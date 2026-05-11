//! Shared engine for the ReCom short-bursts optimizer (Cannon et al. 2020,
//! arXiv: 2011.02288).
//!
//! Short bursts is an MCMC-with-restart heuristic. The chain runs as a
//! single sequential ReCom Markov chain that always accepts every valid
//! proposal (no within-chain rejection); workers parallelize the expensive
//! tree-drawing step exactly as in the vanilla and tilted runners. After
//! every `burst_length` accepted steps the main thread snaps the chain back
//! to the best-scoring plan seen during the burst (including the burst's
//! starting plan), and a new burst begins from that plan.
//!
//! Threading model:
//! - Each worker draws a single candidate tree+split per packet, retrying
//!   internally on non-adjacent district pairs, disconnected merged
//!   subgraphs, or no-balanced-cut splits. Workers report exactly one
//!   `ScoredProposal` per packet; there is no self-loop accounting.
//! - The main thread receives one packet from every worker per round,
//!   picks one proposal (uniform random over the collected proposals,
//!   exactly like `multi_chain` in `run.rs`), applies it to the canonical
//!   chain via `backend.apply_accepted`, broadcasts the diff to every
//!   worker, and advances the step counter by one.
//! - Optional output writers run on separate threads fed by bounded
//!   channels so disk I/O does not block proposal generation.
//!
//! Scoring is delegated to a [`ScoringBackend`]. Production runs use
//! [`crate::recom::IncrementalBackend`] for O(boundary) per-step scoring;
//! tests and ad-hoc explorations can use [`crate::recom::FullRescoreBackend`]
//! to plug in arbitrary closure objectives.
use super::super::{
    make_sampler, node_bound, random_split, sample_dist_pair, RecomParams, RecomVariant,
    ScoringBackend, WorkerBuffers,
};
use super::packets::{
    broadcast_diff, terminate_burst_worker, BurstDiff, BurstJobPacket, BurstResult,
    BurstScorePacket, BurstStatsPacket, ScoredProposal,
};
use super::writers::{start_burst_score_writer, start_burst_stats_writer};
use crate::buffers::graph_connected_buffered;
use crate::graph::Graph;
use crate::partition::Partition;
use crate::spanning_tree::SpanningTreeSampler;
use crate::stats::{ScoresWriter, StatsWriter};
use crossbeam::scope;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Capacity of the bounded channels feeding the stats and score writer
/// threads. Matches the tilted runner.
const WRITER_CHANNEL_CAPACITY: usize = 128;

/// Public type alias for the score type used by the short-bursts engine.
pub type ScoreValue = f64;

/// Draws one candidate proposal, retrying internally on non-adjacent district
/// pairs, disconnected subgraphs, or no-balanced-cut splits. Always returns a
/// [`ScoredProposal`] (short bursts has no within-chain rejection).
fn draw_burst_proposal<B>(
    graph: &Graph,
    partition: &mut Partition,
    state: &B::State,
    params: &RecomParams,
    buffers: &mut WorkerBuffers<B::Scratch>,
    st_sampler: &mut Box<dyn SpanningTreeSampler>,
    backend: &B,
    rng: &mut SmallRng,
) -> ScoredProposal
where
    B: ScoringBackend,
{
    loop {
        let Some((dist_a, dist_b)) = sample_dist_pair(graph, partition, params.variant, rng)
        else {
            continue;
        };
        partition.subgraph(graph, &mut buffers.subgraph, dist_a, dist_b);
        if !graph_connected_buffered(&buffers.subgraph.graph, &mut buffers.connectivity) {
            continue;
        }
        st_sampler.random_spanning_tree_with_parent(
            &buffers.subgraph.graph,
            graph,
            &buffers.subgraph.raw_nodes,
            &mut buffers.spanning_tree,
            rng,
        );
        let split = random_split(
            &buffers.subgraph.graph,
            graph,
            rng,
            &buffers.spanning_tree.st,
            dist_a,
            dist_b,
            &mut buffers.split,
            &mut buffers.proposal,
            &buffers.subgraph.raw_nodes,
            params,
        );
        if split.is_err() {
            continue;
        }

        // Score with temp-apply / revert via the backend so the worker's
        // canonical partition is unchanged. This matches the tilted
        // runner's contract for `score_candidate`.
        let score = backend.score_candidate(
            graph,
            partition,
            state,
            &mut buffers.scratch,
            &buffers.proposal,
        );
        return ScoredProposal {
            id: rng.random::<u64>(),
            proposal: buffers.proposal.clone(),
            score,
        };
    }
}

/// Runs a short-bursts worker thread.
///
/// On each job the worker applies the broadcast diff (a proposal patch or a
/// burst-boundary partition reset), draws one candidate proposal, sends it
/// back, and waits for the next job. The worker's local `partition` and
/// `state` mirror the canonical chain at the start of every draw.
fn run_burst_worker<B>(
    graph: &Graph,
    mut partition: Partition,
    mut state: B::State,
    params: RecomParams,
    backend: B,
    rng_seed: u64,
    buf_size: usize,
    job_recv: Receiver<BurstJobPacket>,
    result_send: Sender<BurstResult>,
) where
    B: ScoringBackend,
{
    let n = graph.pops.len();
    let mut rng: SmallRng = SeedableRng::seed_from_u64(rng_seed);
    let mut buffers = WorkerBuffers::new(
        backend.make_scratch(buf_size),
        n,
        buf_size,
        params.balance_ub,
    );
    let mut st_sampler = make_sampler(&params, buf_size, &mut rng);

    let mut next: BurstJobPacket = job_recv.recv().unwrap();
    while !next.terminate {
        match std::mem::replace(&mut next.diff, BurstDiff::None) {
            BurstDiff::None => {}
            BurstDiff::Apply(proposal) => {
                backend.apply_accepted(graph, &mut partition, &mut state, &proposal);
            }
            BurstDiff::Reset(new_partition) => {
                partition = new_partition;
                state = backend.init_state(graph, &partition);
            }
        }

        let proposal = draw_burst_proposal(
            graph,
            &mut partition,
            &state,
            &params,
            &mut buffers,
            &mut st_sampler,
            &backend,
            &mut rng,
        );
        result_send.send(BurstResult { proposal }).unwrap();
        next = job_recv.recv().unwrap();
    }
}

/// Tracks the best plan seen during the current burst.
///
/// Initialized from the burst seed; updated whenever the chain visits a
/// strictly better plan. At burst boundaries the main thread snaps to this
/// plan and rebuilds backend state from it before the next burst begins.
struct BurstBest<S> {
    partition: Partition,
    state: S,
    score: f64,
}

impl<S: Clone> BurstBest<S> {
    fn new(partition: Partition, state: S, score: f64) -> Self {
        Self {
            partition,
            state,
            score,
        }
    }

    /// Updates the burst-best if `score` is a strict improvement.
    /// Returns whether the burst-best was updated.
    fn observe(
        &mut self,
        partition: &Partition,
        state: &S,
        score: f64,
        maximize: bool,
    ) -> bool {
        let strict = if maximize {
            score > self.score
        } else {
            score < self.score
        };
        if strict {
            self.partition = partition.clone();
            self.state = state.clone();
            self.score = score;
        }
        strict
    }
}

/// Runs a multi-threaded ReCom short-bursts optimizer with optional async
/// output writers.
///
/// The chain is a single sequential ReCom random walk: every successful
/// tree+split is accepted and applied. Workers parallelize the tree-drawing
/// step. After every `burst_length` accepted steps the main thread snaps the
/// chain back to the best-scoring plan seen during the burst.
///
/// When `write_best_only = false`, the stats writer thread is called for
/// every accepted chain step; sample numbers are sequential starting at 1.
/// When `write_best_only = true`, the stats writer thread is called once per
/// burst boundary with the snapped (best-of-burst) plan; sample numbers count
/// bursts, not steps.
///
/// `scores_writer` mirrors `stats_writer`: one row per accepted chain step
/// when `write_best_only = false`, one row per burst boundary when
/// `write_best_only = true`. Each row carries the step's score, the running
/// global best, and (on strict global improvements) per-district scores.
///
/// Because short-bursts emits full partitions to the writer (workers don't
/// produce proposal diffs that the writer can stitch together), the stats
/// writer thread receives a synthetic empty proposal on each call. Writers
/// that require proposal-level data (TSV, JSONL, pcompress) will produce
/// empty/zeroed proposal fields; prefer `assignments`,
/// `canonicalized-assignments`, `canonical`, or `ben` writers.
///
/// # Arguments
///
/// * `graph` - The graph associated with `partition`.
/// * `partition` - The starting partition.
/// * `params` - The chain parameters of the ReCom chain runs.
/// * `n_threads` - The number of worker threads (excluding the main thread).
/// * `backend` - The scoring backend.
/// * `maximize` - If true, maximize the objective. If false, minimize it.
/// * `burst_length` - The number of accepted chain steps per burst.
/// * `stats_writer` - Optional asynchronous writer. Cadence depends on
///   `write_best_only` (see above).
/// * `scores_writer` - Optional asynchronous writer for objective scores.
///   Cadence matches `stats_writer`.
/// * `show_progress` - If true, display a progress bar to stdout.
/// * `write_best_only` - If true, emit one record per burst (the snapped
///   plan); if false, emit one record per accepted chain step.
pub fn multi_short_bursts_with_writer<B>(
    graph: &Graph,
    partition: Partition,
    params: &RecomParams,
    n_threads: usize,
    backend: B,
    maximize: bool,
    burst_length: usize,
    stats_writer: Option<&mut dyn StatsWriter>,
    scores_writer: Option<&mut ScoresWriter>,
    show_progress: bool,
    write_best_only: bool,
) -> Result<Partition, String>
where
    B: ScoringBackend,
{
    if n_threads == 0 {
        return Err("n_threads must be at least 1".to_string());
    }
    if params.variant == RecomVariant::Reversible {
        return Err(
            "Reversible ReCom is not supported by the short bursts optimizer.".to_string(),
        );
    }
    if burst_length == 0 {
        return Err("burst_length must be at least 1".to_string());
    }

    let node_ub = node_bound(&graph.pops, params.max_pop);

    let mut job_sends = vec![];
    let mut job_recvs = vec![];
    for _ in 0..n_threads {
        let (s, r): (Sender<BurstJobPacket>, Receiver<BurstJobPacket>) = unbounded();
        job_sends.push(s);
        job_recvs.push(r);
    }
    let (result_send, result_recv): (Sender<BurstResult>, Receiver<BurstResult>) = unbounded();

    let initial_state = backend.init_state(graph, &partition);
    let initial_score = backend.initial_score(graph, &partition, &initial_state);
    let initial_district_scores = backend.initial_district_scores(&initial_state);

    // The stats writer emits the seed plan in init(), which counts as the
    // first output record. Total output records = num_steps, so the chain
    // produces num_steps - 1 events of its own.
    let effective_records = params.num_steps.saturating_sub(1);

    let progress_bar = if show_progress {
        let pb = ProgressBar::with_draw_target(
            Some(params.num_steps),
            indicatif::ProgressDrawTarget::stdout_with_hz(1),
        );
        pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:100.cyan/blue} {pos:>10}/{len} ({eta_precise})",
            )
            .unwrap()
            .progress_chars("##-"),
        );
        Some(pb)
    } else {
        None
    };

    let scoped_result = scope(|scope| -> Result<Partition, String> {
        // Spawn writer threads, if requested.
        let stats_send = if let Some(writer) = stats_writer {
            let (send, recv): (Sender<BurstStatsPacket>, Receiver<BurstStatsPacket>) =
                bounded(WRITER_CHANNEL_CAPACITY);
            scope.spawn({
                let partition = partition.clone();
                move |_| start_burst_stats_writer(graph, partition, writer, recv)
            });
            Some(send)
        } else {
            None
        };
        let score_send = if let Some(writer) = scores_writer {
            let (send, recv): (Sender<BurstScorePacket>, Receiver<BurstScorePacket>) =
                bounded(WRITER_CHANNEL_CAPACITY);
            scope.spawn(move |_| {
                start_burst_score_writer(writer, initial_score, initial_district_scores, recv)
            });
            Some(send)
        } else {
            None
        };

        // Spawn worker threads.
        for t_idx in 0..n_threads {
            let rng_seed = params.rng_seed + t_idx as u64 + 1;
            let job_recv = job_recvs[t_idx].clone();
            let result_send = result_send.clone();
            let worker_partition = partition.clone();
            let worker_backend = backend.clone();
            let worker_state = initial_state.clone();

            scope.spawn(move |_| {
                run_burst_worker(
                    graph,
                    worker_partition,
                    worker_state,
                    params.clone(),
                    worker_backend,
                    rng_seed,
                    node_ub,
                    job_recv,
                    result_send,
                );
            });
        }

        // Send initial empty diff to all workers so they draw their first
        // round of proposals from the seed plan.
        if effective_records > 0 {
            broadcast_diff(&job_sends, &BurstDiff::None);
        }

        // Canonical chain state, mutated in place by accepted proposals.
        let mut partition = partition;
        let mut state = initial_state.clone();
        let mut score = initial_score;

        // Best-so-far over the entire run. The optimizer returns the best
        // partition seen, not the chain's final mid-burst state. The
        // running score is also reported as `best_score` on every score
        // writer row.
        let mut global_best_partition = partition.clone();
        let mut global_best_state = state.clone();
        let mut global_best = initial_score;

        // Best-so-far within the current burst; the chain snaps to this
        // partition and state at burst boundaries. Initialized to the
        // burst's starting plan so that a burst that only walks downhill
        // still has somewhere sensible to snap.
        let mut burst_best = BurstBest::new(partition.clone(), state.clone(), score);
        let mut step_in_burst: u64 = 0;

        // Sequential sample number for the writers.
        let mut writer_step: u64 = 0;
        // Total accepted chain steps so far.
        let mut step: u64 = 0;

        // Main-thread RNG, used to pick which worker's proposal to apply
        // each round (matches the random-pick interleaving in `run.rs`).
        let mut main_rng: SmallRng = SeedableRng::seed_from_u64(params.rng_seed);

        while step < effective_records {
            // Collect one proposal from every worker.
            let mut proposals: Vec<ScoredProposal> = Vec::with_capacity(n_threads);
            for _ in 0..n_threads {
                proposals.push(result_recv.recv().unwrap().proposal);
            }
            // Sort by random ID so the random pick is reproducible across
            // arrival orders (same technique as `run.rs`).
            proposals.sort_by_key(|p| p.id);
            let chosen = proposals.swap_remove(main_rng.random_range(0..proposals.len()));

            // Apply the chosen proposal to the canonical chain.
            backend.apply_accepted(graph, &mut partition, &mut state, &chosen.proposal);
            score = chosen.score;
            step += 1;
            step_in_burst += 1;

            // Track the best plan within this burst.
            burst_best.observe(&partition, &state, score, maximize);

            // Track running global best.
            let strict_global = if maximize {
                score > global_best
            } else {
                score < global_best
            };
            if strict_global {
                global_best = score;
                global_best_partition = partition.clone();
                global_best_state = state.clone();
            }

            // Per-step writes.
            if !write_best_only {
                writer_step += 1;
                if let Some(send) = stats_send.as_ref() {
                    send.send(BurstStatsPacket {
                        step: writer_step,
                        partition: Some(partition.clone()),
                        terminate: false,
                    })
                    .unwrap();
                }
                if let Some(send) = score_send.as_ref() {
                    let ds = if strict_global {
                        backend.step_district_scores(&state)
                    } else {
                        None
                    };
                    send.send(BurstScorePacket {
                        step: writer_step,
                        score,
                        best_score: global_best,
                        district_scores: ds,
                        terminate: false,
                    })
                    .unwrap();
                }
            }

            // Decide what diff to broadcast for the next round, snapping
            // at burst boundaries.
            let next_diff = if step_in_burst == burst_length as u64 {
                let snapped =
                    if maximize {
                        burst_best.score > score
                    } else {
                        burst_best.score < score
                    };
                if snapped {
                    partition = burst_best.partition.clone();
                    state = burst_best.state.clone();
                    score = burst_best.score;
                }

                // Per-burst writes (write_best_only mode).
                if write_best_only {
                    writer_step += 1;
                    if let Some(send) = stats_send.as_ref() {
                        send.send(BurstStatsPacket {
                            step: writer_step,
                            partition: Some(partition.clone()),
                            terminate: false,
                        })
                        .unwrap();
                    }
                    if let Some(send) = score_send.as_ref() {
                        // Per-district scores are recomputed for the
                        // snapped state when it differs from the running
                        // global best (i.e. the snap surfaced a plan that
                        // is also a global high-water mark).
                        let strict_global_after_snap = if maximize {
                            score > global_best
                        } else {
                            score < global_best
                        };
                        if strict_global_after_snap {
                            global_best = score;
                        }
                        let ds = if strict_global_after_snap {
                            backend.step_district_scores(&state)
                        } else {
                            None
                        };
                        send.send(BurstScorePacket {
                            step: writer_step,
                            score,
                            best_score: global_best,
                            district_scores: ds,
                            terminate: false,
                        })
                        .unwrap();
                    }
                }

                // Reset burst tracking and seed the next burst from the
                // snapped plan.
                burst_best = BurstBest::new(partition.clone(), state.clone(), score);
                step_in_burst = 0;

                if snapped {
                    // Workers must rebuild from the snapped partition.
                    BurstDiff::Reset(partition.clone())
                } else {
                    // The burst's last-applied proposal is already the
                    // snapped plan, so workers are in sync via the
                    // ordinary apply.
                    BurstDiff::Apply(chosen.proposal.clone())
                }
            } else {
                BurstDiff::Apply(chosen.proposal.clone())
            };

            broadcast_diff(&job_sends, &next_diff);

            if let Some(pb) = progress_bar.as_ref() {
                pb.set_position((step + 1).min(params.num_steps));
            }
        }

        // Drain workers and writers.
        for job in job_sends.iter() {
            terminate_burst_worker(job);
        }

        // For write_best_only mode, emit one final record with the
        // global-best plan when the chain ended mid-burst (no snap event
        // has fired yet for the in-progress burst). Skipped when
        // step_in_burst == 0 (the last iteration was a burst boundary
        // and already emitted its snap).
        if write_best_only && step_in_burst > 0 {
            writer_step += 1;
            if let Some(send) = stats_send.as_ref() {
                send.send(BurstStatsPacket {
                    step: writer_step,
                    partition: Some(global_best_partition.clone()),
                    terminate: false,
                })
                .unwrap();
            }
            if let Some(send) = score_send.as_ref() {
                let ds = backend.step_district_scores(&global_best_state);
                send.send(BurstScorePacket {
                    step: writer_step,
                    score: global_best,
                    best_score: global_best,
                    district_scores: ds,
                    terminate: false,
                })
                .unwrap();
            }
        }

        if let Some(send) = stats_send.as_ref() {
            send.send(BurstStatsPacket {
                step: 0,
                partition: None,
                terminate: true,
            })
            .unwrap();
        }
        if let Some(send) = score_send.as_ref() {
            send.send(BurstScorePacket {
                step: 0,
                score: 0.0,
                best_score: 0.0,
                district_scores: None,
                terminate: true,
            })
            .unwrap();
        }

        Ok(global_best_partition)
    });

    if let Some(pb) = progress_bar {
        pb.set_position(params.num_steps);
        pb.finish_and_clear();
    }

    match scoped_result {
        Ok(inner) => inner,
        Err(_panic) => Err("multi_short_bursts panicked in a worker thread".to_string()),
    }
}
