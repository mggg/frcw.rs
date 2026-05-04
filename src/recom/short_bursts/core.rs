//! Shared engine for the ReCom short-bursts optimizer (Cannon et al. 2020,
//! arXiv: 2011.02288).
//!
//! Short bursts is a greedy hill-climbing heuristic over an arbitrary
//! partition-level objective. The chain runs in fixed-length "bursts"; within
//! a burst the worker always accepts every valid proposal and tracks the
//! burst-end best partition. After each round of bursts the main thread
//! adopts any new global best as the seed of the next burst across all
//! workers.
//!
//! Threading model mirrors the tilted runner: workers parallelize the
//! tree-drawing and scoring step, the main thread interleaves their results,
//! and optional output writers run on separate threads fed by bounded
//! channels so disk I/O does not block proposal generation.
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
    send_burst_batch, terminate_burst_worker, BurstJobPacket, BurstResult, BurstScorePacket,
    BurstStatsPacket,
};
use super::writers::{start_burst_score_writer, start_burst_stats_writer};
use crate::buffers::graph_connected_buffered;
use crate::graph::Graph;
use crate::partition::Partition;
use crate::stats::{ScoresWriter, StatsWriter};
use crossbeam::scope;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Capacity of the bounded channels feeding the stats and score writer
/// threads. Matches the tilted runner.
const WRITER_CHANNEL_CAPACITY: usize = 128;

/// Public type alias for the score type used by the short-bursts engine.
pub type ScoreValue = f64;

/// Runs a short-bursts worker thread.
///
/// On each job the worker runs `n_steps` greedy ReCom steps from its current
/// (or freshly-replaced) partition, optionally collecting every accepted step
/// into `BurstResult::all_steps`, and always tracks the burst-end best.
///
/// When the main thread broadcasts a wholesale partition replacement via
/// `BurstJobPacket::diff`, the worker re-initializes its backend state from
/// the new partition: the cached state is no longer valid against the
/// replaced partition. This is the same swap-handling pattern used by the
/// tilted runner.
fn run_burst_worker<B>(
    graph: &Graph,
    mut partition: Partition,
    mut state: B::State,
    params: RecomParams,
    backend: B,
    maximize: bool,
    rng_seed: u64,
    buf_size: usize,
    collect_trace: bool,
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
        if let Some(new_partition) = next.diff.take() {
            partition = new_partition;
            state = backend.init_state(graph, &partition);
        }

        let mut all_steps: Vec<(Partition, ScoreValue)> = Vec::new();
        let mut best: Option<(Partition, ScoreValue)> = None;
        let mut best_score: ScoreValue = backend.current_score(graph, &partition, &state);
        let mut step = 0;
        while step < next.n_steps {
            let dist_pair = sample_dist_pair(graph, &mut partition, params.variant, &mut rng);
            if dist_pair.is_none() {
                continue;
            }
            let (dist_a, dist_b) = dist_pair.unwrap();
            partition.subgraph(graph, &mut buffers.subgraph, dist_a, dist_b);
            if !graph_connected_buffered(&buffers.subgraph.graph, &mut buffers.connectivity) {
                continue;
            }
            st_sampler.random_spanning_tree_with_parent(
                &buffers.subgraph.graph,
                graph,
                &buffers.subgraph.raw_nodes,
                &mut buffers.spanning_tree,
                &mut rng,
            );
            let split = random_split(
                &buffers.subgraph.graph,
                graph,
                &mut rng,
                &buffers.spanning_tree.st,
                dist_a,
                dist_b,
                &mut buffers.split,
                &mut buffers.proposal,
                &buffers.subgraph.raw_nodes,
                &params,
            );
            if split.is_ok() {
                backend.apply_accepted(graph, &mut partition, &mut state, &buffers.proposal);
                let score = backend.current_score(graph, &partition, &state);
                if collect_trace {
                    all_steps.push((partition.clone(), score));
                }
                if (maximize && score >= best_score) || (!maximize && score <= best_score) {
                    best = Some((partition.clone(), score));
                    best_score = score;
                }
                step += 1;
            }
        }
        result_send
            .send(BurstResult { all_steps, best })
            .unwrap();
        next = job_recv.recv().unwrap();
    }
}

/// Runs a multi-threaded ReCom short-bursts optimizer with optional async
/// output writers.
///
/// Workers run short ReCom bursts using the scoring path provided by
/// `backend`. Use [`crate::recom::IncrementalBackend`] for production runs:
/// per-step scoring drops to O(boundary of two affected districts).
///
/// When `write_best_only = false`, the stats writer thread is called for
/// every accepted chain step across all worker bursts; sample numbers are
/// sequential starting at 1. When `write_best_only = true`, the stats writer
/// thread is called only when a new global best is found, and sample numbers
/// count improvements.
///
/// `scores_writer` mirrors `stats_writer`: one row per accepted chain step
/// when `write_best_only = false`, one row per new global best when
/// `write_best_only = true`. Each row carries the step's score, the running
/// global best, and (on new bests) per-district scores; non-best rows omit
/// district scores so the writer replays its last cached vector.
///
/// Because short-bursts workers return full partitions (not individual
/// proposals), the stats writer thread receives a synthetic empty proposal on
/// each call. Writers that require proposal-level data (TSV, JSONL, pcompress)
/// will produce empty/zeroed proposal fields; prefer `assignments`,
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
/// * `burst_length` - The number of accepted steps per burst per worker.
/// * `stats_writer` - Optional asynchronous writer called for every step (or
///   only new bests when `write_best_only = true`).
/// * `scores_writer` - Optional asynchronous writer called whenever a new
///   global best is found.
/// * `show_progress` - If true, display a progress bar to stdout.
/// * `write_best_only` - If true, the stats writer is called only when a new
///   global best is found, and workers skip collecting per-step partitions.
pub fn multi_short_bursts_with_writer<B>(
    graph: &Graph,
    mut partition: Partition,
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
    if params.variant == RecomVariant::Reversible {
        return Err(
            "Reversible ReCom is not supported by the short bursts optimizer.".to_string(),
        );
    }

    let mut step = 1u64;
    let node_ub = node_bound(&graph.pops, params.max_pop);
    let mut job_sends = vec![];
    let mut job_recvs = vec![];
    for _ in 0..n_threads {
        let (s, r): (Sender<BurstJobPacket>, Receiver<BurstJobPacket>) = unbounded();
        job_sends.push(s);
        job_recvs.push(r);
    }
    let (result_send, result_recv): (Sender<BurstResult>, Receiver<BurstResult>) = unbounded();

    // Build initial backend state for writer initialization and worker seeds.
    let initial_state = backend.init_state(graph, &partition);
    let initial_score = backend.initial_score(graph, &partition, &initial_state);
    let initial_district_scores = backend.initial_district_scores(&initial_state);

    // Collect per-step partitions only when the stats writer needs every step.
    let collect_trace = !write_best_only;

    // The stats writer emits the seed plan in init(), which counts as the
    // first output record. To keep the total output record count equal to
    // params.num_steps, we run one fewer chain step.
    let effective_steps = params.num_steps.saturating_sub(1);

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
                    maximize,
                    rng_seed,
                    node_ub,
                    collect_trace,
                    job_recv,
                    result_send,
                );
            });
        }

        if effective_steps > 0 {
            for job in job_sends.iter() {
                send_burst_batch(job, None, burst_length);
            }
        }

        let mut score = initial_score;
        // Running global-best score, monotone non-decreasing under maximize
        // (non-increasing under minimize). Reported as `best_score` on every
        // score-writer row.
        let mut best_score: f64 = initial_score;
        // Main thread's canonical state, kept in sync with `partition` so the
        // score writer can report fresh per-district scores after a new best.
        let mut best_state: B::State = initial_state;
        // Sequential sample number for stats_writer: incremented once per
        // partition emitted to the writer (every step, or every new best).
        let mut writer_step: u64 = 0;

        while step < effective_steps {
            let mut all_packets: Vec<BurstResult> = Vec::with_capacity(n_threads);
            for _ in 0..n_threads {
                all_packets.push(result_recv.recv().unwrap());
            }
            step += (n_threads * burst_length) as u64;

            let mut diff: Option<Partition> = None;

            if write_best_only {
                // Emit only on new global bests; one stats + one score row per
                // improvement.
                for packet in all_packets {
                    if let Some((bp, bs)) = packet.best {
                        if (maximize && bs >= score) || (!maximize && bs <= score) {
                            partition = bp;
                            score = bs;
                            diff = Some(partition.clone());
                            best_state = backend.init_state(graph, &partition);
                            let strict_improvement = (maximize && bs > best_score)
                                || (!maximize && bs < best_score);
                            if strict_improvement {
                                best_score = bs;
                            }
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
                                let ds = backend.step_district_scores(&best_state);
                                send.send(BurstScorePacket {
                                    step: writer_step,
                                    score: bs,
                                    best_score,
                                    district_scores: ds,
                                    terminate: false,
                                })
                                .unwrap();
                            }
                        }
                    }
                }
            } else {
                // Emit every accepted step. The stats writer is capped at
                // effective_steps so total records (init seed + per-step)
                // equal params.num_steps; the score writer matches that cap.
                for packet in all_packets {
                    for (stepped_partition, stepped_score) in packet.all_steps {
                        if writer_step >= effective_steps {
                            break;
                        }
                        writer_step += 1;
                        let is_chain_best = (maximize && stepped_score >= score)
                            || (!maximize && stepped_score <= score);
                        let strict_improvement = (maximize && stepped_score > best_score)
                            || (!maximize && stepped_score < best_score);
                        if is_chain_best {
                            partition = stepped_partition.clone();
                            score = stepped_score;
                            diff = Some(partition.clone());
                            best_state = backend.init_state(graph, &partition);
                            if strict_improvement {
                                best_score = stepped_score;
                            }
                        }
                        if let Some(send) = stats_send.as_ref() {
                            send.send(BurstStatsPacket {
                                step: writer_step,
                                partition: Some(stepped_partition),
                                terminate: false,
                            })
                            .unwrap();
                        }
                        if let Some(send) = score_send.as_ref() {
                            // Per-district scores are only fresh when the
                            // canonical chain advanced to a new best;
                            // otherwise the writer caches and replays the
                            // last vector via `None`.
                            let ds = if strict_improvement {
                                backend.step_district_scores(&best_state)
                            } else {
                                None
                            };
                            send.send(BurstScorePacket {
                                step: writer_step,
                                score: stepped_score,
                                best_score,
                                district_scores: ds,
                                terminate: false,
                            })
                            .unwrap();
                        }
                    }
                }
            }

            if let Some(pb) = progress_bar.as_ref() {
                pb.set_position((step + 1).min(params.num_steps));
            }

            for job in job_sends.iter() {
                send_burst_batch(job, diff.clone(), burst_length);
            }
        }

        for job in job_sends.iter() {
            terminate_burst_worker(job);
        }

        // Tell the writer threads to drain and exit.
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

        Ok(partition)
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
