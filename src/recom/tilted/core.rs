//! Shared engine for tilted-run ReCom optimizers.
//!
//! A tilted run is a continuous Markov chain that always accepts plans with
//! better scores and accepts plans with worse scores according to a pluggable
//! [`AcceptanceRule`]. The two acceptance rules currently provided live in
//! sibling modules: `fixed::FixedAcceptance` accepts a worse plan with a fixed
//! probability, and `metropolis::MetropolisAcceptance` accepts a worse plan
//! with probability `exp(beta * delta)` where `delta` is signed by the
//! optimization direction. Unlike short bursts, there is no burst boundary
//! and no resetting to a global best state.
//!
//! Threading follows the same model as `multi_chain` in `run.rs`: worker
//! threads parallelize the tree-drawing step within a single sequential
//! tilted chain. Workers draw trees, score proposals, and decide accept/reject;
//! the main thread interleaves accepted proposals and rejections. Optional
//! output writers run on separate threads fed by bounded channels, so disk I/O
//! and serialization usually do not block proposal generation; the bound only
//! engages if a writer falls persistently behind the chain.
//!
//! See `docs/tilted_runs_spec.md` for full architecture documentation.
use super::super::{
    make_sampler, node_bound, random_split, sample_dist_pair, RecomParams, RecomProposal,
    RecomVariant, WorkerBuffers,
};
use super::packets::{
    collect_tilted_results, send_tilted_jobs, stop_tilted_workers, ScoredProposal,
    TiltedJobPacket, TiltedResultPacket, TiltedScorePacket, TiltedStatsPacket,
};
use super::writers::{start_tilted_score_writer, start_tilted_stats_writer};
use crate::buffers::graph_connected_buffered;
use crate::graph::Graph;
use crate::partition::Partition;
use crate::spanning_tree::SpanningTreeSampler;
use crate::stats::{ScoresWriter, SelfLoopCounts, SelfLoopReason, StatsWriter};
use crossbeam::scope;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};

/// Capacity of the bounded channels that feed the stats and score writer
/// threads. Large enough to absorb transient writer stalls; small enough to
/// cap worst-case memory growth if a writer stays persistently behind.
const WRITER_CHANNEL_CAPACITY: usize = 128;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Decides whether to accept a proposal that does not improve the current score.
///
/// Implementors carry the parameters of a particular acceptance rule (for
/// example a fixed probability, or an inverse temperature). The trait is
/// `Copy` so each worker thread holds its own value without coordination.
pub trait AcceptanceRule: Send + Copy {
    /// Returns `true` to accept a worse-or-equal proposal, `false` to reject.
    ///
    /// `current` is the score of the current chain state and `proposed` is the
    /// score of the candidate. `maximize` is `true` when larger scores are
    /// improvements. Implementors may consult `rng` for stochastic decisions.
    /// This method is only called when the proposal is not strictly an
    /// improvement; strict improvements are always accepted by the engine.
    fn accept_worse(&self, current: f64, proposed: f64, maximize: bool, rng: &mut SmallRng)
        -> bool;
}

/// Pluggable scoring strategy for ReCom-based optimizers.
///
/// A backend owns whatever per-thread state and scratch buffers it needs to
/// score candidate proposals and to apply accepted proposals to the chain
/// state. The same trait is used by both the tilted runner (which scores
/// candidates with temp-apply / score / revert under [`score_candidate`]) and
/// the short-bursts runner (which always-accepts within a burst and reads
/// the canonical score back via [`current_score`]).
///
/// Two backends ship with the crate:
///
/// - [`super::FullRescoreBackend`] wraps a closure
///   `Fn(&Graph, &Partition) -> f64` and rescores from scratch on every
///   candidate.
/// - [`super::IncrementalBackend`] wraps an
///   [`crate::objectives::IncrementalObjective`] and reads the score off the
///   cached state without mutating the partition.
pub trait ScoringBackend: Send + Clone {
    /// Per-thread cached state. `()` for full-rescore scoring; the cached
    /// objective state for incremental scoring.
    type State: Send + Clone;

    /// Per-thread scratch buffers needed by the backend's hot path. The
    /// full-rescore backend uses this for its revert buffer; the incremental
    /// backend has nothing to cache here.
    type Scratch: Send;

    /// Builds the initial cached state from the starting partition.
    fn init_state(&self, graph: &Graph, partition: &Partition) -> Self::State;

    /// Allocates per-worker scratch buffers sized for proposals up to
    /// `buf_size` nodes.
    fn make_scratch(&self, buf_size: usize) -> Self::Scratch;

    /// Returns the objective score of the starting partition.
    fn initial_score(&self, graph: &Graph, partition: &Partition, state: &Self::State) -> f64;

    /// Returns per-district scores for the score writer's CSV header. An empty
    /// vector switches the writer to the legacy three-column output.
    fn initial_district_scores(&self, state: &Self::State) -> Vec<f64>;

    /// Scores a candidate proposal. Backends may mutate `partition` or
    /// `scratch` so long as `partition` is restored to its original state
    /// before returning.
    fn score_candidate(
        &self,
        graph: &Graph,
        partition: &mut Partition,
        state: &Self::State,
        scratch: &mut Self::Scratch,
        proposal: &RecomProposal,
    ) -> f64;

    /// Applies an accepted proposal to the canonical partition and the cached
    /// state in lockstep. Backends choose the right `partition.update*`
    /// variant based on what they cache.
    fn apply_accepted(
        &self,
        graph: &Graph,
        partition: &mut Partition,
        state: &mut Self::State,
        proposal: &RecomProposal,
    );

    /// Returns per-district scores to forward to the score writer for the
    /// just-accepted step, or `None` to keep replaying the last vector.
    fn step_district_scores(&self, state: &Self::State) -> Option<Vec<f64>>;

    /// Returns the current canonical score from cached state without
    /// rescoring. For backends whose state already caches the score (e.g.
    /// `IncrementalBackend`), this is O(1). For backends without caching (e.g.
    /// `FullRescoreBackend`), the default falls back to a full rescore.
    ///
    /// Used by always-accept runners (short bursts) to read the score after
    /// `apply_accepted` without paying the cost of `score_candidate`'s
    /// temp-apply / revert dance.
    fn current_score(&self, graph: &Graph, partition: &Partition, state: &Self::State) -> f64 {
        self.initial_score(graph, partition, state)
    }
}

/// Main-thread state for the single sequential tilted chain. Generic in the
/// backend's per-thread cached state so `()` for full-rescore and the
/// objective state for incremental backends both fit uniformly.
struct TiltedMainState<S: Send + Clone> {
    /// Total chain steps processed, including accepted proposals and self-loops.
    step: u64,
    /// Canonical chain state.
    partition: Partition,
    /// Backend-cached state (e.g. an `IncrementalObjective::State`).
    backend_state: S,
    /// Objective score of `partition`.
    current_score: f64,
    /// Tilted rejection counts since the last accepted proposal.
    pending_counts: SelfLoopCounts,
}

impl<S: Send + Clone> TiltedMainState<S> {
    /// Creates main-thread chain state from the initial partition, backend
    /// state, and score.
    fn new(partition: Partition, backend_state: S, current_score: f64) -> Self {
        Self {
            step: 0,
            pending_counts: SelfLoopCounts::default(),
            partition,
            backend_state,
            current_score,
        }
    }

    /// Records tilted rejection self-loops.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of consecutive tilted rejections to record.
    /// * `score_send` - Optional channel for per-step objective-score records.
    fn record_rejections(&mut self, count: usize, score_send: Option<&Sender<TiltedScorePacket>>) {
        if count == 0 {
            return;
        }
        let first_step = self.step + 1;
        self.step += count as u64;
        self.pending_counts
            .inc_by(SelfLoopReason::TiltedRejection, count);
        if let Some(send) = score_send {
            send.send(TiltedScorePacket {
                first_step,
                last_step: self.step,
                score: self.current_score,
                district_scores: None,
                terminate: false,
            })
            .unwrap();
        }
    }

    /// Applies an accepted proposal via the backend and sends output packets.
    ///
    /// # Arguments
    ///
    /// * `graph` - Graph backing the partition; passed to `backend.apply_accepted`.
    /// * `backend` - Scoring backend whose `apply_accepted` updates the partition.
    /// * `accepted` - Proposal selected by the main-thread interleaving step.
    /// * `stats_send` - Optional channel for accepted proposal statistics.
    /// * `score_send` - Optional channel for per-step objective-score records.
    fn apply_accepted_proposal<B>(
        &mut self,
        graph: &Graph,
        backend: &B,
        accepted: &ScoredProposal,
        stats_send: Option<&Sender<TiltedStatsPacket>>,
        score_send: Option<&Sender<TiltedScorePacket>>,
    ) where
        B: ScoringBackend<State = S>,
    {
        self.step += 1;
        backend.apply_accepted(
            graph,
            &mut self.partition,
            &mut self.backend_state,
            &accepted.proposal,
        );
        self.current_score = accepted.score;

        if let Some(send) = stats_send {
            send.send(TiltedStatsPacket {
                step: self.step,
                proposal: Some(accepted.proposal.clone()),
                counts: std::mem::take(&mut self.pending_counts),
                terminate: false,
            })
            .unwrap();
        } else {
            self.pending_counts = SelfLoopCounts::default();
        }
        if let Some(send) = score_send {
            send.send(TiltedScorePacket {
                first_step: self.step,
                last_step: self.step,
                score: self.current_score,
                district_scores: backend.step_district_scores(&self.backend_state),
                terminate: false,
            })
            .unwrap();
        }
    }

    /// Sends pending terminal self-loops to the stats-writer thread if one exists.
    ///
    /// # Arguments
    ///
    /// * `stats_send` - Optional channel for accepted proposal statistics.
    fn send_pending_self_loops(&self, stats_send: Option<&Sender<TiltedStatsPacket>>) {
        if self.pending_counts.sum() == 0 {
            return;
        }
        if let Some(send) = stats_send {
            send.send(TiltedStatsPacket {
                step: self.step,
                proposal: None,
                counts: self.pending_counts.clone(),
                terminate: false,
            })
            .unwrap();
        }
    }
}

/// Interleaves one batch of worker rejections/proposals into the sequential chain.
///
/// Rejections increment the chain step without changing state. Accepted proposals
/// and score events are sent to asynchronous writer threads.
///
/// # Arguments
///
/// * `state` - Main-thread canonical chain state.
/// * `loops` - Number of worker tilted rejections to interleave as self-loops.
/// * `proposals` - Worker-accepted proposals to interleave with self-loops.
/// * `params` - ReCom parameters containing the target step count.
/// * `rng` - Main-thread RNG used for event interleaving and proposal selection.
/// * `job_sends` - Worker job channels for broadcasting state updates.
/// * `stats_send` - Optional channel for accepted proposal statistics.
/// * `score_send` - Optional channel for per-step objective-score records.
fn interleave_tilted_round<B>(
    graph: &Graph,
    backend: &B,
    state: &mut TiltedMainState<B::State>,
    mut loops: usize,
    mut proposals: Vec<ScoredProposal>,
    effective_steps: u64,
    rng: &mut SmallRng,
    job_sends: &[Sender<TiltedJobPacket>],
    stats_send: Option<&Sender<TiltedStatsPacket>>,
    score_send: Option<&Sender<TiltedScorePacket>>,
) where
    B: ScoringBackend,
{
    if proposals.is_empty() {
        let remaining = effective_steps.saturating_sub(state.step) as usize;
        state.record_rejections(loops.min(remaining), score_send);
        send_tilted_jobs(job_sends, None, state.current_score);
        return;
    }

    proposals.sort_by_key(|proposal| proposal.id);
    let mut total = loops + proposals.len();
    while total > 0 && state.step < effective_steps {
        let event = rng.random_range(0..total);
        if event < loops {
            state.record_rejections(1, score_send);
            loops -= 1;
            total -= 1;
            continue;
        }

        let accepted = &proposals[rng.random_range(0..proposals.len())];
        state.apply_accepted_proposal(graph, backend, accepted, stats_send, score_send);
        send_tilted_jobs(job_sends, Some(&accepted.proposal), state.current_score);
        break; // need new round (state changed)
    }
}

/// Runs the main-thread collection/interleaving loop after workers are spawned.
///
/// # Arguments
///
/// * `state` - Main-thread canonical chain state.
/// * `params` - ReCom parameters containing the target step count.
/// * `n_threads` - Number of worker packets to collect per round.
/// * `result_recv` - Shared result channel from all workers.
/// * `job_sends` - Worker job channels.
/// * `rng` - Main-thread RNG used for interleaving.
/// * `stats_send` - Optional channel for accepted proposal statistics.
/// * `score_send` - Optional channel for per-step objective-score records.
/// * `progress_bar` - Optional progress bar tracking total chain steps.
fn run_tilted_main_loop<B>(
    graph: &Graph,
    backend: &B,
    state: &mut TiltedMainState<B::State>,
    params: &RecomParams,
    effective_steps: u64,
    n_threads: usize,
    result_recv: &Receiver<TiltedResultPacket>,
    job_sends: &[Sender<TiltedJobPacket>],
    rng: &mut SmallRng,
    stats_send: Option<&Sender<TiltedStatsPacket>>,
    score_send: Option<&Sender<TiltedScorePacket>>,
    progress_bar: Option<&ProgressBar>,
) where
    B: ScoringBackend,
{
    if effective_steps > 0 {
        send_tilted_jobs(job_sends, None, state.current_score);
    }

    let progress_chunk = (params.num_steps / 1000).clamp(1, 1000);
    let mut last_drawn = state.step;
    while state.step < effective_steps {
        let (loops, proposals) = collect_tilted_results(result_recv, n_threads);
        interleave_tilted_round(
            graph,
            backend,
            state,
            loops,
            proposals,
            effective_steps,
            rng,
            job_sends,
            stats_send,
            score_send,
        );
        if let Some(progress_bar) = progress_bar {
            // Progress = records emitted so far (init seed + accepted/
            // rejected steps). Goal = params.num_steps records total.
            let drawn = state.step + 1;
            if drawn - last_drawn >= progress_chunk || state.step == effective_steps {
                progress_bar.set_position(drawn);
                last_drawn = drawn;
            }
        }
    }
}

/// Draws one round of tilted output for a backend-aware worker.
fn draw_tilted_result<B, R>(
    graph: &Graph,
    partition: &mut Partition,
    state: &B::State,
    params: &RecomParams,
    buffers: &mut WorkerBuffers<B::Scratch>,
    st_sampler: &mut Box<dyn SpanningTreeSampler>,
    backend: &B,
    current_score: f64,
    rule: R,
    maximize: bool,
    rng: &mut SmallRng,
) -> TiltedResultPacket
where
    B: ScoringBackend,
    R: AcceptanceRule,
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

        let new_score = backend.score_candidate(
            graph,
            partition,
            state,
            &mut buffers.scratch,
            &buffers.proposal,
        );

        let is_improvement = if maximize {
            new_score >= current_score
        } else {
            new_score <= current_score
        };
        if is_improvement || rule.accept_worse(current_score, new_score, maximize, rng) {
            return TiltedResultPacket {
                rejections: 0,
                proposals: vec![ScoredProposal {
                    id: rng.random::<u64>(),
                    proposal: buffers.proposal.clone(),
                    score: new_score,
                }],
            };
        }

        return TiltedResultPacket {
            rejections: 1,
            proposals: Vec::new(),
        };
    }
}

/// Starts a backend-aware tilted worker thread.
fn start_tilted_worker<B, R>(
    graph: &Graph,
    mut partition: Partition,
    mut state: B::State,
    params: RecomParams,
    backend: B,
    rule: R,
    maximize: bool,
    rng_seed: u64,
    buf_size: usize,
    job_recv: Receiver<TiltedJobPacket>,
    result_send: Sender<TiltedResultPacket>,
) where
    B: ScoringBackend,
    R: AcceptanceRule,
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

    let mut next: TiltedJobPacket = job_recv.recv().unwrap();
    while !next.terminate {
        if let Some(diff) = &next.diff {
            backend.apply_accepted(graph, &mut partition, &mut state, diff);
        }

        let result = draw_tilted_result(
            graph,
            &mut partition,
            &state,
            &params,
            &mut buffers,
            &mut st_sampler,
            &backend,
            next.current_score,
            rule,
            maximize,
            &mut rng,
        );

        result_send.send(result).unwrap();
        next = job_recv.recv().unwrap();
    }
}

/// Runs a tilted-run optimizer with parallel tree drawing.
///
/// Worker threads draw spanning trees in parallel from the same partition state,
/// score each proposal via `backend`, and decide accept/reject with `rule`.
/// The main thread interleaves accepted proposals and rejections (exactly like
/// `multi_chain` in `run.rs`), maintaining a single sequential chain.
///
/// # Arguments
///
/// * `graph` - The graph associated with `partition`.
/// * `partition` - The starting partition.
/// * `params` - The chain parameters. `params.num_steps` is the total number
///   of chain steps (accepted + rejected).
/// * `n_threads` - The number of worker threads for parallel tree drawing.
/// * `backend` - The scoring backend (e.g. [`FullRescoreBackend`]).
/// * `rule` - The acceptance rule applied to non-improving proposals.
/// * `maximize` - If true, higher scores are better. If false, lower scores are better.
/// * `stats_writer` - Optional asynchronous writer for accepted chain proposals
///   and self-loop counts.
/// * `score_writer` - Optional asynchronous writer for per-step objective scores.
/// * `show_progress` - If true, show a progress bar tracking total chain steps.
pub fn multi_tilted_runs_with_writer<B, R>(
    graph: &Graph,
    partition: Partition,
    params: &RecomParams,
    n_threads: usize,
    backend: B,
    rule: R,
    maximize: bool,
    stats_writer: Option<&mut dyn StatsWriter>,
    score_writer: Option<&mut ScoresWriter>,
    show_progress: bool,
) -> Result<Partition, String>
where
    B: ScoringBackend,
    R: AcceptanceRule,
{
    if n_threads == 0 {
        return Err("n_threads must be at least 1".to_string());
    }
    if params.variant == RecomVariant::Reversible {
        return Err("Reversible ReCom is not supported by the tilted run optimizer.".to_string());
    }

    // The stats writer emits the seed plan in `init`, which counts as the
    // first output record. The chain produces num_steps - 1 events of its
    // own so that the total output record count equals num_steps -- the
    // same convention used by `multi_chain` and `multi_short_bursts`.
    let effective_steps = params.num_steps.saturating_sub(1);

    let node_ub = node_bound(&graph.pops, params.max_pop);

    let mut job_sends = vec![];
    let mut job_recvs = vec![];
    for _ in 0..n_threads {
        let (s, r): (Sender<TiltedJobPacket>, Receiver<TiltedJobPacket>) = unbounded();
        job_sends.push(s);
        job_recvs.push(r);
    }
    let (result_send, result_recv): (Sender<TiltedResultPacket>, Receiver<TiltedResultPacket>) =
        unbounded();

    let initial_state = backend.init_state(graph, &partition);
    let current_score = backend.initial_score(graph, &partition, &initial_state);
    let initial_district_scores = backend.initial_district_scores(&initial_state);
    let mut rng: SmallRng = SeedableRng::seed_from_u64(params.rng_seed);
    let progress_bar = if show_progress {
        let progress_bar = ProgressBar::with_draw_target(
            Some(params.num_steps),
            indicatif::ProgressDrawTarget::stdout_with_hz(1),
        );
        progress_bar.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:100.cyan/blue} {pos:>10}/{len} ({eta_precise})",
            )
            .unwrap()
            .progress_chars("##-"),
        );
        Some(progress_bar)
    } else {
        None
    };

    let scoped_result = scope(|scope| -> Result<Partition, String> {
        let progress_bar_ref = progress_bar.as_ref();
        let stats_send = if let Some(writer) = stats_writer {
            let (send, recv): (Sender<TiltedStatsPacket>, Receiver<TiltedStatsPacket>) =
                bounded(WRITER_CHANNEL_CAPACITY);
            scope.spawn({
                let partition = partition.clone();
                move |_| start_tilted_stats_writer(graph, partition, writer, recv)
            });
            Some(send)
        } else {
            None
        };
        let score_send = if let Some(writer) = score_writer {
            let (send, recv): (Sender<TiltedScorePacket>, Receiver<TiltedScorePacket>) =
                bounded(WRITER_CHANNEL_CAPACITY);
            scope.spawn(move |_| {
                start_tilted_score_writer(writer, current_score, initial_district_scores, recv)
            });
            Some(send)
        } else {
            None
        };

        for t_idx in 0..n_threads {
            let rng_seed = params.rng_seed + t_idx as u64 + 1;
            let job_recv = job_recvs[t_idx].clone();
            let result_send = result_send.clone();
            let partition = partition.clone();
            let worker_backend = backend.clone();
            let worker_state = initial_state.clone();

            scope.spawn(move |_| {
                start_tilted_worker(
                    graph,
                    partition,
                    worker_state,
                    params.clone(),
                    worker_backend,
                    rule,
                    maximize,
                    rng_seed,
                    node_ub,
                    job_recv,
                    result_send,
                );
            });
        }

        let mut state = TiltedMainState::<B::State>::new(partition, initial_state, current_score);
        run_tilted_main_loop(
            graph,
            &backend,
            &mut state,
            params,
            effective_steps,
            n_threads,
            &result_recv,
            &job_sends,
            &mut rng,
            stats_send.as_ref(),
            score_send.as_ref(),
            progress_bar_ref,
        );

        state.send_pending_self_loops(stats_send.as_ref());
        if let Some(send) = &stats_send {
            send.send(TiltedStatsPacket {
                step: 0,
                proposal: None,
                counts: SelfLoopCounts::default(),
                terminate: true,
            })
            .unwrap();
        }
        if let Some(send) = &score_send {
            send.send(TiltedScorePacket {
                first_step: 0,
                last_step: 0,
                score: 0.0,
                district_scores: None,
                terminate: true,
            })
            .unwrap();
        }
        stop_tilted_workers(&job_sends);
        Ok(state.partition)
    });

    if let Some(progress_bar) = progress_bar {
        progress_bar.set_position(params.num_steps);
        progress_bar.finish_and_clear();
    }

    match scoped_result {
        Ok(inner) => inner,
        Err(_panic) => Err("multi_tilted_runs panicked in a worker thread".to_string()),
    }
}

/// Runs a tilted-run optimizer without writing chain statistics or scores.
///
/// See [`multi_tilted_runs_with_writer`] for the full implementation and argument
/// documentation.
pub fn multi_tilted_runs<B, R>(
    graph: &Graph,
    partition: Partition,
    params: &RecomParams,
    n_threads: usize,
    backend: B,
    rule: R,
    maximize: bool,
    show_progress: bool,
) -> Result<Partition, String>
where
    B: ScoringBackend,
    R: AcceptanceRule,
{
    multi_tilted_runs_with_writer(
        graph,
        partition,
        params,
        n_threads,
        backend,
        rule,
        maximize,
        None,
        None,
        show_progress,
    )
}
