//! Wire format and channel-dispatch helpers shared between the tilted engine
//! and its writer threads. The packet types here are private to the `tilted`
//! module via `pub(super)`; the dispatch helpers are too.

use crossbeam_channel::{Receiver, Sender};

use super::super::RecomProposal;
use crate::stats::SelfLoopCounts;

/// A unit of work sent from the main thread to a worker.
pub(super) struct TiltedJobPacket {
    /// The accepted proposal to apply, or `None` if no state change.
    pub(super) diff: Option<RecomProposal>,
    /// The current chain score, used by the worker for accept/reject decisions.
    pub(super) current_score: f64,
    /// A sentinel used to kill the worker thread.
    pub(super) terminate: bool,
}

/// The result of one round of work from a worker.
pub(super) struct TiltedResultPacket {
    /// Number of tilted-criterion rejections (self-loops).
    pub(super) rejections: usize,
    /// Accepted proposals from this round.
    pub(super) proposals: Vec<ScoredProposal>,
}

/// A proposal accepted by a worker, plus the score it would produce.
pub(super) struct ScoredProposal {
    /// Random ID used for deterministic interleaving (same technique as `run.rs`).
    pub(super) id: u64,
    /// The proposal accepted by a worker.
    pub(super) proposal: RecomProposal,
    /// The objective score after applying `proposal`.
    pub(super) score: f64,
}

/// A chain-statistics write packet sent from the chain thread to the writer thread.
pub(super) struct TiltedStatsPacket {
    /// The accepted proposal step.
    pub(super) step: u64,
    /// The accepted proposal, or `None` only for the termination sentinel.
    pub(super) proposal: Option<RecomProposal>,
    /// Tilted rejection counts since the previous accepted proposal.
    pub(super) counts: SelfLoopCounts,
    /// A sentinel used to stop the writer thread.
    pub(super) terminate: bool,
}

/// A score write packet sent from the chain thread to the score-writer thread.
pub(super) struct TiltedScorePacket {
    /// First chain step represented by this packet.
    pub(super) first_step: u64,
    /// Last chain step represented by this packet.
    pub(super) last_step: u64,
    /// Objective score for each step in this packet.
    pub(super) score: f64,
    /// Best score seen so far for each step in this packet.
    pub(super) best_score: f64,
    /// Per-district scores to carry forward from this point on, if the chain
    /// state changed. `None` on pure-rejection runs, where the writer keeps
    /// reusing its previously cached vector.
    pub(super) district_scores: Option<Vec<f64>>,
    /// A sentinel used to stop the writer thread.
    pub(super) terminate: bool,
}

/// Sends the next canonical chain state to every worker.
pub(super) fn send_tilted_jobs(
    job_sends: &[Sender<TiltedJobPacket>],
    diff: Option<&RecomProposal>,
    score: f64,
) {
    for job in job_sends.iter() {
        job.send(TiltedJobPacket {
            diff: diff.cloned(),
            current_score: score,
            terminate: false,
        })
        .unwrap();
    }
}

/// Stops all tilted-run worker threads.
pub(super) fn stop_tilted_workers(job_sends: &[Sender<TiltedJobPacket>]) {
    for job in job_sends.iter() {
        job.send(TiltedJobPacket {
            diff: None,
            current_score: 0.0,
            terminate: true,
        })
        .unwrap();
    }
}

/// Blocks until each worker has returned exactly one round of tilted output.
///
/// Returns the total number of tilted rejections and all worker-accepted
/// proposals for this round.
pub(super) fn collect_tilted_results(
    result_recv: &Receiver<TiltedResultPacket>,
    n_threads: usize,
) -> (usize, Vec<ScoredProposal>) {
    let mut rejections = 0;
    let mut proposals = Vec::<ScoredProposal>::new();

    for _ in 0..n_threads {
        let packet = result_recv.recv().unwrap();
        rejections += packet.rejections;
        proposals.extend(packet.proposals);
    }

    (rejections, proposals)
}
