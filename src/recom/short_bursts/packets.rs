//! Wire format and channel-dispatch helpers shared between the short-bursts
//! engine and its writer threads. Packet types and dispatch helpers are
//! private to the `short_bursts` module via `pub(super)`.

use crossbeam_channel::Sender;

use super::super::RecomProposal;
use crate::partition::Partition;

/// Instruction from the main thread to a worker between rounds.
///
/// `Apply` is the common case: the canonical chain advanced by a single
/// accepted proposal, so each worker patches its partition copy
/// incrementally. `Reset` is sent at burst boundaries when the main thread
/// snaps the chain back to the best plan seen during the burst (which may
/// be many steps behind the worker's current state); workers replace their
/// partition wholesale and rebuild backend state. `None` means no state
/// change since the last batch (only used for the very first round).
pub(super) enum BurstDiff {
    None,
    Apply(RecomProposal),
    Reset(Partition),
}

/// A unit of work sent from the main thread to a worker.
pub(super) struct BurstJobPacket {
    /// State change to apply before drawing the next proposal.
    pub(super) diff: BurstDiff,
    /// A sentinel used to kill the worker thread.
    pub(super) terminate: bool,
}

/// One worker-drawn candidate proposal, plus the score it would produce if
/// applied to the current canonical chain state.
pub(super) struct ScoredProposal {
    /// Random ID used for deterministic interleaving across worker arrivals
    /// (same technique as `run.rs` and `tilted::core.rs`).
    pub(super) id: u64,
    /// The candidate proposal.
    pub(super) proposal: RecomProposal,
    /// Objective score after applying `proposal` to the worker's local copy.
    pub(super) score: f64,
}

/// The result of one round of work from a worker. Always carries one
/// successful proposal -- workers retry internally on failed tree draws or
/// disconnected merges and never report self-loops, since short bursts has
/// no within-chain rejection.
pub(super) struct BurstResult {
    pub(super) proposal: Result<ScoredProposal, String>,
}

/// A chain-statistics write packet sent from the main thread to the stats
/// writer thread.
pub(super) struct BurstStatsPacket {
    /// Sequential sample number for the writer.
    pub(super) step: u64,
    /// Partition to emit. `None` only for the termination sentinel.
    pub(super) partition: Option<Partition>,
    /// A sentinel used to stop the writer thread.
    pub(super) terminate: bool,
}

/// A score-record write packet sent from the main thread to the score writer
/// thread.
pub(super) struct BurstScorePacket {
    /// Chain step at which this score event occurred.
    pub(super) step: u64,
    /// Objective score at this event.
    pub(super) score: f64,
    /// Per-district scores to carry forward, or `None` to reuse the writer's
    /// previously cached vector.
    pub(super) district_scores: Option<Vec<f64>>,
    /// A sentinel used to stop the writer thread.
    pub(super) terminate: bool,
}

/// Sends the same diff to every worker.
pub(super) fn broadcast_diff(job_sends: &[Sender<BurstJobPacket>], diff: &BurstDiff) {
    for job in job_sends.iter() {
        job.send(BurstJobPacket {
            diff: diff.clone(),
            terminate: false,
        })
        .unwrap();
    }
}

/// Stops a short-bursts worker thread.
pub(super) fn terminate_burst_worker(send: &Sender<BurstJobPacket>) {
    let _ = send.send(BurstJobPacket {
        diff: BurstDiff::None,
        terminate: true,
    });
}

impl Clone for BurstDiff {
    fn clone(&self) -> Self {
        match self {
            BurstDiff::None => BurstDiff::None,
            BurstDiff::Apply(p) => BurstDiff::Apply(p.clone()),
            BurstDiff::Reset(p) => BurstDiff::Reset(p.clone()),
        }
    }
}
