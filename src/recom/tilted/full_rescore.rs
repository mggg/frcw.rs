//! Full-rescore scoring backend.
//!
//! [`FullRescoreBackend`] wraps a closure `Fn(&Graph, &Partition) -> f64`.
//! On each candidate it temp-applies the proposal, rescores the entire
//! partition from scratch, then reverts. Cheap to set up, but pays
//! `O(closure)` work per candidate. Use this when prototyping new objectives
//! or for objectives that need to read the full partition; for production
//! chains, prefer [`super::IncrementalBackend`].

use super::super::RecomProposal;
use super::core::ScoringBackend;
use crate::graph::Graph;
use crate::partition::Partition;

/// Stores the current state of the two affected districts so the caller can
/// restore them after a temp-apply / score / revert cycle.
fn save_revert_buffer(
    partition: &Partition,
    proposal: &RecomProposal,
    revert: &mut RecomProposal,
) {
    revert.a_label = proposal.a_label;
    revert.b_label = proposal.b_label;
    revert.a_pop = partition.dist_pops[proposal.a_label];
    revert.b_pop = partition.dist_pops[proposal.b_label];
    revert.a_nodes.clear();
    revert
        .a_nodes
        .extend_from_slice(&partition.dist_nodes[proposal.a_label]);
    revert.b_nodes.clear();
    revert
        .b_nodes
        .extend_from_slice(&partition.dist_nodes[proposal.b_label]);
}

#[derive(Clone, Copy)]
pub struct FullRescoreBackend<F>
where
    F: Fn(&Graph, &Partition) -> f64 + Send + Clone + Copy,
{
    pub obj_fn: F,
}

impl<F> ScoringBackend for FullRescoreBackend<F>
where
    F: Fn(&Graph, &Partition) -> f64 + Send + Clone + Copy,
{
    type State = ();
    type Scratch = RecomProposal;

    fn init_state(&self, _graph: &Graph, _partition: &Partition) -> Self::State {}

    fn make_scratch(&self, buf_size: usize) -> Self::Scratch {
        RecomProposal::new_buffer(buf_size)
    }

    fn initial_score(&self, graph: &Graph, partition: &Partition, _state: &Self::State) -> f64 {
        (self.obj_fn)(graph, partition)
    }

    fn initial_district_scores(&self, _state: &Self::State) -> Vec<f64> {
        Vec::new()
    }

    fn score_candidate(
        &self,
        graph: &Graph,
        partition: &mut Partition,
        _state: &Self::State,
        scratch: &mut Self::Scratch,
        proposal: &RecomProposal,
    ) -> f64 {
        save_revert_buffer(partition, proposal, scratch);
        partition.update(proposal);
        let score = (self.obj_fn)(graph, partition);
        partition.update(scratch);
        score
    }

    fn apply_accepted(
        &self,
        _graph: &Graph,
        partition: &mut Partition,
        _state: &mut Self::State,
        proposal: &RecomProposal,
    ) {
        // Full-rescore doesn't read dist_adj, so no benefit to keeping it warm.
        partition.update(proposal);
    }

    fn step_district_scores(&self, _state: &Self::State) -> Option<Vec<f64>> {
        None
    }
}
