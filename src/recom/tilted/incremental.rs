//! Incremental scoring backend.
//!
//! [`IncrementalBackend`] wraps an [`IncrementalObjective`] and maintains a
//! per-thread cached state. Scoring queries the cache directly via
//! `score_proposal` without mutating the partition. Accepting a proposal
//! updates both the partition (`update_with_dist_adj`, which keeps `dist_adj`
//! warm for cut-edge sampling) and the objective state in lockstep. Cost per
//! candidate scales with the boundary of the affected districts rather than
//! the full partition.

use super::super::RecomProposal;
use super::core::ScoringBackend;
use crate::graph::Graph;
use crate::objectives::IncrementalObjective;
use crate::partition::Partition;

#[derive(Clone)]
pub struct IncrementalBackend<O>
where
    O: IncrementalObjective + Send + Clone + 'static,
    O::State: Send + Clone + 'static,
{
    pub objective: O,
}

impl<O> ScoringBackend for IncrementalBackend<O>
where
    O: IncrementalObjective + Send + Clone + 'static,
    O::State: Send + Clone + 'static,
{
    type State = O::State;
    type Scratch = ();

    fn init_state(&self, graph: &Graph, partition: &Partition) -> Self::State {
        self.objective.init(graph, partition)
    }

    fn make_scratch(&self, _buf_size: usize) -> Self::Scratch {}

    fn initial_score(&self, _graph: &Graph, _partition: &Partition, state: &Self::State) -> f64 {
        self.objective.score_state(state)
    }

    fn initial_district_scores(&self, state: &Self::State) -> Vec<f64> {
        self.objective.district_scores(state)
    }

    fn score_candidate(
        &self,
        graph: &Graph,
        _partition: &mut Partition,
        state: &Self::State,
        _scratch: &mut Self::Scratch,
        proposal: &RecomProposal,
    ) -> f64 {
        self.objective.score_proposal(graph, state, proposal)
    }

    fn apply_accepted(
        &self,
        graph: &Graph,
        partition: &mut Partition,
        state: &mut Self::State,
        proposal: &RecomProposal,
    ) {
        self.objective.apply_proposal(graph, state, proposal);
        partition.update_with_dist_adj(proposal, graph);
    }

    fn step_district_scores(&self, state: &Self::State) -> Option<Vec<f64>> {
        Some(self.objective.district_scores(state))
    }
}
