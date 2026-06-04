//! `abs_deviation` objective: drive a target number of districts toward a target
//! population-of-interest share.
//!
//! For each district, `share = pov_counts / total_counts` and
//! `deviation = |share - target|`. The score is the sum of the
//! `n_target_districts` smallest deviations -- the districts whose share is
//! closest to `target`. Lower is better (run in minimize mode).
//!
//! See [`super::ObjectiveConfig::AbsDeviation`] for the JSON schema.

use super::{district_share, leak_str, sum_attr_over, ObjectiveConfig};
use crate::graph::Graph;
use crate::partition::Partition;
use crate::recom::RecomProposal;
use crate::stats::partition_attr_sums;
use serde_json::Value;

/// Per-district cached state for an [`ObjectiveConfig::AbsDeviation`] objective.
#[derive(Clone, Debug)]
pub struct AbsDeviationState {
    /// Per-district population-of-interest share (`pov_counts / total_counts`).
    pub(super) shares: Vec<f64>,
    pub(super) score: f64,
}

/// `|share - target|`, treating a non-finite share (e.g. `total_counts == 0`)
/// as `+inf` so an empty district is never selected among the `n` nearest.
#[inline]
fn deviation(share: f64, target: f64) -> f64 {
    if share.is_finite() {
        (share - target).abs()
    } else {
        f64::INFINITY
    }
}

/// Sum of the `n_target_districts` smallest deviations across all districts.
///
/// `num_dists` is small, so a full sort is negligible.
fn abs_dev_score(shares: &[f64], target: f64, n_target_districts: usize) -> f64 {
    let mut devs: Vec<f64> = shares.iter().map(|&s| deviation(s, target)).collect();
    devs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    devs.iter().take(n_target_districts).sum()
}

/// Per-district shares from per-district count sums.
fn shares_from_counts(pov_counts: &[i32], total_counts: &[i32]) -> Vec<f64> {
    pov_counts
        .iter()
        .zip(total_counts.iter())
        .map(|(&p, &t)| district_share(p, t))
        .collect()
}

impl AbsDeviationState {
    pub(super) fn init(
        graph: &Graph,
        partition: &Partition,
        target: f64,
        n_target_districts: usize,
        pov_counts_col: &str,
        total_counts_col: &str,
    ) -> AbsDeviationState {
        let num_dists = partition.num_dists as usize;

        assert!(n_target_districts > 0);
        assert!(n_target_districts <= num_dists);

        let pov_counts = partition_attr_sums(graph, partition, pov_counts_col);
        let total_counts = partition_attr_sums(graph, partition, total_counts_col);
        debug_assert_eq!(pov_counts.len(), num_dists);
        debug_assert_eq!(total_counts.len(), num_dists);

        let shares = shares_from_counts(&pov_counts, &total_counts);
        let score = abs_dev_score(&shares, target, n_target_districts);
        AbsDeviationState { shares, score }
    }
}

/// Parses an `abs_deviation` objective config. See [`super::make_objective`].
pub(super) fn from_json(data: &Value) -> ObjectiveConfig {
    let target = data["target"].as_f64().unwrap();
    assert!(target > 0.0 && target < 1.0, "'target' must be in (0, 1)");
    let n_target_districts = data["n_target_districts"].as_u64().unwrap_or_else(|| {
        panic!(
            "'n_target_districts' must be a non-negative integer, got {}",
            data["n_target_districts"]
        )
    }) as usize;

    if n_target_districts == 0 {
        panic!("'n_target_districts' must be positive");
    }

    ObjectiveConfig::AbsDeviation {
        target,
        n_target_districts,
        pov_counts_col: leak_str(data, "pov_counts_col"),
        total_counts_col: leak_str(data, "total_counts_col"),
    }
}

/// Node attribute columns required by an `abs_deviation` config.
pub(super) fn required_node_cols(data: &Value) -> Vec<String> {
    vec![
        data["pov_counts_col"].as_str().unwrap().to_string(),
        data["total_counts_col"].as_str().unwrap().to_string(),
    ]
}

/// Full (non-incremental) score over every district.
pub(super) fn full_score(
    graph: &Graph,
    partition: &Partition,
    target: f64,
    n_target_districts: usize,
    pov_counts_col: &str,
    total_counts_col: &str,
) -> f64 {
    let pov_counts = partition_attr_sums(graph, partition, pov_counts_col);
    let total_counts = partition_attr_sums(graph, partition, total_counts_col);
    let shares = shares_from_counts(&pov_counts, &total_counts);
    abs_dev_score(&shares, target, n_target_districts)
}

pub(super) fn score_proposal(
    graph: &Graph,
    state: &AbsDeviationState,
    target: f64,
    n_target_districts: usize,
    pov_counts_col: &str,
    total_counts_col: &str,
    proposal: &RecomProposal,
) -> f64 {
    let new_a = district_share(
        sum_attr_over(graph, pov_counts_col, &proposal.a_nodes),
        sum_attr_over(graph, total_counts_col, &proposal.a_nodes),
    );
    let new_b = district_share(
        sum_attr_over(graph, pov_counts_col, &proposal.b_nodes),
        sum_attr_over(graph, total_counts_col, &proposal.b_nodes),
    );

    let mut shares = state.shares.clone();
    shares[proposal.a_label] = new_a;
    shares[proposal.b_label] = new_b;
    abs_dev_score(&shares, target, n_target_districts)
}

pub(super) fn apply_proposal(
    graph: &Graph,
    state: &mut AbsDeviationState,
    target: f64,
    n_target_districts: usize,
    pov_counts_col: &str,
    total_counts_col: &str,
    proposal: &RecomProposal,
) {
    let new_a = district_share(
        sum_attr_over(graph, pov_counts_col, &proposal.a_nodes),
        sum_attr_over(graph, total_counts_col, &proposal.a_nodes),
    );
    let new_b = district_share(
        sum_attr_over(graph, pov_counts_col, &proposal.b_nodes),
        sum_attr_over(graph, total_counts_col, &proposal.b_nodes),
    );

    state.shares[proposal.a_label] = new_a;
    state.shares[proposal.b_label] = new_b;
    state.score = abs_dev_score(&state.shares, target, n_target_districts);
}
