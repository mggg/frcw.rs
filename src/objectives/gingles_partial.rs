//! `gingles_partial` objective: maximize minority opportunity districts with a
//! next-partial-district tiebreaker.
//!
//! See [`super::ObjectiveConfig::GinglesPartial`] for the JSON schema and full
//! scoring semantics.

use super::{district_share, leak_str, sum_attr_over, ObjectiveConfig};
use crate::graph::Graph;
use crate::partition::Partition;
use crate::recom::RecomProposal;
use crate::stats::partition_attr_sums;
use serde_json::Value;

/// Per-district cached state for an [`ObjectiveConfig::GinglesPartial`] objective.
#[derive(Clone, Debug)]
pub struct GinglesPartialState {
    pub(super) min_pops: Vec<i32>,
    pub(super) total_pops: Vec<i32>,
    pub(super) opportunity_count: usize,
    pub(super) best_below_dist: Option<usize>,
    pub(super) best_below_share: f64,
    pub(super) score: f64,
}

impl GinglesPartialState {
    pub(super) fn init(
        graph: &Graph,
        partition: &Partition,
        threshold: f64,
        min_pop_col: &str,
        total_pop_col: &str,
    ) -> GinglesPartialState {
        let num_dists = partition.num_dists as usize;
        let min_pops = partition_attr_sums(graph, partition, min_pop_col);
        let total_pops = partition_attr_sums(graph, partition, total_pop_col);
        debug_assert_eq!(min_pops.len(), num_dists);
        debug_assert_eq!(total_pops.len(), num_dists);

        let mut opportunity_count: usize = 0;
        let mut best_below_share: f64 = 0.0;
        let mut best_below_dist: Option<usize> = None;
        for d in 0..num_dists {
            let share = min_pops[d] as f64 / total_pops[d] as f64;
            if share >= threshold {
                opportunity_count += 1;
            } else if share > best_below_share || best_below_dist.is_none() {
                best_below_share = share;
                best_below_dist = Some(d);
            }
        }
        let score = opportunity_count as f64 + (best_below_share / threshold);
        GinglesPartialState {
            min_pops,
            total_pops,
            opportunity_count,
            best_below_dist,
            best_below_share,
            score,
        }
    }
}

/// Parses a `gingles_partial` objective config. See [`super::make_objective`].
pub(super) fn from_json(data: &Value) -> ObjectiveConfig {
    let threshold = data["threshold"].as_f64().unwrap();
    assert!(
        threshold > 0.0 && threshold < 1.0,
        "'threshold' must be in (0, 1)"
    );
    ObjectiveConfig::GinglesPartial {
        threshold,
        min_pop_col: leak_str(data, "min_pop"),
        total_pop_col: leak_str(data, "total_pop"),
    }
}

/// Node attribute columns required by a `gingles_partial` config.
pub(super) fn required_node_cols(data: &Value) -> Vec<String> {
    vec![
        data["min_pop"].as_str().unwrap().to_string(),
        data["total_pop"].as_str().unwrap().to_string(),
    ]
}

/// Full (non-incremental) score over every district.
pub(super) fn full_score(
    graph: &Graph,
    partition: &Partition,
    threshold: f64,
    min_pop_col: &str,
    total_pop_col: &str,
) -> f64 {
    let min_pops = partition_attr_sums(graph, partition, min_pop_col);
    let total_pops = partition_attr_sums(graph, partition, total_pop_col);
    let shares: Vec<f64> = min_pops
        .iter()
        .zip(total_pops.iter())
        .map(|(&m, &t)| m as f64 / t as f64)
        .collect();
    let opportunity_count = shares.iter().filter(|&&s| s >= threshold).count();
    let mut sorted_below: Vec<f64> = shares.into_iter().filter(|&s| s < threshold).collect();
    sorted_below.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let next_highest = sorted_below.last().copied().unwrap_or(0.0);
    opportunity_count as f64 + (next_highest / threshold)
}

pub(super) fn score_proposal(
    graph: &Graph,
    state: &GinglesPartialState,
    threshold: f64,
    min_pop_col: &str,
    total_pop_col: &str,
    proposal: &RecomProposal,
) -> f64 {
    let a_label = proposal.a_label;
    let b_label = proposal.b_label;
    let new_a_min = sum_attr_over(graph, min_pop_col, &proposal.a_nodes);
    let new_a_total = sum_attr_over(graph, total_pop_col, &proposal.a_nodes);
    let new_b_min = sum_attr_over(graph, min_pop_col, &proposal.b_nodes);
    let new_b_total = sum_attr_over(graph, total_pop_col, &proposal.b_nodes);

    let (opp_count, _, best_below) = gingles_single_update(
        &state.min_pops,
        &state.total_pops,
        state.opportunity_count,
        state.best_below_dist,
        state.best_below_share,
        threshold,
        a_label,
        b_label,
        new_a_min,
        new_a_total,
        new_b_min,
        new_b_total,
    );

    opp_count as f64 + (best_below / threshold)
}

pub(super) fn apply_proposal(
    graph: &Graph,
    state: &mut GinglesPartialState,
    threshold: f64,
    min_pop_col: &str,
    total_pop_col: &str,
    proposal: &RecomProposal,
) {
    let a_label = proposal.a_label;
    let b_label = proposal.b_label;
    let new_a_min = sum_attr_over(graph, min_pop_col, &proposal.a_nodes);
    let new_a_total = sum_attr_over(graph, total_pop_col, &proposal.a_nodes);
    let new_b_min = sum_attr_over(graph, min_pop_col, &proposal.b_nodes);
    let new_b_total = sum_attr_over(graph, total_pop_col, &proposal.b_nodes);

    let (opp_count, best_dist, best_share) = gingles_single_update(
        &state.min_pops,
        &state.total_pops,
        state.opportunity_count,
        state.best_below_dist,
        state.best_below_share,
        threshold,
        a_label,
        b_label,
        new_a_min,
        new_a_total,
        new_b_min,
        new_b_total,
    );

    state.min_pops[a_label] = new_a_min;
    state.total_pops[a_label] = new_a_total;
    state.min_pops[b_label] = new_b_min;
    state.total_pops[b_label] = new_b_total;
    state.opportunity_count = opp_count;
    state.best_below_dist = best_dist;
    state.best_below_share = best_share;
    state.score = opp_count as f64 + (best_share / threshold);
}

fn gingles_single_update(
    min_pops: &[i32],
    total_pops: &[i32],
    cached_opp: usize,
    cached_best_dist: Option<usize>,
    cached_best_share: f64,
    threshold: f64,
    a_label: usize,
    b_label: usize,
    new_a_min: i32,
    new_a_total: i32,
    new_b_min: i32,
    new_b_total: i32,
) -> (usize, Option<usize>, f64) {
    let old_a_share = district_share(min_pops[a_label], total_pops[a_label]);
    let old_b_share = district_share(min_pops[b_label], total_pops[b_label]);
    let new_a_share = district_share(new_a_min, new_a_total);
    let new_b_share = district_share(new_b_min, new_b_total);

    let mut opp = cached_opp;
    if old_a_share >= threshold {
        opp -= 1;
    }
    if old_b_share >= threshold {
        opp -= 1;
    }
    if new_a_share >= threshold {
        opp += 1;
    }
    if new_b_share >= threshold {
        opp += 1;
    }

    let new_a_below = if new_a_share < threshold {
        new_a_share
    } else {
        f64::NEG_INFINITY
    };
    let new_b_below = if new_b_share < threshold {
        new_b_share
    } else {
        f64::NEG_INFINITY
    };

    let (best_dist, best_share) = match cached_best_dist {
        Some(holder) if holder != a_label && holder != b_label => {
            let mut dist = Some(holder);
            let mut share = cached_best_share;
            if new_a_below > share {
                share = new_a_below;
                dist = Some(a_label);
            }
            if new_b_below > share {
                share = new_b_below;
                dist = Some(b_label);
            }
            if share == f64::NEG_INFINITY {
                (None, 0.0)
            } else {
                (dist, share)
            }
        }
        _ => {
            // Slow path: rescan unchanged districts.
            let mut dist: Option<usize> = None;
            let mut share: f64 = f64::NEG_INFINITY;
            for (d, (&m, &t)) in min_pops.iter().zip(total_pops.iter()).enumerate() {
                if d == a_label || d == b_label {
                    continue;
                }
                let s = district_share(m, t);
                if s < threshold && s > share {
                    share = s;
                    dist = Some(d);
                }
            }
            if new_a_below > share {
                share = new_a_below;
                dist = Some(a_label);
            }
            if new_b_below > share {
                share = new_b_below;
                dist = Some(b_label);
            }
            if share == f64::NEG_INFINITY {
                (None, 0.0)
            } else {
                (dist, share)
            }
        }
    };

    (opp, best_dist, best_share)
}
