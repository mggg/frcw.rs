//! `banded_gingles_partial` objective: maximize minority opportunity districts
//! that fall inside a target band, rewarding districts below the band and
//! penalizing districts above it.
//!
//! See [`super::ObjectiveConfig::BandedGinglesPartial`] for the JSON schema and
//! full scoring semantics.

use super::{district_share, leak_str, sum_attr_over, ObjectiveConfig};
use crate::graph::Graph;
use crate::partition::Partition;
use crate::recom::RecomProposal;
use crate::stats::partition_attr_sums;
use serde_json::Value;

/// Per-district cached state for an [`ObjectiveConfig::BandedGinglesPartial`] objective.
///
/// `in_band_count` tracks districts whose share is in `[lower, upper]`,
/// `best_below_*` track the single highest district strictly below `lower`
/// (the tiebreaker reward, mirroring [`super::GinglesPartialState`]), and
/// `above_penalty_sum` is the running sum of `upper / share` over every
/// district strictly above `upper`. Because the above-band term is a sum
/// rather than a max, it is maintained by the two-district delta on each
/// proposal rather than a rescan.
#[derive(Clone, Debug)]
pub struct BandedGinglesPartialState {
    pub(super) min_pops: Vec<i32>,
    pub(super) total_pops: Vec<i32>,
    pub(super) in_band_count: usize,
    pub(super) best_below_dist: Option<usize>,
    pub(super) best_below_share: f64,
    pub(super) above_penalty_sum: f64,
    pub(super) score: f64,
}

impl BandedGinglesPartialState {
    pub(super) fn init(
        graph: &Graph,
        partition: &Partition,
        lower: f64,
        upper: f64,
        min_pop_col: &str,
        total_pop_col: &str,
    ) -> BandedGinglesPartialState {
        let num_dists = partition.num_dists as usize;
        let min_pops = partition_attr_sums(graph, partition, min_pop_col);
        let total_pops = partition_attr_sums(graph, partition, total_pop_col);
        debug_assert_eq!(min_pops.len(), num_dists);
        debug_assert_eq!(total_pops.len(), num_dists);

        let mut in_band_count: usize = 0;
        let mut above_penalty_sum: f64 = 0.0;
        let mut best_below_share: f64 = 0.0;
        let mut best_below_dist: Option<usize> = None;
        for d in 0..num_dists {
            let share = district_share(min_pops[d], total_pops[d]);
            if share >= lower && share <= upper {
                in_band_count += 1;
            } else if share > upper {
                above_penalty_sum += upper / share;
            } else if share > best_below_share || best_below_dist.is_none() {
                best_below_share = share;
                best_below_dist = Some(d);
            }
        }
        let score = in_band_count as f64 + (best_below_share / lower) + above_penalty_sum;
        BandedGinglesPartialState {
            min_pops,
            total_pops,
            in_band_count,
            best_below_dist,
            best_below_share,
            above_penalty_sum,
            score,
        }
    }
}

/// Parses a `banded_gingles_partial` objective config. See [`super::make_objective`].
pub(super) fn from_json(data: &Value) -> ObjectiveConfig {
    let lower = data["lower_threshold"].as_f64().unwrap_or_else(|| {
        panic!("Missing or non-numeric field 'lower_threshold' in objective config")
    });
    let upper = data["upper_threshold"].as_f64().unwrap_or_else(|| {
        panic!("Missing or non-numeric field 'upper_threshold' in objective config")
    });
    assert!(
        lower > 0.0 && lower < 1.0,
        "'lower_threshold' must be in (0, 1)"
    );
    assert!(
        upper > 0.0 && upper < 1.0,
        "'upper_threshold' must be in (0, 1)"
    );
    assert!(
        lower <= upper,
        "'lower_threshold' must be <= 'upper_threshold'"
    );
    ObjectiveConfig::BandedGinglesPartial {
        lower,
        upper,
        min_pop_col: leak_str(data, "min_pop"),
        total_pop_col: leak_str(data, "total_pop"),
    }
}

/// Node attribute columns required by a `banded_gingles_partial` config.
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
    lower: f64,
    upper: f64,
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
    let in_band_count = shares.iter().filter(|&&s| s >= lower && s <= upper).count();
    let above_penalty: f64 = shares.iter().filter(|&&s| s > upper).map(|&s| upper / s).sum();
    let best_below = shares
        .iter()
        .copied()
        .filter(|&s| s < lower)
        .fold(f64::NEG_INFINITY, f64::max);
    let best_below = if best_below.is_finite() { best_below } else { 0.0 };
    in_band_count as f64 + (best_below / lower) + above_penalty
}

pub(super) fn score_proposal(
    graph: &Graph,
    state: &BandedGinglesPartialState,
    lower: f64,
    upper: f64,
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

    let (in_band, _, best_below, above_sum) = banded_gingles_single_update(
        &state.min_pops,
        &state.total_pops,
        state.in_band_count,
        state.best_below_dist,
        state.best_below_share,
        state.above_penalty_sum,
        lower,
        upper,
        a_label,
        b_label,
        new_a_min,
        new_a_total,
        new_b_min,
        new_b_total,
    );

    in_band as f64 + (best_below / lower) + above_sum
}

pub(super) fn apply_proposal(
    graph: &Graph,
    state: &mut BandedGinglesPartialState,
    lower: f64,
    upper: f64,
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

    let (in_band, best_dist, best_share, above_sum) = banded_gingles_single_update(
        &state.min_pops,
        &state.total_pops,
        state.in_band_count,
        state.best_below_dist,
        state.best_below_share,
        state.above_penalty_sum,
        lower,
        upper,
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
    state.in_band_count = in_band;
    state.best_below_dist = best_dist;
    state.best_below_share = best_share;
    state.above_penalty_sum = above_sum;
    state.score = in_band as f64 + (best_share / lower) + above_sum;
}

/// Penalty-reduced reward contributed by a district above the upper band edge.
///
/// Returns `upper / share` (in `(0, 1)`) when `share > upper`, and `0.0`
/// otherwise. The demerit `1 - upper / share` grows with the overshoot and
/// approaches a maximum of `1` per offending district.
#[inline]
fn above_band_contribution(share: f64, upper: f64) -> f64 {
    if share > upper {
        upper / share
    } else {
        0.0
    }
}

/// Recomputes the banded-gingles aggregates after a two-district swap, without
/// mutating the caller's cached state.
///
/// Returns `(in_band_count, best_below_dist, best_below_share, above_penalty_sum)`.
/// The in-band count and above-band penalty sum are updated by the two-district
/// delta (each category is mutually exclusive). The best-below tiebreaker is a
/// max over districts strictly below `lower`, so it uses the same fast/slow
/// rescan pattern as gingles_partial's single update.
fn banded_gingles_single_update(
    min_pops: &[i32],
    total_pops: &[i32],
    cached_in_band: usize,
    cached_below_dist: Option<usize>,
    cached_below_share: f64,
    cached_above_sum: f64,
    lower: f64,
    upper: f64,
    a_label: usize,
    b_label: usize,
    new_a_min: i32,
    new_a_total: i32,
    new_b_min: i32,
    new_b_total: i32,
) -> (usize, Option<usize>, f64, f64) {
    let old_a_share = district_share(min_pops[a_label], total_pops[a_label]);
    let old_b_share = district_share(min_pops[b_label], total_pops[b_label]);
    let new_a_share = district_share(new_a_min, new_a_total);
    let new_b_share = district_share(new_b_min, new_b_total);

    let in_band_of = |s: f64| s >= lower && s <= upper;

    // In-band count: each of the three share categories is mutually exclusive,
    // so we can subtract the two stale districts and add the two fresh ones.
    let mut in_band = cached_in_band;
    for s in [old_a_share, old_b_share] {
        if in_band_of(s) {
            in_band -= 1;
        }
    }
    for s in [new_a_share, new_b_share] {
        if in_band_of(s) {
            in_band += 1;
        }
    }

    // Above-band penalty: a running sum, so the two-district delta is exact.
    let above_sum = cached_above_sum
        - above_band_contribution(old_a_share, upper)
        - above_band_contribution(old_b_share, upper)
        + above_band_contribution(new_a_share, upper)
        + above_band_contribution(new_b_share, upper);

    // Best below-band share: a max, so mirror gingles_single_update.
    let new_a_below = if new_a_share < lower {
        new_a_share
    } else {
        f64::NEG_INFINITY
    };
    let new_b_below = if new_b_share < lower {
        new_b_share
    } else {
        f64::NEG_INFINITY
    };

    let (best_dist, best_share) = match cached_below_dist {
        Some(holder) if holder != a_label && holder != b_label => {
            let mut dist = Some(holder);
            let mut share = cached_below_share;
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
                if s < lower && s > share {
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

    (in_band, best_dist, best_share, above_sum)
}
