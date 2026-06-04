//! `by_district_abs_deviation` objective: minimize the total absolute distance
//! between a list of target shares and distinct districts.
//!
//! For each district, `share = pov_counts / total_counts`. Given a list of
//! `target_values` (length `k`, with `1 <= k <= num_dists`), the score is the
//! minimum total `|share - target|` over all ways of matching the `k` targets
//! to `k` *distinct* districts -- i.e. each target claims its own district. On
//! the real line this minimum-cost assignment is achieved by sorting both the
//! targets and the district shares and matching order-preservingly, which a
//! small O(n * k) DP computes (with fast paths for the two common shapes).
//!
//! Two shapes have a simpler reading:
//! - all targets equal to `t` (length `k`) rewards the `k` districts closest to
//!   `t` (the sum of the `k` smallest `|share - t|`);
//! - one target per district (`k == num_dists`) is the full sorted bijection.
//!
//! Lower is better (run in minimize mode). See
//! [`super::ObjectiveConfig::ByDistrictAbsDeviation`] for the JSON schema.

use super::{district_share, leak_str, sum_attr_over, ObjectiveConfig};
use crate::graph::Graph;
use crate::partition::Partition;
use crate::recom::RecomProposal;
use crate::stats::partition_attr_sums;
use serde_json::Value;

/// Per-district cached state for an [`ObjectiveConfig::ByDistrictAbsDeviation`]
/// objective.
#[derive(Clone, Debug)]
pub struct ByDistrictAbsDeviationState {
    /// Per-district population-of-interest share (`pov_counts / total_counts`),
    /// indexed by district label (unsorted).
    pub(super) shares: Vec<f64>,
    pub(super) score: f64,
}

/// Effective share for ordering and deviation: a non-finite share (e.g.
/// `total_counts == 0`) is treated as `+inf` so it sorts last and yields a
/// large, deterministic (non-`NaN`) contribution rather than breaking the sort.
#[inline]
fn effective_share(share: f64) -> f64 {
    if share.is_finite() {
        share
    } else {
        f64::INFINITY
    }
}

/// Minimum total `|share - target|` over all matchings of the `k` targets to
/// `k` distinct districts.
///
/// `sorted_targets` must be sorted ascending, with `sorted_targets.len() <=
/// shares.len()`. The general path is an O(n * k) DP over the sorted district
/// shares: `dp[j]` is the cheapest way to assign the first `j` targets to
/// distinct districts seen so far, matched order-preservingly (optimal for an
/// L1 cost on a line). Two cheap special cases bypass the DP.
fn by_district_abs_dev_score(shares: &[f64], sorted_targets: &[f64]) -> f64 {
    let n = shares.len();
    let k = sorted_targets.len();
    debug_assert!(k <= n, "more target values than districts");

    // Bypass: all targets equal `t` -> the `k` districts closest to `t`,
    // i.e. the sum of the `k` smallest `|share - t|`.
    if k > 0 && sorted_targets[0] == sorted_targets[k - 1] {
        let t = sorted_targets[0];
        let mut devs: Vec<f64> = shares
            .iter()
            .map(|&x| (effective_share(x) - t).abs())
            .collect();
        devs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        return devs.iter().take(k).sum();
    }

    let mut sorted: Vec<f64> = shares.iter().map(|&x| effective_share(x)).collect();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    // Bypass: one target per district -> the full sorted bijection.
    if k == n {
        return sorted
            .iter()
            .zip(sorted_targets.iter())
            .map(|(&x, &t)| (x - t).abs())
            .sum();
    }

    // General case: minimum-cost order-preserving assignment via DP.
    // Iterate districts (outer) and targets descending (inner) so `dp[j - 1]`
    // still holds the value from the previous district (0/1-knapsack trick).
    let mut dp = vec![f64::INFINITY; k + 1];
    dp[0] = 0.0;
    for &share in sorted.iter() {
        for j in (1..=k).rev() {
            if dp[j - 1].is_finite() {
                let cost = dp[j - 1] + (share - sorted_targets[j - 1]).abs();
                if cost < dp[j] {
                    dp[j] = cost;
                }
            }
        }
    }
    dp[k]
}

/// Per-district shares from per-district count sums.
fn shares_from_counts(pov_counts: &[i32], total_counts: &[i32]) -> Vec<f64> {
    pov_counts
        .iter()
        .zip(total_counts.iter())
        .map(|(&p, &t)| district_share(p, t))
        .collect()
}

/// Validates that there are at most one target per district.
fn assert_target_len(target_values: &[f64], num_dists: usize) {
    assert!(
        !target_values.is_empty() && target_values.len() <= num_dists,
        "'target_values' must have between 1 and {} entries (one per district), got {}",
        num_dists,
        target_values.len()
    );
}

impl ByDistrictAbsDeviationState {
    pub(super) fn init(
        graph: &Graph,
        partition: &Partition,
        target_values: &[f64],
        pov_counts_col: &str,
        total_counts_col: &str,
    ) -> ByDistrictAbsDeviationState {
        let num_dists = partition.num_dists as usize;
        assert_target_len(target_values, num_dists);

        let pov_counts = partition_attr_sums(graph, partition, pov_counts_col);
        let total_counts = partition_attr_sums(graph, partition, total_counts_col);
        debug_assert_eq!(pov_counts.len(), num_dists);
        debug_assert_eq!(total_counts.len(), num_dists);

        let shares = shares_from_counts(&pov_counts, &total_counts);
        let score = by_district_abs_dev_score(&shares, target_values);
        ByDistrictAbsDeviationState { shares, score }
    }
}

/// Validates a JSON value as a target share and returns it.
fn parse_share(v: &Value, ctx: &str) -> f64 {
    let t = v
        .as_f64()
        .unwrap_or_else(|| panic!("{} must be a number, got {}", ctx, v));
    assert!(
        t.is_finite() && (0.0..=1.0).contains(&t),
        "{} must be a finite share in [0, 1], got {}",
        ctx,
        t
    );
    t
}

/// Parses a `by_district_abs_deviation` objective config.
/// See [`super::make_objective`].
///
/// Two input forms are accepted:
/// - explicit `"target_values": [..]` -- one target per claimed district;
/// - shorthand `"target": t, "n_target_districts": k` -- expands to `k` copies
///   of `t`, rewarding the `k` districts closest to `t`. Handy when many
///   districts share one target.
///
/// `target_values` takes precedence if both are present. The targets are stored
/// **sorted ascending** so that scoring only needs to sort the district shares.
pub(super) fn from_json(data: &Value) -> ObjectiveConfig {
    let mut targets: Vec<f64> = if let Some(arr) =
        data.get("target_values").and_then(|v| v.as_array())
    {
        assert!(!arr.is_empty(), "'target_values' must be non-empty");
        arr.iter()
            .map(|v| parse_share(v, "each 'target_values' entry"))
            .collect()
    } else if data.get("target").is_some() || data.get("n_target_districts").is_some() {
        let target = parse_share(&data["target"], "'target'");
        let n = data["n_target_districts"]
            .as_u64()
            .unwrap_or_else(|| {
                panic!(
                    "'n_target_districts' must be a positive integer, got {}",
                    data["n_target_districts"]
                )
            }) as usize;
        assert!(n >= 1, "'n_target_districts' must be positive");
        vec![target; n]
    } else {
        panic!(
            "by_district_abs_deviation requires either 'target_values' (a list) \
             or the 'target' + 'n_target_districts' shorthand"
        );
    };
    targets.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let target_values: &'static [f64] = Box::leak(targets.into_boxed_slice());

    ObjectiveConfig::ByDistrictAbsDeviation {
        target_values,
        pov_counts_col: leak_str(data, "pov_counts_col"),
        total_counts_col: leak_str(data, "total_counts_col"),
    }
}

/// Node attribute columns required by a `by_district_abs_deviation` config.
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
    target_values: &[f64],
    pov_counts_col: &str,
    total_counts_col: &str,
) -> f64 {
    let num_dists = partition.num_dists as usize;
    assert_target_len(target_values, num_dists);
    let pov_counts = partition_attr_sums(graph, partition, pov_counts_col);
    let total_counts = partition_attr_sums(graph, partition, total_counts_col);
    let shares = shares_from_counts(&pov_counts, &total_counts);
    by_district_abs_dev_score(&shares, target_values)
}

pub(super) fn score_proposal(
    graph: &Graph,
    state: &ByDistrictAbsDeviationState,
    target_values: &[f64],
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
    by_district_abs_dev_score(&shares, target_values)
}

pub(super) fn apply_proposal(
    graph: &Graph,
    state: &mut ByDistrictAbsDeviationState,
    target_values: &[f64],
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
    state.score = by_district_abs_dev_score(&state.shares, target_values);
}
