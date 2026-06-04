//! `election_wins` objective: maximize (or minimize) the number of districts
//! won by a target party across a set of elections, with a partial-district
//! tiebreaker.
//!
//! See [`super::ObjectiveConfig::ElectionWins`] for the JSON schema and full
//! scoring semantics.

use super::{sum_attr_over, Aggregation, ObjectiveConfig};
use crate::graph::Graph;
use crate::partition::Partition;
use crate::recom::RecomProposal;
use crate::stats::partition_attr_sums;
use serde_json::Value;

/// Largest `f64` strictly smaller than 1.0. A tied district is placed here so
/// the fractional tiebreaker stays strictly below the next integer win count.
const ONE_BELOW: f64 = f64::from_bits(1.0f64.to_bits() - 1);

/// Per-district cached state for an [`ObjectiveConfig::ElectionWins`] objective.
///
/// The inner `target_votes` / `other_votes` vectors are indexed
/// `[election_index][district_index]`. `wins`, `best_losing_dist`,
/// `best_losing_tiebreak`, and `election_scores` are indexed
/// `[election_index]`.
#[derive(Clone, Debug)]
pub struct ElectionWinsState {
    pub(super) target_votes: Vec<Vec<i32>>,
    pub(super) other_votes: Vec<Vec<i32>>,
    pub(super) wins: Vec<usize>,
    pub(super) best_losing_dist: Vec<Option<usize>>,
    pub(super) best_losing_tiebreak: Vec<f64>,
    pub(super) election_scores: Vec<f64>,
    pub(super) score: f64,
}

impl ElectionWinsState {
    pub(super) fn init(
        graph: &Graph,
        partition: &Partition,
        elections: &[(&'static str, &'static str)],
        target_a: bool,
        aggregation: Aggregation,
    ) -> ElectionWinsState {
        let num_dists = partition.num_dists as usize;
        let mut target_votes = Vec::with_capacity(elections.len());
        let mut other_votes = Vec::with_capacity(elections.len());
        let mut wins = Vec::with_capacity(elections.len());
        let mut best_losing_dist = Vec::with_capacity(elections.len());
        let mut best_losing_tiebreak = Vec::with_capacity(elections.len());
        let mut election_scores = Vec::with_capacity(elections.len());

        for &(col_a, col_b) in elections {
            let sums_a = partition_attr_sums(graph, partition, col_a);
            let sums_b = partition_attr_sums(graph, partition, col_b);
            debug_assert_eq!(sums_a.len(), num_dists);
            debug_assert_eq!(sums_b.len(), num_dists);

            let (t_vec, o_vec): (Vec<i32>, Vec<i32>) = if target_a {
                (sums_a.clone(), sums_b.clone())
            } else {
                (sums_b.clone(), sums_a.clone())
            };

            let mut e_wins: usize = 0;
            let mut best_tb: f64 = 0.0;
            let mut best_dist: Option<usize> = None;
            for (d, (&t, &o)) in t_vec.iter().zip(o_vec.iter()).enumerate() {
                let (won, tb) = election_district_outcome(t, o);
                if won {
                    e_wins += 1;
                } else if let Some(tb) = tb {
                    if tb > best_tb {
                        best_tb = tb;
                        best_dist = Some(d);
                    }
                }
            }
            election_scores.push(e_wins as f64 + best_tb);
            target_votes.push(t_vec);
            other_votes.push(o_vec);
            wins.push(e_wins);
            best_losing_dist.push(best_dist);
            best_losing_tiebreak.push(best_tb);
        }

        let score = aggregation.apply(&election_scores);
        ElectionWinsState {
            target_votes,
            other_votes,
            wins,
            best_losing_dist,
            best_losing_tiebreak,
            election_scores,
            score,
        }
    }
}

/// Parses an `election_wins` objective config. See [`super::make_objective`].
pub(super) fn from_json(data: &Value) -> ObjectiveConfig {
    let elections_arr = data["elections"]
        .as_array()
        .unwrap_or_else(|| panic!("Missing field 'elections' in objective config"));
    let mut pairs: Vec<(&'static str, &'static str)> = Vec::with_capacity(elections_arr.len());
    for e in elections_arr {
        let a = super::leak_str(e, "votes_a");
        let b = super::leak_str(e, "votes_b");
        pairs.push((a, b));
    }
    let elections: &'static [(&'static str, &'static str)] = Box::leak(pairs.into_boxed_slice());
    let target = data["target"]
        .as_str()
        .unwrap_or_else(|| panic!("Missing field 'target' in objective config"));
    let target_a = match target {
        "a" => true,
        "b" => false,
        other => panic!("Invalid target '{}'. Use 'a' or 'b'.", other),
    };
    let agg_str = data["aggregation"]
        .as_str()
        .unwrap_or_else(|| panic!("Missing field 'aggregation' in objective config"));
    ObjectiveConfig::ElectionWins {
        elections,
        target_a,
        aggregation: Aggregation::from_str(agg_str),
    }
}

/// Node attribute columns required by an `election_wins` config.
pub(super) fn required_node_cols(data: &Value) -> Vec<String> {
    let mut cols = vec![];
    for e in data["elections"].as_array().unwrap() {
        cols.push(e["votes_a"].as_str().unwrap().to_string());
        cols.push(e["votes_b"].as_str().unwrap().to_string());
    }
    cols
}

/// Full (non-incremental) score over every district.
pub(super) fn full_score(
    graph: &Graph,
    partition: &Partition,
    elections: &[(&'static str, &'static str)],
    target_a: bool,
    aggregation: Aggregation,
) -> f64 {
    let mut election_scores: Vec<f64> = Vec::with_capacity(elections.len());
    for &(col_a, col_b) in elections.iter() {
        let sums_a = partition_attr_sums(graph, partition, col_a);
        let sums_b = partition_attr_sums(graph, partition, col_b);

        let mut wins: usize = 0;
        let mut best_losing_tiebreak: f64 = 0.0;
        for (&va, &vb) in sums_a.iter().zip(sums_b.iter()) {
            let total = va as f64 + vb as f64;
            if total == 0.0 {
                continue;
            }
            let target_votes = if target_a { va } else { vb };
            let other_votes = if target_a { vb } else { va };
            if target_votes > other_votes {
                wins += 1;
            } else {
                // Keep ties as losses, but place them strictly below the next
                // integer win count so the fractional tiebreaker stays in [0, 1).
                let tiebreak = if target_votes == other_votes {
                    ONE_BELOW
                } else {
                    (target_votes as f64 / total) / 0.5
                };
                if tiebreak > best_losing_tiebreak {
                    best_losing_tiebreak = tiebreak;
                }
            }
        }
        // Tiebreaker stays in [0, 1), so it sits between integer win counts.
        election_scores.push(wins as f64 + best_losing_tiebreak);
    }
    aggregation.apply(&election_scores)
}

pub(super) fn score_proposal(
    graph: &Graph,
    state: &ElectionWinsState,
    elections: &[(&'static str, &'static str)],
    target_a: bool,
    aggregation: Aggregation,
    proposal: &RecomProposal,
) -> f64 {
    let a_label = proposal.a_label;
    let b_label = proposal.b_label;
    let mut election_scores: Vec<f64> = Vec::with_capacity(elections.len());

    for (e_idx, &(col_a, col_b)) in elections.iter().enumerate() {
        let a_sum_a = sum_attr_over(graph, col_a, &proposal.a_nodes);
        let a_sum_b = sum_attr_over(graph, col_b, &proposal.a_nodes);
        let b_sum_a = sum_attr_over(graph, col_a, &proposal.b_nodes);
        let b_sum_b = sum_attr_over(graph, col_b, &proposal.b_nodes);

        let (new_a_target, new_a_other) = if target_a {
            (a_sum_a, a_sum_b)
        } else {
            (a_sum_b, a_sum_a)
        };
        let (new_b_target, new_b_other) = if target_a {
            (b_sum_a, b_sum_b)
        } else {
            (b_sum_b, b_sum_a)
        };

        let (wins, _best_dist, best_tb) = election_single_update(
            &state.target_votes[e_idx],
            &state.other_votes[e_idx],
            state.wins[e_idx],
            state.best_losing_dist[e_idx],
            state.best_losing_tiebreak[e_idx],
            a_label,
            b_label,
            new_a_target,
            new_a_other,
            new_b_target,
            new_b_other,
        );

        election_scores.push(wins as f64 + best_tb);
    }
    aggregation.apply(&election_scores)
}

pub(super) fn apply_proposal(
    graph: &Graph,
    state: &mut ElectionWinsState,
    elections: &[(&'static str, &'static str)],
    target_a: bool,
    aggregation: Aggregation,
    proposal: &RecomProposal,
) {
    let a_label = proposal.a_label;
    let b_label = proposal.b_label;

    for (e_idx, &(col_a, col_b)) in elections.iter().enumerate() {
        let a_sum_a = sum_attr_over(graph, col_a, &proposal.a_nodes);
        let a_sum_b = sum_attr_over(graph, col_b, &proposal.a_nodes);
        let b_sum_a = sum_attr_over(graph, col_a, &proposal.b_nodes);
        let b_sum_b = sum_attr_over(graph, col_b, &proposal.b_nodes);

        let (new_a_target, new_a_other) = if target_a {
            (a_sum_a, a_sum_b)
        } else {
            (a_sum_b, a_sum_a)
        };
        let (new_b_target, new_b_other) = if target_a {
            (b_sum_a, b_sum_b)
        } else {
            (b_sum_b, b_sum_a)
        };

        let (wins, best_dist, best_tb) = election_single_update(
            &state.target_votes[e_idx],
            &state.other_votes[e_idx],
            state.wins[e_idx],
            state.best_losing_dist[e_idx],
            state.best_losing_tiebreak[e_idx],
            a_label,
            b_label,
            new_a_target,
            new_a_other,
            new_b_target,
            new_b_other,
        );

        // Commit cached sums for the two changed districts.
        state.target_votes[e_idx][a_label] = new_a_target;
        state.other_votes[e_idx][a_label] = new_a_other;
        state.target_votes[e_idx][b_label] = new_b_target;
        state.other_votes[e_idx][b_label] = new_b_other;

        state.wins[e_idx] = wins;
        state.best_losing_dist[e_idx] = best_dist;
        state.best_losing_tiebreak[e_idx] = best_tb;
        state.election_scores[e_idx] = wins as f64 + best_tb;
    }
    state.score = aggregation.apply(&state.election_scores);
}

/// Per-district outcome for election-wins scoring.
///
/// Returns `(counted_as_win, tiebreak_if_loss)`. A district with zero total
/// votes is reported as `(false, None)` so callers skip it entirely -- they
/// must not treat it as a candidate for `best_losing_tiebreak`.
fn election_district_outcome(target: i32, other: i32) -> (bool, Option<f64>) {
    let total = target as f64 + other as f64;
    if total == 0.0 {
        return (false, None);
    }
    if target > other {
        (true, None)
    } else if target == other {
        (false, Some(ONE_BELOW))
    } else {
        (false, Some((target as f64 / total) / 0.5))
    }
}

/// Scans cached per-district vote totals and returns the maximum losing
/// tiebreak (0.0 if no losing districts exist) along with the district index
/// that holds it.
///
/// `skip_a` and `skip_b` are district indices whose cached values are known to
/// be stale (because a proposal is being applied to them) and must not be
/// considered.
fn scan_best_losing_tiebreak(
    target_votes: &[i32],
    other_votes: &[i32],
    skip_a: usize,
    skip_b: usize,
) -> (Option<usize>, f64) {
    let mut best_tb: f64 = 0.0;
    let mut best_dist: Option<usize> = None;
    for (d, (&t, &o)) in target_votes.iter().zip(other_votes.iter()).enumerate() {
        if d == skip_a || d == skip_b {
            continue;
        }
        let (_, tb) = election_district_outcome(t, o);
        if let Some(tb) = tb {
            if tb > best_tb {
                best_tb = tb;
                best_dist = Some(d);
            }
        }
    }
    (best_dist, best_tb)
}

/// Replays the post-proposal computation for a single election and returns
/// the resulting `(wins, best_losing_dist, best_losing_tiebreak)` triple.
///
/// `state_target` / `state_other` are the cached per-district sums for this
/// election; they are read but not mutated.
fn election_single_update(
    state_target: &[i32],
    state_other: &[i32],
    cached_wins: usize,
    cached_best_dist: Option<usize>,
    cached_best_tb: f64,
    a_label: usize,
    b_label: usize,
    new_a_target: i32,
    new_a_other: i32,
    new_b_target: i32,
    new_b_other: i32,
) -> (usize, Option<usize>, f64) {
    let old_a_target = state_target[a_label];
    let old_a_other = state_other[a_label];
    let old_b_target = state_target[b_label];
    let old_b_other = state_other[b_label];
    let (old_a_win, _old_a_tb) = election_district_outcome(old_a_target, old_a_other);
    let (old_b_win, _old_b_tb) = election_district_outcome(old_b_target, old_b_other);
    let (new_a_win, new_a_tb_opt) = election_district_outcome(new_a_target, new_a_other);
    let (new_b_win, new_b_tb_opt) = election_district_outcome(new_b_target, new_b_other);

    let mut wins = cached_wins;
    if old_a_win {
        wins -= 1;
    }
    if old_b_win {
        wins -= 1;
    }
    if new_a_win {
        wins += 1;
    }
    if new_b_win {
        wins += 1;
    }

    let new_a_loss_tb = new_a_tb_opt.unwrap_or(0.0);
    let new_b_loss_tb = new_b_tb_opt.unwrap_or(0.0);

    let (best_dist, best_tb) = match cached_best_dist {
        Some(holder) if holder != a_label && holder != b_label => {
            // Fast path: cached holder is unchanged. Compare against the two
            // new district tiebreaks.
            let mut dist = holder;
            let mut tb = cached_best_tb;
            if new_a_loss_tb > tb {
                tb = new_a_loss_tb;
                dist = a_label;
            }
            if new_b_loss_tb > tb {
                tb = new_b_loss_tb;
                dist = b_label;
            }
            // If the new max is 0.0 and the cached holder was a legitimate
            // losing district with tiebreak 0.0, we still hold that district.
            if tb == 0.0 && cached_best_tb == 0.0 && new_a_loss_tb == 0.0 && new_b_loss_tb == 0.0 {
                (Some(holder), 0.0)
            } else {
                (Some(dist), tb)
            }
        }
        _ => {
            // Slow path: rescan unchanged districts.
            let (mut dist, mut tb) =
                scan_best_losing_tiebreak(state_target, state_other, a_label, b_label);
            if new_a_loss_tb > tb || (dist.is_none() && new_a_tb_opt.is_some()) {
                if new_a_tb_opt.is_some() && new_a_loss_tb >= tb {
                    tb = new_a_loss_tb;
                    dist = Some(a_label);
                }
            }
            if new_b_loss_tb > tb || (dist.is_none() && new_b_tb_opt.is_some()) {
                if new_b_tb_opt.is_some() && new_b_loss_tb >= tb {
                    tb = new_b_loss_tb;
                    dist = Some(b_label);
                }
            }
            (dist, tb)
        }
    };

    (wins, best_dist, best_tb)
}
