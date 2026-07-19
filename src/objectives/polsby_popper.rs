//! `polsby_popper` objective: maximize a Polsby-Popper compactness aggregate
//! across districts.
//!
//! See [`super::ObjectiveConfig::PolsbyPopper`] for the JSON schema and full
//! scoring semantics. This module also owns the two public helpers the CLI uses
//! to prepare a graph for Polsby-Popper scoring: [`polsby_popper_autoderive`]
//! and [`ensure_derived_perim_column`].

use super::{Aggregation, ObjectiveConfig};
use crate::graph::Graph;
use crate::partition::Partition;
use crate::recom::RecomProposal;
use serde_json::Value;
use std::collections::HashSet;

/// Synthetic node-attribute column name used when a `polsby_popper` config
/// supplies `boundary_perim_col` but omits `perim_col` -- the loader writes
/// the derived total-perimeter values here and the objective reads them back
/// through the same key.
const DERIVED_PERIM_COL: &str = "__rustrecom_derived_perim";

/// Per-district cached state for an [`ObjectiveConfig::PolsbyPopper`] objective.
#[derive(Clone, Debug)]
pub struct PolsbyPopperState {
    pub(super) areas: Vec<f64>,
    pub(super) perimeters: Vec<f64>,
    pub(super) district_scores: Vec<f64>,
    pub(super) score: f64,
}

impl PolsbyPopperState {
    pub(super) fn init(
        graph: &Graph,
        partition: &Partition,
        area_col: &str,
        perim_col: &str,
        shared_perim_col: &str,
        aggregation: Aggregation,
    ) -> PolsbyPopperState {
        let area_vals = graph
            .attr
            .get(area_col)
            .unwrap_or_else(|| panic!("Missing node attribute '{}'", area_col));
        let perim_vals = graph
            .attr
            .get(perim_col)
            .unwrap_or_else(|| panic!("Missing node attribute '{}'", perim_col));
        let shared_perim_vals = graph
            .edge_attr
            .get(shared_perim_col)
            .unwrap_or_else(|| panic!("Missing edge attribute '{}'", shared_perim_col));

        let n_dists = partition.num_dists as usize;
        let mut areas = vec![0.0f64; n_dists];
        let mut perimeters = vec![0.0f64; n_dists];

        for (dist, nodes) in partition.dist_nodes.iter().enumerate() {
            for &node in nodes {
                areas[dist] += area_vals[node].parse::<f64>().unwrap_or(0.0);
                perimeters[dist] += perim_vals[node].parse::<f64>().unwrap_or(0.0);
            }
        }

        for (edge_idx, edge) in graph.edges.iter().enumerate() {
            let d_u = partition.assignments[edge.0] as usize;
            let d_v = partition.assignments[edge.1] as usize;
            if d_u == d_v {
                perimeters[d_u] -= 2.0 * shared_perim_vals[edge_idx];
            }
        }

        let district_scores: Vec<f64> = (0..n_dists)
            .map(|d| polsby_popper_score(areas[d], perimeters[d]))
            .collect();
        let score = aggregation.apply(&district_scores);
        PolsbyPopperState {
            areas,
            perimeters,
            district_scores,
            score,
        }
    }
}

/// Parses a `polsby_popper` objective config. See [`super::make_objective`].
pub(super) fn from_json(data: &Value) -> ObjectiveConfig {
    let agg_str = data["aggregation"]
        .as_str()
        .unwrap_or_else(|| panic!("Missing field 'aggregation' in objective config"));
    let boundary_perim_col = data
        .get("boundary_perim_col")
        .and_then(|v| v.as_str())
        .map(|s| &*Box::leak(s.to_owned().into_boxed_str()) as &'static str);
    let perim_col: &'static str = match data.get("perim_col").and_then(|v| v.as_str()) {
        Some(s) => &*Box::leak(s.to_owned().into_boxed_str()),
        None => {
            if boundary_perim_col.is_none() {
                panic!(
                    "polsby_popper config must set 'perim_col' (pre-baked total perimeter column) or 'boundary_perim_col' (auto-derive from shared_perim + boundary_perim)."
                );
            }
            DERIVED_PERIM_COL
        }
    };
    ObjectiveConfig::PolsbyPopper {
        area_col: super::leak_str(data, "area_col"),
        perim_col,
        shared_perim_col: super::leak_str(data, "shared_perim_col"),
        boundary_perim_col,
        aggregation: Aggregation::from_str(agg_str),
    }
}

/// Node attribute columns required by a `polsby_popper` config.
pub(super) fn required_node_cols(data: &Value) -> Vec<String> {
    let mut cols = vec![data["area_col"].as_str().unwrap().to_string()];
    if data
        .get("boundary_perim_col")
        .and_then(|v| v.as_str())
        .is_none()
    {
        let perim = data
            .get("perim_col")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| {
                panic!("polsby_popper config must set 'perim_col' or 'boundary_perim_col'.")
            });
        cols.push(perim.to_string());
    }
    cols
}

/// Node attribute columns that may be absent on some nodes (passed as
/// `partial_columns` to the loader). Polsby-Popper's `boundary_perim_col` is
/// defined only on boundary nodes.
pub(super) fn partial_node_cols(data: &Value) -> Vec<String> {
    data.get("boundary_perim_col")
        .and_then(|v| v.as_str())
        .map(|s| vec![s.to_string()])
        .unwrap_or_default()
}

/// Edge attribute columns required by a `polsby_popper` config.
pub(super) fn required_edge_cols(data: &Value) -> Vec<String> {
    vec![data["shared_perim_col"]
        .as_str()
        .unwrap_or_else(|| panic!("Missing 'shared_perim_col' in polsby_popper config"))
        .to_string()]
}

/// Full (non-incremental) score over every district.
pub(super) fn full_score(
    graph: &Graph,
    partition: &Partition,
    area_col: &str,
    perim_col: &str,
    shared_perim_col: &str,
    aggregation: Aggregation,
) -> f64 {
    let area_vals = graph
        .attr
        .get(area_col)
        .unwrap_or_else(|| panic!("Missing node attribute '{}'", area_col));
    let perim_vals = graph
        .attr
        .get(perim_col)
        .unwrap_or_else(|| panic!("Missing node attribute '{}'", perim_col));
    let shared_perim_vals = graph
        .edge_attr
        .get(shared_perim_col)
        .unwrap_or_else(|| panic!("Missing edge attribute '{}'", shared_perim_col));

    let n_dists = partition.num_dists as usize;
    let mut area_d = vec![0.0f64; n_dists];
    let mut perim_d = vec![0.0f64; n_dists];

    for (dist, nodes) in partition.dist_nodes.iter().enumerate() {
        for &node in nodes {
            area_d[dist] += area_vals[node].parse::<f64>().unwrap_or(0.0);
            perim_d[dist] += perim_vals[node].parse::<f64>().unwrap_or(0.0);
        }
    }

    // Subtract 2 * shared_perim for edges internal to each district
    // (cancels out the double-counted shared boundaries).
    for (edge_idx, edge) in graph.edges.iter().enumerate() {
        let d_u = partition.assignments[edge.0] as usize;
        let d_v = partition.assignments[edge.1] as usize;
        if d_u == d_v {
            perim_d[d_u] -= 2.0 * shared_perim_vals[edge_idx];
        }
    }

    let pp_scores: Vec<f64> = (0..n_dists)
        .map(|d| polsby_popper_score(area_d[d], perim_d[d]))
        .collect();

    aggregation.apply(&pp_scores)
}

pub(super) fn score_proposal(
    graph: &Graph,
    state: &PolsbyPopperState,
    area_col: &str,
    perim_col: &str,
    shared_perim_col: &str,
    aggregation: Aggregation,
    proposal: &RecomProposal,
) -> f64 {
    let shared_perim_vals = graph
        .edge_attr
        .get(shared_perim_col)
        .unwrap_or_else(|| panic!("Missing edge attribute '{}'", shared_perim_col));

    let a_set: HashSet<usize> = proposal.a_nodes.iter().copied().collect();
    let b_set: HashSet<usize> = proposal.b_nodes.iter().copied().collect();

    let (new_a_area, new_a_perim) = compute_proposal_district_geom(
        graph,
        &proposal.a_nodes,
        area_col,
        perim_col,
        shared_perim_vals,
        &a_set,
    );
    let (new_b_area, new_b_perim) = compute_proposal_district_geom(
        graph,
        &proposal.b_nodes,
        area_col,
        perim_col,
        shared_perim_vals,
        &b_set,
    );
    let new_a_score = polsby_popper_score(new_a_area, new_a_perim);
    let new_b_score = polsby_popper_score(new_b_area, new_b_perim);

    aggregate_with_replacements(
        aggregation,
        &state.district_scores,
        proposal.a_label,
        new_a_score,
        proposal.b_label,
        new_b_score,
    )
}

pub(super) fn apply_proposal(
    graph: &Graph,
    state: &mut PolsbyPopperState,
    area_col: &str,
    perim_col: &str,
    shared_perim_col: &str,
    aggregation: Aggregation,
    proposal: &RecomProposal,
) {
    let shared_perim_vals = graph
        .edge_attr
        .get(shared_perim_col)
        .unwrap_or_else(|| panic!("Missing edge attribute '{}'", shared_perim_col));

    let a_set: HashSet<usize> = proposal.a_nodes.iter().copied().collect();
    let b_set: HashSet<usize> = proposal.b_nodes.iter().copied().collect();

    let (new_a_area, new_a_perim) = compute_proposal_district_geom(
        graph,
        &proposal.a_nodes,
        area_col,
        perim_col,
        shared_perim_vals,
        &a_set,
    );
    let (new_b_area, new_b_perim) = compute_proposal_district_geom(
        graph,
        &proposal.b_nodes,
        area_col,
        perim_col,
        shared_perim_vals,
        &b_set,
    );

    state.areas[proposal.a_label] = new_a_area;
    state.areas[proposal.b_label] = new_b_area;
    state.perimeters[proposal.a_label] = new_a_perim;
    state.perimeters[proposal.b_label] = new_b_perim;
    state.district_scores[proposal.a_label] = polsby_popper_score(new_a_area, new_a_perim);
    state.district_scores[proposal.b_label] = polsby_popper_score(new_b_area, new_b_perim);
    state.score = aggregation.apply(&state.district_scores);
}

#[inline]
fn polsby_popper_score(area: f64, perimeter: f64) -> f64 {
    if perimeter <= 0.0 {
        0.0
    } else {
        4.0 * std::f64::consts::PI * area / (perimeter * perimeter)
    }
}

/// Computes `(area, perimeter)` for a proposed district given the raw node
/// lists of the new district. Perimeter uses a hash-set membership test over
/// `nodes` to identify internal edges.
fn compute_proposal_district_geom(
    graph: &Graph,
    nodes: &[usize],
    area_col: &str,
    perim_col: &str,
    shared_perim_vals: &[f64],
    node_set: &HashSet<usize>,
) -> (f64, f64) {
    let area_vals = graph
        .attr
        .get(area_col)
        .unwrap_or_else(|| panic!("Missing node attribute '{}'", area_col));
    let perim_vals = graph
        .attr
        .get(perim_col)
        .unwrap_or_else(|| panic!("Missing node attribute '{}'", perim_col));

    let mut area: f64 = 0.0;
    let mut raw_perim: f64 = 0.0;
    for &node in nodes {
        area += area_vals[node].parse::<f64>().unwrap_or(0.0);
        raw_perim += perim_vals[node].parse::<f64>().unwrap_or(0.0);
    }

    // Subtract 2 * shared_perim for edges internal to the proposed district.
    // Iterate over edges whose low endpoint is in `nodes`; each internal edge
    // is seen exactly once this way.
    let n_graph = graph.edges_start.len();
    let mut internal: f64 = 0.0;
    for &node in nodes {
        let start = graph.edges_start[node];
        let end = if node + 1 < n_graph {
            graph.edges_start[node + 1]
        } else {
            graph.edges.len()
        };
        for e_idx in start..end {
            let edge = graph.edges[e_idx];
            debug_assert_eq!(edge.0, node);
            if node_set.contains(&edge.1) {
                internal += shared_perim_vals[e_idx];
            }
        }
    }
    (area, raw_perim - 2.0 * internal)
}

/// Aggregates `scores` with two district entries replaced by the given new
/// values, without mutating `scores`.
fn aggregate_with_replacements(
    aggregation: Aggregation,
    scores: &[f64],
    a: usize,
    new_a: f64,
    b: usize,
    new_b: f64,
) -> f64 {
    match aggregation {
        Aggregation::Sum | Aggregation::Mean => {
            let mut total = 0.0f64;
            for (i, &s) in scores.iter().enumerate() {
                let v = if i == a {
                    new_a
                } else if i == b {
                    new_b
                } else {
                    s
                };
                total += v;
            }
            if matches!(aggregation, Aggregation::Mean) {
                total / scores.len() as f64
            } else {
                total
            }
        }
        Aggregation::Min => {
            let mut m = f64::INFINITY;
            for (i, &s) in scores.iter().enumerate() {
                let v = if i == a {
                    new_a
                } else if i == b {
                    new_b
                } else {
                    s
                };
                if v < m {
                    m = v;
                }
            }
            m
        }
    }
}

/// Inspects a `polsby_popper` objective config for the auto-derivation
/// triple `(perim_col, boundary_perim_col, shared_perim_col)`.
///
/// Returns `Some(...)` iff the objective is `polsby_popper` and carries a
/// `boundary_perim_col` field, in which case the CLI should call
/// [`ensure_derived_perim_column`] on the loaded graph before running the
/// chain. Returns `None` for every other objective and for Polsby-Popper
/// configs that omit `boundary_perim_col`.
pub fn polsby_popper_autoderive(config: &str) -> Option<(String, String, String)> {
    let data: Value = serde_json::from_str(config).ok()?;
    if data.get("objective").and_then(|v| v.as_str())? != "polsby_popper" {
        return None;
    }
    let boundary = data
        .get("boundary_perim_col")
        .and_then(|v| v.as_str())?
        .to_string();
    let perim = data
        .get("perim_col")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| DERIVED_PERIM_COL.to_string());
    let shared = data
        .get("shared_perim_col")
        .and_then(|v| v.as_str())?
        .to_string();
    Some((perim, boundary, shared))
}

/// Derives a node-level total perimeter column in-place on `graph` from the
/// per-edge `shared_perim_col` and per-node `boundary_perim_col`.
///
/// For each node `n`, writes `graph.attr[perim_col][n]` =
/// `boundary_perim[n] + sum(shared_perim[edge] for edges incident to n)`,
/// where missing or non-numeric boundary entries are treated as zero. This
/// reconstructs the total geometric perimeter of each precinct (outer-hull
/// contribution plus shared boundary with every neighbor) that
/// `ObjectiveConfig::PolsbyPopper` expects as `perim_col`.
///
/// Any prior value stored under `perim_col` is overwritten. `shared_perim_col`
/// must already exist in `graph.edge_attr` (loaded via `required_edge_cols`);
/// `boundary_perim_col` must already exist in `graph.attr`.
pub fn ensure_derived_perim_column(
    graph: &mut Graph,
    perim_col: &str,
    boundary_perim_col: &str,
    shared_perim_col: &str,
) {
    let shared_perim_vals: Vec<f64> = graph
        .edge_attr
        .get(shared_perim_col)
        .unwrap_or_else(|| {
            panic!(
                "Cannot derive '{}': missing edge attribute '{}'",
                perim_col, shared_perim_col
            )
        })
        .clone();
    let boundary_vals: Vec<f64> = graph
        .attr
        .get(boundary_perim_col)
        .unwrap_or_else(|| {
            panic!(
                "Cannot derive '{}': missing node attribute '{}'",
                perim_col, boundary_perim_col
            )
        })
        .iter()
        .map(|s| s.parse::<f64>().unwrap_or(0.0))
        .collect();

    let n = graph.pops.len();
    assert_eq!(
        boundary_vals.len(),
        n,
        "boundary_perim column length does not match node count"
    );
    assert_eq!(
        shared_perim_vals.len(),
        graph.edges.len(),
        "shared_perim column length does not match edge count"
    );

    let mut perim_vals = boundary_vals;
    for (edge_idx, edge) in graph.edges.iter().enumerate() {
        let w = shared_perim_vals[edge_idx];
        perim_vals[edge.0] += w;
        perim_vals[edge.1] += w;
    }

    let as_strings: Vec<String> = perim_vals.iter().map(|v| format!("{}", v)).collect();
    graph.attr.insert(perim_col.to_string(), as_strings);
}
