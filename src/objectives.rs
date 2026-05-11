//! Objective (score) functions for partition optimization.
//!
//! Each objective is configured via a JSON string passed to [`make_objective_fn`]
//! (the closure-based legacy API) or [`make_objective`] (the incremental API).
//!
//! The legacy API evaluates the full objective over every district on every
//! call. The incremental API caches per-district summaries in an
//! [`ObjectiveState`] and updates only the two districts affected by a
//! [`RecomProposal`], avoiding node-count scans on each candidate.
//!
//! The JSON schema for each objective is documented on its corresponding variant
//! of [`ObjectiveConfig`].
use crate::graph::Graph;
use crate::partition::Partition;
use crate::recom::RecomProposal;
use crate::stats::partition_attr_sums;
use serde_json::Value;
use std::collections::HashSet;

/// Aggregation method for per-district scores.
#[derive(Clone, Copy, Debug)]
pub enum Aggregation {
    Mean,
    Min,
    Sum,
}

/// Largest `f64` strictly smaller than 1.0.
const ONE_BELOW: f64 = f64::from_bits(1.0f64.to_bits() - 1);

/// Synthetic node-attribute column name used when a `polsby_popper` config
/// supplies `boundary_perim_col` but omits `perim_col` -- the loader writes
/// the derived total-perimeter values here and the objective reads them back
/// through the same key.
const DERIVED_PERIM_COL: &str = "__frcw_derived_perim";

impl Aggregation {
    fn from_str(s: &str) -> Aggregation {
        match s {
            "mean" => Aggregation::Mean,
            "min" => Aggregation::Min,
            "sum" => Aggregation::Sum,
            other => panic!(
                "Unknown aggregation '{}'. Use 'mean', 'min', or 'sum'.",
                other
            ),
        }
    }

    fn apply(&self, values: &[f64]) -> f64 {
        match self {
            Aggregation::Mean => values.iter().sum::<f64>() / values.len() as f64,
            Aggregation::Min => values.iter().cloned().fold(f64::INFINITY, f64::min),
            Aggregation::Sum => values.iter().sum(),
        }
    }
}

/// A parsed, `Copy`-able representation of an objective function's configuration.
/// The JSON schema for each variant is documented below.
#[derive(Clone, Copy)]
pub enum ObjectiveConfig {
    /// Maximize Gingles opportunity districts with next-partial-district augmentation.
    ///
    /// The score is the number of districts where the minority share exceeds
    /// `threshold`, plus a fractional tiebreaker from the highest sub-threshold
    /// district (so that plans closer to gaining a new opportunity district are
    /// preferred).
    ///
    /// JSON schema:
    /// ```json
    /// {
    ///   "objective": "gingles_partial",
    ///   "threshold": 0.5,
    ///   "min_pop": "BVAP",
    ///   "total_pop": "VAP"
    /// }
    /// ```
    ///
    /// Fields:
    /// - `threshold`: minority share threshold, must be in (0, 1)
    /// - `min_pop`: node attribute column for the minority population (integer-valued)
    /// - `total_pop`: node attribute column for the total population (integer-valued)
    GinglesPartial {
        threshold: f64,
        min_pop_col: &'static str,
        total_pop_col: &'static str,
    },

    /// Maximize (or minimize) the number of districts won by a target party
    /// across a set of elections, with a partial-district tiebreaker.
    ///
    /// For each election, the score is the number of districts where the target
    /// party's vote total exceeds the other party's, plus a fractional tiebreaker
    /// from the closest losing district (so the optimizer has gradient signal
    /// between integer win counts). Per-election scores are aggregated via
    /// `aggregation`.
    ///
    /// JSON schema:
    /// ```json
    /// {
    ///   "objective": "election_wins",
    ///   "elections": [
    ///     {"votes_a": "DEM_GOV_18", "votes_b": "REP_GOV_18"},
    ///     {"votes_a": "DEM_SEN_18", "votes_b": "REP_SEN_18"}
    ///   ],
    ///   "target": "a",
    ///   "aggregation": "mean"
    /// }
    /// ```
    ///
    /// Fields:
    /// - `elections`: array of objects, each with `votes_a` and `votes_b` naming
    ///   node attribute columns containing integer vote counts
    /// - `target`: `"a"` or `"b"` -- which party's wins to count
    /// - `aggregation`: one of `"mean"`, `"min"`, or `"sum"`
    ElectionWins {
        elections: &'static [(&'static str, &'static str)],
        target_a: bool,
        aggregation: Aggregation,
    },

    /// Maximize a Polsby-Popper compactness aggregate across districts.
    ///
    /// The Polsby-Popper score for a district is `4 * pi * area / perimeter^2`.
    /// Per-district scores are aggregated via `aggregation`.
    ///
    /// JSON schema:
    /// ```json
    /// {
    ///   "objective": "polsby_popper",
    ///   "area_col": "area",
    ///   "perim_col": "perim",
    ///   "shared_perim_col": "shared_perim",
    ///   "aggregation": "mean"
    /// }
    /// ```
    ///
    /// Fields:
    /// - `area_col`: node attribute (float-parseable string) giving precinct area
    /// - `perim_col`: node attribute (float-parseable string) giving the total
    ///   perimeter of each precinct (including shared boundaries with neighbors).
    ///   Optional in the JSON when `boundary_perim_col` is supplied; the loader
    ///   derives into a synthetic internal column in that case.
    /// - `shared_perim_col`: edge attribute (`graph.edge_attr`) giving the shared
    ///   perimeter between adjacent precincts; must be loaded via `required_edge_cols`
    ///   and passed to `from_networkx`
    /// - `boundary_perim_col`: optional node attribute naming each precinct's
    ///   outer-hull contribution (nonzero only on boundary nodes). When set, the
    ///   loader will auto-derive the total-perimeter column from
    ///   `shared_perim_col` plus this boundary column. If `perim_col` is also
    ///   given, the derived values are written into that column (overwriting
    ///   any prior value); if `perim_col` is omitted, a synthetic internal
    ///   column name is used. One of `perim_col` or `boundary_perim_col`
    ///   must always be set.
    /// - `aggregation`: one of `"mean"`, `"min"`, or `"sum"`
    PolsbyPopper {
        area_col: &'static str,
        perim_col: &'static str,
        shared_perim_col: &'static str,
        boundary_perim_col: Option<&'static str>,
        aggregation: Aggregation,
    },
}

impl ObjectiveConfig {
    /// Evaluates the objective for the given graph and partition.
    pub fn score(&self, graph: &Graph, partition: &Partition) -> f64 {
        match self {
            ObjectiveConfig::GinglesPartial {
                threshold,
                min_pop_col,
                total_pop_col,
            } => {
                let min_pops = partition_attr_sums(graph, partition, min_pop_col);
                let total_pops = partition_attr_sums(graph, partition, total_pop_col);
                let shares: Vec<f64> = min_pops
                    .iter()
                    .zip(total_pops.iter())
                    .map(|(&m, &t)| m as f64 / t as f64)
                    .collect();
                let opportunity_count = shares.iter().filter(|&s| s >= threshold).count();
                let mut sorted_below: Vec<f64> =
                    shares.into_iter().filter(|s| s < threshold).collect();
                sorted_below.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let next_highest = sorted_below.last().copied().unwrap_or(0.0);
                opportunity_count as f64 + (next_highest / threshold)
            }

            ObjectiveConfig::ElectionWins {
                elections,
                target_a,
                aggregation,
            } => {
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
                        let target_votes = if *target_a { va } else { vb };
                        let other_votes = if *target_a { vb } else { va };
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

            ObjectiveConfig::PolsbyPopper {
                area_col,
                perim_col,
                shared_perim_col,
                boundary_perim_col: _,
                aggregation,
            } => {
                let area_vals = graph
                    .attr
                    .get(*area_col)
                    .unwrap_or_else(|| panic!("Missing node attribute '{}'", area_col));
                let perim_vals = graph
                    .attr
                    .get(*perim_col)
                    .unwrap_or_else(|| panic!("Missing node attribute '{}'", perim_col));
                let shared_perim_vals = graph
                    .edge_attr
                    .get(*shared_perim_col)
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
                    .map(|d| {
                        let p = perim_d[d];
                        if p <= 0.0 {
                            0.0
                        } else {
                            4.0 * std::f64::consts::PI * area_d[d] / (p * p)
                        }
                    })
                    .collect();

                aggregation.apply(&pp_scores)
            }
        }
    }

    /// Parses and caches the integer node attribute columns required by this
    /// objective. Call this once on the mutable graph before starting the chain.
    ///
    /// For [`ObjectiveConfig::ElectionWins`] and [`ObjectiveConfig::GinglesPartial`],
    /// this populates `graph.int_attr` with the parsed vote/population columns so
    /// that the hot-path incremental scoring functions can index directly into
    /// `Vec<i32>` rather than re-parsing strings on every proposal.
    ///
    /// Panics if any required column is missing or contains a non-integer value.
    pub fn cache_graph_cols(&self, graph: &mut Graph) {
        match self {
            ObjectiveConfig::ElectionWins { elections, .. } => {
                for &(col_a, col_b) in elections.iter() {
                    graph.cache_int_col(col_a);
                    graph.cache_int_col(col_b);
                }
            }
            ObjectiveConfig::GinglesPartial {
                min_pop_col,
                total_pop_col,
                ..
            } => {
                graph.cache_int_col(min_pop_col);
                graph.cache_int_col(total_pop_col);
            }
            ObjectiveConfig::PolsbyPopper { .. } => {}
        }
    }
}

/// Parses an objective configuration JSON string into a `Copy` [`ObjectiveConfig`].
///
/// Dispatches on the `"objective"` field:
/// - `"gingles_partial"` -- see [`ObjectiveConfig::GinglesPartial`]
/// - `"election_wins"` -- see [`ObjectiveConfig::ElectionWins`]
/// - `"polsby_popper"` -- see [`ObjectiveConfig::PolsbyPopper`]
pub fn make_objective(config: &str) -> ObjectiveConfig {
    let data: Value = serde_json::from_str(config).unwrap();
    let obj_type = data["objective"].as_str().unwrap();

    // Use Box::leak so that &'static str column names can be captured by Copy closures.
    let leak_str = |v: &Value, field: &str| -> &'static str {
        &*Box::leak(
            v[field]
                .as_str()
                .unwrap_or_else(|| panic!("Missing field '{}' in objective config", field))
                .to_owned()
                .into_boxed_str(),
        )
    };

    match obj_type {
        "gingles_partial" => {
            let threshold = data["threshold"].as_f64().unwrap();
            assert!(
                threshold > 0.0 && threshold < 1.0,
                "'threshold' must be in (0, 1)"
            );
            ObjectiveConfig::GinglesPartial {
                threshold,
                min_pop_col: leak_str(&data, "min_pop"),
                total_pop_col: leak_str(&data, "total_pop"),
            }
        }
        "election_wins" => {
            let elections_arr = data["elections"]
                .as_array()
                .unwrap_or_else(|| panic!("Missing field 'elections' in objective config"));
            let mut pairs: Vec<(&'static str, &'static str)> =
                Vec::with_capacity(elections_arr.len());
            for e in elections_arr {
                let a = leak_str(e, "votes_a");
                let b = leak_str(e, "votes_b");
                pairs.push((a, b));
            }
            let elections: &'static [(&'static str, &'static str)] =
                Box::leak(pairs.into_boxed_slice());
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
        "polsby_popper" => {
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
                area_col: leak_str(&data, "area_col"),
                perim_col,
                shared_perim_col: leak_str(&data, "shared_perim_col"),
                boundary_perim_col,
                aggregation: Aggregation::from_str(agg_str),
            }
        }
        other => panic!(
            "Unknown objective '{}'. Supported: 'gingles_partial', 'election_wins', 'polsby_popper'.",
            other
        ),
    }
}

/// Returns a `Copy + Clone + Send` closure that scores a partition according
/// to the given JSON objective configuration.
///
/// This is the legacy full-score API. For incremental scoring that caches
/// per-district state and only rescans the two districts a `RecomProposal`
/// changes, use [`make_objective`] and the [`IncrementalObjective`] trait.
pub fn make_objective_fn(config: &str) -> impl Fn(&Graph, &Partition) -> f64 + Send + Clone + Copy {
    let obj = make_objective(config);
    move |graph: &Graph, partition: &Partition| -> f64 { obj.score(graph, partition) }
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

/// Returns the node attribute columns required by the given objective config.
///
/// Use this in CLI binaries to ensure these columns are included in the
/// `columns` argument to [`frcw::init::from_networkx`].
pub fn required_node_cols(config: &str) -> Vec<String> {
    let data: Value = serde_json::from_str(config).unwrap();
    match data["objective"].as_str().unwrap() {
        "gingles_partial" => vec![
            data["min_pop"].as_str().unwrap().to_string(),
            data["total_pop"].as_str().unwrap().to_string(),
        ],
        "election_wins" => {
            let mut cols = vec![];
            for e in data["elections"].as_array().unwrap() {
                cols.push(e["votes_a"].as_str().unwrap().to_string());
                cols.push(e["votes_b"].as_str().unwrap().to_string());
            }
            cols
        }
        "polsby_popper" => {
            let mut cols = vec![data["area_col"].as_str().unwrap().to_string()];
            if data.get("boundary_perim_col").and_then(|v| v.as_str()).is_none() {
                let perim = data
                    .get("perim_col")
                    .and_then(|v| v.as_str())
                    .unwrap_or_else(|| {
                        panic!(
                            "polsby_popper config must set 'perim_col' or 'boundary_perim_col'."
                        )
                    });
                cols.push(perim.to_string());
            }
            cols
        }
        _ => vec![],
    }
}

/// Returns the node attribute columns that the objective config references
/// but which may be absent on some nodes. These should be passed to
/// [`frcw::init::from_networkx`] as `partial_columns`: missing entries are
/// stored as `"null"` rather than panicking.
///
/// Today this only returns Polsby-Popper's `boundary_perim_col` (when set),
/// because boundary perimeter is by definition defined only on boundary
/// nodes.
pub fn partial_node_cols(config: &str) -> Vec<String> {
    let data: Value = match serde_json::from_str(config) {
        Ok(v) => v,
        Err(_) => return vec![],
    };
    match data.get("objective").and_then(|v| v.as_str()) {
        Some("polsby_popper") => data
            .get("boundary_perim_col")
            .and_then(|v| v.as_str())
            .map(|s| vec![s.to_string()])
            .unwrap_or_default(),
        _ => vec![],
    }
}

/// Returns the edge attribute columns required by the given objective config.
///
/// Use this in CLI binaries to determine which columns to pass as `edge_float_cols`
/// to [`frcw::init::from_networkx`].
pub fn required_edge_cols(config: &str) -> Vec<String> {
    let data: Value = serde_json::from_str(config).unwrap();
    match data["objective"].as_str().unwrap() {
        "polsby_popper" => vec![data["shared_perim_col"]
            .as_str()
            .unwrap_or_else(|| panic!("Missing 'shared_perim_col' in polsby_popper config"))
            .to_string()],
        _ => vec![],
    }
}

// =====================================================================
// Incremental scoring
// =====================================================================

/// Per-district cached state for an [`ObjectiveConfig::ElectionWins`] objective.
///
/// The inner `target_votes` / `other_votes` vectors are indexed
/// `[election_index][district_index]`. `wins`, `best_losing_dist`,
/// `best_losing_tiebreak`, and `election_scores` are indexed
/// `[election_index]`.
#[derive(Clone, Debug)]
pub struct ElectionWinsState {
    target_votes: Vec<Vec<i32>>,
    other_votes: Vec<Vec<i32>>,
    wins: Vec<usize>,
    best_losing_dist: Vec<Option<usize>>,
    best_losing_tiebreak: Vec<f64>,
    election_scores: Vec<f64>,
    score: f64,
}

/// Per-district cached state for an [`ObjectiveConfig::GinglesPartial`] objective.
#[derive(Clone, Debug)]
pub struct GinglesPartialState {
    min_pops: Vec<i32>,
    total_pops: Vec<i32>,
    opportunity_count: usize,
    best_below_dist: Option<usize>,
    best_below_share: f64,
    score: f64,
}

/// Per-district cached state for an [`ObjectiveConfig::PolsbyPopper`] objective.
#[derive(Clone, Debug)]
pub struct PolsbyPopperState {
    areas: Vec<f64>,
    perimeters: Vec<f64>,
    district_scores: Vec<f64>,
    score: f64,
}

/// Tagged-union cached state for any [`ObjectiveConfig`] variant.
#[derive(Clone, Debug)]
pub enum ObjectiveState {
    ElectionWins(ElectionWinsState),
    GinglesPartial(GinglesPartialState),
    PolsbyPopper(PolsbyPopperState),
}

/// An objective that supports incremental scoring from cached per-district state.
///
/// Tilted runs create one cached `State` at chain start, score candidate
/// proposals against it without mutating it, and mutate it only when the main
/// chain accepts a proposal. This avoids rescoring every district for every
/// candidate proposal.
///
/// The trait method is named `score_state` (not `score`) to avoid shadowing
/// the existing inherent `ObjectiveConfig::score(graph, partition)` full-score
/// method.
pub trait IncrementalObjective: Send + Clone {
    /// Cached per-district state.
    type State: Send + Clone;

    /// Builds initial cached state for `partition`.
    fn init(&self, graph: &Graph, partition: &Partition) -> Self::State;

    /// Returns the aggregate score represented by `state`.
    fn score_state(&self, state: &Self::State) -> f64;

    /// Returns the score that would result after applying `proposal` to the
    /// partition represented by `current`. Must not mutate `current`.
    fn score_proposal(
        &self,
        graph: &Graph,
        current: &Self::State,
        proposal: &RecomProposal,
    ) -> f64;

    /// Mutates `state` to reflect applying `proposal`. Must produce state
    /// equivalent to rebuilding from the updated partition via [`Self::init`].
    fn apply_proposal(
        &self,
        graph: &Graph,
        state: &mut Self::State,
        proposal: &RecomProposal,
    );

    /// Returns a per-district score vector describing `state`, suitable for
    /// emitting alongside the aggregate score. The length of the returned
    /// vector should equal the partition's district count for objectives that
    /// have a natural per-district decomposition (e.g. Polsby-Popper); an
    /// empty vector indicates the objective does not expose per-district
    /// values.
    fn district_scores(&self, state: &Self::State) -> Vec<f64>;

    /// Convenience: score a full partition from scratch via the cached path.
    fn score_partition(&self, graph: &Graph, partition: &Partition) -> f64 {
        self.score_state(&self.init(graph, partition))
    }
}

/// Sums a pre-cached integer node attribute over a set of nodes.
/// Panics if `col` has not been registered via [`Graph::cache_int_col`]
/// (i.e. via [`ObjectiveConfig::cache_graph_cols`] before the chain starts).
fn sum_attr_over(graph: &Graph, col: &str, nodes: &[usize]) -> i32 {
    let values = graph.int_attr.get(col).unwrap_or_else(|| {
        panic!(
            "Column '{}' has not been pre-cached; call cache_graph_cols before running the chain",
            col
        )
    });
    nodes.iter().map(|&n| values[n]).sum()
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

impl ElectionWinsState {
    fn init(
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

impl GinglesPartialState {
    fn init(
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

impl PolsbyPopperState {
    fn init(
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

        let district_scores: Vec<f64> = (0..n_dists).map(|d| polsby_popper_score(areas[d], perimeters[d])).collect();
        let score = aggregation.apply(&district_scores);
        PolsbyPopperState {
            areas,
            perimeters,
            district_scores,
            score,
        }
    }
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

impl IncrementalObjective for ObjectiveConfig {
    type State = ObjectiveState;

    fn init(&self, graph: &Graph, partition: &Partition) -> ObjectiveState {
        match *self {
            ObjectiveConfig::ElectionWins {
                elections,
                target_a,
                aggregation,
            } => ObjectiveState::ElectionWins(ElectionWinsState::init(
                graph,
                partition,
                elections,
                target_a,
                aggregation,
            )),
            ObjectiveConfig::GinglesPartial {
                threshold,
                min_pop_col,
                total_pop_col,
            } => ObjectiveState::GinglesPartial(GinglesPartialState::init(
                graph,
                partition,
                threshold,
                min_pop_col,
                total_pop_col,
            )),
            ObjectiveConfig::PolsbyPopper {
                area_col,
                perim_col,
                shared_perim_col,
                boundary_perim_col: _,
                aggregation,
            } => ObjectiveState::PolsbyPopper(PolsbyPopperState::init(
                graph,
                partition,
                area_col,
                perim_col,
                shared_perim_col,
                aggregation,
            )),
        }
    }

    fn score_state(&self, state: &ObjectiveState) -> f64 {
        match (self, state) {
            (ObjectiveConfig::ElectionWins { .. }, ObjectiveState::ElectionWins(s)) => s.score,
            (ObjectiveConfig::GinglesPartial { .. }, ObjectiveState::GinglesPartial(s)) => s.score,
            (ObjectiveConfig::PolsbyPopper { .. }, ObjectiveState::PolsbyPopper(s)) => s.score,
            _ => panic!("Objective/state variant mismatch"),
        }
    }

    fn score_proposal(
        &self,
        graph: &Graph,
        current: &ObjectiveState,
        proposal: &RecomProposal,
    ) -> f64 {
        match (self, current) {
            (
                ObjectiveConfig::ElectionWins {
                    elections,
                    target_a,
                    aggregation,
                },
                ObjectiveState::ElectionWins(state),
            ) => score_proposal_election_wins(
                graph, state, elections, *target_a, *aggregation, proposal,
            ),
            (
                ObjectiveConfig::GinglesPartial {
                    threshold,
                    min_pop_col,
                    total_pop_col,
                },
                ObjectiveState::GinglesPartial(state),
            ) => score_proposal_gingles(
                graph,
                state,
                *threshold,
                min_pop_col,
                total_pop_col,
                proposal,
            ),
            (
                ObjectiveConfig::PolsbyPopper {
                    area_col,
                    perim_col,
                    shared_perim_col,
                    boundary_perim_col: _,
                    aggregation,
                },
                ObjectiveState::PolsbyPopper(state),
            ) => score_proposal_polsby_popper(
                graph,
                state,
                area_col,
                perim_col,
                shared_perim_col,
                *aggregation,
                proposal,
            ),
            _ => panic!("Objective/state variant mismatch"),
        }
    }

    fn apply_proposal(
        &self,
        graph: &Graph,
        state: &mut ObjectiveState,
        proposal: &RecomProposal,
    ) {
        match (self, state) {
            (
                ObjectiveConfig::ElectionWins {
                    elections,
                    target_a,
                    aggregation,
                },
                ObjectiveState::ElectionWins(s),
            ) => apply_proposal_election_wins(
                graph, s, elections, *target_a, *aggregation, proposal,
            ),
            (
                ObjectiveConfig::GinglesPartial {
                    threshold,
                    min_pop_col,
                    total_pop_col,
                },
                ObjectiveState::GinglesPartial(s),
            ) => apply_proposal_gingles(
                graph,
                s,
                *threshold,
                min_pop_col,
                total_pop_col,
                proposal,
            ),
            (
                ObjectiveConfig::PolsbyPopper {
                    area_col,
                    perim_col,
                    shared_perim_col,
                    boundary_perim_col: _,
                    aggregation,
                },
                ObjectiveState::PolsbyPopper(s),
            ) => apply_proposal_polsby_popper(
                graph,
                s,
                area_col,
                perim_col,
                shared_perim_col,
                *aggregation,
                proposal,
            ),
            _ => panic!("Objective/state variant mismatch"),
        }
    }

    fn district_scores(&self, state: &ObjectiveState) -> Vec<f64> {
        match (self, state) {
            (ObjectiveConfig::PolsbyPopper { .. }, ObjectiveState::PolsbyPopper(s)) => {
                s.district_scores.clone()
            }
            (ObjectiveConfig::GinglesPartial { .. }, ObjectiveState::GinglesPartial(s)) => s
                .min_pops
                .iter()
                .zip(s.total_pops.iter())
                .map(|(&m, &t)| if t == 0 { 0.0 } else { m as f64 / t as f64 })
                .collect(),
            (ObjectiveConfig::ElectionWins { .. }, ObjectiveState::ElectionWins(_)) => Vec::new(),
            _ => panic!("Objective/state variant mismatch"),
        }
    }
}

// ---------- election_wins incremental ----------

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

fn score_proposal_election_wins(
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

fn apply_proposal_election_wins(
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

// ---------- gingles_partial incremental ----------

fn score_proposal_gingles(
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

fn apply_proposal_gingles(
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

fn district_share(min_pop: i32, total_pop: i32) -> f64 {
    min_pop as f64 / total_pop as f64
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

// ---------- polsby_popper incremental ----------

fn score_proposal_polsby_popper(
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

fn apply_proposal_polsby_popper(
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

#[cfg(test)]
mod incremental_tests {
    use super::*;

    /// Builds a 4x4 rect grid with synthetic node and edge attributes.
    ///
    /// Column-major node indexing (nodes 0..=15). The grid is partitioned
    /// into four 2x2 quadrant districts of four nodes each.
    fn make_test_graph_and_partition() -> (Graph, Partition) {
        let mut graph = Graph::rect_grid(4, 4);
        let n = graph.pops.len();

        // Vary populations so test totals are nonuniform.
        graph.pops = (0..n).map(|i| (i + 1) as u32 * 10).collect();
        graph.total_pop = graph.pops.iter().sum();

        // Synthetic integer attributes for election_wins / gingles_partial.
        let dem: Vec<String> = (0..n).map(|i| ((i * 7) % 50 + 10).to_string()).collect();
        let rep: Vec<String> = (0..n).map(|i| ((i * 11) % 55 + 5).to_string()).collect();
        let bvap: Vec<String> = (0..n).map(|i| ((i * 5) % 20 + 1).to_string()).collect();
        let vap: Vec<String> = (0..n).map(|i| ((i * 13) % 40 + 30).to_string()).collect();
        // Float-parseable attributes for polsby_popper.
        let area: Vec<String> = (0..n).map(|i| format!("{}", (i + 1) as f64 * 1.5)).collect();
        let perim: Vec<String> = (0..n).map(|i| format!("{}", (i + 3) as f64 * 2.25)).collect();

        graph.attr.insert("dem".to_string(), dem);
        graph.attr.insert("rep".to_string(), rep);
        graph.attr.insert("bvap".to_string(), bvap);
        graph.attr.insert("vap".to_string(), vap);
        graph.attr.insert("area".to_string(), area);
        graph.attr.insert("perim".to_string(), perim);

        // Synthetic edge attribute for polsby_popper.
        let shared_perim: Vec<f64> = graph
            .edges
            .iter()
            .enumerate()
            .map(|(i, _)| 0.5 + (i as f64) * 0.1)
            .collect();
        graph.edge_attr.insert("shared_perim".to_string(), shared_perim);

        // 4x4 grid, 2x2 quadrant partition. Column-major indexing.
        // col 0: nodes 0..=3, col 1: 4..=7, col 2: 8..=11, col 3: 12..=15.
        // Quadrant layout (1-indexed districts):
        //   col 0-1, row 0-1: d1 -> nodes 0,1,4,5
        //   col 0-1, row 2-3: d2 -> nodes 2,3,6,7
        //   col 2-3, row 0-1: d3 -> nodes 8,9,12,13
        //   col 2-3, row 2-3: d4 -> nodes 10,11,14,15
        let mut assignments = vec![0u32; n];
        for node in [0, 1, 4, 5] {
            assignments[node] = 1;
        }
        for node in [2, 3, 6, 7] {
            assignments[node] = 2;
        }
        for node in [8, 9, 12, 13] {
            assignments[node] = 3;
        }
        for node in [10, 11, 14, 15] {
            assignments[node] = 4;
        }
        let partition = Partition::from_assignments(&graph, &assignments).unwrap();
        (graph, partition)
    }

    /// Builds a valid ReCom proposal that moves a single boundary node from
    /// `dist_take` to `dist_give`. Both districts must currently exist.
    fn boundary_swap_proposal(
        partition: &Partition,
        dist_take: usize,
        dist_give: usize,
        node_to_move: usize,
    ) -> RecomProposal {
        let mut a_nodes = partition.dist_nodes[dist_take].clone();
        a_nodes.retain(|&n| n != node_to_move);
        let mut b_nodes = partition.dist_nodes[dist_give].clone();
        b_nodes.push(node_to_move);
        RecomProposal {
            a_label: dist_take,
            b_label: dist_give,
            a_pop: 0, // unused for scoring
            b_pop: 0,
            a_nodes,
            b_nodes,
        }
    }

    fn static_str(s: &str) -> &'static str {
        Box::leak(s.to_owned().into_boxed_str())
    }

    fn test_election_wins_config() -> ObjectiveConfig {
        let pair: &'static [(&'static str, &'static str)] =
            Box::leak(vec![(static_str("dem"), static_str("rep"))].into_boxed_slice());
        ObjectiveConfig::ElectionWins {
            elections: pair,
            target_a: true,
            aggregation: Aggregation::Mean,
        }
    }

    fn test_gingles_config() -> ObjectiveConfig {
        ObjectiveConfig::GinglesPartial {
            threshold: 0.5,
            min_pop_col: static_str("bvap"),
            total_pop_col: static_str("vap"),
        }
    }

    fn test_polsby_popper_config() -> ObjectiveConfig {
        ObjectiveConfig::PolsbyPopper {
            area_col: static_str("area"),
            perim_col: static_str("perim"),
            shared_perim_col: static_str("shared_perim"),
            boundary_perim_col: None,
            aggregation: Aggregation::Mean,
        }
    }

    fn assert_close(lhs: f64, rhs: f64, label: &str) {
        let tol = 1e-9 * (1.0 + lhs.abs() + rhs.abs());
        assert!(
            (lhs - rhs).abs() <= tol,
            "{}: {} vs {} (diff {})",
            label,
            lhs,
            rhs,
            (lhs - rhs).abs()
        );
    }

    fn run_equivalence_suite(obj: ObjectiveConfig) {
        let (mut graph, partition) = make_test_graph_and_partition();
        obj.cache_graph_cols(&mut graph);

        // Cached init score must match full-score.
        let state = obj.init(&graph, &partition);
        let cached_initial = obj.score_state(&state);
        let full_initial = obj.score(&graph, &partition);
        assert_close(cached_initial, full_initial, "init_score");

        // (dist_take, dist_give, node_to_move) swaps across quadrant boundaries.
        let swaps: &[(usize, usize, usize)] = &[
            (1, 0, 2),  // d2 -> d1: node 2
            (0, 1, 1),  // d1 -> d2: node 1
            (2, 0, 8),  // d3 -> d1: node 8
            (0, 2, 4),  // d1 -> d3: node 4
            (3, 1, 10), // d4 -> d2: node 10
            (3, 2, 11), // d4 -> d3: node 11
        ];

        for &(take, give, node) in swaps {
            let proposal = boundary_swap_proposal(&partition, take, give, node);

            let mut applied = partition.clone();
            applied.update(&proposal);
            let full_after = obj.score(&graph, &applied);

            let score_proposal = obj.score_proposal(&graph, &state, &proposal);
            assert_close(score_proposal, full_after, "score_proposal");

            let mut mut_state = state.clone();
            obj.apply_proposal(&graph, &mut mut_state, &proposal);
            assert_close(obj.score_state(&mut_state), full_after, "apply_then_score");

            let fresh_state = obj.init(&graph, &applied);
            assert_close(
                obj.score_state(&fresh_state),
                obj.score_state(&mut_state),
                "apply_matches_fresh",
            );
        }

        // Multi-step chain: several applied proposals must stay equivalent.
        let chain: &[(usize, usize, usize)] = &[(1, 0, 2), (0, 1, 5), (2, 0, 8)];
        let mut chain_part = partition.clone();
        let mut chain_state = state.clone();
        for &(take, give, node) in chain {
            let proposal = boundary_swap_proposal(&chain_part, take, give, node);
            obj.apply_proposal(&graph, &mut chain_state, &proposal);
            chain_part.update(&proposal);
        }
        let chain_full = obj.score(&graph, &chain_part);
        assert_close(obj.score_state(&chain_state), chain_full, "chain_score");
    }

    #[test]
    fn election_wins_incremental_matches_full_score() {
        run_equivalence_suite(test_election_wins_config());
    }

    #[test]
    fn gingles_partial_incremental_matches_full_score() {
        run_equivalence_suite(test_gingles_config());
    }

    #[test]
    fn polsby_popper_incremental_matches_full_score() {
        run_equivalence_suite(test_polsby_popper_config());
    }

    #[test]
    fn election_wins_min_aggregation() {
        let pair: &'static [(&'static str, &'static str)] =
            Box::leak(vec![(static_str("dem"), static_str("rep"))].into_boxed_slice());
        let obj = ObjectiveConfig::ElectionWins {
            elections: pair,
            target_a: false,
            aggregation: Aggregation::Min,
        };
        run_equivalence_suite(obj);
    }

    #[test]
    fn polsby_popper_min_aggregation() {
        let obj = ObjectiveConfig::PolsbyPopper {
            area_col: static_str("area"),
            perim_col: static_str("perim"),
            shared_perim_col: static_str("shared_perim"),
            boundary_perim_col: None,
            aggregation: Aggregation::Min,
        };
        run_equivalence_suite(obj);
    }

    #[test]
    fn ensure_derived_perim_column_matches_hand_computation() {
        // 2x2 rect grid. Column-major node layout:
        //   0 (0,0)  2 (1,0)
        //   1 (0,1)  3 (1,1)
        // Graph::rect_grid emits edges in sorted (low, high) order per source.
        // For 2x2, the edges are: (0,1), (0,2), (1,3), (2,3).
        let mut graph = Graph::rect_grid(2, 2);

        // Each boundary node contributes 1.0 of outer perimeter.
        let boundary: Vec<String> = (0..graph.pops.len()).map(|_| "1".to_string()).collect();
        graph.attr.insert("boundary_perim".to_string(), boundary);

        // Distinct shared_perim weights so we can tell which edges contributed.
        let shared: Vec<f64> = vec![2.0, 3.0, 4.0, 5.0];
        graph.edge_attr.insert("shared_perim".to_string(), shared);

        ensure_derived_perim_column(
            &mut graph,
            "perim",
            "boundary_perim",
            "shared_perim",
        );

        let perim: Vec<f64> = graph
            .attr
            .get("perim")
            .unwrap()
            .iter()
            .map(|s| s.parse::<f64>().unwrap())
            .collect();

        // Expected per node: boundary (=1) + sum of shared_perim for incident edges.
        // Edges sorted: e0=(0,1)=2, e1=(0,2)=3, e2=(1,3)=4, e3=(2,3)=5.
        // node 0: 1 + 2 + 3 = 6
        // node 1: 1 + 2 + 4 = 7
        // node 2: 1 + 3 + 5 = 9
        // node 3: 1 + 4 + 5 = 10
        assert_close(perim[0], 6.0, "perim[0]");
        assert_close(perim[1], 7.0, "perim[1]");
        assert_close(perim[2], 9.0, "perim[2]");
        assert_close(perim[3], 10.0, "perim[3]");
    }

    #[test]
    fn polsby_popper_district_scores_match_state() {
        let (graph, partition) = make_test_graph_and_partition();
        let obj = test_polsby_popper_config();
        let state = obj.init(&graph, &partition);
        let scores = obj.district_scores(&state);

        // Per-district scores should equal polsby_popper_score(area, perim)
        // computed on the initial state's cached per-district values.
        if let ObjectiveState::PolsbyPopper(s) = &state {
            assert_eq!(scores.len(), s.district_scores.len());
            for (i, &got) in scores.iter().enumerate() {
                assert_close(got, s.district_scores[i], "district_scores[i]");
            }
            // Mean aggregation equals the overall score.
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            assert_close(mean, s.score, "mean == aggregate");
        } else {
            panic!("wrong state variant");
        }
    }

    #[test]
    fn election_wins_multi_election() {
        let pairs: &'static [(&'static str, &'static str)] = Box::leak(
            vec![
                (static_str("dem"), static_str("rep")),
                (static_str("bvap"), static_str("vap")),
            ]
            .into_boxed_slice(),
        );
        let obj = ObjectiveConfig::ElectionWins {
            elections: pairs,
            target_a: true,
            aggregation: Aggregation::Mean,
        };
        run_equivalence_suite(obj);
    }
}