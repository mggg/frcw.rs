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
//! ## Module layout
//!
//! This module is the cross-cutting spine: the [`ObjectiveConfig`] /
//! [`ObjectiveState`] enums, the [`IncrementalObjective`] trait, the JSON
//! parsing/column-requirement entry points, and the dispatch that routes each
//! to the per-score implementation. Each score lives in its own submodule:
//!
//! - [`by_district_abs_deviation`]
//! - [`gingles_partial`]
//! - [`banded_gingles_partial`]
//! - [`election_wins`]
//! - [`polsby_popper`]
//!
//! The JSON schema for each objective is documented on its corresponding variant
//! of [`ObjectiveConfig`].
use crate::graph::Graph;
use crate::partition::Partition;
use crate::recom::RecomProposal;
use serde_json::Value;

mod banded_gingles_partial;
mod by_district_abs_deviation;
mod election_wins;
mod gingles_partial;
mod polsby_popper;

pub use banded_gingles_partial::BandedGinglesPartialState;
pub use by_district_abs_deviation::{AbsDevTotal, ByDistrictAbsDeviationState};
pub use election_wins::ElectionWinsState;
pub use gingles_partial::GinglesPartialState;
pub use polsby_popper::{ensure_derived_perim_column, polsby_popper_autoderive, PolsbyPopperState};

/// Aggregation method for per-district scores.
#[derive(Clone, Copy, Debug)]
pub enum Aggregation {
    Mean,
    Min,
    Sum,
}

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
    /// Minimize the total absolute distance between a list of target shares and
    /// distinct districts.
    ///
    /// For each district, `share = pov_counts / total_counts` (a within-district
    /// share of two populations); alternatively the denominator can be a single
    /// fixed global value, `share = pov_counts / total_count`, giving each
    /// district's share of a global total. Given
    /// `target_values` of length `k` (with `1 <= k <= district count`), the
    /// score is the minimum total `|share - target|` over all matchings of the
    /// `k` targets to `k` *distinct* districts -- each target claims its own
    /// district. On the real line this is achieved by sorting both lists and
    /// matching order-preservingly. For example, with
    /// `target_values = [0.1, 0.4, 0.6]` and district shares `[0.3, 0.5, 0.6]`,
    /// the score is `|0.3 - 0.1| + |0.5 - 0.4| + |0.6 - 0.6| = 0.3`.
    ///
    /// Two shapes have a simpler reading:
    /// - `k` copies of a single value `t` reward the `k` districts closest to
    ///   `t` (the sum of the `k` smallest `|share - t|`);
    /// - one value per district (`k == district count`) is the full sorted
    ///   bijection.
    ///
    /// JSON schema:
    /// ```json
    /// {
    ///   "objective": "by_district_abs_deviation",
    ///   "target_values": [0.1, 0.4, 0.6],
    ///   "pov_counts_col": "BVAP",
    ///   "total_counts_col": "VAP"
    /// }
    /// ```
    ///
    /// As a shorthand for many districts sharing one target, supply `target`
    /// and `n_target_districts` instead of `target_values` (this expands to
    /// `n_target_districts` copies of `target`, and is also accepted under the
    /// alias objective name `"abs_deviation"`):
    /// ```json
    /// {
    ///   "objective": "abs_deviation",
    ///   "target": 0.5,
    ///   "n_target_districts": 30,
    ///   "pov_counts_col": "BVAP",
    ///   "total_counts_col": "VAP"
    /// }
    /// ```
    ///
    /// To divide by a global constant instead of a per-district column, supply
    /// `total_count` in place of `total_counts_col` (each district's share is
    /// then `pov_counts / total_count`):
    /// ```json
    /// {
    ///   "objective": "by_district_abs_deviation",
    ///   "target_values": [0.1, 0.4, 0.6],
    ///   "pov_counts_col": "BVAP",
    ///   "total_count": 1000000
    /// }
    /// ```
    ///
    /// Fields:
    /// - `target_values`: target shares, each a finite value in [0, 1]; length
    ///   must be between 1 and the district count (validated when the chain
    ///   starts)
    /// - `target` + `n_target_districts`: shorthand alternative to
    ///   `target_values`; `target` is a finite share in [0, 1] and
    ///   `n_target_districts` a positive count (<= district count)
    /// - `pov_counts_col`: node attribute column for the population of interest (integer-valued)
    /// - `total_counts_col`: node attribute column for the total population
    ///   (integer-valued); mutually exclusive with `total_count`
    /// - `total_count`: a single positive global value used as the denominator
    ///   for every district; takes precedence over `total_counts_col` if both
    ///   are present
    ByDistrictAbsDeviation {
        /// Stored sorted ascending (see [`make_objective`]).
        target_values: &'static [f64],
        pov_counts_col: &'static str,
        /// Per-district column sum or a fixed global constant.
        total: AbsDevTotal,
    },

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

    /// Maximize Gingles opportunity districts that fall inside a target band,
    /// rewarding districts below the band and penalizing districts above it.
    ///
    /// Each district whose minority share lands in `[lower, upper]` contributes
    /// a full `1.0`. As with `gingles_partial`, the single highest district
    /// strictly below `lower` contributes a fractional reward `share / lower`
    /// in `[0, 1)`, giving the optimizer gradient signal toward gaining a new
    /// in-band district. Symmetrically, *every* district strictly above `upper`
    /// contributes a penalized `upper / share` in `(0, 1)`: the demerit
    /// `1 - upper / share` grows with how far the district overshoots and
    /// approaches a maximum of `1` per offending district.
    ///
    /// So for a band `[0.55, 0.65]` and shares
    /// `[0.43, 0.51, 0.58, 0.63, 0.70, 0.73]` the score is
    /// `2 + 0.51/0.55 + 0.65/0.70 + 0.65/0.73` (two in-band districts, the
    /// highest below-band district as a tiebreaker, and a penalty term for each
    /// of the two above-band districts).
    ///
    /// JSON schema:
    /// ```json
    /// {
    ///   "objective": "banded_gingles_partial",
    ///   "lower_threshold": 0.55,
    ///   "upper_threshold": 0.65,
    ///   "min_pop": "BVAP",
    ///   "total_pop": "VAP"
    /// }
    /// ```
    ///
    /// Fields:
    /// - `lower_threshold`: lower edge of the target band, must be in (0, 1)
    /// - `upper_threshold`: upper edge of the target band, must be in (0, 1)
    ///   and >= `lower_threshold`
    /// - `min_pop`: node attribute column for the minority population (integer-valued)
    /// - `total_pop`: node attribute column for the total population (integer-valued)
    BandedGinglesPartial {
        lower: f64,
        upper: f64,
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
        match *self {
            ObjectiveConfig::ByDistrictAbsDeviation {
                target_values,
                pov_counts_col,
                total,
            } => by_district_abs_deviation::full_score(
                graph,
                partition,
                target_values,
                pov_counts_col,
                total,
            ),
            ObjectiveConfig::GinglesPartial {
                threshold,
                min_pop_col,
                total_pop_col,
            } => {
                gingles_partial::full_score(graph, partition, threshold, min_pop_col, total_pop_col)
            }

            ObjectiveConfig::BandedGinglesPartial {
                lower,
                upper,
                min_pop_col,
                total_pop_col,
            } => banded_gingles_partial::full_score(
                graph,
                partition,
                lower,
                upper,
                min_pop_col,
                total_pop_col,
            ),

            ObjectiveConfig::ElectionWins {
                elections,
                target_a,
                aggregation,
            } => election_wins::full_score(graph, partition, elections, target_a, aggregation),

            ObjectiveConfig::PolsbyPopper {
                area_col,
                perim_col,
                shared_perim_col,
                boundary_perim_col: _,
                aggregation,
            } => polsby_popper::full_score(
                graph,
                partition,
                area_col,
                perim_col,
                shared_perim_col,
                aggregation,
            ),
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
            ObjectiveConfig::ByDistrictAbsDeviation {
                pov_counts_col,
                total,
                ..
            } => {
                graph.cache_int_col(pov_counts_col);
                if let AbsDevTotal::Column(col) = total {
                    graph.cache_int_col(col);
                }
            }
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
            }
            | ObjectiveConfig::BandedGinglesPartial {
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

/// Use `Box::leak` so that `&'static str` column names can be captured by `Copy`
/// configs/closures. Shared by the per-score `from_json` parsers.
fn leak_str(v: &Value, field: &str) -> &'static str {
    &*Box::leak(
        v[field]
            .as_str()
            .unwrap_or_else(|| panic!("Missing field '{}' in objective config", field))
            .to_owned()
            .into_boxed_str(),
    )
}

/// Parses an objective configuration JSON string into a `Copy` [`ObjectiveConfig`].
///
/// Dispatches on the `"objective"` field:
/// - `"by_district_abs_deviation"` (alias: `"abs_deviation"`) -- see
///   [`ObjectiveConfig::ByDistrictAbsDeviation`]
/// - `"gingles_partial"` -- see [`ObjectiveConfig::GinglesPartial`]
/// - `"banded_gingles_partial"` -- see [`ObjectiveConfig::BandedGinglesPartial`]
/// - `"election_wins"` -- see [`ObjectiveConfig::ElectionWins`]
/// - `"polsby_popper"` -- see [`ObjectiveConfig::PolsbyPopper`]
pub fn make_objective(config: &str) -> ObjectiveConfig {
    let data: Value = serde_json::from_str(config).unwrap();
    let obj_type = data["objective"].as_str().unwrap();

    match obj_type {
        // The `abs_deviation` name accepts the `target` + `n_target_districts`
        // shorthand and parses into a ByDistrictAbsDeviation.
        "by_district_abs_deviation" | "abs_deviation" => by_district_abs_deviation::from_json(&data),
        "gingles_partial" => gingles_partial::from_json(&data),
        "banded_gingles_partial" => banded_gingles_partial::from_json(&data),
        "election_wins" => election_wins::from_json(&data),
        "polsby_popper" => polsby_popper::from_json(&data),
        other => panic!(
            "Unknown objective '{}'. Supported: 'by_district_abs_deviation' (alias 'abs_deviation'), 'gingles_partial', 'banded_gingles_partial', 'election_wins', 'polsby_popper'.",
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

/// Returns the node attribute columns required by the given objective config.
///
/// Use this in CLI binaries to ensure these columns are included in the
/// `columns` argument to [`rustrecom::init::from_networkx`].
pub fn required_node_cols(config: &str) -> Vec<String> {
    let data: Value = serde_json::from_str(config).unwrap();
    match data["objective"].as_str().unwrap() {
        "by_district_abs_deviation" | "abs_deviation" => {
            by_district_abs_deviation::required_node_cols(&data)
        }
        "gingles_partial" => gingles_partial::required_node_cols(&data),
        "banded_gingles_partial" => banded_gingles_partial::required_node_cols(&data),
        "election_wins" => election_wins::required_node_cols(&data),
        "polsby_popper" => polsby_popper::required_node_cols(&data),
        _ => vec![],
    }
}

/// Returns the node attribute columns that the objective config references
/// but which may be absent on some nodes. These should be passed to
/// [`rustrecom::init::from_networkx`] as `partial_columns`: missing entries are
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
        Some("polsby_popper") => polsby_popper::partial_node_cols(&data),
        _ => vec![],
    }
}

/// Returns the edge attribute columns required by the given objective config.
///
/// Use this in CLI binaries to determine which columns to pass as `edge_float_cols`
/// to [`rustrecom::init::from_networkx`].
pub fn required_edge_cols(config: &str) -> Vec<String> {
    let data: Value = serde_json::from_str(config).unwrap();
    match data["objective"].as_str().unwrap() {
        "polsby_popper" => polsby_popper::required_edge_cols(&data),
        _ => vec![],
    }
}

// =====================================================================
// Incremental scoring
// =====================================================================

/// Tagged-union cached state for any [`ObjectiveConfig`] variant.
#[derive(Clone, Debug)]
pub enum ObjectiveState {
    ByDistrictAbsDeviation(ByDistrictAbsDeviationState),
    ElectionWins(ElectionWinsState),
    GinglesPartial(GinglesPartialState),
    BandedGinglesPartial(BandedGinglesPartialState),
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
    fn score_proposal(&self, graph: &Graph, current: &Self::State, proposal: &RecomProposal)
        -> f64;

    /// Mutates `state` to reflect applying `proposal`. Must produce state
    /// equivalent to rebuilding from the updated partition via [`Self::init`].
    fn apply_proposal(&self, graph: &Graph, state: &mut Self::State, proposal: &RecomProposal);

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

/// Minority share of a district, `min_pop / total_pop`. Shared by the
/// gingles-family scorers.
fn district_share(min_pop: i32, total_pop: i32) -> f64 {
    min_pop as f64 / total_pop as f64
}

impl IncrementalObjective for ObjectiveConfig {
    type State = ObjectiveState;

    fn init(&self, graph: &Graph, partition: &Partition) -> ObjectiveState {
        match *self {
            ObjectiveConfig::ByDistrictAbsDeviation {
                target_values,
                pov_counts_col,
                total,
            } => ObjectiveState::ByDistrictAbsDeviation(ByDistrictAbsDeviationState::init(
                graph,
                partition,
                target_values,
                pov_counts_col,
                total,
            )),
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
            ObjectiveConfig::BandedGinglesPartial {
                lower,
                upper,
                min_pop_col,
                total_pop_col,
            } => ObjectiveState::BandedGinglesPartial(BandedGinglesPartialState::init(
                graph,
                partition,
                lower,
                upper,
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
            (
                ObjectiveConfig::ByDistrictAbsDeviation { .. },
                ObjectiveState::ByDistrictAbsDeviation(s),
            ) => s.score,
            (ObjectiveConfig::ElectionWins { .. }, ObjectiveState::ElectionWins(s)) => s.score,
            (ObjectiveConfig::GinglesPartial { .. }, ObjectiveState::GinglesPartial(s)) => s.score,
            (
                ObjectiveConfig::BandedGinglesPartial { .. },
                ObjectiveState::BandedGinglesPartial(s),
            ) => s.score,
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
                ObjectiveConfig::ByDistrictAbsDeviation {
                    target_values,
                    pov_counts_col,
                    total,
                },
                ObjectiveState::ByDistrictAbsDeviation(state),
            ) => by_district_abs_deviation::score_proposal(
                graph,
                state,
                target_values,
                pov_counts_col,
                *total,
                proposal,
            ),
            (
                ObjectiveConfig::ElectionWins {
                    elections,
                    target_a,
                    aggregation,
                },
                ObjectiveState::ElectionWins(state),
            ) => election_wins::score_proposal(
                graph,
                state,
                elections,
                *target_a,
                *aggregation,
                proposal,
            ),
            (
                ObjectiveConfig::GinglesPartial {
                    threshold,
                    min_pop_col,
                    total_pop_col,
                },
                ObjectiveState::GinglesPartial(state),
            ) => gingles_partial::score_proposal(
                graph,
                state,
                *threshold,
                min_pop_col,
                total_pop_col,
                proposal,
            ),
            (
                ObjectiveConfig::BandedGinglesPartial {
                    lower,
                    upper,
                    min_pop_col,
                    total_pop_col,
                },
                ObjectiveState::BandedGinglesPartial(state),
            ) => banded_gingles_partial::score_proposal(
                graph,
                state,
                *lower,
                *upper,
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
            ) => polsby_popper::score_proposal(
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

    fn apply_proposal(&self, graph: &Graph, state: &mut ObjectiveState, proposal: &RecomProposal) {
        match (self, state) {
            (
                ObjectiveConfig::ByDistrictAbsDeviation {
                    target_values,
                    pov_counts_col,
                    total,
                },
                ObjectiveState::ByDistrictAbsDeviation(s),
            ) => by_district_abs_deviation::apply_proposal(
                graph,
                s,
                target_values,
                pov_counts_col,
                *total,
                proposal,
            ),
            (
                ObjectiveConfig::ElectionWins {
                    elections,
                    target_a,
                    aggregation,
                },
                ObjectiveState::ElectionWins(s),
            ) => election_wins::apply_proposal(
                graph,
                s,
                elections,
                *target_a,
                *aggregation,
                proposal,
            ),
            (
                ObjectiveConfig::GinglesPartial {
                    threshold,
                    min_pop_col,
                    total_pop_col,
                },
                ObjectiveState::GinglesPartial(s),
            ) => gingles_partial::apply_proposal(
                graph,
                s,
                *threshold,
                min_pop_col,
                total_pop_col,
                proposal,
            ),
            (
                ObjectiveConfig::BandedGinglesPartial {
                    lower,
                    upper,
                    min_pop_col,
                    total_pop_col,
                },
                ObjectiveState::BandedGinglesPartial(s),
            ) => banded_gingles_partial::apply_proposal(
                graph,
                s,
                *lower,
                *upper,
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
            ) => polsby_popper::apply_proposal(
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
            // The distinct-assignment matching makes a per-district deviation
            // ill-defined, so emit the honest per-district shares.
            (
                ObjectiveConfig::ByDistrictAbsDeviation { .. },
                ObjectiveState::ByDistrictAbsDeviation(s),
            ) => s
                .shares
                .iter()
                .map(|&sh| if sh.is_finite() { sh } else { 0.0 })
                .collect(),
            (ObjectiveConfig::PolsbyPopper { .. }, ObjectiveState::PolsbyPopper(s)) => {
                s.district_scores.clone()
            }
            (ObjectiveConfig::GinglesPartial { .. }, ObjectiveState::GinglesPartial(s)) => s
                .min_pops
                .iter()
                .zip(s.total_pops.iter())
                .map(|(&m, &t)| if t == 0 { 0.0 } else { m as f64 / t as f64 })
                .collect(),
            (
                ObjectiveConfig::BandedGinglesPartial { .. },
                ObjectiveState::BandedGinglesPartial(s),
            ) => s
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
        let area: Vec<String> = (0..n)
            .map(|i| format!("{}", (i + 1) as f64 * 1.5))
            .collect();
        let perim: Vec<String> = (0..n)
            .map(|i| format!("{}", (i + 3) as f64 * 2.25))
            .collect();

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
        graph
            .edge_attr
            .insert("shared_perim".to_string(), shared_perim);

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

    // Builders for the three scoring paths of `by_district_abs_deviation`. The
    // fixture has 4 districts; `target_values` must be sorted ascending (the
    // parser guarantees this, but the builders bypass it).

    /// `k < num_dists`, non-uniform -> general DP path.
    fn test_by_district_general_config() -> ObjectiveConfig {
        ObjectiveConfig::ByDistrictAbsDeviation {
            target_values: &[0.1, 0.25, 0.5],
            pov_counts_col: static_str("bvap"),
            total: AbsDevTotal::Column(static_str("vap")),
        }
    }

    /// `k == num_dists` -> full sorted-bijection bypass.
    fn test_by_district_full_bijection_config() -> ObjectiveConfig {
        ObjectiveConfig::ByDistrictAbsDeviation {
            target_values: &[0.05, 0.10, 0.20, 0.30],
            pov_counts_col: static_str("bvap"),
            total: AbsDevTotal::Column(static_str("vap")),
        }
    }

    /// All targets equal -> "k nearest districts" bypass.
    fn test_by_district_uniform_config() -> ObjectiveConfig {
        ObjectiveConfig::ByDistrictAbsDeviation {
            target_values: &[0.2, 0.2],
            pov_counts_col: static_str("bvap"),
            total: AbsDevTotal::Column(static_str("vap")),
        }
    }

    /// Constant global denominator -> shares are each district's fraction of a
    /// fixed total rather than a within-district ratio.
    fn test_by_district_constant_total_config() -> ObjectiveConfig {
        ObjectiveConfig::ByDistrictAbsDeviation {
            target_values: &[0.05, 0.15, 0.25],
            pov_counts_col: static_str("bvap"),
            total: AbsDevTotal::Constant(100.0),
        }
    }

    fn test_banded_gingles_config() -> ObjectiveConfig {
        // The 4x4 test fixture's quadrant shares are roughly
        // [0.082, 0.231, 0.062, 0.257], so this band starts with two
        // below-band districts, one in-band, and one above-band -- exercising
        // every branch of the banded scorer.
        ObjectiveConfig::BandedGinglesPartial {
            lower: 0.10,
            upper: 0.24,
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
    fn by_district_abs_deviation_general_incremental_matches_full_score() {
        run_equivalence_suite(test_by_district_general_config());
    }

    #[test]
    fn by_district_abs_deviation_full_bijection_incremental_matches_full_score() {
        run_equivalence_suite(test_by_district_full_bijection_config());
    }

    #[test]
    fn by_district_abs_deviation_uniform_incremental_matches_full_score() {
        run_equivalence_suite(test_by_district_uniform_config());
    }

    #[test]
    fn by_district_abs_deviation_constant_total_incremental_matches_full_score() {
        run_equivalence_suite(test_by_district_constant_total_config());
    }

    /// Full score for a path graph with one node per district, share = bvap/100.
    /// `targets` need not be sorted; the helper sorts them as the parser would.
    fn by_district_full_score(bvap: &[u32], targets: &[f64]) -> f64 {
        let n = bvap.len();
        let edges: String = (0..n.saturating_sub(1))
            .map(|i| format!("{} {}", i, i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let pops = vec!["1"; n].join(" ");
        let mut graph = Graph::from_edge_list(&edges, &pops).unwrap();
        graph.attr.insert(
            "bvap".to_string(),
            bvap.iter().map(|v| v.to_string()).collect(),
        );
        graph.attr.insert(
            "vap".to_string(),
            (0..n).map(|_| "100".to_string()).collect(),
        );
        let assignments: Vec<u32> = (1..=n as u32).collect();
        let partition = Partition::from_assignments(&graph, &assignments).unwrap();

        let mut sorted = targets.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let target_values: &'static [f64] = Box::leak(sorted.into_boxed_slice());
        let obj = ObjectiveConfig::ByDistrictAbsDeviation {
            target_values,
            pov_counts_col: static_str("bvap"),
            total: AbsDevTotal::Column(static_str("vap")),
        };
        obj.score(&graph, &partition)
    }

    #[test]
    fn by_district_abs_deviation_worked_examples() {
        // User's example: shares [0.3, 0.5, 0.6], targets [0.1, 0.4, 0.6]
        // (k == num_dists) -> sorted-matched 0.2 + 0.1 + 0.0 = 0.3.
        assert_close(
            by_district_full_score(&[30, 50, 60], &[0.1, 0.4, 0.6]),
            0.3,
            "full bijection",
        );

        // Subset (k < num_dists), general DP: shares [0.1, 0.5, 0.9],
        // targets [0.45, 0.85] -> best distinct match 0.45->0.5, 0.85->0.9
        // = 0.05 + 0.05 = 0.10 (beating any matching that uses district 0.1).
        assert_close(
            by_district_full_score(&[10, 50, 90], &[0.45, 0.85]),
            0.10,
            "subset assignment",
        );

        // All-equal targets: shares [0.1, 0.5, 0.9], targets [0.45, 0.45]
        // -> two smallest |share - 0.45| = 0.05 + 0.35.
        assert_close(
            by_district_full_score(&[10, 50, 90], &[0.45, 0.45]),
            0.05 + 0.35,
            "uniform (k nearest)",
        );
    }

    #[test]
    fn by_district_abs_deviation_shorthand_matches_explicit() {
        // The `target` + `n_target_districts` shorthand (and the `abs_deviation`
        // alias) expand to repeated target_values.
        let shorthand = make_objective(
            r#"{"objective":"abs_deviation","target":0.3,"n_target_districts":3,
                "pov_counts_col":"bvap","total_counts_col":"vap"}"#,
        );
        match shorthand {
            ObjectiveConfig::ByDistrictAbsDeviation { target_values, .. } => {
                assert_eq!(target_values.to_vec(), vec![0.3, 0.3, 0.3]);
            }
            _ => panic!("expected ByDistrictAbsDeviation"),
        }

        // Scores identically to the explicit list form.
        let (graph, partition) = make_test_graph_and_partition();
        let explicit = make_objective(
            r#"{"objective":"by_district_abs_deviation","target_values":[0.3,0.3,0.3],
                "pov_counts_col":"bvap","total_counts_col":"vap"}"#,
        );
        assert_close(
            shorthand.score(&graph, &partition),
            explicit.score(&graph, &partition),
            "shorthand vs explicit",
        );
    }

    #[test]
    fn by_district_abs_deviation_total_count_parses_and_scores() {
        // `total_count` parses into a constant denominator.
        let obj = make_objective(
            r#"{"objective":"by_district_abs_deviation","target_values":[0.05,0.15,0.25],
                "pov_counts_col":"bvap","total_count":100}"#,
        );
        match obj {
            ObjectiveConfig::ByDistrictAbsDeviation { total, .. } => {
                assert!(matches!(total, AbsDevTotal::Constant(t) if (t - 100.0).abs() < 1e-12));
            }
            _ => panic!("expected ByDistrictAbsDeviation"),
        }

        // A constant denominator only needs the pov column loaded.
        let cols = required_node_cols(
            r#"{"objective":"by_district_abs_deviation","target_values":[0.05,0.15,0.25],
                "pov_counts_col":"bvap","total_count":100}"#,
        );
        assert_eq!(cols, vec!["bvap".to_string()]);

        // Path graph, one node per district. With total_count = 100, shares are
        // bvap/100 = [0.1, 0.3, 0.5]; targets [0.05, 0.15, 0.25] (k == num_dists)
        // match order-preservingly: 0.05 + 0.15 + 0.25 = 0.45.
        let mut graph = Graph::from_edge_list("0 1\n1 2", "1 1 1").unwrap();
        graph.attr.insert(
            "bvap".to_string(),
            vec!["10", "30", "50"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        let partition = Partition::from_assignments(&graph, &vec![1u32, 2, 3]).unwrap();
        assert_close(
            obj.score(&graph, &partition),
            0.45,
            "constant total worked example",
        );

        // Incremental path agrees with the full score after caching.
        obj.cache_graph_cols(&mut graph);
        assert_close(
            obj.score_partition(&graph, &partition),
            0.45,
            "constant total worked example (incremental)",
        );
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
    fn banded_gingles_partial_incremental_matches_full_score() {
        run_equivalence_suite(test_banded_gingles_config());
    }

    #[test]
    fn gingles_partial_zero_total_pop_district_stays_finite() {
        let mut graph = Graph::from_edge_list("0 1\n1 2", "1 1 1").unwrap();
        graph.attr.insert(
            "bvap".to_string(),
            vec!["0", "30", "60"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        graph.attr.insert(
            "vap".to_string(),
            vec!["0", "100", "100"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        let partition = Partition::from_assignments(&graph, &vec![1u32, 2, 3]).unwrap();

        let obj = ObjectiveConfig::GinglesPartial {
            threshold: 0.5,
            min_pop_col: static_str("bvap"),
            total_pop_col: static_str("vap"),
        };

        // One opportunity district (0.6) plus the highest below district (0.3);
        // the NaN district contributes nothing. Score = 1 + 0.3 / 0.5 = 1.6.
        let full = obj.score(&graph, &partition);
        assert_close(full, 1.6, "gingles zero-total-pop full");

        let mut cache_graph = graph.clone();
        obj.cache_graph_cols(&mut cache_graph);
        let cached = obj.score_partition(&cache_graph, &partition);
        assert_close(cached, full, "gingles zero-total-pop incremental");
    }

    #[test]
    fn banded_gingles_partial_zero_total_pop_district_stays_finite() {
        // Same NaN-share district as the gingles test, scored against a band.
        let mut graph = Graph::from_edge_list("0 1\n1 2", "1 1 1").unwrap();
        graph.attr.insert(
            "bvap".to_string(),
            vec!["0", "30", "60"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        graph.attr.insert(
            "vap".to_string(),
            vec!["0", "100", "100"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        let partition = Partition::from_assignments(&graph, &vec![1u32, 2, 3]).unwrap();

        let obj = ObjectiveConfig::BandedGinglesPartial {
            lower: 0.4,
            upper: 0.7,
            min_pop_col: static_str("bvap"),
            total_pop_col: static_str("vap"),
        };

        // 0.6 is in-band (+1); 0.3 is the highest below-band district
        // (+0.3 / 0.4); the NaN district contributes nothing.
        let expected = 1.0 + 0.3 / 0.4;
        let full = obj.score(&graph, &partition);
        assert_close(full, expected, "banded zero-total-pop full");

        let mut cache_graph = graph.clone();
        obj.cache_graph_cols(&mut cache_graph);
        let cached = obj.score_partition(&cache_graph, &partition);
        assert_close(cached, full, "banded zero-total-pop incremental");
    }

    #[test]
    fn banded_gingles_partial_matches_worked_example() {
        // A 6-node path graph with one node per district lets us set each
        // district's share directly via (bvap, vap).
        let mut graph = Graph::from_edge_list("0 1\n1 2\n2 3\n3 4\n4 5", "1 1 1 1 1 1").unwrap();
        // Target shares [0.43, 0.51, 0.58, 0.63, 0.70, 0.73] via bvap/100.
        let bvap: Vec<String> = ["43", "51", "58", "63", "70", "73"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let vap: Vec<String> = (0..6).map(|_| "100".to_string()).collect();
        graph.attr.insert("bvap".to_string(), bvap);
        graph.attr.insert("vap".to_string(), vap);

        let assignments: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let partition = Partition::from_assignments(&graph, &assignments).unwrap();

        let obj = ObjectiveConfig::BandedGinglesPartial {
            lower: 0.55,
            upper: 0.65,
            min_pop_col: static_str("bvap"),
            total_pop_col: static_str("vap"),
        };

        // Band [0.55, 0.65]: 0.58 and 0.63 are in-band (+2); 0.51 is the highest
        // below-band district (+0.51/0.55); 0.70 and 0.73 are above-band
        // (+0.65/0.70 + 0.65/0.73). 0.43 contributes nothing.
        let expected = 2.0 + 0.51 / 0.55 + 0.65 / 0.70 + 0.65 / 0.73;

        let full = obj.score(&graph, &partition);
        assert_close(full, expected, "banded worked example (full)");

        let mut cache_graph = graph.clone();
        obj.cache_graph_cols(&mut cache_graph);
        let cached = obj.score_partition(&cache_graph, &partition);
        assert_close(cached, expected, "banded worked example (incremental)");
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

        ensure_derived_perim_column(&mut graph, "perim", "boundary_perim", "shared_perim");

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
