use crate::graph::Graph;
use crate::partition::Partition;
use crate::recom::RecomProposal;
use serde_json::Value;

pub trait ChainConstraint: Send + Clone {
    type State: Send + Clone;

    fn init(&self, graph: &Graph, partition: &Partition) -> Self::State;

    fn is_valid_state(&self, state: &Self::State) -> bool;

    fn proposal_valid(&self, graph: &Graph, state: &Self::State, proposal: &RecomProposal) -> bool;

    fn apply_proposal(&self, graph: &Graph, state: &mut Self::State, proposal: &RecomProposal);
}

#[derive(Clone, Copy)]
pub struct NoConstraint;

impl ChainConstraint for NoConstraint {
    type State = ();

    fn init(&self, _graph: &Graph, _partition: &Partition) -> Self::State {}

    fn is_valid_state(&self, _state: &Self::State) -> bool {
        true
    }

    fn proposal_valid(
        &self,
        _graph: &Graph,
        _state: &Self::State,
        _proposal: &RecomProposal,
    ) -> bool {
        true
    }

    fn apply_proposal(&self, _graph: &Graph, _state: &mut Self::State, _proposal: &RecomProposal) {}
}

#[derive(Clone)]
pub struct DistrictShareFloor {
    numerator_col: String,
    denominator_cols: Vec<String>,
    threshold: f64,
}

impl DistrictShareFloor {
    pub fn new(numerator_col: String, denominator_cols: Vec<String>, threshold: f64) -> Self {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            panic!("Constraint threshold must be finite and in [0, 1].");
        }
        if denominator_cols.is_empty() {
            panic!("Constraint denominator_cols must not be empty.");
        }
        Self {
            numerator_col,
            denominator_cols,
            threshold,
        }
    }

    pub fn required_node_cols(&self) -> Vec<String> {
        let mut cols = vec![self.numerator_col.clone()];
        for col in &self.denominator_cols {
            if !cols.contains(col) {
                cols.push(col.clone());
            }
        }
        cols
    }

    pub fn cache_graph_cols(&self, graph: &mut Graph) {
        for col in self.required_node_cols() {
            graph.cache_float_col(&col);
        }
    }

    fn district_valid(&self, numerator: f64, denominator: f64) -> bool {
        share(numerator, denominator) >= self.threshold
    }

    fn proposal_sums(&self, graph: &Graph, nodes: &[usize]) -> (f64, f64) {
        let numerator = sum_cached_float_col(graph, &self.numerator_col, nodes);
        let denominator = self
            .denominator_cols
            .iter()
            .map(|col| sum_cached_float_col(graph, col, nodes))
            .sum();
        (numerator, denominator)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DistrictShareFloorState {
    numerators: Vec<f64>,
    denominators: Vec<f64>,
}

impl ChainConstraint for DistrictShareFloor {
    type State = DistrictShareFloorState;

    fn init(&self, graph: &Graph, partition: &Partition) -> Self::State {
        let mut numerators = Vec::with_capacity(partition.num_dists as usize);
        let mut denominators = Vec::with_capacity(partition.num_dists as usize);

        for nodes in &partition.dist_nodes {
            let (numerator, denominator) = self.proposal_sums(graph, nodes);
            numerators.push(numerator);
            denominators.push(denominator);
        }

        DistrictShareFloorState {
            numerators,
            denominators,
        }
    }

    fn is_valid_state(&self, state: &Self::State) -> bool {
        state
            .numerators
            .iter()
            .zip(&state.denominators)
            .all(|(&num, &den)| self.district_valid(num, den))
    }

    fn proposal_valid(
        &self,
        graph: &Graph,
        _state: &Self::State,
        proposal: &RecomProposal,
    ) -> bool {
        let (a_num, a_den) = self.proposal_sums(graph, &proposal.a_nodes);
        let (b_num, b_den) = self.proposal_sums(graph, &proposal.b_nodes);
        self.district_valid(a_num, a_den) && self.district_valid(b_num, b_den)
    }

    fn apply_proposal(&self, graph: &Graph, state: &mut Self::State, proposal: &RecomProposal) {
        let (a_num, a_den) = self.proposal_sums(graph, &proposal.a_nodes);
        let (b_num, b_den) = self.proposal_sums(graph, &proposal.b_nodes);

        state.numerators[proposal.a_label] = a_num;
        state.denominators[proposal.a_label] = a_den;
        state.numerators[proposal.b_label] = b_num;
        state.denominators[proposal.b_label] = b_den;
    }
}

#[derive(Clone)]
pub enum ConstraintConfig {
    None,
    DistrictShareFloor(DistrictShareFloor),
}

impl ConstraintConfig {
    pub fn required_node_cols(&self) -> Vec<String> {
        match self {
            ConstraintConfig::None => vec![],
            ConstraintConfig::DistrictShareFloor(floor) => floor.required_node_cols(),
        }
    }

    pub fn cache_graph_cols(&self, graph: &mut Graph) {
        match self {
            ConstraintConfig::None => {}
            ConstraintConfig::DistrictShareFloor(floor) => floor.cache_graph_cols(graph),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConstraintState {
    None,
    DistrictShareFloor(DistrictShareFloorState),
}

impl ChainConstraint for ConstraintConfig {
    type State = ConstraintState;

    fn init(&self, graph: &Graph, partition: &Partition) -> Self::State {
        match self {
            ConstraintConfig::None => ConstraintState::None,
            ConstraintConfig::DistrictShareFloor(floor) => {
                ConstraintState::DistrictShareFloor(floor.init(graph, partition))
            }
        }
    }

    fn is_valid_state(&self, state: &Self::State) -> bool {
        match (self, state) {
            (ConstraintConfig::None, ConstraintState::None) => true,
            (
                ConstraintConfig::DistrictShareFloor(floor),
                ConstraintState::DistrictShareFloor(s),
            ) => floor.is_valid_state(s),
            _ => panic!("Mismatched constraint and state types"),
        }
    }

    fn proposal_valid(&self, graph: &Graph, state: &Self::State, proposal: &RecomProposal) -> bool {
        match (self, state) {
            (ConstraintConfig::None, ConstraintState::None) => true,
            (
                ConstraintConfig::DistrictShareFloor(floor),
                ConstraintState::DistrictShareFloor(s),
            ) => floor.proposal_valid(graph, s, proposal),
            _ => panic!("Mismatched constraint and state types"),
        }
    }

    fn apply_proposal(&self, graph: &Graph, state: &mut Self::State, proposal: &RecomProposal) {
        match (self, state) {
            (ConstraintConfig::None, ConstraintState::None) => {}
            (
                ConstraintConfig::DistrictShareFloor(floor),
                ConstraintState::DistrictShareFloor(s),
            ) => {
                floor.apply_proposal(graph, s, proposal);
            }
            _ => panic!("Mismatched constraint and state types"),
        }
    }
}

pub fn make_constraint_value(data: &Value) -> ConstraintConfig {
    match data["constraint"]
        .as_str()
        .unwrap_or_else(|| panic!("Missing field 'constraint' in constraint config"))
    {
        "district_share_floor" => ConstraintConfig::DistrictShareFloor(DistrictShareFloor::new(
            data["numerator_col"]
                .as_str()
                .unwrap_or_else(|| panic!("Missing field 'numerator_col' in constraint config"))
                .to_string(),
            data["denominator_cols"]
                .as_array()
                .unwrap_or_else(|| panic!("Missing field 'denominator_cols' in constraint config"))
                .iter()
                .map(|col| {
                    col.as_str()
                        .unwrap_or_else(|| panic!("denominator_cols entries must be strings"))
                        .to_string()
                })
                .collect(),
            data["threshold"]
                .as_f64()
                .unwrap_or_else(|| panic!("Missing field 'threshold' in constraint config")),
        )),
        other => panic!(
            "Unknown constraint '{}'. Supported: 'district_share_floor'.",
            other
        ),
    }
}

pub fn make_constraint(config: &str) -> ConstraintConfig {
    let data: Value = serde_json::from_str(config).unwrap();
    make_constraint_value(&data)
}

fn sum_cached_float_col(graph: &Graph, col: &str, nodes: &[usize]) -> f64 {
    let values = graph.float_attr.get(col).unwrap_or_else(|| {
        panic!(
            "Column '{}' has not been pre-cached; call cache_graph_cols before running the chain",
            col
        )
    });
    nodes.iter().map(|&node| values[node]).sum()
}

fn share(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}
