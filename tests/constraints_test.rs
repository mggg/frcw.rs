use rustrecom::constraints::{make_constraint, ChainConstraint, ConstraintConfig};
use rustrecom::graph::Graph;
use rustrecom::partition::Partition;
use rustrecom::recom::run::multi_chain_with_constraint;
use rustrecom::recom::RecomProposal;
use rustrecom::recom::{RecomParams, RecomVariant};
use rustrecom::stats::{SelfLoopCounts, SelfLoopReason, StatsWriter};
use std::io::Result as IoResult;
use std::sync::{Arc, Mutex};
use test_fixtures::fixture_with_attributes;

fn path_graph_with_attrs(rep: Vec<&str>, dem: Vec<&str>) -> (Graph, Partition) {
    let mut graph = Graph::from_edge_list("0 1\n1 2\n2 3", "1 1 1 1").unwrap();
    graph.attr.insert(
        "rep".to_string(),
        rep.into_iter().map(|value| value.to_string()).collect(),
    );
    graph.attr.insert(
        "dem".to_string(),
        dem.into_iter().map(|value| value.to_string()).collect(),
    );
    let partition = Partition::from_assignments(&graph, &vec![1, 1, 2, 2]).unwrap();
    (graph, partition)
}

fn rep_floor(threshold: f64) -> String {
    format!(
        r#"{{"constraint":"district_share_floor","numerator_col":"rep","denominator_cols":["rep","dem"],"threshold":{threshold}}}"#
    )
}

struct UnusedWriter;

impl StatsWriter for UnusedWriter {
    fn init(&mut self, _graph: &Graph, _partition: &Partition) -> IoResult<()> {
        panic!("invalid seed should fail before writer init")
    }

    fn step(
        &mut self,
        _step: u64,
        _graph: &Graph,
        _partition: &Partition,
        _proposal: &RecomProposal,
        _counts: &SelfLoopCounts,
    ) -> IoResult<()> {
        panic!("invalid seed should fail before writer step")
    }

    fn close(&mut self) -> IoResult<()> {
        panic!("invalid seed should fail before writer close")
    }
}

#[test]
fn district_share_floor_accepts_integer_columns() {
    let (mut graph, partition) =
        path_graph_with_attrs(vec!["6", "6", "7", "7"], vec!["4", "4", "3", "3"]);
    let constraint = make_constraint(&rep_floor(0.6));
    constraint.cache_graph_cols(&mut graph);
    let state = constraint.init(&graph, &partition);
    assert!(constraint.is_valid_state(&state));
}

#[test]
fn district_share_floor_accepts_float_columns() {
    let (mut graph, partition) = path_graph_with_attrs(
        vec!["0.25", "0.25", "0.40", "0.40"],
        vec!["0.25", "0.25", "0.10", "0.10"],
    );
    let constraint = make_constraint(&rep_floor(0.5));
    constraint.cache_graph_cols(&mut graph);
    let state = constraint.init(&graph, &partition);
    assert!(constraint.is_valid_state(&state));
}

#[test]
fn district_share_floor_rejects_invalid_seed() {
    let (mut graph, partition) =
        path_graph_with_attrs(vec!["6", "6", "7", "7"], vec!["4", "4", "3", "3"]);
    let constraint = make_constraint(&rep_floor(0.65));
    constraint.cache_graph_cols(&mut graph);
    let state = constraint.init(&graph, &partition);
    assert!(!constraint.is_valid_state(&state));
}

#[test]
fn district_share_floor_checks_changed_proposal_districts() {
    let (mut graph, partition) =
        path_graph_with_attrs(vec!["10", "2", "7", "7"], vec!["0", "8", "3", "3"]);
    let constraint = make_constraint(&rep_floor(0.6));
    constraint.cache_graph_cols(&mut graph);
    let state = constraint.init(&graph, &partition);

    let valid = RecomProposal {
        a_label: 0,
        b_label: 1,
        a_pop: 2,
        b_pop: 2,
        a_nodes: vec![0, 1],
        b_nodes: vec![2, 3],
    };
    assert!(constraint.proposal_valid(&graph, &state, &valid));

    let invalid = RecomProposal {
        a_label: 0,
        b_label: 1,
        a_pop: 2,
        b_pop: 2,
        a_nodes: vec![1, 2],
        b_nodes: vec![0, 3],
    };
    assert!(!constraint.proposal_valid(&graph, &state, &invalid));
}

#[test]
fn district_share_floor_apply_matches_reinit() {
    let (mut graph, mut partition) =
        path_graph_with_attrs(vec!["6", "6", "7", "7"], vec!["4", "4", "3", "3"]);
    let constraint = make_constraint(&rep_floor(0.6));
    constraint.cache_graph_cols(&mut graph);
    let mut state = constraint.init(&graph, &partition);
    let proposal = RecomProposal {
        a_label: 0,
        b_label: 1,
        a_pop: 2,
        b_pop: 2,
        a_nodes: vec![0, 2],
        b_nodes: vec![1, 3],
    };

    constraint.apply_proposal(&graph, &mut state, &proposal);
    partition.update(&proposal);
    let rebuilt = constraint.init(&graph, &partition);

    assert_eq!(state, rebuilt);
}

#[test]
fn constrained_runner_rejects_invalid_seed_before_starting() {
    let (mut graph, partition) =
        path_graph_with_attrs(vec!["6", "6", "7", "7"], vec!["4", "4", "3", "3"]);
    let constraint = make_constraint(&rep_floor(0.65));
    constraint.cache_graph_cols(&mut graph);
    let params = RecomParams {
        min_pop: 1,
        max_pop: 3,
        balance_ub: 0,
        num_steps: 10,
        rng_seed: 17,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };

    let result = multi_chain_with_constraint(
        &graph,
        &partition,
        Box::new(UnusedWriter),
        &params,
        1,
        1,
        false,
        constraint,
    );

    assert_eq!(
        result.unwrap_err(),
        "Initial partition does not satisfy the chain constraint"
    );
}

/// Records, across a live chain run, whether any emitted plan ever violated the
/// constraint and how many constraint self-loops the workers reported.
#[derive(Default)]
struct RunRecord {
    /// First step whose emitted partition violated the constraint (`None` if always valid).
    invalid_step: Option<u64>,
    /// Total `ConstraintViolation` self-loops seen across all step/self_loop frames.
    constraint_loops: usize,
}

struct RecordingWriter {
    constraint: ConstraintConfig,
    record: Arc<Mutex<RunRecord>>,
}

impl RecordingWriter {
    fn check(&self, graph: &Graph, partition: &Partition, step: u64) {
        let state = self.constraint.init(graph, partition);
        if !self.constraint.is_valid_state(&state) {
            let mut rec = self.record.lock().unwrap();
            rec.invalid_step.get_or_insert(step);
        }
    }
}

impl StatsWriter for RecordingWriter {
    fn init(&mut self, graph: &Graph, partition: &Partition) -> IoResult<()> {
        self.check(graph, partition, 0);
        Ok(())
    }

    fn step(
        &mut self,
        step: u64,
        graph: &Graph,
        partition: &Partition,
        _proposal: &RecomProposal,
        counts: &SelfLoopCounts,
    ) -> IoResult<()> {
        self.check(graph, partition, step);
        self.record.lock().unwrap().constraint_loops +=
            counts.get(SelfLoopReason::ConstraintViolation);
        Ok(())
    }

    fn self_loop(
        &mut self,
        _step: u64,
        _graph: &Graph,
        _partition: &Partition,
        counts: &SelfLoopCounts,
    ) -> IoResult<()> {
        self.record.lock().unwrap().constraint_loops +=
            counts.get(SelfLoopReason::ConstraintViolation);
        Ok(())
    }

    fn close(&mut self) -> IoResult<()> {
        Ok(())
    }
}

/// A live multithreaded run under a tight constraint must (a) never emit a plan
/// that violates the constraint and (b) actually reject some proposals as
/// constraint self-loops. The 6x6 grid seeds every district at a-share 0.5, so a
/// 0.5 floor leaves the seed valid while rejecting any split that pushes a
/// changed district below 0.5.
#[test]
fn constrained_runner_only_emits_valid_plans_and_self_loops() {
    let (mut graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let constraint = make_constraint(
        r#"{"constraint":"district_share_floor","numerator_col":"a_share","denominator_cols":["a_share","b_share"],"threshold":0.5}"#,
    );
    constraint.cache_graph_cols(&mut graph);

    let params = RecomParams {
        min_pop: 4,
        max_pop: 8,
        balance_ub: 0,
        num_steps: 2000,
        rng_seed: 781234,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };

    let record = Arc::new(Mutex::new(RunRecord::default()));
    let writer = Box::new(RecordingWriter {
        constraint: constraint.clone(),
        record: record.clone(),
    });

    let result =
        multi_chain_with_constraint(&graph, &partition, writer, &params, 4, 1, false, constraint);
    assert!(result.is_ok(), "chain run failed: {:?}", result);

    let rec = record.lock().unwrap();
    assert_eq!(
        rec.invalid_step, None,
        "constraint was violated by an emitted plan at step {:?}",
        rec.invalid_step
    );
    assert!(
        rec.constraint_loops > 0,
        "expected at least one constraint self-loop under a 0.5 floor"
    );
}
