// Functional tests for tilted run optimization.
use frcw::graph::Graph;
use frcw::objectives::{make_objective, make_objective_fn, required_node_cols};
use frcw::partition::Partition;
use frcw::recom::tilted::{
    multi_tilted_runs, multi_tilted_runs_with_writer, AcceptanceRule, FixedAcceptance,
    FullRescoreBackend, IncrementalBackend, MetropolisAcceptance,
};
use frcw::recom::RecomProposal;
use frcw::recom::{RecomParams, RecomVariant};
use frcw::stats::{CanonicalWriter, ScoresWriter, SelfLoopCounts, StatsWriter};
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::io::Result as IOResult;
use std::iter::FromIterator;

use rstest::rstest;
use test_fixtures::fixture_with_attributes;

const RNG_SEED: u64 = 153434375;

// =================================================================================
// == Helpers (same invariant checks as step_test.rs, applied to final partition) ==
// =================================================================================

fn is_connected_subset(graph: &Graph, nodes: &Vec<usize>) -> bool {
    if nodes.is_empty() {
        return true;
    }
    let nodeset = HashSet::<usize>::from_iter(nodes.iter().cloned());
    let mut stack = vec![nodes[0]];
    let mut visited = HashSet::<usize>::from_iter(stack.iter().cloned());
    while let Some(next) = stack.pop() {
        for neighbor in graph.neighbors[next].iter() {
            if nodeset.contains(neighbor) && !visited.contains(neighbor) {
                visited.insert(*neighbor);
                stack.push(*neighbor);
            }
        }
    }
    return visited.len() == nodes.len();
}

fn assert_partition_valid(graph: &Graph, partition: &Partition, min_pop: u32, max_pop: u32) {
    // Node count matches.
    let node_count: usize = partition.dist_nodes.iter().map(|n| n.len()).sum();
    assert_eq!(
        node_count,
        graph.neighbors.len(),
        "Node count mismatch: partition has {}, graph has {}",
        node_count,
        graph.neighbors.len()
    );

    // Total population matches.
    assert_eq!(
        partition.dist_pops.iter().sum::<u32>(),
        graph.total_pop,
        "Total population mismatch"
    );

    // Population bounds.
    for (i, &pop) in partition.dist_pops.iter().enumerate() {
        assert!(
            min_pop <= pop && pop <= max_pop,
            "District {} pop {} outside bounds [{}, {}]",
            i,
            pop,
            min_pop,
            max_pop
        );
    }

    // Population sums match dist_nodes.
    for (i, (nodes, &pop)) in partition
        .dist_nodes
        .iter()
        .zip(partition.dist_pops.iter())
        .enumerate()
    {
        let computed: u32 = nodes.iter().map(|&n| graph.pops[n]).sum();
        assert_eq!(
            computed, pop,
            "District {} pop sum {} != recorded pop {}",
            i, computed, pop
        );
    }

    // Assignments consistent with dist_nodes.
    for (dist, nodes) in partition.dist_nodes.iter().enumerate() {
        for &n in nodes {
            assert_eq!(
                partition.assignments[n] as usize, dist,
                "Node {} assigned to {} but in dist_nodes[{}]",
                n, partition.assignments[n], dist
            );
        }
    }

    // All districts connected.
    for (i, nodes) in partition.dist_nodes.iter().enumerate() {
        assert!(
            is_connected_subset(graph, nodes),
            "District {} is disconnected",
            i
        );
    }
}

/// Simple objective: population of district 0.
/// Useful for testing chain mechanics without depending on the objectives module.
fn dist0_pop_objective(graph: &Graph, partition: &Partition) -> f64 {
    partition.dist_nodes[0]
        .iter()
        .map(|&n| graph.pops[n] as f64)
        .sum()
}

struct CountingStatsWriter {
    init_calls: usize,
    steps: Vec<(u64, usize)>,
}

impl CountingStatsWriter {
    fn new() -> CountingStatsWriter {
        CountingStatsWriter {
            init_calls: 0,
            steps: Vec::new(),
        }
    }
}

impl StatsWriter for CountingStatsWriter {
    fn init(&mut self, _graph: &Graph, _partition: &Partition) -> IOResult<()> {
        self.init_calls += 1;
        Ok(())
    }

    fn step(
        &mut self,
        step: u64,
        _graph: &Graph,
        _partition: &Partition,
        _proposal: &RecomProposal,
        counts: &SelfLoopCounts,
    ) -> IOResult<()> {
        self.steps.push((step, counts.sum()));
        Ok(())
    }

    fn close(&mut self) -> IOResult<()> {
        Ok(())
    }
}

// ====================================
// == Partition validity on 6x6 grid ==
// ====================================

#[rstest]
fn test_tilted_partition_valid_grid(
    #[values(1, 4)] n_threads: usize,
    #[values(true, false)] maximize: bool,
    #[values(0.0, 0.5)] accept_worse_prob: f64,
) {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let min_pop: u32 = 5;
    let max_pop: u32 = 7;
    let params = RecomParams {
        min_pop,
        max_pop,
        num_steps: 500,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };
    let result = multi_tilted_runs(
        &graph,
        partition,
        &params,
        n_threads,
        FullRescoreBackend {
            obj_fn: dist0_pop_objective,
        },
        FixedAcceptance {
            prob: accept_worse_prob,
        },
        maximize,
        false,
    );
    let final_partition = result.expect("tilted run should not fail");
    assert_partition_valid(&graph, &final_partition, min_pop, max_pop);
}

// ============================================
// == Hill-climbing monotonicity on 6x6 grid ==
// ============================================

#[rstest]
fn test_tilted_hill_climbing_maximize_grid() {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let initial_score = dist0_pop_objective(&graph, &partition);
    let params = RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps: 1000,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };
    let final_partition = multi_tilted_runs(
        &graph,
        partition,
        &params,
        1,
        FullRescoreBackend {
            obj_fn: dist0_pop_objective,
        },
        FixedAcceptance { prob: 0.0 }, // pure hill-climbing
        true,
        false,
    )
    .unwrap();
    let final_score = dist0_pop_objective(&graph, &final_partition);
    assert!(
        final_score >= initial_score,
        "Hill-climbing maximize: final {} < initial {}",
        final_score,
        initial_score
    );
}

#[rstest]
fn test_tilted_hill_climbing_minimize_grid() {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let initial_score = dist0_pop_objective(&graph, &partition);
    let params = RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps: 1000,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };
    let final_partition = multi_tilted_runs(
        &graph,
        partition,
        &params,
        1,
        FullRescoreBackend {
            obj_fn: dist0_pop_objective,
        },
        FixedAcceptance { prob: 0.0 }, // pure hill-climbing
        false,
        false,
    )
    .unwrap();
    let final_score = dist0_pop_objective(&graph, &final_partition);
    assert!(
        final_score <= initial_score,
        "Hill-climbing minimize: final {} > initial {}",
        final_score,
        initial_score
    );
}

#[test]
fn test_tilted_rejects_zero_threads() {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let params = RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps: 1,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };
    let err = multi_tilted_runs(
        &graph,
        partition,
        &params,
        0,
        FullRescoreBackend {
            obj_fn: dist0_pop_objective,
        },
        FixedAcceptance { prob: 0.0 },
        true,
        false,
    )
    .expect_err("n_threads=0 should fail fast");
    assert!(err.contains("n_threads must be at least 1"));
}

#[test]
fn test_tilted_returns_terminal_partition_not_best_seen() {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let initial_assignments = partition.assignments.clone();
    let initial_assignments_ref = &initial_assignments;
    let objective = move |_graph: &Graph, p: &Partition| {
        if p.assignments == *initial_assignments_ref {
            1.0
        } else {
            0.0
        }
    };
    let params = RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps: 8,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };
    let final_partition = multi_tilted_runs(
        &graph,
        partition,
        &params,
        1,
        FullRescoreBackend { obj_fn: objective },
        FixedAcceptance { prob: 1.0 },
        true,
        false,
    )
    .unwrap();
    assert_ne!(
        final_partition.assignments, initial_assignments,
        "tilted runs should return the terminal chain state, not the best-seen plan"
    );
}

#[test]
fn test_tilted_stats_writer_records_accepted_steps() {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let params = RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps: 8,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };
    let mut stats_writer = CountingStatsWriter::new();
    multi_tilted_runs_with_writer(
        &graph,
        partition,
        &params,
        1,
        FullRescoreBackend {
            obj_fn: dist0_pop_objective,
        },
        FixedAcceptance { prob: 1.0 },
        true,
        Some(&mut stats_writer),
        None,
        false,
    )
    .unwrap();
    assert_eq!(stats_writer.init_calls, 1);
    // The chain produces (num_steps - 1) events of its own (init seed
    // counts as the first record), so stats_writer.step is invoked for
    // chain steps 1..=num_steps - 1.
    assert_eq!(
        stats_writer.steps,
        (1..=(params.num_steps - 1))
            .map(|step| (step, 0))
            .collect::<Vec<_>>()
    );
}

#[test]
fn test_tilted_scores_writer_records_every_step() {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let params = RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps: 12,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };
    let path = std::env::temp_dir().join(format!(
        "frcw_tilted_scores_{}_{}.csv",
        std::process::id(),
        RNG_SEED
    ));
    let output = Box::new(std::io::BufWriter::new(fs::File::create(&path).unwrap()));
    let mut scores_writer = ScoresWriter::new(output);
    let final_partition = multi_tilted_runs_with_writer(
        &graph,
        partition,
        &params,
        1,
        FullRescoreBackend {
            obj_fn: dist0_pop_objective,
        },
        FixedAcceptance { prob: 0.0 },
        true,
        None,
        Some(&mut scores_writer),
        false,
    )
    .unwrap();
    let scores = fs::read_to_string(&path).unwrap();
    let lines = scores.lines().collect::<Vec<_>>();
    // FullRescoreBackend exposes no per-district scores, so the header is the
    // bare `step,score` (no best_score column).
    assert_eq!(lines[0], "step,score");
    // header + init seed row + (num_steps - 1) chain events = num_steps + 1 lines
    assert_eq!(lines.len(), params.num_steps as usize + 1);

    for (idx, line) in lines.iter().enumerate().skip(1) {
        let fields = line.split(',').collect::<Vec<_>>();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].parse::<usize>().unwrap(), idx - 1);
        fields[1].parse::<f64>().unwrap();
    }
    let final_score = dist0_pop_objective(&graph, &final_partition);
    let last_fields = lines.last().unwrap().split(',').collect::<Vec<_>>();
    assert_eq!(last_fields[1].parse::<f64>().unwrap(), final_score);

    fs::remove_file(path).unwrap();
}

#[rstest]
fn test_tilted_canonical_writer_mixed_ending_counts(
    #[values(1, 2, 4)] n_threads: usize,
    #[values(0.1, 0.5, 0.9)] accept_worse_prob: f64,
    #[values(1, 7, 42, 101, 2025)] seed: u64,
) {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let params = RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps: 250,
        rng_seed: seed,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };
    let path = std::env::temp_dir().join(format!(
        "frcw_tilted_mixed_{}_{}_{}_{}.jsonl",
        std::process::id(),
        n_threads,
        (accept_worse_prob * 10.0) as u64,
        seed,
    ));
    let output = Box::new(std::io::BufWriter::new(fs::File::create(&path).unwrap()));
    let mut writer = CanonicalWriter::new(output);
    multi_tilted_runs_with_writer(
        &graph,
        partition,
        &params,
        n_threads,
        FullRescoreBackend {
            obj_fn: dist0_pop_objective,
        },
        FixedAcceptance {
            prob: accept_worse_prob,
        },
        true,
        Some(&mut writer),
        None,
        false,
    )
    .unwrap();

    let output = fs::read_to_string(&path).unwrap();
    let records: Vec<Value> = output
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect();
    // Init seed plan + (num_steps - 1) chain events = num_steps records.
    assert_eq!(
        records.len(),
        params.num_steps as usize,
        "expected {} records but got {} (n_threads={}, accept_worse_prob={}, seed={})",
        params.num_steps as usize,
        records.len(),
        n_threads,
        accept_worse_prob,
        seed,
    );
    let samples: Vec<u64> = records
        .iter()
        .map(|r| r["sample"].as_u64().unwrap())
        .collect();
    let expected: Vec<u64> = (1..=params.num_steps).collect();
    let expected = if samples.first().copied() == Some(1) && samples.get(1).copied() == Some(1) {
        let mut v = vec![1u64];
        v.extend(1..=(params.num_steps - 1));
        v
    } else {
        expected
    };
    assert_eq!(
        samples, expected,
        "sample numbers not contiguous (n_threads={}, accept_worse_prob={}, seed={})",
        n_threads, accept_worse_prob, seed,
    );

    fs::remove_file(path).unwrap();
}

#[test]
fn test_tilted_canonical_writer_flushes_terminal_self_loops() {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let initial_assignments = partition.assignments.clone();
    let initial_assignments_ref = &initial_assignments;
    let objective = move |_graph: &Graph, p: &Partition| {
        if p.assignments == *initial_assignments_ref {
            1.0
        } else {
            0.0
        }
    };
    let params = RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps: 8,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };
    let path = std::env::temp_dir().join(format!(
        "frcw_tilted_canonical_{}_{}.jsonl",
        std::process::id(),
        RNG_SEED
    ));
    let output = Box::new(std::io::BufWriter::new(fs::File::create(&path).unwrap()));
    let mut writer = CanonicalWriter::new(output);
    let final_partition = multi_tilted_runs_with_writer(
        &graph,
        partition,
        &params,
        1,
        FullRescoreBackend { obj_fn: objective },
        FixedAcceptance { prob: 0.0 },
        true,
        Some(&mut writer),
        None,
        false,
    )
    .unwrap();

    let output = fs::read_to_string(&path).unwrap();
    let records = output
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(records.len(), params.num_steps as usize);
    assert_eq!(
        records.last().unwrap()["sample"].as_u64().unwrap(),
        params.num_steps - 1
    );
    let final_assignment = final_partition
        .assignments
        .iter()
        .map(|assignment| Value::from(*assignment))
        .collect::<Vec<_>>();
    assert_eq!(
        records.last().unwrap()["assignment"].as_array().unwrap(),
        &final_assignment
    );

    fs::remove_file(path).unwrap();
}

// ===================================
// == Objective function unit tests ==
// ===================================

#[test]
fn test_election_wins_scoring_tied() {
    // 6x6 grid: every district has a_share sum = 3, b_share sum = 3 (tied).
    // With target "a": 0 wins, best losing share = 3/6 = 0.5.
    // But 0.5 is not > 0.5, so it counts as a loss.
    // Exact ties should map to the largest float below 1.0 so the tiebreaker
    // remains strictly below the next integer win count.
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"a_share","votes_b":"b_share"}],"target":"a","aggregation":"mean"}"#;
    let obj_fn = make_objective_fn(config);
    let score = obj_fn(&graph, &partition);
    let expected = f64::from_bits(1.0f64.to_bits() - 1);
    assert!(
        score.to_bits() == expected.to_bits(),
        "Expected score {:?} for tied districts, got {:?}",
        expected,
        score
    );
}

#[test]
fn test_election_wins_scoring_target_b() {
    // Same tied districts, but target "b" -- should give the same score.
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"a_share","votes_b":"b_share"}],"target":"b","aggregation":"mean"}"#;
    let obj_fn = make_objective_fn(config);
    let score = obj_fn(&graph, &partition);
    let expected = f64::from_bits(1.0f64.to_bits() - 1);
    assert!(
        score.to_bits() == expected.to_bits(),
        "Expected score {:?} for tied districts with target b, got {:?}",
        expected,
        score
    );
}

#[test]
fn test_required_node_cols_election_wins() {
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"DEM_GOV","votes_b":"REP_GOV"},{"votes_a":"DEM_SEN","votes_b":"REP_SEN"}],"target":"a","aggregation":"mean"}"#;
    let cols = required_node_cols(config);
    assert_eq!(cols, vec!["DEM_GOV", "REP_GOV", "DEM_SEN", "REP_SEN"]);
}

#[test]
fn test_required_node_cols_gingles() {
    let config =
        r#"{"objective":"gingles_partial","threshold":0.5,"min_pop":"BVAP","total_pop":"VAP"}"#;
    let cols = required_node_cols(config);
    assert_eq!(cols, vec!["BVAP", "VAP"]);
}

// ============================================================================
// == Tilted runs on IA counties (99 nodes, 4 districts, real election data) ==
// ============================================================================

#[rstest]
fn test_tilted_iowa_election_wins(
    #[values(1, 4)] n_threads: usize,
    #[values(true, false)] maximize: bool,
) {
    let election_cols = vec!["PRES16D", "PRES16R"];
    let (graph, partition) = fixture_with_attributes("IA", election_cols);
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let pop_tol = 0.2;
    let min_pop = ((1.0 - pop_tol) * avg_pop).floor() as u32;
    let max_pop = ((1.0 + pop_tol) * avg_pop).ceil() as u32;
    let params = RecomParams {
        min_pop,
        max_pop,
        num_steps: 500,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"PRES16D","votes_b":"PRES16R"}],"target":"a","aggregation":"mean"}"#;
    let obj_fn = make_objective_fn(config);
    let final_partition = multi_tilted_runs(
        &graph,
        partition,
        &params,
        n_threads,
        FullRescoreBackend { obj_fn },
        FixedAcceptance { prob: 0.05 },
        maximize,
        false,
    )
    .expect("IA tilted run should not fail");
    assert_partition_valid(&graph, &final_partition, min_pop, max_pop);
}

#[test]
fn test_tilted_iowa_hill_climbing_maximize() {
    let election_cols = vec!["PRES16D", "PRES16R"];
    let (graph, partition) = fixture_with_attributes("IA", election_cols);
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let pop_tol = 0.2;
    let min_pop = ((1.0 - pop_tol) * avg_pop).floor() as u32;
    let max_pop = ((1.0 + pop_tol) * avg_pop).ceil() as u32;
    let params = RecomParams {
        min_pop,
        max_pop,
        num_steps: 500,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"PRES16D","votes_b":"PRES16R"}],"target":"a","aggregation":"mean"}"#;
    let obj_fn = make_objective_fn(config);
    let initial_score = obj_fn(&graph, &partition);
    let final_partition = multi_tilted_runs(
        &graph,
        partition,
        &params,
        1,
        FullRescoreBackend { obj_fn },
        FixedAcceptance { prob: 0.0 },
        true,
        false,
    )
    .unwrap();
    let final_score = obj_fn(&graph, &final_partition);
    assert!(
        final_score >= initial_score,
        "IA hill-climbing maximize: final {} < initial {}",
        final_score,
        initial_score
    );
}

// ================================================================================
// == Tilted runs on VA precincts (2439 nodes, 11 districts, real election data) ==
// ================================================================================

#[rstest]
fn test_tilted_virginia_election_wins(#[values(1, 4)] n_threads: usize) {
    let election_cols = vec!["G18DSEN", "G18RSEN"];
    let (graph, partition) = fixture_with_attributes("VA", election_cols);
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let pop_tol = 0.05;
    let min_pop = ((1.0 - pop_tol) * avg_pop).floor() as u32;
    let max_pop = ((1.0 + pop_tol) * avg_pop).ceil() as u32;
    let params = RecomParams {
        min_pop,
        max_pop,
        num_steps: 500,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"G18DSEN","votes_b":"G18RSEN"}],"target":"a","aggregation":"mean"}"#;
    let obj_fn = make_objective_fn(config);
    let final_partition = multi_tilted_runs(
        &graph,
        partition,
        &params,
        n_threads,
        FullRescoreBackend { obj_fn },
        FixedAcceptance { prob: 0.05 },
        true,
        false,
    )
    .expect("VA tilted run should not fail");
    assert_partition_valid(&graph, &final_partition, min_pop, max_pop);
}

#[test]
fn test_tilted_virginia_multi_election() {
    // Test with two elections at once.
    let election_cols = vec!["G18DSEN", "G18RSEN", "G16DPRS", "G16RPRS"];
    let (graph, partition) = fixture_with_attributes("VA", election_cols);
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let pop_tol = 0.05;
    let min_pop = ((1.0 - pop_tol) * avg_pop).floor() as u32;
    let max_pop = ((1.0 + pop_tol) * avg_pop).ceil() as u32;
    let params = RecomParams {
        min_pop,
        max_pop,
        num_steps: 500,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"G18DSEN","votes_b":"G18RSEN"},{"votes_a":"G16DPRS","votes_b":"G16RPRS"}],"target":"a","aggregation":"mean"}"#;
    let obj_fn = make_objective_fn(config);
    let final_partition = multi_tilted_runs(
        &graph,
        partition,
        &params,
        2,
        FullRescoreBackend { obj_fn },
        FixedAcceptance { prob: 0.05 },
        true,
        false,
    )
    .expect("VA multi-election tilted run should not fail");
    assert_partition_valid(&graph, &final_partition, min_pop, max_pop);
}

// ===========================================
// == Metropolis acceptance rule unit tests ==
// ===========================================

#[test]
fn test_metropolis_accepts_improvements_via_engine_invariant() {
    // The engine never calls accept_worse on a strictly improving proposal,
    // so this test exercises the rule's behavior when delta < 0 only.
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    let rule = MetropolisAcceptance { beta: 1.0 };
    let mut rng = SmallRng::seed_from_u64(42);
    // current=0.0, proposed=-1.0, maximize=true -> delta = -1.0
    // exp(-1.0) ~= 0.3679; just confirm we can call without panicking and the
    // empirical rate over many trials lands close to the analytic value.
    let trials = 20_000;
    let accepts = (0..trials)
        .filter(|_| rule.accept_worse(0.0, -1.0, true, &mut rng))
        .count();
    let rate = accepts as f64 / trials as f64;
    let expected = (-1.0_f64).exp();
    assert!(
        (rate - expected).abs() < 0.02,
        "metropolis empirical rate {} too far from exp(-1) = {}",
        rate,
        expected,
    );
}

#[rstest]
fn test_metropolis_acceptance_rate_decays_with_delta(
    #[values(true, false)] maximize: bool,
) {
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    let rule = MetropolisAcceptance { beta: 1.5 };
    let trials = 30_000;
    // Worse-proposal magnitudes (interpreted in the optimization direction).
    let magnitudes = [0.1, 0.5, 1.0, 2.0];
    let mut rates = Vec::with_capacity(magnitudes.len());
    for (i, mag) in magnitudes.iter().enumerate() {
        let (current, proposed) = if maximize {
            (0.0, -mag)
        } else {
            (0.0, *mag)
        };
        let mut rng = SmallRng::seed_from_u64(2026 + i as u64);
        let accepts = (0..trials)
            .filter(|_| rule.accept_worse(current, proposed, maximize, &mut rng))
            .count();
        let rate = accepts as f64 / trials as f64;
        let expected = (-rule.beta * mag).exp();
        assert!(
            (rate - expected).abs() < 0.02,
            "rate {} far from exp(-beta * {}) = {} (maximize={})",
            rate,
            mag,
            expected,
            maximize,
        );
        rates.push(rate);
    }
    for window in rates.windows(2) {
        assert!(
            window[0] > window[1],
            "metropolis acceptance rate should decrease with worsening delta: {:?}",
            rates,
        );
    }
}

#[test]
fn test_metropolis_zero_beta_accepts_everything() {
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    let rule = MetropolisAcceptance { beta: 0.0 };
    let mut rng = SmallRng::seed_from_u64(7);
    // exp(0 * delta) = 1, so accept_worse must return true for any worse
    // proposal regardless of how bad it is.
    for _ in 0..1_000 {
        assert!(rule.accept_worse(0.0, -1e6, true, &mut rng));
        assert!(rule.accept_worse(0.0, 1e6, false, &mut rng));
    }
}

#[rstest]
fn test_tilted_metropolis_partition_valid_grid(
    #[values(1, 4)] n_threads: usize,
    #[values(true, false)] maximize: bool,
    #[values(0.5, 5.0)] beta: f64,
) {
    let (graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let min_pop: u32 = 5;
    let max_pop: u32 = 7;
    let params = RecomParams {
        min_pop,
        max_pop,
        num_steps: 500,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };
    let final_partition = multi_tilted_runs(
        &graph,
        partition,
        &params,
        n_threads,
        FullRescoreBackend {
            obj_fn: dist0_pop_objective,
        },
        MetropolisAcceptance { beta },
        maximize,
        false,
    )
    .expect("metropolis tilted run should not fail");
    assert_partition_valid(&graph, &final_partition, min_pop, max_pop);
}

// ===========================================================================
// == Cross-backend equivalence: FullRescore and Incremental must agree ==
// ===========================================================================
//
// Plumbing the same `ObjectiveConfig` through both backends with the same RNG
// seed and a single worker thread should produce bit-identical chains.
// `score_candidate` consumes no RNG bits in either backend, so identical
// scores at every candidate => identical accept/reject decisions => identical
// chain trajectories. This catches:
// - drift between `IncrementalObjective::score_proposal` and a from-scratch
//   evaluation of the applied partition,
// - drift in `IncrementalObjective::apply_proposal` across many steps,
// - any future divergence in how the engine threads `B::State` through
//   `apply_accepted` between the two backend impls.

const ELECTION_WINS_CONFIG: &str = r#"{"objective":"election_wins","elections":[{"votes_a":"PRES16D","votes_b":"PRES16R"}],"target":"a","aggregation":"mean"}"#;

// Per-objective math equivalence for `gingles_partial` and `polsby_popper`
// is covered in `src/objectives.rs::tests::*_incremental_matches_full_score`.
// The IA / VA fixtures don't carry the int-typed columns those objectives
// need, so the engine-level cross-backend tests below stay on `election_wins`.

#[rstest]
fn test_backends_agree_on_iowa_election_wins(
    #[values(true, false)] maximize: bool,
    #[values(1, 7, 42, 101, 2025)] seed: u64,
) {
    let (mut graph, partition) = fixture_with_attributes("IA", vec!["PRES16D", "PRES16R"]);
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let pop_tol = 0.2;
    let min_pop = ((1.0 - pop_tol) * avg_pop).floor() as u32;
    let max_pop = ((1.0 + pop_tol) * avg_pop).ceil() as u32;
    let params = RecomParams {
        min_pop,
        max_pop,
        num_steps: 500,
        rng_seed: seed,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };

    let obj_fn = make_objective_fn(ELECTION_WINS_CONFIG);
    let objective = make_objective(ELECTION_WINS_CONFIG);
    objective.cache_graph_cols(&mut graph);

    let final_full = multi_tilted_runs(
        &graph,
        partition.clone(),
        &params,
        1,
        FullRescoreBackend { obj_fn },
        FixedAcceptance { prob: 0.05 },
        maximize,
        false,
    )
    .expect("full-rescore run should not fail");

    let final_inc = multi_tilted_runs(
        &graph,
        partition,
        &params,
        1,
        IncrementalBackend { objective },
        FixedAcceptance { prob: 0.05 },
        maximize,
        false,
    )
    .expect("incremental run should not fail");

    assert_eq!(
        final_full.assignments, final_inc.assignments,
        "backends produced different chains (election_wins, maximize={}, seed={})",
        maximize, seed,
    );
}

#[rstest]
fn test_backends_agree_on_virginia_election_wins(
    #[values(true, false)] maximize: bool,
    #[values(1, 42, 2025)] seed: u64,
) {
    let (mut graph, partition) = fixture_with_attributes("VA", vec!["G18DSEN", "G18RSEN"]);
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let pop_tol = 0.05;
    let min_pop = ((1.0 - pop_tol) * avg_pop).floor() as u32;
    let max_pop = ((1.0 + pop_tol) * avg_pop).ceil() as u32;
    let params = RecomParams {
        min_pop,
        max_pop,
        num_steps: 250,
        rng_seed: seed,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"G18DSEN","votes_b":"G18RSEN"}],"target":"a","aggregation":"mean"}"#;

    let obj_fn = make_objective_fn(config);
    let objective = make_objective(config);
    objective.cache_graph_cols(&mut graph);

    let final_full = multi_tilted_runs(
        &graph,
        partition.clone(),
        &params,
        1,
        FullRescoreBackend { obj_fn },
        FixedAcceptance { prob: 0.1 },
        maximize,
        false,
    )
    .expect("full-rescore run should not fail");

    let final_inc = multi_tilted_runs(
        &graph,
        partition,
        &params,
        1,
        IncrementalBackend { objective },
        FixedAcceptance { prob: 0.1 },
        maximize,
        false,
    )
    .expect("incremental run should not fail");

    assert_eq!(
        final_full.assignments, final_inc.assignments,
        "backends produced different chains (VA election_wins, maximize={}, seed={})",
        maximize, seed,
    );
}

#[rstest]
fn test_backends_agree_under_metropolis(
    #[values(0.5, 5.0)] beta: f64,
    #[values(7, 2025)] seed: u64,
) {
    let (mut graph, partition) = fixture_with_attributes("IA", vec!["PRES16D", "PRES16R"]);
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let pop_tol = 0.2;
    let min_pop = ((1.0 - pop_tol) * avg_pop).floor() as u32;
    let max_pop = ((1.0 + pop_tol) * avg_pop).ceil() as u32;
    let params = RecomParams {
        min_pop,
        max_pop,
        num_steps: 500,
        rng_seed: seed,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };

    let obj_fn = make_objective_fn(ELECTION_WINS_CONFIG);
    let objective = make_objective(ELECTION_WINS_CONFIG);
    objective.cache_graph_cols(&mut graph);

    let final_full = multi_tilted_runs(
        &graph,
        partition.clone(),
        &params,
        1,
        FullRescoreBackend { obj_fn },
        MetropolisAcceptance { beta },
        true,
        false,
    )
    .expect("full-rescore metropolis run should not fail");

    let final_inc = multi_tilted_runs(
        &graph,
        partition,
        &params,
        1,
        IncrementalBackend { objective },
        MetropolisAcceptance { beta },
        true,
        false,
    )
    .expect("incremental metropolis run should not fail");

    assert_eq!(
        final_full.assignments, final_inc.assignments,
        "backends produced different chains (metropolis, beta={}, seed={})",
        beta, seed,
    );
}

// Deeper check on a single config: assert the per-step score trajectory matches
// exactly. If the chains agree at every step (not just at the end), the chain
// trajectories are bit-identical.
#[test]
fn test_backends_agree_on_full_score_trajectory() {
    let (mut graph, partition) = fixture_with_attributes("IA", vec!["PRES16D", "PRES16R"]);
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let pop_tol = 0.2;
    let min_pop = ((1.0 - pop_tol) * avg_pop).floor() as u32;
    let max_pop = ((1.0 + pop_tol) * avg_pop).ceil() as u32;
    let params = RecomParams {
        min_pop,
        max_pop,
        num_steps: 500,
        rng_seed: 20260503,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };

    let obj_fn = make_objective_fn(ELECTION_WINS_CONFIG);
    let objective = make_objective(ELECTION_WINS_CONFIG);
    objective.cache_graph_cols(&mut graph);

    let pid = std::process::id();
    let path_full =
        std::env::temp_dir().join(format!("frcw_xback_full_{}.csv", pid));
    let path_inc =
        std::env::temp_dir().join(format!("frcw_xback_inc_{}.csv", pid));
    {
        let mut full_writer = ScoresWriter::new(Box::new(std::io::BufWriter::new(
            fs::File::create(&path_full).unwrap(),
        )));
        multi_tilted_runs_with_writer(
            &graph,
            partition.clone(),
            &params,
            1,
            FullRescoreBackend { obj_fn },
            FixedAcceptance { prob: 0.05 },
            true,
            None,
            Some(&mut full_writer),
            false,
        )
        .expect("full-rescore run should not fail");
    }
    {
        let mut inc_writer = ScoresWriter::new(Box::new(std::io::BufWriter::new(
            fs::File::create(&path_inc).unwrap(),
        )));
        multi_tilted_runs_with_writer(
            &graph,
            partition,
            &params,
            1,
            IncrementalBackend { objective },
            FixedAcceptance { prob: 0.05 },
            true,
            None,
            Some(&mut inc_writer),
            false,
        )
        .expect("incremental run should not fail");
    }

    let full_text = fs::read_to_string(&path_full).unwrap();
    let inc_text = fs::read_to_string(&path_inc).unwrap();
    let full_lines: Vec<&str> = full_text.lines().collect();
    let inc_lines: Vec<&str> = inc_text.lines().collect();
    assert_eq!(
        full_lines.len(),
        inc_lines.len(),
        "score trajectories have different lengths"
    );

    // Per-step score (column 1) must match exactly. The full-rescore writer
    // emits no district columns; the incremental writer emits per-district
    // columns, so we only compare step + score.
    for (idx, (full_line, inc_line)) in full_lines.iter().zip(inc_lines.iter()).enumerate() {
        if idx == 0 {
            continue; // header
        }
        let full_fields: Vec<&str> = full_line.split(',').collect();
        let inc_fields: Vec<&str> = inc_line.split(',').collect();
        assert_eq!(
            &full_fields[..2],
            &inc_fields[..2],
            "trajectories diverge at line {}: full={} inc={}",
            idx,
            full_line,
            inc_line,
        );
    }

    fs::remove_file(&path_full).unwrap();
    fs::remove_file(&path_inc).unwrap();
}
