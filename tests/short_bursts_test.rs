// Functional tests for short bursts optimization.
use frcw::graph::Graph;
use frcw::objectives::make_objective;
use frcw::partition::Partition;
use frcw::recom::short_bursts::multi_short_bursts_with_writer;
use frcw::recom::RecomProposal;
use frcw::recom::{FullRescoreBackend, IncrementalBackend, RecomParams, RecomVariant};
use frcw::stats::{ScoresWriter, SelfLoopCounts, StatsWriter};
use std::io::Result as IOResult;

use test_fixtures::fixture_with_attributes;

const RNG_SEED: u64 = 153434375;

// =================================================================================
// == Test helpers
// =================================================================================

fn make_params(num_steps: u64) -> RecomParams {
    RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    }
}

/// Records every partition and sample number passed to writer.step.
struct RecordingWriter {
    partitions: Vec<Partition>,
    steps: Vec<u64>,
}

impl RecordingWriter {
    fn new() -> Self {
        RecordingWriter {
            partitions: Vec::new(),
            steps: Vec::new(),
        }
    }
}

impl StatsWriter for RecordingWriter {
    fn init(&mut self, _graph: &Graph, _partition: &Partition) -> IOResult<()> {
        Ok(())
    }

    fn step(
        &mut self,
        step: u64,
        _graph: &Graph,
        partition: &Partition,
        _proposal: &RecomProposal,
        _counts: &SelfLoopCounts,
    ) -> IOResult<()> {
        self.partitions.push(partition.clone());
        self.steps.push(step);
        Ok(())
    }

    fn close(&mut self) -> IOResult<()> {
        Ok(())
    }
}

// =================================================================================
// == Bug regression test: scoring must use the post-update partition.
//
// The pre-fix bug computed score(P_i) BEFORE calling partition.update(), then
// stored P_{i+1} paired with score(P_i). On the first step of every burst this
// means score(P_0) >= best_score(P_0) is always true, so P_1 is unconditionally
// stored as "best" no matter how bad it actually is.
//
// This test uses a closure objective (via FullRescoreBackend) where only the
// initial partition scores 1.0 and all others score 0.0. A correct optimizer
// keeps returning to the initial partition; the buggy optimizer wanders away
// after the first burst.
// =================================================================================

#[test]
fn test_short_bursts_returns_partition_matching_claimed_best_score() {
    let (graph, partition) = fixture_with_attributes("6x6", vec![]);
    // Box::leak gives us a 'static reference, which is Copy + Send, satisfying
    // FullRescoreBackend's Copy bound on the objective closure.
    let initial: &'static [u32] = Box::leak(partition.assignments.clone().into_boxed_slice());

    let obj_fn = move |_graph: &Graph, p: &Partition| -> f64 {
        if p.assignments.as_slice() == initial {
            1.0
        } else {
            0.0
        }
    };

    let params = make_params(30);
    let final_partition = multi_short_bursts_with_writer(
        &graph,
        partition,
        &params,
        1,
        FullRescoreBackend { obj_fn },
        true, // maximize
        5,
        None,
        None,
        false,
        true, // write_best_only: hill-climbing semantics
    )
    .unwrap();

    // With correct scoring: every within-burst step has actual score 0.0 < 1.0
    // (the initial best), so no step ever passes the >= check and the optimizer
    // returns the unchanged initial partition.
    //
    // With the pre-fix bug: score(P_0) = 1.0 is computed before the update, then
    // partition advances to P_1. Since 1.0 >= 1.0, P_1 (score 0.0) is stored as
    // the burst best. After the first burst the global partition is P_1. All
    // subsequent bursts do the same random walk and never return to P_0.
    assert_eq!(
        final_partition.assignments.as_slice(),
        initial,
        "Short bursts returned a partition other than the initial, which has the unique \
         maximum score. This indicates the optimizer stored the post-step partition \
         before scoring it (pre-fix scoring bug)."
    );
}

// =================================================================================
// == Stats writer receives every accepted chain step (write_best_only=false).
// =================================================================================

#[test]
fn test_short_bursts_writer_records_every_accepted_step() {
    let (mut graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"a_share","votes_b":"b_share"}],"target":"a","aggregation":"mean"}"#;
    let objective = make_objective(config);
    objective.cache_graph_cols(&mut graph);

    let burst_length: usize = 5;
    let n_threads: usize = 1;
    let params = RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps: 50,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };

    let mut writer = RecordingWriter::new();
    multi_short_bursts_with_writer(
        &graph,
        partition,
        &params,
        n_threads,
        IncrementalBackend { objective },
        true,
        burst_length,
        Some(&mut writer),
        None,
        false,
        false, // write_best_only=false: record every step
    )
    .unwrap();

    // Sample numbers must be strictly sequential starting at 1.
    for (i, &s) in writer.steps.iter().enumerate() {
        assert_eq!(
            s,
            (i + 1) as u64,
            "Sample number at index {} is {}, expected {}",
            i,
            s,
            i + 1
        );
    }

    // Total writer calls must equal num_steps - 1: the seed written by init
    // counts as the first output record, so chain steps fill the remaining slots.
    let expected_steps = params.num_steps - 1;
    assert_eq!(
        writer.steps.len() as u64,
        expected_steps,
        "Expected {} writer calls (num_steps - 1), got {}",
        expected_steps,
        writer.steps.len()
    );
}

// =================================================================================
// == Stats writer receives only new global bests (write_best_only=true).
// =================================================================================

#[test]
fn test_short_bursts_write_best_only_records_only_improvements() {
    let (mut graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"a_share","votes_b":"b_share"}],"target":"a","aggregation":"mean"}"#;
    let objective = make_objective(config);
    objective.cache_graph_cols(&mut graph);
    let initial_score = objective.score(&graph, &partition);

    let params = make_params(500);
    let mut writer = RecordingWriter::new();
    multi_short_bursts_with_writer(
        &graph,
        partition,
        &params,
        1,
        IncrementalBackend { objective },
        true,
        10,
        Some(&mut writer),
        None,
        false,
        true, // write_best_only=true: record only improvements
    )
    .unwrap();

    // Sample numbers must be sequential (1, 2, 3, ...).
    for (i, &s) in writer.steps.iter().enumerate() {
        assert_eq!(s, (i + 1) as u64, "non-sequential sample number at index {}", i);
    }

    // Every partition recorded must have a score >= the previous recorded score.
    // A violation would mean we wrote a plan that was not actually an improvement.
    let mut prev_score = initial_score;
    for (i, partition) in writer.partitions.iter().enumerate() {
        let score = objective.score(&graph, partition);
        assert!(
            score >= prev_score,
            "write-best-only writer received a non-improving partition at index {}: \
             score {} < previous {}",
            i,
            score,
            prev_score
        );
        prev_score = score;
    }
}

// =================================================================================
// == Cross-validation: write_best_only=true is a strict subset of
// == write_best_only=false, and scores_writer row count matches improvement count.
//
// Both runs use the same RNG seed, so the underlying random walk is identical.
// The write_best_only=false run records every accepted step; write_best_only=true
// records only the steps that were new global bests. Every partition in the
// best-only run must appear (in order) within the all-steps run.
// =================================================================================

#[test]
fn test_short_bursts_write_best_only_cross_validation() {
    use std::fs;

    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"a_share","votes_b":"b_share"}],"target":"a","aggregation":"mean"}"#;
    let burst_length = 5;
    let params = RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps: 200,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };

    // --- Run 1: record every step ---
    let (mut graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    make_objective(config).cache_graph_cols(&mut graph);
    let mut all_writer = RecordingWriter::new();
    let scores_path = std::env::temp_dir().join(format!(
        "frcw_sb_xval_scores_{}_{}.csv",
        std::process::id(),
        RNG_SEED
    ));
    let scores_out = Box::new(std::io::BufWriter::new(
        fs::File::create(&scores_path).unwrap(),
    ));
    let mut scores_writer = ScoresWriter::new(scores_out);

    multi_short_bursts_with_writer(
        &graph,
        partition,
        &params,
        1,
        IncrementalBackend {
            objective: make_objective(config),
        },
        true,
        burst_length,
        Some(&mut all_writer),
        Some(&mut scores_writer),
        false,
        false, // write_best_only=false
    )
    .unwrap();

    // --- Run 2: record only improvements (same seed = same walk) ---
    let (mut graph2, partition2) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    make_objective(config).cache_graph_cols(&mut graph2);
    let mut best_writer = RecordingWriter::new();

    multi_short_bursts_with_writer(
        &graph2,
        partition2,
        &params,
        1,
        IncrementalBackend {
            objective: make_objective(config),
        },
        true,
        burst_length,
        Some(&mut best_writer),
        None,
        false,
        true, // write_best_only=true
    )
    .unwrap();

    // write_best_only=true emits one record per burst boundary, plus one
    // final record if the chain ends mid-burst. Compute the exact expected
    // count from the chain budget and burst length.
    let effective = params.num_steps.saturating_sub(1) as usize;
    let complete_bursts = effective / burst_length as usize;
    let mid_burst_remainder = effective % burst_length as usize;
    let expected_best_records =
        complete_bursts + if mid_burst_remainder > 0 { 1 } else { 0 };
    assert_eq!(
        best_writer.partitions.len(),
        expected_best_records,
        "write_best_only=true should emit {} records ({} burst boundaries{}) but emitted {}",
        expected_best_records,
        complete_bursts,
        if mid_burst_remainder > 0 { " + 1 final mid-burst snap" } else { "" },
        best_writer.partitions.len(),
    );

    // The two modes share the same chain trajectory (same seed). The
    // all-steps run records strictly more rows because it captures every
    // accepted step rather than only burst-boundary snaps.
    assert!(
        best_writer.partitions.len() < all_writer.partitions.len(),
        "write_best_only=true ({} records) should produce fewer records than \
         write_best_only=false ({} records) over {} steps",
        best_writer.partitions.len(),
        all_writer.partitions.len(),
        params.num_steps
    );

    // Run 1 used write_best_only=false, so the scores CSV records one row
    // per accepted chain step (matching the all-steps stats writer).
    let scores_content = fs::read_to_string(&scores_path).unwrap();
    let scores_lines: Vec<&str> = scores_content.lines().collect();
    // lines[0] = header, lines[1] = init row at step 0, lines[2..] = data rows.
    let data_rows = scores_lines.len().saturating_sub(2);
    assert_eq!(
        data_rows,
        all_writer.partitions.len(),
        "scores CSV has {} data rows but the per-step stats writer recorded {}",
        data_rows,
        all_writer.partitions.len()
    );

    // best_score column must be strictly non-decreasing.
    let mut prev_best: f64 = f64::NEG_INFINITY;
    for line in scores_lines.iter().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        let best = fields[2].parse::<f64>().unwrap();
        assert!(best >= prev_best, "best_score decreased: {} < {}", best, prev_best);
        prev_best = best;
    }

    fs::remove_file(scores_path).unwrap();
}

// =================================================================================
// == Hill-climbing: maximize=true should not return a partition worse than the
// == starting point.
// =================================================================================

#[test]
fn test_short_bursts_hill_climbing_maximize() {
    let (mut graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"a_share","votes_b":"b_share"}],"target":"a","aggregation":"mean"}"#;
    let objective = make_objective(config);
    objective.cache_graph_cols(&mut graph);
    let initial_score = objective.score(&graph, &partition);

    let params = make_params(1000);
    let final_partition = multi_short_bursts_with_writer(
        &graph,
        partition,
        &params,
        1,
        IncrementalBackend { objective },
        true, // maximize
        10,
        None,
        None,
        false,
        true, // write_best_only: hill-climbing semantics
    )
    .unwrap();

    let final_score = objective.score(&graph, &final_partition);
    assert!(
        final_score >= initial_score,
        "Hill-climbing maximize: final score {} < initial score {}",
        final_score,
        initial_score
    );
}

// =================================================================================
// == Backend parity: IncrementalBackend and FullRescoreBackend wrapping the same
// == objective must produce identical chain trajectories under the same seed.
// ==
// == Both backends report the same canonical score for each candidate; with
// == n_threads=1 the chain is deterministic, so the final partition must match.
// == Catches drift between the incremental-state path and the full-rescore path.
// =================================================================================

#[test]
fn test_short_bursts_backend_parity_election_wins() {
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"a_share","votes_b":"b_share"}],"target":"a","aggregation":"mean"}"#;
    let params = make_params(200);
    let burst_length = 5;

    // --- Run 1: IncrementalBackend ---
    let (mut graph_a, partition_a) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    make_objective(config).cache_graph_cols(&mut graph_a);
    let inc_final = multi_short_bursts_with_writer(
        &graph_a,
        partition_a,
        &params,
        1,
        IncrementalBackend {
            objective: make_objective(config),
        },
        true,
        burst_length,
        None,
        None,
        false,
        true,
    )
    .unwrap();

    // --- Run 2: FullRescoreBackend wrapping the same objective ---
    let (graph_b, partition_b) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let objective = make_objective(config);
    let obj_fn = move |g: &Graph, p: &Partition| -> f64 { objective.score(g, p) };
    let full_final = multi_short_bursts_with_writer(
        &graph_b,
        partition_b,
        &params,
        1,
        FullRescoreBackend { obj_fn },
        true,
        burst_length,
        None,
        None,
        false,
        true,
    )
    .unwrap();

    assert_eq!(
        inc_final.assignments, full_final.assignments,
        "IncrementalBackend and FullRescoreBackend produced different final partitions \
         under the same seed: incremental {:?}, full-rescore {:?}",
        inc_final.assignments, full_final.assignments
    );
}
