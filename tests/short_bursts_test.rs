// Functional tests for short bursts optimization.
use rustrecom::graph::Graph;
use rustrecom::objectives::{make_objective, IncrementalObjective};
use rustrecom::partition::Partition;
use rustrecom::recom::short_bursts::multi_short_bursts_with_writer;
use rustrecom::recom::RecomProposal;
use rustrecom::recom::{FullRescoreBackend, IncrementalBackend, RecomParams, RecomVariant};
use rustrecom::stats::{ScoresWriter, SelfLoopCounts, StatsWriter};
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
        false,
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
// == Stats writer receives every accepted chain step.
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
        false, // write_improved_scores_only=false
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
// == Score filtering does not affect the stats/output writer.
// =================================================================================

#[test]
fn test_short_bursts_improved_scores_only_does_not_filter_output_writer() {
    let (mut graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    let config = r#"{"objective":"election_wins","elections":[{"votes_a":"a_share","votes_b":"b_share"}],"target":"a","aggregation":"mean"}"#;
    let objective = make_objective(config);
    objective.cache_graph_cols(&mut graph);

    let params = make_params(50);
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
        true, // write_improved_scores_only=true: stats/output still records every step
    )
    .unwrap();

    // Sample numbers must be sequential (1, 2, 3, ...).
    for (i, &s) in writer.steps.iter().enumerate() {
        assert_eq!(
            s,
            (i + 1) as u64,
            "non-sequential sample number at index {}",
            i
        );
    }

    assert_eq!(writer.steps.len() as u64, params.num_steps - 1);
}

// =================================================================================
// == Cross-validation: write_improved_scores_only filters only the scores CSV.
//
// Both runs use the same RNG seed, so the underlying random walk is identical.
// The unfiltered scores run records every accepted step; improved-scores-only
// records only rows that were new global bests.
// =================================================================================

#[test]
fn test_short_bursts_improved_scores_only_cross_validation() {
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
    let all_scores_path = std::env::temp_dir().join(format!(
        "rustrecom_sb_xval_all_scores_{}_{}.csv",
        std::process::id(),
        RNG_SEED
    ));
    let scores_out = Box::new(std::io::BufWriter::new(
        fs::File::create(&all_scores_path).unwrap(),
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
        false, // write_improved_scores_only=false
    )
    .unwrap();

    // --- Run 2: filter only the scores CSV (same seed = same walk) ---
    let (mut graph2, partition2) = fixture_with_attributes("6x6", vec!["a_share", "b_share"]);
    make_objective(config).cache_graph_cols(&mut graph2);
    let mut best_writer = RecordingWriter::new();
    let best_scores_path = std::env::temp_dir().join(format!(
        "rustrecom_sb_xval_best_scores_{}_{}.csv",
        std::process::id(),
        RNG_SEED
    ));
    let best_scores_out = Box::new(std::io::BufWriter::new(
        fs::File::create(&best_scores_path).unwrap(),
    ));
    let mut best_scores_writer = ScoresWriter::new(best_scores_out);

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
        Some(&mut best_scores_writer),
        false,
        true, // write_improved_scores_only=true
    )
    .unwrap();

    assert_eq!(
        best_writer.partitions.len(),
        all_writer.partitions.len(),
        "score filtering must not affect --output-file/stat records"
    );

    // Run 1 did not filter scores, so the scores CSV records one row
    // per accepted chain step (matching the all-steps stats writer).
    let scores_content = fs::read_to_string(&all_scores_path).unwrap();
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

    // Header is the bare `step,score` (election_wins exposes no per-district
    // scores), with no best_score column. Each data row carries a contiguous
    // step index and a parseable score.
    assert_eq!(scores_lines[0], "step,score");
    for (offset, line) in scores_lines.iter().skip(1).enumerate() {
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].parse::<u64>().unwrap(), offset as u64);
        fields[1].parse::<f64>().unwrap();
    }

    let best_scores_content = fs::read_to_string(&best_scores_path).unwrap();
    let best_scores_lines: Vec<&str> = best_scores_content.lines().collect();
    assert_eq!(best_scores_lines[0], "step,score");

    let parse_row = |line: &str| {
        let fields: Vec<&str> = line.split(',').collect();
        (
            fields[0].parse::<u64>().unwrap(),
            fields[1].parse::<f64>().unwrap(),
        )
    };
    let all_rows: Vec<(u64, f64)> = scores_lines
        .iter()
        .skip(1)
        .map(|line| parse_row(line))
        .collect();
    let mut expected_rows = vec![all_rows[0]];
    let mut best = all_rows[0].1;
    for &row in all_rows.iter().skip(1) {
        if row.1 > best {
            expected_rows.push(row);
            best = row.1;
        }
    }
    let best_rows: Vec<(u64, f64)> = best_scores_lines
        .iter()
        .skip(1)
        .map(|line| parse_row(line))
        .collect();
    assert_eq!(best_rows, expected_rows);

    let mut prev_step = 0;
    let mut prev_score = best_rows[0].1;
    for &(step, score) in best_rows.iter().skip(1) {
        assert!(step > prev_step, "improved-only score steps must increase");
        assert!(
            score > prev_score,
            "improved-only score rows must strictly improve"
        );
        prev_step = step;
        prev_score = score;
    }

    fs::remove_file(all_scores_path).unwrap();
    fs::remove_file(best_scores_path).unwrap();
}

// =================================================================================
// == Per-district vector freshness: every scores CSV row must carry the
// == per-district vector of *that row's* plan, not a stale cached vector.
// ==
// == Regression test for per-step recompute. Short bursts previously replayed
// == the last cached district vector on non-best steps (and, under
// == improved-only score output, always emitted the seed vector). We reconstruct each
// == row's plan from a stats writer and recompute its district scores
// == independently, then require an exact match against the CSV d_* columns.
// =================================================================================

#[test]
fn test_short_bursts_scores_district_vector_tracks_each_step() {
    use std::fs;

    // gingles_partial exposes a per-district vector (min_pop / total_pop per
    // district) that shifts as the plan moves. a_share / population are
    // integer columns on the 6x6 fixture.
    let config = r#"{"objective":"gingles_partial","threshold":0.5,"min_pop":"a_share","total_pop":"population"}"#;
    let (mut graph, partition) = fixture_with_attributes("6x6", vec!["a_share", "population"]);
    let scorer = make_objective(config);
    scorer.cache_graph_cols(&mut graph);

    let initial_partition = partition.clone();
    let params = make_params(80);
    let burst_length = 5;

    let mut stats_writer = RecordingWriter::new();
    let scores_path = std::env::temp_dir().join(format!(
        "rustrecom_sb_district_vector_{}_{}.csv",
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
        Some(&mut stats_writer),
        Some(&mut scores_writer),
        false,
        false, // write_improved_scores_only=false
    )
    .unwrap();

    // Independently recompute a plan's per-district vector via the objective.
    let recompute = |p: &Partition| -> Vec<f64> {
        let state = scorer.init(&graph, p);
        scorer.district_scores(&state)
    };

    let parse_districts = |line: &str| -> Vec<f64> {
        line.split(',')
            .skip(2)
            .map(|f| f.parse::<f64>().unwrap())
            .collect()
    };

    let assert_close = |csv: &[f64], expected: &[f64], ctx: &str| {
        assert_eq!(
            csv.len(),
            expected.len(),
            "district vector length mismatch ({})",
            ctx
        );
        for (i, (got, want)) in csv.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-9,
                "district {} mismatch ({}): csv={} recomputed={}",
                i,
                ctx,
                got,
                want
            );
        }
    };

    let scores_content = fs::read_to_string(&scores_path).unwrap();
    let scores_lines: Vec<&str> = scores_content.lines().collect();
    assert!(
        scores_lines[0].starts_with("step,score,d_0"),
        "expected per-district header, got: {}",
        scores_lines[0]
    );

    // Data rows: [0] = step-0 seed (initial plan); [i] for i>=1 = the i-th
    // accepted step, whose plan the stats writer captured at index i-1.
    let data_lines: Vec<&str> = scores_lines[1..].to_vec();
    assert_eq!(
        data_lines.len(),
        stats_writer.partitions.len() + 1,
        "scores rows should be one seed row plus one per accepted step"
    );

    assert_close(
        &parse_districts(data_lines[0]),
        &recompute(&initial_partition),
        "seed row",
    );

    let seed_vec = parse_districts(data_lines[0]);
    let mut saw_change = false;
    for (i, line) in data_lines.iter().enumerate().skip(1) {
        let csv_vec = parse_districts(line);
        assert_close(
            &csv_vec,
            &recompute(&stats_writer.partitions[i - 1]),
            "step row",
        );
        if csv_vec != seed_vec {
            saw_change = true;
        }
    }

    // Sanity: the plan actually moved, so the per-step comparison is not
    // passing vacuously against a constant (e.g. always-seed) vector.
    assert!(
        saw_change,
        "expected the per-district vector to change across steps"
    );

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
        false,
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
