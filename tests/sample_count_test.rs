//! Regression tests for trailing self-loop handling in `multi_chain` and the
//! stats writers.
//!
//! The output record count must be a true total: `init` emits record 0 and the
//! runner performs `num_steps - 1` events, so a finished BEN stream decodes to
//! exactly `num_steps` samples once self-loops sampled after the last accepted
//! proposal are flushed (previously they were dropped, undercounting the tail).
//! The pcompress/JSONL/TSV tests pin the same terminal-tail contract for those
//! writers, which historically inherited the no-op `self_loop` default.

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use binary_ensemble::io::reader::BenStreamReader;
use serde_json::Value;

use frcw::init::from_networkx_value;
use frcw::partition::Partition;
use frcw::recom::run::multi_chain;
use frcw::recom::{RecomParams, RecomProposal, RecomVariant};
use frcw::stats::{
    BenWriter, JSONLWriter, PcompressWriter, SelfLoopCounts, SelfLoopReason, StatsWriter, TSVWriter,
};

const GRAPH_JSON: &str = include_str!("../test_fixtures/graphs/6x6.json");
const NODE_COUNT: usize = 36;
const RNG_SEED: u64 = 8675309;

fn temp_path(tag: &str, ext: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!(
        "frcw_{}_{}_{}.{}",
        tag,
        std::process::id(),
        ts,
        ext
    ));
    path
}

fn load_grid() -> (frcw::graph::Graph, Partition) {
    let data: Value = serde_json::from_str(GRAPH_JSON).unwrap();
    from_networkx_value(data, "population", "district", vec![], vec![], vec![]).unwrap()
}

/// Run `multi_chain` with a `ben` writer to a temp file and decode the frames.
fn run_ben(params: &RecomParams) -> (Result<(), String>, PathBuf, Vec<(Vec<u16>, u16)>) {
    let (graph, partition) = load_grid();
    let path = temp_path("ben_run", "ben");
    let result = {
        let writer = Box::new(BenWriter::new(Box::new(BufWriter::new(
            File::create(&path).unwrap(),
        )))) as Box<dyn StatsWriter>;
        multi_chain(&graph, &partition, writer, params, 1, 1, false)
    };
    let mut reader = BenStreamReader::from_ben(File::open(&path).unwrap()).unwrap();
    let mut frames: Vec<(Vec<u16>, u16)> = Vec::new();
    reader
        .for_each_assignment(|assignment, count| {
            frames.push((assignment.to_vec(), count));
            Ok(true)
        })
        .unwrap();
    (result, path, frames)
}

#[test]
fn standard_run_decodes_to_num_steps() {
    let num_steps = 300u64;
    let params = RecomParams {
        min_pop: 5,
        max_pop: 7,
        num_steps,
        rng_seed: RNG_SEED,
        balance_ub: 0,
        variant: RecomVariant::DistrictPairsRMST,
        region_weights: None,
        edge_weight_keys: vec![],
    };
    let (result, path, frames) = run_ben(&params);
    assert!(result.is_ok(), "accepting run should succeed: {:?}", result);
    let total: u64 = frames.iter().map(|(_, c)| *c as u64).sum();
    assert_eq!(total, num_steps);
    std::fs::remove_file(&path).ok();
}

#[test]
fn all_rejection_run_emits_full_length_then_errors() {
    // Impossible population bounds reject every proposal. Reversible ReCom turns
    // each rejection into a self-loop (rather than retrying forever), so the
    // chain terminates: the output should be the seed repeated num_steps times,
    // and multi_chain should still report no accepted proposals.
    let (_, seed_partition) = load_grid();
    let expected_seed: Vec<u16> = seed_partition
        .assignments
        .iter()
        .map(|&x| x as u16)
        .collect();
    assert_eq!(expected_seed.len(), NODE_COUNT);

    let num_steps = 100u64;
    let params = RecomParams {
        min_pop: 100,
        max_pop: 100,
        num_steps,
        rng_seed: RNG_SEED,
        balance_ub: 10,
        variant: RecomVariant::Reversible,
        region_weights: None,
        edge_weight_keys: vec![],
    };
    let (result, path, frames) = run_ben(&params);
    assert!(
        result.is_err(),
        "all-rejection run should report no accepted proposals"
    );

    let total: u64 = frames.iter().map(|(_, c)| *c as u64).sum();
    assert_eq!(total, num_steps, "full chain length recorded");
    for (assignment, _) in &frames {
        assert_eq!(assignment, &expected_seed, "every record is the seed plan");
    }
    std::fs::remove_file(&path).ok();
}

#[test]
fn pcompress_all_rejection_run_preserves_terminal_tail() {
    // Same all-rejection setup as above: every event lands in the terminal
    // `self_loop` callback, which pcompress previously dropped entirely.
    let (graph, partition) = load_grid();
    let expected_seed: Vec<u8> = partition.assignments.iter().map(|&x| x as u8).collect();

    let num_steps = 25u64;
    let params = RecomParams {
        min_pop: 100,
        max_pop: 100,
        num_steps,
        rng_seed: RNG_SEED,
        balance_ub: 10,
        variant: RecomVariant::Reversible,
        region_weights: None,
        edge_weight_keys: vec![],
    };
    let path = temp_path("pcompress_run", "pcompress");
    let result = {
        let writer = Box::new(PcompressWriter::new(Box::new(BufWriter::new(
            File::create(&path).unwrap(),
        )))) as Box<dyn StatsWriter>;
        multi_chain(&graph, &partition, writer, &params, 1, 1, false)
    };
    assert!(
        result.is_err(),
        "all-rejection run should report no accepted proposals"
    );

    let mut reader = std::io::BufReader::new(File::open(&path).unwrap());
    let mut decoded: Vec<u8> = Vec::new();
    {
        let mut out = BufWriter::new(&mut decoded);
        pcompress::decode::decode(&mut reader, &mut out, 0, false);
    }
    let samples: Vec<Vec<u8>> = String::from_utf8(decoded)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        samples.len() as u64,
        num_steps,
        "full chain length recorded"
    );
    for sample in &samples {
        assert_eq!(sample, &expected_seed, "every record is the seed plan");
    }
    std::fs::remove_file(&path).ok();
}

#[test]
fn tsv_writes_constraint_and_terminal_self_loop_rows() {
    let (graph, partition) = load_grid();
    let path = temp_path("tsv", "tsv");
    {
        let mut writer = TSVWriter::new(Box::new(BufWriter::new(File::create(&path).unwrap())));
        writer.init(&graph, &partition).unwrap();

        let proposal = RecomProposal {
            a_label: 0,
            b_label: 1,
            a_pop: 6,
            b_pop: 6,
            a_nodes: vec![0, 1],
            b_nodes: vec![2, 3],
        };
        let mut step_counts = SelfLoopCounts::default();
        step_counts.inc(SelfLoopReason::NoSplit);
        step_counts.inc_by(SelfLoopReason::ConstraintViolation, 2);
        writer
            .step(4, &graph, &partition, &proposal, &step_counts)
            .unwrap();

        let mut tail_counts = SelfLoopCounts::default();
        tail_counts.inc_by(SelfLoopReason::ConstraintViolation, 3);
        writer
            .self_loop(7, &graph, &partition, &tail_counts)
            .unwrap();
        writer.close().unwrap();
    }

    let contents = std::fs::read_to_string(&path).unwrap();
    let rows: Vec<Vec<&str>> = contents.lines().map(|l| l.split('\t').collect()).collect();
    assert_eq!(rows.len(), 3, "header, accepted row, terminal row");
    assert_eq!(
        rows[0],
        vec![
            "step",
            "non_adjacent",
            "no_split",
            "seam_length",
            "tilted_rejection",
            "a_label",
            "b_label",
            "a_pop",
            "b_pop",
            "a_nodes",
            "b_nodes",
            "constraint_violation",
        ]
    );

    // Accepted row: original column positions unchanged, constraint count appended.
    assert_eq!(rows[1].len(), 12);
    assert_eq!(rows[1][0], "4");
    assert_eq!(rows[1][2], "1", "no_split count");
    assert_eq!(rows[1][5], "0", "a_label");
    assert_eq!(rows[1][8], "6", "b_pop");
    assert_eq!(rows[1][11], "2", "accepted-row constraint count");

    // Terminal row: same width, blank proposal columns, terminal counts kept.
    assert_eq!(rows[2].len(), 12);
    assert_eq!(rows[2][0], "7");
    assert!(rows[2][5..11].iter().all(|cell| cell.is_empty()));
    assert_eq!(rows[2][11], "3", "terminal constraint count");
    std::fs::remove_file(&path).ok();
}

#[test]
fn jsonl_emits_terminal_self_loop_record() {
    let (graph, partition) = load_grid();
    let path = temp_path("jsonl", "jsonl");
    {
        let mut writer = JSONLWriter::new(
            false,
            false,
            false,
            Box::new(BufWriter::new(File::create(&path).unwrap())),
        );
        writer.init(&graph, &partition).unwrap();

        let mut counts = SelfLoopCounts::default();
        counts.inc(SelfLoopReason::NonAdjacent);
        counts.inc_by(SelfLoopReason::ConstraintViolation, 4);
        writer.self_loop(99, &graph, &partition, &counts).unwrap();
        writer.close().unwrap();
    }

    let contents = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<Value> = contents
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(lines.len(), 2, "init record plus terminal record");
    assert!(lines[0].get("init").is_some());

    let record = lines[1]
        .get("self_loop")
        .expect("terminal record uses a distinct top-level key");
    assert!(
        lines[1].get("step").is_none(),
        "terminal record must not masquerade as an accepted step"
    );
    assert_eq!(record["step"], 99);
    assert_eq!(record["counts"]["non_adjacent"], 1);
    assert_eq!(record["counts"]["constraint_violation"], 4);
    std::fs::remove_file(&path).ok();
}
