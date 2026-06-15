//! Regression tests for the trailing self-loop fix in `multi_chain`.
//!
//! The output record count must be a true total: `init` emits record 0 and the
//! runner performs `num_steps - 1` events, so a finished BEN stream decodes to
//! exactly `num_steps` samples once self-loops sampled after the last accepted
//! proposal are flushed (previously they were dropped, undercounting the tail).

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use binary_ensemble::io::reader::BenStreamReader;
use serde_json::Value;

use frcw::init::from_networkx_value;
use frcw::partition::Partition;
use frcw::recom::run::multi_chain;
use frcw::recom::{RecomParams, RecomVariant};
use frcw::stats::{BenWriter, StatsWriter};

const GRAPH_JSON: &str = include_str!("../test_fixtures/graphs/6x6.json");
const NODE_COUNT: usize = 36;
const RNG_SEED: u64 = 8675309;

fn temp_path(tag: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!("frcw_ben_{}_{}_{}.ben", tag, std::process::id(), ts));
    path
}

fn load_grid() -> (frcw::graph::Graph, Partition) {
    let data: Value = serde_json::from_str(GRAPH_JSON).unwrap();
    from_networkx_value(data, "population", "district", vec![], vec![], vec![]).unwrap()
}

/// Run `multi_chain` with a `ben` writer to a temp file and decode the frames.
fn run_ben(params: &RecomParams) -> (Result<(), String>, PathBuf, Vec<(Vec<u16>, u16)>) {
    let (graph, partition) = load_grid();
    let path = temp_path("run");
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
