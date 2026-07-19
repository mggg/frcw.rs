//! End-to-end tests for the `bendl` file writer.
//!
//! These read bundles back in-process with the Rust `BendlReader` rather than
//! shelling out to the crate's `bendl` CLI, which is not guaranteed on PATH in
//! the test environment. One provenance test does invoke the built `rustrecom`
//! binary (via `CARGO_BIN_EXE_rustrecom`) to exercise the real metadata wiring.

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use binary_ensemble::io::bundle::BendlReader;
use binary_ensemble::io::reader::BenStreamReader;
use serde_json::Value;
use sha3::{Digest, Sha3_256};

use rustrecom::bendl::{reorder_graph_json, BendlGraphOrder};
use rustrecom::init::from_networkx_value;
use rustrecom::recom::run::multi_chain;
use rustrecom::recom::{RecomParams, RecomVariant};
use rustrecom::stats::{BendlBenStreamWriter, StatsWriter};

/// The 6x6 grid fixture: 36 nodes, 6 districts of 6 nodes each.
const GRAPH_JSON: &str = include_str!("../test_fixtures/graphs/6x6.json");
const POP_COL: &str = "population";
const ASSIGNMENT_COL: &str = "district";
const NODE_COUNT: usize = 36;
const NUM_DISTS: u16 = 6;
const RNG_SEED: u64 = 8675309;

fn temp_path(tag: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!(
        "rustrecom_bendl_{}_{}_{}.bendl",
        tag,
        std::process::id(),
        ts
    ));
    path
}

fn sha3_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha3_256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Standard (non-reversible) params that accept proposals on the 6x6 grid.
fn grid_params(num_steps: u64) -> RecomParams {
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

/// Reorder `original` per `order`, build the chain from the resulting bytes, and
/// write a bundle to `path`. Mirrors the binaries' bendl arm. Returns the bytes
/// embedded as the Graph asset (the reordered JSON).
fn write_bundle(
    original: &[u8],
    order: &BendlGraphOrder,
    params: &RecomParams,
    path: &PathBuf,
) -> Vec<u8> {
    let embed = reorder_graph_json(original, order).expect("reorder graph");
    let data: Value = serde_json::from_slice(&embed).expect("parse embedded graph");
    let (graph, partition) =
        from_networkx_value(data, POP_COL, ASSIGNMENT_COL, vec![], vec![], vec![])
            .expect("build graph + partition");
    let buf = BufWriter::new(File::create(path).expect("create bundle file"));
    let writer = Box::new(BendlBenStreamWriter::new(
        buf,
        embed.clone(),
        b"{}".to_vec(),
    )) as Box<dyn StatsWriter>;
    multi_chain(&graph, &partition, writer, params, 1, 1, false).expect("chain run");
    embed
}

struct Decoded {
    sample_count: i64,
    total_records: u64,
    frames: Vec<(Vec<u16>, u16)>,
    graph_asset: Vec<u8>,
    metadata_asset: Vec<u8>,
}

fn decode_bundle(path: &PathBuf) -> Decoded {
    let file = File::open(path).expect("open bundle");
    let mut reader = BendlReader::open(file).expect("parse bundle header + directory");
    let sample_count = reader
        .sample_count()
        .expect("finalized bundle has a sample count");

    let graph_entry = reader
        .find_asset_by_name("graph.json")
        .expect("graph asset present")
        .clone();
    let metadata_entry = reader
        .find_asset_by_name("metadata.json")
        .expect("metadata asset present")
        .clone();
    let graph_asset = reader.asset_bytes(&graph_entry).expect("read graph asset");
    let metadata_asset = reader
        .asset_bytes(&metadata_entry)
        .expect("read metadata asset");

    // Integrity: stream and every asset checksum must verify.
    reader.verify_stream_checksum().expect("stream checksum");
    reader
        .verify_all_asset_checksums()
        .expect("asset checksums");

    let raw = reader
        .assignment_stream_reader_unverified()
        .expect("stream reader");
    let mut ben = BenStreamReader::from_ben(raw).expect("decode BEN banner");
    let mut frames: Vec<(Vec<u16>, u16)> = Vec::new();
    ben.for_each_assignment(|assignment, count| {
        frames.push((assignment.to_vec(), count));
        Ok(true)
    })
    .expect("decode stream");
    let total_records = frames.iter().map(|(_, c)| *c as u64).sum();

    Decoded {
        sample_count,
        total_records,
        frames,
        graph_asset,
        metadata_asset,
    }
}

#[test]
fn bundle_roundtrip_assets_present_and_valid() {
    let path = temp_path("roundtrip");
    write_bundle(
        GRAPH_JSON.as_bytes(),
        &BendlGraphOrder::None,
        &grid_params(150),
        &path,
    );
    let decoded = decode_bundle(&path);
    // Graph and Metadata assets are present and parse as JSON.
    let graph: Value = serde_json::from_slice(&decoded.graph_asset).expect("graph asset is JSON");
    assert_eq!(graph["nodes"].as_array().unwrap().len(), NODE_COUNT);
    let _: Value = serde_json::from_slice(&decoded.metadata_asset).expect("metadata asset is JSON");
    std::fs::remove_file(&path).ok();
}

#[test]
fn absolute_sample_count_triple_identity() {
    // header sample_count == decoded record count == num_steps. The chain emits
    // record 0 in init and num_steps - 1 events, so a vanilla run totals exactly
    // num_steps once trailing self-loops are flushed.
    let num_steps = 250u64;
    let path = temp_path("samplecount");
    write_bundle(
        GRAPH_JSON.as_bytes(),
        &BendlGraphOrder::None,
        &grid_params(num_steps),
        &path,
    );
    let decoded = decode_bundle(&path);
    assert_eq!(decoded.sample_count, num_steps as i64);
    assert_eq!(decoded.total_records, num_steps);
    std::fs::remove_file(&path).ok();
}

#[test]
fn mlc_reorder_roundtrip_aligns_stream_to_embedded_graph() {
    let reordered = reorder_graph_json(GRAPH_JSON.as_bytes(), &BendlGraphOrder::Mlc).unwrap();
    assert_ne!(
        reordered,
        GRAPH_JSON.as_bytes(),
        "mlc should reorder the 6x6 grid"
    );

    let path = temp_path("mlc");
    let embed = write_bundle(
        GRAPH_JSON.as_bytes(),
        &BendlGraphOrder::Mlc,
        &grid_params(150),
        &path,
    );
    let decoded = decode_bundle(&path);

    // The embedded Graph asset is exactly the reordered bytes.
    assert_eq!(decoded.graph_asset, embed);
    assert_eq!(decoded.graph_asset, reordered);

    // The crate renumbers `id` to the new position (0..N-1). Original identity is
    // recovered through preserved node attributes, not `id`: here the multiset of
    // (x, y) grid coordinates is unchanged, just reordered.
    let original: Value = serde_json::from_str(GRAPH_JSON).unwrap();
    let embedded: Value = serde_json::from_slice(&decoded.graph_asset).unwrap();
    let embedded_ids: Vec<i64> = embedded["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["id"].as_i64().unwrap())
        .collect();
    assert_eq!(
        embedded_ids,
        (0..NODE_COUNT as i64).collect::<Vec<_>>(),
        "ids are renumbered to positional indices"
    );
    let coords = |g: &Value| -> Vec<(i64, i64)> {
        let mut c: Vec<(i64, i64)> = g["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| (n["x"].as_i64().unwrap(), n["y"].as_i64().unwrap()))
            .collect();
        c.sort_unstable();
        c
    };
    assert_eq!(
        coords(&original),
        coords(&embedded),
        "node data preserved (coordinate multiset unchanged)"
    );

    // The stream is positionally aligned to the reordered graph: each frame has
    // one label per node, all within district range.
    for (assignment, _) in &decoded.frames {
        assert_eq!(assignment.len(), NODE_COUNT);
        assert!(assignment.iter().all(|&d| d < NUM_DISTS));
    }
    std::fs::remove_file(&path).ok();
}

#[test]
fn determinism_external_reorder_equals_internal_mlc() {
    // (graph pre-reordered externally, order=none) must decode identically to
    // (original graph, order=mlc): the chain RNG is a pure function of seed,
    // final graph, and n_threads, so both produce the same stream.
    let reordered = reorder_graph_json(GRAPH_JSON.as_bytes(), &BendlGraphOrder::Mlc).unwrap();

    let path_external = temp_path("det_external");
    write_bundle(
        &reordered,
        &BendlGraphOrder::None,
        &grid_params(200),
        &path_external,
    );
    let path_internal = temp_path("det_internal");
    write_bundle(
        GRAPH_JSON.as_bytes(),
        &BendlGraphOrder::Mlc,
        &grid_params(200),
        &path_internal,
    );

    let external = decode_bundle(&path_external);
    let internal = decode_bundle(&path_internal);
    assert_eq!(external.graph_asset, internal.graph_asset);
    assert_eq!(external.sample_count, internal.sample_count);
    assert_eq!(external.frames, internal.frames);

    std::fs::remove_file(&path_external).ok();
    std::fs::remove_file(&path_internal).ok();
}

/// Read the `metadata.json` asset of a bundle as a JSON object.
fn read_metadata(path: &PathBuf) -> Value {
    let decoded = decode_bundle(path);
    serde_json::from_slice(&decoded.metadata_asset).expect("metadata is JSON")
}

#[test]
fn provenance_via_rustrecom_binary() {
    let rustrecom = env!("CARGO_BIN_EXE_rustrecom");
    let graph_path = "test_fixtures/graphs/6x6.json";

    // order=none: source and embedded hashes are equal.
    let none_path = temp_path("prov_none");
    let status = Command::new(rustrecom)
        .args([
            "--graph-json",
            graph_path,
            "--n-steps",
            "120",
            "--tol",
            "0.25",
            "--pop-col",
            POP_COL,
            "--assignment-col",
            ASSIGNMENT_COL,
            "--rng-seed",
            "1",
            "--variant",
            "district-pairs-rmst",
            "--writer",
            "bendl",
            "--output-file",
            none_path.to_str().unwrap(),
            "--overwrite-output",
        ])
        .status()
        .expect("run rustrecom bendl none");
    assert!(status.success());
    let meta = read_metadata(&none_path);
    let source = meta["source_graph_sha3"].as_str().unwrap();
    let embedded = meta["embedded_graph_sha3"].as_str().unwrap();
    assert_eq!(meta["bendl_graph_order"].as_str().unwrap(), "none");
    assert_eq!(source, embedded, "order=none: hashes equal");
    assert!(
        meta.get("graph_sha3").is_none(),
        "graph_sha3 replaced for bendl"
    );

    // order=mlc: hashes differ, and embedded matches a fresh hash of the Graph
    // asset bytes.
    let mlc_path = temp_path("prov_mlc");
    let status = Command::new(rustrecom)
        .args([
            "--graph-json",
            graph_path,
            "--n-steps",
            "120",
            "--tol",
            "0.25",
            "--pop-col",
            POP_COL,
            "--assignment-col",
            ASSIGNMENT_COL,
            "--rng-seed",
            "1",
            "--variant",
            "district-pairs-rmst",
            "--writer",
            "bendl",
            "--bendl-graph-order",
            "mlc",
            "--output-file",
            mlc_path.to_str().unwrap(),
            "--overwrite-output",
        ])
        .status()
        .expect("run rustrecom bendl mlc");
    assert!(status.success());
    let decoded = decode_bundle(&mlc_path);
    let meta: Value = serde_json::from_slice(&decoded.metadata_asset).unwrap();
    let source = meta["source_graph_sha3"].as_str().unwrap();
    let embedded = meta["embedded_graph_sha3"].as_str().unwrap();
    assert_eq!(meta["bendl_graph_order"].as_str().unwrap(), "mlc");
    assert_ne!(source, embedded, "order=mlc: hashes differ");
    assert_eq!(
        embedded,
        sha3_hex(&decoded.graph_asset),
        "embedded hash matches the embedded Graph asset bytes"
    );

    std::fs::remove_file(&none_path).ok();
    std::fs::remove_file(&mlc_path).ok();
}

#[test]
fn bendl_writer_without_output_file_is_rejected_by_binary() {
    let rustrecom = env!("CARGO_BIN_EXE_rustrecom");
    let output = Command::new(rustrecom)
        .args([
            "--graph-json",
            "test_fixtures/graphs/6x6.json",
            "--n-steps",
            "50",
            "--tol",
            "0.25",
            "--pop-col",
            POP_COL,
            "--assignment-col",
            ASSIGNMENT_COL,
            "--rng-seed",
            "1",
            "--variant",
            "district-pairs-rmst",
            "--writer",
            "bendl",
        ])
        .output()
        .expect("run rustrecom");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("requires '--output-file'"),
        "stderr: {}",
        stderr
    );
}
