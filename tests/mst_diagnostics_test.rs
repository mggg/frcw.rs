use serde_json::{json, Value};
use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn mst_failure_reports_context_and_writes_debug_artifact() {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "rustrecom_mst_diagnostic_{}_{}",
        std::process::id(),
        timestamp
    ));
    let graph_path = root.join("graph.json");
    let debug_directory = root.join("debug");
    fs::create_dir_all(&root).unwrap();

    // The directed-looking adjacency is connected when traversed from node 0,
    // but its undirected edge list is one edge short. This reproduces the old
    // "expected ... edges in MST" failure at the real CLI seam.
    let graph = json!({
        "directed": false,
        "multigraph": false,
        "graph": [],
        "nodes": [
            {"id": 0, "population": 1, "district": 1},
            {"id": 1, "population": 1, "district": 1},
            {"id": 2, "population": 1, "district": 1},
            {"id": 3, "population": 1, "district": 2}
        ],
        "adjacency": [
            [{"id": 2}, {"id": 3}],
            [{"id": 0}],
            [{"id": 1}],
            [{"id": 0}]
        ]
    });
    fs::write(&graph_path, graph.to_string()).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_rustrecom"))
        .args([
            "--graph-json",
            graph_path.to_str().unwrap(),
            "--n-steps",
            "2",
            "--tol",
            "1",
            "--pop-col",
            "population",
            "--assignment-col",
            "district",
            "--rng-seed",
            "17",
            "--variant",
            "cut-edges-mst",
            "--writer",
            "canonical",
        ])
        .env("RUSTRECOM_MST_DEBUG_DIR", &debug_directory)
        .output()
        .expect("run rustrecom");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Cannot construct a spanning tree"));
    assert!(stderr.contains("found 2 of 3 required tree edges"));
    assert!(stderr.contains("Diagnostic written to"));

    let paths = fs::read_dir(&debug_directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    assert_eq!(paths.len(), 1);
    let diagnostic: Value = serde_json::from_slice(&fs::read(&paths[0]).unwrap()).unwrap();
    assert_eq!(diagnostic["failure"]["error"]["kind"], "disconnected");
    assert_eq!(diagnostic["runner"], "chain");
    assert_eq!(diagnostic["worker_index"], 0);
    assert_eq!(diagnostic["rng_seed"], 18);
    assert_eq!(diagnostic["previous_partition"], Value::Null);
    assert_eq!(
        diagnostic["current_partition"]["assignments"],
        json!([0, 0, 0, 1])
    );
    assert_eq!(
        diagnostic["attempted_districts"],
        json!({"a": 0, "b": 1, "indexing": "internal zero-based"})
    );
    assert_eq!(diagnostic["candidate_partition"], Value::Null);
    assert_eq!(diagnostic["merged_subgraph"]["node_count"], 4);
    assert_eq!(diagnostic["merged_subgraph"]["edge_count"], 2);

    fs::remove_file(&paths[0]).unwrap();
    fs::remove_dir(&debug_directory).unwrap();
    fs::remove_file(graph_path).unwrap();
    fs::remove_dir(root).unwrap();
}
