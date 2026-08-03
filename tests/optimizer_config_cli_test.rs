//! End-to-end tests for `--config` mode on the optimizer subcommands: CLI/config
//! output equivalence, exact-config provenance, and schema rejection.

use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const GRAPH_PATH: &str = "test_fixtures/graphs/6x6.json";

fn objective() -> Value {
    json!({
        "objective": "by_district_abs_deviation",
        "target_values": [0.2, 0.2, 0.2],
        "pov_counts_col": "a_share",
        "total_counts_col": "population"
    })
}

fn rustrecom(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rustrecom"))
        .args(args)
        .output()
        .expect("run rustrecom")
}

fn temp_dir(tag: &str) -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "rustrecom_opt_config_{}_{}_{}",
        tag,
        std::process::id(),
        ts
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn metadata_path(output_path: &Path) -> PathBuf {
    let stem = output_path.file_stem().unwrap().to_string_lossy();
    output_path.with_file_name(format!("{stem}_metadata.jsonl"))
}

/// A short-bursts config equivalent to `sb_cli_args`, with paths substituted.
fn sb_config(output: &Path, scores: &Path) -> String {
    json!({
        "version": 1,
        "command": "short-bursts",
        "graph_json": GRAPH_PATH,
        "n_steps": 20,
        "tol": 0.25,
        "pop_col": "population",
        "assignment_col": "district",
        "rng_seed": 17,
        "burst_length": 5,
        "objective": objective(),
        "maximize": false,
        "writer": "canonical",
        "output_file": output.to_str().unwrap(),
        "scores_output_file": scores.to_str().unwrap(),
    })
    .to_string()
}

fn sb_cli_args(output: &Path, scores: &Path) -> Vec<String> {
    [
        "short-bursts",
        "--graph-json",
        GRAPH_PATH,
        "--n-steps",
        "20",
        "--tol",
        "0.25",
        "--pop-col",
        "population",
        "--assignment-col",
        "district",
        "--rng-seed",
        "17",
        "--burst-length",
        "5",
        "--objective",
        &objective().to_string(),
        "--maximize",
        "false",
        "--writer",
        "canonical",
        "--output-file",
        output.to_str().unwrap(),
        "--scores-output-file",
        scores.to_str().unwrap(),
        "--overwrite-output",
    ]
    .iter()
    .map(|arg| arg.to_string())
    .collect()
}

#[test]
fn short_bursts_config_matches_cli_and_preserves_provenance() {
    let cli_dir = temp_dir("sb_cli");
    let cli_output = cli_dir.join("out.jsonl");
    let cli_scores = cli_dir.join("scores.csv");
    let run = rustrecom(
        &sb_cli_args(&cli_output, &cli_scores)
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
    );
    assert!(
        run.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );

    let config_dir = temp_dir("sb_config");
    let config_output = config_dir.join("out.jsonl");
    let config_scores = config_dir.join("scores.csv");
    let raw = sb_config(&config_output, &config_scores);
    let run = rustrecom(&["short-bursts", "--config", &raw, "--overwrite-output"]);
    assert!(
        run.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );

    // Identical sampler inputs give identical records and scores.
    assert_eq!(
        fs::read(&cli_output).unwrap(),
        fs::read(&config_output).unwrap()
    );
    assert_eq!(
        fs::read(&cli_scores).unwrap(),
        fs::read(&config_scores).unwrap()
    );

    // Config mode's sidecar is the exact raw config; CLI mode's is the meta record.
    assert_eq!(
        fs::read_to_string(metadata_path(&config_output)).unwrap(),
        format!("{raw}\n")
    );
    let cli_meta: Value = serde_json::from_str(
        fs::read_to_string(metadata_path(&cli_output))
            .unwrap()
            .trim(),
    )
    .unwrap();
    assert_eq!(cli_meta["meta"]["type"], "short_bursts");

    fs::remove_dir_all(cli_dir).ok();
    fs::remove_dir_all(config_dir).ok();
}

#[test]
fn tilted_config_matches_cli_and_carries_acceptance_options() {
    let tilted_args = |scores: &Path| -> Vec<String> {
        [
            "tilted",
            "--graph-json",
            GRAPH_PATH,
            "--n-steps",
            "20",
            "--tol",
            "0.25",
            "--pop-col",
            "population",
            "--assignment-col",
            "district",
            "--rng-seed",
            "17",
            "--accept-rule",
            "fixed",
            "--accept-worse-prob",
            "0.05",
            "--objective",
            &objective().to_string(),
            "--maximize",
            "false",
            "--scores-output-file",
            scores.to_str().unwrap(),
            "--overwrite-output",
        ]
        .iter()
        .map(|arg| arg.to_string())
        .collect()
    };

    let cli_dir = temp_dir("tilted_cli");
    let cli_scores = cli_dir.join("scores.csv");
    let run = rustrecom(
        &tilted_args(&cli_scores)
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
    );
    assert!(
        run.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );

    let config_dir = temp_dir("tilted_config");
    let config_scores = config_dir.join("scores.csv");
    let raw = json!({
        "version": 1,
        "command": "tilted",
        "graph_json": GRAPH_PATH,
        "n_steps": 20,
        "tol": 0.25,
        "pop_col": "population",
        "assignment_col": "district",
        "rng_seed": 17,
        "accept_rule": "fixed",
        "accept_worse_prob": 0.05,
        "objective": objective(),
        "maximize": false,
        "scores_output_file": config_scores.to_str().unwrap(),
    })
    .to_string();
    let run = rustrecom(&["tilted", "--config", &raw, "--overwrite-output"]);
    assert!(
        run.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );

    assert_eq!(
        fs::read(&cli_scores).unwrap(),
        fs::read(&config_scores).unwrap()
    );
    assert_eq!(
        fs::read_to_string(metadata_path(&config_scores)).unwrap(),
        format!("{raw}\n")
    );

    fs::remove_dir_all(cli_dir).ok();
    fs::remove_dir_all(config_dir).ok();
}

#[test]
fn optimizer_config_mode_rejects_cli_arguments() {
    let dir = temp_dir("sb_mixed");
    let raw = sb_config(&dir.join("out.jsonl"), &dir.join("scores.csv"));
    let run = rustrecom(&[
        "short-bursts",
        "--config",
        &raw,
        "--burst-length",
        "9",
        "--overwrite-output",
    ]);
    assert!(!run.status.success());
    assert!(String::from_utf8_lossy(&run.stderr)
        .contains("cannot be combined with CLI arguments: burst_length"));
    fs::remove_dir_all(dir).ok();
}

#[test]
fn optimizer_config_mode_rejects_wrong_command_and_unknown_fields() {
    let dir = temp_dir("sb_schema");
    let raw = sb_config(&dir.join("out.jsonl"), &dir.join("scores.csv"));

    let mut document: Value = serde_json::from_str(&raw).unwrap();
    document["command"] = json!("tilted");
    let run = rustrecom(&["short-bursts", "--config", &document.to_string()]);
    assert!(!run.status.success());
    assert!(String::from_utf8_lossy(&run.stderr).contains("Expected command 'short-bursts'"));

    let mut document: Value = serde_json::from_str(&raw).unwrap();
    document["pop_tol"] = json!(0.25);
    let run = rustrecom(&["short-bursts", "--config", &document.to_string()]);
    assert!(!run.status.success());
    assert!(String::from_utf8_lossy(&run.stderr).contains("unknown field `pop_tol`"));

    // Tilted rejects an acceptance rule the CLI's value set would have caught.
    let run = rustrecom(&[
        "tilted",
        "--config",
        &json!({
            "version": 1,
            "command": "tilted",
            "graph_json": GRAPH_PATH,
            "n_steps": 20,
            "tol": 0.25,
            "pop_col": "population",
            "assignment_col": "district",
            "rng_seed": 17,
            "accept_rule": "metropolis",
            "objective": objective(),
        })
        .to_string(),
    ]);
    assert!(!run.status.success());
    assert!(String::from_utf8_lossy(&run.stderr).contains("unknown acceptance rule 'metropolis'"));

    fs::remove_dir_all(dir).ok();
}
