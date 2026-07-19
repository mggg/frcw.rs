//! End-to-end CLI tests for `rustrecom tilted` acceptance-rule configuration.

use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const GRAPH_PATH: &str = "test_fixtures/graphs/6x6.json";
const POP_COL: &str = "population";
const ASSIGNMENT_COL: &str = "district";
const N_STEPS: u64 = 25;

fn temp_path(tag: &str, ext: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!(
        "rustrecom_tilted_cli_{}_{}_{}.{}",
        tag,
        std::process::id(),
        ts,
        ext
    ));
    path
}

fn metadata_path(output_path: &Path) -> PathBuf {
    let stem = output_path.file_stem().unwrap().to_str().unwrap();
    output_path.with_file_name(format!("{}_metadata.jsonl", stem))
}

fn l1_objective() -> &'static str {
    r#"{"objective":"by_district_abs_deviation","target_values":[0.2,0.2,0.2],
        "pov_counts_col":"a_share","total_counts_col":"population"}"#
}

fn base_args(scores_path: &Path) -> Vec<String> {
    vec![
        "--graph-json".to_string(),
        GRAPH_PATH.to_string(),
        "--n-steps".to_string(),
        N_STEPS.to_string(),
        "--tol".to_string(),
        "0.25".to_string(),
        "--pop-col".to_string(),
        POP_COL.to_string(),
        "--assignment-col".to_string(),
        ASSIGNMENT_COL.to_string(),
        "--rng-seed".to_string(),
        "17".to_string(),
        "--objective".to_string(),
        l1_objective().to_string(),
        "--maximize".to_string(),
        "false".to_string(),
        "--scores-output-file".to_string(),
        scores_path.to_str().unwrap().to_string(),
        "--overwrite-output".to_string(),
    ]
}

fn run_tilted(tag: &str, extra_args: &[&str]) -> (PathBuf, PathBuf, Value) {
    let rustrecom = env!("CARGO_BIN_EXE_rustrecom");
    let scores_path = temp_path(tag, "csv");
    let meta_path = metadata_path(&scores_path);
    let mut args = base_args(&scores_path);
    args.extend(extra_args.iter().map(|arg| arg.to_string()));

    let output = Command::new(rustrecom)
        .arg("tilted")
        .args(&args)
        .output()
        .expect("run rustrecom tilted");
    assert!(
        output.status.success(),
        "status: {:?}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let metadata = fs::read_to_string(&meta_path).expect("metadata sidecar");
    let line: Value = serde_json::from_str(metadata.trim()).expect("metadata JSON line");
    (scores_path, meta_path, line["meta"].clone())
}

fn score_rows(path: &Path) -> Vec<(u64, f64)> {
    let csv = fs::read_to_string(path).expect("scores CSV");
    let mut rows = Vec::new();
    for line in csv.lines().skip(1) {
        let mut cols = line.split(',');
        let step = cols.next().unwrap().parse::<u64>().unwrap();
        let score = cols.next().unwrap().parse::<f64>().unwrap();
        rows.push((step, score));
    }
    rows
}

#[test]
fn default_accept_rule_is_linear_and_metadata_tracks_l1_minimization() {
    let (scores_path, meta_path, meta) = run_tilted("default_linear", &[]);

    assert_eq!(meta["accept_rule"].as_str(), Some("linear"));
    assert_eq!(meta["maximize"].as_bool(), Some(false));
    assert!(
        meta.get("acceptance_beta").is_none(),
        "default beta should not be serialized as an explicit user parameter"
    );

    let rows = score_rows(&scores_path);
    assert_eq!(rows.first().map(|row| row.0), Some(0));
    assert_eq!(rows.len(), N_STEPS as usize);
    assert_eq!(rows.last().map(|row| row.0), Some(N_STEPS - 1));
    assert!(rows.iter().all(|(_, score)| score.is_finite()));

    fs::remove_file(scores_path).ok();
    fs::remove_file(meta_path).ok();
}

#[test]
fn linear_acceptance_beta_is_user_visible_and_scores_stay_finite() {
    let (scores_path, meta_path, meta) = run_tilted(
        "linear_beta",
        &["--accept-rule", "linear", "--acceptance-beta", "10"],
    );

    assert_eq!(meta["accept_rule"].as_str(), Some("linear"));
    assert_eq!(meta["acceptance_beta"].as_f64(), Some(10.0));

    let rows = score_rows(&scores_path);
    assert_eq!(
        rows.len(),
        N_STEPS as usize,
        "seed row plus num_steps - 1 chain events"
    );
    assert!(rows.iter().all(|(_, score)| score.is_finite()));

    fs::remove_file(scores_path).ok();
    fs::remove_file(meta_path).ok();
}

#[test]
fn write_improved_scores_only_records_strict_improvements() {
    let (scores_path, meta_path, meta) =
        run_tilted("improved_scores", &["--write-improved-scores-only"]);

    assert_eq!(meta["write_improved_scores_only"].as_bool(), Some(true));

    let rows = score_rows(&scores_path);
    assert_eq!(rows.first().map(|row| row.0), Some(0));
    let mut prev = rows[0];
    for &row in rows.iter().skip(1) {
        assert!(row.0 > prev.0, "improved-only score steps must increase");
        assert!(
            row.1 < prev.1,
            "minimizing improved-only scores must strictly decrease"
        );
        prev = row;
    }

    fs::remove_file(scores_path).ok();
    fs::remove_file(meta_path).ok();
}

#[test]
fn exponential_uses_shared_acceptance_beta() {
    let (scores_path, meta_path, meta) = run_tilted(
        "exponential_beta",
        &["--accept-rule", "exponential", "--acceptance-beta", "3.5"],
    );

    assert_eq!(meta["accept_rule"].as_str(), Some("exponential"));
    assert_eq!(meta["acceptance_beta"].as_f64(), Some(3.5));

    fs::remove_file(scores_path).ok();
    fs::remove_file(meta_path).ok();
}

#[test]
fn fixed_acceptance_rejects_shared_beta_to_avoid_ambiguous_configuration() {
    let rustrecom = env!("CARGO_BIN_EXE_rustrecom");
    let scores_path = temp_path("reject_fixed_beta", "csv");
    let mut args = base_args(&scores_path);
    args.extend([
        "--accept-rule".to_string(),
        "fixed".to_string(),
        "--accept-worse-prob".to_string(),
        "0.5".to_string(),
        "--acceptance-beta".to_string(),
        "2".to_string(),
    ]);

    let output = Command::new(rustrecom)
        .arg("tilted")
        .args(&args)
        .output()
        .expect("run rustrecom tilted");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--acceptance-beta") && stderr.contains("--accept-rule linear"),
        "stderr:\n{}",
        stderr
    );

    fs::remove_file(&scores_path).ok();
    fs::remove_file(metadata_path(&scores_path)).ok();
}

#[test]
fn linear_acceptance_beta_must_be_positive() {
    let rustrecom = env!("CARGO_BIN_EXE_rustrecom");
    for (tag, beta) in [("zero_beta", "0"), ("negative_beta", "-0.1")] {
        let scores_path = temp_path(tag, "csv");
        let mut args = base_args(&scores_path);
        args.extend(["--accept-rule".to_string(), "linear".to_string()]);
        args.push(format!("--acceptance-beta={}", beta));

        let output = Command::new(rustrecom)
            .arg("tilted")
            .args(&args)
            .output()
            .expect("run rustrecom tilted");
        assert!(!output.status.success(), "beta {} should be rejected", beta);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("--acceptance-beta") && stderr.contains("positive"),
            "stderr:\n{}",
            stderr
        );

        fs::remove_file(&scores_path).ok();
        fs::remove_file(metadata_path(&scores_path)).ok();
    }
}

#[test]
fn exponential_acceptance_beta_allows_zero_but_not_negative() {
    let (_, _, meta) = run_tilted(
        "exponential_zero_beta",
        &["--accept-rule", "exponential", "--acceptance-beta", "0"],
    );
    assert_eq!(meta["accept_rule"].as_str(), Some("exponential"));
    assert_eq!(meta["acceptance_beta"].as_f64(), Some(0.0));

    let rustrecom = env!("CARGO_BIN_EXE_rustrecom");
    let scores_path = temp_path("negative_exponential_beta", "csv");
    let mut args = base_args(&scores_path);
    args.extend(["--accept-rule".to_string(), "exponential".to_string()]);
    args.push("--acceptance-beta=-0.1".to_string());

    let output = Command::new(rustrecom)
        .arg("tilted")
        .args(&args)
        .output()
        .expect("run rustrecom tilted");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--acceptance-beta") && stderr.contains("non-negative"),
        "stderr:\n{}",
        stderr
    );

    fs::remove_file(&scores_path).ok();
    fs::remove_file(metadata_path(&scores_path)).ok();
}
