use binary_ensemble::io::bundle::BendlReader;
use serde_json::{json, Value};
use std::{fs, path::PathBuf, process::Command};

const BASE_ARGS: &[&str] = &[
    "--graph-json",
    "test_fixtures/graphs/6x6.json",
    "--n-steps",
    "25",
    "--tol",
    "0.25",
    "--pop-col",
    "population",
    "--assignment-col",
    "district",
    "--rng-seed",
    "17",
    "--variant",
    "district-pairs-mst",
    "--writer",
    "canonical",
];

fn run_chain(extra_args: &[&str]) -> Vec<u8> {
    let output = Command::new(env!("CARGO_BIN_EXE_rustrecom"))
        .args(BASE_ARGS)
        .args(extra_args)
        .output()
        .expect("run rustrecom");
    assert!(
        output.status.success(),
        "status: {:?}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

/// A config equivalent to `BASE_ARGS`: envelope, required fields, and any
/// requested overrides. Optional fields are left to their defaults, which the
/// stdout-equivalence test below depends on.
fn config(writer: &str, output: Option<&str>, constraint: Option<Value>) -> String {
    let mut document = json!({
        "version": 1,
        "command": "chain",
        "graph_json": "test_fixtures/graphs/6x6.json",
        "n_steps": 25,
        "tol": 0.25,
        "pop_col": "population",
        "assignment_col": "district",
        "rng_seed": 17,
        "variant": "district-pairs-mst",
        "writer": writer,
    });
    if let Some(path) = output {
        document["output_file"] = json!(path);
    }
    if let Some(spec) = constraint {
        document["constraint"] = spec;
    }
    document.to_string()
}

fn run_config(config: &str, extra_args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rustrecom"))
        .arg("--config")
        .arg(config)
        .args(extra_args)
        .output()
        .expect("run rustrecom config mode")
}

fn temp_output(name: &str, suffix: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "rustrecom_config_test_{}_{}_{}",
        std::process::id(),
        name,
        suffix
    ))
}

fn metadata_path(output_path: &PathBuf) -> PathBuf {
    let stem = output_path.file_stem().unwrap().to_string_lossy();
    output_path.with_file_name(format!("{stem}_metadata.jsonl"))
}

#[test]
fn show_progress_does_not_change_stdout() {
    assert_eq!(run_chain(&[]), run_chain(&["--show-progress"]));
}

#[test]
fn empty_constraint_does_not_change_stdout() {
    assert_eq!(run_chain(&[]), run_chain(&["--constraint", ""]));
}

#[test]
fn config_mode_matches_cli_stdout() {
    let raw = config("canonical", None, None);
    let output = run_config(&raw, &[]);
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, run_chain(&[]));
}

#[test]
fn default_metadata_does_not_claim_subsampling() {
    let output = Command::new(env!("CARGO_BIN_EXE_rustrecom"))
        .args(&BASE_ARGS[..BASE_ARGS.len() - 2])
        .args(["--writer", "jsonl"])
        .output()
        .expect("run rustrecom");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let metadata: Value = serde_json::from_str(stdout.lines().next().unwrap()).unwrap();
    assert!(metadata["meta"].get("sample_interval").is_none());
}

#[test]
fn sample_interval_keeps_seed_and_original_sample_numbers() {
    let output = run_chain(&["--sample-interval", "7"]);
    let samples = String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|line| {
            serde_json::from_str::<Value>(line).unwrap()["sample"]
                .as_u64()
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(samples, vec![1, 8, 15, 22]);
}

#[test]
fn config_sample_interval_matches_cli() {
    let mut document: Value = serde_json::from_str(&config("canonical", None, None)).unwrap();
    document["sample_interval"] = json!(7);
    let output = run_config(&document.to_string(), &[]);
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, run_chain(&["--sample-interval", "7"]));
}

#[test]
fn sample_interval_rejects_invalid_values_and_writers() {
    let mut document: Value = serde_json::from_str(&config("jsonl", None, None)).unwrap();
    document["sample_interval"] = json!(2);
    let output = run_config(&document.to_string(), &[]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("only supported by assignment-producing writers"));

    let mut document: Value = serde_json::from_str(&config("canonical", None, None)).unwrap();
    document["sample_interval"] = json!(0);
    let output = run_config(&document.to_string(), &[]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("must be at least 1"));
}

#[test]
fn config_file_argument_matches_inline_config() {
    let config_path = temp_output("config_file", "config.json");
    let raw = config("canonical", None, None);
    fs::write(&config_path, &raw).unwrap();
    let from_file = Command::new(env!("CARGO_BIN_EXE_rustrecom"))
        .arg("--config")
        .arg(&config_path)
        .output()
        .expect("run rustrecom --config <path>");
    assert!(
        from_file.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&from_file.stderr)
    );
    assert_eq!(from_file.stdout, run_config(&raw, &[]).stdout);
    fs::remove_file(config_path).unwrap();
}

#[test]
fn config_stdin_argument_matches_inline_config() {
    use std::io::Write;
    use std::process::Stdio;

    let raw = config("canonical", None, None);
    let mut child = Command::new(env!("CARGO_BIN_EXE_rustrecom"))
        .arg("--config")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn rustrecom --config -");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(raw.as_bytes())
        .unwrap();
    let from_stdin = child.wait_with_output().expect("run rustrecom --config -");
    assert!(
        from_stdin.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&from_stdin.stderr)
    );
    assert_eq!(from_stdin.stdout, run_config(&raw, &[]).stdout);
}

#[test]
fn config_mode_rejects_cli_arguments() {
    let raw = config("canonical", None, None);
    let output = run_config(&raw, &["--writer", "canonical"]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("cannot be combined with CLI arguments: writer"));
}

#[test]
fn config_mode_rejects_unknown_fields_and_wrong_envelopes() {
    // Typos fail loudly instead of being silently ignored.
    let mut document: Value = serde_json::from_str(&config("canonical", None, None)).unwrap();
    document["pop_tol"] = json!(0.25);
    let output = run_config(&document.to_string(), &[]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown field `pop_tol`"));

    // A config addressed to a different subcommand is refused.
    let mut document: Value = serde_json::from_str(&config("canonical", None, None)).unwrap();
    document["command"] = json!("short-bursts");
    let output = run_config(&document.to_string(), &[]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Expected command 'chain'"));
}

#[test]
fn config_mode_writes_exact_sidecar_for_canonical_output() {
    let output_path = temp_output("canonical", "plans.jsonl");
    let sidecar_path = metadata_path(&output_path);
    let _ = fs::remove_file(&output_path);
    let _ = fs::remove_file(&sidecar_path);

    let raw = config("canonical", output_path.to_str(), None);
    let output = run_config(&raw, &["--overwrite-output"]);
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output_path.exists());
    assert_eq!(
        fs::read_to_string(&sidecar_path).unwrap(),
        format!("{raw}\n")
    );

    fs::remove_file(output_path).unwrap();
    fs::remove_file(sidecar_path).unwrap();
}

#[test]
fn config_mode_embeds_exact_config_in_jsonl_metadata() {
    let output_path = temp_output("jsonl", "plans.jsonl");
    let sidecar_path = metadata_path(&output_path);
    let _ = fs::remove_file(&output_path);
    let _ = fs::remove_file(&sidecar_path);

    let raw = config("jsonl", output_path.to_str(), None);
    let output = run_config(&raw, &["--overwrite-output"]);
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let first_line = fs::read_to_string(&output_path)
        .unwrap()
        .lines()
        .next()
        .unwrap()
        .to_string();
    let metadata: Value = serde_json::from_str(&first_line).unwrap();
    assert_eq!(metadata["meta"]["config"], raw);
    assert!(!sidecar_path.exists());

    fs::remove_file(output_path).unwrap();
}

#[test]
fn config_mode_embeds_exact_config_in_bendl_metadata() {
    let output_path = temp_output("bendl", "plans.bendl");
    let sidecar_path = metadata_path(&output_path);
    let _ = fs::remove_file(&output_path);
    let _ = fs::remove_file(&sidecar_path);

    let raw = config("bendl", output_path.to_str(), None);
    let output = run_config(&raw, &["--overwrite-output"]);
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let file = fs::File::open(&output_path).expect("open bundle");
    let mut reader = BendlReader::open(file).expect("parse bundle");
    let entry = reader
        .find_asset_by_name("metadata.json")
        .expect("metadata asset present")
        .clone();
    let metadata = reader.asset_bytes(&entry).expect("read metadata asset");
    assert_eq!(String::from_utf8(metadata).unwrap(), raw);
    assert!(!sidecar_path.exists());

    fs::remove_file(output_path).unwrap();
}

#[test]
fn config_mode_runs_a_single_constraint() {
    let constraint = json!({
        "constraint": "district_share_floor",
        "numerator_col": "population",
        "denominator_cols": ["population"],
        "threshold": 0.5
    });
    let raw = config("canonical", None, Some(constraint));
    let output = run_config(&raw, &[]);
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // The v1 schema has exactly one `constraint` slot; a plural field is a
    // schema violation, not a silently dropped list.
    let mut document: Value = serde_json::from_str(&config("canonical", None, None)).unwrap();
    document["constraints"] = json!([]);
    let output = run_config(&document.to_string(), &[]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown field `constraints`"));
}
