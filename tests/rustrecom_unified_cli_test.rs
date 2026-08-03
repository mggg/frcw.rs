//! End-to-end tests for the unified `rustrecom` CLI and its `frcw` compatibility alias: the
//! default-subcommand shim, subcommand dispatch, short-bursts coverage, and the two intentional
//! behavior changes (nonzero exit on engine errors; tilted rejecting reversible before creating
//! output files). Most cases run through the `frcw` alias, which doubles as coverage that the
//! alias behaves identically to the primary binary.

use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

const GRAPH_PATH: &str = "test_fixtures/graphs/6x6.json";
const OBJECTIVE: &str = r#"{"objective":"by_district_abs_deviation","target_values":[0.2,0.2,0.2],
    "pov_counts_col":"a_share","total_counts_col":"population"}"#;

fn frcw(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_frcw"))
        .args(args)
        .output()
        .expect("run frcw")
}

fn rustrecom(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rustrecom"))
        .args(args)
        .output()
        .expect("run rustrecom")
}

fn temp_path(tag: &str, ext: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!(
        "frcw_unified_cli_{}_{}_{}.{}",
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

/// Common chain arguments in the legacy bare form (no subcommand token).
fn bare_chain_args() -> Vec<&'static str> {
    vec![
        "--graph-json",
        GRAPH_PATH,
        "--n-steps",
        "5",
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
    ]
}

fn short_bursts_args(scores_path: &Path) -> Vec<String> {
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
        OBJECTIVE,
        "--maximize",
        "false",
        "--overwrite-output",
        "--scores-output-file",
    ]
    .iter()
    .map(|arg| arg.to_string())
    .chain([scores_path.to_str().unwrap().to_string()])
    .collect()
}

// --- Default-subcommand shim ---------------------------------------------

#[test]
fn no_arguments_selects_chain_and_reports_chain_required_arguments() {
    let output = frcw(&[]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--graph-json") && stderr.contains("--variant"),
        "expected chain's required arguments, got:\n{}",
        stderr
    );
}

#[test]
fn legacy_bare_chain_arguments_still_run_the_chain() {
    let output = frcw(&bare_chain_args());
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // The chain's jsonl writer emits the meta record on stdout.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let first: Value = serde_json::from_str(stdout.lines().next().unwrap()).unwrap();
    assert_eq!(first["meta"]["chain_variant"], "district-pairs-mst");
}

#[test]
fn deprecated_rmst_variants_still_run_and_warn() {
    for (old, new) in [
        ("cut-edges-rmst", "cut-edges-mst"),
        ("district-pairs-rmst", "district-pairs-mst"),
    ] {
        let mut args = bare_chain_args();
        *args.last_mut().unwrap() = old;
        let output = rustrecom(&args);
        assert!(
            output.status.success(),
            "stderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(&format!("`{old}` is deprecated"))
                && stderr.contains(&format!("use `{new}` instead")),
            "stderr:\n{stderr}"
        );
    }
}

#[test]
fn explicit_chain_subcommand_matches_bare_form() {
    let mut args = vec!["chain"];
    args.extend(bare_chain_args());
    let explicit = frcw(&args);
    let bare = frcw(&bare_chain_args());
    assert!(explicit.status.success());
    assert_eq!(explicit.stdout, bare.stdout);
}

#[test]
fn unknown_first_token_is_an_unrecognized_subcommand() {
    let output = frcw(&["nonsense"]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unrecognized subcommand 'nonsense'"),
        "stderr:\n{}",
        stderr
    );
}

#[test]
fn help_lists_all_three_subcommands() {
    let output = frcw(&["--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    for subcommand in ["chain", "short-bursts", "tilted"] {
        assert!(stdout.contains(subcommand), "missing {}", subcommand);
    }
    // Per-subcommand help still works and shows subcommand-specific options.
    let chain_help = frcw(&["chain", "--help"]);
    assert!(chain_help.status.success());
    assert!(String::from_utf8_lossy(&chain_help.stdout).contains("--constraint"));
}

#[test]
fn version_propagates_to_subcommands_and_survives_legacy_options() {
    let version = env!("CARGO_PKG_VERSION");
    for args in [
        vec!["--version"],
        vec!["tilted", "--version"],
        // A leading legacy chain option must not hide --version.
        vec!["--graph-json", "x", "--version"],
    ] {
        let output = frcw(&args);
        assert!(output.status.success(), "args: {:?}", args);
        assert!(
            String::from_utf8_lossy(&output.stdout).contains(version),
            "args: {:?}",
            args
        );
    }
}

// --- rustrecom primary name and frcw alias --------------------------------

#[test]
fn each_binary_name_reports_itself_in_version_and_usage() {
    let version = env!("CARGO_PKG_VERSION");
    let rustrecom_version = rustrecom(&["--version"]);
    assert!(String::from_utf8_lossy(&rustrecom_version.stdout)
        .contains(&format!("rustrecom {}", version)));
    let frcw_version = frcw(&["--version"]);
    assert!(String::from_utf8_lossy(&frcw_version.stdout).contains(&format!("frcw {}", version)));

    // Usage lines in clap errors carry the invoked name, so old scripts that
    // match on `frcw` stderr keep seeing `frcw`.
    let rustrecom_usage = rustrecom(&[]);
    assert!(String::from_utf8_lossy(&rustrecom_usage.stderr).contains("Usage: rustrecom chain"));
    let frcw_usage = frcw(&[]);
    assert!(String::from_utf8_lossy(&frcw_usage.stderr).contains("Usage: frcw chain"));
}

#[test]
fn alias_output_matches_primary_binary_output() {
    let mut args = vec!["chain"];
    args.extend(bare_chain_args());
    let primary = rustrecom(&args);
    let alias = frcw(&args);
    assert!(primary.status.success());
    assert_eq!(primary.stdout, alias.stdout);
}

// --- Deprecated binary-name shims -----------------------------------------

#[test]
fn frcw_warns_about_deprecation_but_rustrecom_does_not() {
    let alias = frcw(&["--version"]);
    assert!(alias.status.success());
    let alias_stderr = String::from_utf8_lossy(&alias.stderr);
    assert!(
        alias_stderr.contains("`frcw` binary is deprecated")
            && alias_stderr.contains("use `rustrecom` instead"),
        "stderr:\n{}",
        alias_stderr
    );
    let primary = rustrecom(&["--version"]);
    assert!(
        primary.stderr.is_empty(),
        "the primary name must not warn:\n{}",
        String::from_utf8_lossy(&primary.stderr)
    );
}

#[test]
fn frcw_short_bursts_shim_forwards_legacy_invocations_and_warns() {
    let scores_path = temp_path("legacy_sb_shim", "csv");
    // Old-style argument list: no subcommand token.
    let args = short_bursts_args(&scores_path);
    let output = Command::new(env!("CARGO_BIN_EXE_frcw_short_bursts"))
        .args(&args[1..])
        .output()
        .expect("run frcw_short_bursts shim");
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("`frcw_short_bursts` binary is deprecated")
            && stderr.contains("use `rustrecom short-bursts` instead"),
        "stderr:\n{}",
        stderr
    );
    let meta_path = metadata_path(&scores_path);
    let meta: Value = serde_json::from_str(fs::read_to_string(&meta_path).unwrap().trim()).unwrap();
    assert_eq!(meta["meta"]["type"], "short_bursts");

    fs::remove_file(&scores_path).ok();
    fs::remove_file(&meta_path).ok();
}

#[test]
fn frcw_tilted_shim_forwards_help_and_keeps_top_level_version() {
    let shim = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_frcw_tilted"))
            .args(args)
            .output()
            .expect("run frcw_tilted shim")
    };
    let help = shim(&["--help"]);
    assert!(help.status.success());
    let help_stdout = String::from_utf8_lossy(&help.stdout);
    assert!(
        help_stdout.contains("--accept-rule"),
        "--help must show the tilted options:\n{}",
        help_stdout
    );
    assert!(String::from_utf8_lossy(&help.stderr).contains("use `rustrecom tilted` instead"));

    // A bare `--version` stays top-level, matching the old binary's output.
    let version = shim(&["--version"]);
    assert!(version.status.success());
    assert!(String::from_utf8_lossy(&version.stdout)
        .contains(&format!("frcw_tilted {}", env!("CARGO_PKG_VERSION"))));
}

// --- Short-bursts coverage ------------------------------------------------

#[test]
fn short_bursts_writes_records_scores_and_metadata() {
    let scores_path = temp_path("sb_full", "csv");
    let output_path = temp_path("sb_full_out", "jsonl");
    let mut args = short_bursts_args(&scores_path);
    args.extend([
        "--writer".to_string(),
        "canonical".to_string(),
        "--output-file".to_string(),
        output_path.to_str().unwrap().to_string(),
    ]);
    let output = frcw(&args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert!(fs::metadata(&output_path).unwrap().len() > 0);
    let scores = fs::read_to_string(&scores_path).unwrap();
    assert!(scores.lines().count() > 1, "scores CSV should have rows");

    // The sidecar derives from --output-file when it is present.
    let meta_path = metadata_path(&output_path);
    let meta: Value = serde_json::from_str(fs::read_to_string(&meta_path).unwrap().trim()).unwrap();
    assert_eq!(meta["meta"]["type"], "short_bursts");
    assert_eq!(meta["meta"]["burst_length"], 5);

    fs::remove_file(&scores_path).ok();
    fs::remove_file(&output_path).ok();
    fs::remove_file(&meta_path).ok();
}

#[test]
fn score_only_short_bursts_run_still_writes_metadata_sidecar() {
    let scores_path = temp_path("sb_scores_only", "csv");
    let args = short_bursts_args(&scores_path);
    let output = frcw(&args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let meta_path = metadata_path(&scores_path);
    let meta: Value = serde_json::from_str(fs::read_to_string(&meta_path).unwrap().trim()).unwrap();
    assert_eq!(meta["meta"]["type"], "short_bursts");
    assert!(meta["meta"].get("output_file").is_none());

    fs::remove_file(&scores_path).ok();
    fs::remove_file(&meta_path).ok();
}

#[test]
fn colliding_output_and_scores_paths_are_rejected() {
    let path = temp_path("sb_collision", "csv");
    let mut args = short_bursts_args(&path);
    args.extend([
        "--writer".to_string(),
        "canonical".to_string(),
        "--output-file".to_string(),
        path.to_str().unwrap().to_string(),
    ]);
    let output = frcw(&args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("'--output-file' and '--scores-output-file' must be different"),
        "stderr:\n{}",
        stderr
    );
    assert!(!path.exists(), "collision must fail before file creation");
}

// --- Intentional change 1: engine errors exit nonzero ---------------------

#[test]
fn burst_length_zero_engine_error_exits_nonzero() {
    let scores_path = temp_path("sb_burst_zero", "csv");
    let mut args = short_bursts_args(&scores_path);
    let burst_index = args.iter().position(|a| a == "--burst-length").unwrap();
    args[burst_index + 1] = "0".to_string();
    let output = frcw(&args.iter().map(String::as_str).collect::<Vec<_>>());
    assert!(
        !output.status.success(),
        "engine errors must exit nonzero now"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("burst_length must be at least 1"),
        "stderr:\n{}",
        stderr
    );

    fs::remove_file(&scores_path).ok();
    fs::remove_file(metadata_path(&scores_path)).ok();
}

// --- Intentional change 2: tilted rejects reversible pre-side-effects -----

#[test]
fn tilted_rejects_reversible_without_creating_output_files() {
    let scores_path = temp_path("tilted_reversible", "csv");
    let args: Vec<String> = [
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
        "--variant",
        "reversible",
        "--objective",
        OBJECTIVE,
        "--maximize",
        "false",
        "--overwrite-output",
        "--scores-output-file",
    ]
    .iter()
    .map(|arg| arg.to_string())
    .chain([scores_path.to_str().unwrap().to_string()])
    .collect();
    let output = frcw(&args.iter().map(String::as_str).collect::<Vec<_>>());

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Reversible ReCom is not supported by the tilted run optimizer."),
        "stderr:\n{}",
        stderr
    );
    // The rejection must land before any output file is created or truncated.
    assert!(!scores_path.exists(), "scores file must not be created");
    assert!(
        !metadata_path(&scores_path).exists(),
        "metadata sidecar must not be created"
    );
}
