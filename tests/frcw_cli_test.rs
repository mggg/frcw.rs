use std::process::Command;

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
    "district-pairs-rmst",
    "--writer",
    "canonical",
];

fn run_frcw(extra_args: &[&str]) -> Vec<u8> {
    let output = Command::new(env!("CARGO_BIN_EXE_frcw"))
        .args(BASE_ARGS)
        .args(extra_args)
        .output()
        .expect("run frcw");
    assert!(
        output.status.success(),
        "status: {:?}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

#[test]
fn show_progress_does_not_change_stdout() {
    assert_eq!(run_frcw(&[]), run_frcw(&["--show-progress"]));
}
