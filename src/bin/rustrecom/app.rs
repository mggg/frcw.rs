//! Shared entry point for the `rustrecom` binary and its `frcw` compatibility
//! alias (`src/bin/frcw.rs`). [`invoked_name`] makes each spelling report
//! itself in help, usage, and version output.

use crate::{chain, short_bursts, tilted};
use clap::Command;
use mimalloc::MiMalloc;
use std::ffi::OsString;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// The file stem of argv[0], so the `frcw` alias shows `frcw` in usage and
/// version text while the primary binary shows `rustrecom`.
fn invoked_name() -> String {
    std::env::args_os()
        .next()
        .map(std::path::PathBuf::from)
        .and_then(|path| {
            path.file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "rustrecom".to_string())
}

/// Inserts `chain` after the program name when no subcommand was given, so the
/// pre-unification bare chain invocation keeps working.
///
/// The first token selects the shim only when it starts with `-` (and is not a
/// help/version flag) or is absent entirely. Testing the leading `-` rather
/// than an allowlist of subcommand names means a future subcommand needs no
/// shim change; a bare `help` token or an unknown subcommand name passes
/// through to clap untouched.
fn with_default_subcommand(mut args: Vec<OsString>) -> Vec<OsString> {
    let insert_chain = match args.get(1) {
        None => true,
        Some(first) => {
            let is_help_or_version =
                first == "-h" || first == "--help" || first == "-V" || first == "--version";
            !is_help_or_version && first.to_string_lossy().starts_with('-')
        }
    };
    if insert_chain {
        args.insert(1, OsString::from("chain"));
    }
    args
}

fn cli() -> Command {
    Command::new(invoked_name())
        .version(env!("CARGO_PKG_VERSION"))
        .author("Data and Democracy Lab <https://data-democracy.org>")
        .propagate_version(true)
        .subcommand_required(true)
        .subcommand(chain::command())
        .subcommand(short_bursts::command())
        .subcommand(tilted::command())
}

pub fn main() {
    let args = with_default_subcommand(std::env::args_os().collect());
    let matches = cli().get_matches_from(args);
    // Engine and runtime errors surface here and exit nonzero; parameter
    // validation failures inside each command still panic with their original
    // message payloads.
    let result = match matches.subcommand() {
        Some(("chain", sub)) => {
            chain::run(sub).map_err(|e| format!("Error during chain execution: {}", e))
        }
        Some(("short-bursts", sub)) => {
            short_bursts::run(sub).map_err(|e| format!("Error during optimization: {}", e))
        }
        Some(("tilted", sub)) => {
            tilted::run(sub).map_err(|e| format!("Error during optimization: {}", e))
        }
        _ => unreachable!("subcommand_required guarantees a known subcommand"),
    };
    if let Err(message) = result {
        eprintln!("{}", message);
        std::process::exit(1);
    }
}
