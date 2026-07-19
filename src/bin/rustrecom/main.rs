//! Unified CLI for rustrecom: `chain`, `short-bursts`, and `tilted`
//! subcommands. Also built under the compatibility name `frcw`
//! (`src/bin/frcw.rs`) so existing scripts keep working.
//!
//! A bare `rustrecom --graph-json ...` invocation (no subcommand) still runs
//! the chain so GerryChain.jl, README examples, and existing scripts keep
//! working; see `app::with_default_subcommand`.

mod app;
mod chain;
mod common;
mod short_bursts;
mod tilted;

fn main() {
    app::main()
}
