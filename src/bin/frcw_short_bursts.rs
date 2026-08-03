//! Deprecated shim for the pre-unification `frcw_short_bursts` binary: prints a deprecation
//! warning and forwards to the `short-bursts` subcommand of the `rustrecom` CLI (see `app::main`,
//! which dispatches on argv[0]).

#[path = "rustrecom/app.rs"]
mod app;
#[path = "rustrecom/chain.rs"]
mod chain;
#[path = "rustrecom/common.rs"]
mod common;
#[path = "rustrecom/short_bursts.rs"]
mod short_bursts;
#[path = "rustrecom/tilted.rs"]
mod tilted;

fn main() {
    app::main()
}
