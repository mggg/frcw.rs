//! Compatibility alias for the `rustrecom` CLI: the same entry point built
//! under the pre-rename `frcw` name so GerryChain.jl and existing scripts
//! keep working. `argv[0]`-based naming in `app::invoked_name` makes this
//! binary report itself as `frcw` in help, usage, and version output.

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
