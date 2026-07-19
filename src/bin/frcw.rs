//! Deprecated compatibility alias for the `rustrecom` CLI: the same entry point built under the
//! pre-rename `frcw` name so gerrytools' mgrp runner and existing scripts keep working.
//! `argv[0]`-based naming in `app` makes this binary report itself as `frcw` in help, usage, and
//! version output, and prints a deprecation warning pointing at `rustrecom`.

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
