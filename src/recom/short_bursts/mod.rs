//! ReCom-based short-bursts optimizer (Cannon et al. 2020, arXiv: 2011.02288).
//!
//! Short bursts is a greedy hill-climbing heuristic over an arbitrary
//! partition-level objective. [`core`] holds the shared engine; `packets` and
//! `writers` are private modules implementing the worker/writer wire format
//! and writer-thread loops.

pub mod core;
mod packets;
mod writers;

pub use core::{multi_short_bursts_with_writer, ScoreValue};
