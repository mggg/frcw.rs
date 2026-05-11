//! Tilted-run ReCom optimizer.
//!
//! [`core`] holds the shared engine, generic in an [`AcceptanceRule`] and a
//! [`ScoringBackend`]. [`fixed`] and [`metropolis`] each provide a concrete
//! acceptance rule; [`full_rescore`] and [`incremental`] each provide a
//! concrete scoring backend.

pub mod core;
pub mod fixed;
pub mod full_rescore;
pub mod incremental;
pub mod metropolis;
mod packets;
mod writers;

pub use core::{
    multi_tilted_runs, multi_tilted_runs_with_writer, AcceptanceRule, ScoringBackend,
};
pub use fixed::FixedAcceptance;
pub use full_rescore::FullRescoreBackend;
pub use incremental::IncrementalBackend;
pub use metropolis::MetropolisAcceptance;
