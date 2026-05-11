//! Fixed-probability acceptance rule for tilted runs.
//!
//! `FixedAcceptance` accepts a non-improving proposal with a constant
//! probability `prob`, independent of how much worse the proposal is. The
//! engine that consumes this rule lives in [`super::core`].

use rand::rngs::SmallRng;
use rand::Rng;

use super::core::AcceptanceRule;

/// Accepts a worse-or-equal proposal with a fixed probability.
///
/// Use `FixedAcceptance { prob: 0.0 }` for pure hill-climbing and
/// `FixedAcceptance { prob: 1.0 }` for an unconditional accept (random walk).
/// Values must be in `[0.0, 1.0]`; the CLI validates this at parse time.
#[derive(Clone, Copy, Debug)]
pub struct FixedAcceptance {
    pub prob: f64,
}

impl AcceptanceRule for FixedAcceptance {
    fn accept_worse(
        &self,
        _current: f64,
        _proposed: f64,
        _maximize: bool,
        rng: &mut SmallRng,
    ) -> bool {
        rng.random::<f64>() < self.prob
    }
}
