//! Linear acceptance rule for tilted runs.
//!
//! `LinearAcceptance` accepts a non-improving proposal with probability
//! `1 - beta * loss`, clamped at zero once the scaled loss is at least `1.0`.

use rand::rngs::SmallRng;
use rand::Rng;

use super::core::AcceptanceRule;

/// Accepts a worse-or-equal proposal with probability `max(0, 1 - beta * loss)`.
#[derive(Clone, Copy, Debug)]
pub struct LinearAcceptance {
    pub beta: f64,
}

impl AcceptanceRule for LinearAcceptance {
    fn accept_worse(
        &self,
        current: f64,
        proposed: f64,
        maximize: bool,
        rng: &mut SmallRng,
    ) -> bool {
        let loss = if maximize {
            current - proposed
        } else {
            proposed - current
        };
        rng.random::<f64>() < (1.0 - self.beta * loss).max(0.0)
    }
}
