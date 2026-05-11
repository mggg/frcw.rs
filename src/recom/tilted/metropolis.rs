//! Metropolis acceptance rule for tilted runs.
//!
//! `MetropolisAcceptance` accepts a non-improving proposal with probability
//! `exp(beta * delta)`, where `delta` is signed by the optimization direction
//! so that worse plans always produce `delta <= 0` (and therefore an
//! acceptance probability in `(0, 1]`). Larger `beta` makes the chain pickier
//! about how much worse it is willing to tolerate; `beta = 0` reduces to a
//! random walk and `beta -> infinity` reduces to hill-climbing. The engine
//! that consumes this rule lives in [`super::core`].

use rand::rngs::SmallRng;
use rand::Rng;

use super::core::AcceptanceRule;

/// Accepts a worse-or-equal proposal with probability `exp(beta * delta)`.
///
/// `delta = proposed - current` when `maximize`, otherwise `delta = current -
/// proposed`. For non-improving proposals `delta <= 0`, so the acceptance
/// probability lies in `(0, 1]` and decays as the proposal gets worse.
/// `beta` must be non-negative; the CLI validates this at parse time.
#[derive(Clone, Copy, Debug)]
pub struct MetropolisAcceptance {
    pub beta: f64,
}

impl AcceptanceRule for MetropolisAcceptance {
    fn accept_worse(
        &self,
        current: f64,
        proposed: f64,
        maximize: bool,
        rng: &mut SmallRng,
    ) -> bool {
        let delta = if maximize {
            proposed - current
        } else {
            current - proposed
        };
        rng.random::<f64>() < (self.beta * delta).exp()
    }
}
