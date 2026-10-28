//! Orthogonal hyper-connections: 4-stream SO(4) residual read/write wrapper (spec §1.2, §2.1, §4).
//!
//! - [`StreamInit`]: projected token features `[B, T, d]` to four near-identical streams `[B, T, 4, d]`.
//! - [`OhcStage`]: per backbone stage, `read` the streams into the operator input `u` and `write`
//!   `X' = Q X + γ b fᵀ` back, with `Q ∈ SO(4)` built from a unit-quaternion pair ([`QuaternionPair`]).
//! - [`OhcReadout`]: four streams to one `[B, T, d]` bank.
//!
//! Equations, reference versions and every adaptation of the paper: `docs/references/ohc.md`.

mod readout;
mod rotation;
mod stage;
mod streams;
mod relay_streams;

pub use readout::{OhcReadout, OhcReadoutConfig};
pub use rotation::{QUATERNION_EPSILON, QuaternionPair, ROTATION_LOGITS};
pub use stage::{GATE_NORM_EPSILON, OhcRead, OhcStage, OhcStageConfig, READ_WEIGHT_FLOOR};
pub use streams::{StreamInit, StreamInitConfig};
pub use relay_streams::{RelayStreamInit, RelayStreamInitConfig};

/// Number of persistent residual streams (spec §0 `S`); the quaternion-pair mixer is defined for exactly 4.
pub const STREAMS: usize = 4;

#[cfg(test)]
mod tests;
