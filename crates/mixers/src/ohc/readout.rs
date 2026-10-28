//! Four streams to one `[B, T, d]` bank (spec §4: `H_early` after stage 5, `H_final` after stage 15).
//!
//! Learned output contraction from the pinned Megatron-LM hybrid stack.
//! Returns the raw sum; the model owns the single subsequent final norm.
//!
//! ```text
//! x'_t = vec(X_t) / sqrt(mean(vec(X_t)²) + ε)         [4d]
//! c_t  = σ(s · x'_t W_c + β_c) + ε                    [4]    (W_c: [4d, 4])
//! H_t  = Σ_s c_ts X_ts                                [d]
//! ```
//!
//! A zero (padded) position reads out as exactly `0`.

use burn::module::Param;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;

use super::STREAMS;

#[derive(Config, Debug)]
pub struct OhcReadoutConfig {
    #[config(default = 512)]
    pub d_model: usize,
    /// `ε` of the contraction gate RMS and gate floor (reference `layernorm_epsilon`).
    #[config(default = 1e-5)]
    pub epsilon: f64,
}

impl OhcReadoutConfig {
    pub fn init(&self, device: &Device) -> OhcReadout {
        let flat = STREAMS * self.d_model;
        // Xavier-uniform limit of the reference `hc_head_fn: [n, n·C]`.
        let limit = (6.0 / (flat + STREAMS) as f64).sqrt();
        OhcReadout {
            d_model: self.d_model,
            epsilon: self.epsilon,
            gate_projection: LinearConfig::new(flat, STREAMS)
                .with_bias(false)
                .with_initializer(Initializer::Uniform {
                    min: -limit,
                    max: limit,
                })
                .init(device),
            gate_bias: Initializer::Zeros.init([STREAMS], device),
            gate_scale: Initializer::Ones.init([1], device),
        }
    }
}

/// Learned weighted contraction of the four raw streams.
#[derive(Module, Debug)]
pub struct OhcReadout {
    pub d_model: usize,
    pub epsilon: f64,
    /// `W_c: [4d, 4]`.
    pub gate_projection: Linear,
    /// `β_c`, `[4]`.
    pub gate_bias: Param<Tensor<1>>,
    /// `s`, `[1]`.
    pub gate_scale: Param<Tensor<1>>,
}

impl OhcReadout {
    /// `[B, T, 4, d] → [B, T, d]`.
    pub fn readout(&self, streams: Tensor<4>) -> Tensor<3> {
        let streams = streams.cast(burn::tensor::DType::F32);
        let [batch, tokens, stream_count, width] = streams.dims();
        assert_eq!(
            stream_count, STREAMS,
            "oHC uses {STREAMS} streams, got {stream_count}"
        );
        assert_eq!(
            width, self.d_model,
            "stream width {width} differs from the configured {}",
            self.d_model
        );

        let flat = streams.clone().reshape([batch, tokens, STREAMS * width]);
        let inverse_rms = (flat.clone().square().mean_dim(2) + self.epsilon)
            .sqrt()
            .recip();
        let weights = sigmoid(
            self.gate_projection.forward(flat)
                * inverse_rms
                * self.gate_scale.val().unsqueeze::<3>()
                + self.gate_bias.val().unsqueeze::<3>(),
        ) + self.epsilon;
        (streams * weights.unsqueeze_dim::<4>(3))
            .sum_dim(2)
            .squeeze_dim::<3>(2)
    }
}
