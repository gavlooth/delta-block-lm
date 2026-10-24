//! One backbone stage's oHC read and write (spec §2.1).
//!
//! For each position the streams `X_t ∈ R[4, d]` give, through one projection of the RMS-normalized flattened
//! streams (mHC Eq. 4 as used by oHC §5), three input-dependent maps:
//!
//! ```text
//! x'_t   = vec(X_t) / sqrt(mean(vec(X_t)²) + 1e-6)                      [4d]
//! p_t    = x'_t W                                                       [16]  (W: [4d, 16])
//! a_t    = σ(α_read · p_t[0..4] + β_read) + 1e-6                         read weights
//! b_t    = 2 σ(α_write · p_t[4..8] + β_write)                            write weights
//! ℓ_t    = α_mix · p_t[8..16] + β_mix,  Q_t = L_q R_r̄ (quaternion pair)  mixer in SO(4)
//! ```
//!
//! and the stage computes
//!
//! ```text
//! u_t      = Σ_s a_ts RMSNorm(X_ts)            (read, spec §2.1)
//! f_t      = F(u)_t                            (the stage operator, outside this crate)
//! X'_t     = Q_t X_t + γ b_t f_tᵀ              (write, spec §2.1)
//! ```
//!
//! `α_mix` is fixed at its initial value (oHC §5 "Implementation"); `α_read`, `α_write`, `γ` and the biases
//! are trainable. Every map is per position, so documents and padding stay separate: a zero position reads
//! `u = 0`, and writes `0` whenever the operator output is `0` there.

use burn::module::Param;
use burn::nn::{Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;

use crate::STREAMS;
use crate::rotation::{QuaternionPair, ROTATION_LOGITS};

/// `ε` inside the RMS of the flattened streams that feed the gate projection (value of the reference's
/// `norm_eps`; placed under the square root so the gradient stays finite at zero (padded) positions).
pub const GATE_NORM_EPSILON: f64 = 1e-6;

/// Lower limit added to the read weights, as in the reference's `h_pre = σ(·) + 1e-6`.
pub const READ_WEIGHT_FLOOR: f64 = 1e-6;

/// Gate channels per position: 4 read, 4 write, 8 rotation logits.
const GATE_CHANNELS: usize = 2 * STREAMS + ROTATION_LOGITS;

#[derive(Config, Debug)]
pub struct OhcStageConfig {
    #[config(default = "dre_core::Precision::Fp32")]
    pub precision: dre_core::Precision,
    /// Width of each stream and of the stage operator.
    #[config(default = 512)]
    pub d_model: usize,
    /// Initial `α_read`, `α_write` and the fixed `α_mix` (reference `mhc_init_gating_factor`).
    #[config(default = 0.01)]
    pub gate_scale_init: f64,
    /// Initial residual-write scale `γ` (spec §2.1: conservative; LayerScale value for depth ≤ 18).
    #[config(default = 0.1)]
    pub write_scale_init: f64,
    /// `ε` of the per-stream RMSNorm in the read.
    #[config(default = 1e-5)]
    pub norm_epsilon: f64,
}

impl OhcStageConfig {
    pub fn init(&self, device: &Device) -> OhcStage {
        let flat = STREAMS * self.d_model;
        // Xavier-uniform limit of the reference projection `[n·C → n² + 2n]` (24 outputs at n = 4); this
        // module keeps the 16 outputs oHC reads (4 read, 4 write, 8 rotation logits).
        let limit = (6.0 / (flat + STREAMS * STREAMS + 2 * STREAMS) as f64).sqrt();
        OhcStage {
            mixed: self.precision == dre_core::Precision::Bf16Mixed,
            d_model: self.d_model,
            gate_projection: LinearConfig::new(flat, GATE_CHANNELS)
                .with_bias(false)
                .with_initializer(Initializer::Uniform {
                    min: -limit,
                    max: limit,
                })
                .init(device),
            read_bias: Initializer::Zeros.init([STREAMS], device),
            write_bias: Initializer::Zeros.init([STREAMS], device),
            mix_bias: Initializer::Zeros.init([ROTATION_LOGITS], device),
            read_scale: Initializer::Constant {
                value: self.gate_scale_init,
            }
            .init([1], device),
            write_gate_scale: Initializer::Constant {
                value: self.gate_scale_init,
            }
            .init([1], device),
            mix_scale: self.gate_scale_init,
            write_scale: Initializer::Constant {
                value: self.write_scale_init,
            }
            .init([1], device),
            stream_norm: RmsNormConfig::new(self.d_model)
                .with_epsilon(self.norm_epsilon)
                .init(device),
            rotation: QuaternionPair::new(device),
        }
    }
}

/// oHC wrapper of one backbone stage: [`read`](OhcStage::read) the streams into the operator input,
/// [`write`](OhcStage::write) the operator output back.
#[derive(Module, Debug)]
pub struct OhcStage {
    mixed: bool,
    pub d_model: usize,
    /// `W: [4d, 16]`, shared source of read, write and rotation gates.
    pub gate_projection: Linear,
    pub read_bias: Param<Tensor<1>>,
    pub write_bias: Param<Tensor<1>>,
    pub mix_bias: Param<Tensor<1>>,
    /// `α_read`, `[1]`.
    pub read_scale: Param<Tensor<1>>,
    /// `α_write`, `[1]`.
    pub write_gate_scale: Param<Tensor<1>>,
    /// `α_mix`, fixed (oHC §5: frozen at its initial value for orthogonal mixers).
    pub mix_scale: f64,
    /// Residual-write scale `γ`, `[1]`.
    pub write_scale: Param<Tensor<1>>,
    /// RMSNorm applied to each stream before the weighted read.
    pub stream_norm: RmsNorm,
    pub rotation: QuaternionPair,
}

/// Result of [`OhcStage::read`]: the operator input and what [`OhcStage::write`] needs.
#[derive(Debug, Clone)]
pub struct OhcRead {
    /// `u`, the operator input, `[B, T, d]`.
    pub work: Tensor<3>,
    /// The streams the read was computed from, `[B, T, 4, d]`.
    pub streams: Tensor<4>,
    /// `Q`, `[B, T, 4, 4]`, each in `SO(4)`.
    pub mix: Tensor<4>,
    /// `a`, `[B, T, 4]`.
    pub read_weights: Tensor<3>,
    /// `b`, `[B, T, 4]`.
    pub write_weights: Tensor<3>,
}

impl OhcStage {
    /// Initializes a newly inserted wrapper as an FP32 identity without disabling learning.
    ///
    /// Only the residual-write scale and mixer channels are zeroed. Read/write gates and
    /// stream normalization keep their normal initialization, and `mix_scale` stays nonzero.
    /// The operator must still run: its output supplies the first write-scale gradient.
    pub fn identity_init(mut self) -> Self {
        self.write_scale = self.write_scale.map(|value| Tensor::zeros(value.dims(), &value.device())).set_require_grad(true);
        self.mix_bias = self.mix_bias.map(|value| Tensor::zeros(value.dims(), &value.device())).set_require_grad(true);
        self.gate_projection.weight = self.gate_projection.weight.map(|weight| {
            let [rows, columns] = weight.dims();
            let zeros = Tensor::zeros([rows, ROTATION_LOGITS], &weight.device());
            weight.slice_assign([0..rows, 2 * STREAMS..columns], zeros).detach()
        }).set_require_grad(true);
        self
    }

    /// Gates from the streams and the operator input `u_t = Σ_s a_ts RMSNorm(X_ts)`.
    pub fn read(&self, streams: Tensor<4>) -> OhcRead {
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

        let flat = streams.clone().cast(burn::tensor::DType::F32).reshape([batch, tokens, STREAMS * width]);
        let inverse_rms = (flat.clone().square().mean_dim(2) + GATE_NORM_EPSILON)
            .sqrt()
            .recip();
        let projected = dre_core::project(&self.gate_projection, flat, self.mixed) * inverse_rms;

        let read_weights = sigmoid(
            projected.clone().narrow(2, 0, STREAMS) * self.read_scale.val().unsqueeze::<3>()
                + self.read_bias.val().unsqueeze::<3>(),
        ) + READ_WEIGHT_FLOOR;
        let write_weights = sigmoid(
            projected.clone().narrow(2, STREAMS, STREAMS)
                * self.write_gate_scale.val().unsqueeze::<3>()
                + self.write_bias.val().unsqueeze::<3>(),
        ) * 2.0;
        let mix_logits = projected.narrow(2, 2 * STREAMS, ROTATION_LOGITS) * self.mix_scale
            + self.mix_bias.val().unsqueeze::<3>();
        let mix = self.rotation.rotation(mix_logits);

        let work = (self.stream_norm.forward(streams.clone().cast(burn::tensor::DType::F32))
            * read_weights.clone().unsqueeze_dim::<4>(3))
        .sum_dim(2)
        .squeeze_dim::<3>(2);
        OhcRead {
            work,
            streams,
            mix,
            read_weights,
            write_weights,
        }
    }

    /// `X' = Q X + γ b fᵀ` with `f` the operator output `[B, T, d]` for `read.work`.
    pub fn write(&self, read: OhcRead, operator_output: Tensor<3>) -> Tensor<4> {
        assert_eq!(
            operator_output.dims(),
            read.work.dims(),
            "operator output must have the shape of its input"
        );
        let mixed = read.mix.matmul(read.streams.cast(burn::tensor::DType::F32));
        let write =
            read.write_weights.unsqueeze_dim::<4>(3) * self.write_scale.val().unsqueeze::<4>();
        let streams = mixed + write * operator_output.unsqueeze_dim::<4>(2);
        if self.mixed { streams.cast(burn::tensor::DType::BF16) } else { streams }
    }

    /// Read, apply `operator` to `u`, write.
    pub fn forward(
        &self,
        streams: Tensor<4>,
        operator: impl FnOnce(Tensor<3>) -> Tensor<3>,
    ) -> Tensor<4> {
        let read = self.read(streams);
        let output = operator(read.work.clone());
        self.write(read, output)
    }
}
