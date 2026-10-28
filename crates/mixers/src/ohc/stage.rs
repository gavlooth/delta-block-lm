//! One backbone stage's oHC read and write (spec §2.1).
//!
//! For each position the streams `X_t ∈ R[4, d]` give, through one projection of the RMS-normalized flattened
//! streams (mHC Eq. 4 as used by oHC §5), three input-dependent maps:
//!
//! ```text
//! x'_t   = vec(X_t) / (sqrt(mean(vec(X_t)²)) + 1e-6)                    [4d]
//! p_t    = x'_t W                                                       [16]  (W: [4d, 16])
//! a_t    = σ(α_read · p_t[0..4] + β_read) + 1e-6                         read weights
//! b_t    = 2 σ(α_write · p_t[4..8] + β_write)                            write weights
//! ℓ_t    = α_mix · p_t[8..16] + β_mix,  Q_t = L_q R_r̄ (quaternion pair)  mixer in SO(4)
//! ```
//!
//! and the stage computes
//!
//! ```text
//! u_t      = Σ_s a_ts X_ts                     (raw weighted read, Eq. 1)
//! f_t      = F(u)_t                            (the stage operator, outside this crate)
//! X'_t     = Q_t X_t + b_t f_tᵀ                (write, Eq. 1)
//! ```
//!
//! `α_mix` is fixed at its initial value (oHC §5 "Implementation"); `α_read`, `α_write` and the biases
//! are trainable. Every map is per position, so documents and padding stay separate: a zero position reads
//! `u = 0`, and writes `0` whenever the operator output is `0` there.

use burn::module::Param;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;

use super::rotation::{QuaternionPair, ROTATION_LOGITS};
use super::{OhcBackend, STREAMS};

/// `ε` outside the flattened-stream RMS, matching pinned upstream `native_proj_rms`.
pub const GATE_NORM_EPSILON: f64 = 1e-6;

/// Lower limit added to the read weights, as in the reference's `h_pre = σ(·) + 1e-6`.
pub const READ_WEIGHT_FLOOR: f64 = 1e-6;

/// Gate channels per position: 4 read, 4 write, 8 rotation logits.
const GATE_CHANNELS: usize = 2 * STREAMS + ROTATION_LOGITS;

#[derive(Config, Debug)]
pub struct OhcStageConfig {
    /// Width of each stream and of the stage operator.
    #[config(default = 512)]
    pub d_model: usize,
    /// Initial `α_read`, `α_write` and the fixed `α_mix` (reference `mhc_init_gating_factor`).
    #[config(default = 0.01)]
    pub gate_scale_init: f64,
}

impl OhcStageConfig {
    pub fn init(&self, device: &Device) -> OhcStage {
        let flat = STREAMS * self.d_model;
        // Xavier-uniform limit of the reference projection `[n·C → n² + 2n]` (24 outputs at n = 4); this
        // module keeps the 16 outputs oHC reads (4 read, 4 write, 8 rotation logits).
        let limit = (6.0 / (flat + STREAMS * STREAMS + 2 * STREAMS) as f64).sqrt();
        OhcStage {
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
            rotation: QuaternionPair::new(device),
        }
        .identity_init()
    }
}

/// oHC wrapper of one backbone stage: [`read`](OhcStage::read) the streams into the operator input,
/// [`write`](OhcStage::write) the operator output back.
#[derive(Module, Debug)]
pub struct OhcStage {
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
    /// Starts residual transport at exactly the FP32 identity, without suppressing branch writes.
    /// Rotation columns and bias are zero; read/write gates retain their normal initialization.
    /// The fixed nonzero `mix_scale` lets rotation columns learn from the first update.
    pub fn identity_init(mut self) -> Self {
        self.mix_bias = self
            .mix_bias
            .map(|value| Tensor::zeros(value.dims(), &value.device()))
            .set_require_grad(true);
        self.gate_projection.weight = self
            .gate_projection
            .weight
            .map(|weight| {
                let [rows, columns] = weight.dims();
                let zeros = Tensor::zeros([rows, ROTATION_LOGITS], &weight.device());
                weight
                    .slice_assign([0..rows, 2 * STREAMS..columns], zeros)
                    .detach()
            })
            .set_require_grad(true);
        self
    }

    /// Gates from the streams and the raw operator input `u_t = Σ_s a_ts X_ts`.
    pub fn read(&self, streams: Tensor<4>) -> OhcRead {
        self.read_selected(streams, OhcBackend::Tensor)
    }
    pub fn read_selected(&self, streams: Tensor<4>, backend: OhcBackend) -> OhcRead {
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

        let projected = if matches!(backend, OhcBackend::FusedGate | OhcBackend::TransposedGate) {
            backend.project_gate(streams.clone(), self.gate_projection.weight.val())
        } else {
            let flat = streams.clone().cast(burn::tensor::DType::F32).reshape([
                batch,
                tokens,
                STREAMS * width,
            ]);
            backend.gate_norm(streams.clone(), self.gate_projection.forward(flat))
        };

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
        let mix = self
            .rotation
            .rotation_selected(mix_logits, backend.fused_read());
        let work = backend.weighted_read(streams.clone(), read_weights.clone());
        OhcRead {
            work,
            streams,
            mix,
            read_weights,
            write_weights,
        }
    }

    /// `X' = Q X + b fᵀ` with `f` the operator output `[B, T, d]` for `read.work`.
    pub fn write(
        &self,
        read: OhcRead,
        operator_output: Tensor<3>,
        valid: Tensor<2>,
        backend: OhcBackend,
    ) -> Tensor<4> {
        assert_eq!(
            operator_output.dims(),
            read.work.dims(),
            "operator output must have the shape of its input"
        );
        let [batch, tokens, _, width] = read.streams.dims();
        let streams = backend.write(
            read.streams,
            read.mix,
            read.write_weights,
            operator_output,
        );
        streams.mask_fill(
            valid
                .equal_elem(0.0)
                .reshape([batch, tokens, 1, 1])
                .expand([batch, tokens, STREAMS, width]),
            0.0,
        )
    }

    /// Read, apply `operator` to `u`, write.
    pub fn forward(
        &self,
        streams: Tensor<4>,
        valid: Tensor<2>,
        backend: OhcBackend,
        operator: impl FnOnce(Tensor<3>) -> Tensor<3>,
    ) -> Tensor<4> {
        let read = self.read_selected(streams, backend);
        let output = operator(read.work.clone());
        self.write(read, output, valid, backend)
    }
}
