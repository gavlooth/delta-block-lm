//! Trainable causal GDN-2 and KronLSTM with optional HOLA exact memory.
//! See `README.md` for recurrence equations, reference provenance and check commands.
//! Training explicitly selects the tensor solver or the opt-in fused Kron GPU scan.
//! The default remains the differentiable tensor solver until paired measurements.
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::{
    Distribution,
    activation::{sigmoid, silu},
};
use serde::{Deserialize, Serialize};
pub mod attribution;
mod cache_reference;
pub mod checks;
pub mod chunk;
pub mod gdn2_reference;
mod hola;
pub mod normalization;
pub mod profile;
mod projections;
pub use normalization::NormalizationBackend;
pub mod reference;
pub use checks::{NumericalReport, numerical_checks};
pub use projections::{
    ProjectionBackend, benchmark as projection_benchmark,
    numerical_checks as projection_numerical_checks,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MixerKind {
    Gdn2,
    KronLstm,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScanBackend {
    #[default]
    TensorSolver,
    FusedGpu,
}
impl ScanBackend {
    pub fn is_tensor_solver(&self) -> bool {
        *self == Self::TensorSolver
    }
    pub fn training_algorithm(&self) -> &'static str {
        match self {
            Self::TensorSolver => "differentiable_chunk_tensor_solver",
            Self::FusedGpu => "fused_kron_gpu",
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HolaBackend {
    #[default]
    Tensor,
    FusedGpu,
    FusedGpuPairs,
    FusedGpuPairsNorms,
}
impl HolaBackend {
    pub fn is_tensor(&self) -> bool {
        *self == Self::Tensor
    }
    pub fn is_fused(&self) -> bool {
        !self.is_tensor()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CachePolicy {
    None,
    Surprise,
    Recency,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MixerConfig {
    pub kind: MixerKind,
    pub cache_policy: CachePolicy,
    #[serde(skip_serializing_if = "ScanBackend::is_tensor_solver")]
    pub scan_backend: ScanBackend,
    #[serde(skip_serializing_if = "HolaBackend::is_tensor")]
    pub hola_backend: HolaBackend,
    #[serde(skip_serializing_if = "ProjectionBackend::is_separate")]
    pub projection_backend: ProjectionBackend,
    #[serde(skip_serializing_if = "NormalizationBackend::is_tensor")]
    pub normalization_backend: NormalizationBackend,
    pub d_model: usize,
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub chunk_size: usize,
    pub cache_capacity: usize,
}
/// Allocated exact-attention and selection work, including masked entries.
#[derive(Clone, Debug, Default, Serialize)]
pub struct AttentionWork {
    pub history_pairs: usize,
    pub current_chunk_pairs: usize,
    pub sink_pairs: usize,
    pub selection_rank_pairs: usize,
    /// Input elements to argsort; not an estimate of its comparison count.
    pub selection_sort_elements: usize,
}
impl Default for MixerConfig {
    fn default() -> Self {
        Self {
            kind: MixerKind::Gdn2,
            cache_policy: CachePolicy::None,
            scan_backend: ScanBackend::TensorSolver,
            hola_backend: HolaBackend::Tensor,
            projection_backend: ProjectionBackend::Separate,
            normalization_backend: NormalizationBackend::Tensor,
            d_model: 128,
            heads: 4,
            key_dim: 32,
            value_dim: 64,
            chunk_size: 16,
            cache_capacity: 16,
        }
    }
}
impl MixerConfig {
    pub fn init(&self, device: &Device) -> Mixer {
        assert!(
            self.d_model > 0
                && self.heads > 0
                && self.key_dim > 0
                && self.value_dim > 0
                && self.chunk_size > 0
        );
        assert!(
            (1..=chunk::MAX_CHUNK_SIZE).contains(&self.chunk_size),
            "computational chunk_size must be 1..=16"
        );
        assert!(self.cache_policy == CachePolicy::None || self.cache_capacity > 0);
        if self.scan_backend == ScanBackend::FusedGpu {
            assert_eq!(
                self.kind,
                MixerKind::KronLstm,
                "fused GPU scan requires KronLSTM"
            );
            assert_eq!(
                (self.key_dim, self.value_dim),
                (32, 64),
                "fused Kron scan requires key32/value64"
            );
            chunk::assert_fused_device(device);
        }
        if self.hola_backend.is_fused() {
            chunk::assert_fused_device(device);
            if self.cache_policy != CachePolicy::None {
                assert_eq!(
                    (self.key_dim, self.value_dim),
                    (32, 64),
                    "fused HOLA requires key32/value64"
                );
                assert!(
                    (1..=64).contains(&self.cache_capacity),
                    "fused HOLA capacity must be 1..=64"
                );
            }
        }
        if !self.normalization_backend.is_tensor() {
            chunk::assert_fused_device(device);
        }
        if self.projection_backend == ProjectionBackend::MergedHeuristicOutput {
            chunk::assert_fused_device(device);
        }
        let kw = self.heads * self.key_dim;
        let vw = self.heads * self.value_dim;
        let linear = |i, o| LinearConfig::new(i, o).with_bias(false).init(device);
        let conv = |channels| {
            Param::from_tensor(Tensor::random(
                [4, channels],
                Distribution::Uniform(-0.1, 0.1),
                device,
            ))
        };
        let scalar = self.kind == MixerKind::KronLstm;
        Mixer {
            q: linear(self.d_model, kw),
            k: linear(self.d_model, kw),
            v: linear(self.d_model, vw),
            decay: linear(self.d_model, kw),
            erase: linear(self.d_model, if scalar { self.heads } else { kw }),
            write: linear(self.d_model, if scalar { self.heads } else { vw }),
            output_gate: linear(self.d_model, vw),
            output: linear(vw, self.d_model),
            q_conv: conv(kw),
            k_conv: conv(kw),
            v_conv: conv(vw),
            decay_scale: Param::from_tensor(Tensor::zeros([self.heads], device)),
            decay_bias: Param::from_tensor(Tensor::full([kw], -4.0, device)),
            norm: RmsNormConfig::new(self.value_dim).init(device),
            cache: (self.cache_policy != CachePolicy::None).then(|| ExactMemory {
                q_norm: RmsNormConfig::new(self.key_dim).init(device),
                k_norm: RmsNormConfig::new(self.key_dim).init(device),
                gate: Param::from_tensor(Tensor::full([self.heads], -4.0, device)),
                sink: Param::from_tensor(Tensor::zeros([self.heads], device)),
            }),
            kind: self.kind,
            cache_policy: self.cache_policy,
            scan_backend: self.scan_backend,
            hola_backend: self.hola_backend,
            projection_backend: self.projection_backend,
            normalization_backend: self.normalization_backend,
            heads: self.heads,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
            chunk_size: self.chunk_size,
            cache_capacity: self.cache_capacity,
        }
    }
    pub fn recurrent_state_bytes(&self, batch: usize) -> usize {
        batch * self.heads * self.key_dim * self.value_dim * 4
    }
    pub fn exact_memory_bytes(&self, batch: usize) -> usize {
        if self.cache_policy == CachePolicy::None {
            0
        } else {
            batch
                * self.heads
                * (self.cache_capacity + self.chunk_size)
                * (self.key_dim + self.value_dim + 3)
                * 4
        }
    }
    pub fn attention_work(&self, batch: usize, tokens: usize) -> AttentionWork {
        let mut work = AttentionWork::default();
        if self.cache_policy == CachePolicy::None {
            return work;
        }
        assert!(self.chunk_size > 0);
        for first in (0..tokens).step_by(self.chunk_size) {
            let n = self.chunk_size.min(tokens - first);
            let rows = batch * self.heads;
            work.history_pairs += rows * n * self.cache_capacity;
            work.current_chunk_pairs += rows * n * n;
            work.sink_pairs += rows * n;
            work.selection_rank_pairs +=
                rows * (self.cache_capacity + n) * (self.cache_capacity + n);
            work.selection_sort_elements += rows * (self.cache_capacity + n);
        }
        work
    }
}

#[derive(Module, Debug)]
pub struct Mixer {
    q: Linear,
    k: Linear,
    v: Linear,
    decay: Linear,
    erase: Linear,
    write: Linear,
    output_gate: Linear,
    output: Linear,
    q_conv: Param<Tensor<2>>,
    k_conv: Param<Tensor<2>>,
    v_conv: Param<Tensor<2>>,
    decay_scale: Param<Tensor<1>>,
    decay_bias: Param<Tensor<1>>,
    norm: RmsNorm,
    cache: Option<ExactMemory>,
    #[module(skip)]
    kind: MixerKind,
    #[module(skip)]
    cache_policy: CachePolicy,
    #[module(skip)]
    scan_backend: ScanBackend,
    #[module(skip)]
    hola_backend: HolaBackend,
    #[module(skip)]
    projection_backend: ProjectionBackend,
    #[module(skip)]
    normalization_backend: NormalizationBackend,
    #[module(skip)]
    heads: usize,
    #[module(skip)]
    key_dim: usize,
    #[module(skip)]
    value_dim: usize,
    #[module(skip)]
    chunk_size: usize,
    #[module(skip)]
    cache_capacity: usize,
}
#[derive(Module, Debug)]
struct ExactMemory {
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    gate: Param<Tensor<1>>,
    sink: Param<Tensor<1>>,
}
impl Mixer {
    pub fn set_normalization_backend(&mut self, backend: NormalizationBackend) {
        if !backend.is_tensor() {
            chunk::assert_fused_device(&self.q.weight.val().device());
        }
        self.normalization_backend = backend;
    }
    /// Changes execution only; trainable tensors and their identities remain unchanged.
    pub fn set_scan_backend(&mut self, backend: ScanBackend) {
        if backend == ScanBackend::FusedGpu {
            assert_eq!(
                self.kind,
                MixerKind::KronLstm,
                "fused GPU scan requires KronLSTM"
            );
            assert_eq!(
                (self.key_dim, self.value_dim),
                (32, 64),
                "fused Kron scan requires key32/value64"
            );
            chunk::assert_fused_device(&self.q.weight.val().device());
        }
        self.scan_backend = backend;
    }
    pub fn set_hola_backend(&mut self, backend: HolaBackend) {
        if backend.is_fused() {
            assert_eq!(
                (self.key_dim, self.value_dim),
                (32, 64),
                "fused HOLA requires key32/value64"
            );
            assert!(
                (1..=64).contains(&self.cache_capacity),
                "fused HOLA capacity must be 1..=64"
            );
            chunk::assert_fused_device(&self.q.weight.val().device());
        }
        self.hola_backend = backend;
    }
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let [b, t, _] = input.dims();
        let device = input.device();
        self.forward_masked(
            input,
            Tensor::ones([b, t], &device),
            Tensor::zeros([b, t], &device),
        )
    }
    pub fn forward_masked(
        &self,
        input: Tensor<3>,
        valid: Tensor<2>,
        reset: Tensor<2>,
    ) -> Tensor<3> {
        self.forward_profiled(input, valid, reset, &mut profile::Unprofiled)
    }
    /// Same differentiable forward, with optional non-overlapping section observation.
    pub fn forward_profiled<P: profile::Sections>(
        &self,
        input: Tensor<3>,
        valid: Tensor<2>,
        reset: Tensor<2>,
        profile: &mut P,
    ) -> Tensor<3> {
        let [b, t, _] = input.dims();
        assert!(t > 0);
        assert_eq!(valid.dims(), [b, t]);
        assert_eq!(reset.dims(), [b, t]);
        let (input, documents) = profile.section("Masks and document IDs", || {
            (
                input * valid.clone().unsqueeze_dim::<3>(2),
                (reset.clone() * valid.clone()).cumsum(1),
            )
        });
        let conv = |x, weight| causal_conv(x, weight, valid.clone(), documents.clone());
        let mut projected = self.project_input(input.clone(), profile);
        let raw_q = projected[0]
            .take()
            .unwrap_or_else(|| profile.section("Projection: q", || self.q.forward(input.clone())));
        let raw_q = profile.section("Short causal convolutions", || {
            conv(raw_q, self.q_conv.val())
        });
        let raw_q = profile.section("L2 normalization and gate activations", || {
            silu(raw_q).reshape([b, t, self.heads, self.key_dim])
        });
        let raw_k = projected[1]
            .take()
            .unwrap_or_else(|| profile.section("Projection: k", || self.k.forward(input.clone())));
        let raw_k = profile.section("Short causal convolutions", || {
            conv(raw_k, self.k_conv.val())
        });
        let raw_k = profile.section("L2 normalization and gate activations", || {
            silu(raw_k).reshape([b, t, self.heads, self.key_dim])
        });
        let raw_v = projected[2]
            .take()
            .unwrap_or_else(|| profile.section("Projection: v", || self.v.forward(input.clone())));
        let raw_v = profile.section("Short causal convolutions", || {
            conv(raw_v, self.v_conv.val())
        });
        let raw_v = profile.section("L2 normalization and gate activations", || {
            silu(raw_v).reshape([b, t, self.heads, self.value_dim])
        });
        let (q, k, v) = profile.section("L2 normalization and gate activations", || {
            (
                self.normalization_backend.l2(raw_q.clone()),
                self.normalization_backend.l2(raw_k.clone()),
                if self.kind == MixerKind::KronLstm {
                    self.normalization_backend.l2(raw_v.clone())
                } else {
                    raw_v.clone()
                },
            )
        });
        let decay = projected[3].take().unwrap_or_else(|| {
            profile.section("Projection: decay", || self.decay.forward(input.clone()))
        });
        let alpha = profile.section("L2 normalization and gate activations", || {
            self.normalization_backend.decay(
                decay.reshape([b, t, self.heads, self.key_dim]),
                self.decay_bias.val(),
                self.decay_scale.val(),
            )
        });
        let erase_width = if self.kind == MixerKind::Gdn2 {
            self.key_dim
        } else {
            1
        };
        let write_width = if self.kind == MixerKind::Gdn2 {
            self.value_dim
        } else {
            1
        };
        let erase = projected[4].take().unwrap_or_else(|| {
            profile.section("Projection: erase", || self.erase.forward(input.clone()))
        });
        let erase = profile.section("L2 normalization and gate activations", || {
            sigmoid(erase).reshape([b, t, self.heads, erase_width])
        });
        let write = projected[5].take().unwrap_or_else(|| {
            profile.section("Projection: write", || self.write.forward(input.clone()))
        });
        let write = profile.section("L2 normalization and gate activations", || {
            sigmoid(write).reshape([b, t, self.heads, write_width])
        });
        let recurrent = profile.section("Recurrent scan", || {
            chunk::scan_selected(
                self.scan_backend,
                self.kind,
                chunk::ScanInput {
                    q,
                    k,
                    v,
                    alpha,
                    erase,
                    write,
                },
                valid.clone(),
                reset,
                self.chunk_size,
                None,
            )
        });
        let mut mixed = recurrent.output;
        if let Some(cache) = &self.cache {
            let (cache_q, cache_k) = profile.section("HOLA RMSNorms", || {
                (
                    hola::normalize(
                        self.hola_backend,
                        self.normalization_backend,
                        &cache.q_norm,
                        raw_q,
                    ),
                    hola::normalize(
                        self.hola_backend,
                        self.normalization_backend,
                        &cache.k_norm,
                        raw_k,
                    ),
                )
            });
            let exact = profile.section("HOLA read", || {
                hola::read_selected(
                    self.hola_backend,
                    cache_q,
                    cache_k,
                    raw_v,
                    recurrent.scores,
                    valid.clone(),
                    documents,
                    self.cache_capacity,
                    self.chunk_size,
                    self.cache_policy,
                    cache.sink.val(),
                )
            });
            mixed = profile.section("L2 normalization and gate activations", || {
                mixed + exact * sigmoid(cache.gate.val()).reshape([1, 1, self.heads, 1])
            });
        }
        let gate = projected[6].take().unwrap_or_else(|| {
            profile.section("Projection: output_gate", || {
                self.output_gate.forward(input)
            })
        });
        let mixed = if self.normalization_backend == NormalizationBackend::FusedNormsGates {
            profile.section("Mixer output RMSNorm", || {
                self.normalization_backend
                    .gated_rms(
                        &self.norm,
                        mixed,
                        gate.reshape([b, t, self.heads, self.value_dim]),
                    )
                    .reshape([b, t, self.heads * self.value_dim])
            })
        } else {
            let mixed = profile.section("L2 normalization and gate activations", || {
                mixed * sigmoid(gate).reshape([b, t, self.heads, self.value_dim])
            });
            profile.section("Mixer output RMSNorm", || {
                self.normalization_backend.rms(&self.norm, mixed).reshape([
                    b,
                    t,
                    self.heads * self.value_dim,
                ])
            })
        };
        let output = profile.section("Projection: output", || self.project_output(mixed));
        profile.section("Masks and document IDs", || {
            output * valid.unsqueeze_dim::<3>(2)
        })
    }
}

/// Causal short convolution; a lag never crosses a reset or an invalid source.
fn causal_conv(
    x: Tensor<3>,
    weight: Tensor<2>,
    valid: Tensor<2>,
    documents: Tensor<2>,
) -> Tensor<3> {
    let [b, t, c] = x.dims();
    let width = weight.dims()[0];
    let mut output = Tensor::zeros([b, t, c], &x.device());
    for lag in 0..width.min(t) {
        let n = t - lag;
        let source = x.clone().slice([0..b, 0..n, 0..c]);
        let same = documents
            .clone()
            .slice([0..b, lag..t])
            .equal(documents.clone().slice([0..b, 0..n]))
            .float();
        let active = same * valid.clone().slice([0..b, 0..n]) * valid.clone().slice([0..b, lag..t]);
        let contribution = source
            * weight
                .clone()
                .slice([lag..lag + 1, 0..c])
                .reshape([1, 1, c])
            * active.unsqueeze_dim::<3>(2);
        let old = output.clone().slice([0..b, lag..t, 0..c]);
        output = output.slice_assign([0..b, lag..t, 0..c], old + contribution);
    }
    output
}
