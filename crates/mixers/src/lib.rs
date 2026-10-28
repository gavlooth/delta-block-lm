//! Trainable causal KronLSTM with optional HOLA exact memory.
//! See `README.md` for recurrence equations, reference provenance and check commands.
//! Training explicitly selects the tensor solver or an opt-in fused GPU scan.
//! The default remains the differentiable tensor solver until paired measurements.
use burn::module::{Initializer, Module, Param};
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
pub mod convolution;
pub mod gate_bias_checks;
pub use convolution::ConvolutionBackend;
mod hola;
pub mod prefix_retrieval;
pub mod prefix_checks;
pub mod lngram;
pub mod normalization;
pub mod ohc;
pub mod profile;
mod projections;
pub use normalization::NormalizationBackend;
pub mod reference;
pub mod wavelet;
pub mod sbd;
pub use checks::{NumericalReport, numerical_checks};
pub use projections::{
    ProjectionBackend, benchmark as projection_benchmark,
    numerical_checks as projection_numerical_checks,
};
pub use wavelet::{WaveletBackend, WaveletDyn};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MixerKind {
    KronLstm,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScanBackend {
    #[default]
    TensorSolver,
    FusedGpu,
    FusedParallel128,
    FusedParallel256,
}
impl ScanBackend {
    pub fn is_tensor_solver(&self) -> bool {
        *self == Self::TensorSolver
    }
    pub fn is_fused(&self) -> bool {
        !self.is_tensor_solver()
    }
    pub fn partitions(&self) -> usize {
        match self {
            Self::FusedParallel128 => 2,
            Self::FusedParallel256 => 4,
            _ => 1,
        }
    }
    pub fn training_algorithm(&self) -> &'static str {
        match self {
            Self::TensorSolver => "differentiable_chunk_tensor_solver",
            Self::FusedGpu => "fused_kron_gpu",
            Self::FusedParallel128 => "fused_kron_parallel128_gpu",
            Self::FusedParallel256 => "fused_kron_parallel256_gpu",
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
    #[serde(skip_serializing_if = "ConvolutionBackend::is_tensor")]
    pub convolution_backend: ConvolutionBackend,
    pub d_model: usize,
    /// Total model depth, used only for residual-output initialization.
    pub num_layers: usize,
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
            kind: MixerKind::KronLstm,
            cache_policy: CachePolicy::None,
            scan_backend: ScanBackend::TensorSolver,
            hola_backend: HolaBackend::Tensor,
            projection_backend: ProjectionBackend::Separate,
            normalization_backend: NormalizationBackend::Tensor,
            convolution_backend: ConvolutionBackend::Tensor,
            d_model: 128,
            num_layers: 1,
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
                && self.num_layers > 0
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
        if self.scan_backend.is_fused() {
            assert_eq!(
                (self.key_dim, self.value_dim),
                (32, 64),
                "fused scan requires key32/value64"
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
        if self.convolution_backend == ConvolutionBackend::FusedGpu {
            chunk::assert_fused_device(device);
        }
        if self.projection_backend == ProjectionBackend::MergedHeuristicOutput {
            chunk::assert_fused_device(device);
        }
        let kw = self.heads * self.key_dim;
        let vw = self.heads * self.value_dim;
        // Pinned official model initialization: Normal(.02), with a smaller
        // projection into the residual stream. Paper gate equations still keep
        // their beta/c biases even though the release omits those offsets.
        let normal = Initializer::Normal { mean: 0.0, std: 0.02 };
        let linear = |i, o| LinearConfig::new(i, o).with_bias(false)
            .with_initializer(normal.clone()).init(device);
        // Eq. 11 and the output-gate definition require learned offsets.
        // The paper gives no bias initializer; zeros preserve neutral initial gates.
        let gate = |i, o| {
            let mut layer = linear(i, o);
            layer.bias = Some(Initializer::Zeros.init([o], device));
            layer
        };
        let conv = |channels| normal.init([4, channels], device);
        let output_gate = Linear {
            weight: Initializer::Zeros.init([self.d_model, vw], device),
            bias: Some(Initializer::Constant { value: 3.0 }.init([vw], device)),
        };
        let output = LinearConfig::new(vw, self.d_model).with_bias(false)
            .with_initializer(Initializer::Normal {
                mean: 0.0, std: 0.02 / (2.0 * self.num_layers as f64).sqrt(),
            }).init(device);
        // Official A_log ~ log U(0,16); dt ~ log U(.001,.1), then inverse softplus.
        let decay_scale = Tensor::<1>::random([self.heads], Distribution::Uniform(0.0, 16.0), device).log();
        let dt = (Tensor::<1>::random([kw], Distribution::Uniform(0.0, 1.0), device)
            * (0.1f64.ln() - 0.001f64.ln()) + 0.001f64.ln()).exp().clamp_min(1e-4);
        // dt is at least .001 here, so this equivalent inverse is well resolved in f32.
        let decay_bias = dt.clone() + (dt.neg().exp().neg() + 1.0).log();
        Mixer {
            q: linear(self.d_model, kw),
            k: linear(self.d_model, kw),
            v: linear(self.d_model, vw),
            decay: linear(self.d_model, kw),
            erase: gate(self.d_model, self.heads),
            write: gate(self.d_model, self.heads),
            output_gate,
            output,
            q_conv: conv(kw),
            k_conv: conv(kw),
            v_conv: conv(vw),
            decay_scale: Param::from_tensor(decay_scale),
            decay_bias: Param::from_tensor(decay_bias),
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
            prefix_retrieval: None,
            projection_backend: self.projection_backend,
            normalization_backend: self.normalization_backend,
            convolution_backend: self.convolution_backend,
            in_block_reads: sbd::InBlockReads::None,
            noisy_read_backend: sbd::NoisyReadBackend::Tensor,
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
    prefix_retrieval: Option<prefix_retrieval::PrefixReadConfig>,
    #[module(skip)]
    projection_backend: ProjectionBackend,
    #[module(skip)]
    normalization_backend: NormalizationBackend,
    #[module(skip)]
    convolution_backend: ConvolutionBackend,
    #[module(skip)]
    in_block_reads: sbd::InBlockReads,
    #[module(skip)]
    noisy_read_backend: sbd::NoisyReadBackend,
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
    pub fn with_sbd_reads(mut self, reads: sbd::InBlockReads, backend: sbd::NoisyReadBackend) -> Self {
        assert!(reads.is_none() || self.cache.is_some(), "sbd.in_block_reads=hola_read requires HOLA in every layer");
        self.in_block_reads = reads;
        self.noisy_read_backend = backend;
        self
    }
    pub fn set_noisy_read_backend(&mut self, backend: sbd::NoisyReadBackend) {
        self.noisy_read_backend = backend;
    }
    pub fn set_convolution_backend(&mut self, backend: ConvolutionBackend) {
        if backend == ConvolutionBackend::FusedGpu {
            chunk::assert_fused_device(&self.q.weight.val().device());
        }
        self.convolution_backend = backend;
    }
    pub fn set_normalization_backend(&mut self, backend: NormalizationBackend) {
        if !backend.is_tensor() {
            chunk::assert_fused_device(&self.q.weight.val().device());
        }
        self.normalization_backend = backend;
    }
    /// Changes execution only; trainable tensors and their identities remain unchanged.
    pub fn set_scan_backend(&mut self, backend: ScanBackend) {
        if backend.is_fused() {
            assert_eq!(
                (self.key_dim, self.value_dim),
                (32, 64),
                "fused scan requires key32/value64"
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
    pub fn set_prefix_retrieval(&mut self, config: prefix_retrieval::PrefixReadConfig) {
        assert!(self.cache.is_some(), "prefix retrieval requires HOLA");
        assert!(config.block_size > 0 && config.block_size % self.chunk_size == 0);
        assert!(config.blocks_per_query > 0);
        if !config.backend.is_tensor() {
            assert_eq!((self.key_dim,self.value_dim),(32,64));
        }
        self.prefix_retrieval = Some(config);
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
        self.forward_inner(input, valid, reset, profile, None)
    }
    /// Clean prefix plus one temporary block. Only the prefix advances recurrent memory.
    pub fn forward_sbd(&self, input: Tensor<3>, valid: Tensor<2>, reset: Tensor<2>, start: usize) -> Tensor<3> {
        self.forward_inner(input, valid, reset, &mut profile::Unprofiled, Some(start))
    }
    fn forward_inner<P: profile::Sections>(
        &self, input: Tensor<3>, valid: Tensor<2>, reset: Tensor<2>, profile: &mut P,
        block_start: Option<usize>,
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
        let conv = |x, weight| {
            self.convolution_backend
                .forward(x, weight, valid.clone(), documents.clone())
        };
        let mut projected = self.project_input(input.clone(), profile);
        let raw_q = projected[0]
            .take()
            .unwrap_or_else(|| profile.section("Projection: q", || self.q.forward(input.clone())));
        let clean_raw_q = P::CAPTURE_CLEAN_MIXERS.then(||raw_q.clone());
        let raw_q = profile.section("Short causal convolutions", || {
            conv(raw_q, self.q_conv.val())
        });
        let raw_q = profile.section("L2 normalization and gate activations", || {
            silu(raw_q).reshape([b, t, self.heads, self.key_dim])
        });
        let raw_k = projected[1]
            .take()
            .unwrap_or_else(|| profile.section("Projection: k", || self.k.forward(input.clone())));
        let clean_raw_k = P::CAPTURE_CLEAN_MIXERS.then(||raw_k.clone());
        let raw_k = profile.section("Short causal convolutions", || {
            conv(raw_k, self.k_conv.val())
        });
        let raw_k = profile.section("L2 normalization and gate activations", || {
            silu(raw_k).reshape([b, t, self.heads, self.key_dim])
        });
        let raw_v = projected[2]
            .take()
            .unwrap_or_else(|| profile.section("Projection: v", || self.v.forward(input.clone())));
        let clean_raw_v = P::CAPTURE_CLEAN_MIXERS.then(||raw_v.clone());
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
                self.normalization_backend.l2(raw_v.clone()),
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
        let erase = projected[4].take().unwrap_or_else(|| {
            profile.section("Projection: erase", || self.erase.forward(input.clone()))
        });
        let erase = profile.section("L2 normalization and gate activations", || {
            sigmoid(erase).reshape([b, t, self.heads, 1])
        });
        let write = projected[5].take().unwrap_or_else(|| {
            profile.section("Projection: write", || self.write.forward(input.clone()))
        });
        let write = profile.section("L2 normalization and gate activations", || {
            sigmoid(write).reshape([b, t, self.heads, 1])
        });
        let clean_scan = P::CAPTURE_CLEAN_MIXERS.then(||chunk::ScanInput {
            q:q.clone(),k:k.clone(),v:v.clone(),alpha:alpha.clone(),erase:erase.clone(),write:write.clone(),
        });
        let clean_starts = if P::CAPTURE_CLEAN_MIXERS {profile.clean_starts().unwrap_or(&[]).to_vec()} else {Vec::new()};
        let mut clean_states = Vec::new();
        let recurrent = profile.section("Recurrent scan", || {
            let inputs = chunk::ScanInput { q, k, v, alpha, erase, write };
            if let Some(start) = block_start {
                assert!(start > 0 && start < t);
                let take = |x: Tensor<4>, first: usize, last: usize| {
                    let width = x.dims()[3];
                    x.slice([0..b, first..last, 0..self.heads, 0..width])
                };
                let prefix = chunk::scan_selected(self.scan_backend, self.kind, chunk::ScanInput {
                    q: take(inputs.q.clone(), 0, start), k: take(inputs.k.clone(), 0, start),
                    v: take(inputs.v.clone(), 0, start), alpha: take(inputs.alpha.clone(), 0, start),
                    erase: take(inputs.erase.clone(), 0, start), write: take(inputs.write.clone(), 0, start),
                }, valid.clone().slice([0..b, 0..start]), reset.clone().slice([0..b, 0..start]), self.chunk_size, None);
                let n = t - start;
                let single = |x: Tensor<4>| {
                    let width = x.dims()[3];
                    take(x, start, t).reshape([b*n, 1, self.heads, width])
                };
                let initial = prefix.final_state.clone().unsqueeze_dim::<5>(1)
                    .expand([b, n, self.heads, self.key_dim, self.value_dim])
                    .reshape([b*n, self.heads, self.key_dim, self.value_dim]);
                let active = chunk::scan_selected(self.scan_backend, self.kind, chunk::ScanInput {
                    q: single(inputs.q), k: single(inputs.k), v: single(inputs.v),
                    alpha: single(inputs.alpha), erase: single(inputs.erase), write: single(inputs.write),
                }, valid.clone().slice([0..b, start..t]).reshape([b*n, 1]),
                reset.clone().slice([0..b, start..t]).reshape([b*n, 1]), self.chunk_size, Some(initial));
                chunk::ScanOutput {
                    output: Tensor::cat(vec![prefix.output, active.output.reshape([b,n,self.heads,self.value_dim])], 1),
                    scores: Tensor::cat(vec![prefix.scores, Tensor::zeros([b,n,self.heads,1], &input.device())], 1),
                    final_state: prefix.final_state,
                }
            } else if !clean_starts.is_empty() {
                let result=self.scan_sbd_snapshots::<true>(inputs,valid.clone(),reset.clone(),&clean_starts);
                clean_states=result.states;
                chunk::ScanOutput {output:result.output.unwrap(),scores:result.scores,final_state:result.final_state}
            } else {
                chunk::scan_selected(self.scan_backend, self.kind, inputs, valid.clone(), reset.clone(), self.chunk_size, None)
            }
        });
        if P::CAPTURE_RECURRENT_STATE {
            profile.recurrent_state(&recurrent.final_state);
        }
        if P::CAPTURE_CLEAN_MIXERS {
            assert!(block_start.is_none(),"clean mixer capture requires a clean pass");
            profile.clean_mixer(sbd::CleanMixer {
                raw:[clean_raw_q.unwrap(),clean_raw_k.unwrap(),clean_raw_v.unwrap()],
                keys:raw_k.clone(),values:raw_v.clone(),scan:clean_scan.unwrap(),
                scores:recurrent.scores.clone(),valid:valid.clone(),reset:reset.clone(),
                starts:clean_starts,states:clean_states,
            });
        }
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
            let exact = if let Some(config) = self.prefix_retrieval {
                if P::CAPTURE_CACHE_MEMBERSHIP && block_start.is_none() {
                    let replay = profile.replayed_cache_selection();
                    let prefix_replay = profile.replayed_prefix_visibility();
                    let (read,selections) = hola::read_prefix_with_selection(cache_q,cache_k,raw_v,
                        recurrent.scores,valid.clone(),documents,self.cache_capacity,self.chunk_size,
                        self.cache_policy,cache.sink.val(),config,replay.as_deref(),prefix_replay.as_deref());
                    for selection in selections {
                        profile.cache_membership(selection.chunk_start,selection.chunk_start+self.chunk_size<t,
                            &selection.kept_slots,&selection.positions);
                        profile.prefix_visibility(selection.prefix_visible.as_ref());
                    }
                    read.output
                } else {
                    hola::read_prefix(cache_q, cache_k, raw_v, recurrent.scores,
                        valid.clone(), documents, self.cache_capacity, self.chunk_size,
                        self.cache_policy, cache.sink.val(), block_start, config, self.in_block_reads)
                }
            } else if let Some(start) = block_start {
                hola::read_frozen(cache_q, cache_k, raw_v, recurrent.scores,
                    valid.clone(), documents, self.cache_capacity, self.chunk_size,
                    self.cache_policy, cache.sink.val(), start, self.hola_backend,
                    self.in_block_reads, self.noisy_read_backend)
            } else if P::CAPTURE_CACHE_MEMBERSHIP {
                assert_eq!(
                    self.hola_backend,
                    HolaBackend::Tensor,
                    "cache membership diagnostic requires the tensor HOLA path"
                );
                let replay = profile.replayed_cache_selection();
                let (read, selections) = profile.section("HOLA read", || {
                    hola::read_with_selection(
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
                        replay.as_deref(),
                    )
                });
                for selection in selections {
                    profile.cache_membership(
                        selection.chunk_start,
                        selection.chunk_start + self.chunk_size < t,
                        &selection.kept_slots,
                        &selection.positions,
                    );
                }
                read.output
            } else {
                profile.section("HOLA read", || {
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
                })
            };
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
