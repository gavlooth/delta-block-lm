use anyhow::{Context, Result, ensure};
use delta_mixers::{
    CachePolicy, HolaBackend, MixerConfig, MixerKind, NormalizationBackend, ProjectionBackend,
    ScanBackend,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

fn is_zero(value: &usize) -> bool {
    *value == 0
}
fn lngram_one_bit(value: &delta_mixers::lngram::SurrogateGradient) -> bool {
    *value == delta_mixers::lngram::SurrogateGradient::OneBit
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    pub name: String,
    pub mixer: String,
    pub cache: String,
}
impl Arm {
    pub fn mixer_config(&self, c: &Config) -> Result<MixerConfig> {
        let kind = match self.mixer.as_str() {
            "gdn2" => MixerKind::Gdn2,
            "kronlstm" => MixerKind::KronLstm,
            value => anyhow::bail!("unknown mixer {value}"),
        };
        let cache_policy = match self.cache.as_str() {
            "none" => CachePolicy::None,
            "surprise" => CachePolicy::Surprise,
            "recency" => CachePolicy::Recency,
            value => anyhow::bail!("unknown cache {value}"),
        };
        ensure!(
            c.sbd.as_ref().is_none_or(|s| s.in_block_reads.is_none()) || cache_policy != CachePolicy::None,
            "sbd.in_block_reads=hola_read requires HOLA in every layer; arm {} has cache=none", self.name
        );
        ensure!(
            c.scan_backend.supports(kind),
            "scan selector does not support mixer kind: use fused_gdn2 for GDN-2, existing fused selectors for KronLSTM"
        );
        Ok(MixerConfig {
            d_model: c.d_model,
            heads: c.heads,
            key_dim: c.key_dim,
            value_dim: c.value_dim,
            chunk_size: c.chunk_size,
            cache_capacity: c.cache_capacity,
            cache_policy,
            kind,
            scan_backend: c.scan_backend,
            hola_backend: c.hola_backend,
            projection_backend: c.projection_backend,
            normalization_backend: c.normalization_backend,
            convolution_backend: c.convolution_backend,
        })
    }
    pub fn training_algorithm(&self, c: &Config) -> &'static str {
        let fused_hola = self.cache != "none" && c.hola_backend.is_fused();
        match (c.scan_backend, fused_hola) {
            (ScanBackend::TensorSolver, false) => "differentiable_chunk_tensor_solver",
            (ScanBackend::FusedGpu, false) => "fused_kron_gpu",
            (ScanBackend::TensorSolver, true) => "tensor_solver_fused_hola_gpu",
            (ScanBackend::FusedGpu, true) => "fused_kron_and_hola_gpu",
            (ScanBackend::FusedParallel128, false) => "fused_kron_parallel128_gpu",
            (ScanBackend::FusedParallel256, false) => "fused_kron_parallel256_gpu",
            (ScanBackend::FusedParallel128, true) => "fused_kron_parallel128_and_hola_gpu",
            (ScanBackend::FusedParallel256, true) => "fused_kron_parallel256_and_hola_gpu",
            (ScanBackend::FusedGdn2, false) => "fused_gdn2_gpu",
            (ScanBackend::FusedGdn2, true) => "fused_gdn2_and_hola_gpu",
        }
    }
    pub fn uses_custom_gpu(&self, c: &Config) -> bool {
        c.scan_backend.is_fused()
            || (self.cache != "none" && c.hola_backend.is_fused())
            || (!c.prefix_block_retrieval_backend.is_tensor()
                && c.prefix_block_retrieval.as_ref().is_some_and(|r|!r.layers.is_empty()))
            || !c.normalization_backend.is_tensor()
            || c.convolution_backend == delta_mixers::ConvolutionBackend::FusedGpu
            || !c.update_backend.is_tensor()
            || !c.loss_backend.is_tensor()
            || !c.epilogue_backend.is_native()
            || !c.ohc_backend.is_tensor()
            || !c.wavelet_backend.is_tensor()
            || (!c.lngram_layers.is_empty()
                && c.lngram_backend == delta_mixers::lngram::LngramBackend::FusedGpu)
            || c.sbd.as_ref().is_some_and(|s| !s.joint_execution.is_tensor())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadBackend {
    #[default]
    Native,
    Padded320,
    Heuristic,
}
impl HeadBackend {
    pub fn is_native(&self) -> bool {
        *self == Self::Native
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EpilogueBackend {
    #[default]
    Native,
    AnalyticBackward,
    FusedPointwise,
}
impl EpilogueBackend {
    pub fn is_native(&self) -> bool {
        *self == Self::Native
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateBackend {
    #[default]
    Tensor,
    FusedGpu,
}
impl UpdateBackend {
    pub fn is_tensor(&self) -> bool {
        *self == Self::Tensor
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LossBackend {
    #[default]
    Tensor,
    Fused64,
    Fused256,
}
impl LossBackend {
    pub fn is_tensor(&self) -> bool {
        *self == Self::Tensor
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SoftSelectionBackend {
    #[default]
    Cpu,
    Gpu,
    GpuTiled,
}
impl SoftSelectionBackend {
    pub fn is_cpu(&self) -> bool {
        *self == Self::Cpu
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SbdFeedback {
    #[default]
    SoftFeedback,
    HardMask,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SbdBlockExecution {
    #[default]
    PerBlock,
    Batched,
}
impl SbdBlockExecution {
    pub fn is_per_block(&self) -> bool {
        *self == Self::PerBlock
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SbdJointOutput {
    #[default]
    Factorized,
    Codpula,
}
impl SbdJointOutput {
    pub fn is_factorized(&self) -> bool { *self == Self::Factorized }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SbdCommitment {
    #[default]
    Fixed,
    Dependence,
}
impl SbdCommitment {
    pub fn is_fixed(&self) -> bool { *self == Self::Fixed }
}
fn dependence_threshold() -> f64 { 0.05 }
fn is_default_dependence_threshold(value: &f64) -> bool { *value == dependence_threshold() }

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CocommitExecution {
    Doubled,
    #[default]
    Separate,
}
impl CocommitExecution {
    pub fn is_separate(&self) -> bool { *self == Self::Separate }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CocommitConfig {
    pub tail_layers: usize,
    pub rounds: usize,
    pub training_bundle_max: usize,
    pub visible_loss_weight: f64,
    #[serde(skip_serializing_if = "CocommitExecution::is_separate")]
    pub execution: CocommitExecution,
}
impl Default for CocommitConfig {
    fn default() -> Self {
        Self { tail_layers: 1, rounds: 1, training_bundle_max: 16, visible_loss_weight: 0.2,
            execution: CocommitExecution::Separate }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct SbdConfig {
    pub block_lengths: Vec<usize>,
    pub feedback: SbdFeedback,
    #[serde(skip_serializing_if = "SbdBlockExecution::is_per_block")]
    pub block_execution: SbdBlockExecution,
    #[serde(skip_serializing_if = "SbdJointOutput::is_factorized")]
    pub joint_output: SbdJointOutput,
    #[serde(skip_serializing_if = "crate::codpula::CodpulaExecution::is_tensor")]
    pub joint_execution: crate::codpula::CodpulaExecution,
    #[serde(skip_serializing_if = "SbdCommitment::is_fixed")]
    pub commitment: SbdCommitment,
    /// Appendix C Figures 12–13 report 0.05; the paper gives no global default.
    #[serde(skip_serializing_if = "is_default_dependence_threshold")]
    pub dependence_threshold: f64,
    #[serde(skip_serializing_if = "delta_mixers::sbd::InBlockReads::is_none")]
    pub in_block_reads: delta_mixers::sbd::InBlockReads,
    #[serde(skip_serializing_if = "delta_mixers::sbd::NoisyReadBackend::is_tensor")]
    pub noisy_read_backend: delta_mixers::sbd::NoisyReadBackend,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cocommit: Option<CocommitConfig>,
    /// Optional exact row chunk size for SBD backward accumulation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rows_per_pass: Option<usize>,
    pub candidate_width: usize,
    pub commit_count: usize,
    /// None selects fixed-count commitment; Some selects paper equation 8 (nats).
    pub entropy_threshold: Option<f64>,
    pub generate_tokens: usize,
}
impl Default for SbdConfig {
    fn default() -> Self {
        Self { block_lengths: vec![8], feedback: SbdFeedback::SoftFeedback,
            block_execution: SbdBlockExecution::PerBlock,
            joint_output: SbdJointOutput::Factorized,
            joint_execution: crate::codpula::CodpulaExecution::Tensor,
            commitment: SbdCommitment::Fixed, dependence_threshold: dependence_threshold(),
            candidate_width: 4, commit_count: 2, entropy_threshold: None, generate_tokens: 32,
            in_block_reads: delta_mixers::sbd::InBlockReads::None,
            noisy_read_backend: delta_mixers::sbd::NoisyReadBackend::Tensor,
            cocommit: None, rows_per_pass: None }
    }
}

// Keep the historical SBD revision; optional reads enter identity through their selectors.
const SBD_MODEL_REVISION: &str = "sbd-recurrence-only-v1";
const PREFIX_RETRIEVAL_REVISION: &str = "prefix-block-moba-own-v1";
const COCOMMIT_MODEL_REVISION: &str = "cocommit-full-stream-marker-event-v1";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Omitted means the historical causal architecture and serialized config.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sbd: Option<SbdConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_block_retrieval: Option<delta_mixers::prefix_retrieval::PrefixBlockRetrieval>,
    #[serde(skip_serializing_if = "delta_mixers::prefix_retrieval::PrefixRetrievalBackend::is_tensor")]
    pub prefix_block_retrieval_backend: delta_mixers::prefix_retrieval::PrefixRetrievalBackend,
    pub task: String,
    pub backend: String,
    /// Burn's compile-time CubeCL fusion backend; omitted means historical unfused execution.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub fusion: bool,
    /// Burn's compile-time autotuner; omitted preserves the historical heuristic matmul.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub autotune: bool,
    pub artifact_dir: PathBuf,
    pub corpus_manifest: PathBuf,
    pub cache_dir: PathBuf,
    pub seed: u64,
    pub seeds: Vec<u64>,
    pub arms: Vec<Arm>,
    pub d_model: usize,
    pub layers: usize,
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub chunk_size: usize,
    pub cache_capacity: usize,
    #[serde(skip_serializing_if = "ScanBackend::is_tensor_solver")]
    pub scan_backend: ScanBackend,
    #[serde(skip_serializing_if = "HolaBackend::is_tensor")]
    pub hola_backend: HolaBackend,
    #[serde(skip_serializing_if = "ProjectionBackend::is_separate")]
    pub projection_backend: ProjectionBackend,
    #[serde(skip_serializing_if = "NormalizationBackend::is_tensor")]
    pub normalization_backend: NormalizationBackend,
    #[serde(skip_serializing_if = "delta_mixers::ConvolutionBackend::is_tensor")]
    pub convolution_backend: delta_mixers::ConvolutionBackend,
    #[serde(skip_serializing_if = "HeadBackend::is_native")]
    pub head_backend: HeadBackend,
    #[serde(skip_serializing_if = "EpilogueBackend::is_native")]
    pub epilogue_backend: EpilogueBackend,
    #[serde(skip_serializing_if = "UpdateBackend::is_tensor")]
    pub update_backend: UpdateBackend,
    #[serde(skip_serializing_if = "LossBackend::is_tensor")]
    pub loss_backend: LossBackend,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub wavelet_layers: Vec<usize>,
    #[serde(skip_serializing_if = "delta_mixers::WaveletBackend::is_tensor")]
    pub wavelet_backend: delta_mixers::WaveletBackend,
    /// Four orthogonal residual streams; omitted preserves the single-stream architecture.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub ohc_streams: bool,
    #[serde(skip_serializing_if = "delta_mixers::ohc::OhcBackend::is_tensor")]
    pub ohc_backend: delta_mixers::ohc::OhcBackend,
    /// Three target injections are independent of the per-injection n-gram orders.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub lngram_layers: Vec<usize>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub lngram_orders: Vec<usize>,
    #[serde(skip_serializing_if = "is_zero")]
    pub lngram_routes: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub lngram_bits: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub lngram_memory_dim: usize,
    #[serde(skip_serializing_if = "delta_mixers::lngram::LngramBackend::is_tensor")]
    pub lngram_backend: delta_mixers::lngram::LngramBackend,
    #[serde(skip_serializing_if = "lngram_one_bit")]
    pub lngram_surrogate: delta_mixers::lngram::SurrogateGradient,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub lngram_no_smoothing: bool,
    pub context: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f32,
    pub gradient_clip: f32,
    pub train_tokens: usize,
    pub budget_file: Option<PathBuf>,
    pub eval_examples: usize,
    pub checkpoint_every: usize,
    pub association_counts: Vec<usize>,
    pub key_symbols: usize,
    pub value_symbols: usize,
    pub eval_seed: u64,
    pub pilot_steps: usize,
    pub pilot_seconds: f64,
    pub comparison_seconds: f64,
    pub max_seconds: f64,
    pub soft_candidate_width: usize,
    #[serde(skip_serializing_if = "SoftSelectionBackend::is_cpu")]
    pub soft_selection_backend: SoftSelectionBackend,
    pub soft_steps: usize,
    pub soft_eta: f64,
    pub soft_temperature: f64,
    pub hard_mask_only: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            sbd: None,
            prefix_block_retrieval: None,
            prefix_block_retrieval_backend: delta_mixers::prefix_retrieval::PrefixRetrievalBackend::Tensor,
            task: "recall".into(),
            backend: "cpu".into(),
            fusion: false,
            autotune: false,
            artifact_dir: "artifacts/smoke".into(),
            corpus_manifest: "data/books/manifest.json".into(),
            cache_dir: ".cache/cubecl".into(),
            seed: 1,
            seeds: vec![1, 2, 3],
            arms: vec![Arm {
                name: "gdn2".into(),
                mixer: "gdn2".into(),
                cache: "none".into(),
            }],
            d_model: 128,
            layers: 2,
            heads: 4,
            key_dim: 32,
            value_dim: 64,
            chunk_size: 16,
            cache_capacity: 16,
            scan_backend: ScanBackend::TensorSolver,
            hola_backend: HolaBackend::Tensor,
            projection_backend: ProjectionBackend::Separate,
            normalization_backend: NormalizationBackend::Tensor,
            convolution_backend: delta_mixers::ConvolutionBackend::Tensor,
            head_backend: HeadBackend::Native,
            epilogue_backend: EpilogueBackend::Native,
            update_backend: UpdateBackend::Tensor,
            loss_backend: LossBackend::Tensor,
            ohc_streams: false,
            ohc_backend: delta_mixers::ohc::OhcBackend::Tensor,
            wavelet_layers: Vec::new(),
            wavelet_backend: delta_mixers::WaveletBackend::Tensor,
            lngram_layers: Vec::new(),
            lngram_orders: Vec::new(),
            lngram_routes: 0,
            lngram_bits: 0,
            lngram_memory_dim: 0,
            lngram_backend: delta_mixers::lngram::LngramBackend::Tensor,
            lngram_surrogate: delta_mixers::lngram::SurrogateGradient::OneBit,
            lngram_no_smoothing: false,
            context: 128,
            batch_size: 1,
            learning_rate: 0.001,
            weight_decay: 0.01,
            gradient_clip: 1.0,
            train_tokens: 2048,
            budget_file: None,
            eval_examples: 16,
            checkpoint_every: 8,
            association_counts: vec![4, 8, 16],
            key_symbols: 64,
            value_symbols: 64,
            eval_seed: 0x5eeda11,
            pilot_steps: 3,
            pilot_seconds: 10.0,
            comparison_seconds: 60.0,
            max_seconds: 90.0,
            soft_candidate_width: 4,
            soft_selection_backend: SoftSelectionBackend::Cpu,
            soft_steps: 4,
            soft_eta: 0.5,
            soft_temperature: 1.0,
            hard_mask_only: false,
        }
    }
}
impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let c: Self = serde_json::from_slice(
            &std::fs::read(path).with_context(|| format!("reading {}", path.display()))?,
        )?;
        c.validate()?;
        Ok(c)
    }
    pub fn validate(&self) -> Result<()> {
        if let Some(retrieval) = &self.prefix_block_retrieval {
            ensure!(self.chunk_size > 0 && retrieval.block_size > 0
                && retrieval.block_size % self.chunk_size == 0,
                "prefix retrieval block_size must be a positive multiple of chunk_size");
            ensure!(retrieval.blocks_per_query >= 1, "prefix retrieval blocks_per_query must be positive");
            ensure!(retrieval.layers.iter().all(|&layer| layer < self.layers),
                "prefix retrieval layer indices must be in range");
            ensure!(retrieval.layers.is_empty() || (self.cache_capacity > 0 && self.arms.iter().all(|arm| arm.cache != "none")),
                "prefix retrieval requires HOLA in every selected layer");
        }
        ensure!(self.prefix_block_retrieval_backend.is_tensor()
            || (self.prefix_block_retrieval.is_some() && self.backend == "vulkan"
                && self.key_dim == 32 && self.value_dim == 64 && (1..=64).contains(&self.cache_capacity) && !self.fusion),
            "fused prefix retrieval requires selected layers, K32/V64 and non-fusion Vulkan");
        if let Some(sbd) = &self.sbd {
            ensure!(self.task == "lm", "SBD requires byte language modeling");
            ensure!(sbd.rows_per_pass.is_none_or(|rows| rows > 0),
                "sbd.rows_per_pass must be positive when specified");
            ensure!(!self.autotune, "SBD variable-shape matmuls are not covered by the autotune preflight");
            ensure!(!sbd.block_lengths.is_empty() && sbd.block_lengths.iter().all(|&n| n > 0),
                "SBD block lengths must be positive");
            ensure!(sbd.candidate_width > 0 && sbd.candidate_width <= self.vocab()
                && sbd.commit_count > 0 && sbd.generate_tokens > 0, "invalid SBD decoding counts");
            ensure!(sbd.entropy_threshold.is_none_or(|x| x.is_finite() && x >= 0.0),
                "SBD entropy threshold must be finite and nonnegative");
            ensure!(sbd.dependence_threshold.is_finite() && sbd.dependence_threshold > 0.0,
                "SBD dependence threshold must be finite and positive");
            ensure!(sbd.commitment != SbdCommitment::Dependence || sbd.joint_output == SbdJointOutput::Codpula,
                "dependence commitment requires sbd.joint_output=codpula");
            ensure!(sbd.joint_output != SbdJointOutput::Codpula || sbd.block_lengths.iter().all(|&n| n <= 16),
                "Codpula block lengths must be at most 16");
            ensure!(sbd.joint_execution.is_tensor()
                || (sbd.joint_output == SbdJointOutput::Codpula
                    && matches!(self.backend.as_str(),"vulkan" | "cuda") && !self.fusion),
                "sbd.joint_execution=fused_gpu requires Codpula and explicit non-fusion GPU execution");
            ensure!(
                sbd.noisy_read_backend.is_tensor()
                    || (matches!(self.backend.as_str(),"vulkan" | "cuda") && !self.fusion
                        && self.key_dim == 32 && self.value_dim == 64 && self.chunk_size <= 16
                        && self.cache_capacity <= 64 && sbd.block_lengths.iter().all(|&n| n <= 16)),
                "sbd.noisy_read_backend=fused_gpu requires explicit non-fusion GPU execution, key32/value64, chunk/query count <=16 and capacity <=64"
            );
            if let Some(cocommit) = &sbd.cocommit {
                ensure!((1..=self.layers).contains(&cocommit.tail_layers),
                    "sbd.cocommit.tail_layers must be within 1..=layers");
                ensure!(cocommit.rounds == 1, "sbd.cocommit currently requires rounds=1");
                ensure!(cocommit.training_bundle_max > 0,
                    "sbd.cocommit.training_bundle_max must be positive");
                ensure!(cocommit.visible_loss_weight.is_finite() && cocommit.visible_loss_weight >= 0.0,
                    "sbd.cocommit.visible_loss_weight must be finite and nonnegative");
                ensure!(sbd.in_block_reads == delta_mixers::sbd::InBlockReads::HolaRead,
                    "sbd.cocommit requires in_block_reads=hola_read");
                ensure!(self.cache_capacity > 0 && self.arms.iter().all(|arm| arm.cache != "none"),
                    "sbd.cocommit requires HOLA cache layers");
            }
        }
        ensure!(
            self.ohc_backend.is_tensor()
                || (self.ohc_streams
                    && matches!(self.backend.as_str(), "vulkan" | "cuda")
                    && !self.fusion),
            "fused oHC requires ohc_streams and an explicit non-fusion GPU backend"
        );
        ensure!(
            !self.ohc_backend.fused_read() || self.d_model <= 1024,
            "fused oHC read supports d_model <= 1024"
        );
        ensure!(
            self.wavelet_layers.iter().all(|&layer| layer < self.layers)
                && self
                    .wavelet_layers
                    .iter()
                    .enumerate()
                    .all(|(i, layer)| !self.wavelet_layers[..i].contains(layer)),
            "wavelet_layers must contain unique in-range block indices"
        );
        ensure!(
            self.wavelet_layers.is_empty() || self.d_model % 8 == 0,
            "wavelet requires d_model divisible by eight gate groups"
        );
        ensure!(
            self.wavelet_backend.is_tensor()
                || (!self.wavelet_layers.is_empty()
                    && matches!(self.backend.as_str(), "vulkan" | "cuda")
                    && !self.fusion),
            "fused wavelet requires enabled layers and an explicit non-fusion GPU backend"
        );
        ensure!(
            matches!(self.task.as_str(), "recall" | "lm"),
            "task must be recall or lm"
        );
        ensure!(
            !self.fusion || matches!(self.backend.as_str(), "vulkan" | "cuda"),
            "fusion requires an explicit GPU backend"
        );
        ensure!(
            !self.autotune || matches!(self.backend.as_str(), "vulkan" | "cuda"),
            "autotune requires an explicit GPU backend"
        );
        ensure!(
            (self.update_backend.is_tensor() && self.loss_backend.is_tensor())
                || (matches!(self.backend.as_str(), "vulkan" | "cuda") && !self.fusion),
            "fused update/loss require an explicit non-fusion GPU backend"
        );
        ensure!(
            self.soft_selection_backend.is_cpu()
                || matches!(self.backend.as_str(), "vulkan" | "cuda"),
            "GPU soft selection requires an explicit GPU backend"
        );
        ensure!(
            self.head_backend.is_native() || self.task == "lm",
            "experimental head paths require the byte-LM vocabulary"
        );
        ensure!(
            self.head_backend != HeadBackend::Heuristic
                || matches!(self.backend.as_str(), "vulkan" | "cuda"),
            "heuristic head requires a GPU backend"
        );
        ensure!(
            self.projection_backend != ProjectionBackend::MergedHeuristicOutput
                || matches!(self.backend.as_str(), "vulkan" | "cuda"),
            "heuristic output projection requires a GPU backend"
        );
        if self.autotune {
            crate::gemm_guard::validate(self)?;
        }
        ensure!(
            self.context >= 8 && self.batch_size > 0 && self.layers > 0 && self.d_model > 0,
            "invalid model shape"
        );
        let mut lngram_layers = std::collections::BTreeSet::new();
        ensure!(
            self.lngram_layers
                .iter()
                .all(|&i| i < self.layers && lngram_layers.insert(i)),
            "lngram_layers must be distinct existing block indices"
        );
        ensure!(
            self.lngram_backend != delta_mixers::lngram::LngramBackend::FusedGpu
                || (matches!(self.backend.as_str(), "vulkan" | "cuda") && !self.fusion),
            "fused_gpu Lngram requires an explicit non-fusion GPU backend"
        );
        if !self.lngram_layers.is_empty() {
            let config = self.lngram_config();
            let mut orders = std::collections::BTreeSet::new();
            ensure!(
                config.routes > 0
                    && config.memory_dim > 0
                    && (1..=8).contains(&config.bits_per_route),
                "Lngram routes/memory_dim must be positive and bits must be 1..=8"
            );
            ensure!(
                !config.orders.is_empty()
                    && config.orders.iter().all(|&n| n > 0 && orders.insert(n)),
                "Lngram orders must be distinct positive integers"
            );
            for &n in &config.orders {
                let rows = u32::try_from(n)
                    .ok()
                    .and_then(|n| config.alphabet().checked_pow(n))
                    .and_then(|rows| rows.checked_mul(config.routes));
                ensure!(
                    rows.is_some_and(|rows| rows <= i32::MAX as usize),
                    "Lngram table exceeds i32 row addresses"
                );
            }
        }
        ensure!(
            self.heads == 4 && self.key_dim == 32 && self.value_dim == 64 && self.chunk_size == 16,
            "preserve inherited 4x32x64, chunk16 kernel dimensions"
        );
        ensure!(
            self.eval_examples > 0 && self.pilot_steps > 0 && self.checkpoint_every > 0,
            "counts must be positive"
        );
        ensure!(
            self.learning_rate.is_finite() && self.learning_rate > 0.0 && self.gradient_clip > 0.0,
            "invalid optimizer"
        );
        ensure!(
            self.max_seconds.is_finite()
                && self.max_seconds > 0.0
                && self.pilot_seconds > 0.0
                && self.comparison_seconds > 0.0,
            "invalid deadlines"
        );
        ensure!(
            (!self.scan_backend.is_fused()
                && !self.hola_backend.is_fused()
                && self.normalization_backend.is_tensor()
                && self.epilogue_backend.is_native()
                && self.convolution_backend != delta_mixers::ConvolutionBackend::FusedGpu)
                || matches!(self.backend.as_str(), "vulkan" | "cuda"),
            "fused_gpu requires an explicit GPU backend; no CPU fallback"
        );
        ensure!(
            !self.hola_backend.is_fused()
                || !self.arms.iter().any(|arm| arm.cache != "none")
                || (1..=64).contains(&self.cache_capacity),
            "fused HOLA capacity must be 1..=64"
        );
        ensure!(
            !self.arms.is_empty()
                && self.arms.iter().all(|a| !a.name.is_empty()
                    && a.name
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')),
            "arm names must be safe path components"
        );
        let mut names = std::collections::BTreeSet::new();
        for arm in &self.arms {
            ensure!(names.insert(&arm.name), "duplicate arm name");
            arm.mixer_config(self)?;
        }
        ensure!(
            !self.association_counts.is_empty()
                && self
                    .association_counts
                    .iter()
                    .all(|&n| n > 0 && n <= self.key_symbols && 5 * n + 2 <= self.context),
            "associations need distinct keys and room for all queries"
        );
        ensure!(
            self.key_symbols > 0 && self.value_symbols > 0,
            "empty symbol vocabulary"
        );
        ensure!(
            self.soft_candidate_width > 0
                && self.soft_steps > 0
                && (0.0..=1.0).contains(&self.soft_eta)
                && self.soft_temperature > 0.0,
            "invalid feedback schedule"
        );
        Ok(())
    }
    pub fn lngram_config(&self) -> delta_mixers::lngram::LngramConfig {
        delta_mixers::lngram::LngramConfig::new()
            .with_d_model(self.d_model)
            .with_routes(if self.lngram_routes == 0 {
                (self.d_model / 8).max(1)
            } else {
                self.lngram_routes
            })
            .with_bits_per_route(if self.lngram_bits == 0 {
                4
            } else {
                self.lngram_bits
            })
            .with_memory_dim(if self.lngram_memory_dim == 0 {
                8
            } else {
                self.lngram_memory_dim
            })
            .with_orders(if self.lngram_orders.is_empty() {
                vec![2, 3]
            } else {
                self.lngram_orders.clone()
            })
            .with_backend(self.lngram_backend)
            .with_surrogate(self.lngram_surrogate)
            .with_smoothing(!self.lngram_no_smoothing)
    }
    pub fn vocab(&self) -> usize {
        if self.task == "lm" {
            257
        } else {
            4 + self.key_symbols + self.value_symbols
        }
    }
    pub fn tokens_per_step(&self) -> usize {
        self.context * self.batch_size
    }
    pub fn training_fingerprint(&self, target_tokens: usize) -> Result<String> {
        let bytes = if self.sbd.as_ref().is_some_and(|s|s.cocommit.is_some()) {
            serde_json::to_vec(&(self,target_tokens,SBD_MODEL_REVISION,
                self.prefix_block_retrieval.as_ref().map(|_|PREFIX_RETRIEVAL_REVISION),COCOMMIT_MODEL_REVISION))?
        } else if self.prefix_block_retrieval.is_some() {
            serde_json::to_vec(&(self,target_tokens,
                self.sbd.as_ref().map(|_|SBD_MODEL_REVISION),PREFIX_RETRIEVAL_REVISION))?
        } else if self.sbd.is_some() {
            serde_json::to_vec(&(self, target_tokens, SBD_MODEL_REVISION))?
        } else {
            // Preserve historical causal checkpoint and metrics identities.
            serde_json::to_vec(&(self, target_tokens))?
        };
        Ok(crate::data::hash(&bytes))
    }
    pub fn pilot_fingerprint(&self) -> Result<String> {
        let mut identity = serde_json::json!({"config":self,"precision":"float32","training_algorithm":self.scan_backend.training_algorithm(),"optimizer":"AdamW"});
        if self.sbd.is_some() {
            identity["sbd_model_revision"] = SBD_MODEL_REVISION.into();
        }
        if self.prefix_block_retrieval.is_some() {
            identity["prefix_block_retrieval_revision"] = PREFIX_RETRIEVAL_REVISION.into();
        }
        if self.sbd.as_ref().is_some_and(|s|s.cocommit.is_some()) {
            identity["cocommit_model_revision"] = COCOMMIT_MODEL_REVISION.into();
        }
        Ok(crate::data::hash(&serde_json::to_vec(&identity)?))
    }
    pub fn resolved_tokens(&self) -> Result<usize> {
        if let Some(path) = &self.budget_file {
            let v: serde_json::Value = serde_json::from_slice(
                &std::fs::read(path)
                    .with_context(|| format!("pilot budget {} is required", path.display()))?,
            )?;
            ensure!(
                v["pilot_config_sha256"].as_str() == Some(self.pilot_fingerprint()?.as_str()),
                "stale pilot budget: complete config/cache/optimizer changed; run a new pilot in a fresh artifact directory"
            );
            ensure!(
                v["task"] == self.task
                    && v["context"] == self.context
                    && v["batch_size"] == self.batch_size
                    && v["d_model"] == self.d_model
                    && v["layers"] == self.layers
                    && v["backend"] == self.backend,
                "pilot budget belongs to a different experiment"
            );
            ensure!(
                v["arms"] == serde_json::to_value(&self.arms)?,
                "pilot must time every comparison arm"
            );
            Ok(v["train_tokens"]
                .as_u64()
                .context("missing pilot token budget")? as usize)
        } else {
            ensure!(self.train_tokens > 0, "zero training-token budget");
            Ok(self.train_tokens)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prefix_retrieval_serialization_and_identity() {
        let baseline = Config::default();
        let value = serde_json::to_value(&baseline).unwrap();
        assert!(value.get("prefix_block_retrieval").is_none());
        assert!(value.get("prefix_block_retrieval_backend").is_none());
        let restored: Config = serde_json::from_value(value).unwrap();
        assert_eq!(baseline.pilot_fingerprint().unwrap(), restored.pilot_fingerprint().unwrap());
        assert_eq!(baseline.training_fingerprint(2048).unwrap(), restored.training_fingerprint(2048).unwrap());
        let mut enabled = baseline.clone();
        enabled.arms[0].cache = "surprise".into();
        let historical = enabled.clone();
        enabled.prefix_block_retrieval = Some(delta_mixers::prefix_retrieval::PrefixBlockRetrieval {
            layers: vec![0], block_size: 64, blocks_per_query: 2,
        });
        assert!(enabled.validate().is_ok());
        assert_ne!(historical.pilot_fingerprint().unwrap(), enabled.pilot_fingerprint().unwrap());
        assert_ne!(historical.training_fingerprint(2048).unwrap(), enabled.training_fingerprint(2048).unwrap());
        let previous = crate::data::hash(&serde_json::to_vec(&(&enabled,2048)).unwrap());
        assert_ne!(previous,enabled.training_fingerprint(2048).unwrap());
        enabled.prefix_block_retrieval.as_mut().unwrap().block_size = 0;
        assert!(enabled.validate().is_err());
        enabled.prefix_block_retrieval.as_mut().unwrap().block_size = 17;
        assert!(enabled.validate().is_err());
        enabled.prefix_block_retrieval.as_mut().unwrap().block_size = 64;
        enabled.prefix_block_retrieval.as_mut().unwrap().blocks_per_query = 0;
        assert!(enabled.validate().is_err());
        enabled.prefix_block_retrieval.as_mut().unwrap().blocks_per_query = 2;
        enabled.prefix_block_retrieval.as_mut().unwrap().layers = vec![enabled.layers];
        assert!(enabled.validate().is_err());
        enabled.prefix_block_retrieval.as_mut().unwrap().layers = vec![0];
        enabled.arms[0].cache = "none".into();
        assert!(enabled.validate().is_err());
    }
    #[test]
    fn lngram_omission_preserves_serialization_and_enabled_changes_identity() {
        let baseline = Config::default();
        let value = serde_json::to_value(&baseline).unwrap();
        assert!(
            value
                .as_object()
                .unwrap()
                .keys()
                .all(|key| !key.starts_with("lngram_"))
        );
        let decoded: Config = serde_json::from_value(value).unwrap();
        assert!(decoded.lngram_layers.is_empty());
        let mut enabled = baseline.clone();
        enabled.lngram_layers = vec![0, 1];
        assert_ne!(
            baseline.pilot_fingerprint().unwrap(),
            enabled.pilot_fingerprint().unwrap()
        );
        assert_eq!(enabled.lngram_config().orders, vec![2, 3]);
        enabled.lngram_layers = vec![0, 0];
        assert!(enabled.validate().is_err());
        enabled.lngram_layers = vec![enabled.layers];
        assert!(enabled.validate().is_err());
        enabled.lngram_layers = vec![0];
        enabled.lngram_orders = vec![2, 2];
        assert!(enabled.validate().is_err());
        let mut fused = Config::default();
        fused.lngram_layers = vec![0];
        fused.lngram_backend = delta_mixers::lngram::LngramBackend::FusedGpu;
        assert!(fused.validate().is_err());
        fused.backend = "vulkan".into();
        assert!(fused.validate().is_ok());
        fused.fusion = true;
        assert!(fused.validate().is_err());
    }
    #[test]
    fn pilot_identity_includes_cache_and_optimizer() {
        let c = Config::default();
        let baseline = c.pilot_fingerprint().unwrap();
        let mut changed = c.clone();
        changed.cache_capacity = 64;
        assert_ne!(baseline, changed.pilot_fingerprint().unwrap());
        let mut changed = c.clone();
        changed.learning_rate *= 2.0;
        assert_ne!(baseline, changed.pilot_fingerprint().unwrap());
        let mut changed = c;
        changed.gradient_clip *= 2.0;
        assert_ne!(baseline, changed.pilot_fingerprint().unwrap());
        let mut changed = Config::default();
        changed.scan_backend = ScanBackend::FusedGpu;
        assert_ne!(
            Config::default().pilot_fingerprint().unwrap(),
            changed.pilot_fingerprint().unwrap()
        );
        let mut changed = Config::default();
        changed.hola_backend = HolaBackend::FusedGpu;
        assert_ne!(
            Config::default().pilot_fingerprint().unwrap(),
            changed.pilot_fingerprint().unwrap()
        );
        for backend in [HolaBackend::FusedGpuPairs, HolaBackend::FusedGpuPairsNorms] {
            changed.hola_backend = backend;
            assert_ne!(
                Config::default().pilot_fingerprint().unwrap(),
                changed.pilot_fingerprint().unwrap()
            );
        }
        let mut changed = Config::default();
        changed.fusion = true;
        assert_ne!(
            Config::default().pilot_fingerprint().unwrap(),
            changed.pilot_fingerprint().unwrap()
        );
        let mut changed = Config::default();
        changed.autotune = true;
        assert_ne!(
            Config::default().pilot_fingerprint().unwrap(),
            changed.pilot_fingerprint().unwrap()
        );
        let mut changed = Config::default();
        changed.projection_backend = ProjectionBackend::Merged;
        assert_ne!(
            Config::default().pilot_fingerprint().unwrap(),
            changed.pilot_fingerprint().unwrap()
        );
        let mut changed = Config::default();
        changed.normalization_backend = NormalizationBackend::FusedNormsGates;
        assert_ne!(
            Config::default().pilot_fingerprint().unwrap(),
            changed.pilot_fingerprint().unwrap()
        );
        for backend in [
            delta_mixers::ConvolutionBackend::PaddedTensor,
            delta_mixers::ConvolutionBackend::FusedGpu,
        ] {
            let mut changed = Config::default();
            changed.convolution_backend = backend;
            assert_ne!(
                Config::default().pilot_fingerprint().unwrap(),
                changed.pilot_fingerprint().unwrap()
            );
        }
        let mut changed = Config::default();
        changed.update_backend = UpdateBackend::FusedGpu;
        assert_ne!(
            Config::default().pilot_fingerprint().unwrap(),
            changed.pilot_fingerprint().unwrap()
        );
        let mut changed = Config::default();
        changed.loss_backend = LossBackend::Fused256;
        assert_ne!(
            Config::default().pilot_fingerprint().unwrap(),
            changed.pilot_fingerprint().unwrap()
        );
    }
    #[test]
    fn ohc_is_explicit_and_changes_architecture_identity() {
        let mut c = Config::default();
        let identity = c.pilot_fingerprint().unwrap();
        c.ohc_streams = true;
        c.validate().unwrap();
        assert_ne!(identity, c.pilot_fingerprint().unwrap());
        c.ohc_backend = delta_mixers::ohc::OhcBackend::FusedGpu;
        assert!(c.validate().is_err());
        c.backend = "vulkan".into();
        c.validate().unwrap();
        c.ohc_streams = false;
        assert!(c.validate().is_err());
        c.ohc_streams = true;
        for backend in [
            delta_mixers::ohc::OhcBackend::FusedReadWrite,
            delta_mixers::ohc::OhcBackend::FusedGate,
            delta_mixers::ohc::OhcBackend::TransposedGate,
        ] {
            c.ohc_backend = backend;
            c.d_model = 256;
            c.validate().unwrap();
            let encoded = serde_json::to_value(&c).unwrap();
            let decoded: Config = serde_json::from_value(encoded).unwrap();
            assert_eq!(decoded.ohc_backend, backend);
            c.d_model = 1025;
            assert!(c.validate().is_err());
        }
    }
    #[test]
    fn legacy_selector_defaults_without_changing_serialized_identity() {
        let config = Config::default();
        let legacy = serde_json::to_value(&config).unwrap();
        assert!(legacy.get("scan_backend").is_none());
        assert!(legacy.get("hola_backend").is_none());
        assert!(legacy.get("fusion").is_none());
        assert!(legacy.get("autotune").is_none());
        assert!(legacy.get("projection_backend").is_none());
        assert!(legacy.get("normalization_backend").is_none());
        assert!(legacy.get("convolution_backend").is_none());
        assert!(legacy.get("head_backend").is_none());
        assert!(legacy.get("update_backend").is_none());
        assert!(legacy.get("loss_backend").is_none());
        assert!(legacy.get("ohc_streams").is_none());
        assert!(legacy.get("ohc_backend").is_none());
        assert!(legacy.get("soft_selection_backend").is_none());
        let decoded: Config = serde_json::from_value(legacy.clone()).unwrap();
        assert_eq!(decoded.scan_backend, ScanBackend::TensorSolver);
        assert_eq!(decoded.hola_backend, HolaBackend::Tensor);
        assert!(!decoded.fusion);
        assert!(!decoded.autotune);
        assert_eq!(decoded.projection_backend, ProjectionBackend::Separate);
        assert_eq!(decoded.normalization_backend, NormalizationBackend::Tensor);
        assert_eq!(
            decoded.convolution_backend,
            delta_mixers::ConvolutionBackend::Tensor
        );
        assert_eq!(decoded.head_backend, HeadBackend::Native);
        assert_eq!(decoded.update_backend, UpdateBackend::Tensor);
        assert_eq!(decoded.loss_backend, LossBackend::Tensor);
        assert_eq!(decoded.soft_selection_backend, SoftSelectionBackend::Cpu);
        assert!(!decoded.ohc_streams);
        assert_eq!(decoded.ohc_backend, delta_mixers::ohc::OhcBackend::Tensor);
        assert_eq!(serde_json::to_value(decoded).unwrap(), legacy);
    }
    #[test]
    fn automatic_fusion_is_explicit_and_gpu_only() {
        let mut config = Config::default();
        config.fusion = true;
        assert!(config.validate().is_err());
        config.backend = "vulkan".into();
        config.validate().unwrap();
        let encoded = serde_json::to_value(&config).unwrap();
        assert_eq!(encoded["fusion"], true);
        assert!(serde_json::from_value::<Config>(encoded).unwrap().fusion);
    }
    #[test]
    fn autotune_is_explicit_and_gpu_only() {
        let mut config = Config::default();
        config.autotune = true;
        assert!(config.validate().is_err());
        config.backend = "vulkan".into();
        config.task = "lm".into();
        config.scan_backend = ScanBackend::FusedGpu;
        config.hola_backend = HolaBackend::FusedGpu;
        config.projection_backend = ProjectionBackend::Merged;
        config.arms[0].mixer = "kronlstm".into();
        config.validate().unwrap();
        let encoded = serde_json::to_value(&config).unwrap();
        assert_eq!(encoded["autotune"], true);
        assert!(serde_json::from_value::<Config>(encoded).unwrap().autotune);
        let mut padded = config.clone();
        padded.head_backend = HeadBackend::Padded320;
        padded.task = "lm".into();
        assert_ne!(
            config.pilot_fingerprint().unwrap(),
            padded.pilot_fingerprint().unwrap()
        );
    }
    #[test]
    fn parallel_scan_selectors_are_explicit_distinct_and_gpu_only() {
        let baseline = Config::default().pilot_fingerprint().unwrap();
        let mut fingerprints = Vec::new();
        for backend in [
            ScanBackend::FusedGpu,
            ScanBackend::FusedParallel128,
            ScanBackend::FusedParallel256,
        ] {
            let mut config = Config::default();
            config.scan_backend = backend;
            assert!(config.validate().is_err());
            config.backend = "vulkan".into();
            assert!(config.arms[0].mixer_config(&config).is_err());
            config.arms[0].mixer = "kronlstm".into();
            config.arms[0].mixer_config(&config).unwrap();
            let encoded = serde_json::to_value(&config).unwrap();
            assert_eq!(
                serde_json::from_value::<Config>(encoded)
                    .unwrap()
                    .scan_backend,
                backend
            );
            let identity = config.pilot_fingerprint().unwrap();
            assert_ne!(identity, baseline);
            assert!(!fingerprints.contains(&identity));
            fingerprints.push(identity);
        }
    }
    #[test]
    fn fused_selectors_reject_cpu_and_gdn_scan_without_fallback() {
        let mut config = Config::default();
        config.scan_backend = ScanBackend::FusedGpu;
        assert!(config.validate().is_err());
        config.backend = "vulkan".into();
        assert!(config.arms[0].mixer_config(&config).is_err());
        config.scan_backend = ScanBackend::TensorSolver;
        config.backend = "cpu".into();
        for backend in [
            HolaBackend::FusedGpu,
            HolaBackend::FusedGpuPairs,
            HolaBackend::FusedGpuPairsNorms,
        ] {
            config.hola_backend = backend;
            assert!(config.validate().is_err());
        }
    }
    #[test]
    fn update_and_loss_selectors_are_explicit_gpu_only() {
        for (update_backend, loss_backend) in [
            (UpdateBackend::FusedGpu, LossBackend::Tensor),
            (UpdateBackend::Tensor, LossBackend::Fused64),
            (UpdateBackend::Tensor, LossBackend::Fused256),
        ] {
            let mut config = Config {
                update_backend,
                loss_backend,
                ..Config::default()
            };
            assert!(config.validate().is_err());
            config.backend = "vulkan".into();
            config.validate().unwrap();
            let baseline = Config::default().pilot_fingerprint().unwrap();
            assert_ne!(baseline, config.pilot_fingerprint().unwrap());
            config.fusion = true;
            assert!(config.validate().is_err());
        }
    }
    #[test]
    fn soft_selection_is_explicit_and_gpu_only() {
        let mut config = Config::default();
        let baseline = config.pilot_fingerprint().unwrap();
        for backend in [SoftSelectionBackend::Gpu, SoftSelectionBackend::GpuTiled] {
            config.soft_selection_backend = backend;
            config.backend = "cpu".into();
            assert!(config.validate().is_err());
            config.backend = "vulkan".into();
            config.validate().unwrap();
            assert_ne!(baseline, config.pilot_fingerprint().unwrap());
            let encoded = serde_json::to_value(&config).unwrap();
            assert_eq!(
                serde_json::from_value::<Config>(encoded)
                    .unwrap()
                    .soft_selection_backend,
                backend
            );
        }
    }
    #[test]
    fn wavelet_is_omitted_by_default_and_changes_identity_when_enabled() {
        let mut c = Config::default();
        let original = c.pilot_fingerprint().unwrap();
        let encoded = serde_json::to_value(&c).unwrap();
        assert!(encoded.get("wavelet_layers").is_none());
        assert!(encoded.get("wavelet_backend").is_none());
        c.wavelet_layers = vec![0];
        c.validate().unwrap();
        assert_ne!(original, c.pilot_fingerprint().unwrap());
        c.wavelet_layers = vec![0, 0];
        assert!(c.validate().is_err());
        c.wavelet_layers = vec![c.layers];
        assert!(c.validate().is_err());
        c.wavelet_layers = vec![0];
        c.wavelet_backend = delta_mixers::WaveletBackend::FusedGpu;
        assert!(c.validate().is_err());
        c.backend = "vulkan".into();
        c.validate().unwrap();
        c.fusion = true;
        assert!(c.validate().is_err());
    }
    #[test]
    fn sbd_revision_changes_only_enabled_model_fingerprints() {
        let causal = Config::default();
        let target_tokens = causal.train_tokens;
        let legacy_training = |c: &Config| crate::data::hash(&serde_json::to_vec(&(c, target_tokens)).unwrap());
        let legacy_pilot = |c: &Config| crate::data::hash(&serde_json::to_vec(
            &serde_json::json!({"config":c,"precision":"float32","training_algorithm":c.scan_backend.training_algorithm(),"optimizer":"AdamW"}),
        ).unwrap());
        assert_eq!(causal.training_fingerprint(target_tokens).unwrap(), legacy_training(&causal));
        assert_eq!(causal.pilot_fingerprint().unwrap(), legacy_pilot(&causal));
        let enabled = Config { task: "lm".into(), sbd: Some(SbdConfig::default()), ..causal };
        assert_ne!(enabled.training_fingerprint(target_tokens).unwrap(), legacy_training(&enabled));
        assert_ne!(enabled.pilot_fingerprint().unwrap(), legacy_pilot(&enabled));
    }
    #[test]
    fn sbd_defaults_preserve_causal_serialization() {
        let c = Config::default();
        assert!(serde_json::to_value(&c).unwrap().get("sbd").is_none());
        let mut enabled = Config { task: "lm".into(), sbd: Some(SbdConfig::default()), ..c.clone() };
        enabled.validate().unwrap();
        assert_ne!(c.pilot_fingerprint().unwrap(), enabled.pilot_fingerprint().unwrap());
        assert_eq!(enabled.sbd.as_ref().unwrap().feedback, SbdFeedback::SoftFeedback);
        enabled.sbd.as_mut().unwrap().block_lengths = vec![1,4,8,16];
        enabled.sbd.as_mut().unwrap().candidate_width = 257;
        enabled.sbd.as_mut().unwrap().commit_count = 3;
        enabled.validate().unwrap();
        enabled.sbd.as_mut().unwrap().block_lengths = vec![0];
        assert!(enabled.validate().is_err());
    }
    #[test]
    fn block_execution_defaults_preserve_sbd_identity() {
        let mut c = Config {task:"lm".into(),sbd:Some(SbdConfig::default()),..Config::default()};
        let omitted = serde_json::to_value(&c).unwrap();
        assert!(omitted["sbd"].get("block_execution").is_none());
        let training = c.training_fingerprint(c.train_tokens).unwrap();
        let pilot = c.pilot_fingerprint().unwrap();
        let mut explicit = omitted.clone();
        explicit["sbd"]["block_execution"] = "per_block".into();
        let explicit: Config = serde_json::from_value(explicit).unwrap();
        assert_eq!(explicit.sbd.as_ref().unwrap().block_execution,SbdBlockExecution::PerBlock);
        assert_eq!(serde_json::to_value(&explicit).unwrap(),omitted);
        assert_eq!(explicit.training_fingerprint(c.train_tokens).unwrap(),training);
        assert_eq!(explicit.pilot_fingerprint().unwrap(),pilot);
        c.sbd.as_mut().unwrap().block_execution = SbdBlockExecution::Batched;
        let encoded = serde_json::to_value(&c).unwrap();
        assert_eq!(encoded["sbd"]["block_execution"],"batched");
        let restored: Config = serde_json::from_value(encoded).unwrap();
        assert_eq!(restored.sbd.as_ref().unwrap().block_execution,SbdBlockExecution::Batched);
        assert_ne!(c.training_fingerprint(c.train_tokens).unwrap(),training);
        assert_ne!(c.pilot_fingerprint().unwrap(),pilot);
        let mut invalid = omitted;
        invalid["sbd"]["block_execution"] = "automatic".into();
        assert!(serde_json::from_value::<Config>(invalid).is_err());
    }
    #[test]
    fn in_block_read_default_identity_and_hola_requirement() {
        let mut c = Config {task:"lm".into(),sbd:Some(SbdConfig::default()),..Config::default()};
        let omitted = serde_json::to_value(&c).unwrap();
        assert!(omitted["sbd"].get("in_block_reads").is_none());
        assert!(omitted["sbd"].get("noisy_read_backend").is_none());
        let old_training = c.training_fingerprint(c.train_tokens).unwrap();
        let old_pilot = c.pilot_fingerprint().unwrap();
        let mut explicit = omitted.clone();
        explicit["sbd"]["in_block_reads"] = "none".into();
        explicit["sbd"]["noisy_read_backend"] = "tensor".into();
        let explicit: Config = serde_json::from_value(explicit).unwrap();
        assert_eq!(serde_json::to_value(&explicit).unwrap(),omitted);
        assert_eq!(explicit.training_fingerprint(c.train_tokens).unwrap(),old_training);
        assert_eq!(explicit.pilot_fingerprint().unwrap(),old_pilot);
        c.sbd.as_mut().unwrap().in_block_reads = delta_mixers::sbd::InBlockReads::HolaRead;
        c.arms = vec![Arm {name:"without-hola".into(),mixer:"kronlstm".into(),cache:"none".into()}];
        assert!(c.validate().unwrap_err().to_string().contains("requires HOLA"));
        c.arms[0].cache = "surprise".into();
        c.validate().unwrap();
        assert_ne!(c.training_fingerprint(c.train_tokens).unwrap(),old_training);
        assert_ne!(c.pilot_fingerprint().unwrap(),old_pilot);
        assert!(serde_json::from_value::<Config>(serde_json::json!({"in_block_reads":"hola_read"})).is_err());
    }
    #[test]
    fn joint_output_defaults_preserve_historical_sbd_identity() {
        let mut c = Config {task:"lm".into(),sbd:Some(SbdConfig::default()),..Config::default()};
        let omitted = serde_json::to_value(&c).unwrap();
        for field in ["joint_output","joint_execution","commitment","dependence_threshold"] {
            assert!(omitted["sbd"].get(field).is_none());
        }
        let training = c.training_fingerprint(c.train_tokens).unwrap();
        let pilot = c.pilot_fingerprint().unwrap();
        let mut explicit = omitted.clone();
        explicit["sbd"]["joint_output"] = "factorized".into();
        explicit["sbd"]["joint_execution"] = "tensor".into();
        explicit["sbd"]["commitment"] = "fixed".into();
        explicit["sbd"]["dependence_threshold"] = 0.05.into();
        let explicit: Config = serde_json::from_value(explicit).unwrap();
        assert_eq!(serde_json::to_value(&explicit).unwrap(),omitted);
        assert_eq!(explicit.training_fingerprint(c.train_tokens).unwrap(),training);
        assert_eq!(explicit.pilot_fingerprint().unwrap(),pilot);
        c.sbd.as_mut().unwrap().joint_output = SbdJointOutput::Codpula;
        c.validate().unwrap();
        assert_ne!(c.training_fingerprint(c.train_tokens).unwrap(),training);
        assert_ne!(c.pilot_fingerprint().unwrap(),pilot);
        c.sbd.as_mut().unwrap().commitment = SbdCommitment::Dependence;
        c.validate().unwrap();
        let encoded = serde_json::to_value(&c).unwrap();
        assert_eq!(encoded["sbd"]["joint_output"],"codpula");
        assert_eq!(encoded["sbd"]["commitment"],"dependence");
        assert_eq!(serde_json::to_value(serde_json::from_value::<Config>(encoded.clone()).unwrap()).unwrap(),encoded);
        c.sbd.as_mut().unwrap().joint_output = SbdJointOutput::Factorized;
        assert!(c.validate().is_err());
        c.sbd.as_mut().unwrap().joint_output = SbdJointOutput::Codpula;
        c.sbd.as_mut().unwrap().block_lengths = vec![17];
        assert!(c.validate().is_err());
        c.sbd.as_mut().unwrap().block_lengths = vec![1,4,16];
        for invalid in [0.0,-0.1,f64::NAN,f64::INFINITY] {
            c.sbd.as_mut().unwrap().dependence_threshold = invalid;
            assert!(c.validate().is_err());
        }
        c.sbd.as_mut().unwrap().dependence_threshold = 0.05;
        c.sbd.as_mut().unwrap().joint_execution = crate::codpula::CodpulaExecution::FusedGpu;
        assert!(c.validate().is_err());
        c.backend = "vulkan".into();
        c.validate().unwrap();
        c.fusion = true;
        assert!(c.validate().is_err());
        for field in ["joint_output","joint_execution","commitment"] {
            let mut invalid = omitted.clone();
            invalid["sbd"][field] = "automatic".into();
            assert!(serde_json::from_value::<Config>(invalid).is_err());
        }
    }
}
