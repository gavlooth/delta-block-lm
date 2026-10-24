use anyhow::{Context, Result, ensure};
use delta_mixers::{
    CachePolicy, HolaBackend, MixerConfig, MixerKind, NormalizationBackend, ProjectionBackend,
    ScanBackend,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

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
            c.scan_backend != ScanBackend::FusedGpu || kind == MixerKind::KronLstm,
            "fused_gpu is KronLSTM-only; use tensor_solver for GDN-2"
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
        })
    }
    pub fn training_algorithm(&self, c: &Config) -> &'static str {
        let fused_hola = self.cache != "none" && c.hola_backend == HolaBackend::FusedGpu;
        match (c.scan_backend == ScanBackend::FusedGpu, fused_hola) {
            (false, false) => "differentiable_chunk_tensor_solver",
            (true, false) => "fused_kron_gpu",
            (false, true) => "tensor_solver_fused_hola_gpu",
            (true, true) => "fused_kron_and_hola_gpu",
        }
    }
    pub fn uses_custom_gpu(&self, c: &Config) -> bool {
        c.scan_backend == ScanBackend::FusedGpu
            || (self.cache != "none" && c.hola_backend == HolaBackend::FusedGpu)
            || !c.normalization_backend.is_tensor()
            || !c.update_backend.is_tensor()
            || !c.loss_backend.is_tensor()
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

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
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
    #[serde(skip_serializing_if = "HeadBackend::is_native")]
    pub head_backend: HeadBackend,
    #[serde(skip_serializing_if = "UpdateBackend::is_tensor")]
    pub update_backend: UpdateBackend,
    #[serde(skip_serializing_if = "LossBackend::is_tensor")]
    pub loss_backend: LossBackend,
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
    pub soft_steps: usize,
    pub soft_eta: f64,
    pub soft_temperature: f64,
    pub hard_mask_only: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
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
            head_backend: HeadBackend::Native,
            update_backend: UpdateBackend::Tensor,
            loss_backend: LossBackend::Tensor,
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
            self.head_backend.is_native() || self.task == "lm",
            "experimental head paths require the byte-LM vocabulary"
        );
        ensure!(
            self.head_backend != HeadBackend::Heuristic
                || matches!(self.backend.as_str(), "vulkan" | "cuda"),
            "heuristic head requires a GPU backend"
        );
        ensure!(
            self.context >= 8 && self.batch_size > 0 && self.layers > 0 && self.d_model > 0,
            "invalid model shape"
        );
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
            (self.scan_backend != ScanBackend::FusedGpu
                && self.hola_backend != HolaBackend::FusedGpu
                && self.normalization_backend.is_tensor())
                || matches!(self.backend.as_str(), "vulkan" | "cuda"),
            "fused_gpu requires an explicit GPU backend; no CPU fallback"
        );
        ensure!(
            self.hola_backend != HolaBackend::FusedGpu
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
    pub fn pilot_fingerprint(&self) -> Result<String> {
        Ok(crate::data::hash(&serde_json::to_vec(
            &serde_json::json!({"config":self,"precision":"float32","training_algorithm":self.scan_backend.training_algorithm(),"optimizer":"AdamW"}),
        )?))
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
    fn legacy_selector_defaults_without_changing_serialized_identity() {
        let config = Config::default();
        let legacy = serde_json::to_value(&config).unwrap();
        assert!(legacy.get("scan_backend").is_none());
        assert!(legacy.get("hola_backend").is_none());
        assert!(legacy.get("fusion").is_none());
        assert!(legacy.get("autotune").is_none());
        assert!(legacy.get("projection_backend").is_none());
        assert!(legacy.get("normalization_backend").is_none());
        assert!(legacy.get("head_backend").is_none());
        assert!(legacy.get("update_backend").is_none());
        assert!(legacy.get("loss_backend").is_none());
        let decoded: Config = serde_json::from_value(legacy.clone()).unwrap();
        assert_eq!(decoded.scan_backend, ScanBackend::TensorSolver);
        assert_eq!(decoded.hola_backend, HolaBackend::Tensor);
        assert!(!decoded.fusion);
        assert!(!decoded.autotune);
        assert_eq!(decoded.projection_backend, ProjectionBackend::Separate);
        assert_eq!(decoded.normalization_backend, NormalizationBackend::Tensor);
        assert_eq!(decoded.head_backend, HeadBackend::Native);
        assert_eq!(decoded.update_backend, UpdateBackend::Tensor);
        assert_eq!(decoded.loss_backend, LossBackend::Tensor);
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
    fn fused_selectors_reject_cpu_and_gdn_scan_without_fallback() {
        let mut config = Config::default();
        config.scan_backend = ScanBackend::FusedGpu;
        assert!(config.validate().is_err());
        config.backend = "vulkan".into();
        assert!(config.arms[0].mixer_config(&config).is_err());
        config.scan_backend = ScanBackend::TensorSolver;
        config.hola_backend = HolaBackend::FusedGpu;
        config.backend = "cpu".into();
        assert!(config.validate().is_err());
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
}
