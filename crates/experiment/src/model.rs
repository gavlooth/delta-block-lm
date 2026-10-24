use crate::{
    config::{Arm, Config, HeadBackend, LossBackend, UpdateBackend},
    data::Batch,
};
use anyhow::{Result, ensure};
use burn::{
    module::{Initializer, ModuleVisitor, Param},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    optim::GradientsParams,
    prelude::*,
    tensor::activation::{log_softmax, silu},
};
use delta_mixers::{
    Mixer,
    profile::{Sections, Unprofiled},
};

#[derive(Module, Debug)]
pub struct Block {
    pub(super) norm: RmsNorm,
    pub(super) mixer: Mixer,
    pub(super) ff_norm: RmsNorm,
    pub(super) up: Linear,
    pub(super) down: Linear,
}
#[derive(Module, Debug)]
pub struct LanguageModel {
    pub(super) embedding: Embedding,
    pub(super) blocks: Vec<Block>,
    pub(super) norm: RmsNorm,
    pub(super) head: Linear,
    #[module(skip)]
    pub(super) normalization_backend: delta_mixers::NormalizationBackend,
    #[module(skip)]
    pub(super) head_backend: HeadBackend,
    #[module(skip)]
    pub(super) loss_backend: LossBackend,
    #[module(skip)]
    pub(super) update_backend: UpdateBackend,
}
impl LanguageModel {
    pub fn new(c: &Config, arm: &Arm, device: &Device) -> Result<Self> {
        let init = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };
        let linear = |a, b| {
            LinearConfig::new(a, b)
                .with_bias(false)
                .with_initializer(init.clone())
                .init(device)
        };
        let norm = || {
            RmsNormConfig::new(c.d_model)
                .with_epsilon(1e-6)
                .init(device)
        };
        let mut blocks = Vec::with_capacity(c.layers);
        for _ in 0..c.layers {
            blocks.push(Block {
                norm: norm(),
                mixer: arm.mixer_config(c)?.init(device),
                ff_norm: norm(),
                up: linear(c.d_model, 2 * c.d_model),
                down: linear(2 * c.d_model, c.d_model),
            });
        }
        Ok(Self {
            embedding: EmbeddingConfig::new(c.vocab(), c.d_model)
                .with_initializer(init.clone())
                .init(device),
            blocks,
            norm: norm(),
            head: linear(c.d_model, c.vocab()),
            normalization_backend: c.normalization_backend,
            head_backend: c.head_backend,
            loss_backend: c.loss_backend,
            update_backend: c.update_backend,
        })
    }
    pub fn set_scan_backend(&mut self, backend: delta_mixers::ScanBackend) {
        for block in &mut self.blocks {
            block.mixer.set_scan_backend(backend);
        }
    }
    pub fn set_hola_backend(&mut self, backend: delta_mixers::HolaBackend) {
        for block in &mut self.blocks {
            block.mixer.set_hola_backend(backend);
        }
    }
    pub fn set_projection_backend(&mut self, backend: delta_mixers::ProjectionBackend) {
        for block in &mut self.blocks {
            block.mixer.set_projection_backend(backend);
        }
    }
    pub fn set_normalization_backend(&mut self, backend: delta_mixers::NormalizationBackend) {
        self.normalization_backend = backend;
        for block in &mut self.blocks {
            block.mixer.set_normalization_backend(backend);
        }
    }
    pub fn set_head_backend(&mut self, backend: HeadBackend) {
        self.head_backend = backend;
    }
    pub fn set_loss_backend(&mut self, backend: LossBackend) {
        self.loss_backend = backend;
    }
    pub fn set_update_backend(&mut self, backend: UpdateBackend) {
        self.update_backend = backend;
    }
    pub fn loss(&self, logits: Tensor<3>, batch: &Batch) -> Tensor<1> {
        loss_selected(logits, batch, self.loss_backend)
    }
    pub(super) fn head_forward(&self, hidden: Tensor<3>) -> Tensor<3> {
        match self.head_backend {
            HeadBackend::Native => self.head.forward(hidden),
            HeadBackend::Padded320 => {
                let [b, t, _] = hidden.dims();
                let weight = self.head.weight.val();
                let [width, vocab] = weight.dims();
                assert_eq!(vocab, 257);
                // The 63 added columns are constants, never parameters or optimizer state.
                let padding = Tensor::<2>::zeros([width, 320 - vocab], &hidden.device());
                let padded = Tensor::cat(vec![weight, padding], 1);
                burn::tensor::module::linear(hidden, padded, None).slice([0..b, 0..t, 0..vocab])
            }
            HeadBackend::Heuristic => {
                let [b, t, width] = hidden.dims();
                let weight = self.head.weight.val();
                let vocab = weight.dims()[1];
                heuristic_matmul(hidden.reshape([b * t, width]), weight).reshape([b, t, vocab])
            }
        }
    }
    pub fn forward(&self, batch: &Batch) -> Tensor<3> {
        self.forward_profiled(batch, &mut Unprofiled)
    }
    pub fn forward_profiled<P: Sections>(&self, batch: &Batch, profile: &mut P) -> Tensor<3> {
        let mut hidden =
            profile.section("Embedding", || self.embedding.forward(batch.input.clone()));
        for (index, block) in self.blocks.iter().enumerate() {
            profile.block(Some(index));
            let normalized = profile.section("Block RMSNorm: mixer", || {
                self.normalization_backend.rms(&block.norm, hidden.clone())
            });
            let update = block.mixer.forward_profiled(
                normalized,
                batch.valid.clone(),
                batch.reset.clone(),
                profile,
            );
            hidden = profile.section("Residual adds", || hidden + update);
            let normalized = profile.section("Block RMSNorm: MLP", || {
                self.normalization_backend
                    .rms(&block.ff_norm, hidden.clone())
            });
            let update = profile.section("MLP: up", || block.up.forward(normalized));
            let update = profile.section("MLP: SiLU", || silu(update));
            let update = profile.section("MLP: down", || block.down.forward(update));
            hidden = profile.section("Residual adds", || hidden + update);
        }
        profile.block(None);
        profile.section("Final norm and vocabulary head", || {
            self.head_forward(self.normalization_backend.rms(&self.norm, hidden))
        })
    }
}

pub(super) fn heuristic_matmul(a: Tensor<2>, b: Tensor<2>) -> Tensor<2> {
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    {
        delta_kernels::heuristic_matmul(a, b)
    }
    #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
    {
        let _ = (a, b);
        panic!("heuristic matmul requires a GPU feature")
    }
}
pub fn loss_selected(logits: Tensor<3>, batch: &Batch, backend: LossBackend) -> Tensor<1> {
    let [b, t, v] = logits.dims();
    let chosen = if backend.is_tensor() {
        log_softmax(logits.reshape([b * t, v]), 1)
            .gather(1, batch.target.clone().reshape([b * t, 1]))
            .reshape([b, t])
            .neg()
    } else {
        #[cfg(any(feature = "vulkan", feature = "cuda"))]
        {
            delta_kernels::cross_entropy_rows(
                logits.reshape([b * t, v]),
                batch.target.clone().reshape([b * t]).float(),
                if backend == LossBackend::Fused64 {
                    64
                } else {
                    256
                },
            )
            .reshape([b, t])
        }
        #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
        {
            panic!("fused cross entropy requires a GPU feature")
        }
    };
    (chosen * batch.weight.clone()).sum() / batch.weight.clone().sum()
}
pub fn scalar(tensor: Tensor<1>) -> Result<f64> {
    let data = tensor
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("reading scalar: {e:?}"))?;
    let value = data[0] as f64;
    ensure!(value.is_finite(), "nonfinite loss/state");
    Ok(value)
}
/// Check every gradient before optimizer writes, not only the scalar loss.
pub fn checked_gradients(model: &LanguageModel, loss: Tensor<1>) -> Result<GradientsParams> {
    checked_gradients_profiled(model, loss, &mut Unprofiled)
}
pub fn checked_gradients_profiled<P: Sections>(
    model: &LanguageModel,
    loss: Tensor<1>,
    profile: &mut P,
) -> Result<GradientsParams> {
    let grads = profile.section("Backward (total only)", || {
        GradientsParams::from_grads(loss.backward(), model)
    });
    check_parameter_gradients(model, &grads, profile)?;
    Ok(grads)
}
pub(super) fn check_parameter_gradients<P: Sections>(
    model: &LanguageModel,
    grads: &GradientsParams,
    profile: &mut P,
) -> Result<()> {
    struct Check<'a> {
        grads: &'a GradientsParams,
        missing: usize,
        norms: Vec<Tensor<1>>,
        backend: UpdateBackend,
    }
    impl ModuleVisitor for Check<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            if let Some(g) = self.grads.get::<D>(param.id) {
                // A squared sum catches NaN and overflow; reduce once on device per parameter.
                self.norms.push(if self.backend.is_tensor() {
                    g.square().sum()
                } else {
                    crate::update::square_sum(g)
                });
            } else {
                self.missing += 1;
            }
        }
    }
    let total = profile.section("Gradient checks: reductions", || -> Result<Tensor<1>> {
        let mut check = Check {
            grads,
            missing: 0,
            norms: Vec::new(),
            backend: model.update_backend,
        };
        model.visit(&mut check);
        ensure!(
            check.missing == 0,
            "{} trainable parameters have no gradient: refusing partial/readout-only training",
            check.missing
        );
        Ok(Tensor::cat(check.norms, 0).sum())
    })?;
    let finite = profile
        .section("Host/device: gradient norm readback", || scalar(total))
        .is_ok();
    ensure!(finite, "nonfinite parameter gradients");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data;

    struct Parameters(Vec<Vec<f32>>);
    impl ModuleVisitor for Parameters {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            self.0
                .push(param.val().into_data().try_to_vec::<f32>().unwrap());
        }
    }
    #[test]
    fn padded_head_preserves_parameters_logits_and_vjps() {
        let mut c = Config::default();
        c.task = "lm".into();
        c.d_model = 8;
        c.layers = 1;
        let device = Device::ndarray().autodiff();
        device.seed(3107);
        let mut model = LanguageModel::new(&c, &c.arms[0], &device).unwrap();
        let hidden = Tensor::<3>::random(
            [2, 3, 8],
            burn::tensor::Distribution::Uniform(-0.1, 0.1),
            &device,
        )
        .require_grad();
        let weight = model.head.weight.val();
        let count = model.num_params();
        let original = model.head_forward(hidden.clone());
        let grads = original.clone().sum().backward();
        let dx = hidden.grad(&grads).unwrap();
        let dw = weight.grad(&grads).unwrap();
        model.set_head_backend(HeadBackend::Padded320);
        let padded = model.head_forward(hidden.clone());
        let grads = padded.clone().sum().backward();
        assert_eq!(count, model.num_params());
        assert_eq!(model.head.weight.dims(), [8, 257]);
        for (a, b) in [
            ((original - padded).abs().max(), 2e-6),
            ((dx - hidden.grad(&grads).unwrap()).abs().max(), 2e-6),
            ((dw - weight.grad(&grads).unwrap()).abs().max(), 2e-6),
        ] {
            assert!(a.into_data().try_to_vec::<f32>().unwrap()[0] < b);
        }
    }
    #[test]
    fn full_mixer_training_and_optimizer_resume_match() {
        let mut c = Config::default();
        c.d_model = 8;
        c.layers = 1;
        c.context = 16;
        c.association_counts = vec![2];
        c.batch_size = 1;
        let device = Device::ndarray().autodiff();
        device.seed(719);
        let arm = &c.arms[0];
        let mut model = LanguageModel::new(&c, arm, &device).unwrap();
        let batch = Batch::new(&[data::recall(&c, 1, 0, false)], &device);
        let mut before = Parameters(Vec::new());
        model.blocks[0].mixer.visit(&mut before);
        let mut optim = crate::train::optimizer(&c);
        let objective = model.loss(model.forward(&batch), &batch);
        let grads = checked_gradients(&model, objective).unwrap();
        model = optim.step(0.001, model, grads);
        let mut after = Parameters(Vec::new());
        model.blocks[0].mixer.visit(&mut after);
        assert!(
            before
                .0
                .iter()
                .zip(&after.0)
                .any(|(a, b)| a.iter().zip(b).any(|(a, b)| (a - b).abs() > 1e-8)),
            "mixer weights did not train"
        );
        let path =
            std::env::temp_dir().join(format!("delta-experiment-resume-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        model
            .clone()
            .save_file(path.join("model.burnpack"))
            .unwrap();
        optim.save(path.join("optimizer.burnpack")).unwrap();
        let restored = crate::checkpoint::load_model(
            LanguageModel::new(&c, arm, &device).unwrap(),
            path.join("model.burnpack"),
            &device,
        )
        .unwrap();
        let mut restored_optim =
            crate::checkpoint::load_optimizer(&c, path.join("optimizer.burnpack")).unwrap();
        let grads = checked_gradients(&model, model.loss(model.forward(&batch), &batch)).unwrap();
        let restored_grads =
            checked_gradients(&restored, restored.loss(restored.forward(&batch), &batch)).unwrap();
        let model = optim.step(0.001, model, grads);
        let restored = restored_optim.step(0.001, restored, restored_grads);
        let difference = (model.forward(&batch) - restored.forward(&batch))
            .abs()
            .max()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()[0];
        assert!(
            difference < 2e-6,
            "next resumed AdamW step diverged: {difference}"
        );
        std::fs::remove_dir_all(path).unwrap();
    }
}
