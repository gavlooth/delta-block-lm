//! Explicit reduction-only versus reduction-plus-gating scopes; historical tensor path is default.
use burn::{nn::RmsNorm, prelude::*};
use serde::{Deserialize, Serialize};
#[path = "normalization_checks.rs"]
mod checks;
pub use checks::numerical_checks;
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationBackend {
    #[default]
    Tensor,
    /// Custom L2 and learned RMS reductions, leaving elementwise gates separate.
    FusedNorms,
    /// Also fuse sigmoid/output multiply into RMSNorm and softplus/exp decay.
    FusedNormsGates,
}
impl NormalizationBackend {
    pub fn is_tensor(&self) -> bool {
        *self == Self::Tensor
    }
    pub fn l2(self, x: Tensor<4>) -> Tensor<4> {
        if self.is_tensor() {
            // Floor before sqrt so the zero-vector backward never evaluates 0/0.
            return x.clone() / x.square().sum_dim(3).clamp_min(1e-24).sqrt();
        }
        #[cfg(any(feature = "vulkan", feature = "cuda"))]
        {
            delta_kernels::l2_norm(x)
        }
        #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
        {
            let _ = x;
            panic!("custom normalization requires a GPU feature")
        }
    }
    pub fn rms<const D: usize>(self, norm: &RmsNorm, x: Tensor<D>) -> Tensor<D> {
        if self.is_tensor() {
            return norm.forward(x);
        }
        #[cfg(any(feature = "vulkan", feature = "cuda"))]
        {
            delta_kernels::rms_norm(x, norm.gamma.val(), None, norm.epsilon)
        }
        #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
        {
            let _ = (norm, x);
            panic!("custom normalization requires a GPU feature")
        }
    }
    pub fn gated_rms(self, norm: &RmsNorm, x: Tensor<4>, gate: Tensor<4>) -> Tensor<4> {
        if self != Self::FusedNormsGates {
            return self.rms(norm, x * burn::tensor::activation::sigmoid(gate));
        }
        #[cfg(any(feature = "vulkan", feature = "cuda"))]
        {
            delta_kernels::rms_norm(x, norm.gamma.val(), Some(gate), norm.epsilon)
        }
        #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
        {
            let _ = (norm, x, gate);
            panic!("custom normalization requires a GPU feature")
        }
    }
    pub fn decay(self, x: Tensor<4>, bias: Tensor<1>, scale: Tensor<1>) -> Tensor<4> {
        if self != Self::FusedNormsGates {
            let [_, _, h, k] = x.dims();
            return (burn::tensor::activation::softplus(x + bias.reshape([1, 1, h, k]), 1.)
                * scale.exp().reshape([1, 1, h, 1]))
            .neg()
            .exp();
        }
        #[cfg(any(feature = "vulkan", feature = "cuda"))]
        {
            delta_kernels::decay(x, bias, scale)
        }
        #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
        {
            let _ = (x, bias, scale);
            panic!("custom normalization requires a GPU feature")
        }
    }
}

/// Resident representative per-block normalization/gating graph. No projections,
/// convolutions, recurrence, HOLA read, optimizer or safety checks are timed here.
pub fn benchmark(
    device: &Device,
    tokens: usize,
    batch: usize,
    warmups: usize,
    trials: usize,
) -> Result<impl Serialize + use<>, String> {
    use burn::{nn::RmsNormConfig, tensor::Distribution};
    use std::time::Instant;
    #[derive(Serialize)]
    struct Row {
        normalization_backend: NormalizationBackend,
        forward_samples_ms: Vec<f64>,
        forward_backward_samples_ms: Vec<f64>,
        forward_median_ms: f64,
        forward_backward_median_ms: f64,
    }
    let device = device.clone().autodiff();
    let modes = [
        NormalizationBackend::Tensor,
        NormalizationBackend::FusedNorms,
        NormalizationBackend::FusedNormsGates,
    ];
    device.seed(2917);
    let raw = [32, 32, 64, 32, 64, 64, 256, 256].map(|w| {
        Tensor::<4>::random(
            [batch, tokens, if w == 256 { 1 } else { 4 }, w],
            Distribution::Uniform(-0.5, 0.5),
            &device,
        )
        .require_grad()
    });
    let qnorm = RmsNormConfig::new(32).init(&device);
    let knorm = RmsNormConfig::new(32).init(&device);
    let outnorm = RmsNormConfig::new(64).init(&device);
    let blocknorm = RmsNormConfig::new(256).with_epsilon(1e-6).init(&device);
    let ffnorm = RmsNormConfig::new(256).with_epsilon(1e-6).init(&device);
    let bias = Tensor::<1>::full([128], -4., &device).require_grad();
    let scale = Tensor::<1>::zeros([4], &device).require_grad();
    let run = |mode: NormalizationBackend| {
        vec![
            mode.l2(raw[0].clone()),
            mode.l2(raw[1].clone()),
            mode.l2(raw[2].clone()),
            mode.rms(&qnorm, raw[0].clone()),
            mode.rms(&knorm, raw[1].clone()),
            mode.decay(raw[3].clone(), bias.clone(), scale.clone()),
            mode.gated_rms(&outnorm, raw[4].clone(), raw[5].clone()),
            mode.rms(&blocknorm, raw[6].clone()),
            mode.rms(&ffnorm, raw[7].clone()),
        ]
    };
    let mut forward: [Vec<f64>; 3] = core::array::from_fn(|_| Vec::new());
    let mut backward: [Vec<f64>; 3] = core::array::from_fn(|_| Vec::new());
    for iteration in 0..warmups + trials {
        let mut order = [iteration % 3, (iteration + 1) % 3, (iteration + 2) % 3];
        if iteration % 2 == 1 {
            order.reverse();
        }
        for index in order {
            device.sync().map_err(|e| format!("{e:?}"))?;
            let start = Instant::now();
            let y = run(modes[index]);
            device.sync().map_err(|e| format!("{e:?}"))?;
            let f = start.elapsed().as_secs_f64() * 1000.;
            drop(y);
            device.sync().map_err(|e| format!("{e:?}"))?;
            let start = Instant::now();
            let objective = run(modes[index])
                .into_iter()
                .map(|x| x.square().sum())
                .reduce(|a, b| a + b)
                .unwrap();
            let grads = objective.backward();
            device.sync().map_err(|e| format!("{e:?}"))?;
            let fb = start.elapsed().as_secs_f64() * 1000.;
            std::hint::black_box(grads);
            if iteration >= warmups {
                forward[index].push(f);
                backward[index].push(fb);
            }
        }
    }
    fn median(x: &[f64]) -> f64 {
        let mut x = x.to_vec();
        x.sort_by(f64::total_cmp);
        if x.len() % 2 == 1 {
            x[x.len() / 2]
        } else {
            (x[x.len() / 2 - 1] + x[x.len() / 2]) / 2.
        }
    }
    Ok(modes
        .into_iter()
        .enumerate()
        .map(|(i, mode)| Row {
            normalization_backend: mode,
            forward_median_ms: median(&forward[i]),
            forward_backward_median_ms: median(&backward[i]),
            forward_samples_ms: forward[i].clone(),
            forward_backward_samples_ms: backward[i].clone(),
        })
        .collect::<Vec<_>>())
}
