//! Explicit short-convolution execution choices; omission preserves historical arithmetic.
use burn::prelude::*;
use serde::{Deserialize, Serialize};
#[path = "convolution_checks.rs"]
mod checks;
pub use checks::numerical_checks;
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConvolutionBackend {
    #[default]
    Tensor,
    PaddedTensor,
    FusedGpu,
}
impl ConvolutionBackend {
    pub fn is_tensor(&self) -> bool {
        *self == Self::Tensor
    }
    pub fn forward(
        self,
        x: Tensor<3>,
        weight: Tensor<2>,
        valid: Tensor<2>,
        documents: Tensor<2>,
    ) -> Tensor<3> {
        match self {
            Self::Tensor => crate::causal_conv(x, weight, valid, documents),
            Self::PaddedTensor => padded(x, weight, valid, documents),
            Self::FusedGpu => {
                #[cfg(any(feature = "vulkan", feature = "cuda"))]
                {
                    delta_kernels::causal_conv(x, weight, valid, documents)
                }
                #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
                {
                    let _ = (x, weight, valid, documents);
                    panic!("custom convolution requires GPU feature")
                }
            }
        }
    }
}
fn padded(x: Tensor<3>, weight: Tensor<2>, valid: Tensor<2>, documents: Tensor<2>) -> Tensor<3> {
    let [b, t, c] = x.dims();
    let mut output = None;
    for lag in 0..weight.dims()[0].min(t) {
        let n = t - lag;
        let same = documents
            .clone()
            .slice([0..b, lag..t])
            .equal(documents.clone().slice([0..b, 0..n]))
            .float();
        let active = same * valid.clone().slice([0..b, 0..n]) * valid.clone().slice([0..b, lag..t]);
        let contribution = x.clone().slice([0..b, 0..n, 0..c])
            * weight
                .clone()
                .slice([lag..lag + 1, 0..c])
                .reshape([1, 1, c])
            * active.unsqueeze_dim::<3>(2);
        let contribution = if lag == 0 {
            contribution
        } else {
            Tensor::cat(
                vec![Tensor::zeros([b, lag, c], &x.device()), contribution],
                1,
            )
        };
        output = Some(match output {
            None => contribution,
            Some(old) => old + contribution,
        });
    }
    output.expect("nonempty convolution")
}
#[derive(Serialize)]
pub struct Benchmark {
    layers: usize,
    channels: [usize; 3],
    results: Vec<BenchmarkRow>,
}
#[derive(Serialize)]
struct BenchmarkRow {
    convolution_backend: ConvolutionBackend,
    forward_median_ms: f64,
    forward_backward_median_ms: f64,
    forward_samples_ms: Vec<f64>,
    forward_backward_samples_ms: Vec<f64>,
}
/// Resident q/k/v fixtures, four actual layers, independent leaf VJPs, alternating F/FB.
pub fn benchmark(
    device: &Device,
    tokens: usize,
    batch: usize,
    warmups: usize,
    trials: usize,
) -> Result<Benchmark, String> {
    use std::time::Instant;
    let device = device.clone().autodiff();
    device.seed(731);
    let fixtures: [_; 4] = core::array::from_fn(|_| {
        [128, 128, 256].map(|c| {
            (
                Tensor::<3>::random(
                    [batch, tokens, c],
                    burn::tensor::Distribution::Uniform(-0.5, 0.5),
                    &device,
                ),
                Tensor::<2>::random(
                    [4, c],
                    burn::tensor::Distribution::Uniform(-0.1, 0.1),
                    &device,
                ),
            )
        })
    });
    let valid = Tensor::ones([batch, tokens], &device);
    let docs = Tensor::zeros([batch, tokens], &device);
    let mut results = Vec::new();
    for backend in [
        SelfBackend::Tensor,
        SelfBackend::PaddedTensor,
        SelfBackend::FusedGpu,
    ] {
        let mut samples: [Vec<f64>; 2] = core::array::from_fn(|_| Vec::new());
        for iteration in 0..warmups + trials {
            for mode in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                device.sync().map_err(|e| format!("{e:?}"))?;
                let start = Instant::now();
                for layer in &fixtures {
                    let leaves = layer.each_ref().map(|(x, w)| {
                        (
                            x.clone().detach().require_grad(),
                            w.clone().detach().require_grad(),
                        )
                    });
                    let roots = leaves.each_ref().map(|(x, w)| {
                        backend
                            .forward(x.clone(), w.clone(), valid.clone(), docs.clone())
                            .sum()
                    });
                    let root = roots.into_iter().reduce(|a, b| a + b).unwrap();
                    if mode == 1 {
                        let grads = root.backward();
                        for (x, w) in leaves {
                            let _ = (
                                x.grad(&grads).ok_or("missing input VJP")?,
                                w.grad(&grads).ok_or("missing weight VJP")?,
                            );
                        }
                    }
                }
                device.sync().map_err(|e| format!("{e:?}"))?;
                if iteration >= warmups {
                    samples[mode].push(start.elapsed().as_secs_f64() * 1000.);
                }
            }
        }
        let median = |s: &[f64]| {
            let mut v = s.to_vec();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let [forward_samples_ms, forward_backward_samples_ms] = samples;
        results.push(BenchmarkRow {
            convolution_backend: backend,
            forward_median_ms: median(&forward_samples_ms),
            forward_backward_median_ms: median(&forward_backward_samples_ms),
            forward_samples_ms,
            forward_backward_samples_ms,
        });
    }
    Ok(Benchmark {
        layers: 4,
        channels: [128, 128, 256],
        results,
    })
}
use ConvolutionBackend as SelfBackend;
