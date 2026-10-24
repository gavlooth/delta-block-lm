//! Execution-only weight packing: the seven original Linear parameters remain the leaves.
use crate::Mixer;
use burn::{nn::Linear, prelude::*};
use serde::{Deserialize, Serialize};
use std::time::Instant;

const MODES: [ProjectionBackend; 3] = [
    ProjectionBackend::Separate,
    ProjectionBackend::Grouped,
    ProjectionBackend::Merged,
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionBackend {
    #[default]
    Separate,
    /// Three contractions: Q/K/decay, V/output gate, and erase/write.
    Grouped,
    /// One contraction for all seven input projections; output stays dependent on the scan.
    Merged,
    /// Merged inputs, with the dependent output contraction pinned to the library heuristic.
    MergedHeuristicOutput,
}
impl ProjectionBackend {
    pub fn is_separate(&self) -> bool {
        *self == Self::Separate
    }
}

fn project<const N: usize>(input: Tensor<3>, layers: [&Linear; N]) -> [Tensor<3>; N] {
    let [batch, tokens, _] = input.dims();
    let widths = layers.map(|layer| layer.weight.dims()[1]);
    let weight = Tensor::cat(layers.map(|layer| layer.weight.val()).into(), 1);
    let output = burn::tensor::module::linear(input, weight, None);
    let mut first = 0;
    core::array::from_fn(|index| {
        let end = first + widths[index];
        let result = output.clone().slice([0..batch, 0..tokens, first..end]);
        first = end;
        result
    })
}

impl Mixer {
    /// Changes contractions only, preserving parameter names, shapes, IDs and optimizer state.
    pub fn set_projection_backend(&mut self, backend: ProjectionBackend) {
        if backend == ProjectionBackend::MergedHeuristicOutput {
            crate::chunk::assert_fused_device(&self.output.weight.val().device());
        }
        self.projection_backend = backend;
    }

    pub(crate) fn project_input<P: crate::profile::Sections>(
        &self,
        input: Tensor<3>,
        profile: &mut P,
    ) -> [Option<Tensor<3>>; 7] {
        match self.projection_backend {
            ProjectionBackend::Separate => core::array::from_fn(|_| None),
            ProjectionBackend::Grouped => {
                let [q, k, decay] = profile.section("Projection: grouped Q/K/decay", || {
                    project(input.clone(), [&self.q, &self.k, &self.decay])
                });
                let [v, gate] = profile.section("Projection: grouped V/output_gate", || {
                    project(input.clone(), [&self.v, &self.output_gate])
                });
                let [erase, write] = profile.section("Projection: grouped erase/write", || {
                    project(input, [&self.erase, &self.write])
                });
                [q, k, v, decay, erase, write, gate].map(Some)
            }
            ProjectionBackend::Merged | ProjectionBackend::MergedHeuristicOutput => profile
                .section("Projection: merged input", || {
                    project(
                        input,
                        [
                            &self.q,
                            &self.k,
                            &self.v,
                            &self.decay,
                            &self.erase,
                            &self.write,
                            &self.output_gate,
                        ],
                    )
                    .map(Some)
                }),
        }
    }

    pub(crate) fn project_output(&self, input: Tensor<3>) -> Tensor<3> {
        if self.projection_backend != ProjectionBackend::MergedHeuristicOutput {
            return self.output.forward(input);
        }
        #[cfg(any(feature = "vulkan", feature = "cuda"))]
        {
            let [batch, tokens, width] = input.dims();
            let weight = self.output.weight.val();
            let output = weight.dims()[1];
            delta_kernels::heuristic_matmul(input.reshape([batch * tokens, width]), weight)
                .reshape([batch, tokens, output])
        }
        #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
        {
            let _ = input;
            panic!("heuristic output projection requires a GPU feature")
        }
    }
}

fn outputs(mixer: &Mixer, input: Tensor<3>) -> [Tensor<3>; 7] {
    let packed = mixer.project_input(input.clone(), &mut crate::profile::Unprofiled);
    let layers = [
        &mixer.q,
        &mixer.k,
        &mixer.v,
        &mixer.decay,
        &mixer.erase,
        &mixer.write,
        &mixer.output_gate,
    ];
    let mut packed = packed.into_iter();
    layers.map(|layer| {
        packed
            .next()
            .unwrap()
            .unwrap_or_else(|| layer.forward(input.clone()))
    })
}

fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .into_data()
        .try_to_vec::<f32>()
        .expect("f32 projection check")
}

fn compare(actual: &[f32], expected: &[f64]) -> Result<f64, String> {
    if actual.len() != expected.len() {
        return Err("projection element count".into());
    }
    let mut maximum = 0.0f64;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual as f64 - expected).abs();
        if !actual.is_finite() || !expected.is_finite() || error > 2e-5 + 3e-4 * expected.abs() {
            return Err(format!(
                "projection[{index}] {actual} != {expected}, abs error {error}"
            ));
        }
        maximum = maximum.max(error);
    }
    Ok(maximum)
}

/// Independent scalar-f64 dot products and analytic input/every-weight VJPs.
/// Includes both gate-width conventions, strided input, tail widths and zeros.
pub fn numerical_checks(device: &Device) -> Result<impl Serialize + use<>, String> {
    #[derive(Serialize)]
    struct Case {
        kind: crate::MixerKind,
        zero: bool,
        projection_backend: ProjectionBackend,
        batch: usize,
        tokens: usize,
        strided_input: bool,
        max_abs_error: f64,
        all_seven_weight_and_input_gradients: bool,
    }
    #[derive(Serialize)]
    struct Report {
        passed: bool,
        absolute_tolerance: f64,
        relative_tolerance: f64,
        cases: Vec<Case>,
    }
    let device = device.clone().autodiff();
    let mut cases = Vec::new();
    for kind in [crate::MixerKind::Gdn2, crate::MixerKind::KronLstm] {
        for zero in [false, true] {
            device.seed(1039);
            let config = crate::MixerConfig {
                kind,
                d_model: 8,
                heads: 2,
                key_dim: 3,
                value_dim: 5,
                chunk_size: 4,
                ..Default::default()
            };
            let base = config.init(&device);
            let (batch, tokens, width) = (2, 7, config.d_model);
            // Swapping the last axes gives a genuinely strided backend input.
            let storage: Vec<f32> = (0..batch * tokens * width)
                .map(|i| {
                    if zero {
                        0.
                    } else {
                        ((i * 13 + 7) as f64 * 0.031).sin() as f32
                    }
                })
                .collect();
            let input =
                Tensor::<3>::from_data(TensorData::new(storage, [batch, width, tokens]), &device)
                    .swap_dims(1, 2)
                    .require_grad();
            let input_values = values(input.clone());
            let layers = [
                &base.q,
                &base.k,
                &base.v,
                &base.decay,
                &base.erase,
                &base.write,
                &base.output_gate,
            ];
            let weights = layers.map(|layer| values(layer.weight.val()));
            let widths = layers.map(|layer| layer.weight.dims()[1]);
            let upstream: [Vec<f32>; 7] = core::array::from_fn(|field| {
                (0..batch * tokens * widths[field])
                    .map(|i| {
                        if zero {
                            0.
                        } else {
                            ((i * 3 + field * 17 + 1) as f64 * 0.027).cos() as f32
                        }
                    })
                    .collect()
            });
            let mut expected_input = vec![0.; input_values.len()];
            let expected_weights: [Vec<f64>; 7] = core::array::from_fn(|field| {
                let mut gradient = vec![0.; weights[field].len()];
                for row in 0..batch * tokens {
                    for k in 0..width {
                        for column in 0..widths[field] {
                            let adjoint = upstream[field][row * widths[field] + column] as f64;
                            gradient[k * widths[field] + column] +=
                                input_values[row * width + k] as f64 * adjoint;
                            expected_input[row * width + k] +=
                                weights[field][k * widths[field] + column] as f64 * adjoint;
                        }
                    }
                }
                gradient
            });
            let expected_output: [Vec<f64>; 7] = core::array::from_fn(|field| {
                (0..batch * tokens * widths[field])
                    .map(|index| {
                        let row = index / widths[field];
                        let column = index % widths[field];
                        (0..width)
                            .map(|k| {
                                input_values[row * width + k] as f64
                                    * weights[field][k * widths[field] + column] as f64
                            })
                            .sum()
                    })
                    .collect()
            });
            for mode in MODES {
                let mut mixer = base.clone();
                mixer.set_projection_backend(mode);
                let output = outputs(&mixer, input.clone());
                let mut maximum = 0.0f64;
                for field in 0..7 {
                    maximum = maximum.max(compare(
                        &values(output[field].clone()),
                        &expected_output[field],
                    )?);
                }
                let loss = output
                    .into_iter()
                    .enumerate()
                    .map(|(field, output)| {
                        (output
                            * Tensor::<3>::from_data(
                                TensorData::new(
                                    upstream[field].clone(),
                                    [batch, tokens, widths[field]],
                                ),
                                &device,
                            ))
                        .sum()
                    })
                    .reduce(|a, b| a + b)
                    .unwrap();
                let grads = loss.backward();
                maximum = maximum.max(compare(
                    &values(input.grad(&grads).ok_or("missing input gradient")?),
                    &expected_input,
                )?);
                let layers = [
                    &mixer.q,
                    &mixer.k,
                    &mixer.v,
                    &mixer.decay,
                    &mixer.erase,
                    &mixer.write,
                    &mixer.output_gate,
                ];
                for field in 0..7 {
                    maximum = maximum.max(compare(
                        &values(
                            layers[field]
                                .weight
                                .val()
                                .grad(&grads)
                                .ok_or("missing weight gradient")?,
                        ),
                        &expected_weights[field],
                    )?);
                    if layers[field].weight.id
                        != [
                            &base.q,
                            &base.k,
                            &base.v,
                            &base.decay,
                            &base.erase,
                            &base.write,
                            &base.output_gate,
                        ][field]
                            .weight
                            .id
                    {
                        return Err("projection packing changed a parameter ID".into());
                    }
                }
                cases.push(Case {
                    kind,
                    zero,
                    projection_backend: mode,
                    batch,
                    tokens,
                    strided_input: true,
                    max_abs_error: maximum,
                    all_seven_weight_and_input_gradients: true,
                });
            }
        }
    }
    Ok(Report {
        passed: true,
        absolute_tolerance: 2e-5,
        relative_tolerance: 3e-4,
        cases,
    })
}

/// Resident seven-input-projection cost, with packing/slices and their autodiff retained.
/// This excludes the dependent output projection, scan, optimizer and safety checks.
pub fn benchmark(
    config: &crate::MixerConfig,
    device: &Device,
    tokens: usize,
    batch: usize,
    warmups: usize,
    trials: usize,
) -> Result<impl Serialize + use<>, String> {
    #[derive(Serialize)]
    struct Row {
        projection_backend: ProjectionBackend,
        forward_samples_ms: Vec<f64>,
        forward_median_ms: f64,
        forward_backward_samples_ms: Vec<f64>,
        forward_backward_median_ms: f64,
    }
    #[derive(Serialize)]
    struct Report {
        context: usize,
        batch: usize,
        results: Vec<Row>,
        scope: &'static str,
    }
    let device = device.clone().autodiff();
    device.seed(1039);
    let base = config.init(&device);
    let input = Tensor::<3>::random(
        [batch, tokens, config.d_model],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &device,
    )
    .require_grad();
    let mixers = MODES.map(|mode| {
        let mut mixer = base.clone();
        mixer.set_projection_backend(mode);
        mixer
    });
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
            let output = outputs(&mixers[index], input.clone());
            device.sync().map_err(|e| format!("{e:?}"))?;
            let ms = start.elapsed().as_secs_f64() * 1000.;
            drop(output);
            device.sync().map_err(|e| format!("{e:?}"))?;
            let start = Instant::now();
            let objective = outputs(&mixers[index], input.clone())
                .into_iter()
                .map(|x| x.square().sum())
                .reduce(|a, b| a + b)
                .unwrap();
            let gradients = objective.backward();
            device.sync().map_err(|e| format!("{e:?}"))?;
            let fb_ms = start.elapsed().as_secs_f64() * 1000.;
            drop(gradients);
            if iteration >= warmups {
                forward[index].push(ms);
                backward[index].push(fb_ms);
            }
        }
    }
    let median = |samples: &[f64]| {
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted[sorted.len() / 2]
    };
    let results = (0..3)
        .map(|index| Row {
            projection_backend: MODES[index],
            forward_median_ms: median(&forward[index]),
            forward_samples_ms: std::mem::take(&mut forward[index]),
            forward_backward_median_ms: median(&backward[index]),
            forward_backward_samples_ms: std::mem::take(&mut backward[index]),
        })
        .collect();
    Ok(Report {
        context: tokens,
        batch,
        results,
        scope: "resident seven input projections, packing and unpacking included; fwd/backward includes sum-of-squares objective and input/every-weight VJPs, excludes optimizer/checks/scan/output projection",
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn packed_projections_match_independent_f64_and_preserve_leaf_ids() {
        super::numerical_checks(&burn::prelude::Device::ndarray()).unwrap();
    }

    #[test]
    fn masked_reset_tail_layers_keep_outputs_and_all_gradients() {
        use super::*;
        use burn::{
            module::{Module, ModuleVisitor, Param},
            optim::GradientsParams,
        };
        struct Snapshot<'a>(&'a GradientsParams, Vec<Vec<f32>>);
        impl ModuleVisitor for Snapshot<'_> {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                self.1.push(values(
                    self.0
                        .get::<D>(param.id)
                        .expect("every layer parameter gradient"),
                ));
            }
        }
        let device = Device::ndarray().autodiff();
        let valid = Tensor::<2>::from_data(
            [
                [1.0f32, 1., 1., 0., 1., 1., 0.],
                [0., 1., 1., 1., 1., 0., 0.],
            ],
            &device,
        );
        let reset = Tensor::<2>::from_data(
            [
                [1.0f32, 0., 1., 0., 0., 0., 1.],
                [0., 1., 0., 0., 1., 0., 0.],
            ],
            &device,
        );
        let input = Tensor::<3>::from_data(
            TensorData::new(
                (0..2 * 7 * 8)
                    .map(|i| ((i * 7 + 1) as f64 * 0.017).sin() as f32)
                    .collect::<Vec<_>>(),
                [2, 8, 7],
            ),
            &device,
        )
        .swap_dims(1, 2)
        .require_grad();
        for kind in [crate::MixerKind::Gdn2, crate::MixerKind::KronLstm] {
            for cache_policy in [
                crate::CachePolicy::None,
                crate::CachePolicy::Surprise,
                crate::CachePolicy::Recency,
            ] {
                device.seed(719);
                let base = crate::MixerConfig {
                    kind,
                    cache_policy,
                    d_model: 8,
                    heads: 2,
                    key_dim: 3,
                    value_dim: 5,
                    chunk_size: 4,
                    cache_capacity: 3,
                    ..Default::default()
                }
                .init(&device);
                let expected = base.forward_masked(input.clone(), valid.clone(), reset.clone());
                let grads = expected.clone().square().sum().backward();
                let input_gradient = values(input.grad(&grads).unwrap())
                    .into_iter()
                    .map(f64::from)
                    .collect::<Vec<_>>();
                let grads = GradientsParams::from_grads(grads, &base);
                let mut snapshot = Snapshot(&grads, Vec::new());
                base.visit(&mut snapshot);
                let expected_gradients = snapshot.1;
                let expected = values(expected)
                    .into_iter()
                    .map(f64::from)
                    .collect::<Vec<_>>();
                for mode in [ProjectionBackend::Grouped, ProjectionBackend::Merged] {
                    let mut candidate = base.clone();
                    candidate.set_projection_backend(mode);
                    let output =
                        candidate.forward_masked(input.clone(), valid.clone(), reset.clone());
                    compare(&values(output.clone()), &expected).unwrap();
                    let grads = output.square().sum().backward();
                    compare(&values(input.grad(&grads).unwrap()), &input_gradient).unwrap();
                    let grads = GradientsParams::from_grads(grads, &candidate);
                    let mut snapshot = Snapshot(&grads, Vec::new());
                    candidate.visit(&mut snapshot);
                    assert_eq!(expected_gradients.len(), snapshot.1.len());
                    for (actual, expected) in snapshot.1.iter().zip(&expected_gradients) {
                        compare(
                            actual,
                            &expected.iter().copied().map(f64::from).collect::<Vec<_>>(),
                        )
                        .unwrap();
                    }
                }
            }
        }
    }
}
