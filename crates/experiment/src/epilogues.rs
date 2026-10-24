//! Execution-only SiLU choices and bounded resident MLP/residual attribution.
use crate::{config::EpilogueBackend, profile::median, runtime};
use anyhow::{Result, ensure};
use burn::{prelude::*, tensor::Distribution};
use serde_json::{Value, json};
use std::time::Instant;

pub const MODES: [EpilogueBackend; 3] = [
    EpilogueBackend::Native,
    EpilogueBackend::AnalyticBackward,
    EpilogueBackend::FusedPointwise,
];
impl EpilogueBackend {
    pub fn silu(self, x: Tensor<3>) -> Tensor<3> {
        if self.is_native() {
            return burn::tensor::activation::silu(x);
        }
        #[cfg(any(feature = "vulkan", feature = "cuda"))]
        {
            delta_kernels::silu_analytic(x, self == Self::FusedPointwise)
        }
        #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
        {
            let _ = x;
            panic!("analytic SiLU requires an explicit GPU backend")
        }
    }
}
fn values(x: Tensor<3>) -> Vec<f32> {
    x.into_data().try_to_vec::<f32>().expect("f32 epilogue")
}
fn compare(actual: Vec<f32>, expected: &[f64]) -> Result<f64> {
    ensure!(actual.len() == expected.len(), "epilogue reference shape");
    let mut maximum = 0.0f64;
    for (i, (a, &e)) in actual.into_iter().zip(expected).enumerate() {
        let error = (a as f64 - e).abs();
        ensure!(
            a.is_finite() && e.is_finite() && error <= 2e-5 + 3e-4 * e.abs(),
            "epilogue reference element {i}: {a} != {e}, error {error}"
        );
        maximum = maximum.max(error);
    }
    Ok(maximum)
}
fn fixture(data: Vec<f32>, shape: [usize; 3], strided: bool, device: &Device) -> Tensor<3> {
    let [b, t, w] = shape;
    if strided {
        Tensor::from_data(TensorData::new(data, [b, w, t]), device).swap_dims(1, 2)
    } else {
        Tensor::from_data(TensorData::new(data, shape), device)
    }
}
/// Independent scalar-f64 output and input VJPs, including both residual parents.
/// GPU invocations test all three implementations; CPU tests only the library path.
pub fn numerical_checks(device: &Device) -> Result<Value> {
    let device = device.clone().autodiff();
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    let gpu = matches!(
        device.clone().inner().as_dispatch(),
        burn::backend::DispatchDevice::Cube(_)
    );
    #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
    let gpu = false;
    let modes = if gpu { &MODES[..] } else { &MODES[..1] };
    let mut cases = Vec::new();
    for width in [1, 3, 35, 256, 512] {
        for strided in [false, true] {
            for zero in [false, true] {
                let shape = [2, 7, width];
                let n = shape.iter().product();
                let input = (0..n)
                    .map(|i| {
                        if zero {
                            0.
                        } else {
                            match i % 13 {
                                0 => -1000.,
                                1 => 1000.,
                                2 => -100.,
                                3 => 100.,
                                4 => -30.,
                                5 => 30.,
                                6 => 0.,
                                7 => -1e-6,
                                8 => 1e-6,
                                _ => ((i * 17 + 3) as f64 * 0.013).sin() as f32,
                            }
                        }
                    })
                    .collect();
                let x = fixture(input, shape, strided, &device).require_grad();
                let upstream = fixture(
                    (0..n)
                        .map(|i| {
                            if i % 11 == 0 {
                                0.
                            } else {
                                ((i * 19 + 1) as f64 * 0.031).cos() as f32
                            }
                        })
                        .collect(),
                    shape,
                    strided,
                    &device,
                );
                let residual = fixture(
                    (0..n).map(|i| ((i + 1) as f32 * 0.07).sin()).collect(),
                    shape,
                    strided,
                    &device,
                )
                .require_grad();
                let xv = values(x.clone());
                let uv = values(upstream.clone());
                let rv = values(residual.clone());
                let mut y = Vec::with_capacity(n);
                let mut dx = Vec::with_capacity(n);
                let mut add = Vec::with_capacity(n);
                for i in 0..n {
                    let z = xv[i] as f64;
                    // Stable scalar reference, independently evaluated from the GPU formula.
                    let s = if z >= 0. {
                        1. / (1. + (-z).exp())
                    } else {
                        let e = z.exp();
                        e / (1. + e)
                    };
                    y.push(z * s);
                    dx.push(uv[i] as f64 * (s + z * s * (1. - s)));
                    add.push(z + rv[i] as f64);
                }
                for &mode in modes {
                    let result = mode.silu(x.clone());
                    let output_error = compare(values(result.clone()), &y)?;
                    let grads = (result * upstream.clone()).sum().backward();
                    let gradient_error = compare(
                        values(
                            x.grad(&grads)
                                .ok_or_else(|| anyhow::anyhow!("missing SiLU dx"))?,
                        ),
                        &dx,
                    )?;
                    cases.push(json!({"operator":"silu","epilogue_backend":mode,"shape":shape,
                        "strided":strided,"zero":zero,"output_max_abs":output_error,"input_vjp_max_abs":gradient_error}));
                }
                #[cfg(any(feature = "vulkan", feature = "cuda"))]
                if gpu {
                    // Autodiff's multiply can materialize dy; test the raw strided VJP too.
                    let error = compare(
                        values(delta_kernels::silu_vjp(x.clone(), upstream.clone())),
                        &dx,
                    )?;
                    cases.push(
                        json!({"operator":"explicit_silu_vjp","shape":shape,"strided":strided,
                        "zero":zero,"input_vjp_max_abs":error}),
                    );
                }
                let result = x.clone() + residual.clone();
                let output_error = compare(values(result.clone()), &add)?;
                let grads = (result * upstream.clone()).sum().backward();
                let expected: Vec<_> = uv.iter().map(|&x| x as f64).collect();
                let left = compare(
                    values(
                        x.grad(&grads)
                            .ok_or_else(|| anyhow::anyhow!("missing residual dx"))?,
                    ),
                    &expected,
                )?;
                let right = compare(
                    values(
                        residual
                            .grad(&grads)
                            .ok_or_else(|| anyhow::anyhow!("missing residual dr"))?,
                    ),
                    &expected,
                )?;
                // Shared-parent residual must accumulate twice, not overwrite a parent gradient.
                let grads = ((x.clone() + x.clone()) * upstream.clone())
                    .sum()
                    .backward();
                let twice: Vec<_> = expected.iter().map(|v| 2. * v).collect();
                let shared = compare(
                    values(
                        x.grad(&grads)
                            .ok_or_else(|| anyhow::anyhow!("missing shared residual dx"))?,
                    ),
                    &twice,
                )?;
                cases.push(json!({"operator":"native_residual","shape":shape,"strided":strided,"zero":zero,
                    "output_max_abs":output_error,"left_vjp_max_abs":left,"right_vjp_max_abs":right,"shared_parent_vjp_max_abs":shared}));
            }
        }
    }
    Ok(json!({"passed":true,"absolute_tolerance":2e-5,"relative_tolerance":3e-4,"cases":cases}))
}

/// Resident isolated SiLU, native residual, and actual up->SiLU->down->residual graph.
/// No normalization is moved across either matmul. Width and parameters match the model.
pub fn benchmark(
    device: &Device,
    tokens: usize,
    batch: usize,
    width: usize,
    warmups: usize,
    trials: usize,
) -> Result<Value> {
    ensure!(
        warmups > 0 && trials >= 3,
        "epilogue sampling requires warmup and >=3 trials"
    );
    let device = device.clone().autodiff();
    device.seed(2917);
    let x = Tensor::<3>::random(
        [batch, tokens, width],
        Distribution::Uniform(-0.5, 0.5),
        &device,
    )
    .require_grad();
    let activation = Tensor::<3>::random(
        [batch, tokens, 2 * width],
        Distribution::Uniform(-0.5, 0.5),
        &device,
    )
    .require_grad();
    let residual = Tensor::<3>::random(
        [batch, tokens, width],
        Distribution::Uniform(-0.5, 0.5),
        &device,
    )
    .require_grad();
    let up = Tensor::<2>::random([width, 2 * width], Distribution::Normal(0., 0.02), &device)
        .require_grad();
    let down = Tensor::<2>::random([2 * width, width], Distribution::Normal(0., 0.02), &device)
        .require_grad();
    runtime::sync(&device)?;
    let mut rows = Vec::new();
    for scope in ["silu", "residual", "mlp"] {
        let modes = if scope == "residual" {
            &MODES[..1]
        } else {
            &MODES[..]
        };
        let mut forward: [Vec<f64>; 3] = core::array::from_fn(|_| Vec::new());
        let mut backward: [Vec<f64>; 3] = core::array::from_fn(|_| Vec::new());
        let run = |mode: EpilogueBackend| match scope {
            "silu" => mode.silu(activation.clone()),
            "residual" => x.clone() + residual.clone(),
            _ => {
                let a = burn::tensor::module::linear(x.clone(), up.clone(), None);
                burn::tensor::module::linear(mode.silu(a), down.clone(), None) + residual.clone()
            }
        };
        for iteration in 0..warmups + trials {
            let mut order: Vec<_> = (0..modes.len())
                .map(|i| (i + iteration) % modes.len())
                .collect();
            if iteration % 2 == 1 {
                order.reverse();
            }
            for index in order {
                runtime::sync(&device)?;
                let start = Instant::now();
                let y = run(modes[index]);
                runtime::sync(&device)?;
                let f = start.elapsed().as_secs_f64() * 1000.;
                drop(y);
                runtime::sync(&device)?;
                let start = Instant::now();
                let grads = run(modes[index]).square().sum().backward();
                runtime::sync(&device)?;
                let fb = start.elapsed().as_secs_f64() * 1000.;
                std::hint::black_box(grads);
                if iteration >= warmups {
                    forward[index].push(f);
                    backward[index].push(fb);
                }
            }
        }
        for (i, &mode) in modes.iter().enumerate() {
            rows.push(json!({"scope":scope,"epilogue_backend":mode,
                "forward_samples_ms":forward[i],"forward_median_ms":median(&forward[i]),
                "forward_backward_samples_ms":backward[i],"forward_backward_median_ms":median(&backward[i])}));
        }
    }
    let native_ms = |scope: &str| {
        rows.iter()
            .find(|r| r["scope"] == scope && r["epilogue_backend"] == "native")
            .unwrap()["forward_backward_median_ms"]
            .as_f64()
            .unwrap()
    };
    let removable = native_ms("silu") + native_ms("residual");
    Ok(
        json!({"seed":2917,"batch":batch,"tokens":tokens,"width":width,"rows":rows,
        "optimistic_isolated_fb_removable_ms":removable,
        "optimistic_isolated_fb_fraction_of_native_mlp":removable / native_ms("mlp"),
        "ceiling_caveat":"isolated activation/add FB includes square+sum objective and barriers; sum is an optimistic attribution estimate, not a proven integrated speedup bound; residual implementation is unchanged",
        "analytic_saved_input_bytes":batch * tokens * 2 * width * 4,
        "analytic_vjp_output_bytes":batch * tokens * 2 * width * 4,
        "fused_pointwise_forward_dispatches":1,"analytic_vjp_dispatches":1,
        "memory_scope":"logical f32 tensor bytes, not allocator peak; analytic VJP allocates dx only, no sigmoid intermediate; fullmodel report samples pool reservations"}),
    )
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    #[test]
    fn native_epilogue_scalar_f64_outputs_and_vjps() {
        super::numerical_checks(&burn::prelude::Device::ndarray()).unwrap();
    }
}
