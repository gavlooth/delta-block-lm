//! Independent scalar-f64 formulas, not autodiff of the tensor reference.
use crate::normalization::NormalizationBackend as Backend;
use burn::{nn::RmsNormConfig, prelude::*};
use serde::Serialize;
fn values<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.into_data()
        .try_to_vec::<f32>()
        .expect("f32 normalization check")
}
fn compare(label: &str, actual: Vec<f32>, expected: &[f64]) -> Result<f64, String> {
    if actual.len() != expected.len() {
        return Err(format!("{label} length"));
    }
    let mut maximum = 0.0f64;
    for (i, (a, &e)) in actual.into_iter().zip(expected).enumerate() {
        let error = (a as f64 - e).abs();
        if !a.is_finite() || !e.is_finite() || error > 2e-5 + 3e-4 * e.abs() {
            return Err(format!("{label}[{i}] {a} != {e}, abs error {error}"));
        }
        maximum = maximum.max(error);
    }
    Ok(maximum)
}
fn strided(data: Vec<f32>, shape: [usize; 4], device: &Device) -> Tensor<4> {
    // Swap storage axes rather than slicing a contiguous allocation.
    let [b, t, h, w] = shape;
    Tensor::<4>::from_data(TensorData::new(data, [b, w, h, t]), device).swap_dims(1, 3)
}
#[derive(Serialize)]
pub struct Case {
    operator: &'static str,
    width: usize,
    epsilon: f64,
    zero: bool,
    input_scale: f64,
    strided_inputs: bool,
    internal_and_tail_padding: bool,
    all_input_parameter_gradients: bool,
    max_abs_error: f64,
}
#[derive(Serialize)]
pub struct Report {
    passed: bool,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    cases: Vec<Case>,
}
pub fn numerical_checks(device: &Device) -> Result<Report, String> {
    let device = device.clone().autodiff();
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    let gpu = matches!(
        device.clone().inner().as_dispatch(),
        burn::backend::DispatchDevice::Cube(_)
    );
    #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
    let gpu = false;
    let modes = if gpu {
        vec![
            Backend::Tensor,
            Backend::FusedNorms,
            Backend::FusedNormsGates,
        ]
    } else {
        vec![Backend::Tensor]
    };
    let mut cases = Vec::new();
    for width in [3, 32, 35, 64, 256] {
        for (zero, input_scale) in [(false, 1.0), (false, 1e-3), (false, 1e-12), (false, 1e-14), (true, 0.0)] {
            let shape = [2, 7, 2, width];
            let n = shape.iter().product();
            let data = (0..n)
                .map(|i| {
                    // Pre-zero physical padding instead of a multiply that would
                    // materialize x and erase its noncontiguous layout.
                    let token = i / (width * 2 * 7) * 7 + i % 7;
                    if zero || [0, 3, 6, 9, 13].contains(&token) {
                        0.
                    } else {
                        (((i * 17 + 3) as f64 * 0.013).sin() * input_scale) as f32
                    }
                })
                .collect();
            let x = strided(data, shape, &device).require_grad();
            let gate = strided(
                (0..n)
                    .map(|i| match i % 11 {
                        0 => -30.,
                        1 => 30.,
                        2 => 0.,
                        _ => ((i * 7) as f32 * 0.07).cos(),
                    })
                    .collect(),
                shape,
                &device,
            )
            .require_grad();
            let upstream = strided(
                (0..n)
                    .map(|i| {
                        if i % 13 == 0 {
                            0.
                        } else {
                            ((i * 19 + 1) as f64 * 0.031).cos() as f32
                        }
                    })
                    .collect(),
                shape,
                &device,
            );
            let weight = Tensor::<1>::from_data(
                TensorData::new(
                    (0..width)
                        .map(|i| {
                            if i % 7 == 0 {
                                0.
                            } else {
                                ((i + 3) as f32 * 0.17).sin()
                            }
                        })
                        .collect::<Vec<_>>(),
                    [width],
                ),
                &device,
            )
            .require_grad();
            let mask_values = (0..14)
                .map(|i| {
                    if [0, 3, 6, 9, 13].contains(&i) {
                        0.
                    } else {
                        1.
                    }
                })
                .collect::<Vec<_>>();
            let mask =
                Tensor::<2>::from_data(TensorData::new(mask_values.clone(), [2, 7]), &device)
                    .reshape([2, 7, 1, 1]);
            let xv = values(x.clone());
            let gv = values(gate.clone());
            let uv = values(upstream.clone());
            let wv = values(weight.clone());
            // L2 uses a norm floor; RMS retains its separate additive epsilons.
            for (mode, epsilon) in [(0, 1e-12), (1, 1e-6), (1, 1e-5), (2, 1e-5)] {
                let mut y = vec![0.; n];
                let mut dx = vec![0.; n];
                let mut dg = vec![0.; n];
                let mut dw = vec![0.; width];
                for row in 0..28 {
                    let mask = mask_values[row / 2] as f64;
                    let mut z = vec![0.; width];
                    let mut s = vec![1.; width];
                    for c in 0..width {
                        let i = row * width + c;
                        if mode == 2 {
                            s[c] = 1. / (1. + (-(gv[i] as f64)).exp());
                        }
                        z[c] = xv[i] as f64 * mask * s[c];
                    }
                    let divisor = if mode == 0 { 1. } else { width as f64 };
                    let square = z.iter().map(|z| z * z).sum::<f64>();
                    let inverse = if mode == 0 {
                        square.sqrt().max(epsilon).recip()
                    } else {
                        (square / divisor + epsilon).sqrt().recip()
                    };
                    let norm_derivative = mode != 0 || square >= epsilon * epsilon;
                    let dot = (0..width)
                        .map(|c| {
                            uv[row * width + c] as f64
                                * mask
                                * z[c]
                                * if mode == 0 { 1. } else { wv[c] as f64 }
                        })
                        .sum::<f64>();
                    for c in 0..width {
                        let i = row * width + c;
                        let w = if mode == 0 { 1. } else { wv[c] as f64 };
                        y[i] = z[c] * inverse * w * mask;
                        let dz = (uv[i] as f64 * mask * w
                            - if norm_derivative { z[c] * dot * inverse * inverse / divisor } else { 0.0 })
                            * inverse;
                        dx[i] = dz * s[c] * mask;
                        dg[i] = dz * z[c] * (1. - s[c]);
                        dw[c] += uv[i] as f64 * mask * z[c] * inverse;
                    }
                }
                for &backend in &modes {
                    if mode == 2 && backend == Backend::FusedNorms {
                        continue;
                    }
                    let norm = RmsNormConfig::new(width)
                        .with_epsilon(epsilon)
                        .init(&device);
                    // Reuse the exact leaf as gamma; no parameter initialization is the oracle.
                    let norm = burn::nn::RmsNorm {
                        gamma: burn::module::Param::from_tensor(weight.clone()),
                        ..norm
                    };
                    let input = x.clone();
                    let result = if mode == 0 {
                        backend.l2(input)
                    } else if mode == 1 {
                        backend.rms(&norm, input)
                    } else {
                        backend.gated_rms(&norm, input, gate.clone())
                    } * mask.clone();
                    let mut error = compare("normalization output", values(result.clone()), &y)?;
                    let grads = (result * upstream.clone()).sum().backward();
                    error = error.max(compare(
                        "normalization dx",
                        values(x.grad(&grads).ok_or("missing dx")?),
                        &dx,
                    )?);
                    if mode != 0 {
                        error = error.max(compare(
                            "normalization dgamma",
                            values(norm.gamma.val().grad(&grads).ok_or("missing dgamma")?),
                            &dw,
                        )?);
                    }
                    if mode == 2 {
                        error = error.max(compare(
                            "normalization dgate",
                            values(gate.grad(&grads).ok_or("missing dgate")?),
                            &dg,
                        )?);
                    }
                    cases.push(Case {
                        operator: match (mode, backend) {
                            (0, Backend::Tensor) => "tensor_l2",
                            (0, _) => "fused_l2",
                            (1, Backend::Tensor) => "tensor_rms",
                            (1, _) => "fused_rms",
                            (_, Backend::Tensor) => "tensor_gated_rms",
                            _ => "fused_gated_rms",
                        },
                        width,
                        epsilon,
                        zero,
                        input_scale,
                        strided_inputs: true,
                        internal_and_tail_padding: true,
                        all_input_parameter_gradients: true,
                        max_abs_error: error,
                    });
                }
            }
        }
    }
    for zero in [false, true] {
        let shape = [2, 7, 2, 3];
        let n = 84;
        let x = strided(
            (0..n)
                .map(|i| {
                    if zero {
                        0.
                    } else {
                        match i % 9 {
                            0 => -40.,
                            1 => 40.,
                            2 => 20.,
                            _ => ((i * 3) as f32 * 0.11).sin(),
                        }
                    }
                })
                .collect(),
            shape,
            &device,
        )
        .require_grad();
        let bias = Tensor::<1>::from_data([0., 0.3, -4., 0., -0.7, -3.], &device).require_grad();
        let scale = Tensor::<1>::from_data([-0.2, 0.4], &device).require_grad();
        let upstream = strided(
            (0..n)
                .map(|i| {
                    if i % 5 == 0 {
                        0.
                    } else {
                        ((i * 7) as f32 * 0.03).cos()
                    }
                })
                .collect(),
            shape,
            &device,
        );
        let mask_values = (0..14)
            .map(|i| {
                if [0, 3, 6, 9, 13].contains(&i) {
                    0.
                } else {
                    1.
                }
            })
            .collect::<Vec<_>>();
        let mask = Tensor::<2>::from_data(TensorData::new(mask_values.clone(), [2, 7]), &device)
            .reshape([2, 7, 1, 1]);
        let xv = values(x.clone());
        let bv = values(bias.clone());
        let sv = values(scale.clone());
        let uv = values(upstream.clone());
        let mut y = vec![0.; n];
        let mut dx = vec![0.; n];
        let mut db = vec![0.; 6];
        let mut ds = vec![0.; 2];
        for i in 0..n {
            let c = i % 6;
            let h = c / 3;
            let mask = mask_values[i / 6] as f64;
            let z = xv[i] as f64 + bv[c] as f64;
            let sp = if z > 20. { z } else { z.exp().ln_1p() };
            let derivative = if z > 20. { 1. } else { 1. / (1. + (-z).exp()) };
            let scale = (sv[h] as f64).exp();
            let alpha = (-sp * scale).exp();
            y[i] = alpha * mask;
            dx[i] = -uv[i] as f64 * mask * alpha * scale * derivative;
            db[c] += dx[i];
            ds[h] += -uv[i] as f64 * mask * alpha * scale * sp;
        }
        for &backend in &modes {
            let result = backend.decay(x.clone(), bias.clone(), scale.clone()) * mask.clone();
            let mut error = compare("decay output", values(result.clone()), &y)?;
            let grads = (result * upstream.clone()).sum().backward();
            for (name, actual, expected) in [
                ("decay dx", values(x.grad(&grads).ok_or("decay dx")?), &dx),
                (
                    "decay dbias",
                    values(bias.grad(&grads).ok_or("decay dbias")?),
                    &db,
                ),
                (
                    "decay dscale",
                    values(scale.grad(&grads).ok_or("decay dscale")?),
                    &ds,
                ),
            ] {
                error = error.max(compare(name, actual, expected)?);
            }
            cases.push(Case {
                operator: if backend == Backend::FusedNormsGates {
                    "fused_decay"
                } else {
                    "tensor_decay"
                },
                width: 3,
                epsilon: 0.,
                zero,
                input_scale: if zero { 0.0 } else { 1.0 },
                strided_inputs: true,
                internal_and_tail_padding: true,
                all_input_parameter_gradients: true,
                max_abs_error: error,
            });
        }
    }
    Ok(Report {
        passed: true,
        absolute_tolerance: 2e-5,
        relative_tolerance: 3e-4,
        cases,
    })
}
#[cfg(all(test, feature = "cpu"))]
mod tests {
    #[test]
    fn independent_f64_normalization_and_gating() {
        super::numerical_checks(&burn::prelude::Device::ndarray()).unwrap();
    }
    #[test]
    #[should_panic(expected = "no CPU fallback")]
    fn custom_normalization_never_falls_back_to_cpu() {
        crate::MixerConfig {
            normalization_backend: crate::NormalizationBackend::FusedNormsGates,
            ..Default::default()
        }
        .init(&burn::prelude::Device::ndarray());
    }
}
