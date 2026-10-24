use super::{ScanInput, scan};
use crate::{
    MixerKind,
    reference::{self, Token},
};
use burn::prelude::*;

fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor.into_data().try_to_vec::<f32>().unwrap()
}

fn assert_close(name: &str, actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{name}");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && e.is_finite() && (f64::from(a) - e).abs() <= 2e-5 + 3e-4 * e.abs(),
            "{name}[{i}]: {a} != {e}"
        );
    }
}

// Independent of the normalized trigonometric acceptance fixture. A padded reset
// between active tokens must preserve the initial-state path; a later valid reset
// must sever it. Check both common-solver callers and all computational partitions.
fn initial_key_reset_padding(device: Device) {
    let device = device.autodiff();
    let (b, t, h, k, v) = (2, 5, 2, 3, 4);
    let valid = [
        true, false, true, true, false, true, true, false, true, false,
    ];
    let reset = [
        false, true, false, true, false, true, false, true, false, true,
    ];
    let initial_data: Vec<f32> = (0..b * h * k * v)
        .map(|i| 0.03 * (1 + i % 7) as f32)
        .collect();
    let upstream: Vec<f32> = (0..b * t * h * v)
        .map(|i| 0.04 * ((i % 9) as f32 - 3.))
        .collect();
    for kind in [MixerKind::KronLstm, MixerKind::Gdn2] {
        let widths = [
            k,
            k,
            v,
            k,
            if kind == MixerKind::KronLstm { 1 } else { k },
            if kind == MixerKind::KronLstm { 1 } else { v },
        ];
        let data: Vec<Vec<f32>> = widths
            .iter()
            .enumerate()
            .map(|(field, &width)| {
                (0..b * t * h * width)
                    .map(|i| {
                        if field == 3 {
                            if i % (h * k) == 1 && (i / (h * k)) % t == 2 {
                                0.
                            } else {
                                0.75 + 0.02 * (i % 5) as f32
                            }
                        } else if field == 4 {
                            0.6
                        } else if field == 5 {
                            if (i / (h * width)) % t == 2 {
                                0.
                            } else {
                                -0.25
                            }
                        } else {
                            0.08 * (((i + 3 * field) % 7) as f32 - 2.)
                        }
                    })
                    .collect()
            })
            .collect();
        let mut expected_output = vec![0.; b * t * h * v];
        let mut expected_key = vec![0.; b * t * h * k];
        let mut expected_initial = vec![0.; b * h * k * v];
        let mut initial_key_effect = 0.0f64;
        for row in 0..b {
            for head in 0..h {
                let tokens: Vec<Token> = (0..t)
                    .map(|p| {
                        let fields: Vec<Vec<f64>> = data
                            .iter()
                            .zip(widths)
                            .map(|(field, width)| {
                                let start = ((row * t + p) * h + head) * width;
                                field[start..start + width]
                                    .iter()
                                    .map(|&x| f64::from(x))
                                    .collect()
                            })
                            .collect();
                        Token {
                            q: fields[0].clone(),
                            k: fields[1].clone(),
                            v: fields[2].clone(),
                            alpha: fields[3].clone(),
                            erase: fields[4].clone(),
                            write: fields[5].clone(),
                        }
                    })
                    .collect();
                let sf = (row * h + head) * k * v;
                let initial: Vec<f64> = initial_data[sf..sf + k * v]
                    .iter()
                    .map(|&x| f64::from(x))
                    .collect();
                let seeds: Vec<Vec<f64>> = (0..t)
                    .map(|p| {
                        let start = ((row * t + p) * h + head) * v;
                        upstream[start..start + v]
                            .iter()
                            .map(|&x| f64::from(x))
                            .collect()
                    })
                    .collect();
                let valid = &valid[row * t..(row + 1) * t];
                let reset = &reset[row * t..(row + 1) * t];
                let forward = reference::scan(kind, &tokens, valid, reset, Some(&initial));
                let backward = reference::backward(
                    kind,
                    &tokens,
                    valid,
                    reset,
                    &seeds,
                    &vec![0.; k * v],
                    Some(&initial),
                    &vec![0.; t],
                );
                let zero_initial = reference::backward(
                    kind,
                    &tokens,
                    valid,
                    reset,
                    &seeds,
                    &vec![0.; k * v],
                    None,
                    &vec![0.; t],
                );
                expected_initial[sf..sf + k * v].copy_from_slice(&backward.initial_state);
                for p in 0..t {
                    let of = ((row * t + p) * h + head) * v;
                    let kf = ((row * t + p) * h + head) * k;
                    expected_output[of..of + v].copy_from_slice(&forward.output[p]);
                    expected_key[kf..kf + k].copy_from_slice(&backward.tokens[p].k);
                    if row == 0 && p == 0 {
                        initial_key_effect = initial_key_effect.max(
                            backward.tokens[p]
                                .k
                                .iter()
                                .zip(&zero_initial.tokens[p].k)
                                .map(|(a, b)| (a - b).abs())
                                .fold(0., f64::max),
                        );
                    }
                    if !valid[p] {
                        assert!(backward.tokens[p].k.iter().all(|&x| x == 0.));
                        assert!(forward.output[p].iter().all(|&x| x == 0.));
                    }
                }
            }
        }
        assert!(
            initial_key_effect > 1e-4,
            "fixture must expose nonzero initial-state key dependence"
        );
        assert!(expected_initial[..h * k * v].iter().any(|x| x.abs() > 1e-4));
        assert!(expected_initial[h * k * v..].iter().all(|&x| x == 0.));
        for chunk in [1, 2, 5, 16] {
            let case = format!("{kind:?}/B{b}/T{t}/H{h}/K{k}/V{v}/chunk{chunk}/output_only");
            let x: Vec<Tensor<4>> = data
                .iter()
                .zip(widths)
                .map(|(data, width)| {
                    Tensor::from_data(TensorData::new(data.clone(), [b, t, h, width]), &device)
                        .require_grad()
                })
                .collect();
            let initial =
                Tensor::from_data(TensorData::new(initial_data.clone(), [b, h, k, v]), &device)
                    .require_grad();
            let metadata = |flags: &[bool]| {
                Tensor::from_data(
                    TensorData::new(
                        flags
                            .iter()
                            .map(|&x| if x { 1. } else { 0. })
                            .collect::<Vec<f32>>(),
                        [b, t],
                    ),
                    &device,
                )
            };
            let output = scan(
                kind,
                ScanInput {
                    q: x[0].clone(),
                    k: x[1].clone(),
                    v: x[2].clone(),
                    alpha: x[3].clone(),
                    erase: x[4].clone(),
                    write: x[5].clone(),
                },
                metadata(&valid),
                metadata(&reset),
                chunk,
                Some(initial.clone()),
            );
            assert_close(
                &format!("{case}/output"),
                &values(output.output.clone()),
                &expected_output,
            );
            let grads = (output.output
                * Tensor::from_data(TensorData::new(upstream.clone(), [b, t, h, v]), &device))
            .sum()
            .backward();
            assert_close(
                &format!("{case}/key VJP"),
                &values(x[1].grad(&grads).unwrap()),
                &expected_key,
            );
            assert_close(
                &format!("{case}/initial-state VJP"),
                &values(initial.grad(&grads).unwrap()),
                &expected_initial,
            );
        }
    }
}

#[cfg(feature = "cpu")]
#[test]
fn tensor_solver_initial_key_reset_padding_cpu() {
    initial_key_reset_padding(Device::ndarray());
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "shares the GPU; run explicitly under the repository GPU lock"]
fn tensor_solver_initial_key_reset_padding_vulkan() {
    initial_key_reset_padding(Device::vulkan(Default::default()));
}
