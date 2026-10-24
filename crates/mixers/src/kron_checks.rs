//! Independent f64 acceptance for fused outputs, state, physical scores and every float input.
use super::{ATOL, Fixture, RTOL, compare, scan_input, values};
use crate::{
    MixerKind, ScanBackend, chunk,
    reference::{self, Token},
};
use burn::prelude::*;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct KronCheck {
    pub mixer: MixerKind,
    pub scan_backend: ScanBackend,
    pub tokens: usize,
    pub objective: &'static str,
    pub strided_inputs: bool,
    pub output_max_abs: f64,
    pub state_max_abs: f64,
    pub score_max_abs: f64,
    pub input_gradient_max_abs: f64,
    pub initial_state_gradient_max_abs: f64,
    pub continuation_max_abs: f64,
    pub continuation_gradient_max_abs: f64,
}

/// Both paths check every output's VJP against an independent sequential f64 derivative.
pub fn kron_numerical_checks(device: &Device) -> Result<Vec<KronCheck>, String> {
    let device = device.clone().autodiff();
    let mut backends = vec![ScanBackend::TensorSolver];
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    if matches!(
        device.clone().inner().as_dispatch(),
        burn::backend::DispatchDevice::Cube(_)
    ) {
        backends.push(ScanBackend::FusedGpu);
    }
    let mut reports = Vec::new();
    for backend in backends {
        reports.extend(kron_numerical_checks_selected(
            &device, backend, None, None,
        )?);
    }
    Ok(reports)
}

/// Runs the existing fixtures for exactly one requested path, without another path's gate.
/// Objective indices are combined=0, output=1, final state=2 and score=3.
pub fn kron_numerical_checks_selected(
    device: &Device,
    backend: ScanBackend,
    tokens: Option<usize>,
    objective: Option<usize>,
) -> Result<Vec<KronCheck>, String> {
    if tokens.is_some_and(|n| !(1..=512).contains(&n)) {
        return Err("Kron reference token count must be 1..=512".into());
    }
    if objective.is_some_and(|mode| mode > 3) {
        return Err("Kron reference objective must be 0..=3".into());
    }
    let device = device.clone().autodiff();
    if backend.is_fused() {
        chunk::assert_fused_device(&device);
    }
    let times = tokens.map_or_else(|| vec![1, 7, 35], |n| vec![n]);
    let mut reports = Vec::new();
    for time in times {
        for mode in 0..4 {
            if objective.is_some_and(|selected| selected != mode) {
                continue;
            }
            reports.push(check(&device, backend, MixerKind::KronLstm, time, mode, false).map_err(|error| {
                let objective = match mode {
                    0 => "output+final_state+score",
                    1 => "output_only",
                    2 => "final_state_only",
                    _ => "score_only",
                };
                format!("KronLSTM backend={backend:?} B2/T{time}/H4/K32/V64/chunk16 objective={objective} fixture=nonzero_initial_zero_alpha_signed_and_zero_correction_zero_norm_padding_reset: {error}")
            })?);
        }
    }
    if tokens.is_none() && objective.is_none() {
        reports.push(check(&device, backend, MixerKind::KronLstm, 35, 0, true)?);
    }
    Ok(reports)
}

/// Independent GDN-2 output/state/score/initial VJPs at the original tolerances.
pub fn gdn_numerical_checks_selected(
    device: &Device,
    backend: ScanBackend,
    tokens: Option<usize>,
    objective: Option<usize>,
) -> Result<Vec<KronCheck>, String> {
    if !backend.supports(MixerKind::Gdn2) {
        return Err("GDN-2 requires tensor_solver or fused_gdn2".into());
    }
    if tokens.is_some_and(|n| !(1..=512).contains(&n))
        || objective.is_some_and(|mode| mode > 3) {
        return Err("GDN-2 reference expects tokens 1..=512 and objective 0..=3".into());
    }
    let device = device.clone().autodiff();
    if backend.is_fused() { chunk::assert_fused_device(&device); }
    let mut reports = Vec::new();
    for time in tokens.map_or_else(|| vec![1, 7, 35], |n| vec![n]) {
        for mode in 0..4 {
            if objective.is_some_and(|selected| selected != mode) { continue; }
            reports.push(check(&device, backend, MixerKind::Gdn2, time, mode, false)
                .map_err(|error| format!("GDN-2 {backend:?} T{time} objective={mode}: {error}"))?);
        }
    }
    if tokens.is_none() && objective.is_none() {
        reports.push(check(&device, backend, MixerKind::Gdn2, 35, 0, true)?);
    }
    Ok(reports)
}

fn check(
    device: &Device,
    backend: ScanBackend,
    kind: MixerKind,
    time: usize,
    mode: usize,
    strided: bool,
) -> Result<KronCheck, String> {
    let mut f = Fixture::new(kind, time);
    // Both non-reset initial-state dependence and reset-discarded initial state are exercised.
    f.reset[0] = false;
    if time == 1 {
        f.valid.fill(true);
    }
    for row in 0..f.b {
        if time > 9 {
            f.reset[row * time + 9] = true;
        } // reset on padding must be ignored
        if row == 1 && time > 16 {
            f.valid[row * time + 16..(row + 1) * time].fill(false); // includes an entire padding chunk
        }
        for p in 0..time {
            for h in 0..f.h {
                for field in [1, 2] {
                    let width = f.widths[field];
                    let start = ((row * time + p) * f.h + h) * width;
                    let norm = f.data[field][start..start + width]
                        .iter()
                        .map(|x| x * x)
                        .sum::<f32>()
                        .sqrt();
                    for x in &mut f.data[field][start..start + width] {
                        *x /= norm;
                    }
                }
                let scalar = (row * time + p) * f.h + h;
                if p == 1 {
                    for field in [4, 5] {
                        let width = f.widths[field];
                        f.data[field][scalar * width..(scalar + 1) * width].fill(0.);
                    } // exactly zero edit with nonzero key/value norms
                } else if p == 3 {
                    let width = f.widths[5];
                    f.data[5][scalar * width..(scalar + 1) * width].fill(-0.2);
                } else if p == 4 || p == 6 {
                    let field = if p == 4 { 1 } else { 2 };
                    let width = f.widths[field];
                    let start = scalar * width;
                    f.data[field][start..start + width].fill(0.); // zero key/value norm subgradient
                }
                if p == 2 || p == 16 {
                    let start = ((row * time + p) * f.h + h) * f.k;
                    for x in &mut f.data[3][start..start + if p == 2 { 1 } else { f.k }] {
                        *x = 0.;
                    }
                }
            }
        }
    }
    let initial_data = (0..f.b * f.h * f.k * f.v)
        .map(|i| 0.03 * ((i * 11 + 7) as f32 * 0.017).sin())
        .collect::<Vec<_>>();
    let score_seed = (0..f.b * time * f.h)
        .map(|i| 0.07 * ((i * 7 + 2) as f32 * 0.13).cos())
        .collect::<Vec<_>>();
    let use_output = mode == 0 || mode == 1;
    let use_final = mode == 0 || mode == 2;
    let use_scores = mode == 0 || mode == 3;
    let mut expected_output = vec![0.; f.b * time * f.h * f.v];
    let mut expected_state = vec![0.; initial_data.len()];
    let mut expected_scores = vec![0.; score_seed.len()];
    let mut expected_grads = f.data.iter().map(|x| vec![0.; x.len()]).collect::<Vec<_>>();
    let mut expected_initial = vec![0.; initial_data.len()];
    for row in 0..f.b {
        for head in 0..f.h {
            let tokens = (0..time)
                .map(|p| {
                    let fields = (0..6)
                        .map(|field| {
                            let width = f.widths[field];
                            let first = ((row * time + p) * f.h + head) * width;
                            f.data[field][first..first + width]
                                .iter()
                                .map(|&x| f64::from(x))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    Token {
                        q: fields[0].clone(),
                        k: fields[1].clone(),
                        v: fields[2].clone(),
                        alpha: fields[3].clone(),
                        erase: fields[4].clone(),
                        write: fields[5].clone(),
                    }
                })
                .collect::<Vec<_>>();
            let sf = (row * f.h + head) * f.k * f.v;
            let initial = initial_data[sf..sf + f.k * f.v]
                .iter()
                .map(|&x| f64::from(x))
                .collect::<Vec<_>>();
            let valid = &f.valid[row * time..(row + 1) * time];
            let reset = &f.reset[row * time..(row + 1) * time];
            let forward =
                reference::scan(kind, &tokens, valid, reset, Some(&initial));
            let upstream = (0..time)
                .map(|p| {
                    let first = ((row * time + p) * f.h + head) * f.v;
                    f.upstream[first..first + f.v]
                        .iter()
                        .map(|&x| if use_output { f64::from(x) } else { 0. })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let final_seed = f.final_adjoint[sf..sf + f.k * f.v]
                .iter()
                .map(|&x| if use_final { f64::from(x) } else { 0. })
                .collect::<Vec<_>>();
            let scores = (0..time)
                .map(|p| {
                    if use_scores {
                        f64::from(score_seed[(row * time + p) * f.h + head])
                    } else {
                        0.
                    }
                })
                .collect::<Vec<_>>();
            let backward = reference::backward(
                kind,
                &tokens,
                valid,
                reset,
                &upstream,
                &final_seed,
                Some(&initial),
                &scores,
            );
            expected_state[sf..sf + f.k * f.v].copy_from_slice(&forward.final_state);
            expected_initial[sf..sf + f.k * f.v].copy_from_slice(&backward.initial_state);
            for p in 0..time {
                let first = ((row * time + p) * f.h + head) * f.v;
                expected_output[first..first + f.v].copy_from_slice(&forward.output[p]);
                expected_scores[(row * time + p) * f.h + head] = forward.scores[p];
                for (field, grad) in backward.tokens[p].fields().into_iter().enumerate() {
                    let width = f.widths[field];
                    let first = ((row * time + p) * f.h + head) * width;
                    expected_grads[field][first..first + width].copy_from_slice(grad);
                }
            }
        }
    }
    let inputs = || {
        if strided {
            f.data
                .iter()
                .zip(f.widths)
                .map(|(data, width)| {
                    strided_upload(data.clone(), [f.b, time, f.h, width], 1, 2, device)
                        .require_grad()
                })
                .collect::<Vec<_>>()
        } else {
            f.inputs(device)
        }
    };
    let x = inputs();
    let initial = if strided {
        strided_upload(initial_data, [f.b, f.h, f.k, f.v], 2, 3, device)
    } else {
        Tensor::from_data(TensorData::new(initial_data, [f.b, f.h, f.k, f.v]), device)
    }
    .require_grad();
    let (valid, reset) = if strided {
        (
            strided_upload(
                f.valid.iter().map(|&x| if x { 1. } else { 0. }).collect(),
                [f.b, time],
                0,
                1,
                device,
            ),
            strided_upload(
                f.reset.iter().map(|&x| if x { 1. } else { 0. }).collect(),
                [f.b, time],
                0,
                1,
                device,
            ),
        )
    } else {
        f.metadata(device)
    };
    let out = chunk::scan_selected(
        backend,
        kind,
        scan_input(&x),
        valid.clone(),
        reset.clone(),
        16,
        Some(initial.clone()),
    );
    let output_error = compare(
        "Kron output",
        &values(out.output.clone()),
        &expected_output,
        ATOL,
        RTOL,
    )?;
    let state_error = compare(
        "Kron state",
        &values(out.final_state.clone()),
        &expected_state,
        ATOL,
        RTOL,
    )?;
    let score_error = compare(
        "Kron physical HOLA score",
        &values(out.scores.clone()),
        &expected_scores,
        ATOL,
        RTOL,
    )?;
    let mut terms = Vec::new();
    if use_output {
        terms.push(
            (out.output
                * Tensor::from_data(
                    TensorData::new(f.upstream.clone(), [f.b, time, f.h, f.v]),
                    device,
                ))
            .sum(),
        );
    }
    if use_final {
        terms.push(
            (out.final_state
                * Tensor::from_data(
                    TensorData::new(f.final_adjoint.clone(), [f.b, f.h, f.k, f.v]),
                    device,
                ))
            .sum(),
        );
    }
    if use_scores {
        terms.push(
            (out.scores
                * Tensor::from_data(
                    TensorData::new(score_seed.clone(), [f.b, time, f.h, 1]),
                    device,
                ))
            .sum(),
        );
    }
    let grads = Tensor::cat(terms, 0).sum().backward();
    let mut gradient_error = 0.0f64;
    for (field, tensor) in x.iter().enumerate() {
        let actual = tensor
            .grad(&grads)
            .map(values)
            .unwrap_or_else(|| vec![0.; f.data[field].len()]);
        gradient_error = gradient_error.max(compare(
            &format!("Kron gradient field {field}"),
            &actual,
            &expected_grads[field],
            ATOL,
            RTOL,
        )?);
    }
    let initial_error = compare(
        "Kron initial-state VJP",
        &values(
            initial
                .grad(&grads)
                .ok_or("missing initial-state gradient")?,
        ),
        &expected_initial,
        ATOL,
        RTOL,
    )?;
    let (continuation_error, continuation_gradient_error) = if time > 1 {
        let x = inputs();
        let initial = initial.clone().detach().require_grad();
        let split = 3.min(time - 1);
        let section = |start, end| {
            let xs = x
                .iter()
                .map(|x| {
                    x.clone()
                        .slice([0..f.b, start..end, 0..f.h, 0..x.dims()[3]])
                })
                .collect::<Vec<_>>();
            (
                scan_input(&xs),
                valid.clone().slice([0..f.b, start..end]),
                reset.clone().slice([0..f.b, start..end]),
            )
        };
        let (a, av, ar) = section(0, split);
        let a = chunk::scan_selected(
            backend,
            kind,
            a,
            av,
            ar,
            16,
            Some(initial.clone()),
        );
        let (b, bv, br) = section(split, time);
        let b = chunk::scan_selected(
            backend,
            kind,
            b,
            bv,
            br,
            16,
            Some(a.final_state),
        );
        let output = Tensor::cat(vec![a.output, b.output], 1);
        let scores = Tensor::cat(vec![a.scores, b.scores], 1);
        let mut error = compare(
            "Kron continuation output",
            &values(output.clone()),
            &expected_output,
            ATOL,
            RTOL,
        )?;
        error = error.max(compare(
            "Kron continuation state",
            &values(b.final_state.clone()),
            &expected_state,
            ATOL,
            RTOL,
        )?);
        error = error.max(compare(
            "Kron continuation score",
            &values(scores.clone()),
            &expected_scores,
            ATOL,
            RTOL,
        )?);
        let mut terms = Vec::new();
        if use_output {
            terms.push(
                (output
                    * Tensor::from_data(
                        TensorData::new(f.upstream.clone(), [f.b, time, f.h, f.v]),
                        device,
                    ))
                .sum(),
            );
        }
        if use_final {
            terms.push(
                (b.final_state
                    * Tensor::from_data(
                        TensorData::new(f.final_adjoint.clone(), [f.b, f.h, f.k, f.v]),
                        device,
                    ))
                .sum(),
            );
        }
        if use_scores {
            terms.push(
                (scores
                    * Tensor::from_data(TensorData::new(score_seed, [f.b, time, f.h, 1]), device))
                .sum(),
            );
        }
        let grads = Tensor::cat(terms, 0).sum().backward();
        let mut grad_error = compare(
            "continued initial-state VJP",
            &values(
                initial
                    .grad(&grads)
                    .ok_or("missing continued initial-state gradient")?,
            ),
            &expected_initial,
            ATOL,
            RTOL,
        )?;
        for (field, tensor) in x.iter().enumerate() {
            let actual = tensor
                .grad(&grads)
                .map(values)
                .unwrap_or_else(|| vec![0.; f.data[field].len()]);
            grad_error = grad_error.max(compare(
                &format!("continued input VJP field {field}"),
                &actual,
                &expected_grads[field],
                ATOL,
                RTOL,
            )?);
        }
        (error, grad_error)
    } else {
        (0., 0.)
    };
    Ok(KronCheck {
        mixer: kind,
        scan_backend: backend,
        tokens: time,
        strided_inputs: strided,
        objective: match mode {
            0 => "output+final_state+score",
            1 => "output_only",
            2 => "final_state_only",
            _ => "score_only",
        },
        output_max_abs: output_error,
        state_max_abs: state_error,
        score_max_abs: score_error,
        input_gradient_max_abs: gradient_error,
        initial_state_gradient_max_abs: initial_error,
        continuation_max_abs: continuation_error,
        continuation_gradient_max_abs: continuation_gradient_error,
    })
}

/// Pack a transposed physical buffer on the CPU, then expose the original
/// logical axes as a genuine noncontiguous device view.
fn strided_upload<const D: usize>(
    data: Vec<f32>,
    shape: [usize; D],
    a: usize,
    b: usize,
    device: &Device,
) -> Tensor<D> {
    let mut physical = shape;
    physical.swap(a, b);
    let mut packed = vec![0.; data.len()];
    for (index, value) in data.into_iter().enumerate() {
        let mut coordinates = [0; D];
        let mut remaining = index;
        for axis in (0..D).rev() {
            coordinates[axis] = remaining % shape[axis];
            remaining /= shape[axis];
        }
        coordinates.swap(a, b);
        let offset = coordinates
            .iter()
            .zip(physical)
            .fold(0, |offset, (&x, width)| offset * width + x);
        packed[offset] = value;
    }
    Tensor::from_data(TensorData::new(packed, physical), device).swap_dims(a, b)
}
