//! Independent sequential f64 reverse-mode oracle. Rotation uses Hamilton products,
//! not the production matrix construction. The scalar tape differentiates every input
//! and trainable parameter; no production tensor operation supplies reference gradients.
use super::*;
use burn::tensor::TensorData;
use serde::Serialize;

#[derive(Default)]
struct Tape {
    v: Vec<f64>,
    edges: Vec<Vec<(usize, f64)>>,
}
impl Tape {
    fn node(&mut self, v: f64, e: Vec<(usize, f64)>) -> usize {
        let i = self.v.len();
        self.v.push(v);
        self.edges.push(e);
        i
    }
    fn constant(&mut self, v: f64) -> usize {
        self.node(v, vec![])
    }
    fn add(&mut self, a: usize, b: usize) -> usize {
        self.node(self.v[a] + self.v[b], vec![(a, 1.), (b, 1.)])
    }
    fn mul(&mut self, a: usize, b: usize) -> usize {
        self.node(self.v[a] * self.v[b], vec![(a, self.v[b]), (b, self.v[a])])
    }
    fn scale(&mut self, a: usize, s: f64) -> usize {
        self.node(self.v[a] * s, vec![(a, s)])
    }
    fn bias(&mut self, a: usize, b: f64) -> usize {
        self.node(self.v[a] + b, vec![(a, 1.)])
    }
    fn rsqrt(&mut self, a: usize) -> usize {
        let v = self.v[a].sqrt().recip();
        self.node(v, vec![(a, -0.5 * v * v * v)])
    }
    fn sigmoid(&mut self, a: usize) -> usize {
        let v = 1. / (1. + (-self.v[a]).exp());
        self.node(v, vec![(a, v * (1. - v))])
    }
    fn sum(&mut self, a: &[usize]) -> usize {
        let mut y = self.constant(0.);
        for &x in a {
            y = self.add(y, x);
        }
        y
    }
    fn inv_norm(&mut self, x: &[usize], scale: f64, eps: f64) -> usize {
        let squares: Vec<_> = x.iter().map(|&v| self.mul(v, v)).collect();
        let s = self.sum(&squares);
        let s = self.scale(s, scale);
        let s = self.bias(s, eps);
        self.rsqrt(s)
    }
    fn gradients(&self, loss: usize) -> Vec<f64> {
        let mut g = vec![0.; self.v.len()];
        g[loss] = 1.;
        for i in (0..self.v.len()).rev() {
            for &(p, d) in &self.edges[i] {
                g[p] += g[i] * d;
            }
        }
        g
    }
}
fn hamilton(t: &mut Tape, a: &[usize], b: &[usize]) -> Vec<usize> {
    const TERMS: [[(usize, usize, f64); 4]; 4] = [
        [(0, 0, 1.), (1, 1, -1.), (2, 2, -1.), (3, 3, -1.)],
        [(0, 1, 1.), (1, 0, 1.), (2, 3, 1.), (3, 2, -1.)],
        [(0, 2, 1.), (1, 3, -1.), (2, 0, 1.), (3, 1, 1.)],
        [(0, 3, 1.), (1, 2, 1.), (2, 1, -1.), (3, 0, 1.)],
    ];
    TERMS
        .iter()
        .map(|terms| {
            let v: Vec<_> = terms
                .iter()
                .map(|&(i, j, s)| {
                    let v = t.mul(a[i], b[j]);
                    t.scale(v, s)
                })
                .collect();
            t.sum(&v)
        })
        .collect()
}
fn rotate(t: &mut Tape, q: &[usize], r: &[usize], x: &[usize]) -> Vec<usize> {
    let qx = hamilton(t, q, x);
    let rc: Vec<_> = r
        .iter()
        .enumerate()
        .map(|(i, &v)| t.scale(v, if i == 0 { 1. } else { -1. }))
        .collect();
    hamilton(t, &qx, &rc)
}
fn values<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.into_data().try_to_vec::<f32>().unwrap()
}
fn compare(label: &str, a: &[f32], e: &[f64]) -> Result<f64, String> {
    if a.len() != e.len() {
        return Err(format!("{label}: length"));
    }
    let mut max = 0.0f64;
    for (i, (&a, &e)) in a.iter().zip(e).enumerate() {
        let d = (a as f64 - e).abs();
        if !a.is_finite() || !e.is_finite() || d > 2e-5 + 3e-4 * e.abs() {
            return Err(format!("{label}[{i}] {a} != {e}; error {d}"));
        }
        max = max.max(d);
    }
    Ok(max)
}
fn parameters(init: &StreamInit, s: &OhcStage, h: &OhcReadout) -> Vec<Vec<f32>> {
    vec![
        values(init.scales.val()),
        values(init.offsets.val()),
        values(s.gate_projection.weight.val()),
        values(s.read_bias.val()),
        values(s.write_bias.val()),
        values(s.mix_bias.val()),
        values(s.read_scale.val()),
        values(s.write_gate_scale.val()),
        values(s.write_scale.val()),
        values(s.stream_norm.gamma.val()),
        values(h.gate_projection.weight.val()),
        values(h.gate_bias.val()),
        values(h.gate_scale.val()),
        values(h.norm.gamma.val()),
    ]
}
fn oracle(
    x: &[f32],
    extra: &[f32],
    p: &[Vec<f32>],
    valid: &[f32],
    d: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let mut t = Tape::default();
    let xi: Vec<_> = x.iter().map(|&v| t.constant(v as f64)).collect();
    let fi: Vec<_> = extra.iter().map(|&v| t.constant(v as f64)).collect();
    let pi: Vec<Vec<_>> = p
        .iter()
        .map(|v| v.iter().map(|&v| t.constant(v as f64)).collect())
        .collect();
    let mut out = Vec::new();
    let mut loss = t.constant(0.);
    for n in 0..valid.len() {
        let zero = t.constant(0.);
        let mut streams = vec![zero; 4 * d];
        if valid[n] != 0. {
            for s in 0..4 {
                for c in 0..d {
                    let v = t.mul(xi[n * d + c], pi[0][s * d + c]);
                    streams[s * d + c] = t.add(v, pi[1][s * d + c]);
                }
            }
        }
        let inv = t.inv_norm(&streams, 1. / (4 * d) as f64, 1e-6);
        let mut projected = Vec::new();
        for k in 0..16 {
            let terms: Vec<_> = (0..4 * d)
                .map(|j| t.mul(streams[j], pi[2][j * 16 + k]))
                .collect();
            let v = t.sum(&terms);
            projected.push(t.mul(v, inv));
        }
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut logits = Vec::new();
        for s in 0..4 {
            let v = t.mul(projected[s], pi[6][0]);
            let v = t.add(v, pi[3][s]);
            let v = t.sigmoid(v);
            a.push(t.bias(v, 1e-6));
            let v = t.mul(projected[4 + s], pi[7][0]);
            let v = t.add(v, pi[4][s]);
            let v = t.sigmoid(v);
            b.push(t.scale(v, 2.));
        }
        for k in 0..8 {
            let v = t.scale(projected[8 + k], 0.01);
            let v = t.add(v, pi[5][k]);
            logits.push(t.bias(v, if k % 4 == 0 { 1. } else { 0. }));
        }
        let iq = t.inv_norm(&logits[..4], 1., 1e-12);
        let ir = t.inv_norm(&logits[4..], 1., 1e-12);
        let q: Vec<_> = logits[..4].iter().map(|&v| t.mul(v, iq)).collect();
        let r: Vec<_> = logits[4..].iter().map(|&v| t.mul(v, ir)).collect();
        let norms: Vec<_> = (0..4)
            .map(|s| t.inv_norm(&streams[s * d..(s + 1) * d], 1. / d as f64, 1e-5))
            .collect();
        let mut work = Vec::new();
        for c in 0..d {
            let terms: Vec<_> = (0..4)
                .map(|s| {
                    let v = t.mul(streams[s * d + c], norms[s]);
                    let v = t.mul(v, pi[9][c]);
                    t.mul(v, a[s])
                })
                .collect();
            work.push(t.sum(&terms));
        }
        let mut written = vec![zero; 4 * d];
        for c in 0..d {
            let column: Vec<_> = (0..4).map(|s| streams[s * d + c]).collect();
            let mixed = rotate(&mut t, &q, &r, &column);
            let f = t.scale(work[c], 0.2);
            let f = t.add(f, fi[n * d + c]);
            for s in 0..4 {
                if valid[n] != 0. {
                    let v = t.mul(b[s], pi[8][0]);
                    let v = t.mul(v, f);
                    written[s * d + c] = t.add(mixed[s], v);
                }
            }
        }
        let inv = t.inv_norm(&written, 1. / (4 * d) as f64, 1e-5);
        let mut weights = Vec::new();
        for s in 0..4 {
            let terms: Vec<_> = (0..4 * d)
                .map(|j| t.mul(written[j], pi[10][j * 4 + s]))
                .collect();
            let v = t.sum(&terms);
            let v = t.mul(v, inv);
            let v = t.mul(v, pi[12][0]);
            let v = t.add(v, pi[11][s]);
            let v = t.sigmoid(v);
            weights.push(t.bias(v, 1e-5));
        }
        let combined: Vec<_> = (0..d)
            .map(|c| {
                let terms: Vec<_> = (0..4)
                    .map(|s| t.mul(written[s * d + c], weights[s]))
                    .collect();
                t.sum(&terms)
            })
            .collect();
        let inv = t.inv_norm(&combined, 1. / d as f64, 1e-5);
        for c in 0..d {
            let y = t.mul(combined[c], inv);
            let y = t.mul(y, pi[13][c]);
            out.push(t.v[y]);
            let z = t.scale(y, ((n * d + c + 1) as f64 * 0.37).sin());
            loss = t.add(loss, z);
            let z = t.scale(work[c], 0.017 * ((n * d + c + 2) as f64 * 0.19).cos());
            loss = t.add(loss, z);
        }
        for (j, &y) in written.iter().enumerate() {
            let z = t.scale(y, 0.013 * ((n * 4 * d + j + 3) as f64 * 0.11).sin());
            loss = t.add(loss, z);
        }
    }
    let g = t.gradients(loss);
    (
        out,
        xi.iter().map(|&i| g[i]).collect(),
        fi.iter().map(|&i| g[i]).collect(),
        pi.iter()
            .map(|v| v.iter().map(|&i| g[i]).collect())
            .collect(),
    )
}
#[derive(Serialize)]
pub struct Report {
    passed: bool,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    cases: Vec<serde_json::Value>,
    so4: serde_json::Value,
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
            OhcBackend::Tensor,
            OhcBackend::FusedGpu,
            OhcBackend::FusedReadWrite,
        ]
    } else {
        vec![OhcBackend::Tensor]
    };
    let write_cases = write_checks(&device, &modes)?;
    let read_cases = read_component_checks(&device, &modes)?;
    let mut cases = Vec::new();
    for (tokens, d) in [(1, 3), (5, 7), (17, 3)] {
        for zero in [false, true] {
            for strided in [false, true] {
                let batch = 2;
                let n = batch * tokens;
                device.seed(1771 + tokens as u64);
                let mut init = StreamInitConfig::new().with_d_model(d).init(&device);
                if zero {
                    init.offsets = init
                        .offsets
                        .map(|v| Tensor::zeros(v.dims(), &v.device()))
                        .set_require_grad(true);
                }
                let stage = OhcStageConfig::new().with_d_model(d).init(&device);
                let head = OhcReadoutConfig::new().with_d_model(d).init(&device);
                let raw: Vec<f32> = (0..n * d)
                    .map(|i| {
                        if zero {
                            0.
                        } else {
                            ((i + 1) as f32 * 0.13).sin() * 0.4
                        }
                    })
                    .collect();
                let x = if strided {
                    Tensor::<3>::from_data(TensorData::new(raw, [batch, d, tokens]), &device)
                        .swap_dims(1, 2)
                } else {
                    Tensor::<3>::from_data(TensorData::new(raw, [batch, tokens, d]), &device)
                }
                .require_grad();
                let extra = Tensor::<3>::from_data(
                    TensorData::new(
                        (0..n * d)
                            .map(|i| {
                                if zero {
                                    0.
                                } else {
                                    ((i + 3) as f32 * 0.23).cos() * 0.1
                                }
                            })
                            .collect::<Vec<_>>(),
                        [batch, tokens, d],
                    ),
                    &device,
                )
                .require_grad();
                let valid: Vec<f32> = (0..n)
                    .map(|i| {
                        if tokens > 1 && (i % tokens == tokens - 1 || i == 2) {
                            0.
                        } else {
                            1.
                        }
                    })
                    .collect();
                let mask = Tensor::<2>::from_data(
                    TensorData::new(valid.clone(), [batch, tokens]),
                    &device,
                );
                let params = parameters(&init, &stage, &head);
                let (expected, dx, df, dp) = oracle(
                    &values(x.clone()),
                    &values(extra.clone()),
                    &params,
                    &valid,
                    d,
                );
                let adj = Tensor::<3>::from_data(
                    TensorData::new(
                        (0..n * d)
                            .map(|i| ((i + 1) as f64 * 0.37).sin() as f32)
                            .collect::<Vec<_>>(),
                        [batch, tokens, d],
                    ),
                    &device,
                );
                let work_adj = Tensor::<3>::from_data(
                    TensorData::new(
                        (0..n * d)
                            .map(|i| (0.017 * ((i + 2) as f64 * 0.19).cos()) as f32)
                            .collect::<Vec<_>>(),
                        [batch, tokens, d],
                    ),
                    &device,
                );
                let write_adj = Tensor::<4>::from_data(
                    TensorData::new(
                        (0..n * 4 * d)
                            .map(|i| (0.013 * ((i + 3) as f64 * 0.11).sin()) as f32)
                            .collect::<Vec<_>>(),
                        [batch, tokens, 4, d],
                    ),
                    &device,
                );
                for &backend in &modes {
                    let streams = init.forward(x.clone(), mask.clone());
                    let read = stage.read_selected(streams, backend);
                    let work = read.work.clone();
                    let update = work.clone() * 0.2 + extra.clone();
                    let written = stage.write(read, update, mask.clone(), backend);
                    let y = head.readout(written.clone());
                    let mut max = compare("readout", &values(y.clone()), &expected)?;
                    let g = ((y * adj.clone()).sum()
                        + (work * work_adj.clone()).sum()
                        + (written * write_adj.clone()).sum())
                    .backward();
                    max = max.max(compare(
                        "dx",
                        &values(x.grad(&g).ok_or("missing dx")?),
                        &dx,
                    )?);
                    max = max.max(compare(
                        "operator input gradient",
                        &values(extra.grad(&g).ok_or("missing df")?),
                        &df,
                    )?);
                    macro_rules! check {
                        ($i:expr,$p:expr) => {
                            max = max.max(compare(
                                stringify!($p),
                                &values(
                                    $p.val()
                                        .grad(&g)
                                        .ok_or(concat!("missing ", stringify!($p)))?,
                                ),
                                &dp[$i],
                            )?)
                        };
                    }
                    check!(0, init.scales);
                    check!(1, init.offsets);
                    check!(2, stage.gate_projection.weight);
                    check!(3, stage.read_bias);
                    check!(4, stage.write_bias);
                    check!(5, stage.mix_bias);
                    check!(6, stage.read_scale);
                    check!(7, stage.write_gate_scale);
                    check!(8, stage.write_scale);
                    check!(9, stage.stream_norm.gamma);
                    check!(10, head.gate_projection.weight);
                    check!(11, head.gate_bias);
                    check!(12, head.gate_scale);
                    check!(13, head.norm.gamma);
                    // Every operation is token-local: changing one position cannot affect any other position,
                    // including positions across resets. Test actual production paths rather than the oracle.
                    let run = |z: Tensor<3>| {
                        let r = stage.read_selected(init.forward(z, mask.clone()), backend);
                        let f = r.work.clone() * 0.2 + extra.clone();
                        head.readout(stage.write(r, f, mask.clone(), backend))
                    };
                    let original = values(run(x.clone()));
                    let changed = x
                        .clone()
                        .slice_assign([0..1, 0..1, 0..d], Tensor::full([1, 1, d], 3., &device));
                    let changed = values(run(changed));
                    for i in d..n * d {
                        if original[i] != changed[i] {
                            return Err(format!("oHC crossed token/document boundary at {i}"));
                        }
                    }
                    cases.push(serde_json::json!({"backend":backend,"batch":batch,"tokens":tokens,"width":d,"zero_features_operator":zero,"strided_features":strided,"padding_and_short_documents":true,"all_14_parameter_gradients":true,"causality_max_abs":0,"max_abs_error":max}));
                }
            }
        }
    }
    // Random logits and exact identity: check the matrix, determinant and rotation norm.
    let rot = rotation::QuaternionPair::new(&device);
    let mut worst = 0.0f64;
    let mut det_error = 0.0f64;
    let mut norm_error = 0.0f64;
    for random in [false, true] {
        let logits: Vec<f32> = (0..13 * 8)
            .map(|i| {
                if random {
                    ((i * 17 + 3) as f32 * 0.31).sin() * 0.7
                } else {
                    0.
                }
            })
            .collect();
        let logits = Tensor::from_data(TensorData::new(logits, [1, 13, 8]), &device);
        let q: Vec<f32> = modes
            .iter()
            .flat_map(|mode| values(rot.rotation_selected(logits.clone(), mode.fused_read())))
            .collect();
        for m in q.chunks_exact(16) {
            for i in 0..4 {
                for j in 0..4 {
                    let dot: f64 = (0..4)
                        .map(|k| m[k * 4 + i] as f64 * m[k * 4 + j] as f64)
                        .sum();
                    worst = worst.max((dot - if i == j { 1. } else { 0. }).abs());
                }
            }
            let mut det = 0.;
            for a in 0..4 {
                for b in 0..4 {
                    for c in 0..4 {
                        for e in 0..4 {
                            let p = [a, b, c, e];
                            if (0..4).any(|i| (i + 1..4).any(|j| p[i] == p[j])) {
                                continue;
                            }
                            let inversions = (0..4)
                                .map(|i| (i + 1..4).filter(|&j| p[i] > p[j]).count())
                                .sum::<usize>();
                            det += if inversions % 2 == 0 { 1. } else { -1. }
                                * (0..4).map(|i| m[i * 4 + p[i]] as f64).product::<f64>();
                        }
                    }
                }
            }
            det_error = det_error.max((det - 1.).abs());
            let x = [0.3, -0.7, 0.2, 0.8];
            let before: f64 = x.iter().map(|v| v * v).sum();
            let after: f64 = (0..4)
                .map(|i| {
                    let v: f64 = (0..4).map(|j| m[i * 4 + j] as f64 * x[j]).sum();
                    v * v
                })
                .sum();
            norm_error = norm_error.max((before - after).abs());
            if !random {
                for i in 0..4 {
                    for j in 0..4 {
                        if m[i * 4 + j] != if i == j { 1. } else { 0. } {
                            return Err("zero logits not exact identity".into());
                        }
                    }
                }
            }
        }
    }
    if worst > 2e-5 || det_error > 2e-5 || norm_error > 2e-5 {
        return Err(format!("SO4 failed {worst} {det_error} {norm_error}"));
    }
    let init = StreamInitConfig::new().with_d_model(7).init(&device);
    let s = OhcStageConfig::new().with_d_model(7).init(&device);
    let v = Tensor::<2>::ones([1, 5], &device);
    let x = Tensor::<3>::random(
        [1, 5, 7],
        burn::tensor::Distribution::Uniform(-0.4, 0.4),
        &device,
    );
    let read = s.read(init.forward(x.clone(), v.clone()));
    let initial_matrix = values(read.mix.clone());
    let near_identity = initial_matrix
        .chunks_exact(16)
        .flat_map(|m| {
            (0..16).map(move |i| (m[i] as f64 - if i / 4 == i % 4 { 1. } else { 0. }).abs())
        })
        .fold(0., f64::max);
    if near_identity > 0.2 {
        return Err(format!(
            "initial rotation not near identity: {near_identity}"
        ));
    }
    let s = s.identity_init();
    let streams = init.forward(x, v.clone());
    let y = s.forward(streams.clone(), v, OhcBackend::Tensor, |u| u * 0.2);
    let identity = values((streams - y).abs().max())[0];
    if identity != 0. {
        return Err(format!("identity initialization: {identity}"));
    }
    let mask_values = vec![1., 0., 1., 0., 0., 0.];
    let valid = Tensor::<2>::from_data(TensorData::new(mask_values.clone(), [2, 3]), &device);
    let clean: Vec<f32> = (0..42)
        .map(|i| {
            if mask_values[i / 7] == 0. {
                0.
            } else {
                (i as f32 * 0.13).sin()
            }
        })
        .collect();
    let dirty: Vec<f32> = clean
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if mask_values[i / 7] == 0. {
                if i % 2 == 0 { f32::NAN } else { f32::INFINITY }
            } else {
                v
            }
        })
        .collect();
    let clean = Tensor::<3>::from_data(TensorData::new(clean, [2, 3, 7]), &device).require_grad();
    let dirty = Tensor::<3>::from_data(TensorData::new(dirty, [2, 3, 7]), &device).require_grad();
    let a = init.forward(clean.clone(), valid.clone());
    let b = init.forward(dirty.clone(), valid);
    let expected: Vec<f64> = values(a.clone()).iter().map(|&x| x as f64).collect();
    let mut padding_error = compare("dirty padding output", &values(b.clone()), &expected)?;
    let ga = a.square().sum().backward();
    let gb = b.square().sum().backward();
    for (label, a, b) in [
        (
            "dirty dx",
            values(clean.grad(&ga).ok_or("clean dx")?),
            values(dirty.grad(&gb).ok_or("dirty dx")?),
        ),
        (
            "dirty scale",
            values(init.scales.val().grad(&ga).ok_or("clean scales")?),
            values(init.scales.val().grad(&gb).ok_or("dirty scales")?),
        ),
        (
            "dirty offset",
            values(init.offsets.val().grad(&ga).ok_or("clean offsets")?),
            values(init.offsets.val().grad(&gb).ok_or("dirty offsets")?),
        ),
    ] {
        let expected: Vec<f64> = a.iter().map(|&v| v as f64).collect();
        padding_error = padding_error.max(compare(label, &b, &expected)?);
    }
    cases.push(serde_json::json!({"dirty_padding_nan_inf_and_empty_row":true,"max_abs_error":padding_error}));
    Ok(Report {
        passed: true,
        absolute_tolerance: 2e-5,
        relative_tolerance: 3e-4,
        cases: [cases, write_cases, read_cases].concat(),
        so4: serde_json::json!({"qtq_max_abs":worst,"determinant_error":det_error,"squared_norm_error":norm_error,"initial_q_minus_i_max_abs":near_identity,"identity_init_max_abs":identity}),
    })
}
#[cfg(test)]
#[test]
fn cpu_matches_independent_f64_and_is_token_local() {
    numerical_checks(&Device::ndarray()).unwrap();
}

fn write_checks(device: &Device, modes: &[OhcBackend]) -> Result<Vec<serde_json::Value>, String> {
    let mut cases = Vec::new();
    for strided in [false, true] {
        let (b, t, d) = (2, 3, 5);
        let raw = |n: usize, k: f32| {
            (0..n)
                .map(|i| ((i + 1) as f32 * k).sin() * 0.3)
                .collect::<Vec<_>>()
        };
        let x = if strided {
            Tensor::<4>::from_data(
                TensorData::new(raw(b * t * 4 * d, 0.13), [b, t, d, 4]),
                device,
            )
            .swap_dims(2, 3)
        } else {
            Tensor::<4>::from_data(
                TensorData::new(raw(b * t * 4 * d, 0.13), [b, t, 4, d]),
                device,
            )
        }
        .require_grad();
        let q = if strided {
            Tensor::<4>::from_data(TensorData::new(raw(b * t * 16, 0.17), [b, t, 4, 4]), device)
                .swap_dims(2, 3)
        } else {
            Tensor::<4>::from_data(TensorData::new(raw(b * t * 16, 0.17), [b, t, 4, 4]), device)
        }
        .require_grad();
        let w = if strided {
            Tensor::<3>::from_data(TensorData::new(raw(b * t * 4, 0.19), [b, 4, t]), device)
                .swap_dims(1, 2)
        } else {
            Tensor::<3>::from_data(TensorData::new(raw(b * t * 4, 0.19), [b, t, 4]), device)
        }
        .require_grad();
        let f = if strided {
            Tensor::<3>::from_data(TensorData::new(raw(b * t * d, 0.23), [b, d, t]), device)
                .swap_dims(1, 2)
        } else {
            Tensor::<3>::from_data(TensorData::new(raw(b * t * d, 0.23), [b, t, d]), device)
        }
        .require_grad();
        let u = if strided {
            Tensor::<4>::from_data(
                TensorData::new(raw(b * t * 4 * d, 0.29), [b, t, d, 4]),
                device,
            )
            .swap_dims(2, 3)
        } else {
            Tensor::<4>::from_data(
                TensorData::new(raw(b * t * 4 * d, 0.29), [b, t, 4, d]),
                device,
            )
        };
        let (xv, qv, wv, fv, uv) = (
            values(x.clone()),
            values(q.clone()),
            values(w.clone()),
            values(f.clone()),
            values(u.clone()),
        );
        let (mut y, mut dx, mut dq, mut dw, mut df) = (
            vec![0.; b * t * 4 * d],
            vec![0.; b * t * 4 * d],
            vec![0.; b * t * 16],
            vec![0.; b * t * 4],
            vec![0.; b * t * d],
        );
        for n in 0..b * t {
            for s in 0..4 {
                for c in 0..d {
                    let i = (n * 4 + s) * d + c;
                    let up = uv[i] as f64;
                    y[i] = wv[n * 4 + s] as f64 * fv[n * d + c] as f64;
                    dw[n * 4 + s] += up * fv[n * d + c] as f64;
                    df[n * d + c] += up * wv[n * 4 + s] as f64;
                    for k in 0..4 {
                        y[i] += qv[n * 16 + s * 4 + k] as f64 * xv[(n * 4 + k) * d + c] as f64;
                        dq[n * 16 + s * 4 + k] += up * xv[(n * 4 + k) * d + c] as f64;
                        dx[(n * 4 + k) * d + c] += up * qv[n * 16 + s * 4 + k] as f64;
                    }
                }
            }
        }
        for &backend in modes {
            let out = backend.write(
                x.clone(),
                q.clone(),
                w.clone(),
                f.clone(),
                Tensor::ones([1], device),
            );
            let mut max = compare("write", &values(out.clone()), &y)?;
            let g = (out * u.clone()).sum().backward();
            for (label, a, e) in [
                ("write dx", values(x.grad(&g).ok_or("write dx")?), &dx),
                ("write dq", values(q.grad(&g).ok_or("write dq")?), &dq),
                ("write dw", values(w.grad(&g).ok_or("write dw")?), &dw),
                ("write df", values(f.grad(&g).ok_or("write df")?), &df),
            ] {
                max = max.max(compare(label, &a, e)?);
            }
            cases.push(serde_json::json!({"write_only":true,"backend":backend,"strided_streams_rotation_gates_operator_adjoint":strided,"max_abs_error":max}));
        }
    }
    Ok(cases)
}

fn read_component_checks(
    device: &Device,
    modes: &[OhcBackend],
) -> Result<Vec<serde_json::Value>, String> {
    let mut cases = Vec::new();
    let (b, t, d) = (2, 3, 5);
    for strided in [false, true] {
        let raw = |n: usize, k: f32| {
            (0..n)
                .map(|i| ((i + 1) as f32 * k).sin() * 0.3)
                .collect::<Vec<_>>()
        };
        let tensor3 = |last: usize, k: f32| {
            if strided {
                Tensor::<3>::from_data(TensorData::new(raw(b * t * last, k), [b, last, t]), device)
                    .swap_dims(1, 2)
            } else {
                Tensor::<3>::from_data(TensorData::new(raw(b * t * last, k), [b, t, last]), device)
            }
        };
        let x = if strided {
            Tensor::<4>::from_data(
                TensorData::new(raw(b * t * 4 * d, 0.13), [b, t, d, 4]),
                device,
            )
            .swap_dims(2, 3)
        } else {
            Tensor::<4>::from_data(
                TensorData::new(raw(b * t * 4 * d, 0.13), [b, t, 4, d]),
                device,
            )
        }
        .require_grad();
        let p = tensor3(16, 0.17).require_grad();
        let a = tensor3(4, 0.19).require_grad();
        let logits = tensor3(8, 0.23).require_grad();
        let gamma = if strided {
            Tensor::<2>::from_data(TensorData::new(raw(2 * d, 0.29), [d, 2]), device)
                .narrow(1, 0, 1)
                .squeeze_dim::<1>(1)
        } else {
            Tensor::<1>::from_data(TensorData::new(raw(d, 0.29), [d]), device)
        }
        .require_grad();
        let (ug, ur, uq) = (tensor3(16, 0.31), tensor3(d, 0.37), tensor3(16, 0.41));
        let (ugv, urv, uqv) = (values(ug.clone()), values(ur.clone()), values(uq.clone()));
        let mut tape = Tape::default();
        let inputs: Vec<Vec<_>> = [
            values(x.clone()),
            values(p.clone()),
            values(a.clone()),
            values(gamma.clone()),
            values(logits.clone()),
        ]
        .iter()
        .map(|v| v.iter().map(|&z| tape.constant(z as f64)).collect())
        .collect();
        let mut expected = [Vec::new(), Vec::new(), Vec::new()];
        let mut loss = tape.constant(0.);
        let epsilon = 0.00023;
        for n in 0..b * t {
            let xs = &inputs[0][n * 4 * d..(n + 1) * 4 * d];
            let inv = tape.inv_norm(xs, 1. / (4 * d) as f64, 1e-6);
            for k in 0..16 {
                let y = tape.mul(inputs[1][n * 16 + k], inv);
                expected[0].push(tape.v[y]);
                let v = tape.scale(y, ugv[n * 16 + k] as f64);
                loss = tape.add(loss, v);
            }
            let norms: Vec<_> = (0..4)
                .map(|s| tape.inv_norm(&xs[s * d..(s + 1) * d], 1. / d as f64, epsilon))
                .collect();
            for c in 0..d {
                let terms: Vec<_> = (0..4)
                    .map(|s| {
                        let v = tape.mul(xs[s * d + c], norms[s]);
                        tape.mul(v, inputs[2][n * 4 + s])
                    })
                    .collect();
                let y = tape.sum(&terms);
                let y = tape.mul(y, inputs[3][c]);
                expected[1].push(tape.v[y]);
                let v = tape.scale(y, urv[n * d + c] as f64);
                loss = tape.add(loss, v);
            }
            let raw: Vec<_> = (0..8)
                .map(|k| tape.bias(inputs[4][n * 8 + k], if k % 4 == 0 { 1. } else { 0. }))
                .collect();
            let iq = tape.inv_norm(&raw[..4], 1., 1e-12);
            let ir = tape.inv_norm(&raw[4..], 1., 1e-12);
            let q: Vec<_> = raw[..4].iter().map(|&v| tape.mul(v, iq)).collect();
            let r: Vec<_> = raw[4..].iter().map(|&v| tape.mul(v, ir)).collect();
            let mut matrix = vec![0; 16];
            for j in 0..4 {
                let basis: Vec<_> = (0..4)
                    .map(|i| tape.constant(if i == j { 1. } else { 0. }))
                    .collect();
                let col = rotate(&mut tape, &q, &r, &basis);
                for i in 0..4 {
                    matrix[i * 4 + j] = col[i];
                }
            }
            for k in 0..16 {
                let y = matrix[k];
                expected[2].push(tape.v[y]);
                let v = tape.scale(y, uqv[n * 16 + k] as f64);
                loss = tape.add(loss, v);
            }
        }
        let grads = tape.gradients(loss);
        let expected_grads: Vec<Vec<_>> = inputs
            .iter()
            .map(|v| v.iter().map(|&i| grads[i]).collect())
            .collect();
        let mut norm = burn::nn::RmsNormConfig::new(d)
            .with_epsilon(epsilon)
            .init(device);
        norm.gamma = burn::module::Param::from_tensor(gamma.clone());
        let rotation = rotation::QuaternionPair::new(device);
        for &backend in modes {
            let gate = backend.gate_norm(x.clone(), p.clone());
            let read = backend.weighted_read(x.clone(), a.clone(), &norm);
            let matrix = rotation
                .rotation_selected(logits.clone(), backend.fused_read())
                .reshape([b, t, 16]);
            let mut max = 0.0f64;
            for (label, actual, expected) in [
                ("gate norm", values(gate.clone()), &expected[0]),
                ("weighted read", values(read.clone()), &expected[1]),
                ("quaternion", values(matrix.clone()), &expected[2]),
            ] {
                max = max.max(compare(label, &actual, expected)?);
            }
            let g = ((gate * ug.clone()).sum()
                + (read * ur.clone()).sum()
                + (matrix * uq.clone()).sum())
            .backward();
            let actual = [
                values(x.grad(&g).ok_or("read dx")?),
                values(p.grad(&g).ok_or("read dp")?),
                values(a.grad(&g).ok_or("read da")?),
                values(gamma.grad(&g).ok_or("read dgamma")?),
                values(logits.grad(&g).ok_or("read dlogits")?),
            ];
            for (actual, expected) in actual.iter().zip(&expected_grads) {
                max = max.max(compare("read all input gradients", actual, expected)?);
            }
            cases.push(serde_json::json!({"read_primitives_independent_f64":true,"backend":backend,"strided_streams_projected_gates_weights_gamma_logits_adjoints":strided,"custom_norm_epsilon":epsilon,"max_abs_error":max}));
        }
    }
    Ok(cases)
}
