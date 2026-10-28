//! `ohc` acceptance tests (PLAN §7): `QᵀQ ≈ I` and `det Q ≈ 1` at initialization and for random gates,
//! stream diversity basics, padding and document separation, finite gradients, and results against an
//! independent `f64` oracle (PLAN §8.1). The oracle builds `Q` by direct Hamilton products `q x r̄`, not by the
//! sign matrix the tensor path uses. The tests run on ndarray; `ohc_stage_stream_init_and_readout_match_f64_oracle_cuda`
//! repeats the forward oracle on the Spark GPU, including the BF16 gate.

use burn::module::Param;
use burn::prelude::*;
use burn::tensor::TensorData;
use dre_core::{BatchLayout, DocumentLayout, gpu_device};

use super::*;

// ---------------------------------------------------------------- host helpers

/// SplitMix64 with Box–Muller normals, so fixtures are reproducible without a dependency.
struct HostRng(u64);

impl HostRng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normals(&mut self, count: usize, std: f64) -> Vec<f64> {
        (0..count)
            .map(|_| {
                let (u1, u2) = (self.uniform(), self.uniform());
                std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect()
    }
}

fn tensor<const D: usize>(values: &[f64], shape: [usize; D], device: &Device) -> Tensor<D> {
    let values: Vec<f32> = values.iter().map(|&v| v as f32).collect();
    Tensor::from_data(TensorData::new(values, shape), device)
}

fn host<const D: usize>(tensor: Tensor<D>) -> Vec<f64> {
    tensor
        .cast(burn::tensor::DType::F32)
        .into_data()
        .try_to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect()
}

fn rms(values: &[f64]) -> f64 {
    (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
}

/// PLAN §8.1 elementwise rule `|x − ref| ≤ atol + rtol·|ref|` with `atol = 1e-5·max(rms(ref), scale)`;
/// prints max absolute error, max relative error and p50/p99 of the error ratio.
fn assert_close(name: &str, got: &[f64], reference: &[f64], scale: f64) {
    assert_eq!(got.len(), reference.len(), "{name}: length differs");
    let bf16 = std::env::var_os("DRE_BF16_GATE").is_some();
    let rtol = if bf16 { 1e-2 } else { 1e-4 };
    let atol = if bf16 { 1e-2 } else { 1e-5 } * rms(reference).max(scale);
    let mut max_abs = 0.0f64;
    let mut max_rel: Option<f64> = None;
    let mut ratios = Vec::with_capacity(got.len());
    for (&x, &r) in got.iter().zip(reference) {
        assert!(x.is_finite(), "{name}: non-finite value {x}");
        let error = (x - r).abs();
        max_abs = max_abs.max(error);
        if r.abs() > atol {
            max_rel = Some(max_rel.unwrap_or(0.0).max(error / r.abs()));
        }
        ratios.push(error / (atol + rtol * r.abs()));
    }
    ratios.sort_by(f64::total_cmp);
    let quantile = |p: f64| ratios[((ratios.len() - 1) as f64 * p).round() as usize];
    let relative = max_rel.map_or("none: all |ref| <= atol".to_string(), |v| {
        format!("{v:.3e}")
    });
    println!(
        "{name}: max abs {max_abs:.3e}, max rel {relative}, ratio p50 {:.3e} p99 {:.3e} (atol {atol:.3e})",
        quantile(0.5),
        quantile(0.99)
    );
    assert!(
        *ratios.last().unwrap() <= 1.0,
        "{name}: worst error ratio {}",
        ratios.last().unwrap()
    );
}

// ---------------------------------------------------------------- f64 oracle

type Matrix4 = [[f64; 4]; 4];

fn hamilton(p: [f64; 4], q: [f64; 4]) -> [f64; 4] {
    // (w_p w_q − p⃗·q⃗, w_p q⃗ + w_q p⃗ + p⃗ × q⃗), oHC Eq. (14).
    [
        p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
        p[0] * q[1] + q[0] * p[1] + p[2] * q[3] - p[3] * q[2],
        p[0] * q[2] + q[0] * p[2] + p[3] * q[1] - p[1] * q[3],
        p[0] * q[3] + q[0] * p[3] + p[1] * q[2] - p[2] * q[1],
    ]
}

/// Matrix of `x ↦ q x r̄` from eight logits, column by column.
fn oracle_rotation(logits: &[f64]) -> Matrix4 {
    let unit = |v: [f64; 4]| {
        let norm = (v.iter().map(|x| x * x).sum::<f64>() + QUATERNION_EPSILON).sqrt();
        v.map(|x| x / norm)
    };
    let q = unit([1.0 + logits[0], logits[1], logits[2], logits[3]]);
    let r = unit([1.0 + logits[4], logits[5], logits[6], logits[7]]);
    let r_conjugate = [r[0], -r[1], -r[2], -r[3]];
    let mut matrix = [[0.0; 4]; 4];
    for k in 0..4 {
        let mut basis = [0.0; 4];
        basis[k] = 1.0;
        let column = hamilton(hamilton(q, basis), r_conjugate);
        for i in 0..4 {
            matrix[i][k] = column[i];
        }
    }
    matrix
}

fn det4(m: &Matrix4) -> f64 {
    let det3 = |rows: [usize; 3], cols: [usize; 3]| {
        let a = |i: usize, j: usize| m[rows[i]][cols[j]];
        a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1))
            - a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0))
            + a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0))
    };
    let cols = |skip: usize| {
        let mut out = [0usize; 3];
        let mut n = 0;
        for c in 0..4 {
            if c != skip {
                out[n] = c;
                n += 1;
            }
        }
        out
    };
    (0..4)
        .map(|j| if j % 2 == 0 { 1.0 } else { -1.0 } * m[0][j] * det3([1, 2, 3], cols(j)))
        .sum()
}

/// Largest `|QᵀQ − I|` entry and `|det Q − 1|` over `[N, 4, 4]` row-major matrices.
fn orthogonality_errors(matrices: &[f64]) -> (f64, f64) {
    let mut gram_error = 0.0f64;
    let mut det_error = 0.0f64;
    for chunk in matrices.chunks(16) {
        let m: Matrix4 = std::array::from_fn(|i| std::array::from_fn(|j| chunk[i * 4 + j]));
        for i in 0..4 {
            for j in 0..4 {
                let dot: f64 = (0..4).map(|k| m[k][i] * m[k][j]).sum();
                gram_error = gram_error.max((dot - if i == j { 1.0 } else { 0.0 }).abs());
            }
        }
        det_error = det_error.max((det4(&m) - 1.0).abs());
    }
    (gram_error, det_error)
}

fn sigmoid64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Host copy of an [`OhcStage`]'s parameters.
#[derive(Clone)]
struct StageOracle {
    d: usize,
    norm_gamma: Vec<f64>,
    norm_epsilon: f64,
    /// `[4d, 16]` row-major.
    weight: Vec<f64>,
    read_bias: Vec<f64>,
    write_bias: Vec<f64>,
    mix_bias: Vec<f64>,
    read_scale: f64,
    write_gate_scale: f64,
    mix_scale: f64,
    write_scale: f64,
}

impl StageOracle {
    fn from_stage(stage: &OhcStage) -> Self {
        let weight = stage.gate_projection.weight.val();
        assert_eq!(weight.dims(), [STREAMS * stage.d_model, 16]);
        Self {
            d: stage.d_model,
            norm_gamma: host(stage.stream_norm.gamma.val()),
            norm_epsilon: stage.stream_norm.epsilon,
            weight: host(weight),
            read_bias: host(stage.read_bias.val()),
            write_bias: host(stage.write_bias.val()),
            mix_bias: host(stage.mix_bias.val()),
            read_scale: host(stage.read_scale.val())[0],
            write_gate_scale: host(stage.write_gate_scale.val())[0],
            mix_scale: stage.mix_scale,
            write_scale: host(stage.write_scale.val())[0],
        }
    }

    /// One position: returns `(u, Q, a, b)`.
    fn gates_and_read(&self, x: &[f64]) -> (Vec<f64>, Matrix4, Vec<f64>, Vec<f64>) {
        let d = self.d;
        let flat_rms =
            (x.iter().map(|v| v * v).sum::<f64>() / (4 * d) as f64 + GATE_NORM_EPSILON).sqrt();
        let projected: Vec<f64> = (0..16)
            .map(|o| {
                (0..4 * d)
                    .map(|i| x[i] * self.weight[i * 16 + o])
                    .sum::<f64>()
                    / flat_rms
            })
            .collect();
        let a: Vec<f64> = (0..4)
            .map(|s| {
                sigmoid64(self.read_scale * projected[s] + self.read_bias[s]) + READ_WEIGHT_FLOOR
            })
            .collect();
        let b: Vec<f64> = (0..4)
            .map(|s| 2.0 * sigmoid64(self.write_gate_scale * projected[4 + s] + self.write_bias[s]))
            .collect();
        let logits: Vec<f64> = (0..8)
            .map(|k| self.mix_scale * projected[8 + k] + self.mix_bias[k])
            .collect();
        let q = oracle_rotation(&logits);
        let mut u = vec![0.0; d];
        for s in 0..4 {
            let stream = &x[s * d..(s + 1) * d];
            let stream_rms =
                (stream.iter().map(|v| v * v).sum::<f64>() / d as f64 + self.norm_epsilon).sqrt();
            for c in 0..d {
                u[c] += a[s] * stream[c] / stream_rms * self.norm_gamma[c];
            }
        }
        (u, q, a, b)
    }

    /// Positions `[P, 4, d]`, operator applied elementwise to `u`; returns `(u [P, d], X' [P, 4, d])`.
    fn forward(&self, x: &[f64], operator: fn(f64) -> f64) -> (Vec<f64>, Vec<f64>) {
        let d = self.d;
        let mut work = Vec::new();
        let mut out = Vec::new();
        for position in x.chunks(4 * d) {
            let (u, q, _, b) = self.gates_and_read(position);
            for s in 0..4 {
                for c in 0..d {
                    let mixed: f64 = (0..4).map(|k| q[s][k] * position[k * d + c]).sum();
                    out.push(mixed + self.write_scale * b[s] * operator(u[c]));
                }
            }
            work.extend(u);
        }
        (work, out)
    }
}

fn oracle_readout(readout: &OhcReadout, x: &[f64]) -> Vec<f64> {
    let d = readout.d_model;
    let eps = readout.epsilon;
    let weight = host(readout.gate_projection.weight.val());
    let bias = host(readout.gate_bias.val());
    let scale = host(readout.gate_scale.val())[0];
    let gamma = host(readout.norm.gamma.val());
    let mut out = Vec::new();
    for position in x.chunks(4 * d) {
        let flat_rms = (position.iter().map(|v| v * v).sum::<f64>() / (4 * d) as f64 + eps).sqrt();
        let weights: Vec<f64> = (0..4)
            .map(|s| {
                let p: f64 = (0..4 * d)
                    .map(|i| position[i] * weight[i * 4 + s])
                    .sum::<f64>()
                    / flat_rms;
                sigmoid64(p * scale + bias[s]) + eps
            })
            .collect();
        let combined: Vec<f64> = (0..d)
            .map(|c| (0..4).map(|s| weights[s] * position[s * d + c]).sum())
            .collect();
        let combined_rms = (combined.iter().map(|v| v * v).sum::<f64>() / d as f64 + eps).sqrt();
        out.extend(
            combined
                .iter()
                .zip(&gamma)
                .map(|(v, g)| v / combined_rms * g),
        );
    }
    out
}

/// Replaces the rotation biases and gate projection with large random values so `Q` is far from `I`.
fn randomize_gates(stage: OhcStage, rng: &mut HostRng, device: &Device) -> OhcStage {
    let mut stage = stage;
    let [inputs, outputs] = stage.gate_projection.weight.dims();
    stage.gate_projection.weight = Param::from_tensor(tensor(
        &rng.normals(inputs * outputs, 1.0),
        [inputs, outputs],
        device,
    ));
    stage.mix_bias = Param::from_tensor(tensor(
        &rng.normals(ROTATION_LOGITS, 1.5),
        [ROTATION_LOGITS],
        device,
    ));
    stage.read_bias = Param::from_tensor(tensor(&rng.normals(STREAMS, 1.0), [STREAMS], device));
    stage.write_bias = Param::from_tensor(tensor(&rng.normals(STREAMS, 1.0), [STREAMS], device));
    stage.mix_scale = 0.5;
    stage
}

// ---------------------------------------------------------------- tests

#[test]
fn ohc_quaternion_pair_is_special_orthogonal() {
    let device = Device::ndarray();
    let pair = QuaternionPair::new(&device);
    let mut rng = HostRng(11);
    let (batch, tokens) = (3, 64);

    // Random gates: logits large enough to reach rotations far from the identity.
    let logits = rng.normals(batch * tokens * ROTATION_LOGITS, 1.5);
    let rotations = host(pair.rotation(tensor(&logits, [batch, tokens, ROTATION_LOGITS], &device)));
    let (gram_error, det_error) = orthogonality_errors(&rotations);
    println!("random gates: max |QᵀQ − I| {gram_error:.3e}, max |det Q − 1| {det_error:.3e}");
    assert!(gram_error < 1e-5 && det_error < 1e-5);

    let reference: Vec<f64> = logits
        .chunks(ROTATION_LOGITS)
        .flat_map(|l| oracle_rotation(l).into_iter().flatten().collect::<Vec<_>>())
        .collect();
    assert_close(
        "quaternion pair vs Hamilton product",
        &rotations,
        &reference,
        0.5,
    );
    let farthest = rotations
        .chunks(16)
        .map(|m| {
            (0..4)
                .map(|i| (m[i * 4 + i] - 1.0).abs())
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max);
    assert!(
        farthest > 1.0,
        "random gates should include rotations far from I, got {farthest}"
    );

    // Zero logits: the identity, exactly (oHC §5).
    let identity = host(pair.rotation(Tensor::zeros([1, 2, ROTATION_LOGITS], &device)));
    for m in identity.chunks(16) {
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(m[i * 4 + j], if i == j { 1.0 } else { 0.0 });
            }
        }
    }
}

#[test]
fn ohc_mixer_at_init_is_near_identity_rotation() {
    let device = Device::ndarray();
    let mut rng = HostRng(12);
    for d in [24, 512] {
        let stage = OhcStageConfig::new().with_d_model(d).init(&device);
        let (batch, tokens) = (2, 16);
        let x = tensor(
            &rng.normals(batch * tokens * STREAMS * d, 1.0),
            [batch, tokens, STREAMS, d],
            &device,
        );
        let read = stage.read(x);
        assert_eq!(read.work.dims(), [batch, tokens, d]);
        assert_eq!(read.mix.dims(), [batch, tokens, 4, 4]);
        let mix = host(read.mix);
        let (gram_error, det_error) = orthogonality_errors(&mix);
        let distance = mix
            .chunks(16)
            .flat_map(|m| (0..16).map(move |e| (m[e] - if e % 5 == 0 { 1.0 } else { 0.0 }).abs()))
            .fold(0.0, f64::max);
        println!(
            "d={d}: max |QᵀQ − I| {gram_error:.3e}, max |det Q − 1| {det_error:.3e}, max |Q − I| {distance:.3e}"
        );
        assert!(gram_error < 1e-5 && det_error < 1e-5);
        assert!(
            distance > 0.0 && distance < 0.2,
            "Q at init should be near, not exactly, I: {distance}"
        );
        // Reference gate values at init: read weights ≈ σ(0) = 0.5, write weights ≈ 2σ(0) = 1.
        assert!(
            host(read.read_weights)
                .iter()
                .all(|a| (a - 0.5).abs() < 0.05)
        );
        assert!(
            host(read.write_weights)
                .iter()
                .all(|b| (b - 1.0).abs() < 0.1)
        );
    }
}

#[test]
fn ohc_random_gates_stay_special_orthogonal_and_preserve_norm() {
    let device = Device::ndarray();
    let mut rng = HostRng(13);
    let d = 24;
    let stage = randomize_gates(
        OhcStageConfig::new().with_d_model(d).init(&device),
        &mut rng,
        &device,
    );
    let (batch, tokens) = (2, 32);
    let x_host = rng.normals(batch * tokens * STREAMS * d, 1.0);
    let x = tensor(&x_host, [batch, tokens, STREAMS, d], &device);
    let read = stage.read(x);
    let mix = host(read.mix.clone());
    let (gram_error, det_error) = orthogonality_errors(&mix);
    println!("random stage gates: max |QᵀQ − I| {gram_error:.3e}, max |det Q − 1| {det_error:.3e}");
    assert!(gram_error < 1e-5 && det_error < 1e-5);

    // With f = 0 the write is Q X alone, which keeps ‖X_t‖_F per position (spec §2.1 guarantee).
    let out = host(stage.write(read, Tensor::zeros([batch, tokens, d], &device)));
    let norms = |v: &[f64]| {
        v.chunks(STREAMS * d)
            .map(|p| p.iter().map(|x| x * x).sum::<f64>().sqrt())
            .collect::<Vec<_>>()
    };
    assert_close("‖QX‖ vs ‖X‖", &norms(&out), &norms(&x_host), 1.0);
}

#[test]
fn ohc_stage_stream_init_and_readout_match_f64_oracle() {
    stage_stream_init_and_readout_match_f64_oracle(Device::ndarray(), dre_core::Precision::Fp32);
}

#[test]
#[ignore = "Spark CUDA; requires DRE_TEST_GPU=1"]
fn ohc_stage_stream_init_and_readout_match_f64_oracle_cuda() {
    assert_eq!(std::env::var("DRE_TEST_GPU").as_deref(), Ok("1"));
    let precision = if std::env::var_os("DRE_BF16_GATE").is_some() { dre_core::Precision::Bf16Mixed } else { dre_core::Precision::Fp32 };
    stage_stream_init_and_readout_match_f64_oracle(gpu_device(), precision);
}

fn stage_stream_init_and_readout_match_f64_oracle(device: Device, precision: dre_core::Precision) {
    let mut rng = HostRng(14);
    let d = 24;
    let (batch, tokens) = (2, 5);
    let layout = BatchLayout::new(
        DocumentLayout::one_document_per_row(&[5, 5], tokens),
        &device,
    );

    let init = StreamInitConfig::new().with_precision(precision).with_d_model(d).init(&device);
    let features_host = rng.normals(batch * tokens * d, 1.0);
    let streams = init.forward(tensor(&features_host, [batch, tokens, d], &device), &layout);
    let scales = host(init.scales.val());
    let offsets = host(init.offsets.val());
    let streams_reference: Vec<f64> = features_host
        .chunks(d)
        .flat_map(|e| {
            (0..STREAMS * d)
                .map(|i| e[i % d] * scales[i] + offsets[i])
                .collect::<Vec<_>>()
        })
        .collect();
    assert_close(
        "stream init",
        &host(streams.clone()),
        &streams_reference,
        1.0,
    );

    for (label, stage) in [
        (
            "init gates",
            OhcStageConfig::new().with_precision(precision).with_d_model(d).init(&device),
        ),
        (
            "random gates",
            randomize_gates(
                OhcStageConfig::new().with_precision(precision).with_d_model(d).init(&device),
                &mut rng,
                &device,
            ),
        ),
    ] {
        let oracle = StageOracle::from_stage(&stage);
        let streams_host = host(streams.clone());
        let read = stage.read(streams.clone());
        let work = host(read.work.clone());
        let output = read.work.clone().tanh();
        let out = host(stage.write(read, output));
        let (work_reference, out_reference) = oracle.forward(&streams_host, f64::tanh);
        assert_close(&format!("{label}: read u"), &work, &work_reference, 1.0);
        assert_close(&format!("{label}: write X'"), &out, &out_reference, 1.0);
    }

    let readout = OhcReadoutConfig::new().with_precision(precision).with_d_model(d).init(&device);
    let bank = host(readout.readout(streams.clone()));
    assert_close(
        "readout",
        &bank,
        &oracle_readout(&readout, &host(streams)),
        1.0,
    );
}

#[test]
fn ohc_padding_stays_zero_and_documents_stay_separate() {
    let device = Device::ndarray();
    let mut rng = HostRng(15);
    let d = 24;
    let tokens = 6;
    let layout = BatchLayout::new(
        DocumentLayout::packed(vec![vec![2, 3], vec![4]], tokens),
        &device,
    );
    let valid: Vec<bool> = host(layout.valid.clone())
        .into_iter()
        .map(|v| v == 1.0)
        .collect();
    let init = StreamInitConfig::new().with_d_model(d).init(&device);
    let stage = randomize_gates(
        OhcStageConfig::new().with_d_model(d).init(&device),
        &mut rng,
        &device,
    );
    let readout = OhcReadoutConfig::new().with_d_model(d).init(&device);

    let run = |features: &[f64]| {
        // Padded features are nonzero on purpose: the stream init must remove them.
        let streams = init.forward(tensor(features, [2, tokens, d], &device), &layout);
        let read = stage.read(streams.clone());
        let work = read.work.clone();
        let output = work.clone().tanh();
        let written = stage.write(read, output);
        let bank = readout.readout(written.clone());
        (host(streams), host(work), host(written), host(bank))
    };
    let features = rng.normals(2 * tokens * d, 1.0);
    let (streams, work, written, bank) = run(&features);
    for (position, &is_valid) in valid.iter().enumerate() {
        let groups = [
            (
                "streams",
                &streams[position * 4 * d..(position + 1) * 4 * d],
            ),
            ("read", &work[position * d..(position + 1) * d]),
            ("write", &written[position * 4 * d..(position + 1) * 4 * d]),
            ("readout", &bank[position * d..(position + 1) * d]),
        ];
        for (name, values) in groups {
            if is_valid {
                assert!(
                    values.iter().any(|&v| v != 0.0),
                    "{name} is zero at valid position {position}"
                );
            } else {
                assert!(
                    values.iter().all(|&v| v == 0.0),
                    "{name} is nonzero at padded position {position}"
                );
            }
        }
    }

    // Changing one token (row 0, position 1, first document) changes only that position. The first run
    // may have used autotune candidates; compare runs made after it, which use the selected kernels.
    let (_, _, written, bank) = run(&features);
    let (_, _, written_again, bank_again) = run(&features);
    assert!(
        written == written_again && bank == bank_again,
        "repeated runs differ"
    );
    let changed_position = 1;
    let mut changed = features.clone();
    for value in &mut changed[changed_position * d..(changed_position + 1) * d] {
        *value += 0.5;
    }
    let (_, _, written_changed, bank_changed) = run(&changed);
    for position in 0..2 * tokens {
        let same = written[position * 4 * d..(position + 1) * 4 * d]
            == written_changed[position * 4 * d..(position + 1) * 4 * d]
            && bank[position * d..(position + 1) * d]
                == bank_changed[position * d..(position + 1) * d];
        assert_eq!(same, position != changed_position, "position {position}");
    }
}

#[test]
fn ohc_stream_init_is_near_identical_but_distinct() {
    let device = Device::ndarray();
    let mut rng = HostRng(16);
    for d in [24, 512] {
        let tokens = 8;
        let layout = BatchLayout::new(
            DocumentLayout::one_document_per_row(&[tokens], tokens),
            &device,
        );
        let init = StreamInitConfig::new().with_d_model(d).init(&device);
        let streams = host(init.forward(
            tensor(&rng.normals(tokens * d, 1.0), [1, tokens, d], &device),
            &layout,
        ));
        let (mut lowest, mut highest) = (f64::INFINITY, f64::NEG_INFINITY);
        for position in streams.chunks(4 * d) {
            for i in 0..4 {
                for j in i + 1..4 {
                    let (a, b) = (&position[i * d..(i + 1) * d], &position[j * d..(j + 1) * d]);
                    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                    let cosine = dot / (rms(a) * rms(b) * d as f64);
                    lowest = lowest.min(cosine);
                    highest = highest.max(cosine);
                }
            }
        }
        println!("d={d}: pairwise stream cosine in [{lowest:.6}, {highest:.6}]");
        assert!(
            lowest > 0.99,
            "streams should start near-identical: {lowest}"
        );
        assert!(
            highest < 1.0 - 1e-6,
            "streams should start distinct: {highest}"
        );
    }
}

#[test]
fn ohc_gradients_match_f64_oracle() {
    let device = Device::ndarray().autodiff();
    let mut rng = HostRng(17);
    let d = 24;
    let tokens = 3;
    let stage = randomize_gates(
        OhcStageConfig::new().with_d_model(d).init(&device),
        &mut rng,
        &device,
    );
    let oracle = StageOracle::from_stage(&stage);
    let x_host = rng.normals(tokens * STREAMS * d, 1.0);
    let loss_weights = rng.normals(tokens * STREAMS * d, 1.0);

    let x = tensor(&x_host, [1, tokens, STREAMS, d], &device).require_grad();
    let out = stage.forward(x.clone(), |u| u.tanh());
    let loss = (out * tensor(&loss_weights, [1, tokens, STREAMS, d], &device)).sum();
    let grads = loss.backward();

    let oracle_loss = |oracle: &StageOracle, x: &[f64]| -> f64 {
        oracle
            .forward(x, f64::tanh)
            .1
            .iter()
            .zip(&loss_weights)
            .map(|(o, w)| o * w)
            .sum()
    };
    let step = |value: f64| 1e-6 * value.abs().max(1.0);
    let central = |plus: f64, minus: f64, h: f64| (plus - minus) / (2.0 * h);

    let x_reference: Vec<f64> = (0..x_host.len())
        .map(|i| {
            let h = step(x_host[i]);
            let (mut plus, mut minus) = (x_host.clone(), x_host.clone());
            plus[i] += h;
            minus[i] -= h;
            central(oracle_loss(&oracle, &plus), oracle_loss(&oracle, &minus), h)
        })
        .collect();
    assert_close("∂L/∂X", &host(x.grad(&grads).unwrap()), &x_reference, 1.0);

    let weight_reference: Vec<f64> = (0..oracle.weight.len())
        .map(|i| {
            let h = step(oracle.weight[i]);
            let (mut plus, mut minus) = (oracle.clone(), oracle.clone());
            plus.weight[i] += h;
            minus.weight[i] -= h;
            central(oracle_loss(&plus, &x_host), oracle_loss(&minus, &x_host), h)
        })
        .collect();
    let weight_grad = host(stage.gate_projection.weight.val().grad(&grads).unwrap());
    assert_close("∂L/∂W", &weight_grad, &weight_reference, 1.0);

    type Field = fn(&mut StageOracle) -> &mut f64;
    let scalars: [(&str, Field, Tensor<1>); 3] = [
        ("∂L/∂α_read", |o| &mut o.read_scale, stage.read_scale.val()),
        (
            "∂L/∂α_write",
            |o| &mut o.write_gate_scale,
            stage.write_gate_scale.val(),
        ),
        ("∂L/∂γ", |o| &mut o.write_scale, stage.write_scale.val()),
    ];
    for (name, field, parameter) in scalars {
        let (mut plus, mut minus) = (oracle.clone(), oracle.clone());
        let h = step(*field(&mut plus));
        *field(&mut plus) += h;
        *field(&mut minus) -= h;
        let reference = central(oracle_loss(&plus, &x_host), oracle_loss(&minus, &x_host), h);
        assert_close(
            name,
            &host(parameter.grad(&grads).unwrap()),
            &[reference],
            1.0,
        );
    }
    let bias_reference: Vec<f64> = (0..ROTATION_LOGITS)
        .map(|k| {
            let h = step(oracle.mix_bias[k]);
            let (mut plus, mut minus) = (oracle.clone(), oracle.clone());
            plus.mix_bias[k] += h;
            minus.mix_bias[k] -= h;
            central(oracle_loss(&plus, &x_host), oracle_loss(&minus, &x_host), h)
        })
        .collect();
    assert_close(
        "∂L/∂β_mix",
        &host(stage.mix_bias.val().grad(&grads).unwrap()),
        &bias_reference,
        1.0,
    );
}

#[test]
fn ohc_gradients_finite_through_padded_chain() {
    let device = Device::ndarray().autodiff();
    let mut rng = HostRng(18);
    let d = 24;
    let tokens = 7;
    let layout = BatchLayout::new(
        DocumentLayout::packed(vec![vec![3, 2], vec![4]], tokens),
        &device,
    );
    let init = StreamInitConfig::new().with_d_model(d).init(&device);
    let stages: Vec<OhcStage> = (0..2)
        .map(|_| OhcStageConfig::new().with_d_model(d).init(&device))
        .collect();
    let readout = OhcReadoutConfig::new().with_d_model(d).init(&device);

    let features =
        tensor(&rng.normals(2 * tokens * d, 1.0), [2, tokens, d], &device).require_grad();
    let mut streams = init.forward(features.clone(), &layout);
    for stage in &stages {
        streams = stage.forward(streams, |u| u.tanh() * layout.valid.clone());
    }
    let bank = readout.readout(streams);
    let loss = (bank * tensor(&rng.normals(2 * tokens * d, 1.0), [2, tokens, d], &device)).sum();
    let grads = loss.backward();

    let check = |name: &str, gradient: Option<Vec<f64>>| {
        let gradient = gradient.unwrap_or_else(|| panic!("{name}: no gradient"));
        assert!(
            gradient.iter().all(|g| g.is_finite()),
            "{name}: non-finite gradient"
        );
        assert!(
            gradient.iter().any(|&g| g != 0.0),
            "{name}: gradient is all zero"
        );
        println!(
            "{name}: ‖grad‖ = {:.3e}",
            gradient.iter().map(|g| g * g).sum::<f64>().sqrt()
        );
    };
    let feature_grad = host(features.grad(&grads).unwrap());
    let valid = host(layout.valid.clone());
    for (position, &v) in valid.iter().enumerate() {
        let g = &feature_grad[position * d..(position + 1) * d];
        if v == 0.0 {
            assert!(
                g.iter().all(|&x| x == 0.0),
                "gradient at padded position {position} must be exactly 0"
            );
        }
    }
    check("features", Some(feature_grad));
    check("stream scales", init.scales.val().grad(&grads).map(host));
    check("stream offsets", init.offsets.val().grad(&grads).map(host));
    for (index, stage) in stages.iter().enumerate() {
        check(
            &format!("stage {index} W"),
            stage.gate_projection.weight.val().grad(&grads).map(host),
        );
        check(
            &format!("stage {index} α_read"),
            stage.read_scale.val().grad(&grads).map(host),
        );
        check(
            &format!("stage {index} α_write"),
            stage.write_gate_scale.val().grad(&grads).map(host),
        );
        check(
            &format!("stage {index} γ"),
            stage.write_scale.val().grad(&grads).map(host),
        );
        check(
            &format!("stage {index} β_read"),
            stage.read_bias.val().grad(&grads).map(host),
        );
        check(
            &format!("stage {index} β_write"),
            stage.write_bias.val().grad(&grads).map(host),
        );
        check(
            &format!("stage {index} β_mix"),
            stage.mix_bias.val().grad(&grads).map(host),
        );
        check(
            &format!("stage {index} norm"),
            stage.stream_norm.gamma.val().grad(&grads).map(host),
        );
    }
    check(
        "readout W",
        readout.gate_projection.weight.val().grad(&grads).map(host),
    );
    check("readout β", readout.gate_bias.val().grad(&grads).map(host));
    check("readout s", readout.gate_scale.val().grad(&grads).map(host));
    check(
        "readout norm",
        readout.norm.gamma.val().grad(&grads).map(host),
    );
}

#[test]
fn exp004_ohc_stream_init_dirty_padding_preserves_parameter_gradients_cpu() {
    let device=Device::ndarray().autodiff();
    device.seed(2721);
    let width=3;
    let init=StreamInitConfig::new().with_d_model(width).init(&device);
    for documents in [vec![vec![2,1],vec![1],vec![],vec![]],vec![vec![];4]] {
        let host_layout=DocumentLayout::packed(documents,4).with_document_padding(4);
        let valid=host_layout.document_metadata_host().source_valid;
        let layout=BatchLayout::new(host_layout,&device);
        let clean=(0..4*4*width).map(|index|if valid[index/width]==0.0 {0.0}
            else {(index%11) as f32*0.1-0.4}).collect::<Vec<_>>();
        let sentinels=[f32::NAN,f32::INFINITY,f32::NEG_INFINITY,1e30,-1e30];
        let dirty=clean.iter().enumerate().map(|(index,&value)|if valid[index/width]==0.0 {
            sentinels[index%sentinels.len()]
        } else {value}).collect::<Vec<_>>();
        let run=|values:&[f32]| {
            let input=Tensor::<3>::from_data(TensorData::new(values.to_vec(),[4,4,width]),&device).require_grad();
            let output=init.forward(input.clone(),&layout);
            let gradients=output.clone().square().sum().backward();
            (host(output),host(input.grad(&gradients).expect("input gradient")),
                host(init.scales.val().grad(&gradients).expect("scale gradient")),
                host(init.offsets.val().grad(&gradients).expect("offset gradient")))
        };
        let clean=run(&clean);let dirty=run(&dirty);
        assert_eq!(dirty,clean,"inactive values must not affect outputs or any trainable gradient");
        for values in [&dirty.0,&dirty.1,&dirty.2,&dirty.3] {
            assert!(values.iter().all(|value|value.is_finite()));
        }
        for (position,&active) in valid.iter().enumerate() {
            if active==0.0 {
                assert!(dirty.0[position*STREAMS*width..(position+1)*STREAMS*width].iter().all(|&value|value==0.0));
                assert!(dirty.1[position*width..(position+1)*width].iter().all(|&value|value==0.0));
            }
        }
        if valid.iter().all(|&active|active==0.0) {
            assert!(dirty.2.iter().chain(&dirty.3).all(|&value|value==0.0));
        } else {
            assert!(dirty.1.iter().any(|&value|value!=0.0));
            assert!(dirty.2.iter().any(|&value|value!=0.0));
            assert!(dirty.3.iter().any(|&value|value!=0.0));
        }
    }
}
