//! Before any autotuned training, independently probe every active fused-LM linear and both VJPs.
use crate::config::Config;
#[cfg(any(feature = "vulkan", test))]
use crate::config::HeadBackend;
use anyhow::{Result, ensure};
use burn::prelude::*;
use delta_mixers::{HolaBackend, ProjectionBackend, ScanBackend};
#[cfg(any(feature = "vulkan", test))]
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub fn validate(c: &Config) -> Result<()> {
    ensure!(
        c.backend == "vulkan" && c.task == "lm" && !c.fusion,
        "guarded autotune is supported only for the unfused local Vulkan byte-LM"
    );
    ensure!(
        c.scan_backend == ScanBackend::FusedGpu && c.hola_backend == HolaBackend::FusedGpu,
        "autotune requires fused_gpu scan and HOLA; tensor_solver/tensor HOLA are not certified"
    );
    ensure!(
        matches!(
            c.projection_backend,
            ProjectionBackend::Merged | ProjectionBackend::MergedHeuristicOutput
        ),
        "autotune requires merged input projections so every executed GEMM is covered by preflight"
    );
    ensure!(
        c.normalization_backend.is_tensor(),
        "guarded autotune currently certifies tensor normalization/gating"
    );
    Ok(())
}

#[cfg(any(feature = "vulkan", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Role {
    MlpUp,
    MlpDown,
    MergedInput,
    MixerOutput,
    Head,
}
#[cfg(any(feature = "vulkan", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Shape {
    role: Role,
    batch: usize,
    context: usize,
    input: usize,
    output: usize,
    physical_output: usize,
    heuristic: bool,
}
#[cfg(any(feature = "vulkan", test))]
fn shapes(c: &Config) -> Result<[Shape; 5]> {
    let _ = c
        .batch_size
        .checked_mul(c.context)
        .ok_or_else(|| anyhow::anyhow!("GEMM row count overflow"))?;
    let doubled = c
        .d_model
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("MLP width overflow"))?;
    let key = c.heads * c.key_dim;
    let value = c.heads * c.value_dim;
    let make = |role, input, output, physical_output, heuristic| Shape {
        role,
        batch: c.batch_size,
        context: c.context,
        input,
        output,
        physical_output,
        heuristic,
    };
    Ok([
        make(Role::MlpUp, c.d_model, doubled, doubled, false),
        make(Role::MlpDown, doubled, c.d_model, c.d_model, false),
        make(
            Role::MergedInput,
            c.d_model,
            3 * key + 2 * value + 2 * c.heads,
            3 * key + 2 * value + 2 * c.heads,
            false,
        ),
        make(
            Role::MixerOutput,
            value,
            c.d_model,
            c.d_model,
            c.projection_backend == ProjectionBackend::MergedHeuristicOutput,
        ),
        make(
            Role::Head,
            c.d_model,
            c.vocab(),
            if c.head_backend == HeadBackend::Padded320 {
                320
            } else {
                c.vocab()
            },
            c.head_backend == HeadBackend::Heuristic,
        ),
    ])
}
#[cfg(any(feature = "vulkan", test))]
fn fixture(index: usize, seed: u32) -> f32 {
    let bits = (index as u32)
        .wrapping_mul(1_664_525)
        .wrapping_add(seed)
        .rotate_left(13);
    ((bits % 257) as i32 - 128) as f32 / 1024.
}
#[cfg(any(feature = "vulkan", test))]
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Expected {
    forward: [f64; 4],
    input_vjp: [f64; 4],
    weight_vjp: [f64; 4],
}
#[cfg(any(feature = "vulkan", test))]
fn reference(s: Shape) -> Expected {
    let rows = s.batch * s.context;
    let mut expected = Expected {
        forward: [0.; 4],
        input_vjp: [0.; 4],
        weight_vjp: [0.; 4],
    };
    for (q, (r, j)) in [
        (0, 0),
        (rows / 2, s.output / 2),
        (rows - 1, s.output - 1),
        (rows / 3, s.output / 3),
    ]
    .into_iter()
    .enumerate()
    {
        expected.forward[q] = (0..s.input)
            .map(|p| fixture(r * s.input + p, 3107) as f64 * fixture(p * s.output + j, 719) as f64)
            .sum();
    }
    for (q, (r, p)) in [
        (0, 0),
        (rows / 2, s.input / 2),
        (rows - 1, s.input - 1),
        (rows / 3, s.input / 3),
    ]
    .into_iter()
    .enumerate()
    {
        expected.input_vjp[q] = (0..s.output)
            .map(|j| fixture(r * s.output + j, 991) as f64 * fixture(p * s.output + j, 719) as f64)
            .sum();
    }
    for (q, (p, j)) in [
        (0, 0),
        (s.input / 2, s.output / 2),
        (s.input - 1, s.output - 1),
        (s.input / 3, s.output / 3),
    ]
    .into_iter()
    .enumerate()
    {
        expected.weight_vjp[q] = (0..rows)
            .map(|r| fixture(r * s.input + p, 3107) as f64 * fixture(r * s.output + j, 991) as f64)
            .sum();
    }
    expected
}
#[cfg(any(feature = "vulkan", test))]
fn compare(actual: &[f32; 4], expected: &[f64; 4]) -> (bool, f64) {
    actual
        .iter()
        .zip(expected)
        .fold((true, 0.), |(passed, maximum), (&a, &e)| {
            let error = (a as f64 - e).abs();
            (
                passed && a.is_finite() && error <= 2e-5 + 3e-4 * e.abs(),
                f64::max(maximum, error),
            )
        })
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use super::*;
    use crate::runtime;
    use anyhow::Context;
    use cubecl_environment::persistence::Database;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::time::Instant;
    #[derive(Serialize, Deserialize)]
    struct Certificate {
        schema: u32,
        identity: String,
        tune_cache_sha256: String,
        passed: bool,
        shapes: Vec<Shape>,
        expected: Vec<Expected>,
    }

    /// Read only matmul tuning entries, not compilation/session records. No cache is modified here.
    fn cache_digest() -> Result<Option<String>> {
        let db = Database::open_active().context("opening persistent GEMM tune cache")?;
        let mut namespaces: Vec<_> = db
            .summary()
            .into_iter()
            .filter(|n| n.namespace.starts_with("autotune/") && n.namespace.contains("matmul"))
            .collect();
        namespaces.sort_by(|a, b| a.namespace.cmp(&b.namespace));
        if namespaces.is_empty() {
            return Ok(None);
        }
        let mut digest = Sha256::new();
        for ns in namespaces {
            let mut entries: Vec<([u8; 32], [u8; 32])> = Vec::with_capacity(ns.entries as usize);
            db.scan(&ns.namespace, &mut |key, value| {
                entries.push((Sha256::digest(key).into(), Sha256::digest(value).into()));
            });
            ensure!(
                entries.len() as u64 == ns.entries,
                "incomplete matmul cache scan; refusing an unverifiable cache verdict"
            );
            entries.sort_unstable();
            digest.update((ns.namespace.len() as u64).to_le_bytes());
            digest.update(ns.namespace.as_bytes());
            for (key, value) in entries {
                digest.update(key);
                digest.update(value);
            }
        }
        Ok(Some(format!("{:x}", digest.finalize())))
    }
    fn collect3(tensor: Tensor<3>, points: [(usize, usize); 4], context: usize) -> [f32; 4] {
        let selected = points.map(|(row, col)| {
            tensor
                .clone()
                .slice([
                    row / context..row / context + 1,
                    row % context..row % context + 1,
                    col..col + 1,
                ])
                .reshape([1])
        });
        Tensor::cat(selected.into(), 0)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
            .try_into()
            .unwrap()
    }
    fn collect2(tensor: Tensor<2>, points: [(usize, usize); 4]) -> [f32; 4] {
        let selected = points.map(|(row, col)| {
            tensor
                .clone()
                .slice([row..row + 1, col..col + 1])
                .reshape([1])
        });
        Tensor::cat(selected.into(), 0)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
            .try_into()
            .unwrap()
    }
    fn probe(c: &Config, s: Shape, device: &Device) -> Result<[[f32; 4]; 3]> {
        let train = device.clone().autodiff();
        let rows = s.batch * s.context;
        let x = Tensor::<3>::from_data(
            TensorData::new(
                (0..rows * s.input)
                    .map(|i| fixture(i, 3107))
                    .collect::<Vec<_>>(),
                [s.batch, s.context, s.input],
            ),
            &train,
        )
        .require_grad();
        let weight = Tensor::<2>::from_data(
            TensorData::new(
                (0..s.input * s.output)
                    .map(|i| fixture(i, 719))
                    .collect::<Vec<_>>(),
                [s.input, s.output],
            ),
            &train,
        )
        .require_grad();
        let upstream = Tensor::<3>::from_data(
            TensorData::new(
                (0..rows * s.output)
                    .map(|i| fixture(i, 991))
                    .collect::<Vec<_>>(),
                [s.batch, s.context, s.output],
            ),
            &train,
        );
        let y = if s.role == Role::Head {
            crate::model::project_head(x.clone(), weight.clone(), c.head_backend)
        } else if s.heuristic {
            crate::model::heuristic_matmul(x.clone().reshape([rows, s.input]), weight.clone())
                .reshape([s.batch, s.context, s.output])
        } else {
            burn::tensor::module::linear(x.clone(), weight.clone(), None)
        };
        let grads = (y.clone() * upstream).sum().backward();
        let dx = x.grad(&grads).context("missing preflight input VJP")?;
        let dw = weight
            .grad(&grads)
            .context("missing preflight weight VJP")?;
        let forward = collect3(
            y,
            [
                (0, 0),
                (rows / 2, s.output / 2),
                (rows - 1, s.output - 1),
                (rows / 3, s.output / 3),
            ],
            s.context,
        );
        let input = collect3(
            dx,
            [
                (0, 0),
                (rows / 2, s.input / 2),
                (rows - 1, s.input - 1),
                (rows / 3, s.input / 3),
            ],
            s.context,
        );
        let weight = collect2(
            dw,
            [
                (0, 0),
                (s.input / 2, s.output / 2),
                (s.input - 1, s.output - 1),
                (s.input / 3, s.output / 3),
            ],
        );
        runtime::sync(device)?;
        Ok([forward, input, weight])
    }
    pub(super) fn run(c: &Config, device: &Device) -> Result<Value> {
        let started = Instant::now();
        let geometries = shapes(c)?;
        let identity = crate::data::hash(&serde_json::to_vec(
            &json!({"schema":1,"shapes":geometries,
            "head_backend":c.head_backend,"projection_backend":c.projection_backend,"device":runtime::identity(device),
            "guard_source":crate::data::hash(include_bytes!("gemm_guard.rs")),
            "model_source":crate::data::hash(include_bytes!("model.rs")),
            "operator_source":crate::data::hash(include_bytes!("../../kernels/src/matmul_ops.rs")),
            "projections_source":crate::data::hash(include_bytes!("../../mixers/src/projections.rs")),
            "dependency_lock":crate::data::hash(include_bytes!("../../../Cargo.lock")),"dtype":"f32"}),
        )?);
        let certificate_path = c
            .cache_dir
            .join("gemm-verdicts")
            .join(format!("{identity}.json"));
        let report_path = c.artifact_dir.join("gemm-preflight.json");
        let mut report = json!({"schema":1,"passed":false,"identity":identity,"device":runtime::identity(device),
            "shapes":geometries,"cases":[],"report_path":report_path,"certificate_path":certificate_path,
            "absolute_tolerance":2e-5,"relative_tolerance":3e-4,"fresh_device_probes":true,
            "scope":"Every unique active fused-LM linear: real rank3 forward, input VJP and parameter VJP; constant padded head columns excluded from loss/parameters. No tensor-solver/Tensor-HOLA certification.",
            "cache_protocol":"Warm all selections first so current candidate checksums are resolved; hash sorted raw matmul tune entries; reuse cached f64 expectations only for identical shapes, execution, driver and source/dependency identity; always recheck fresh device outputs before any real optimizer step."});
        runtime::write_json(&report_path, &report)?;
        let mut actual = Vec::with_capacity(geometries.len());
        for shape in geometries {
            ensure!(
                started.elapsed().as_secs_f64() < c.max_seconds.min(290.),
                "GEMM preflight deadline before training"
            );
            report["active_shape"] = json!(shape);
            runtime::write_json(&report_path, &report)?;
            actual.push(probe(c, shape, device)?);
        }
        let tune_cache_sha256 =
            cache_digest()?.context("no persistent matmul tune entries after preflight")?;
        let cached = std::fs::read(&certificate_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Certificate>(&bytes).ok())
            .filter(|old| {
                old.schema == 1
                    && old.passed
                    && old.identity == identity
                    && old.tune_cache_sha256 == tune_cache_sha256
                    && old.shapes == geometries
                    && old.expected.len() == geometries.len()
            });
        let cache_hit = cached.is_some();
        let expected = if let Some(old) = cached {
            old.expected
        } else {
            geometries.into_iter().map(reference).collect()
        };
        let mut passed = true;
        for ((shape, actual), expected) in geometries.into_iter().zip(actual).zip(&expected) {
            let f = compare(&actual[0], &expected.forward);
            let x = compare(&actual[1], &expected.input_vjp);
            let w = compare(&actual[2], &expected.weight_vjp);
            let ok = f.0 && x.0 && w.0;
            passed &= ok;
            report["cases"]
                .as_array_mut()
                .unwrap()
                .push(json!({"shape":shape,"passed":ok,
                "forward_max_abs":f.1,"input_vjp_max_abs":x.1,"weight_vjp_max_abs":w.1,
                "actual_f32":actual,"expected_f64":expected}));
        }
        report["passed"] = json!(passed);
        report["verdict_cache_hit"] = json!(cache_hit && passed);
        report["tune_cache_sha256"] = json!(tune_cache_sha256);
        report["elapsed_ms"] = json!(started.elapsed().as_secs_f64() * 1000.);
        report["sampled_reserved_bytes"] = json!(runtime::sampled_reserved(device));
        runtime::write_json(&report_path, &report)?;
        if passed {
            runtime::write_json(
                &certificate_path,
                &Certificate {
                    schema: 1,
                    identity,
                    tune_cache_sha256,
                    passed,
                    shapes: geometries.into(),
                    expected,
                },
            )?;
        }
        ensure!(
            passed,
            "GEMM preflight rejected this shape/tune-cache before training; unchanged f64 tolerances, see {}",
            report_path.display()
        );
        Ok(report)
    }
}
pub fn ensure(c: &Config, device: &Device) -> Result<Value> {
    validate(c)?;
    #[cfg(feature = "vulkan")]
    {
        vulkan::run(c, device)
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = device;
        anyhow::bail!("GEMM preflight requires the Vulkan feature")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn guard_requires_the_fused_merged_lm() {
        let mut c = Config::default();
        c.task = "lm".into();
        c.backend = "vulkan".into();
        c.scan_backend = ScanBackend::FusedGpu;
        c.hola_backend = HolaBackend::FusedGpu;
        c.projection_backend = ProjectionBackend::Merged;
        validate(&c).unwrap();
        let mut bad = c.clone();
        bad.scan_backend = ScanBackend::TensorSolver;
        assert!(validate(&bad).is_err());
        let mut bad = c.clone();
        bad.hola_backend = HolaBackend::Tensor;
        assert!(validate(&bad).is_err());
        let mut bad = c.clone();
        bad.projection_backend = ProjectionBackend::Grouped;
        assert!(validate(&bad).is_err());
        c.fusion = true;
        assert!(validate(&c).is_err());
    }
    #[test]
    fn guard_derives_all_actual_gemms_and_rejects_wrong_probes() {
        let mut c = Config::default();
        c.task = "lm".into();
        c.d_model = 256;
        c.batch_size = 8;
        c.context = 512;
        c.projection_backend = ProjectionBackend::MergedHeuristicOutput;
        c.head_backend = HeadBackend::Padded320;
        let shapes = shapes(&c).unwrap();
        assert_eq!(shapes.len(), 5);
        assert_eq!(shapes[2].output, 904);
        assert!(shapes[3].heuristic);
        assert_eq!(shapes[4].physical_output, 320);
        assert_eq!(shapes[4].output, 257);
        let expected = reference(shapes[0]);
        assert!(expected.forward.iter().all(|x| x.is_finite()));
        assert!(!compare(&[1.; 4], &[0.; 4]).0);
        assert!(!compare(&[f32::NAN; 4], &[0.; 4]).0);
        assert!(compare(&[0.; 4], &[0.; 4]).0);
    }
}
