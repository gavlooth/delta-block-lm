//! Isolated checkpoint/repeat diagnostics. Never writes a training run's metrics or checkpoints.
use crate::{
    checkpoint,
    config::Config,
    data::{self, Batch},
    model::{self, LanguageModel},
    runtime, train,
};
use anyhow::{Context, Result, ensure};
use burn::{
    module::{Module, ModuleVisitor, Param},
    prelude::*,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const ATOL: f64 = 1e-5;
const RTOL: f64 = 1e-4;
fn float_hash(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}
struct Parameters {
    path: Vec<String>,
    values: Vec<Value>,
    elements: usize,
    nonfinite: usize,
    error: Option<String>,
}
impl ModuleVisitor for Parameters {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_owned());
    }
    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let tensor = param.val();
        let shape = tensor.dims().to_vec();
        let dtype = format!("{:?}", tensor.dtype());
        let device = format!("{:?}", tensor.device());
        match tensor.into_data().try_to_vec_as::<f32>() {
            Ok(values) => {
                self.elements += values.len();
                self.nonfinite += values.iter().filter(|v| !v.is_finite()).count();
                self.values.push(json!({"slot":self.values.len(),"module_path":self.path.join("."),"param_id":param.id.val(),"shape":shape,"dtype":dtype,"device":device,"sha256":float_hash(&values)}));
            }
            Err(error) => self.error = Some(format!("parameter readback: {error:?}")),
        }
    }
}
fn parameters(model: &LanguageModel) -> Result<Value> {
    let mut visitor = Parameters {
        path: Vec::new(),
        values: Vec::new(),
        elements: 0,
        nonfinite: 0,
        error: None,
    };
    model.visit(&mut visitor);
    if let Some(error) = visitor.error {
        anyhow::bail!(error);
    }
    let signatures: Vec<_> = visitor
        .values
        .iter()
        .map(|p| {
            json!([
                p["slot"],
                p["module_path"],
                p["shape"],
                p["dtype"],
                p["sha256"]
            ])
        })
        .collect();
    Ok(
        json!({"sha256":data::hash(&serde_json::to_vec(&signatures)?),"parameter_tensors":visitor.values.len(),"elements":visitor.elements,"nonfinite_values":visitor.nonfinite,"parameters":visitor.values}),
    )
}
fn parameter_comparison(before: &Value, after: &Value) -> Value {
    let a = before["parameters"].as_array().expect("parameter snapshot");
    let b = after["parameters"].as_array().expect("parameter snapshot");
    let mut changed = Vec::new();
    for index in 0..a.len().max(b.len()) {
        let left = a.get(index);
        let right = b.get(index);
        let same = match (left, right) {
            (Some(x), Some(y)) => ["slot", "module_path", "shape", "dtype", "sha256"]
                .iter()
                .all(|field| x[*field] == y[*field]),
            _ => false,
        };
        if !same {
            changed.push(json!({"slot":index,"before":left,"after":right}));
        }
    }
    json!({"passed":changed.is_empty()&&before["nonfinite_values"]==0&&after["nonfinite_values"]==0,"bitwise_weights_equal":changed.is_empty(),"changed":changed,"parameter_ids_ignored":"record loading may replace identity; comparison uses ordered logical values, shape, dtype and module path"})
}
struct Forward {
    values: Vec<f32>,
    summary: Value,
}
fn forward(model: &LanguageModel, batch: &Batch, device: &Device) -> Result<Forward> {
    let logits = model.forward(batch);
    let shape = logits.dims().to_vec();
    let nll = model::scalar(model::loss(logits.clone(), batch))?;
    runtime::sync(device)?;
    let values = logits
        .into_data()
        .try_to_vec_as::<f32>()
        .map_err(|e| anyhow::anyhow!("logit readback: {e:?}"))?;
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite diagnostic logits"
    );
    let summary = json!({"shape":shape,"nll":nll,"sha256":float_hash(&values),"first_logits":&values[..values.len().min(16)]});
    Ok(Forward { values, summary })
}
fn logits_comparison(a: &Forward, b: &Forward) -> Result<Value> {
    ensure!(
        a.values.len() == b.values.len(),
        "roundtrip changed logit shape"
    );
    let mut max_abs = 0.0f64;
    let mut max_scaled = 0.0f64;
    let mut square_error = 0.0f64;
    let mut changed = 0usize;
    for (&x, &y) in a.values.iter().zip(&b.values) {
        ensure!(x.is_finite() && y.is_finite(), "nonfinite comparison input");
        let delta = (x as f64 - y as f64).abs();
        max_abs = max_abs.max(delta);
        square_error += delta * delta;
        max_scaled = max_scaled.max(delta / (ATOL + RTOL * (x.abs().max(y.abs()) as f64)));
        changed += usize::from(x.to_bits() != y.to_bits());
    }
    Ok(
        json!({"passed":max_scaled<=1.0,"bitwise_equal":changed==0,"changed_elements":changed,"max_abs_error":max_abs,"max_scaled_error":max_scaled,"rms_error":(square_error/a.values.len() as f64).sqrt(),"atol":ATOL,"rtol":RTOL,"nll_delta":b.summary["nll"].as_f64().unwrap()-a.summary["nll"].as_f64().unwrap()}),
    )
}
fn phase(
    c: &Config,
    arm: &crate::config::Arm,
    model: &LanguageModel,
    rows: &[data::Example],
    device: &Device,
    directory: &Path,
) -> Result<Value> {
    std::fs::create_dir_all(directory)?;
    let before = parameters(model)?;
    let train_batch = Batch::new(rows, &device.clone().autodiff());
    let eval_batch = Batch::new(rows, device);
    let training_mode = forward(model, &train_batch, device)?;
    let after_training_mode = parameters(model)?;
    let inference = model.valid();
    let after_valid = parameters(&inference)?;
    let first = forward(&inference, &eval_batch, device)?;
    let after_first = parameters(&inference)?;
    let second = forward(&inference, &eval_batch, device)?;
    let after_second = parameters(&inference)?;
    let checkpoint = directory.join("fresh-model.burnpack");
    ensure!(
        !checkpoint.exists(),
        "diagnostic checkpoint path must be fresh"
    );
    model.clone().save_file(&checkpoint)?;
    runtime::sync(device)?;
    let after_save = parameters(model)?;
    let load_target = LanguageModel::new(c, arm, &device.clone().autodiff())?;
    let loaded = checkpoint::load_model(load_target, &checkpoint, device)?;
    let loaded_parameters = parameters(&loaded)?;
    let loaded_inference = loaded.valid();
    let restored_first = forward(&loaded_inference, &eval_batch, device)?;
    let after_loaded_first = parameters(&loaded_inference)?;
    let restored_second = forward(&loaded_inference, &eval_batch, device)?;
    let after_loaded_second = parameters(&loaded_inference)?;
    let comparisons = json!({
        "source_training_vs_valid_logits":logits_comparison(&training_mode,&first)?,
        "source_repeat_logits":logits_comparison(&first,&second)?,
        "roundtrip_logits":logits_comparison(&first,&restored_first)?,
        "restored_repeat_logits":logits_comparison(&restored_first,&restored_second)?,
        "training_forward_parameter_mutation":parameter_comparison(&before,&after_training_mode),
        "valid_parameter_values":parameter_comparison(&before,&after_valid),
        "first_eval_parameter_mutation":parameter_comparison(&before,&after_first),
        "repeat_eval_parameter_mutation":parameter_comparison(&before,&after_second),
        "save_parameter_mutation":parameter_comparison(&before,&after_save),
        "roundtrip_parameter_values":parameter_comparison(&before,&loaded_parameters),
        "loaded_first_eval_parameter_mutation":parameter_comparison(&loaded_parameters,&after_loaded_first),
        "loaded_repeat_eval_parameter_mutation":parameter_comparison(&loaded_parameters,&after_loaded_second)
    });
    let passed = comparisons
        .as_object()
        .unwrap()
        .values()
        .all(|v| v["passed"] == true);
    let report = json!({"passed":passed,"load_guard":"synchronize constructed target before record application, then synchronize uploads","checkpoint":checkpoint,"fixture_sha256":data::fixture_hash(rows),"comparisons":comparisons,
        "forward":{"training_mode":training_mode.summary,"first_eval":first.summary,"repeat_eval":second.summary,"restored_first_eval":restored_first.summary,"restored_repeat_eval":restored_second.summary},
        "parameter_snapshots":{"before":before,"after_training_forward":after_training_mode,"after_valid":after_valid,"after_first_eval":after_first,"after_repeat_eval":after_second,"after_save":after_save,"loaded":loaded_parameters,"after_loaded_first_eval":after_loaded_first,"after_loaded_repeat_eval":after_loaded_second}});
    runtime::write_json(&directory.join("result.json"), &report)?;
    Ok(report)
}

pub fn checkpoint_check(
    c: &Config,
    device: &Device,
    diagnostic_backend: &str,
    arm_name: Option<&str>,
    seed: u64,
    checkpoint_override: Option<&Path>,
    output: Option<&Path>,
) -> Result<Value> {
    ensure!(
        checkpoint_override.is_none() || arm_name.is_some(),
        "--resume for a diagnostic requires --arm"
    );
    let corpus = train::loaded_corpus(c)?;
    let validation = train::evaluations(c, corpus.as_ref(), "validation")?;
    let identity = train::data_identity(c, corpus.as_ref(), &validation);
    let target_tokens = c.resolved_tokens()?;
    let contract =
        train::metrics_contract(c, &identity, &runtime::identity(device), target_tokens)?;
    let rows = &validation[..c.batch_size.min(validation.len())];
    let arms: Vec<_> = if let Some(name) = arm_name {
        vec![
            c.arms
                .iter()
                .find(|a| a.name == name)
                .context("diagnostic arm absent from config")?,
        ]
    } else {
        c.arms.iter().collect()
    };
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let directory = output.unwrap_or(&c.artifact_dir).join("checkpoint-diagnostics").join(format!(
        "{diagnostic_backend}-seed{seed}-{}-{stamp}",
        std::process::id()
    ));
    ensure!(
        !directory.exists(),
        "diagnostic output directory must be unique"
    );
    std::fs::create_dir_all(&directory)?;
    let mut reports = Vec::new();
    for arm in arms {
        runtime::sync(device)?;
        device.memory_cleanup();
        device.seed(seed);
        let latest = c
            .artifact_dir
            .join(format!("{}-seed{seed}/latest.json", arm.name));
        let source: Option<PathBuf> = if let Some(path) = checkpoint_override {
            Some(path.to_owned())
        } else if latest.exists() {
            let record: Value = serde_json::from_slice(&std::fs::read(&latest)?)?;
            Some(
                record["checkpoint"]
                    .as_str()
                    .context("invalid checkpoint pointer")?
                    .into(),
            )
        } else {
            None
        };
        let mut model = LanguageModel::new(c, arm, &device.clone().autodiff())?;
        if let Some(path) = &source {
            let state: Value = serde_json::from_slice(&std::fs::read(path.join("state.json"))?)?;
            ensure!(
                state["config_sha256"] == contract["config_sha256"]
                    && state["data"] == identity
                    && state["arm"] == arm.name
                    && state["seed"] == seed,
                "diagnostic checkpoint identity mismatch"
            );
            model = checkpoint::load_model(model, path.join("model.burnpack"), device)?;
        }
        let source_optimizer_path = source.as_ref().map(|path| path.join("optimizer.burnpack"));
        let mut optimizer = if let Some(path) = &source_optimizer_path {
            checkpoint::load_optimizer(c, path)?
        } else {
            train::optimizer(c)
        };
        let arm_dir = directory.join(&arm.name);
        let original = phase(c, arm, &model, rows, device, &arm_dir.join("source"))?;
        // One isolated real update reproduces optimizer-produced tensor layout, not a training outcome.
        let batch = Batch::new(rows, &device.clone().autodiff());
        let objective = model::loss(model.forward(&batch), &batch);
        let gradient = model::checked_gradients(&model, objective)?;
        model = optimizer.step(c.learning_rate, model, gradient);
        runtime::sync(device)?;
        let after_step = phase(
            c,
            arm,
            &model,
            rows,
            device,
            &arm_dir.join("after-one-diagnostic-adamw-step"),
        )?;
        let optimizer_bytes = optimizer.into_bytes()?;
        let optimizer_hash = data::hash(&optimizer_bytes);
        let optimizer_signature = checkpoint::optimizer_signature(c, optimizer_bytes.clone())?;
        let optimizer_checkpoint = arm_dir.join("optimizer-before-continuation.burnpack");
        ensure!(
            !optimizer_checkpoint.exists(),
            "diagnostic optimizer path must be unique"
        );
        std::fs::write(&optimizer_checkpoint, &*optimizer_bytes)?;
        let mut loaded_optimizer = train::optimizer(c).from_bytes(optimizer_bytes)?;
        let restored_optimizer_bytes = loaded_optimizer.into_bytes()?;
        let restored_optimizer_hash = data::hash(&restored_optimizer_bytes);
        let restored_optimizer_signature =
            checkpoint::optimizer_signature(c, restored_optimizer_bytes)?;
        let target = LanguageModel::new(c, arm, &device.clone().autodiff())?;
        let mut restored = checkpoint::load_model(
            target,
            after_step["checkpoint"]
                .as_str()
                .context("scratch checkpoint path")?,
            device,
        )?;
        let reference_gradient =
            model::checked_gradients(&model, model::loss(model.forward(&batch), &batch))?;
        let restored_gradient =
            model::checked_gradients(&restored, model::loss(restored.forward(&batch), &batch))?;
        model = optimizer.step(c.learning_rate, model, reference_gradient);
        restored = loaded_optimizer.step(c.learning_rate, restored, restored_gradient);
        runtime::sync(device)?;
        let continued_parameters =
            parameter_comparison(&parameters(&model)?, &parameters(&restored)?);
        let eval_batch = Batch::new(rows, device);
        let continued_logits = logits_comparison(
            &forward(&model.valid(), &eval_batch, device)?,
            &forward(&restored.valid(), &eval_batch, device)?,
        )?;
        let reference_state_after = checkpoint::optimizer_signature(c, optimizer.into_bytes()?)?;
        let restored_state_after =
            checkpoint::optimizer_signature(c, loaded_optimizer.into_bytes()?)?;
        let continuation = json!({"passed":optimizer_signature==restored_optimizer_signature&&reference_state_after==restored_state_after&&continued_parameters["passed"]==true&&continued_logits["passed"]==true,
            "optimizer_checkpoint":optimizer_checkpoint,"raw_optimizer_container_sha256":optimizer_hash,"raw_restored_optimizer_container_sha256":restored_optimizer_hash,
            "optimizer_semantic_state_exact_before_step":optimizer_signature==restored_optimizer_signature,
            "optimizer_semantic_state_exact_after_step":reference_state_after==restored_state_after,
            "optimizer_canonical_before":optimizer_signature,"restored_optimizer_canonical_before":restored_optimizer_signature,
            "reference_optimizer_canonical_after_step":reference_state_after,"restored_optimizer_canonical_after_step":restored_state_after,
            "next_step_parameters":continued_parameters,"next_step_logits":continued_logits});
        runtime::write_json(&arm_dir.join("optimizer-continuation.json"), &continuation)?;
        reports.push(json!({"arm":arm,"seed":seed,"source_checkpoint":source,"source_kind":if source.is_some(){"existing checkpoint loaded into device"}else {"fresh seeded initialization; no checkpoint available"},"source_optimizer_checkpoint":source_optimizer_path,"source_optimizer_semantic_state_validated":source.is_some(),
            "passed":original["passed"]==true&&after_step["passed"]==true&&continuation["passed"]==true,"source":original,"after_one_scratch_adamw_step":after_step,"optimizer_continuation":continuation}));
        let partial = json!({"schema":1,"command":"checkpoint-check","diagnostic_backend":diagnostic_backend,"source_backend":c.backend,"device":runtime::identity(device),"config":c,"data":identity,"diagnostic_directory":directory,"reports":reports});
        runtime::write_json(&directory.join("partial-results.json"), &partial)?;
    }
    let passed = reports.iter().all(|r| r["passed"] == true);
    let result = json!({"schema":1,"command":"checkpoint-check","passed":passed,"diagnostic_backend":diagnostic_backend,"source_backend":c.backend,"device":runtime::identity(device),"config":c,"data":identity,"diagnostic_directory":directory,"atol":ATOL,"rtol":RTOL,"reports":reports,
        "interpretation":"Synchronized, value/id-validated loading is applied to source models. Existing source optimizer records are restored and semantically validated before scratch steps; missing or silently skipped state is fatal. Repeated same-model evaluation and bitwise parameter hashes check model stability, while canonical complete optimizer tensor/scalar/config state and next-step parameters/logits check continuation. Raw container order is not semantic equality. Fresh unique files preserve previous failures. No original metrics/checkpoints are written, and no complete upstream GPU root-cause proof is claimed."});
    runtime::write_json(&directory.join("result.json"), &result)?;
    Ok(result)
}
