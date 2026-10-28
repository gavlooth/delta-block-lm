//! Evaluation-only certification from explicitly pinned original checkpoints and training evidence.
use crate::{checkpoint, config::Config, data, model::LanguageModel, runtime, train};
use anyhow::{Result, ensure};
use burn::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: u32,
    source_config_sha256: String,
    entries: Vec<Entry>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    arm: String,
    seed: u64,
    checkpoint: PathBuf,
    checkpoint_sha256: String,
    state_sha256: String,
    training_metrics_path: PathBuf,
    training_metrics_sha256: String,
}

fn read_training_evidence(path: &Path, expected_sha256: &str) -> Result<(Vec<u8>, Value)> {
    let bytes = std::fs::read(path)?;
    ensure!(
        data::hash(&bytes) == expected_sha256,
        "original training evidence bytes changed"
    );
    let value = serde_json::from_slice(&bytes)?;
    Ok((bytes, value))
}
fn finite_evaluation(value: &Value) -> bool {
    ["nll", "perplexity", "accuracy"]
        .iter()
        .all(|field| value[*field].as_f64().is_some_and(f64::is_finite))
}
fn fresh_baseline(
    c: &Config,
    arm: &crate::config::Arm,
    seed: u64,
    rows: &[data::Example],
    device: &Device,
) -> Result<Value> {
    runtime::sync(device)?;
    device.seed(seed);
    let clock = Instant::now();
    let initial = LanguageModel::new(c, arm, &device.clone().autodiff())?;
    runtime::sync(device)?;
    let construction_seconds = clock.elapsed().as_secs_f64();
    let clock = Instant::now();
    let first = train::evaluate(&initial, c, rows, device)?;
    let repeated = train::evaluate(&initial, c, rows, device)?;
    ensure!(
        finite_evaluation(&first) && finite_evaluation(&repeated),
        "nonfinite regenerated baseline"
    );
    runtime::sync(device)?;
    Ok(
        json!({"passed":first==repeated,"seed":seed,"validation":first,"repeated_validation":repeated,
        "construction_and_sync_seconds":construction_seconds,"heldout_evaluation_seconds":clock.elapsed().as_secs_f64(),
        "definition":"Regenerated fresh seeded initialization under the current configuration/backend; not a verified bitwise replay of historical initial weights, whose fingerprint was not recorded."}),
    )
}

pub fn run(c: &Config, config_path: &Path, device: &Device, manifest_path: &Path) -> Result<Value> {
    let manifest_bytes = std::fs::read(manifest_path)?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
    ensure!(
        manifest.schema == 2
            && checkpoint::file_sha256(config_path)? == manifest.source_config_sha256,
        "certification source configuration changed or manifest obsolete; regenerate schema2 with the batch script"
    );
    ensure!(
        c.seeds.len() == 3 && c.seeds.iter().collect::<BTreeSet<_>>().len() == 3,
        "certification requires exactly three paired seeds"
    );
    ensure!(
        manifest.entries.len() == c.arms.len() * c.seeds.len(),
        "manifest must explicitly cover every configured arm and seed"
    );
    let arms: BTreeMap<_, _> = c.arms.iter().map(|a| (a.name.as_str(), a)).collect();
    let mut seen = BTreeSet::new();
    for entry in &manifest.entries {
        ensure!(
            arms.contains_key(entry.arm.as_str())
                && c.seeds.contains(&entry.seed)
                && seen.insert((entry.arm.as_str(), entry.seed)),
            "unknown/duplicate certification arm or seed"
        );
    }
    let corpus = train::loaded_corpus(c)?;
    let validation = train::evaluations(c, corpus.as_ref(), "validation")?;
    let identity = train::data_identity(c, corpus.as_ref(), &validation);
    let target_tokens = c.resolved_tokens()?;
    let mut source_contract =
        train::metrics_contract(c, &identity, &runtime::identity(device), target_tokens)?;
    // This exception applies ONLY to immutable original training provenance, never an evaluation cache.
    source_contract
        .as_object_mut()
        .unwrap()
        .remove("measurement_implementation_version");
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let directory = c.artifact_dir.join("certified-evaluations").join(format!(
        "{}-{}-{stamp}",
        c.backend,
        std::process::id()
    ));
    ensure!(
        !directory.exists(),
        "certification namespace must be unique"
    );
    std::fs::create_dir_all(&directory)?;
    let mut runs = Vec::new();
    let mut certificates = Vec::new();
    for entry in &manifest.entries {
        let arm = arms[entry.arm.as_str()];
        let model_path = entry.checkpoint.join("model.burnpack");
        let state_path = entry.checkpoint.join("state.json");
        let state: Value = serde_json::from_slice(&std::fs::read(&state_path)?)?;
        checkpoint::validate_state_metadata(&state)?;
        ensure!(
            checkpoint::file_sha256(&model_path)? == entry.checkpoint_sha256
                && checkpoint::file_sha256(&state_path)? == entry.state_sha256,
            "pinned checkpoint/state changed"
        );
        let (metrics_bytes, training_metrics) =
            read_training_evidence(&entry.training_metrics_path, &entry.training_metrics_sha256)?;
        let raw_metrics_copy = directory.join(format!(
            "{}-seed{}-original-metrics.json",
            entry.arm, entry.seed
        ));
        std::fs::write(&raw_metrics_copy, &metrics_bytes)?;
        source_contract["arm"] = serde_json::to_value(arm)?;
        source_contract["seed"] = json!(entry.seed);
        source_contract["training_algorithm"] = json!(arm.training_algorithm(c));
        source_contract["custom_gpu_kernel_proof"] = json!(arm.uses_custom_gpu(c));
        for (field, value) in source_contract.as_object().unwrap() {
            ensure!(
                training_metrics.get(field) == Some(value),
                "original training evidence mismatch in {field}"
            );
        }
        ensure!(
            training_metrics["completed"] == true && training_metrics["tokens"] == target_tokens,
            "original training budget incomplete"
        );
        ensure!(
            state["config_sha256"] == source_contract["config_sha256"]
                && state["data"] == identity
                && state["arm"] == entry.arm
                && state["seed"] == entry.seed
                && state["tokens"] == target_tokens,
            "explicit checkpoint identity/budget mismatch"
        );
        runtime::sync(device)?;
        device.memory_cleanup();
        let baseline = fresh_baseline(c, arm, entry.seed, &validation, device)?;
        let clock = Instant::now();
        let target = LanguageModel::new(c, arm, &device.clone().autodiff())?;
        let model = checkpoint::load_model(target, &model_path, device)?;
        let load_seconds = clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        let first = train::evaluate(&model, c, &validation, device)?;
        let repeated = train::evaluate(&model, c, &validation, device)?;
        let (test, repeated_test) = if corpus.is_some() {
            let rows = train::evaluations(c, corpus.as_ref(), "test")?;
            (
                Some(train::evaluate(&model, c, &rows, device)?),
                Some(train::evaluate(&model, c, &rows, device)?),
            )
        } else {
            (None, None)
        };
        let retrieval = if corpus.is_some() {
            Some(train::evaluate(
                &model,
                c,
                &crate::lm_retrieval::examples(c),
                device,
            )?)
        } else {
            None
        };
        for metrics in [&first, &repeated]
            .into_iter()
            .chain(test.iter())
            .chain(repeated_test.iter())
            .chain(retrieval.iter())
        {
            ensure!(finite_evaluation(metrics), "nonfinite certified evaluation");
        }
        let evaluation_seconds = clock.elapsed().as_secs_f64();
        let immutable = checkpoint::file_sha256(&model_path)? == entry.checkpoint_sha256
            && checkpoint::file_sha256(&state_path)? == entry.state_sha256
            && checkpoint::file_sha256(&entry.training_metrics_path)?
                == entry.training_metrics_sha256;
        let passed =
            baseline["passed"] == true && first == repeated && test == repeated_test && immutable;
        let certificate_path = directory.join(format!("{}-seed{}.json", entry.arm, entry.seed));
        let certificate = json!({"schema":1,"measurement_implementation_version":train::MEASUREMENT_VERSION,"passed":passed,
            "source_config_sha256":manifest.source_config_sha256,"config_sha256":source_contract["config_sha256"],"config":c,"data":identity,"device":runtime::identity(device),"arm":arm,"seed":entry.seed,
            "checkpoint":entry.checkpoint,"checkpoint_sha256":entry.checkpoint_sha256,"checkpoint_state_sha256":entry.state_sha256,"source_files_immutable":immutable,
            "logical_parameters_and_ids_match_saved_record":true,"validation":first,"repeated_validation":repeated,"repeated_evaluation_exact":first==repeated,"test":test,"repeated_test":repeated_test,"test_repeat_exact":if test.is_some(){Some(test==repeated_test)}else {None},"byte_associative_retrieval_diagnostic":retrieval,
            "initial_validation":baseline["validation"],"initial_validation_trusted":baseline["passed"]==true,"regenerated_baseline":baseline,
            "checkpoint_load_seconds":load_seconds,"heldout_evaluation_seconds":evaluation_seconds,"new_training_tokens":0,
            "training_metrics_source":entry.training_metrics_path,"training_metrics_sha256":entry.training_metrics_sha256,"original_training_metrics_bytes_copy":raw_metrics_copy,"original_training_metrics_snapshot":training_metrics,
            "training_measurement_note":"Original training file bytes are authoritative and copied verbatim; throughput/state-memory evidence is not remeasured. Historical pre-fix initial/final evaluations are uncertified. Current learning comparison uses a regenerated repeat-stable baseline, not assumed historical initial weights."});
        runtime::write_json(&certificate_path, &certificate)?;
        let mut certified_run = training_metrics.clone();
        certified_run["uncertified_original_initial_validation"] =
            certified_run["initial_validation"].clone();
        certified_run["initial_validation"] = baseline["validation"].clone();
        certified_run["initial_validation_trusted"] = json!(baseline["passed"] == true);
        certified_run["uncertified_original_validation"] = certified_run["validation"].clone();
        certified_run["validation"] = first;
        certified_run["test"] = json!(test);
        certified_run["byte_associative_retrieval_diagnostic"] = json!(retrieval);
        certified_run["measurement_implementation_version"] = json!(train::MEASUREMENT_VERSION);
        certified_run["evaluation_certification"] = json!({"passed":passed,"certificate":certificate_path,"checkpoint_sha256":entry.checkpoint_sha256,"training_metrics_sha256":entry.training_metrics_sha256,"new_training_tokens":0});
        runs.push(certified_run);
        certificates.push(json!({"arm":entry.arm,"seed":entry.seed,"passed":passed,"certificate":certificate_path}));
        runtime::write_json(
            &directory.join("partial-results.json"),
            &json!({"measurement_implementation_version":train::MEASUREMENT_VERSION,"certificates":certificates}),
        )?;
    }
    let passed = certificates.iter().all(|c| c["passed"] == true);
    let mut comparison = train::comparison_summary(c, &c.seeds, &runs, true)?;
    if !passed {
        comparison["decision_eligible"] = json!(false);
        comparison["pareto_shortlist"] = json!([]);
    }
    let result = json!({"schema":1,"command":"certify","measurement_implementation_version":train::MEASUREMENT_VERSION,"passed":passed,
        "config":c,"source_config_sha256":manifest.source_config_sha256,"manifest":manifest_path,"manifest_sha256":data::hash(&manifest_bytes),"data":identity,"device":runtime::identity(device),
        "certification_directory":directory,"certificates":certificates,"comparison":comparison,"new_training_tokens":0,
        "interpretation":"Quality is re-evaluated from explicit immutable checkpoints with synchronized value/id-validated loading and exact repeats. The initial baseline is freshly regenerated and repeat-checked under the current backend, not a bitwise historical replay. Original training throughput is pinned evidence, not remeasured. Original metrics/stdout/checkpoints are untouched; raw source bytes are copied separately. Pre-fix initial/final evaluations remain historical. Missing/untrusted baselines disable improvement eligibility; recall skill and uncertainty gates remain separate."});
    runtime::write_json(&directory.join("comparison.json"), &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_training_bytes_are_authoritative_not_reserialized_numbers() {
        let bytes = br#"{"seed":9007199254740993,"nll":0.84551240822557006,"tokens":1024}"#;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("delta-evidence-{}-{stamp}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("original.json");
        std::fs::write(&path, bytes).unwrap();
        let digest = data::hash(bytes);
        let (actual, value) = read_training_evidence(&path, &digest).unwrap();
        assert_eq!(actual.as_slice(), bytes.as_slice());
        assert_eq!(value["seed"], 9007199254740993u64);
        let mut altered = bytes.to_vec();
        altered.push(b'\n');
        std::fs::write(&path, altered).unwrap();
        assert!(
            read_training_evidence(&path, &digest).is_err(),
            "even semantically equivalent reserialization must fail the byte pin"
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
