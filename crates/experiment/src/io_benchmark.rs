//! Bounded IO attribution; resident fixtures are overhead bounds, not durable checkpoints.
use crate::{
    checkpoint,
    config::{Arm, Config},
    data::{self, Batch, Corpus, Example, HostBatch},
    model::{self, LanguageModel},
    profile, runtime, train,
};
use anyhow::{Context, Result, ensure};
use burn::{module::Module, optim::ModuleOptimizer, prelude::*};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path, time::{Instant, SystemTime, UNIX_EPOCH}};

// Frozen pre-change Batch::new, kept only as an independently executable benchmark baseline.
fn legacy_batch(examples: &[Example], device: &Device) -> Batch {
    let b = examples.len();
    let t = examples[0].input.len();
    let ints = |f: fn(&Example) -> &Vec<i32>| {
        Tensor::from_data(TensorData::new(examples.iter().flat_map(|e| f(e).iter().copied()).collect::<Vec<_>>(), [b, t]), device)
    };
    let weight = examples.iter().flat_map(|e| e.weight.iter().copied()).collect::<Vec<_>>();
    let mut reset = vec![0.0f32; b * t];
    for row in 0..b { reset[row * t] = 1.0; }
    Batch {
        input: ints(|e| &e.input), target: ints(|e| &e.target),
        weight: Tensor::from_data(TensorData::new(weight, [b, t]), device),
        valid: Tensor::ones([b, t], device),
        reset: Tensor::from_data(TensorData::new(reset, [b, t]), device),
    }
}
fn timed<T>(device: &Device, run: impl FnOnce() -> Result<T>) -> Result<(T, f64)> {
    runtime::sync(device)?;
    let start = Instant::now();
    let value = run()?;
    runtime::sync(device)?;
    Ok((value, start.elapsed().as_secs_f64() * 1000.))
}
fn exact_batch(batch: &Batch, rows: &[Example]) -> Result<()> {
    let shape = [rows.len(), rows[0].input.len()];
    let input = batch.input.clone().into_data().try_to_vec_as::<i32>().map_err(|e| anyhow::anyhow!("input readback: {e:?}"))?;
    let target = batch.target.clone().into_data().try_to_vec_as::<i32>().map_err(|e| anyhow::anyhow!("target readback: {e:?}"))?;
    let weight = batch.weight.clone().into_data().try_to_vec_as::<f32>().map_err(|e| anyhow::anyhow!("weight readback: {e:?}"))?;
    let valid = batch.valid.clone().into_data().try_to_vec_as::<f32>().map_err(|e| anyhow::anyhow!("valid readback: {e:?}"))?;
    let reset = batch.reset.clone().into_data().try_to_vec_as::<f32>().map_err(|e| anyhow::anyhow!("reset readback: {e:?}"))?;
    ensure!(batch.input.dims() == shape && batch.target.dims() == shape && batch.weight.dims() == shape && batch.valid.dims() == shape && batch.reset.dims() == shape, "batch shape changed");
    for (r, row) in rows.iter().enumerate() {
        for t in 0..shape[1] {
            let i = r * shape[1] + t;
            ensure!(input[i] == row.input[t] && target[i] == row.target[t] && weight[i].to_bits() == row.weight[t].to_bits() && valid[i].to_bits() == 1.0f32.to_bits() && reset[i].to_bits() == (if t == 0 { 1.0f32 } else { 0.0f32 }).to_bits(), "batch differs at row {r}, token {t}");
        }
    }
    Ok(())
}
fn step(c: &Config, model: LanguageModel, optim: &mut ModuleOptimizer, batch: &Batch) -> Result<LanguageModel> {
    let objective = model.loss(model.forward(batch), batch);
    ensure!(model::scalar(objective.clone())?.is_finite(), "nonfinite IO fixture loss");
    let gradients = model::checked_gradients(&model, objective)?;
    Ok(optim.step(c.learning_rate, model, gradients))
}
fn deadline(c: &Config, start: Instant) -> Result<()> {
    ensure!(start.elapsed().as_secs_f64() < c.max_seconds.min(900.), "IO benchmark deadline reached; partial report retained");
    Ok(())
}
fn summarize(samples: &BTreeMap<String, Vec<f64>>) -> Value {
    Value::Object(samples.iter().map(|(name, values)| (name.clone(), json!({"samples_ms":values,"median_ms":profile::median(values)}))).collect())
}
fn record(samples: &mut BTreeMap<String, Vec<f64>>, name: &str, ms: f64, measured: bool) {
    if measured { samples.entry(name.into()).or_default().push(ms); }
}

fn shape(c: &Config, arm: &Arm, corpus: &Corpus, device: &Device, directory: &Path, warmups: usize, trials: usize, start: Instant) -> Result<Value> {
    std::fs::create_dir_all(directory)?;
    let train_device = device.clone().autodiff();
    device.seed(c.seed);
    let rows = data::examples(c, Some(corpus), c.seed, 0, c.batch_size, "train")?;
    let resident = Batch::new(&rows, &train_device);
    exact_batch(&resident, &rows)?;
    exact_batch(&legacy_batch(&rows, &train_device), &rows)?;
    let mut optimizer = train::optimizer(c);
    let initial = LanguageModel::new(c, arm, &train_device)?;
    let validation = train::evaluations(c, Some(corpus), "validation")?;
    let initial_evaluation = train::evaluate(&initial, c, &validation, device)?;
    let (base, cold_step_ms) = timed(device, || step(c, initial, &mut optimizer, &resident))?;
    let before_signature = checkpoint::optimizer_signature(c, optimizer.into_bytes()?)?;
    let base_path = directory.join("resident-base.burnpack");
    base.clone().save_file(&base_path)?;
    checkpoint::validate_model(&base, &base_path)
        .context("resident base serialization self-check before any comparison")?;
    ensure!(before_signature["tensors"].as_object().is_some_and(|v| !v.is_empty()), "checkpoint fixture must contain real optimizer moments");
    let state = train::State {
        schema: 1,
        config_sha256: data::hash(&serde_json::to_vec(&(c, c.resolved_tokens()?))?),
        data: train::data_identity(c, Some(corpus), &validation),
        arm: arm.name.clone(), seed: c.seed, step: 1, tokens: c.tokens_per_step(),
        training_seconds: cold_step_ms / 1000.0, initial_evaluation,
    };
    let expected_state = serde_json::to_value(&state)?;
    let mut samples = BTreeMap::new();
    let mut comparisons = Vec::new();
    let mut peak_reserved = runtime::sampled_reserved(device);
    let mut checkpoint_bytes = 0;
    let mut first_checkpoint = Value::Null;
    let mut cold_io = Value::Null;
    let measured_rows = data::examples(c, Some(corpus), c.seed, c.batch_size, c.batch_size, "train")?;
    // One cold observation, excluded warmups, then measured trials. Alternate design order.
    for iteration in 0..1 + warmups + trials {
        deadline(c, start)?;
        let measured = iteration > warmups;
        let (generated, rows_ms) = timed(device, || data::examples(c, Some(corpus), c.seed, c.batch_size, c.batch_size, "train"))?;
        let (packed, pack_ms) = timed(device, || Ok(HostBatch::new(&generated)))?;
        let (uploaded, upload_ms) = timed(device, || Ok(packed.upload(&train_device)))?;
        exact_batch(&uploaded, &generated)?;
        let (_, legacy_ms) = timed(device, || Ok(legacy_batch(&generated, &train_device)))?;
        let (_, capacity_ms) = timed(device, || Ok(Batch::new(&generated, &train_device)))?;
        for (name, ms) in [("corpus_rows", rows_ms), ("exact_capacity_host_pack", pack_ms), ("packed_upload", upload_ms), ("legacy_pack_upload", legacy_ms), ("exact_capacity_pack_upload", capacity_ms)] {
            record(&mut samples, name, ms, measured);
        }
        if iteration == 0 { cold_io = json!({"corpus_rows_ms":rows_ms,"host_pack_ms":pack_ms,"upload_ms":upload_ms,"legacy_pack_upload_ms":legacy_ms,"exact_capacity_pack_upload_ms":capacity_ms}); }
        let next_resident = Batch::new(&generated, &train_device);
        let order = if iteration % 2 == 0 { ["legacy", "exact_capacity", "resident"] } else { ["resident", "exact_capacity", "legacy"] };
        let reference_path = directory.join(format!("continuation-reference-{iteration}.burnpack"));
        let mut expected_optimizer = None;
        for mode in order {
            deadline(c, start)?;
            let model = base.clone();
            // A resident comparison must not silently include an unchecked optimizer restore.
            // Clone the real state; the file-restore design is measured separately below.
            let mut optim = optimizer.clone();
            checkpoint::validate_model(&model, &base_path)
                .with_context(|| format!("iteration {iteration}, {mode}: pre-step resident parameters"))?;
            ensure!(checkpoint::optimizer_signature(c, optim.into_bytes()?)? == before_signature,
                "iteration {iteration}, {mode}: pre-step resident optimizer differs");
            let ((updated, owned_batch), ms) = timed(device, || {
                if mode == "resident" {
                    Ok((step(c, model, &mut optim, &next_resident)?, None))
                } else {
                    let rows = data::examples(c, Some(corpus), c.seed, c.batch_size, c.batch_size, "train")?;
                    let batch = if mode == "legacy" { legacy_batch(&rows, &train_device) } else { Batch::new(&rows, &train_device) };
                    let updated = step(c, model, &mut optim, &batch)?;
                    // Training holds its batch through the completion barrier. Match that lifetime
                    // rather than dropping uploaded tensors before timed() synchronizes.
                    Ok((updated, Some(batch)))
                }
            })?;
            exact_batch(owned_batch.as_ref().unwrap_or(&next_resident), &generated)
                .with_context(|| format!("iteration {iteration}, {mode}: actual timed batch values"))?;
            checkpoint::validate_model(&base, &base_path)
                .with_context(|| format!("iteration {iteration}, {mode}: retained base mutated after step"))?;
            ensure!(checkpoint::optimizer_signature(c, optimizer.into_bytes()?)? == before_signature,
                "iteration {iteration}, {mode}: retained optimizer mutated after step");
            record(&mut samples, &format!("full_optimizer_step_{mode}"), ms, measured);
            if let Some(bytes) = runtime::sampled_reserved(device) { peak_reserved = Some(peak_reserved.unwrap_or(0).max(bytes)); }
            // Exact logical parameters including IDs, and all optimizer tensors/scalars/metadata.
            // Every iteration is a same-state, same-next-row comparison, not diverging trajectories.
            let signature = checkpoint::optimizer_signature(c, optim.into_bytes()?)?;
            if let Some(expected) = &expected_optimizer {
                if let Err(error) = checkpoint::validate_model(&updated, &reference_path) {
                    let actual_path = directory.join(format!("continuation-{mode}-{iteration}.burnpack"));
                    updated.clone().save_file(&actual_path)?;
                    runtime::write_json(&directory.join("continuation-mismatch.json"),
                        &json!({"iteration":iteration,"mode":mode,"phase":"input-design comparison before checkpoint save/load","reference":reference_path,"actual":actual_path,"reference_optimizer":expected,"actual_optimizer":signature}))?;
                    return Err(error).with_context(|| format!("iteration {iteration}, {mode}: input-design next-step mismatch before checkpoint transaction; actual record retained"));
                }
                ensure!(&signature == expected, "iteration {iteration}, {mode}: input design changed next-step optimizer state");
            } else {
                updated.clone().save_file(&reference_path)?;
                checkpoint::validate_model(&updated, &reference_path)
                    .with_context(|| format!("iteration {iteration}, {mode}: continuation reference serialization self-check"))?;
                expected_optimizer = Some(signature);
            }
        }
        deadline(c, start)?;
        let saved = directory.join(format!("checkpoint-{iteration}"));
        let (_, save_ms) = timed(device, || train::checkpoint(c, &saved, &base, &optimizer, &state))?;
        checkpoint::validate_model(&base, saved.join("model.burnpack"))
            .context("atomic save differs from the retained pre-update fixture")?;
        let (loaded_state, state_ms) = timed(device, || {
            let value: Value = serde_json::from_slice(&std::fs::read(saved.join("state.json"))?)?;
            ensure!(value == expected_state, "checkpoint config/data/counters changed");
            let latest: Value = serde_json::from_slice(&std::fs::read(directory.join("latest.json"))?)?;
            ensure!(latest["checkpoint"] == json!(saved) && latest["step"] == 1 && latest["tokens"] == c.tokens_per_step(), "atomic checkpoint completion pointer changed");
            Ok(value)
        })?;
        let (target, initialize_ms) = timed(device, || LanguageModel::new(c, arm, &train_device))?;
        let (loaded, load_ms) = timed(device, || checkpoint::load_model(target, saved.join("model.burnpack"), device))
            .context("guarded checkpoint model restore")?;
        let (_, validate_ms) = timed(device, || checkpoint::validate_model(&loaded, saved.join("model.burnpack")))?;
        let (mut restored_optimizer, optimizer_ms) = timed(device, || checkpoint::load_optimizer(c, saved.join("optimizer.burnpack")))?;
        let restored_signature = checkpoint::optimizer_signature(c, restored_optimizer.into_bytes()?)?;
        ensure!(before_signature == restored_signature, "checkpoint optimizer IDs/moments/counters changed");
        let (continued, continuation_ms) = timed(device, || step(c, loaded, &mut restored_optimizer, &next_resident))?;
        checkpoint::validate_model(&continued, &reference_path)
            .context("restored checkpoint next-step continuation")?;
        ensure!(Some(checkpoint::optimizer_signature(c, restored_optimizer.into_bytes()?)?) == expected_optimizer, "checkpoint next-step optimizer continuation differs");
        for (name, ms) in [("checkpoint_atomic_save", save_ms), ("checkpoint_state_validation", state_ms), ("checkpoint_target_initialization", initialize_ms), ("checkpoint_guarded_model_load_including_validation", load_ms), ("checkpoint_model_validation_standalone", validate_ms), ("checkpoint_optimizer_load_including_validation", optimizer_ms), ("checkpoint_restored_continuation", continuation_ms)] {
            record(&mut samples, name, ms, measured);
        }
        checkpoint_bytes = std::fs::metadata(saved.join("model.burnpack"))?.len() + std::fs::metadata(saved.join("optimizer.burnpack"))?.len() + std::fs::metadata(saved.join("state.json"))?.len();
        if iteration == 0 {
            first_checkpoint = json!({"path":saved,"state":loaded_state,"model_sha256":checkpoint::file_sha256(&saved.join("model.burnpack"))?,"optimizer_sha256":checkpoint::file_sha256(&saved.join("optimizer.burnpack"))?,"save_ms":save_ms,"model_load_including_validation_ms":load_ms,"optimizer_load_including_validation_ms":optimizer_ms,"optimizer_canonical":before_signature});
        }
        comparisons.push(json!({"iteration":iteration,"phase":if iteration==0 {"cold"} else if measured {"measured"} else {"warmup"},"batch_values_exact":true,"parameters_and_ids_exact":true,"optimizer_moments_ids_counters_exact":true,"restored_next_step_exact":true}));
        runtime::write_json(&directory.join("partial.json"), &json!({"completed":false,"samples":summarize(&samples),"comparisons":comparisons,"first_checkpoint":first_checkpoint}))?;
    }
    Ok(json!({"completed":true,"config":c,"arm":arm,"batch_size":c.batch_size,"context":c.context,"initial_fixture_sha256":data::fixture_hash(&rows),"fixture_sha256":data::fixture_hash(&measured_rows),"parameters":base.num_params(),"cold_optimizer_step_ms":cold_step_ms,"cold_io":cold_io,"first_checkpoint":first_checkpoint,"timings":summarize(&samples),"checks":comparisons,
        "work":{"packed_host_payload_bytes":c.tokens_per_step()*16,"resident_batch_logical_bytes":c.tokens_per_step()*20,"checkpoint_bytes":checkpoint_bytes,"sampled_peak_pool_reserved_bytes":peak_reserved,"pool_measurement":"sampled pool reservation, not physical/live peak; includes correctness fixtures","exact_capacity_host_vectors":4,"batch_tensor_creations":5,"full_step_designs":["legacy_flat_map_collect","exact_capacity_owned_upload","same_step_resident_batch"],"checkpoint_designs":["original_atomic_save_and_guarded_validated_restore","same_step_resident_model_optimizer_continuation"],"resident_checkpoint_note":"resident continuation is full_optimizer_step_resident; original model and optimizer state prepared outside its clock. It has no durability and is only an IO-overhead bound, never a replacement save/load policy"}}))
}

pub fn run(c: &Config, arm: &Arm, device: &Device, warmups: usize, trials: usize) -> Result<Value> {
    ensure!(c.task == "lm", "io-benchmark requires a real corpus LM configuration");
    ensure!(warmups <= 3 && (1..=8).contains(&trials), "IO benchmark requires warmups <= 3 and trials 1..=8");
    let start = Instant::now();
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let directory = c.artifact_dir.join("io").join(format!("{}-{stamp}", std::process::id()));
    std::fs::create_dir_all(&directory)?;
    let (corpus, corpus_load_ms) = timed(device, || train::loaded_corpus(c))?;
    let corpus = corpus.context("LM corpus required")?;
    let mut reports = Vec::new();
    let mut failure = None;
    for batch in [1, 8] {
        for context in [128, 512] {
            let mut configured = c.clone();
            configured.batch_size = batch;
            configured.context = context;
            let shape_dir = directory.join(format!("b{batch}-t{context}"));
            match deadline(c, start).and_then(|()| shape(&configured, arm, &corpus, device, &shape_dir, warmups, trials, start)) {
                Ok(report) => reports.push(report),
                Err(error) => { failure = Some(format!("B{batch} T{context}: {error:#}")); break; }
            }
            runtime::write_json(&directory.join("partial.json"), &json!({"completed":false,"reports":reports}))?;
        }
        if failure.is_some() { break; }
    }
    let result = json!({"schema":1,"command":"io-benchmark","completed":failure.is_none() && reports.len()==4,"passed":failure.is_none() && reports.len()==4,"error":failure,"device":runtime::identity(device),"config":c,"arm":arm,"warmups":warmups,"trials":trials,"directory":directory,"corpus_load_and_validation_ms":corpus_load_ms,"corpus_manifest_sha256":corpus.manifest_sha256,"reports":reports,
        "timing_policy":"synchronize before/after measured regions; exact checks and fixture resets outside clocks; first observation excluded from warm samples; host clocks include final empty-queue sync overhead; checkpoint load includes mandatory validation, separate validation is not additive; file cache is OS-managed (no cold-disk claim); atomic rename matches training, no new fsync durability claim",
        "decision":"measurements are observations, not automatic adoption. Exact-capacity packing removes avoidable vector growth without changing data/order/config. No async prefetch or checkpoint safety bypass. Resident fixtures bound removable IO only; training still generates the next deterministic rows each step."});
    runtime::write_json(&directory.join("result.json"), &result)?;
    Ok(result)
}
