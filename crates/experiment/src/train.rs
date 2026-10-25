use crate::{
    checkpoint,
    config::{Arm, Config},
    data::{self, Batch, Corpus, Example},
    model::{self, LanguageModel},
    runtime,
};
use anyhow::{Context, Result, ensure};
use burn::{
    grad_clipping::GradientClippingConfig,
    module::Module,
    optim::{AdamWConfig, ModuleOptimizer},
    prelude::*,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path, time::Instant};

pub(crate) const MEASUREMENT_VERSION: &str = "guarded-validated-load-v2";
pub(crate) fn optimizer_config(c: &Config) -> AdamWConfig {
    AdamWConfig::new()
        .with_weight_decay(c.weight_decay)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(c.gradient_clip)))
}
pub(crate) fn optimizer(c: &Config) -> ModuleOptimizer {
    let optimizer = if c.update_backend.is_tensor() {
        optimizer_config(c).init()
    } else {
        crate::update::FusedAdamW {
            decay: c.weight_decay,
            clip: c.gradient_clip,
        }
        .into()
    };
    if c.lngram_layers.is_empty() {
        return optimizer;
    }
    optimizer.with_group(
        burn::module::ParamGroup::from_regex(r"(^|\.)lngram\.tables\.[0-9]+\.weight$")
            .expect("Lngram table parameter path"),
        crate::update::LngramAdamW {
            fused: !c.update_backend.is_tensor(),
            clip: c.gradient_clip,
        },
        None,
    )
}
pub(crate) fn loaded_corpus(c: &Config) -> Result<Option<Corpus>> {
    if c.task == "lm" {
        Ok(Some(Corpus::load(&c.corpus_manifest)?))
    } else {
        Ok(None)
    }
}
pub(crate) fn evaluations(
    c: &Config,
    corpus: Option<&Corpus>,
    split: &str,
) -> Result<Vec<Example>> {
    data::examples(c, corpus, c.eval_seed, 0, c.eval_examples, split)
}
pub(crate) fn data_identity(c: &Config, corpus: Option<&Corpus>, eval: &[Example]) -> Value {
    json!({"algorithm":"splitmix64-v1","training_seed_paired_across_arms":true,"eval_seed":c.eval_seed,"evaluation_sha256":data::fixture_hash(eval),"heldout_window_selection":"stratified-midpoint-disjoint-v1","corpus_manifest_sha256":corpus.map(|x|&x.manifest_sha256),"recall_partition":"sha256(hex_sha256(input_and_labels)) bit disjoint","tokenizer":if c.task=="lm" {"utf8-bytes-v1-bos256"}else {"recall-symbols-v1"}})
}
fn peak(current: &mut Option<u64>, device: &Device) {
    if let Some(value) = runtime::sampled_reserved(device) {
        *current = Some(current.unwrap_or(0).max(value));
    }
}
fn attention_work(c: &Config, arm: &Arm) -> Result<Value> {
    let mut work = arm.mixer_config(c)?.attention_work(c.batch_size, c.context);
    work.history_pairs *= c.layers;
    work.current_chunk_pairs *= c.layers;
    work.sink_pairs *= c.layers;
    work.selection_rank_pairs *= c.layers;
    work.selection_sort_elements *= c.layers;
    Ok(serde_json::to_value(work)?)
}

pub fn evaluate(
    model: &LanguageModel,
    c: &Config,
    examples: &[Example],
    device: &Device,
) -> Result<Value> {
    let model = model.valid();
    let device = device.clone();
    let mut nll = 0.0;
    let mut weight = 0.0;
    let mut correct = 0usize;
    let mut groups: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for rows in examples.chunks(c.batch_size) {
        let batch = Batch::new(rows, &device);
        let logits = model.forward(&batch);
        let loss = model::scalar(model.loss(logits.clone(), &batch))?;
        let predictions = logits
            .argmax(2)
            .reshape([rows.len() * c.context])
            .into_data()
            .try_to_vec_as::<i32>()
            .map_err(|e| anyhow::anyhow!("predictions: {e:?}"))?;
        let tokens: usize = rows
            .iter()
            .map(|r| r.weight.iter().filter(|&&v| v > 0.0).count())
            .sum();
        nll += loss * tokens as f64;
        weight += tokens as f64;
        for (row, example) in rows.iter().enumerate() {
            for t in 0..c.context {
                if example.weight[t] == 0.0 {
                    continue;
                }
                let hit = predictions[row * c.context + t] == example.target[t];
                correct += usize::from(hit);
                if let Some((count, distance)) = example.groups[t] {
                    let distance_bin = if distance < 32 {
                        "0-31"
                    } else if distance < 64 {
                        "32-63"
                    } else if distance < 128 {
                        "64-127"
                    } else if distance < 256 {
                        "128-255"
                    } else {
                        "256+"
                    };
                    for key in [
                        format!("associations/{count}"),
                        format!("distance/{distance_bin}"),
                        format!("associations/{count}/distance/{distance_bin}"),
                    ] {
                        let entry = groups.entry(key).or_default();
                        entry.0 += usize::from(hit);
                        entry.1 += 1;
                    }
                }
            }
        }
    }
    ensure!(weight > 0.0, "evaluation has no targets");
    let nll = nll / weight;
    let by_group: BTreeMap<_, _> = groups
        .into_iter()
        .map(|(key, (hits, total))| {
            (
                key,
                json!({"correct":hits,"count":total,"accuracy":hits as f64/total as f64}),
            )
        })
        .collect();
    Ok(
        json!({"nll":nll,"perplexity":nll.exp(),"accuracy":correct as f64/weight,"predicted_tokens":weight as usize,"by_group":by_group}),
    )
}
#[derive(Deserialize, Serialize)]
pub(crate) struct State {
    pub(crate) schema: u32,
    pub(crate) config_sha256: String,
    pub(crate) data: Value,
    pub(crate) arm: String,
    pub(crate) seed: u64,
    pub(crate) step: usize,
    pub(crate) tokens: usize,
    pub(crate) training_seconds: f64,
    pub(crate) initial_evaluation: Value,
}
pub(crate) fn checkpoint(
    c: &Config,
    path: &Path,
    model: &LanguageModel,
    optim: &ModuleOptimizer,
    state: &State,
) -> Result<()> {
    if !path.exists() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let pending = path.with_extension(format!("pending-{}-{stamp}", std::process::id()));
        std::fs::create_dir_all(&pending)?;
        model.clone().save_file(pending.join("model.burnpack"))?;
        optim.save(pending.join("optimizer.burnpack"))?;
        runtime::write_json(&pending.join("state.json"), state)?;
        std::fs::rename(pending, path)?;
    } else {
        let previous: State = serde_json::from_slice(&std::fs::read(path.join("state.json"))?)?;
        ensure!(
            previous.config_sha256 == state.config_sha256
                && previous.data == state.data
                && previous.arm == state.arm
                && previous.seed == state.seed
                && previous.tokens == state.tokens,
            "checkpoint path already belongs to another run"
        );
        checkpoint::validate_model(model, path.join("model.burnpack"))?;
        ensure!(
            checkpoint::optimizer_file_signature(c, path.join("optimizer.burnpack"))?
                == checkpoint::optimizer_signature(c, optim.into_bytes()?)?,
            "checkpoint destination collision: optimizer state differs; existing evidence was not overwritten"
        );
    }
    runtime::write_json(
        &path
            .parent()
            .context("checkpoint parent")?
            .join("latest.json"),
        &json!({"checkpoint":path,"step":state.step,"tokens":state.tokens}),
    )
}

pub fn run(
    c: &Config,
    arm: &Arm,
    seed: u64,
    device: &Device,
    resume: Option<&Path>,
) -> Result<Value> {
    let corpus = loaded_corpus(c)?;
    let validation = evaluations(c, corpus.as_ref(), "validation")?;
    let data = data_identity(c, corpus.as_ref(), &validation);
    let target_tokens = c.resolved_tokens()?;
    let quantum = c.tokens_per_step();
    let fingerprint = data::hash(&serde_json::to_vec(&(c, target_tokens))?);
    ensure!(
        target_tokens >= quantum && target_tokens % quantum == 0,
        "token budget must be a positive multiple of batch*context ({quantum})"
    );
    let run_dir = c.artifact_dir.join(format!("{}-seed{seed}", arm.name));
    std::fs::create_dir_all(&run_dir)?;
    device.memory_cleanup();
    runtime::sync(device)?;
    let reservation_baseline = runtime::sampled_reserved(device);
    device.seed(seed);
    let train_device = device.clone().autodiff();
    let mut model = LanguageModel::new(c, arm, &train_device)?;
    let mut optim = optimizer(c);
    let mut step = 0usize;
    let mut tokens = 0usize;
    let mut previous_seconds = 0.0;
    let initial;
    let initial_trusted;
    if let Some(path) = resume {
        let state: State = serde_json::from_slice(&std::fs::read(path.join("state.json"))?)?;
        ensure!(
            state.schema == 1
                && state.config_sha256 == fingerprint
                && state.data == data
                && state.arm == arm.name
                && state.seed == seed,
            "resume mismatch: config, seed, arm or dataset changed"
        );
        model = checkpoint::load_model(model, path.join("model.burnpack"), device)?;
        optim = checkpoint::load_optimizer(c, path.join("optimizer.burnpack"))?;
        step = state.step;
        tokens = state.tokens;
        previous_seconds = state.training_seconds;
        initial = state.initial_evaluation;
        initial_trusted = false;
    } else {
        initial = evaluate(&model, c, &validation, device)?;
        let repeated = evaluate(&model, c, &validation, device)?;
        ensure!(
            initial == repeated
                && ["nll", "perplexity", "accuracy"]
                    .iter()
                    .all(|field| initial[*field].as_f64().is_some_and(f64::is_finite)),
            "fresh initial evaluation is nonfinite or not repeat-stable"
        );
        initial_trusted = true;
    }
    let parameters = model.num_params();
    let start_tokens = tokens;
    let mut last_loss = None;
    let mut peak_reserved = runtime::sampled_reserved(device);
    let started = Instant::now();
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(resume.is_some())
        .truncate(resume.is_none())
        .open(run_dir.join("training.jsonl"))?;
    while tokens < target_tokens && started.elapsed().as_secs_f64() < c.max_seconds {
        let rows = data::examples(
            c,
            corpus.as_ref(),
            seed,
            step * c.batch_size,
            c.batch_size,
            "train",
        )?;
        let batch = Batch::new(&rows, &train_device);
        let loss = if c.sbd.is_some() {
            crate::sbd::training_loss(&model,c,&rows,&batch,seed.wrapping_add(step as u64))?
        } else {
            model.loss(model.forward(&batch), &batch)
        };
        peak(&mut peak_reserved, device);
        let value = model::scalar(loss.clone())?;
        let grads = model::checked_gradients(&model, loss)?;
        peak(&mut peak_reserved, device);
        model = optim.step(c.learning_rate, model, grads);
        runtime::sync(device)?;
        peak(&mut peak_reserved, device);
        step += 1;
        tokens += quantum;
        last_loss = Some(value);
        use std::io::Write;
        writeln!(
            log,
            "{}",
            json!({"step":step,"tokens":tokens,"loss":value,"elapsed_seconds":started.elapsed().as_secs_f64()})
        )?;
        if step % c.checkpoint_every == 0 {
            let state = State {
                schema: 1,
                config_sha256: fingerprint.clone(),
                data: data.clone(),
                arm: arm.name.clone(),
                seed,
                step,
                tokens,
                training_seconds: previous_seconds + started.elapsed().as_secs_f64(),
                initial_evaluation: initial.clone(),
            };
            checkpoint(
                c,
                &run_dir.join(format!("checkpoint-{step}")),
                &model,
                &optim,
                &state,
            )?;
        }
    }
    runtime::sync(device)?;
    let training_seconds = started.elapsed().as_secs_f64();
    let total_training_seconds = previous_seconds + training_seconds;
    let state = State {
        schema: 1,
        config_sha256: fingerprint.clone(),
        data: data.clone(),
        arm: arm.name.clone(),
        seed,
        step,
        tokens,
        training_seconds: total_training_seconds,
        initial_evaluation: initial.clone(),
    };
    let checkpoint_path = run_dir.join(format!("checkpoint-{step}"));
    checkpoint(c, &checkpoint_path, &model, &optim, &state)?;
    let final_eval = evaluate(&model, c, &validation, device)?;
    let test = if corpus.is_some() {
        Some(evaluate(
            &model,
            c,
            &evaluations(c, corpus.as_ref(), "test")?,
            device,
        )?)
    } else {
        None
    };
    let retrieval = if corpus.is_some() {
        Some(evaluate(
            &model,
            c,
            &crate::lm_retrieval::examples(c),
            device,
        )?)
    } else {
        None
    };
    let decode_batch = Batch::new(&validation[..1], device);
    let inference = model.valid();
    let _ = model::scalar(inference.forward(&decode_batch).sum())?;
    runtime::sync(device)?;
    let clock = Instant::now();
    for _ in 0..3 {
        let _ = model::scalar(inference.forward(&decode_batch).sum())?;
        runtime::sync(device)?;
    }
    let decode_ms = clock.elapsed().as_secs_f64() * 1000.0 / 3.0;
    let mix = arm.mixer_config(c)?;
    let mut metrics = json!({"schema":1,"task":c.task,"arm":arm,"seed":seed,"config":c,"config_sha256":fingerprint,"data":data,
        "device":runtime::identity(device),"precision":"float32","optimizer":"AdamW","all_model_parameters_trained":true,
        "measurement_implementation_version":MEASUREMENT_VERSION,
        "training_algorithm":arm.training_algorithm(c),"custom_gpu_kernel_proof":arm.uses_custom_gpu(c),
        "parameters":parameters,"recurrent_state_bytes":mix.recurrent_state_bytes(c.batch_size)*c.layers,"cache_including_current_chunk_bytes":mix.exact_memory_bytes(c.batch_size)*c.layers,
        "convolution_state_bytes":c.batch_size*c.layers*3*c.heads*(2*c.key_dim+c.value_dim)*4,
        "hola_work_per_model_forward":attention_work(c,arm)?,"hola_work_note":"exact allocated history/current-chunk/sink pairs and rank-comparison pairs, including masked/unfilled slots; sort_elements is not a sort-comparison count; backward work not inferred",
        "sampled_peak_reserved_device_bytes":peak_reserved,"reservation_baseline_after_cleanup_bytes":reservation_baseline,"peak_gpu_memory_bytes":Value::Null,
        "memory_note":"cleanup requested before each run; runtime may retain reservations across runs; high-water sampled after forward/backward/step, not an exact GPU peak; integrated Intel GPU shares system memory",
        "tokens":tokens,"target_tokens":target_tokens,"steps":step,"completed":tokens==target_tokens,"training_seconds_this_invocation":training_seconds,
        "trained_tokens_this_invocation":tokens-start_tokens,"total_training_seconds":total_training_seconds,"training_tokens_per_second":tokens as f64/total_training_seconds,
        "last_training_loss":last_loss,"validation":final_eval,"test":test,
        "byte_associative_retrieval_diagnostic":retrieval,
        "warm_decode_latency_ms":decode_ms,"decode_method":"next-token logits by full causal-prefix recomputation, context fixed; no incremental-state optimization",
        "checkpoint":checkpoint_path});
    if c.sbd.is_some() {
        metrics["sbd_validation"] = crate::sbd::evaluate(&model,c,&validation,device)?;
        metrics["sbd_generation"] = crate::sbd::generate(&model,c,&validation[0].input[..validation[0].input.len().min(17)],device)?;
    }
    metrics["initial_validation"] = initial;
    metrics["initial_validation_trusted"] = json!(initial_trusted);
    metrics["initial_validation_provenance"] = json!(if initial_trusted {
        "fresh initialization, repeat-checked"
    } else {
        "historical checkpoint baseline; certify a regenerated baseline before learning claims"
    });
    runtime::write_json(&run_dir.join("metrics.json"), &metrics)?;
    Ok(metrics)
}

pub fn pilot(c: &Config, device: &Device) -> Result<Value> {
    let corpus = loaded_corpus(c)?;
    let mut arms = Vec::new();
    let mut slowest = f64::INFINITY;
    for arm in &c.arms {
        device.seed(c.seed);
        let train_device = device.clone().autodiff();
        let mut model = LanguageModel::new(c, arm, &train_device)?;
        let mut optim = optimizer(c);
        let rows = data::examples(c, corpus.as_ref(), c.seed, 0, c.batch_size, "train")?;
        let batch = Batch::new(&rows,&train_device);
        let first = Instant::now();
        let loss = if c.sbd.is_some() {
            crate::sbd::training_loss(&model,c,&rows,&batch,c.seed)?
        } else {model.loss(model.forward(&batch),&batch)};
        let grads = model::checked_gradients(&model, loss)?;
        model = optim.step(c.learning_rate, model, grads);
        runtime::sync(device)?;
        let first_use = first.elapsed().as_secs_f64();
        let clock = Instant::now();
        let mut steps = 0;
        while steps < c.pilot_steps
            && (steps == 0 || clock.elapsed().as_secs_f64() < c.pilot_seconds)
        {
            let rows = data::examples(
                c,
                corpus.as_ref(),
                c.seed,
                (steps + 1) * c.batch_size,
                c.batch_size,
                "train",
            )?;
            let batch = Batch::new(&rows, &train_device);
            let loss = if c.sbd.is_some() {
                crate::sbd::training_loss(&model,c,&rows,&batch,c.seed.wrapping_add(steps as u64 + 1))?
            } else {model.loss(model.forward(&batch),&batch)};
            let grads = model::checked_gradients(&model, loss)?;
            model = optim.step(c.learning_rate, model, grads);
            runtime::sync(device)?;
            steps += 1;
        }
        let seconds = clock.elapsed().as_secs_f64();
        let throughput = (steps * c.tokens_per_step()) as f64 / seconds;
        ensure!(
            throughput.is_finite() && throughput > 0.0,
            "invalid pilot throughput"
        );
        slowest = slowest.min(throughput);
        arms.push(json!({"arm":arm,"training_algorithm":arm.training_algorithm(c),"custom_gpu_kernel_proof":arm.uses_custom_gpu(c),"hola_work_per_model_forward":attention_work(c,arm)?,"first_use_forward_backward_step_seconds":first_use,"warm_steps":steps,"warm_seconds":seconds,"tokens_per_second":throughput}));
        drop(model);
        device.memory_cleanup();
    }
    let quantum = c.tokens_per_step();
    let tokens =
        ((slowest * c.comparison_seconds / quantum as f64).floor() as usize).max(1) * quantum;
    let result = json!({"schema":1,"pilot_config_sha256":c.pilot_fingerprint()?,"task":c.task,"context":c.context,"batch_size":c.batch_size,"d_model":c.d_model,"layers":c.layers,"backend":c.backend,"arms":c.arms,
        "device":runtime::identity(device),"measurements":arms,"pilot_seconds_limit_per_arm":c.pilot_seconds,"target_training_seconds_per_arm":c.comparison_seconds,
        "budget_policy":"slowest measured warm full-model AdamW throughput times target seconds, rounded down to complete batches","train_tokens":tokens});
    runtime::write_json(&c.artifact_dir.join("budget.json"), &result)?;
    Ok(result)
}

fn recall_skill_diagnostic(c: &Config, rows: &[&Value]) -> Result<Value> {
    if c.task != "recall" {
        return Ok(Value::Null);
    }
    let baseline = 1.0 / c.value_symbols as f64;
    let z = 1.959963984540054f64;
    let z2 = z * z;
    let mut per_seed = Vec::with_capacity(rows.len());
    for row in rows {
        let trials = row["validation"]["predicted_tokens"]
            .as_u64()
            .context("recall diagnostic needs target count")?;
        let accuracy = row["validation"]["accuracy"]
            .as_f64()
            .context("recall diagnostic needs accuracy")?;
        ensure!(
            trials > 0 && accuracy.is_finite() && (0.0..=1.0).contains(&accuracy),
            "invalid recall diagnostic trials"
        );
        let correct = (accuracy * trials as f64).round() as u64;
        let n = trials as f64;
        let p = correct as f64 / n;
        let denominator = 1.0 + z2 / n;
        let center = (p + z2 / (2.0 * n)) / denominator;
        let radius = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denominator;
        let lower = (center - radius).max(0.0);
        let upper = (center + radius).min(1.0);
        per_seed.push(json!({"seed":row["seed"],"correct":correct,"trials":trials,"accuracy":p,
            "wilson_95_lower":lower,"wilson_95_upper":upper,"lower_bound_above_random":lower>baseline}));
    }
    let demonstrated = !per_seed.is_empty()
        && per_seed
            .iter()
            .all(|s| s["lower_bound_above_random"] == true);
    Ok(
        json!({"random_accuracy_baseline":baseline,"baseline":"uniform guessing among value symbols, not the full input vocabulary",
        "criterion":"every paired seed has two-sided95% Wilson lower bound strictly above random_accuracy_baseline",
        "recall_skill_demonstrated":demonstrated,"per_seed":per_seed,
        "uncertainty_note":"Wilson intervals assume iid Bernoulli trials. Queries within an example may be correlated; evaluation examples are reused across seeds. Seeds are not pooled. This is a conservative eligibility diagnostic, not proof of skill or architectural superiority."}),
    )
}

pub(crate) fn comparison_summary(
    c: &Config,
    seeds: &[u64],
    runs: &[Value],
    decision: bool,
) -> Result<Value> {
    let seed_order: BTreeMap<_, _> = seeds
        .iter()
        .enumerate()
        .map(|(rank, &seed)| (seed, rank))
        .collect();
    ensure!(
        seed_order.len() == seeds.len(),
        "paired seeds must be distinct"
    );
    let mut summary = Vec::new();
    for arm in &c.arms {
        let mut rows: Vec<_> = runs
            .iter()
            .filter(|r| r["arm"]["name"] == arm.name)
            .collect();
        ensure!(
            rows.len() == seeds.len(),
            "comparison must contain every paired seed for each arm"
        );
        rows.sort_unstable_by_key(|r| {
            seed_order
                .get(&r["seed"].as_u64().unwrap_or(u64::MAX))
                .copied()
                .unwrap_or(usize::MAX)
        });
        ensure!(
            rows.iter()
                .zip(seeds)
                .all(|(row, &seed)| row["seed"].as_u64() == Some(seed)),
            "comparison contains duplicate or unknown arm seeds"
        );
        let metric = if c.task == "recall" {
            "accuracy"
        } else {
            "nll"
        };
        let values: Vec<f64> = rows
            .iter()
            .map(|r| r["validation"][metric].as_f64().unwrap())
            .collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let sd =
            (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
        let throughput = rows
            .iter()
            .map(|r| r["training_tokens_per_second"].as_f64().unwrap())
            .sum::<f64>()
            / rows.len() as f64;
        let trusted = rows.iter().all(|r| r["initial_validation_trusted"] == true);
        let improvement = trusted.then(|| {
            rows.iter()
                .map(|r| {
                    r["initial_validation"]["nll"].as_f64().unwrap()
                        - r["validation"]["nll"].as_f64().unwrap()
                })
                .sum::<f64>()
                / rows.len() as f64
        });
        let learning_seeds = trusted.then(|| {
            rows.iter()
                .filter(|r| {
                    r["validation"]["nll"].as_f64().unwrap()
                        < r["initial_validation"]["nll"].as_f64().unwrap()
                })
                .count()
        });
        let skill = recall_skill_diagnostic(c, &rows)?;
        summary.push(json!({"arm":arm.name,"validation_metric":metric,"mean":mean,"seed_standard_deviation":sd,"paired_values":values,
            "initial_validation_trusted":trusted,"mean_heldout_nll_improvement":improvement,"learning_seed_count":learning_seeds,
            "random_accuracy_baseline":skill["random_accuracy_baseline"],"recall_skill":skill,
            "mean_training_tokens_per_second":throughput,"parameters":rows[0]["parameters"],
            "state_and_cache_bytes":rows[0]["recurrent_state_bytes"].as_u64().unwrap()+rows[0]["cache_including_current_chunk_bytes"].as_u64().unwrap()}));
    }
    let complete = runs.iter().all(|r| r["completed"] == true);
    let mut shortlist = Vec::new();
    let learning = summary.iter().all(|r| {
        r["mean_heldout_nll_improvement"]
            .as_f64()
            .is_some_and(|v| v > 0.0)
    });
    let recall_skill = c.task != "recall"
        || summary
            .iter()
            .all(|r| r["recall_skill"]["recall_skill_demonstrated"] == true);
    let eligible = decision && complete && learning && recall_skill;
    if eligible {
        for candidate in &summary {
            let dominated = summary.iter().any(|other| {
                let quality = |x: &Value| {
                    let v = x["mean"].as_f64().unwrap();
                    if c.task == "recall" { v } else { -v }
                };
                let q = quality(other) >= quality(candidate);
                let speed = other["mean_training_tokens_per_second"].as_f64().unwrap()
                    >= candidate["mean_training_tokens_per_second"]
                        .as_f64()
                        .unwrap();
                let mem = other["state_and_cache_bytes"].as_u64().unwrap()
                    <= candidate["state_and_cache_bytes"].as_u64().unwrap();
                let strict = quality(other) > quality(candidate)
                    || other["mean_training_tokens_per_second"]
                        != candidate["mean_training_tokens_per_second"]
                    || other["state_and_cache_bytes"] != candidate["state_and_cache_bytes"];
                q && speed && mem && strict
            });
            if !dominated {
                shortlist.push(candidate["arm"].clone());
            }
        }
    }
    Ok(
        json!({"schema":1,"task":c.task,"decision_requested":decision,"decision_eligible":eligible,
        "heldout_learning_demonstrated":learning,"random_accuracy_baseline":if c.task=="recall" {json!(1.0/c.value_symbols as f64)}else {Value::Null},
        "recall_skill_demonstrated":if c.task=="recall" {json!(recall_skill)}else {Value::Null},
        "paired_seeds":seeds,"all_budgets_completed":complete,"summary":summary,"pareto_shortlist":shortlist,"runs":runs,
        "interpretation":"NLL improvement is separate from recall skill. Recall shortlists require every arm/seed Wilson lower-bound diagnostic above value-symbol guessing, with stated independence limitations. Three paired seeds and a quality/throughput/state-memory Pareto screen do not prove architectural superiority or transfer to language modeling."}),
    )
}

pub(crate) fn metrics_contract(
    c: &Config,
    data: &Value,
    device: &Value,
    target_tokens: usize,
) -> Result<Value> {
    Ok(
        json!({"schema":1,"measurement_implementation_version":MEASUREMENT_VERSION,"task":c.task,
        "config_sha256":data::hash(&serde_json::to_vec(&(c,target_tokens))?),"target_tokens":target_tokens,
        "data":data,"device":device,"precision":"float32","optimizer":"AdamW",
        "all_model_parameters_trained":true,"training_algorithm":c.arms[0].training_algorithm(c),"custom_gpu_kernel_proof":c.arms[0].uses_custom_gpu(c)}),
    )
}

fn read_completed_metrics(path: &Path, expected: &Value) -> Result<Option<Value>> {
    if !path.exists() {
        return Ok(None);
    }
    let metrics: Value = serde_json::from_slice(
        &std::fs::read(path).with_context(|| format!("reading {}", path.display()))?,
    )?;
    for (field, value) in expected
        .as_object()
        .context("invalid internal metrics contract")?
    {
        ensure!(
            metrics.get(field) == Some(value),
            "cached metrics {} mismatch in {field}; refusing stale/uncertified results. Use explicit train --resume to regenerate metrics, or certify explicit original checkpoints into separate reports",
            path.display()
        );
    }
    if !metrics["completed"]
        .as_bool()
        .context("cached metrics lack completion status")?
    {
        return Ok(None);
    }
    ensure!(
        metrics["tokens"] == expected["target_tokens"],
        "completed cached metrics did not finish their fixed token budget"
    );
    Ok(Some(metrics))
}

pub fn compare(c: &Config, device: &Device, decision: bool) -> Result<Value> {
    let seeds = if decision {
        ensure!(
            c.seeds.len() == 3,
            "architectural decision requires exactly three paired seeds"
        );
        let set: std::collections::BTreeSet<_> = c.seeds.iter().collect();
        ensure!(set.len() == 3, "decision seeds must be distinct");
        c.seeds.clone()
    } else {
        vec![c.seed]
    };
    let corpus = loaded_corpus(c)?;
    let validation = evaluations(c, corpus.as_ref(), "validation")?;
    let data = data_identity(c, corpus.as_ref(), &validation);
    let target_tokens = c.resolved_tokens()?;
    let mut expected = metrics_contract(c, &data, &runtime::identity(device), target_tokens)?;
    let mut runs = Vec::new();
    for &seed in &seeds {
        for arm in &c.arms {
            expected["arm"] = serde_json::to_value(arm)?;
            expected["seed"] = json!(seed);
            expected["training_algorithm"] = json!(arm.training_algorithm(c));
            expected["custom_gpu_kernel_proof"] = json!(arm.uses_custom_gpu(c));
            let metrics_path = c
                .artifact_dir
                .join(format!("{}-seed{seed}/metrics.json", arm.name));
            if let Some(metrics) = read_completed_metrics(&metrics_path, &expected)? {
                runs.push(metrics);
                continue;
            }
            let latest = c
                .artifact_dir
                .join(format!("{}-seed{seed}/latest.json", arm.name));
            let resume = if latest.exists() {
                let record: Value = serde_json::from_slice(&std::fs::read(latest)?)?;
                Some(std::path::PathBuf::from(
                    record["checkpoint"]
                        .as_str()
                        .context("invalid latest checkpoint")?,
                ))
            } else {
                ensure!(
                    !metrics_path.exists(),
                    "incomplete metrics lack a resumable checkpoint"
                );
                None
            };
            runs.push(run(c, arm, seed, device, resume.as_deref())?);
        }
    }
    let result = comparison_summary(c, &seeds, &runs, decision)?;
    runtime::write_json(
        &c.artifact_dir.join(if decision {
            "comparison.json"
        } else {
            "smoke.json"
        }),
        &result,
    )?;
    Ok(result)
}

#[cfg(test)]
mod summary_tests {
    use super::*;
    #[test]
    fn chance_recall_cannot_recommend_architecture_even_when_nll_improves() {
        let mut c = Config::default();
        c.arms.push(Arm {
            name: "kronlstm".into(),
            mixer: "kronlstm".into(),
            cache: "none".into(),
        });
        let mut runs = Vec::new();
        for seed in [1u64, 2, 3] {
            for arm in &c.arms {
                runs.push(json!({"arm":arm,"seed":seed,"completed":true,"parameters":100,
                "training_tokens_per_second":100.0,"recurrent_state_bytes":32,"cache_including_current_chunk_bytes":0,
                "initial_validation":{"nll":5.0},"initial_validation_trusted":true,"validation":{"nll":4.9,"accuracy":1.0/64.0,"predicted_tokens":1024}}));
            }
        }
        let chance = comparison_summary(&c, &[1, 2, 3], &runs, true).unwrap();
        assert_eq!(chance["heldout_learning_demonstrated"], true);
        assert_eq!(chance["random_accuracy_baseline"], 1.0 / 64.0);
        assert_eq!(chance["recall_skill_demonstrated"], false);
        assert_eq!(chance["decision_eligible"], false);
        assert!(chance["pareto_shortlist"].as_array().unwrap().is_empty());
        for run in &mut runs {
            run["validation"]["accuracy"] = json!(0.5);
        }
        let skilled = comparison_summary(&c, &[1, 2, 3], &runs, true).unwrap();
        assert_eq!(skilled["recall_skill_demonstrated"], true);
        assert_eq!(skilled["decision_eligible"], true);
        assert_eq!(skilled["pareto_shortlist"].as_array().unwrap().len(), 2);
        for run in &mut runs {
            run.as_object_mut()
                .unwrap()
                .remove("initial_validation_trusted");
        }
        let untrusted = comparison_summary(&c, &[1, 2, 3], &runs, true).unwrap();
        assert_eq!(untrusted["heldout_learning_demonstrated"], false);
        assert_eq!(untrusted["decision_eligible"], false);
        assert!(untrusted["summary"][0]["mean_heldout_nll_improvement"].is_null());
    }
    #[test]
    fn completed_metrics_reuse_preserves_measurements_and_rejects_stale_contracts() {
        let c = Config::default();
        let data =
            json!({"evaluation_sha256":"fixed-eval","corpus_manifest_sha256":"fixed-corpus"});
        let device = json!({"device":"cpu-oracle","identity":"CPU ndarray"});
        let tokens = 2048;
        let mut expected = metrics_contract(&c, &data, &device, tokens).unwrap();
        expected["arm"] = serde_json::to_value(&c.arms[0]).unwrap();
        expected["seed"] = json!(1);
        let mut metrics = expected.clone();
        metrics["completed"] = json!(true);
        metrics["tokens"] = json!(tokens);
        metrics["last_training_loss"] = json!(1.25);
        metrics["sampled_peak_reserved_device_bytes"] = json!(8765);
        metrics["training_seconds_this_invocation"] = json!(12.5);
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "delta-metrics-cache-{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("metrics.json");
        std::fs::write(&path, serde_json::to_vec(&metrics).unwrap()).unwrap();
        let reused = read_completed_metrics(&path, &expected).unwrap().unwrap();
        assert_eq!(
            reused, metrics,
            "reusing completed metrics changed original measured loss/memory/timing"
        );
        let mut changed = c.clone();
        changed.d_model *= 2;
        let mut changed_config = metrics_contract(&changed, &data, &device, tokens).unwrap();
        changed_config["arm"] = expected["arm"].clone();
        changed_config["seed"] = json!(1);
        assert!(read_completed_metrics(&path, &changed_config).is_err());
        let mut changed_data = expected.clone();
        changed_data["data"]["evaluation_sha256"] = json!("changed-eval");
        assert!(read_completed_metrics(&path, &changed_data).is_err());
        let mut changed_device = expected.clone();
        changed_device["device"]["identity"] = json!("different-device");
        assert!(read_completed_metrics(&path, &changed_device).is_err());
        let mut changed_seed = expected.clone();
        changed_seed["seed"] = json!(2);
        assert!(read_completed_metrics(&path, &changed_seed).is_err());
        let mut changed_precision = expected.clone();
        changed_precision["precision"] = json!("float16");
        assert!(read_completed_metrics(&path, &changed_precision).is_err());
        let mut pre_fix = metrics.clone();
        pre_fix
            .as_object_mut()
            .unwrap()
            .remove("measurement_implementation_version");
        std::fs::write(&path, serde_json::to_vec(&pre_fix).unwrap()).unwrap();
        assert!(
            read_completed_metrics(&path, &expected).is_err(),
            "pre-fix measurements must not masquerade as certified cache entries"
        );
        metrics["completed"] = json!(false);
        std::fs::write(&path, serde_json::to_vec(&metrics).unwrap()).unwrap();
        assert!(
            read_completed_metrics(&path, &expected).unwrap().is_none(),
            "incomplete metrics must resume rather than reuse"
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn summary_preserves_explicit_pairing_for_shuffled_sources() {
        let mut c = Config::default();
        c.arms.push(Arm {
            name: "kronlstm".into(),
            mixer: "kronlstm".into(),
            cache: "none".into(),
        });
        let mut runs = Vec::new();
        for seed in [1u64, 2, 3] {
            for arm in &c.arms {
                runs.push(json!({"arm":arm,"seed":seed,"completed":true,"parameters":100,
                "training_tokens_per_second":100.0,"recurrent_state_bytes":32,"cache_including_current_chunk_bytes":0,
                "initial_validation":{"nll":5.0},"initial_validation_trusted":true,
                "validation":{"nll":4.9,"accuracy":seed as f64/10.0,"predicted_tokens":1024}}));
            }
        }
        runs.reverse();
        let summary = comparison_summary(&c, &[3, 1, 2], &runs, false).unwrap();
        for arm in summary["summary"].as_array().unwrap() {
            assert_eq!(arm["paired_values"], json!([0.3, 0.1, 0.2]));
            assert_eq!(arm["recall_skill"]["per_seed"][0]["seed"], 3);
            assert_eq!(arm["recall_skill"]["per_seed"][1]["seed"], 1);
            assert_eq!(arm["recall_skill"]["per_seed"][2]["seed"], 2);
        }
        let duplicate = runs[2]["seed"].clone();
        runs[0]["seed"] = duplicate;
        assert!(comparison_summary(&c, &[3, 1, 2], &runs, false).is_err());
    }
    #[test]
    fn checkpoint_collision_compares_model_and_optimizer_without_overwrite() {
        let mut c = Config::default();
        c.d_model = 8;
        c.layers = 1;
        c.context = 16;
        c.association_counts = vec![2];
        let device = Device::ndarray().autodiff();
        device.seed(883);
        let mut model = LanguageModel::new(&c, &c.arms[0], &device).unwrap();
        let mut optim = optimizer(&c);
        let batch = Batch::new(&[data::recall(&c, 1, 0, false)], &device);
        let gradient =
            model::checked_gradients(&model, model.loss(model.forward(&batch), &batch)).unwrap();
        model = optim.step(c.learning_rate, model, gradient);
        let original_model = model.clone();
        let state = State {
            schema: 1,
            config_sha256: "fixed-config".into(),
            data: json!({"fixture":"fixed"}),
            arm: c.arms[0].name.clone(),
            seed: 1,
            step: 1,
            tokens: 16,
            training_seconds: 1.0,
            initial_evaluation: json!({"nll":5.0}),
        };
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "delta-checkpoint-collision-{}-{stamp}",
            std::process::id()
        ));
        let path = directory.join("checkpoint-1");
        checkpoint(&c, &path, &model, &optim, &state).unwrap();
        let model_hash = checkpoint::file_sha256(&path.join("model.burnpack")).unwrap();
        let optim_hash = checkpoint::file_sha256(&path.join("optimizer.burnpack")).unwrap();
        checkpoint(&c, &path, &model, &optim, &state).unwrap();
        let gradient =
            model::checked_gradients(&model, model.loss(model.forward(&batch), &batch)).unwrap();
        model = optim.step(c.learning_rate, model, gradient);
        assert!(
            checkpoint(&c, &path, &model, &optim, &state).is_err(),
            "different weights under identical metadata must not reuse a checkpoint"
        );
        assert!(
            checkpoint(&c, &path, &original_model, &optim, &state).is_err(),
            "different optimizer moments must not reuse a checkpoint"
        );
        assert_eq!(
            model_hash,
            checkpoint::file_sha256(&path.join("model.burnpack")).unwrap()
        );
        assert_eq!(
            optim_hash,
            checkpoint::file_sha256(&path.join("optimizer.burnpack")).unwrap()
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
