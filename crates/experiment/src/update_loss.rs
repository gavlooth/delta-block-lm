//! Bounded, interleaved checks/update/loss evaluation; no recommended configuration changes.
use crate::{
    checkpoint,
    config::{Config, LossBackend, UpdateBackend},
    data::{self, Batch},
    model::{self, LanguageModel},
    profile::{Runner, median},
    runtime, train,
};
use anyhow::{Result, ensure};
#[cfg(any(feature = "vulkan", feature = "cuda"))]
use burn::optim::{AdamWConfig, Optimizer};
use burn::{
    module::{Module, ModuleVisitor, Param},
    optim::GradientsParams,
    prelude::*,
};
use delta_mixers::profile::Unprofiled;
use serde_json::{Value, json};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
#[cfg(any(feature = "vulkan", feature = "cuda"))]
fn values<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.into_data().try_to_vec::<f32>().unwrap()
}
#[cfg(any(feature = "vulkan", feature = "cuda"))]
fn compare(name: &str, a: &[f32], b: &[f64]) -> Result<f64> {
    ensure!(a.len() == b.len(), "{name}: lengths");
    let mut max = 0f64;
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        let e = (f64::from(a) - b).abs();
        ensure!(
            a.is_finite() && b.is_finite() && e <= 2e-5 + 3e-4 * b.abs(),
            "{name}[{i}]: {a} vs {b}, error {e}"
        );
        max = max.max(e);
    }
    Ok(max)
}
#[cfg(any(feature = "vulkan", feature = "cuda"))]
fn tensor2(data: &[f32], rows: usize, width: usize, strided: bool, device: &Device) -> Tensor<2> {
    if strided {
        let transposed = (0..width)
            .flat_map(|c| (0..rows).map(move |r| data[r * width + c]))
            .collect::<Vec<_>>();
        Tensor::from_data(TensorData::new(transposed, [width, rows]), device).swap_dims(0, 1)
    } else {
        Tensor::from_data(TensorData::new(data.to_vec(), [rows, width]), device)
    }
}
#[cfg(any(feature = "vulkan", feature = "cuda"))]
pub fn reference(
    c: &Config,
    device: &Device,
    historical: Option<&std::path::Path>,
) -> Result<Value> {
    let device = device.clone().autodiff();
    let mut loss_cases = Vec::new();
    for (rows, width) in [(1, 1), (3, 7), (5, 257), (7, 35)] {
        for strided in [false, true] {
            for zeros in [false, true] {
                for lanes in [64, 256] {
                    let x = (0..rows * width)
                        .map(|i| {
                            if zeros {
                                0.
                            } else {
                                ((i * 37 % 101) as f32 - 50.) * 0.21
                            }
                        })
                        .collect::<Vec<_>>();
                    let target = (0..rows)
                        .map(|r| (r * 19 % width) as f32)
                        .collect::<Vec<_>>();
                    let weights = (0..rows)
                        .map(|r| if r % 3 == 1 { 0. } else { 0.5 + r as f32 * 0.3 })
                        .collect::<Vec<_>>();
                    let total = weights.iter().map(|&w| w as f64).sum::<f64>();
                    let mut ce = Vec::new();
                    let mut dx = Vec::new();
                    for r in 0..rows {
                        let row = &x[r * width..(r + 1) * width];
                        let maximum = row
                            .iter()
                            .map(|&x| x as f64)
                            .fold(f64::NEG_INFINITY, f64::max);
                        let sum = row.iter().map(|&x| (x as f64 - maximum).exp()).sum::<f64>();
                        ce.push((maximum - row[target[r] as usize] as f64) + sum.ln());
                        for (i, &z) in row.iter().enumerate() {
                            dx.push(
                                ((z as f64 - maximum).exp() / sum
                                    - f64::from(i == target[r] as usize))
                                    * weights[r] as f64
                                    / total,
                            );
                        }
                    }
                    let expected = ce
                        .iter()
                        .zip(&weights)
                        .map(|(z, &w)| z * w as f64)
                        .sum::<f64>()
                        / total;
                    let dw = ce
                        .iter()
                        .map(|z| (z - expected) / total)
                        .collect::<Vec<_>>();
                    let input = tensor2(&x, rows, width, strided, &device).require_grad();
                    let weight = Tensor::<1>::from_data(TensorData::new(weights, [rows]), &device)
                        .require_grad();
                    let labels = Tensor::<1>::from_data(TensorData::new(target, [rows]), &device);
                    let output = delta_kernels::cross_entropy_rows(input.clone(), labels, lanes);
                    let objective = (output.clone() * weight.clone()).sum() / weight.clone().sum();
                    let gradients = objective.clone().backward();
                    loss_cases.push(json!({"rows":rows,"width":width,"strided":strided,"zeros":zeros,"lanes":lanes,
                "loss_max_abs":compare("loss",&values(objective),&[expected])?,"row_max_abs":compare("rows",&values(output),&ce)?,
                "dx_max_abs":compare("loss dx",&values(input.grad(&gradients).unwrap()),&dx)?,"dw_max_abs":compare("loss weights",&values(weight.grad(&gradients).unwrap()),&dw)?}));
                }
            }
        }
    }
    let mut optimizer_cases = Vec::new();
    for strided in [false, true] {
        for scale in [0., 0.02, 2., 100.] {
            for decay in [0., 0.1] {
                let rows = 7;
                let width = 35;
                let input = (0..rows * width)
                    .map(|i| ((i * 17 % 53) as f32 - 26.) * 0.03)
                    .collect::<Vec<_>>();
                let mut fused = tensor2(&input, rows, width, strided, &device);
                let mut reference = fused.clone();
                let mut state = None;
                let mut ref_state = None;
                let native = AdamWConfig::new().with_weight_decay(decay)
                    .with_beta_1(c.optimizer_beta1).with_beta_2(c.optimizer_beta2)
                    .with_epsilon(c.optimizer_epsilon).build();
                let mut fused_config = c.clone();
                fused_config.update_backend = UpdateBackend::FusedGpu;
                let fused_optim = crate::update::AdamW::new(&fused_config, decay, 1.0);
                let mut scalar_x = input.iter().map(|&v| v as f64).collect::<Vec<_>>();
                let mut m = vec![0f64; rows * width];
                let mut v = m.clone();
                let mut error = 0f64;
                for step in 1..=5 {
                    let gradient = (0..rows * width)
                        .map(|i| ((i * 31 + step * 7) % 97) as f32 * scale - scale * 48.)
                        .collect::<Vec<_>>();
                    let g = tensor2(&gradient, rows, width, strided, &device);
                    let norm = gradient
                        .iter()
                        .map(|&v| (v as f64).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let coefficient = (1. / (norm + f32::MIN_POSITIVE as f64)).min(1.);
                    for i in 0..gradient.len() {
                        let g = gradient[i] as f64 * coefficient;
                        // Match Burn's f32 hyperparameter constants, not decimal-f64 substitutes.
                        m[i] = m[i] * c.optimizer_beta1 as f64 + g * (1f32 - c.optimizer_beta1) as f64;
                        v[i] = v[i] * c.optimizer_beta2 as f64 + g * g * (1f32 - c.optimizer_beta2) as f64;
                        let delta = (m[i] / (1f32 - c.optimizer_beta1.powi(step as i32)) as f64)
                            / ((v[i] / (1f32 - c.optimizer_beta2.powi(step as i32)) as f64).sqrt()
                                + c.optimizer_epsilon as f64);
                        scalar_x[i] = scalar_x[i] * (1. - 0.001 * decay as f64) - delta * 0.001;
                    }
                    // This single-tensor oracle supplies an already globally clipped gradient.
                    let clipped = g*coefficient;
                    (fused, state) = fused_optim.step(0.001, fused, clipped.clone(), state);
                    (reference, ref_state) = native.step(0.001, reference, clipped, ref_state);
                    error = error.max(compare("optimizer f64", &values(fused.clone()), &scalar_x)?);
                    compare(
                        "optimizer native",
                        &values(fused.clone()),
                        &values(reference.clone())
                            .into_iter()
                            .map(f64::from)
                            .collect::<Vec<_>>(),
                    )?;
                    let state = state.as_ref().unwrap();
                    compare("first moment", &values(state.momentum.moment_1.clone()), &m)?;
                    compare(
                        "second moment",
                        &values(state.momentum.moment_2.clone()),
                        &v,
                    )?;
                }
                optimizer_cases.push(json!({"strided":strided,"gradient_scale":scale,"weight_decay":decay,"steps":5,"max_abs":error}));
            }
        }
    }
    let mut bad = Vec::new();
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        let sum = crate::update::square_sum(Tensor::<1>::from_data([value], &device));
        ensure!(
            model::scalar(sum).is_err(),
            "nonfinite/overflow gradient was accepted"
        );
        bad.push(format!("{value:?}"));
    }
    let mut small = c.clone();
    small.d_model = 8;
    small.layers = 1;
    small.heads = 4;
    small.key_dim = 32;
    small.value_dim = 64;
    small.context = 16;
    small.batch_size = 1;
    // The missing-gradient rejection fixture is a tiny causal backbone, not
    // the caller's dimension-dependent Lngram or denoising configuration.
    small.lngram_layers.clear();
    small.wavelet_layers.clear();
    small.sbd = None;
    for backend in [UpdateBackend::Tensor, UpdateBackend::FusedGpu] {
        small.update_backend = backend;
        let model = LanguageModel::new(&small, &small.arms[0], &device)?;
        ensure!(
            model::checked_gradients(&model, model.head.weight.val().sum()).is_err(),
            "missing gradients accepted"
        );
    }
    let continuation = if let Some(historical) = historical {
        let mut native_config = c.clone();
        native_config.update_backend = UpdateBackend::Tensor;
        native_config.loss_backend = LossBackend::Tensor;
        let mut fused_config = native_config.clone();
        fused_config.update_backend = UpdateBackend::FusedGpu;
        fused_config.loss_backend = LossBackend::Fused256;
        let restored = checkpoint::load_model(
            LanguageModel::new(&native_config, &c.arms[0], &device)?,
            historical.join("model.burnpack"),
            &device,
        )?;
        let mut native = restored.clone();
        let mut fused = restored;
        fused.set_update_backend(UpdateBackend::FusedGpu);
        fused.set_loss_backend(LossBackend::Fused256);
        let mut native_optimizer =
            checkpoint::load_optimizer(&native_config, historical.join("optimizer.burnpack"))?;
        let mut fused_optimizer =
            checkpoint::load_optimizer(&fused_config, historical.join("optimizer.burnpack"))?;
        let corpus = train::loaded_corpus(c)?;
        let rows = data::examples(
            c,
            corpus.as_ref(),
            c.seed,
            checkpoint::load_state(historical.join("state.json"))?.step * c.batch_size,
            c.batch_size,
            "train",
        )?;
        let batch = Batch::new(&rows, &device);
        let mut continuation = Vec::new();
        for _ in 0..3 {
            let a = model::checked_gradients(&native, native.loss(native.forward(&batch), &batch))?;
            let b = model::checked_gradients(&fused, fused.loss(fused.forward(&batch), &batch))?;
            let gradient_error = compare_models(&native, Some(&a), &fused, Some(&b))?;
            native = native_optimizer.step(c.learning_rate, native, a);
            fused = fused_optimizer.step(c.learning_rate, fused, b);
            continuation.push(json!({"all_gradients_max_abs":gradient_error,"updated_parameters_max_abs":compare_models(&native,None,&fused,None)?}));
        }
        continuation
    } else {
        Vec::new()
    };
    let optional_maximum_continuation = optional_maximum_checkpoint(c, &device)?;
    let global_clipping = crate::update::clipping_checks(&device)?;
    let mut native_groups = c.clone(); native_groups.update_backend = UpdateBackend::Tensor;
    let mut fused_groups = c.clone(); fused_groups.update_backend = UpdateBackend::FusedGpu;
    let groups = json!({"native":crate::update::optimizer_group_checks(&native_groups,&device)?,
        "fused":crate::update::optimizer_group_checks(&fused_groups,&device)?});
    let module_updates = module_update_checks(c, &device)?;
    let report = json!({"passed":true,"loss_cases":loss_cases,"optimizer_cases":optimizer_cases,"nonfinite_rejections":bad,"missing_gradient_rejected":true,"historical_checkpoint":historical,"historical_three_step_continuation":continuation,"optional_maximum_checkpoint_continuation":optional_maximum_continuation,"absolute_tolerance":2e-5,"relative_tolerance":3e-4});
    let mut report = report;
    report["global_clipping"] = global_clipping;
    report["parameter_groups"] = groups;
    report["global_native_fused_schedule_resume"] = module_updates;
    report["soft_head_sparse_clock_restart"] = json!({
        "native":crate::update::soft_head_checks(&native_groups,&device)?,
        "fused":crate::update::soft_head_checks(&fused_groups,&device)?});
    Ok(report)
}
#[cfg(not(any(feature = "vulkan", feature = "cuda")))]
pub fn reference(_: &Config, _: &Device, _: Option<&std::path::Path>) -> Result<Value> {
    anyhow::bail!("GPU feature required")
}

/// Multi-tensor oracle: a subthreshold pair, unequal large tensors, sign changes,
/// and zero gradients; moments expose clipping errors hidden by Adam's first update.
#[cfg(any(feature = "vulkan", feature = "cuda"))]
fn module_update_checks(c: &Config, device: &Device) -> Result<Value> {
    let mut native_config = c.clone();
    native_config.update_backend = UpdateBackend::Tensor;
    native_config.train_tokens = c.tokens_per_step()*20;
    native_config.budget_file = None;
    native_config.gradient_clip = 1.0;
    native_config.warmup_fraction = 0.1;
    native_config.min_learning_rate_ratio = 0.1;
    let mut fused_config = native_config.clone();
    fused_config.update_backend = UpdateBackend::FusedGpu;
    let mut native = burn::nn::LinearConfig::new(1,1).init(device);
    let mut fused = native.clone();
    let mut native_optimizer = train::optimizer(&native_config);
    let mut fused_optimizer = train::optimizer(&fused_config);
    let mut scalar = [f64::from(values(native.weight.val())[0]),
        f64::from(values(native.bias.as_ref().unwrap().val())[0])];
    let mut first = [0.0f64;2]; let mut second = [0.0f64;2];
    let directory = c.artifact_dir.join(format!("global-optimizer-{}-{}",std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()));
    std::fs::create_dir_all(&directory)?;
    let mut resumed = None::<(burn::nn::Linear,crate::update::NeuralOptimizer)>;
    let mut maximum = 0.0f64;
    let mut schedule = Vec::new();
    for completed in 0..20usize {
        let raw = [[0.8f32,0.8], [4.0,2.0], [-0.2,0.5], [0.0,0.0]][completed%4];
        let norm = raw.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let coefficient = (1.0/norm).min(1.0);
        // Independent one-based Appendix A7 schedule for this 20-update fixture.
        let factor = if completed < 2 {(completed+1) as f64/2.0}
            else {0.1+0.9*0.5*(1.0+(std::f64::consts::PI*(completed-1) as f64/18.0).cos())};
        let lr = c.learning_rate*factor;
        schedule.push(lr);
        let gradient = |layer: &burn::nn::Linear| {
            let mut gradients = GradientsParams::new();
            gradients.register(layer.weight.id,Tensor::<2>::from_data([[raw[0]]],device));
            gradients.register(layer.bias.as_ref().unwrap().id,Tensor::<1>::from_data([raw[1]],device));
            gradients
        };
        let a = gradient(&native); let b = gradient(&fused);
        native = native_optimizer.step(c.learning_rate,native,a);
        fused = fused_optimizer.step(c.learning_rate,fused,b);
        for index in 0..2 {
            let g = f64::from(raw[index])*coefficient;
            first[index] = first[index]*f64::from(c.optimizer_beta1)+g*f64::from(1.0-c.optimizer_beta1);
            second[index] = second[index]*f64::from(c.optimizer_beta2)+g*g*f64::from(1.0-c.optimizer_beta2);
            let update = (first[index]/f64::from(1.0-c.optimizer_beta1.powi((completed+1) as i32)))
                / ((second[index]/f64::from(1.0-c.optimizer_beta2.powi((completed+1) as i32))).sqrt()
                    +f64::from(c.optimizer_epsilon));
            scalar[index] = scalar[index]*(if index==0 {1.0-lr*f64::from(c.weight_decay)} else {1.0})-lr*update;
        }
        let pair = |layer: &burn::nn::Linear| [values(layer.weight.val())[0],
            values(layer.bias.as_ref().unwrap().val())[0]];
        maximum = maximum.max(compare("global module f64",&pair(&fused),&scalar)?);
        compare("global module native",&pair(&native),&pair(&fused).map(f64::from))?;
        let signature = checkpoint::optimizer_signature(&fused_config,fused_optimizer.into_bytes()?)?;
        for (field,expected) in [("moment_1",first),("moment_2",second)] {
            let reader = burn_pack::Reader::from_bytes(fused_optimizer.into_bytes()?)?;
            let mut checked = 0;
            for tensor in reader.into_tensors()? {
                let (name,dtype,shape,id,bytes) = tensor.into_parts()?;
                if !name.ends_with(field) {continue;}
                let index = if id == Some(fused.weight.id.val()) {0}else{1};
                let data = TensorData::from_bytes(bytes,shape,dtype).try_to_vec::<f32>()
                    .map_err(|e|anyhow::anyhow!("global moment readback: {e:?}"))?;
                compare(field,&data,&[expected[index]])?;
                checked += 1;
            }
            ensure!(checked==2,"global module moment check omitted a tensor");
        }
        if let Some((model,mut optimizer)) = resumed.take() {
            let g = gradient(&model);
            let continued = optimizer.step(c.learning_rate,model,g);
            ensure!(pair(&continued)==pair(&fused),"resumed optimizer parameters differ");
            ensure!(checkpoint::optimizer_signature(&fused_config,optimizer.into_bytes()?)?==signature,
                "resumed schedule/moments differ");
            resumed = Some((continued,optimizer));
        }
        if completed == 6 {
            let path = directory.join("optimizer.burnpack");
            fused_optimizer.save(&path)?;
            resumed = Some((fused.clone(),checkpoint::load_optimizer(&fused_config,&path)?));
        }
    }
    Ok(json!({"passed":true,"steps":20,"resume_after":7,"learning_rates":schedule,
        "global_counter":fused_optimizer.completed_steps(),"f64_maximum_error":maximum,
        "native_fused_equal_within_unchanged_tolerance":true,"resumed_exact":true,"directory":directory}))
}

/// A valid native AMSGrad checkpoint can be consumed by the configured ordinary AdamW.
/// Its unused maximum must survive load, device migration, updates and another roundtrip.
#[cfg(any(feature = "vulkan", feature = "cuda"))]
fn optional_maximum_checkpoint(c: &Config, device: &Device) -> Result<Value> {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let directory = c
        .artifact_dir
        .join(format!("optional-maximum-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&directory)?;
    let mut layer = burn::nn::LinearConfig::new(2, 2)
        .with_bias(false)
        .init(device);
    let input = Tensor::<2>::from_data([[1.0f32, -0.2], [0.3, 0.4]], device);
    let gradients = |layer: &burn::nn::Linear, input: Tensor<2>| {
        GradientsParams::from_grads(layer.forward(input).square().sum().backward(), layer)
    };
    let mut producer = crate::update::NeuralOptimizer::new(c,
        train::optimizer_config(c).with_amsgrad(true).init());
    let gradient = gradients(&layer, input);
    layer = producer.step(c.learning_rate, layer, gradient);
    let model_path = directory.join("model.burnpack");
    let optimizer_path = directory.join("optimizer.burnpack");
    layer.clone().save_file(&model_path)?;
    producer.save(&optimizer_path)?;
    let target = burn::nn::LinearConfig::new(2, 2)
        .with_bias(false)
        .init(device);
    runtime::sync(device)?;
    let restored = target.try_load_file(&model_path)?;
    runtime::sync(device)?;
    ensure!(
        restored.weight.id == layer.weight.id,
        "checkpoint parameter ID changed"
    );
    ensure!(
        values(restored.weight.val()) == values(layer.weight.val()),
        "checkpoint values changed"
    );
    let mut native_config = c.clone();
    native_config.update_backend = UpdateBackend::Tensor;
    let mut fused_config = native_config.clone();
    fused_config.update_backend = UpdateBackend::FusedGpu;
    let mut native_optimizer = checkpoint::load_optimizer(&native_config, &optimizer_path)?;
    let mut fused_optimizer = checkpoint::load_optimizer(&fused_config, &optimizer_path)?;
    let maxima = |optimizer: &crate::update::NeuralOptimizer| -> Result<Value> {
        let signature = checkpoint::optimizer_signature(&native_config, optimizer.into_bytes()?)?;
        let tensors = signature["tensors"]
            .as_object()
            .expect("typed optimizer signature");
        let maximum = tensors
            .iter()
            .filter(|(name, _)| name.contains("max_moment_2"))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect::<serde_json::Map<String, Value>>();
        ensure!(
            !maximum.is_empty(),
            "regression checkpoint has no optional maximum"
        );
        Ok(Value::Object(maximum))
    };
    let original_maximum = maxima(&native_optimizer)?;
    ensure!(
        maxima(&fused_optimizer)? == original_maximum,
        "loaded optional maximum changed"
    );
    let mut native = restored.clone();
    let mut fused = restored;
    let mut maximum_parameter_error = 0.0f64;
    for _ in 0..3 {
        let zero = Tensor::<2>::zeros([2, 2], device);
        let a = gradients(&native, zero.clone());
        let b = gradients(&fused, zero);
        native = native_optimizer.step(c.learning_rate, native, a);
        fused = fused_optimizer.step(c.learning_rate, fused, b);
        maximum_parameter_error = maximum_parameter_error.max(compare(
            "optional-maximum checkpoint continuation",
            &values(fused.weight.val()),
            &values(native.weight.val())
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>(),
        )?);
        ensure!(
            maxima(&native_optimizer)? == original_maximum,
            "native maximum changed with AMSGrad disabled"
        );
        ensure!(
            maxima(&fused_optimizer)? == original_maximum,
            "fused update dropped or changed optional maximum"
        );
    }
    let continued = directory.join("continued-optimizer.burnpack");
    fused_optimizer.save(&continued)?;
    let restored_optimizer = checkpoint::load_optimizer(&fused_config, &continued)?;
    ensure!(
        maxima(&restored_optimizer)? == original_maximum,
        "continued checkpoint lost optional maximum"
    );
    Ok(json!({"passed":true,"directory":directory,"steps":3,
        "ordinary_AdamW_parameters_max_abs":maximum_parameter_error,
        "optional_maximum_exactly_preserved":true}))
}

#[cfg(any(feature = "vulkan", feature = "cuda"))]
struct Snapshot<'a> {
    gradients: Option<&'a GradientsParams>,
    values: Vec<f32>,
}
#[cfg(any(feature = "vulkan", feature = "cuda"))]
impl ModuleVisitor for Snapshot<'_> {
    fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
        self.values.extend(values(
            self.gradients
                .map(|g| g.get::<D>(p.id).unwrap())
                .unwrap_or_else(|| p.val()),
        ));
    }
}
#[cfg(any(feature = "vulkan", feature = "cuda"))]
fn compare_models(
    a: &LanguageModel,
    ga: Option<&GradientsParams>,
    b: &LanguageModel,
    gb: Option<&GradientsParams>,
) -> Result<f64> {
    let mut a_values = Snapshot {
        gradients: ga,
        values: Vec::new(),
    };
    a.visit(&mut a_values);
    let mut b_values = Snapshot {
        gradients: gb,
        values: Vec::new(),
    };
    b.visit(&mut b_values);
    compare(
        "all model parameters/gradients",
        &a_values.values,
        &b_values
            .values
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>(),
    )
}
struct CopyGradients<'a> {
    source: &'a GradientsParams,
    copy: GradientsParams,
}
impl ModuleVisitor for CopyGradients<'_> {
    fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
        self.copy
            .register(p.id, self.source.get::<D>(p.id).unwrap());
    }
}
fn copy_gradients(model: &LanguageModel, source: &GradientsParams) -> GradientsParams {
    let mut v = CopyGradients {
        source,
        copy: GradientsParams::new(),
    };
    model.visit(&mut v);
    v.copy
}
#[cfg(any(feature = "vulkan", feature = "cuda"))]
fn count(run: impl FnOnce() -> Result<()>) -> Result<Value> {
    use std::{
        collections::BTreeMap,
        sync::{Arc, Mutex},
    };
    #[derive(Default)]
    struct Counter(Mutex<BTreeMap<&'static str, usize>>);
    impl cubecl::logging::LaunchObserver for Counter {
        fn launched(&self, kernel: &'static str) {
            *self.0.lock().unwrap().entry(kernel).or_default() += 1;
        }
    }
    let counter = Arc::new(Counter::default());
    let guard = cubecl::logging::LaunchObservation::new(counter.clone());
    run()?;
    drop(guard);
    let kernels = counter.0.lock().unwrap();
    Ok(json!({"issued":kernels.values().sum::<usize>(),"by_kernel":*kernels}))
}
#[cfg(not(any(feature = "vulkan", feature = "cuda")))]
fn count(run: impl FnOnce() -> Result<()>) -> Result<Value> {
    run()?;
    Ok(json!(null))
}

pub fn benchmark(c: &Config, device: &Device, warmups: usize, trials: usize) -> Result<Value> {
    ensure!(
        !c.fusion
            && !c.autotune
            && c.task == "lm"
            && [1, 8].contains(&c.batch_size)
            && [128, 512].contains(&c.context),
        "requires historical non-fusion B1/8 C128/512 LM"
    );
    ensure!(
        warmups >= 1 && trials >= 3 && trials <= 31,
        "bounded warmups/trials"
    );
    let start = Instant::now();
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let directory = c
        .artifact_dir
        .join(format!("update-loss-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&directory)?;
    let path = directory.join("report.json");
    let corpus = train::loaded_corpus(c)?;
    let train_device = device.clone().autodiff();
    train_device.seed(c.seed);
    let base = LanguageModel::new(c, &c.arms[0], &train_device)?;
    let modes = [
        (UpdateBackend::Tensor, LossBackend::Tensor),
        (UpdateBackend::FusedGpu, LossBackend::Tensor),
        (UpdateBackend::Tensor, LossBackend::Fused64),
        (UpdateBackend::Tensor, LossBackend::Fused256),
        (UpdateBackend::FusedGpu, LossBackend::Fused256),
    ];
    let configs = modes
        .iter()
        .map(|&(update_backend, loss_backend)| {
            let mut c = c.clone();
            c.update_backend = update_backend;
            c.loss_backend = loss_backend;
            c
        })
        .collect::<Vec<_>>();
    let mut runners = configs
        .iter()
        .map(|c| {
            let mut model = base.clone();
            model.set_update_backend(c.update_backend);
            model.set_loss_backend(c.loss_backend);
            Runner::new(model, c)
        })
        .collect::<Vec<_>>();
    let mut samples = vec![Vec::new(); modes.len()];
    let mut report = json!({"schema":1,"completed":false,"config":c,"device":runtime::identity(device),"modes":modes,"samples_ms":samples,"directory":directory});
    runtime::write_json(&path, &report)?;
    for trial in 0..warmups + trials {
        ensure!(
            start.elapsed().as_secs_f64() < c.max_seconds.min(290.),
            "benchmark deadline; partial report retained"
        );
        for offset in 0..modes.len() {
            let mode = if trial % 2 == 0 {
                (offset + trial) % modes.len()
            } else {
                (modes.len() + trial - offset % modes.len()) % modes.len()
            };
            let ms = runners[mode].step(
                &configs[mode],
                corpus.as_ref(),
                &train_device,
                &mut Unprofiled,
            )?;
            if trial >= warmups {
                samples[mode].push(ms);
            }
        }
        report["samples_ms"] = json!(samples);
        runtime::write_json(&path, &report)?;
    }
    let mut dispatches = Vec::new();
    for mode in 0..modes.len() {
        dispatches.push(count(|| {
            runners[mode].step(
                &configs[mode],
                corpus.as_ref(),
                &train_device,
                &mut Unprofiled,
            )?;
            Ok(())
        })?);
    }
    report["dispatches"] = json!(dispatches);
    report["medians_ms"] = json!(samples.iter().map(|x| median(x)).collect::<Vec<_>>());
    // Actual-shape resident checks+clipping+update, separate from full-model step timing.
    let rows = data::examples(c, corpus.as_ref(), c.seed, 0, c.batch_size, "train")?;
    let batch = Batch::new(&rows, &train_device);
    let grads = model::checked_gradients(&base, base.loss(base.forward(&batch), &batch))?;
    let mut isolated = vec![Vec::new(); 2];
    let mut optims = [train::optimizer(&configs[0]), train::optimizer(&configs[1])];
    for trial in 0..warmups + trials {
        for mode in if trial % 2 == 0 { [0, 1] } else { [1, 0] } {
            let mut model = base.clone();
            model.set_update_backend(configs[mode].update_backend);
            let gradients = copy_gradients(&model, &grads);
            runtime::sync(device)?;
            let clock = Instant::now();
            model::check_parameter_gradients(&model, &gradients, &mut Unprofiled)?;
            let output = optims[mode].step(c.learning_rate, model, gradients);
            runtime::sync(device)?;
            if trial >= warmups {
                isolated[mode].push(clock.elapsed().as_secs_f64() * 1000.);
            }
            std::hint::black_box(output);
        }
    }
    report["isolated_guarded_update_ms"] = json!(isolated);
    let hidden = Tensor::<3>::random(
        [c.batch_size, c.context, c.d_model],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &train_device,
    )
    .require_grad();
    let mut loss_samples = vec![vec![Vec::new(); 2]; 3];
    for trial in 0..warmups + trials {
        for index in 0..3 {
            let mode = (index + trial) % 3;
            let backend = [
                LossBackend::Tensor,
                LossBackend::Fused64,
                LossBackend::Fused256,
            ][mode];
            for backward in if trial % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                runtime::sync(device)?;
                let clock = Instant::now();
                let logits =
                    base.head_forward(base.normalization_backend.rms(&base.norm, hidden.clone()));
                let objective = model::loss_selected(logits, &batch, backend);
                if backward {
                    std::hint::black_box(objective.backward());
                } else {
                    std::hint::black_box(objective);
                }
                runtime::sync(device)?;
                if trial >= warmups {
                    loss_samples[mode][usize::from(backward)]
                        .push(clock.elapsed().as_secs_f64() * 1000.);
                }
            }
        }
    }
    report["isolated_final_norm_head_loss_f_fb_ms"] = json!(loss_samples);
    // Both directions of the unchanged state layout and validation boundary.
    let checkpoint_path = directory.join("optimizer.burnpack");
    optims[1].save(&checkpoint_path)?;
    checkpoint::load_optimizer(&configs[0], &checkpoint_path)?;
    checkpoint::load_optimizer(&configs[1], &checkpoint_path)?;
    report["optimizer_layout_cross_load_passed"] = json!(true);
    report["sampled_reserved_bytes"] = json!(runtime::sampled_reserved(device));
    report["completed"] = json!(true);
    runtime::write_json(&path, &report)?;
    Ok(report)
}
