use crate::{
    config::Config,
    data::{self, Batch},
    model::{self, LanguageModel},
    profile::{Runner, median},
    runtime, train,
};
use anyhow::{Result, ensure};
use burn::{
    module::{ModuleVisitor, Param},
    optim::GradientsParams,
    prelude::*,
    tensor::Distribution,
};
use delta_mixers::{
    lngram::{LngramBackend, LngramLayout},
    profile::Unprofiled,
};
use serde_json::{Value, json};
use std::time::Instant;

fn values<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.into_data().iter::<f32>().collect()
}
struct Snapshot<'a> {
    gradients: Option<&'a GradientsParams>,
    values: Vec<Vec<f32>>,
}
impl ModuleVisitor for Snapshot<'_> {
    fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
        self.values.push(values(match self.gradients {
            Some(g) => g.get::<D>(p.id).expect("all gradients"),
            None => p.val(),
        }));
    }
}
fn snapshot(model: &LanguageModel, gradients: Option<&GradientsParams>) -> Vec<Vec<f32>> {
    let mut s = Snapshot {
        gradients,
        values: Vec::new(),
    };
    model.visit(&mut s);
    s.values
}
fn compare(a: &[Vec<f32>], b: &[Vec<f32>]) -> Result<f64> {
    ensure!(a.len() == b.len(), "Lngram model parameter count");
    let mut error = 0.0f64;
    for (a, b) in a.iter().zip(b) {
        ensure!(a.len() == b.len(), "Lngram model shape");
        for (&a, &b) in a.iter().zip(b) {
            let e = (a as f64 - b as f64).abs();
            ensure!(
                a.is_finite() && b.is_finite() && e <= 2e-5 + 3e-4 * (b as f64).abs(),
                "Lngram CPU model: {a} vs {b}"
            );
            error = error.max(e);
        }
    }
    Ok(error)
}

#[cfg(feature = "cpu")]
fn model_gate(c: &Config, device: &Device) -> Result<Value> {
    let mut config = c.clone();
    config.backend = "cpu".into();
    config.fusion = false;
    config.autotune = false;
    config.d_model = 32;
    config.layers = 3;
    config.context = 11;
    config.batch_size = 2;
    config.scan_backend = delta_mixers::ScanBackend::TensorSolver;
    config.hola_backend = delta_mixers::HolaBackend::Tensor;
    config.projection_backend = delta_mixers::ProjectionBackend::Separate;
    config.normalization_backend = delta_mixers::NormalizationBackend::Tensor;
    config.convolution_backend = delta_mixers::ConvolutionBackend::Tensor;
    config.head_backend = crate::config::HeadBackend::Native;
    config.loss_backend = crate::config::LossBackend::Tensor;
    config.update_backend = crate::config::UpdateBackend::Tensor;
    config.epilogue_backend = crate::config::EpilogueBackend::Native;
    // Component gate: other Step 7 sublayers are covered by their own gates and kron-model-check.
    config.ohc_streams = false;
    config.ohc_backend = crate::config::Config::default().ohc_backend;
    config.wavelet_layers.clear();
    config.lngram_layers = vec![0, 1, 2];
    config.lngram_routes = config.d_model / 2;
    config.lngram_bits = 2;
    config.lngram_memory_dim = 3;
    config.lngram_backend = LngramBackend::Tensor;
    config.lngram_orders = vec![2, 3];
    config.lngram_no_smoothing = false;
    let cpu = Device::ndarray().autodiff();
    cpu.seed(919);
    let mut base = LanguageModel::new(&config, &config.arms[0], &cpu)?;
    for block in &mut base.blocks {
        let module = block.lngram.as_mut().unwrap();
        for table in &mut module.tables {
            table.weight = Param::from_tensor(Tensor::random(
                table.weight.val().dims(),
                Distribution::Uniform(-0.2, 0.2),
                &cpu,
            ));
        }
        let smooth = module.smoothing.as_mut().unwrap();
        smooth.weight = Param::from_tensor(Tensor::random(
            [4, 32],
            Distribution::Uniform(-0.1, 0.1),
            &cpu,
        ));
    }
    let rows: Vec<_> = (0..2)
        .map(|b| data::Example {
            input: (0..11).map(|t| ((t * 7 + b * 3) % 31) as i32).collect(),
            target: (0..11).map(|t| ((t * 3 + b * 9) % 31) as i32).collect(),
            weight: vec![1.; 11],
            groups: vec![None; 11],
        })
        .collect();
    let reference_batch = Batch::new(&rows, &cpu);
    let reference = base.forward(&reference_batch);
    let gradients =
        model::checked_gradients(&base, base.loss(reference.clone(), &reference_batch))?;
    let reference_gradients = snapshot(&base, Some(&gradients));
    let reference_update =
        train::optimizer(&config).step(config.learning_rate, base.clone(), gradients);
    let mut modes = Vec::new();
    let mut backends = vec![LngramBackend::Tensor, LngramBackend::FlatSelect];
    if matches!(c.backend.as_str(), "vulkan" | "cuda") {
        backends.push(LngramBackend::FusedGpu);
    }
    for mode in backends {
        let target = device.clone().autodiff();
        let mut model = base.clone().fork(&target);
        model.set_lngram_backend(mode);
        if matches!(c.backend.as_str(), "vulkan" | "cuda") {
            model.set_scan_backend(c.scan_backend);
            model.set_hola_backend(c.hola_backend);
            model.set_projection_backend(c.projection_backend);
            model.set_convolution_backend(c.convolution_backend);
            model.set_normalization_backend(c.normalization_backend);
            model.set_epilogue_backend(c.epilogue_backend);
            model.set_head_backend(c.head_backend);
            model.set_loss_backend(c.loss_backend);
            model.set_update_backend(c.update_backend);
        }
        let batch = Batch::new(&rows, &target);
        let logits = model.forward(&batch);
        let output_error = compare(&[values(logits.clone())], &[values(reference.clone())])?;
        let grads = model::checked_gradients(&model, model.loss(logits, &batch))?;
        let gradient_error = compare(&snapshot(&model, Some(&grads)), &reference_gradients)?;
        let mut target_config = config.clone();
        target_config.backend = c.backend.clone();
        target_config.update_backend = c.update_backend;
        let updated = train::optimizer(&target_config).step(config.learning_rate, model, grads);
        let update_error = compare(
            &snapshot(&updated, None),
            &snapshot(&reference_update, None),
        )?;
        modes.push(json!({"backend":mode,"logits_max_abs":output_error,"all_gradients_max_abs":gradient_error,"updated_parameters_max_abs":update_error}));
    }
    // The optimizer's production path must not decay any unindexed, nonzero row with fresh state.
    let target = device.clone().autodiff();
    let model = base.fork(&target);
    let table_ids = model.lngram_table_ids();
    let mut grads = GradientsParams::new();
    struct Zeros<'a>(&'a mut GradientsParams);
    impl ModuleVisitor for Zeros<'_> {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            self.0.register(
                p.id,
                Tensor::<D>::zeros(p.val().dims(), &p.val().device()).inner(),
            );
        }
    }
    model.visit(&mut Zeros(&mut grads));
    let mut before = Vec::new();
    for block in &model.blocks {
        for table in &block.lngram.as_ref().unwrap().tables {
            let dims = table.weight.val().dims();
            let g = Tensor::<2>::zeros(dims, &target)
                .slice_assign([1..2, 0..dims[1]], Tensor::ones([1, dims[1]], &target));
            grads.register(table.weight.id, g.inner());
            before.push(values(table.weight.val()));
        }
    }
    let mut optimizer_config = config.clone();
    optimizer_config.update_backend = c.update_backend;
    let updated = train::optimizer(&optimizer_config).step(config.learning_rate, model, grads);
    let mut index = 0;
    for block in &updated.blocks {
        for table in &block.lngram.as_ref().unwrap().tables {
            let width = table.weight.val().dims()[1];
            let after = values(table.weight.val());
            for i in 0..after.len() {
                ensure!(
                    (after[i] != before[index][i]) == (i / width == 1),
                    "Lngram AdamW changed an unindexed row"
                );
                if i / width == 1 {
                    let delta = (before[index][i] - after[i]) as f64;
                    ensure!(
                        (delta - 5.0 * config.learning_rate).abs()
                            <= 2e-5 + 3e-4 * (5.0 * config.learning_rate),
                        "Lngram table learning-rate multiplier was not applied"
                    );
                }
            }
            index += 1;
        }
    }
    Ok(
        json!({"passed":true,"modes":modes,"parameter_tensors":reference_gradients.len(),"table_parameter_ids":table_ids.len(),"fresh_state_only_indexed_rows_changed":true,"table_learning_rate_multiplier":5,"table_weight_decay":0}),
    )
}
#[cfg(not(feature = "cpu"))]
fn model_gate(_c: &Config, _device: &Device) -> Result<Value> {
    anyhow::bail!("Lngram model gate requires CPU feature")
}

pub fn reference(c: &Config, device: &Device) -> Result<Value> {
    let component = delta_mixers::lngram::numerical_checks(device).map_err(anyhow::Error::msg)?;
    let model = model_gate(c, device)?;
    Ok(
        json!({"passed":true,"component":component,"cpu_model":model,"device":runtime::identity(device),"tolerance":{"absolute":2e-5,"relative":3e-4}}),
    )
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
        fn launched(&self, k: &'static str) {
            *self.0.lock().unwrap().entry(k).or_default() += 1;
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
    Ok(Value::Null)
}

pub fn benchmark(c: &Config, device: &Device, warmups: usize, trials: usize) -> Result<Value> {
    ensure!(
        warmups >= 2 && trials >= 11,
        "Lngram requires two warmups and >=11 paired trials"
    );
    ensure!(
        !c.lngram_layers.is_empty(),
        "configure Lngram injection layers"
    );
    let target = device.clone().autodiff();
    target.seed(c.seed);
    let base = LanguageModel::new(c, &c.arms[0], &target)?;
    let corpus = train::loaded_corpus(c)?;
    let module = base.blocks[c.lngram_layers[0]]
        .lngram
        .as_ref()
        .unwrap()
        .clone();
    let input = Tensor::<3>::random(
        [c.batch_size, c.context, c.d_model],
        Distribution::Uniform(-0.5, 0.5),
        &target,
    )
    .require_grad();
    let valid = Tensor::ones([c.batch_size, c.context], &target);
    let reset = Tensor::<2>::zeros([c.batch_size, c.context], &target).slice_assign(
        [0..c.batch_size, 0..1],
        Tensor::ones([c.batch_size, 1], &target),
    );
    let layout = LngramLayout::new(valid, reset);
    let mut isolated = Vec::new();
    let mut f: [Vec<f64>; 2] = core::array::from_fn(|_| Vec::new());
    let mut fb = f.clone();
    for iteration in 0..warmups + trials {
        for mode in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
            let mut layer = module.clone();
            layer.backend = if mode == 0 {
                LngramBackend::Tensor
            } else {
                LngramBackend::FusedGpu
            };
            for backward in if iteration % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                runtime::sync(&target)?;
                let clock = Instant::now();
                let y = layer.forward(input.clone(), &layout);
                if backward {
                    std::hint::black_box(y.square().sum().backward());
                } else {
                    std::hint::black_box(y);
                }
                runtime::sync(&target)?;
                if iteration >= warmups {
                    if backward {
                        fb[mode].push(clock.elapsed().as_secs_f64() * 1000.)
                    } else {
                        f[mode].push(clock.elapsed().as_secs_f64() * 1000.)
                    }
                }
            }
        }
    }
    for mode in 0..2 {
        let mut layer = module.clone();
        layer.backend = if mode == 0 {
            LngramBackend::Tensor
        } else {
            LngramBackend::FusedGpu
        };
        let launches = count(|| {
            std::hint::black_box(
                layer
                    .forward(input.clone(), &layout)
                    .square()
                    .sum()
                    .backward(),
            );
            runtime::sync(&target)
        })?;
        isolated.push(json!({"backend":layer.backend,"forward_median_ms":median(&f[mode]),"forward_backward_median_ms":median(&fb[mode]),"forward_samples_ms":f[mode],"forward_backward_samples_ms":fb[mode],"forward_backward_launches":launches}));
    }
    let mut configs = [c.clone(), c.clone(), c.clone()];
    configs[0].lngram_layers.clear();
    configs[1].lngram_backend = LngramBackend::Tensor;
    configs[2].lngram_backend = LngramBackend::FusedGpu;
    let mut models = [base.clone(), base.clone(), base.clone()];
    for b in &mut models[0].blocks {
        b.lngram = None;
        b.ohc_lngram = None;
    }
    models[1].set_lngram_backend(LngramBackend::Tensor);
    models[2].set_lngram_backend(LngramBackend::FusedGpu);
    let parameters: Vec<_> = models.iter().map(|m| m.num_params()).collect();
    let mut runners: Vec<_> = models
        .into_iter()
        .zip(&configs)
        .map(|(m, c)| Runner::new(m, c))
        .collect();
    let mut step: [Vec<f64>; 3] = core::array::from_fn(|_| Vec::new());
    for iteration in 0..warmups + trials {
        let order = if iteration % 2 == 0 {
            [0, 1, 2]
        } else {
            [2, 1, 0]
        };
        for mode in order {
            let ms =
                runners[mode].step(&configs[mode], corpus.as_ref(), &target, &mut Unprofiled)?;
            if iteration >= warmups {
                step[mode].push(ms);
            }
        }
    }
    let mut integrated = Vec::new();
    for mode in 0..3 {
        let launches = count(|| {
            runners[mode].step(&configs[mode], corpus.as_ref(), &target, &mut Unprofiled)?;
            Ok(())
        })?;
        integrated.push(json!({"selection":(["off","tensor","fused_gpu"][mode]),"parameters":parameters[mode],"parameter_bytes":parameters[mode]*4,"median_ms":median(&step[mode]),"samples_ms":step[mode],"launches":launches}));
    }
    let config = c.lngram_config();
    let table_bytes: usize = config
        .orders
        .iter()
        .map(|&n| config.table_rows(n) * config.memory_dim * 4)
        .sum();
    let slots: usize = config.orders.iter().sum();
    Ok(json!({
        "completed":true,"config":c,"seed":c.seed,"warmups":warmups,"trials":trials,
        "device":runtime::identity(device),"precision":"float32",
        "fused_design":"shared_exact_symbol_cache_exclusive_route_column_vjp",
        "isolated":isolated,"guarded_steps":integrated,
        "table_bytes_per_injection":table_bytes,"table_bytes_total":table_bytes*c.lngram_layers.len(),
        "one_bit_counterfactual_retained_bytes_per_injection":c.batch_size*c.context*config.routes*config.bits_per_route*config.memory_dim*slots*4,
        "fused_counterfactual_retained_bytes":0,
        "fused_symbol_cache_bytes_per_injection":c.batch_size*c.context*config.routes*4,
        "sampled_pool_reserved_bytes":runtime::sampled_reserved(&target),
        "memory_note":"logical tensor sizes and sampled pool reservations, not physical peak; table gradients and two Adam moments are dense; counterfactual retained bytes apply to tensor, not fused",
        "method":"paired alternating order, resident isolated inputs, synchronized AD-recording F and F+B with square-sum objective; complete guarded step includes real corpus preparation/upload, all parameters, loss readback, finite checks, clipping and AdamW; initialization/JIT warmups/report IO excluded"
    }))
}

/// Paired tiny byte-LM learning check; same seed, examples and step budget, no quality comparison.
pub fn learning_smoke(c: &Config, device: &Device) -> Result<Value> {
    let target = device.clone().autodiff();
    let mut config = c.clone();
    config.d_model = 32;
    config.layers = 3;
    config.context = 16;
    config.batch_size = 1;
    config.lngram_layers = vec![0, 1, 2];
    config.lngram_routes = config.d_model / 2;
    config.lngram_bits = 2;
    config.lngram_memory_dim = 3;
    config.pilot_steps = 200;
    target.seed(c.seed);
    let base = LanguageModel::new(&config, &config.arms[0], &target)?;
    let mut off = base.clone();
    for b in &mut off.blocks {
        b.lngram = None;
        b.ohc_lngram = None;
    }
    let mut configs = [config.clone(), config.clone()];
    configs[0].lngram_layers.clear();
    let mut models = [Some(off), Some(base)];
    let mut optimizers = [train::optimizer(&configs[0]), train::optimizer(&configs[1])];
    let rows: Vec<_> = (0..4)
        .map(|i| data::Example {
            input: (0..16).map(|t| ((t + i) % 4) as i32 + 65).collect(),
            target: (0..16).map(|t| ((t + i + 1) % 4) as i32 + 65).collect(),
            weight: vec![1.; 16],
            groups: vec![None; 16],
        })
        .collect();
    let batches: Vec<_> = rows
        .iter()
        .map(|r| Batch::new(std::slice::from_ref(r), &target))
        .collect();
    let mut losses: [Vec<f64>; 2] = core::array::from_fn(|_| Vec::new());
    for step in 0..200 {
        for mode in if step % 2 == 0 { [0, 1] } else { [1, 0] } {
            let model = models[mode].take().unwrap();
            let batch = &batches[step % 4];
            let loss = model.loss(model.forward(batch), batch);
            losses[mode].push(model::scalar(loss.clone())?);
            let g = model::checked_gradients(&model, loss)?;
            models[mode] = Some(optimizers[mode].step(config.learning_rate, model, g));
        }
    }
    runtime::sync(&target)?;
    let mut results = Vec::new();
    for mode in 0..2 {
        let first = losses[mode][..10].iter().sum::<f64>() / 10.;
        let last = losses[mode][190..].iter().sum::<f64>() / 10.;
        ensure!(last < first, "Lngram learning smoke did not decrease");
        results.push(json!({"enabled":mode==1,"first10_mean":first,"last10_mean":last,"losses":losses[mode]}));
    }
    Ok(
        json!({"passed":true,"steps":200,"seed":c.seed,"config":config,"task":"fixed synthetic byte-pattern next-token learning, not heldout quality","results":results}),
    )
}

#[derive(Module, Debug)]
struct TableParameters {
    tables: Vec<burn::nn::Embedding>,
}

/// Per-injection resident F/FB attribution, separate from ordinary paired full steps.
pub fn attribution(c: &Config, device: &Device, warmups: usize, trials: usize) -> Result<Value> {
    use delta_mixers::lngram::attribution::{Fixture, Part};
    ensure!(
        matches!(c.backend.as_str(), "vulkan" | "cuda") && !c.fusion,
        "Lngram attribution requires non-fusion GPU"
    );
    ensure!(
        warmups >= 2 && trials >= 11,
        "Lngram attribution requires two warmups and eleven trials"
    );
    let target = device.clone().autodiff();
    target.seed(c.seed);
    let mut layer = c.lngram_config().init(&target);
    layer.backend = LngramBackend::Tensor;
    let fixture = Fixture::new(&layer, c.batch_size, c.context, c.d_model, &target);
    let mut parts = Vec::new();
    for part in Fixture::parts(&layer) {
        let mut f = Vec::new();
        let mut fb = Vec::new();
        for iteration in 0..warmups + trials {
            for backward in if iteration % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                runtime::sync(&target)?;
                let clock = Instant::now();
                let y = fixture.forward(&layer, part);
                if backward {
                    std::hint::black_box(y.backward());
                } else {
                    std::hint::black_box(y);
                }
                runtime::sync(&target)?;
                if iteration >= warmups {
                    if backward {
                        fb.push(clock.elapsed().as_secs_f64() * 1000.)
                    } else {
                        f.push(clock.elapsed().as_secs_f64() * 1000.)
                    }
                }
            }
        }
        let forward_launches = count(|| {
            std::hint::black_box(fixture.forward(&layer, part));
            runtime::sync(&target)
        })?;
        let backward_launches = count(|| {
            std::hint::black_box(fixture.forward(&layer, part).backward());
            runtime::sync(&target)
        })?;
        parts.push(json!({"part":part.name(&layer),"instances":1,
            "forward_median_ms":median(&f),"forward_backward_median_ms":median(&fb),
            "derived_backward_ms":median(&fb)-median(&f),"forward_samples_ms":f,"forward_backward_samples_ms":fb,
            "forward_launches":forward_launches,"forward_backward_launches":backward_launches}));
    }
    let tables = TableParameters {
        tables: layer.tables.clone(),
    };
    let root = (0..layer.orders.len())
        .map(|o| fixture.forward(&layer, Part::Gather(o)))
        .reduce(|a, b| a + b)
        .unwrap();
    let source = GradientsParams::from_grads(root.backward(), &tables);
    let copy = |tables: &TableParameters| {
        let mut g = GradientsParams::new();
        for table in &tables.tables {
            g.register(
                table.weight.id,
                source.get::<2>(table.weight.id).expect("table VJP"),
            );
        }
        g
    };
    let mut optimizer = burn::optim::ModuleOptimizer::from(crate::update::LngramAdamW {
        fused: !c.update_backend.is_tensor(),
        clip: c.gradient_clip,
    });
    let mut current = Some(tables);
    let mut samples = Vec::new();
    for iteration in 0..warmups + trials {
        let tables = current.take().unwrap();
        let gradients = copy(&tables);
        runtime::sync(&target)?;
        let clock = Instant::now();
        current = Some(optimizer.step(c.learning_rate, tables, gradients));
        runtime::sync(&target)?;
        if iteration >= warmups {
            samples.push(clock.elapsed().as_secs_f64() * 1000.);
        }
    }
    let table_launches = count(|| {
        let tables = current.take().unwrap();
        let g = copy(&tables);
        current = Some(optimizer.step(c.learning_rate, tables, g));
        runtime::sync(&target)
    })?;
    let result = json!({"completed":true,"config":c,"device":runtime::identity(device),"seed":c.seed,
        "warmups":warmups,"trials":trials,"per_injection":parts,
        "table_AdamW":{"instances":1,"median_ms":median(&samples),"samples_ms":samples,"launches":table_launches},
        "memory":{"one_bit_counterfactual_difference_bytes_per_injection":c.batch_size*c.context*layer.routes*layer.bits_per_route*layer.memory_dim*layer.orders.iter().sum::<usize>()*4,
            "fused_counterfactual_retained_bytes":0,
            "fused_saved_logits_bytes_per_injection":c.batch_size*c.context*layer.routes*layer.bits_per_route*4,
            "fused_symbol_cache_bytes_per_injection":c.batch_size*c.context*layer.routes*4},
        "method":"resident independent leaf inputs; actual production operators; alternating synchronized F/FB with sum roots; per injection, not three-injection totals. Gather FB includes zero-filled dense table gradient and deterministic route scatter. Counterfactual current rows are prepared outside timing, as production reuses current gather. Readout excludes smoothing. Codes/addresses includes sigmoid probability VJP; integer codes have no derivative. Fused memory combines codes, address, gather and both analytic VJPs; not additive with baseline rows. AdamW uses real gather-gradient support, dense moments, x5 lr, zero decay, clipping and warmed state; gradient generation and whole-model finite guards excluded. Do not sum independent stage medians into a full-step prediction."});
    runtime::write_json(&c.artifact_dir.join("lngram-attribution.json"), &result)?;
    Ok(result)
}
