//! oHC acceptance and paired component/guarded-step experiments.
use crate::{
    config::Config,
    data::{self, Batch},
    model::{self, LanguageModel},
    profile::{self, Runner},
    runtime, train,
};
use anyhow::{Result, ensure};
use burn::{
    module::{ModuleVisitor, Param},
    optim::GradientsParams,
    prelude::*,
};
use delta_mixers::{
    ohc::{OhcBackend, OhcReadoutConfig, OhcStageConfig, StreamInitConfig},
    profile::Unprofiled,
};
use serde_json::{Value, json};
use std::time::Instant;
struct Snapshot<'a> {
    grads: Option<&'a GradientsParams>,
    values: Vec<Vec<f32>>,
}
impl ModuleVisitor for Snapshot<'_> {
    fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
        let x = if let Some(g) = self.grads {
            g.get::<D>(p.id).expect("checked gradient")
        } else {
            p.val()
        };
        self.values.push(x.into_data().try_to_vec::<f32>().unwrap());
    }
}
fn snapshot(m: &LanguageModel, g: Option<&GradientsParams>) -> Vec<Vec<f32>> {
    let mut s = Snapshot {
        grads: g,
        values: vec![],
    };
    m.visit(&mut s);
    s.values
}
fn equivalent(a: &[Vec<f32>], b: &[Vec<f32>]) -> Result<f64> {
    ensure!(a.len() == b.len(), "parameter count mismatch");
    let mut max = 0.0f64;
    for (a, b) in a.iter().zip(b) {
        ensure!(a.len() == b.len(), "shape mismatch");
        for (&a, &b) in a.iter().zip(b) {
            let d = (a as f64 - b as f64).abs();
            ensure!(
                a.is_finite() && b.is_finite() && d <= 2e-5 + 3e-4 * (b as f64).abs(),
                "model gate: {a} != {b}, error {d}"
            );
            max = max.max(d);
        }
    }
    Ok(max)
}
fn values<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.into_data().try_to_vec::<f32>().unwrap()
}
fn batch(device: &Device, changed: Option<usize>) -> Batch {
    let mut input: Vec<i32> = (0..18).map(|i| 1 + (i * 13 % 251) as i32).collect();
    if let Some(i) = changed {
        input[i] = 127;
    }
    let mut valid = vec![1.; 18];
    valid[7] = 0.;
    valid[8] = 0.;
    valid[17] = 0.;
    Batch {
        input: Tensor::from_data(TensorData::new(input, [2, 9]), device),
        target: Tensor::from_data(TensorData::new(vec![3i32; 18], [2, 9]), device),
        weight: Tensor::from_data(TensorData::new(valid.clone(), [2, 9]), device),
        valid: Tensor::from_data(TensorData::new(valid, [2, 9]), device),
        reset: Tensor::from_data(
            TensorData::new(
                vec![
                    1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
                ],
                [2, 9],
            ),
            device,
        ),
    }
}
fn model_gate(c: &Config, device: &Device) -> Result<Value> {
    let mut small = c.clone();
    small.task = "lm".into();
    small.d_model = 8;
    small.layers = 1;
    small.context = 9;
    small.batch_size = 2;
    small.ohc_streams = true;
    small.ohc_backend = OhcBackend::Tensor;
    small.scan_backend = delta_mixers::ScanBackend::TensorSolver;
    small.hola_backend = delta_mixers::HolaBackend::Tensor;
    small.projection_backend = delta_mixers::ProjectionBackend::Separate;
    small.normalization_backend = delta_mixers::NormalizationBackend::Tensor;
    small.convolution_backend = delta_mixers::ConvolutionBackend::Tensor;
    small.head_backend = crate::config::HeadBackend::Native;
    small.epilogue_backend = crate::config::EpilogueBackend::Native;
    small.loss_backend = crate::config::LossBackend::Tensor;
    small.update_backend = crate::config::UpdateBackend::Tensor;
    // Component gate: other Step 7 sublayers are covered by their own gates and kron-model-check.
    small.wavelet_layers.clear();
    small.lngram_layers.clear();
    let cpu = Device::ndarray().autodiff();
    cpu.seed(819);
    let reference = LanguageModel::new(&small, &small.arms[0], &cpu)?;
    let rb = batch(&cpu, None);
    let logits = reference.forward(&rb);
    let loss = reference.loss(logits.clone(), &rb);
    let reference_loss = model::scalar(loss.clone())?;
    let rg = model::checked_gradients(&reference, loss)?;
    let gpu = c.backend != "cpu";
    let target_device = device.clone().autodiff();
    let mut target = reference.clone().fork(&target_device);
    if gpu {
        target.set_ohc_backend(OhcBackend::FusedReadWrite);
    }
    let tb = batch(&target_device, None);
    let tl = target.forward(&tb);
    let target_loss = target.loss(tl.clone(), &tb);
    let tg = model::checked_gradients(&target, target_loss.clone())?;
    let logits_error = equivalent(&[values(tl.clone())], &[values(logits)])?;
    let grads_error = equivalent(
        &snapshot(&target, Some(&tg)),
        &snapshot(&reference, Some(&rg)),
    )?;
    let ref_updated = train::optimizer(&small).step(small.learning_rate, reference, rg);
    let target_updated = train::optimizer(&small).step(small.learning_rate, target.clone(), tg);
    let update_error = equivalent(
        &snapshot(&target_updated, None),
        &snapshot(&ref_updated, None),
    )?;
    let base = values(tl);
    let vocab = small.vocab();
    let mut causality = 0.0f64;
    for changed in [3, 5, 12] {
        let new = values(target.forward(&batch(&target_device, Some(changed))));
        for pos in 0..18 {
            let same_row = pos / 9 == changed / 9;
            let same_doc = if changed < 9 {
                (pos < 4) == (changed < 4)
            } else {
                (pos < 11) == (changed < 11)
            };
            if !same_row || !same_doc || pos < changed {
                for k in 0..vocab {
                    let d = (new[pos * vocab + k] as f64 - base[pos * vocab + k] as f64).abs();
                    ensure!(
                        d <= 2e-6,
                        "causal/document regression at {pos} from {changed}: {d}"
                    );
                    causality = causality.max(d);
                }
            }
        }
    }
    Ok(
        json!({"passed":true,"cpu_tensor_vs_selected_gpu":gpu,"logits_max_abs":logits_error,"all_parameter_gradients_max_abs":grads_error,"AdamW_parameters_max_abs":update_error,"reference_loss":reference_loss,"target_loss":model::scalar(target_loss)?,"causality_max_abs":causality,"padding_internal_resets_tail":true}),
    )
}
pub fn reference(c: &Config, device: &Device) -> Result<Value> {
    Ok(
        json!({"component":delta_mixers::ohc::numerical_checks(device).map_err(anyhow::Error::msg)?,"model":model_gate(c,device)?}),
    )
}
fn timed<T>(device: &Device, f: impl FnOnce() -> T) -> Result<(T, f64)> {
    runtime::sync(device)?;
    let now = Instant::now();
    let y = f();
    runtime::sync(device)?;
    Ok((y, now.elapsed().as_secs_f64() * 1000.))
}
fn summary(v: &[f64]) -> Value {
    json!({"median_ms":profile::median(v),"samples_ms":v})
}
fn paired(base: &[f64], candidate: &[f64]) -> Value {
    let differences: Vec<_> = base.iter().zip(candidate).map(|(a, b)| a - b).collect();
    let mid = profile::median(&differences);
    let deviations: Vec<_> = differences.iter().map(|v| (v - mid).abs()).collect();
    let mad = profile::median(&deviations);
    json!({"baseline_minus_candidate_ms":differences,"median_ms":mid,"MAD_ms":mad,"candidate_wins":differences.iter().filter(|&&x|x>0.).count(),"beyond_noise":mid>2.*mad&&differences.iter().filter(|&&x|x>0.).count()>=base.len()*9/10})
}
#[cfg(any(feature = "vulkan", feature = "cuda"))]
fn launches<T>(device: &Device, run: impl FnOnce() -> T) -> Result<Value> {
    use std::{
        collections::BTreeMap,
        sync::{Arc, Mutex},
    };
    #[derive(Default)]
    struct Count(Mutex<BTreeMap<&'static str, usize>>);
    impl cubecl::logging::LaunchObserver for Count {
        fn launched(&self, name: &'static str) {
            *self.0.lock().unwrap().entry(name).or_default() += 1;
        }
    }
    runtime::sync(device)?;
    let count = Arc::new(Count::default());
    let guard = cubecl::logging::LaunchObservation::new(count.clone());
    let result = run();
    runtime::sync(device)?;
    drop(guard);
    drop(result);
    let kernels = count.0.lock().unwrap();
    Ok(json!({"issued":kernels.values().sum::<usize>(),"by_kernel":*kernels}))
}
#[cfg(not(any(feature = "vulkan", feature = "cuda")))]
fn launches<T>(_: &Device, _: impl FnOnce() -> T) -> Result<Value> {
    Ok(Value::Null)
}
fn isolated(c: &Config, device: &Device, warmups: usize, trials: usize) -> Result<Value> {
    let d = c.d_model;
    let device = device.clone().autodiff();
    device.seed(c.seed);
    let x = Tensor::<3>::random(
        [c.batch_size, c.context, d],
        burn::tensor::Distribution::Uniform(-0.4, 0.4),
        &device,
    )
    .require_grad();
    let valid = Tensor::<2>::ones([c.batch_size, c.context], &device);
    let init = StreamInitConfig::new().with_d_model(d).init(&device);
    let stage = OhcStageConfig::new().with_d_model(d).init(&device);
    let head = OhcReadoutConfig::new().with_d_model(d).init(&device);
    let run = |mode| {
        let read = stage.read_selected(init.forward(x.clone(), valid.clone()), mode);
        let update = read.work.clone() * 0.2;
        head.readout(stage.write(read, update, valid.clone(), mode))
    };
    let modes = [
        OhcBackend::Tensor,
        OhcBackend::FusedGpu,
        OhcBackend::FusedReadWrite,
    ];
    let mut fs: [Vec<f64>; 3] = core::array::from_fn(|_| vec![]);
    let mut bs = fs.clone();
    for iter in 0..warmups + trials {
        let mut order = [iter % 3, (iter + 1) % 3, (iter + 2) % 3];
        if iter % 2 == 1 {
            order.reverse();
        }
        for j in order {
            let (y, f) = timed(&device, || run(modes[j]))?;
            drop(y);
            let (g, b) = timed(&device, || run(modes[j]).square().sum().backward())?;
            drop(g);
            if iter >= warmups {
                fs[j].push(f);
                bs[j].push(b);
            }
        }
    }
    let dispatches = modes
        .iter()
        .map(|&mode| {
            Ok(json!({
                "forward":launches(&device, ||run(mode))?,
                "forward_backward":launches(&device, ||run(mode).square().sum().backward())?
            }))
        })
        .collect::<Result<Vec<Value>>>()?;
    Ok(
        json!({"scope":"stream init + one complete stage with F(u)=0.2u + learned normalized readout; resident f32 inputs; no model projections/optimizer","modes":(0..3).map(|i|json!({"ohc_backend":modes[i],"forward":summary(&fs[i]),"forward_backward":summary(&bs[i]),"dispatches":dispatches[i]})).collect::<Vec<_>>(),"paired_forward":paired(&fs[0],&fs[1]),"paired_forward_backward":paired(&bs[0],&bs[1]),"paired_write_to_readwrite_forward":paired(&fs[1],&fs[2]),"paired_write_to_readwrite_forward_backward":paired(&bs[1],&bs[2]),"pool_reserved_bytes":runtime::sampled_reserved(&device)}),
    )
}
pub fn benchmark(c: &Config, device: &Device, warmups: usize, trials: usize) -> Result<Value> {
    ensure!(
        c.backend != "cpu" && !c.fusion && !c.autotune,
        "oHC comparison needs explicit non-fusion GPU without autotune"
    );
    ensure!(
        warmups >= 2 && trials >= 11,
        "oHC protocol requires >=2 warmups and >=11 trials"
    );
    let path = c.artifact_dir.join("benchmark.json");
    let started = Instant::now();
    let mut report = json!({"schema":1,"command":"ohc-benchmark","completed":false,"config":c,"device":runtime::identity(device),"dtype":"f32","warmups_per_mode":warmups,"trials":trials,"protocol":"paired, alternating forward/reverse mode order, identical batches and step indices, independent optimizer trajectories; clocks synchronize before and after; complete steps include host batch generation/upload, logits/loss readback, every parameter gradient check, clipping and AdamW","memory_limitation":"pool reservations include all live scratch models and allocator reuse; not physical peak"});
    runtime::write_json(&path, &report)?;
    report["isolated"] = isolated(c, device, warmups, trials)?;
    runtime::write_json(&path, &report)?;
    device.memory_cleanup();
    let corpus = train::loaded_corpus(c)?;
    let td = device.clone().autodiff();
    let mut configs: Vec<_> = (0..4).map(|_| c.clone()).collect();
    configs[0].ohc_streams = false;
    configs[0].ohc_backend = OhcBackend::Tensor;
    configs[1].ohc_streams = true;
    configs[1].ohc_backend = OhcBackend::Tensor;
    configs[2].ohc_streams = true;
    configs[2].ohc_backend = OhcBackend::FusedGpu;
    configs[3].ohc_streams = true;
    configs[3].ohc_backend = OhcBackend::FusedReadWrite;
    td.seed(c.seed);
    let off = LanguageModel::new(&configs[0], &c.arms[0], &td)?;
    let _ = snapshot(&off, None);
    let tensor = off.clone().with_ohc(c.d_model, &td);
    let mut fused = tensor.clone();
    fused.set_ohc_backend(OhcBackend::FusedGpu);
    let mut readwrite = tensor.clone();
    readwrite.set_ohc_backend(OhcBackend::FusedReadWrite);
    let params = vec![
        off.num_params(),
        tensor.num_params(),
        fused.num_params(),
        readwrite.num_params(),
    ];
    let mut runners = vec![
        Runner::new(off, &configs[0]),
        Runner::new(tensor, &configs[1]),
        Runner::new(fused, &configs[2]),
        Runner::new(readwrite, &configs[3]),
    ];
    let mut steps: [Vec<f64>; 4] = core::array::from_fn(|_| vec![]);
    let mut reserved = 0;
    for iter in 0..warmups + trials {
        let mut order = [iter % 4, (iter + 1) % 4, (iter + 2) % 4, (iter + 3) % 4];
        if iter % 2 == 1 {
            order.reverse();
        }
        for j in order {
            ensure!(
                started.elapsed().as_secs_f64() < c.max_seconds.min(290.),
                "oHC benchmark deadline"
            );
            let ms = runners[j].step(&configs[j], corpus.as_ref(), &td, &mut Unprofiled)?;
            if iter >= warmups {
                steps[j].push(ms);
            }
            reserved = reserved.max(runtime::sampled_reserved(&td).unwrap_or(0));
        }
        report["completed_trials"] = json!(iter.saturating_sub(warmups) + 1);
        runtime::write_json(&path, &report)?;
    }
    let dispatches: Vec<_> = runners
        .iter_mut()
        .zip(&configs)
        .map(|(r, c)| profile::dispatches(r, c, corpus.as_ref(), &td))
        .collect::<Result<_>>()?;
    let n = c.batch_size * c.context;
    let d = c.d_model;
    let stages = 2 * c.layers;
    let overhead = params[1] - params[0];
    report["complete_guarded_step"] = json!({"modes":(0..4).map(|j|json!({"ohc_streams":configs[j].ohc_streams,"ohc_backend":configs[j].ohc_backend,"parameters":params[j],"step":summary(&steps[j]),"dispatches":dispatches[j]})).collect::<Vec<_>>(),"paired_on_vs_off_tensor":paired(&steps[0],&steps[1]),"paired_tensor_vs_fused":paired(&steps[1],&steps[2]),"paired_write_to_readwrite":paired(&steps[2],&steps[3]),"sampled_pool_reserved_bytes":reserved});
    report["memory"] = json!({"added_parameters":overhead,"added_parameter_bytes_f32":overhead*4,"table_bytes":0,"single_stream_bytes":n*d*4,"four_stream_bytes":n*4*d*4,"extra_residual_bytes":n*3*d*4,"per_stage_retained_stream_bytes":n*4*d*4,"all_stage_retained_stream_bytes":stages*n*4*d*4,"per_stage_rotation_bytes":n*16*4,"per_stage_projected_gate_bytes":n*16*4,"fused_write_retained_inputs_bytes":n*(4*d+16+4+d)*4,"fused_write_backward_output_bytes":n*(4*d+16+4+d)*4,"scope":"exact tensor sizes, not summed physical peak; autodiff retains additional normalized streams, gates and sublayer activations; fused write retains X,Q,gamma*b,F and recomputes reductions"});
    report["completed"] = json!(true);
    report["elapsed_seconds"] = json!(started.elapsed().as_secs_f64());
    runtime::write_json(&path, &report)?;
    Ok(report)
}
pub fn learning_smoke(c: &Config, device: &Device) -> Result<Value> {
    let td = device.clone().autodiff();
    let corpus = train::loaded_corpus(c)?;
    let steps = c.pilot_steps;
    ensure!(steps >= 200, "oHC learning smoke requires >=200 steps");
    let mut result = Vec::new();
    let mut off_config = c.clone();
    off_config.ohc_streams = false;
    off_config.ohc_backend = OhcBackend::Tensor;
    td.seed(c.seed);
    let base = LanguageModel::new(&off_config, &c.arms[0], &td)?;
    let _ = snapshot(&base, None);
    for enabled in [false, true] {
        let mut cfg = c.clone();
        cfg.ohc_streams = enabled;
        if !enabled {
            cfg.ohc_backend = OhcBackend::Tensor;
        }
        td.seed(c.seed);
        let mut m = if enabled {
            base.clone().with_ohc(c.d_model, &td)
        } else {
            base.clone()
        };
        m.set_ohc_backend(cfg.ohc_backend);
        let mut optim = train::optimizer(&cfg);
        let mut losses = Vec::new();
        let now = Instant::now();
        for step in 0..steps {
            let rows = data::examples(
                &cfg,
                corpus.as_ref(),
                cfg.seed,
                step * cfg.batch_size,
                cfg.batch_size,
                "train",
            )?;
            let batch = Batch::new(&rows, &td);
            let loss = m.loss(m.forward(&batch), &batch);
            losses.push(model::scalar(loss.clone())?);
            let g = model::checked_gradients(&m, loss)?;
            m = optim.step(cfg.learning_rate, m, g);
            ensure!(
                now.elapsed().as_secs_f64() < c.max_seconds.min(280.),
                "learning smoke deadline"
            );
        }
        let first = losses[..20].iter().sum::<f64>() / 20.;
        let last = losses[steps - 20..].iter().sum::<f64>() / 20.;
        ensure!(last < first, "oHC smoke loss did not decrease");
        result.push(json!({"ohc_streams":enabled,"ohc_backend":cfg.ohc_backend,"seed":c.seed,"steps":steps,"tokens":steps*c.batch_size*c.context,"first20_mean_loss":first,"last20_mean_loss":last,"losses":losses,"seconds":now.elapsed().as_secs_f64()}));
    }
    let result = json!({"passed":true,"config":c,"device":runtime::identity(device),"arms":result,"scope":"short optimization stability only; not a quality comparison"});
    runtime::write_json(&c.artifact_dir.join("learning-smoke.json"), &result)?;
    Ok(result)
}
pub fn attribution(c: &Config, device: &Device) -> Result<Value> {
    use burn::tensor::activation::sigmoid;
    let device = device.clone().autodiff();
    device.seed(c.seed);
    let (b, t, d) = (c.batch_size, c.context, c.d_model);
    let init = StreamInitConfig::new().with_d_model(d).init(&device);
    let stage = OhcStageConfig::new().with_d_model(d).init(&device);
    let head = OhcReadoutConfig::new().with_d_model(d).init(&device);
    let x = Tensor::<3>::random(
        [b, t, d],
        burn::tensor::Distribution::Uniform(-0.4, 0.4),
        &device,
    )
    .require_grad();
    let valid = Tensor::<2>::ones([b, t], &device);
    let streams = init
        .forward(x.clone(), valid.clone())
        .detach()
        .require_grad();
    let p = stage
        .gate_projection
        .forward(streams.clone().reshape([b, t, 4 * d]))
        .detach()
        .require_grad();
    let read = stage.read(streams.clone());
    let a = read.read_weights.clone().detach().require_grad();
    let w = read.write_weights.clone().detach().require_grad();
    let q = read.mix.clone().detach().require_grad();
    let logits = (p.clone().narrow(2, 8, 8) * stage.mix_scale
        + stage.mix_bias.val().unsqueeze::<3>())
    .detach()
    .require_grad();
    let update = (read.work * 0.2).detach().require_grad();
    let names = [
        "stream_init",
        "gate_GEMM_4d_to_16",
        "gate_RMS_and_sigmoid_epilogue",
        "stream_RMS_and_weighted_read",
        "quaternion_to_SO4",
        "write_tensor",
        "write_fused",
        "read_complete",
        "readout",
        "four_stream_memory_traffic_proxy",
        "gate_RMS_fused_only",
        "stream_RMS_weighted_read_fused",
        "quaternion_to_SO4_fused",
        "read_complete_fused",
    ];
    let run = |part| -> Tensor<1> {
        match part {
            0 => init
                .forward(x.clone(), valid.clone())
                .reshape([b * t * 4 * d]),
            1 => stage
                .gate_projection
                .forward(streams.clone().reshape([b, t, 4 * d]))
                .reshape([b * t * 16]),
            2 => {
                let flat = streams.clone().reshape([b, t, 4 * d]);
                let inverse = (flat.square().mean_dim(2) + 1e-6).sqrt().recip();
                let z = p.clone() * inverse;
                let a = sigmoid(
                    z.clone().narrow(2, 0, 4) * stage.read_scale.val().unsqueeze::<3>()
                        + stage.read_bias.val().unsqueeze::<3>(),
                ) + 1e-6;
                let w = sigmoid(
                    z.clone().narrow(2, 4, 4) * stage.write_gate_scale.val().unsqueeze::<3>()
                        + stage.write_bias.val().unsqueeze::<3>(),
                ) * 2.;
                let q = z.narrow(2, 8, 8) * stage.mix_scale + stage.mix_bias.val().unsqueeze::<3>();
                Tensor::cat(vec![a, w, q], 2).reshape([b * t * 16])
            }
            3 => (stage.stream_norm.forward(streams.clone()) * a.clone().unsqueeze_dim::<4>(3))
                .sum_dim(2)
                .reshape([b * t * d]),
            4 => stage
                .rotation
                .rotation(logits.clone())
                .reshape([b * t * 16]),
            5 | 6 => if part == 5 {
                OhcBackend::Tensor
            } else {
                OhcBackend::FusedGpu
            }
            .write(
                streams.clone(),
                q.clone(),
                w.clone(),
                update.clone(),
                stage.write_scale.val(),
            )
            .reshape([b * t * 4 * d]),
            7 | 13 => {
                let r = stage.read_selected(
                    streams.clone(),
                    if part == 13 {
                        OhcBackend::FusedReadWrite
                    } else {
                        OhcBackend::Tensor
                    },
                );
                Tensor::cat(
                    vec![
                        r.work.reshape([b * t * d]),
                        r.mix.reshape([b * t * 16]),
                        r.write_weights.reshape([b * t * 4]),
                    ],
                    0,
                )
            }
            8 => head.readout(streams.clone()).reshape([b * t * d]),
            10 => OhcBackend::FusedReadWrite
                .gate_norm(streams.clone(), p.clone())
                .reshape([b * t * 16]),
            11 => OhcBackend::FusedReadWrite
                .weighted_read(streams.clone(), a.clone(), &stage.stream_norm)
                .reshape([b * t * d]),
            12 => stage
                .rotation
                .rotation_selected(logits.clone(), true)
                .reshape([b * t * 16]),
            _ => (streams.clone() + streams.clone() * 1e-7).reshape([b * t * 4 * d]),
        }
    };
    let mut rows = Vec::new();
    for (part, name) in names.iter().enumerate() {
        let (mut f, mut fb) = (Vec::new(), Vec::new());
        for trial in 0..13 {
            let (y, ms) = timed(&device, || run(part))?;
            drop(y);
            let (g, bs) = timed(&device, || run(part).square().sum().backward())?;
            drop(g);
            if trial >= 2 {
                f.push(ms);
                fb.push(bs);
            }
        }
        rows.push(json!({"part":name,"forward":summary(&f),"forward_backward":summary(&fb),"forward_launches":launches(&device,||run(part))?,"forward_backward_launches":launches(&device,||run(part).square().sum().backward())?}));
    }
    let report = json!({"completed":true,"config":c,"device":runtime::identity(&device),"warmups":2,"trials":11,"rows":rows,"stream_bytes_f32":b*t*4*d*4,"scope":"resident per-stage pieces; each backward has its own square-sum objective and independently differentiable intermediate inputs; read_complete concatenates work, rotation and write gates so every read path receives an adjoint; pieces are not additive end-to-end times; memory proxy is a scale+add read/write on four streams, not measured physical bandwidth"});
    runtime::write_json(&c.artifact_dir.join("attribution.json"), &report)?;
    Ok(report)
}
#[cfg(test)]
#[test]
fn cpu_ohc_model_gate() {
    model_gate(&Config::default(), &Device::ndarray()).unwrap();
}
