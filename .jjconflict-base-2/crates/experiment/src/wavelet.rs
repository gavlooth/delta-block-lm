//! Step 7 wavelet acceptance, paired cost, and matched-budget learning smoke.
use crate::{
    config::{Arm, Config},
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
};
use delta_mixers::{WaveletBackend, WaveletDyn, profile::Unprofiled};
use serde_json::{Value, json};
use std::time::Instant;
fn values<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.into_data().try_to_vec::<f32>().unwrap()
}
struct Snapshot<'a> {
    grads: Option<&'a GradientsParams>,
    parts: Vec<Vec<f32>>,
}
impl ModuleVisitor for Snapshot<'_> {
    fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
        self.parts.push(values(match self.grads {
            Some(g) => g.get::<D>(p.id).expect("all gradients checked"),
            None => p.val(),
        }));
    }
}
fn snapshot(m: &LanguageModel, g: Option<&GradientsParams>) -> Vec<Vec<f32>> {
    let mut s = Snapshot {
        grads: g,
        parts: vec![],
    };
    m.visit(&mut s);
    s.parts
}
fn close(a: &[Vec<f32>], b: &[Vec<f32>]) -> Result<f64> {
    ensure!(a.len() == b.len(), "model parameter count");
    let mut max = 0f64;
    for (a, b) in a.iter().zip(b) {
        ensure!(a.len() == b.len(), "model element count");
        for (&a, &b) in a.iter().zip(b) {
            let error = (a as f64 - b as f64).abs();
            ensure!(
                a.is_finite() && b.is_finite() && error <= 2e-5 + 3e-4 * (b as f64).abs(),
                "CPU model gate: {a} vs {b}, absolute error {error}"
            );
            max = max.max(error);
        }
    }
    Ok(max)
}
fn select(m: &mut LanguageModel, mode: usize) {
    for block in &mut m.blocks {
        if mode == 0 {
            block.wavelet = None;
            block.wavelet_norm = None;
            block.ohc_wavelet = None;
        } else if let Some(w) = block.wavelet.as_mut() {
            w.backend = if mode == 1 {
                WaveletBackend::Tensor
            } else {
                WaveletBackend::FusedGpu
            };
        }
    }
}
pub fn reference(c: &Config, device: &Device) -> Result<Value> {
    let component = delta_mixers::wavelet::numerical_checks(device).map_err(anyhow::Error::msg)?;
    let gate = model_gate(c, device)?;
    Ok(json!({"passed":true,"component":component,"cpu_model_gate":gate}))
}
#[cfg(not(feature = "cpu"))]
fn model_gate(_c: &Config, _device: &Device) -> Result<Value> {
    anyhow::bail!("wavelet model gate requires the cpu feature")
}
#[cfg(feature = "cpu")]
fn model_gate(c: &Config, device: &Device) -> Result<Value> {
    let mut tiny = Config::default();
    tiny.task = "lm".into();
    tiny.context = 17;
    tiny.batch_size = 2;
    tiny.layers = 1;
    tiny.d_model = 16;
    tiny.wavelet_layers = vec![0];
    tiny.seed = c.seed;
    let arm = Arm {
        name: "kronlstm".into(),
        mixer: "kronlstm".into(),
        cache: "none".into(),
    };
    let corpus = train::loaded_corpus(&tiny)?;
    let rows = data::examples(&tiny, corpus.as_ref(), tiny.seed, 0, 2, "train")?;
    let cpu = Device::ndarray().autodiff();
    cpu.seed(tiny.seed);
    let original = LanguageModel::new(&tiny, &arm, &cpu)?;
    let batch = Batch::new(&rows, &cpu);
    let logits = original.forward(&batch);
    let objective = original.loss(logits.clone(), &batch);
    let gradients = model::checked_gradients(&original, objective)?;
    let expected = snapshot(&original, Some(&gradients));
    let updated = train::optimizer(&tiny).step(tiny.learning_rate, original.clone(), gradients);
    let expected_updated = snapshot(&updated, None);
    let gpu = device.clone().autodiff();
    let base = original.fork(&gpu);
    let batch = Batch::new(&rows, &gpu);
    let mut checks = vec![];
    for mode in if c.backend == "cpu" {
        vec![1]
    } else {
        vec![1, 2]
    } {
        let mut candidate = base.clone();
        select(&mut candidate, mode);
        let actual = candidate.forward(&batch);
        let err = close(&[values(actual.clone())], &[values(logits.clone())])?;
        let grads = model::checked_gradients(&candidate, candidate.loss(actual, &batch))?;
        let grad_error = close(&snapshot(&candidate, Some(&grads)), &expected)?;
        let next = train::optimizer(&tiny).step(tiny.learning_rate, candidate, grads);
        let update_error = close(&snapshot(&next, None), &expected_updated)?;
        checks.push(json!({"backend":if mode==1{"tensor"}else{"fused_gpu"},"logits_max_abs":err,"all_gradients_max_abs":grad_error,"adamw_max_abs":update_error,"parameter_tensors":expected.len()}));
    }
    Ok(
        json!({"passed":true,"absolute_tolerance":2e-5,"relative_tolerance":3e-4,"shape":[2,17,16],"checks":checks}),
    )
}
#[cfg(any(feature = "vulkan", feature = "cuda"))]
fn launches(device: &Device, run: impl FnOnce() -> Result<()>) -> Result<Value> {
    use std::{
        collections::BTreeMap,
        sync::{Arc, Mutex},
    };
    #[derive(Default)]
    struct Count(Mutex<BTreeMap<&'static str, usize>>);
    impl cubecl::logging::LaunchObserver for Count {
        fn launched(&self, k: &'static str) {
            *self.0.lock().unwrap().entry(k).or_default() += 1;
        }
    }
    runtime::sync(device)?;
    let count = Arc::new(Count::default());
    let guard = cubecl::logging::LaunchObservation::new(count.clone());
    run()?;
    runtime::sync(device)?;
    drop(guard);
    let counts = count.0.lock().unwrap();
    Ok(json!({"issued_kernels":counts.values().sum::<usize>(),"by_kernel":*counts}))
}
#[cfg(not(any(feature = "vulkan", feature = "cuda")))]
fn launches(_device: &Device, _run: impl FnOnce() -> Result<()>) -> Result<Value> {
    anyhow::bail!("GPU build required")
}
fn paired(base: &[f64], other: &[f64]) -> Value {
    let differences = base
        .iter()
        .zip(other)
        .map(|(a, b)| a - b)
        .collect::<Vec<_>>();
    let med = median(&differences);
    let mad = median(
        &differences
            .iter()
            .map(|x| (x - med).abs())
            .collect::<Vec<_>>(),
    );
    json!({"baseline_minus_candidate_samples_ms":differences,"median_saved_ms":med,"median_absolute_deviation_ms":mad,"candidate_wins":differences.iter().filter(|&&x|x>0.).count(),"trials":differences.len(),"beyond_noise":med>3.*mad&&differences.iter().filter(|&&x|x>0.).count()*10>=differences.len()*9})
}
fn isolated(c: &Config, device: &Device, warmups: usize, trials: usize) -> Result<Value> {
    let train_device = device.clone().autodiff();
    train_device.seed(c.seed);
    let base = WaveletDyn::new(c.d_model, 4, 8, WaveletBackend::Tensor, &train_device);
    let x = Tensor::<3>::random(
        [c.batch_size, c.context, c.d_model],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        &train_device,
    )
    .require_grad();
    let valid = Tensor::ones([c.batch_size, c.context], &train_device);
    let reset = Tensor::zeros([c.batch_size, c.context], &train_device).slice_assign(
        [0..c.batch_size, 0..1],
        Tensor::ones([c.batch_size, 1], &train_device),
    );
    let modules = [base.clone(), {
        let mut m = base;
        m.backend = WaveletBackend::FusedGpu;
        m
    }];
    let mut samples: [[Vec<f64>; 2]; 2] =
        core::array::from_fn(|_| core::array::from_fn(|_| vec![]));
    let run = |backend: usize, backward: usize| -> Result<()> {
        let root = modules[backend]
            .forward(x.clone(), valid.clone(), reset.clone())
            .sum();
        if backward == 1 {
            let grads = root.backward();
            std::hint::black_box(x.grad(&grads).expect("input VJP"));
        } else {
            std::hint::black_box(root);
        }
        Ok(())
    };
    for iteration in 0..warmups + trials {
        for mode in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
            for backward in if iteration % 2 == 0 { [0, 1] } else { [1, 0] } {
                runtime::sync(device)?;
                let start = Instant::now();
                run(mode, backward)?;
                runtime::sync(device)?;
                if iteration >= warmups {
                    samples[mode][backward].push(start.elapsed().as_secs_f64() * 1000.);
                }
            }
        }
    }
    let rows=(0..2).map(|i|Ok(json!({"backend":if i==0{"tensor"}else{"fused_gpu"},"forward_samples_ms":samples[i][0],"forward_backward_samples_ms":samples[i][1],"forward_median_ms":median(&samples[i][0]),"forward_backward_median_ms":median(&samples[i][1]),"forward_launches":launches(device,||run(i,0))?,"forward_backward_launches":launches(device,||run(i,1))?}))).collect::<Result<Vec<_>>>()?;
    Ok(
        json!({"instances":1,"shape":[c.batch_size,c.context,c.d_model],"results":rows,"paired_forward":paired(&samples[0][0],&samples[1][0]),"paired_forward_backward":paired(&samples[0][1],&samples[1][1]),"scope":"resident full WaveletDyn, parameter and input VJP, includes metadata; synchronization brackets every sample"}),
    )
}
pub fn benchmark(c: &Config, device: &Device, warmups: usize, trials: usize) -> Result<Value> {
    ensure!(
        c.backend != "cpu" && !c.fusion && !c.wavelet_layers.is_empty(),
        "requires non-fusion GPU and wavelet_layers"
    );
    ensure!(
        warmups >= 2 && trials >= 11,
        "requires two warmups and at least eleven trials"
    );
    let isolated = isolated(c, device, warmups, trials)?;
    device.memory_cleanup();
    let corpus = train::loaded_corpus(c)?;
    let train_device = device.clone().autodiff();
    train_device.seed(c.seed);
    let base = LanguageModel::new(c, &c.arms[0], &train_device)?;
    let mut runners = (0..3)
        .map(|mode| {
            let mut m = base.clone();
            select(&mut m, mode);
            Runner::new(m, c)
        })
        .collect::<Vec<_>>();
    let mut samples: [Vec<f64>; 3] = core::array::from_fn(|_| vec![]);
    let mut peak = runtime::sampled_reserved(device);
    for iteration in 0..warmups + trials {
        for mode in if iteration % 2 == 0 {
            [0, 1, 2]
        } else {
            [2, 1, 0]
        } {
            let ms = runners[mode].step(c, corpus.as_ref(), &train_device, &mut Unprofiled)?;
            if iteration >= warmups {
                samples[mode].push(ms);
            }
            if let Some(n) = runtime::sampled_reserved(device) {
                peak = Some(peak.unwrap_or(0).max(n));
            }
        }
    }
    let mut rows = vec![];
    for mode in 0..3 {
        let name = ["off", "tensor", "fused_gpu"][mode];
        rows.push(json!({"mode":name,"step_samples_ms":samples[mode],"step_median_ms":median(&samples[mode]),"dispatches":crate::profile::dispatches(&mut runners[mode],c,corpus.as_ref(),&train_device)?}));
    }
    let d = c.d_model;
    let n = c.batch_size * c.context * d;
    Ok(
        json!({"schema":1,"command":"wavelet-benchmark","completed":true,"config":c,"device":runtime::identity(device),"dtype":"f32","warmups":warmups,"trials":trials,"isolated":isolated,"integrated":rows,"paired_tensor_to_fused":paired(&samples[1],&samples[2]),"paired_off_to_tensor":paired(&samples[0],&samples[1]),"paired_off_to_fused":paired(&samples[0],&samples[2]),"protocol":"matched initial common weights; independent AdamW trajectories, alternating forward/reverse order; same row indices/seed; steps include host batches, forward, loss, every gradient check, clipping and AdamW","memory":{"component_parameters_each":3*d*d+40*d+40,"with_prenorm_parameters_each":3*d*d+41*d+40,"table_bytes":0,"fixed_filter_device_bytes":0,"band_output_bytes_each":5*n*4,"fused_smoothing_vjp_retains_input":false,"fused_shared_metadata_bytes_each":2*c.batch_size*c.context*4,"sampled_reserved_bytes":peak,"limitation":"allocator pool sample across all trajectories; not physical peak or live activations; bands exclude projection, gate, value, pointwise and library matmul autodiff storage"}}),
    )
}
pub fn smoke(c: &Config, device: &Device) -> Result<Value> {
    ensure!(
        !c.wavelet_layers.is_empty(),
        "smoke requires enabled wavelet layers"
    );
    let corpus = train::loaded_corpus(c)?;
    let device = device.clone().autodiff();
    device.seed(c.seed);
    let base = LanguageModel::new(c, &c.arms[0], &device)?;
    let steps = 200;
    let mut losses: [Vec<f64>; 2] = core::array::from_fn(|_| vec![]);
    let mut models = [
        {
            let mut m = base.clone();
            select(&mut m, 0);
            Some(m)
        },
        Some(base),
    ];
    let mut optim = [train::optimizer(c), train::optimizer(c)];
    for step in 0..steps {
        let rows = data::examples(
            c,
            corpus.as_ref(),
            c.seed,
            step * c.batch_size,
            c.batch_size,
            "train",
        )?;
        let batch = Batch::new(&rows, &device);
        for i in if step % 2 == 0 { [0, 1] } else { [1, 0] } {
            let m = models[i].take().unwrap();
            let loss = m.loss(m.forward(&batch), &batch);
            losses[i].push(model::scalar(loss.clone())?);
            let grads = model::checked_gradients(&m, loss)?;
            models[i] = Some(optim[i].step(c.learning_rate, m, grads));
        }
    }
    let rows=(0..2).map(|i|{let first=losses[i][..20].iter().sum::<f64>()/20.;let last=losses[i][steps-20..].iter().sum::<f64>()/20.;json!({"enabled":i==1,"first20_mean_loss":first,"last20_mean_loss":last,"decreased":last<first,"losses":losses[i]})}).collect::<Vec<_>>();
    ensure!(
        rows.iter().all(|r| r["decreased"] == true),
        "smoke loss did not decrease"
    );
    Ok(
        json!({"completed":true,"passed":true,"config":c,"steps_each":steps,"training_tokens_each":steps*c.batch_size*c.context,"seed":c.seed,"results":rows,"interpretation":"finite-gradient learning smoke only; no quality comparison"}),
    )
}
