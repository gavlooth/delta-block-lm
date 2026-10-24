//! Bounded HOLA attribution and paired guarded-step comparison.
use crate::{
    config::{Arm, Config},
    runtime,
};
use anyhow::{Result, ensure};
use burn::{nn::RmsNormConfig, prelude::*};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Instant,
};

#[cfg(any(feature = "vulkan", feature = "cuda"))]
#[derive(Default)]
struct Timings(Mutex<BTreeMap<&'static str, Vec<(f64, String)>>>);
#[cfg(any(feature = "vulkan", feature = "cuda"))]
impl cubecl::logging::LaunchObserver for Timings {
    fn launched(&self, _: &'static str) {}
    fn timing(&self) -> cubecl::logging::TimingRequest {
        cubecl::logging::TimingRequest::Resolved
    }
    fn timed(
        &self,
        kernel: &'static str,
        duration: cubecl::logging::Duration,
        method: cubecl::logging::TimingMethod,
    ) {
        self.0
            .lock()
            .unwrap()
            .entry(kernel)
            .or_default()
            .push((duration.as_secs_f64() * 1000., format!("{method:?}")));
    }
}

pub fn cost(
    c: &Config,
    _arm: &Arm,
    device: &Device,
    _contexts: &[usize],
    warmups: usize,
    trials: usize,
) -> Result<Value> {
    ensure!(
        warmups > 0 && trials > 0 && trials <= 31,
        "positive bounded samples required"
    );
    ensure!(
        c.backend == "vulkan" && !c.fusion && !c.autotune,
        "HOLA cost requires non-fusion Vulkan without autotune"
    );
    ensure!(c.hola_backend.is_fused(), "HOLA cost requires an explicit fused read selector");
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    {
        let device = device.clone().autodiff();
        let mut cases = Vec::new();
        for batch in [1, 8] {
            for capacity in [16, 64] {
                device.seed(c.seed);
                let input = |width| {
                    Tensor::<4>::random(
                        [batch, 512, 4, width],
                        burn::tensor::Distribution::Uniform(-0.5, 0.5),
                        &device,
                    )
                    .require_grad()
                };
                let q = input(32);
                let k = input(32);
                let v = input(64);
                let scores = input(1).detach();
                let valid = Tensor::<2>::ones([batch, 512], &device);
                let docs = valid.clone().zeros_like();
                let sink = Tensor::<1>::zeros([4], &device).require_grad();
                let qnorm = RmsNormConfig::new(32).with_epsilon(1e-5).init(&device);
                let knorm = RmsNormConfig::new(32).with_epsilon(1e-5).init(&device);
                let run = |norms: bool, backward: bool| {
                    let output = if norms {
                        if c.hola_backend == delta_mixers::HolaBackend::FusedGpuPairsNorms {
                            delta_mixers::NormalizationBackend::FusedNorms.rms(&qnorm, q.clone())
                                + delta_mixers::NormalizationBackend::FusedNorms
                                    .rms(&knorm, k.clone())
                        } else {
                            qnorm.forward(q.clone()) + knorm.forward(k.clone())
                        }
                    } else {
                        delta_kernels::hola_read(
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            scores.clone(),
                            valid.clone(),
                            docs.clone(),
                            sink.clone(),
                            capacity,
                            16,
                            false,
                            false,
                            c.hola_backend != delta_mixers::HolaBackend::FusedGpu,
                        )
                        .output
                    };
                    let gradients = backward.then(|| output.clone().backward());
                    std::hint::black_box((output, gradients));
                };
                for _ in 0..warmups {
                    for norms in [false, true] {
                        run(norms, true);
                        runtime::sync(&device)?;
                    }
                }
                let mut rows = Vec::new();
                for norms in [false, true] {
                    let mut ordinary = Vec::new();
                    let mut forward = Vec::new();
                    for trial in 0..trials {
                        for backward in if trial % 2 == 0 {
                            [false, true]
                        } else {
                            [true, false]
                        } {
                            runtime::sync(&device)?;
                            let clock = Instant::now();
                            run(norms, backward);
                            runtime::sync(&device)?;
                            let ms = clock.elapsed().as_secs_f64() * 1000.;
                            if backward {
                                ordinary.push(ms);
                            } else {
                                forward.push(ms);
                            }
                        }
                    }
                    let observed = Arc::new(Timings::default());
                    let guard = cubecl::logging::LaunchObservation::new(observed.clone());
                    for _ in 0..trials {
                        run(norms, true);
                        runtime::sync(&device)?;
                    }
                    drop(guard);
                    rows.push(json!({"scope":if norms {"two_learned_RMSNorms"} else {"fused_read"},"ordinary_forward_samples_ms":forward,"ordinary_forward_median_ms":crate::profile::median(&forward),"ordinary_fb_samples_ms":ordinary,"ordinary_fb_median_ms":crate::profile::median(&ordinary),"serialized_kernel_samples_ms_and_method":*observed.0.lock().unwrap()}));
                }
                cases.push(json!({"batch":batch,"context":512,"capacity":capacity,"heads":4,"key":32,"value":64,"chunk":16,"measurements":rows}));
            }
        }
        Ok(
            json!({"command":"hola-cost","completed":true,"config":c,"device":runtime::identity(&device),"warmups":warmups,"trials":trials,"protocol":"resident independent seeded uniform[-.5,.5] leaves; valid all one/doc zero; ordinary forward+backward outer sync; separate serialized per-launch observation includes timing method; per-layer, not extrapolated model costs", "cases":cases}),
        )
    }
    #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
    anyhow::bail!("HOLA cost requires GPU features")
}
