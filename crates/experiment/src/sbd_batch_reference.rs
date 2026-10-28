//! Fixed-tolerance equivalence of block-axis batching and historical per-block SBD.
use crate::{
    config::{Arm, Config, SbdBlockExecution, SbdConfig, SbdFeedback},
    data::{Batch, Example},
    kron_benchmark::{parameters, values},
    model::{self, LanguageModel},
    sbd::{sample_training_blocks, TrainingBlock},
    sbd_training::CleanInputs,
    soft::SoftState,
};
use anyhow::{Context, Result, ensure};
use burn::{module::Module, prelude::*, tensor::activation::log_softmax};
use delta_mixers::sbd::{InBlockReads, NoisyReadBackend};
use serde_json::{Value, json};

type ParameterSnapshot = Vec<(Vec<usize>, Vec<f32>)>;

fn compare(actual: &[f32], reference: &[f32]) -> Result<Value> {
    crate::sbd::compare(actual, &reference.iter().map(|&x| f64::from(x)).collect::<Vec<_>>())
}

fn gradient_snapshot(model: &LanguageModel, loss: Tensor<1>, noisy: bool) -> Result<ParameterSnapshot> {
    let unused = model::unused_denoising_parameters(noisy, false);
    let (gradients, report) = model::checked_gradients_with_unused(model, loss, &unused, &mut delta_mixers::profile::Unprofiled)?;
    let mut visited = model.clone();
    if !noisy {
        ensure!(report.get("sbd_mask").is_some_and(|count| count.missing_steps == 1), "causal draw fabricated a mask gradient");
        visited.sbd_mask = None;
    }
    parameters(&visited, Some(&gradients))
}

fn compare_gradients(actual: &ParameterSnapshot, reference: &ParameterSnapshot) -> Result<Value> {
    ensure!(actual.len() == reference.len(), "batched SBD parameter count differs");
    let mut reports = Vec::with_capacity(reference.len());
    let mut maximum_absolute_error = 0.0f64;
    let mut maximum_tolerance_ratio = 0.0f64;
    let mut elements = 0usize;
    for (index, ((actual_shape, actual), (reference_shape, reference))) in actual.iter().zip(reference).enumerate() {
        ensure!(actual_shape == reference_shape, "batched SBD parameter {index} shape differs");
        let mut report = compare(actual, reference).with_context(|| format!("batched SBD parameter {index}, shape {reference_shape:?}"))?;
        maximum_absolute_error = maximum_absolute_error.max(report["max_abs"].as_f64().unwrap());
        maximum_tolerance_ratio = maximum_tolerance_ratio.max(report["maximum_tolerance_ratio"].as_f64().unwrap());
        elements += reference.len();
        report["parameter_index"] = json!(index);
        report["shape"] = json!(reference_shape);
        reports.push(report);
    }
    Ok(json!({
        "passed": true,
        "parameter_tensors": reports.len(),
        "compared_elements": elements,
        "max_abs": maximum_absolute_error,
        "maximum_tolerance_ratio": maximum_tolerance_ratio,
        "parameters": reports,
    }))
}

fn rows(count: usize, length: usize) -> Vec<Example> {
    (0..count).map(|row| {
        let sequence = std::iter::once(256).chain((0..length).map(|position| ((position * 37 + row * 53 + 11) % 256) as i32)).collect::<Vec<_>>();
        Example {
            input: sequence[..length].to_vec(),
            target: sequence[1..].to_vec(),
            weight: vec![1.0; length],
            groups: vec![None; length],
        }
    }).collect()
}

fn fixture_config(c: &Config, mixer: &str, cache: &str, step7: bool, reads: InBlockReads, retrieval: bool) -> (Config, Arm) {
    let mut cfg = c.clone();
    cfg.task = "lm".into();
    cfg.backend = "cpu".into();
    cfg.fusion = false;
    cfg.autotune = false;
    cfg.d_model = 16;
    cfg.layers = 1;
    cfg.heads = 4;
    cfg.key_dim = 32;
    cfg.value_dim = 64;
    cfg.chunk_size = 16;
    cfg.cache_capacity = 4;
    cfg.scan_backend = delta_mixers::ScanBackend::TensorSolver;
    cfg.hola_backend = delta_mixers::HolaBackend::Tensor;
    cfg.prefix_block_retrieval_backend = delta_mixers::prefix_retrieval::PrefixRetrievalBackend::Tensor;
    cfg.prefix_block_retrieval = if retrieval {
        c.prefix_block_retrieval.clone().map(|mut retrieval| { retrieval.layers = vec![0]; retrieval })
    } else { None };
    cfg.projection_backend = delta_mixers::ProjectionBackend::Separate;
    cfg.normalization_backend = delta_mixers::NormalizationBackend::Tensor;
    cfg.convolution_backend = delta_mixers::ConvolutionBackend::Tensor;
    cfg.head_backend = crate::config::HeadBackend::Native;
    cfg.loss_backend = crate::config::LossBackend::Tensor;
    cfg.update_backend = crate::config::UpdateBackend::Tensor;
    cfg.epilogue_backend = crate::config::EpilogueBackend::Native;
    cfg.ohc_streams = step7;
    cfg.ohc_backend = delta_mixers::ohc::OhcBackend::Tensor;
    cfg.wavelet_layers = if step7 { vec![0] } else { vec![] };
    cfg.wavelet_backend = delta_mixers::WaveletBackend::Tensor;
    cfg.lngram_layers = if step7 { vec![0] } else { vec![] };
    cfg.lngram_backend = delta_mixers::lngram::LngramBackend::Tensor;
    cfg.lngram_routes = cfg.d_model;
    cfg.lngram_bits = 1;
    cfg.lngram_memory_dim = 2;
    cfg.lngram_orders = vec![2, 3];
    cfg.soft_selection_backend = crate::config::SoftSelectionBackend::Cpu;
    cfg.sbd = Some(SbdConfig { block_lengths: vec![1, 4, 8, 16], in_block_reads: reads, ..SbdConfig::default() });
    let arm = Arm { name: format!("{mixer}-{cache}"), mixer: mixer.into(), cache: cache.into() };
    cfg.arms = vec![arm.clone()];
    (cfg, arm)
}

fn requested_backends(model: &mut LanguageModel, cfg: &mut Config, c: &Config, reads: InBlockReads) {
    cfg.backend = c.backend.clone();
    cfg.scan_backend = delta_mixers::ScanBackend::FusedGpu;
    cfg.hola_backend = delta_mixers::HolaBackend::FusedGpu;
    cfg.sbd.as_mut().unwrap().noisy_read_backend = if reads.is_none() { NoisyReadBackend::Tensor } else { NoisyReadBackend::FusedGpu };
    model.set_scan_backend(cfg.scan_backend);
    model.set_hola_backend(cfg.hola_backend);
    model.set_noisy_read_backend(cfg.sbd.as_ref().unwrap().noisy_read_backend);
    if let Some(retrieval) = &cfg.prefix_block_retrieval {
        cfg.prefix_block_retrieval_backend = c.prefix_block_retrieval_backend;
        model.blocks[0].mixer.set_prefix_retrieval(delta_mixers::prefix_retrieval::PrefixReadConfig {
            block_size: retrieval.block_size,
            blocks_per_query: retrieval.blocks_per_query,
            backend: c.prefix_block_retrieval_backend,
        });
    }
    cfg.convolution_backend = c.convolution_backend;
    cfg.normalization_backend = c.normalization_backend;
    cfg.ohc_backend = if cfg.ohc_streams { c.ohc_backend } else { delta_mixers::ohc::OhcBackend::Tensor };
    cfg.wavelet_backend = if cfg.wavelet_layers.is_empty() { delta_mixers::WaveletBackend::Tensor } else { c.wavelet_backend };
    cfg.lngram_backend = if cfg.lngram_layers.is_empty() { delta_mixers::lngram::LngramBackend::Tensor } else { c.lngram_backend };
    model.set_convolution_backend(cfg.convolution_backend);
    model.set_normalization_backend(cfg.normalization_backend);
    model.set_ohc_backend(cfg.ohc_backend);
    model.set_wavelet_backend(cfg.wavelet_backend);
    model.set_lngram_backend(cfg.lngram_backend);
}

fn late_boundary(c: &Config) -> usize {
    c.prefix_block_retrieval.as_ref().map_or(19, |retrieval| {
        (retrieval.blocks_per_query + 1) * retrieval.block_size + c.chunk_size + 3
    })
}

fn soft(c: &Config, observed: Vec<Option<usize>>, active: Vec<bool>, device: &Device) -> Result<SoftState> {
    SoftState::new(c.vocab(), c.sbd.as_ref().unwrap().candidate_width, observed, active, device, c.soft_selection_backend)
}

fn active_values(logits: Tensor<3>, lengths: &[usize], vocab: usize) -> Vec<f32> {
    let width = logits.dims()[1];
    let values = values(logits);
    let mut selected = Vec::with_capacity(lengths.iter().sum::<usize>() * vocab);
    for (block, &length) in lengths.iter().enumerate() {
        selected.extend_from_slice(&values[block * width * vocab..(block * width + length) * vocab]);
    }
    selected
}

fn logits_case(model: &LanguageModel, c: &Config, device: &Device) -> Result<Value> {
    let width = 16;
    let length = late_boundary(c) + width + 2;
    let rows = rows(2, length);
    let batch = Batch::new(&rows, device);
    // Repeated rows, unaligned boundaries and a three-token tail share one padded batch.
    let blocks = [(0, 1), (1, 9), (0, 17), (1, length - 2)];
    let lengths = [16, 8, 4, 3];
    let mut trace = CleanInputs::default();
    let _ = model.forward_profiled(&batch, &mut trace);
    let row_starts = (0..rows.len()).map(|row| blocks.iter().filter_map(|&(r, start)| (r == row).then_some(start)).collect::<Vec<_>>()).collect::<Vec<_>>();
    let caches = row_starts.iter().enumerate().map(|(row, starts)| model.training_cache(&trace, row, starts)).collect::<Vec<_>>();
    let mut observed = Vec::with_capacity(blocks.len() * width);
    let mut active = Vec::with_capacity(blocks.len() * width);
    let mut reference = Vec::with_capacity(lengths.iter().sum::<usize>() * c.vocab());
    let mut reference_teacher = Vec::with_capacity(reference.capacity());
    let feedback = c.sbd.as_ref().unwrap().feedback;
    for (index, (&(row, start), &n)) in blocks.iter().zip(&lengths).enumerate() {
        let visible = (0..n).map(|position| (position % 3 == 1).then_some(rows[row].target[start - 1 + position] as usize)).collect::<Vec<_>>();
        let mut state = soft(c, visible.clone(), vec![true; n], device)?;
        let local_index = blocks[..index].iter().filter(|&&(r, _)| r == row).count();
        if feedback == SbdFeedback::SoftFeedback {
            let teacher = model.cached_block(&caches[row], local_index, &state, false).logits.detach();
            reference_teacher.extend(values(teacher.clone()));
            state.update(teacher.reshape([n, c.vocab()]),c.soft_temperature,c.sbd.as_ref().unwrap().candidate_width,model.soft_confidence.as_ref().unwrap())?;
        }
        reference.extend(values(model.cached_block(&caches[row], local_index, &state, feedback == SbdFeedback::HardMask).logits));
        observed.extend(visible);
        observed.extend(std::iter::repeat_n(None, width - n));
        active.extend(std::iter::repeat_n(true, n));
        active.extend(std::iter::repeat_n(false, width - n));
    }
    let cache = model.training_cache_batched(&trace, &blocks);
    let mut state = soft(c, observed, active, device)?;
    let teacher_report = if feedback == SbdFeedback::SoftFeedback {
        let teacher = model.cached_blocks(&cache, &state, false).logits.detach();
        let report = compare(&active_values(teacher.clone(), &lengths, c.vocab()), &reference_teacher).context("batched SBD detached teacher logits")?;
        state.update(teacher.reshape([blocks.len() * width, c.vocab()]),c.soft_temperature,c.sbd.as_ref().unwrap().candidate_width,model.soft_confidence.as_ref().unwrap())?;
        Some(report)
    } else { None };
    let actual = model.cached_blocks(&cache, &state, feedback == SbdFeedback::HardMask).logits;
    let report = compare(&active_values(actual, &lengths, c.vocab()), &reference).context("batched SBD padded versus unpadded logits")?;
    let retrieval_candidates = c.prefix_block_retrieval.as_ref().map(|retrieval| {
        let cut = blocks[3].1 / c.chunk_size * c.chunk_size;
        cut / retrieval.block_size
    });
    if let (Some(candidates), Some(retrieval)) = (retrieval_candidates, &c.prefix_block_retrieval) {
        ensure!(candidates > retrieval.blocks_per_query, "retrieval fixture must rank more full earlier blocks than its selection budget");
    }
    Ok(json!({
        "rows": rows.len(), "clean_length": length, "blocks": blocks,
        "active_lengths": lengths, "padded_length": width,
        "right_padding_positions": lengths.iter().map(|&n| width - n).sum::<usize>(),
        "completed_retrieval_candidates_at_tail": retrieval_candidates,
        "teacher_logits": teacher_report, "logits": report,
    }))
}

fn seed_for(c: &Config, rows: &[Example], predicate: impl Fn(&[TrainingBlock]) -> bool) -> Result<(u64, Vec<TrainingBlock>)> {
    for seed in 0..65_536 {
        let blocks = sample_training_blocks(c, rows, seed)?;
        if predicate(&blocks) { return Ok((seed, blocks)); }
    }
    anyhow::bail!("no deterministic SBD reference draw satisfied its fixture within 65536 seeds")
}

#[derive(Clone, Copy)]
enum CausalExpectation { None, AllObserved, Reconstructed }

fn causal_objective(model: &LanguageModel, c: &Config, rows: &[Example], batch: &Batch, blocks: &[TrainingBlock]) -> Result<(Tensor<1>, Vec<Value>)> {
    let mut loss=model.loss(model.forward(batch),batch);
    let device=batch.input.device();
    let mut reports=Vec::new();
    for block in blocks {
        let row=&rows[block.row];
        let n=block.masked.len();
        let observed=(0..n).map(|i|(!block.masked[i]).then_some(row.target[block.start+i] as usize)).collect();
        let mut state=soft(c,observed,vec![true;n],&device)?;
        if c.sbd.as_ref().unwrap().feedback==SbdFeedback::SoftFeedback {
            let teacher=model.valid();
            let output=crate::sbd::block_forward(&teacher,&row.input[..block.start+1],&state,false,&device);
            crate::sbd::update_feedback(model,c,output,&mut state)?;
        }
        let output=crate::sbd::block_forward(model,&row.input[..block.start+1],&state,c.sbd.as_ref().unwrap().feedback==SbdFeedback::HardMask,&device);
        loss=loss+crate::sbd::masked_training_loss(output.logits,block,row,rows.len(),c.vocab(),&device);
        reports.push(json!({"row":block.row,"start":block.start,"length":n,"noisy_input_consumed":true}));
    }
    Ok((loss,reports))
}

fn loss_case(model: &LanguageModel, c: &Config, rows: &[Example], seed: u64, blocks: &[TrainingBlock], expectation: CausalExpectation, device: &Device) -> Result<Value> {
    let batch = Batch::new(rows, device);
    let mut per_block = c.clone();
    per_block.sbd.as_mut().unwrap().block_execution = SbdBlockExecution::PerBlock;
    let mut batched = per_block.clone();
    batched.sbd.as_mut().unwrap().block_execution = SbdBlockExecution::Batched;
    let noisy = !blocks.is_empty();
    let reference_loss = crate::sbd::training_loss(model, &per_block, rows, &batch, seed)?.loss;
    let reference_value = values(reference_loss.clone());
    let reference_gradients = gradient_snapshot(model, reference_loss, noisy)?;
    let actual_loss = crate::sbd::training_loss(model, &batched, rows, &batch, seed)?.loss;
    let actual_value = values(actual_loss.clone());
    let actual_gradients = gradient_snapshot(model, actual_loss, noisy)?;
    let loss_report = compare(&actual_value, &reference_value).context("batched SBD total training loss")?;
    let gradient_report = compare_gradients(&actual_gradients, &reference_gradients)?;
    let causal = match expectation {
        CausalExpectation::None => None,
        CausalExpectation::AllObserved | CausalExpectation::Reconstructed => {
            if matches!(expectation, CausalExpectation::AllObserved) {
                ensure!(blocks.is_empty(), "all-observed fixture unexpectedly selected a masked block");
            }
            let (expected_loss, logits_reports) = causal_objective(model, c, rows, &batch, blocks)?;
            let expected_value = values(expected_loss.clone());
            let expected_gradients = gradient_snapshot(model, expected_loss, noisy)?;
            let bit_identical = reference_value.iter().zip(&actual_value).zip(&expected_value).all(|((&a, &b), &e)| a.to_bits() == b.to_bits() && a.to_bits() == e.to_bits());
            if matches!(expectation, CausalExpectation::AllObserved) {
                ensure!(bit_identical, "all-observed loss must be exactly clean NTP in both execution modes");
            }
            Some(json!({
                "objective": if matches!(expectation, CausalExpectation::AllObserved) { "clean_ntp" } else { "clean_ntp_plus_reconstructed_full_noisy_blocks" },
                "loss_bit_identical": bit_identical,
                "noisy_blocks": logits_reports,
                "per_block_loss": compare(&reference_value, &expected_value)?,
                "batched_loss": compare(&actual_value, &expected_value)?,
                "per_block_all_parameter_gradients": compare_gradients(&reference_gradients, &expected_gradients)?,
                "batched_all_parameter_gradients": compare_gradients(&actual_gradients, &expected_gradients)?,
            }))
        }
    };
    Ok(json!({
        "seed": seed, "rows": rows.len(), "targets_per_row": rows[0].target.len(),
        "configured_training_block_lengths": c.sbd.as_ref().unwrap().training_block_lengths,
        "draw": blocks.iter().map(|block| json!({"row":block.row,"start":block.start,"configured_length":block.length,"masked":block.masked})).collect::<Vec<_>>(),
        "per_block_loss": reference_value[0], "batched_loss": actual_value[0],
        "total_training_loss": loss_report, "all_parameter_gradients": gradient_report,
        "causal_expectation": causal,
    }))
}

fn training_cases(model: &LanguageModel, c: &Config, device: &Device) -> Result<Vec<Value>> {
    let mut reports=Vec::new();
    let mut cfg=c.clone();
    cfg.soft_training_probability=1.0;
    for length in [1usize,4,8,16] {
        cfg.sbd.as_mut().unwrap().training_block_lengths=vec![length];
        let samples=rows(2,19);
        let (seed,blocks)=seed_for(&cfg,&samples,|blocks|[0,1].iter().all(|&r|blocks.iter().any(|b|b.row==r)))?;
        ensure!(blocks.iter().all(|b|b.length==length && b.masked.len()==length && b.start+length<=19/length*length),"partial noisy training target entered full-block objective");
        reports.push(loss_case(model,&cfg,&samples,seed,&blocks,CausalExpectation::Reconstructed,device)?);
    }
    cfg.sbd.as_mut().unwrap().training_block_lengths=vec![4,8];
    let observed=rows(2,9);
    let (seed,blocks)=seed_for(&cfg,&observed,|blocks|blocks.is_empty())?;
    reports.push(loss_case(model,&cfg,&observed,seed,&blocks,CausalExpectation::AllObserved,device)?);
    let variable=rows(4,35);
    let mut drawn=std::collections::BTreeSet::new();
    for seed in 0..128 {
        let blocks=sample_training_blocks(&cfg,&variable,seed)?;
        if let Some(first)=blocks.first() {
            ensure!(blocks.iter().all(|b|b.length==first.length && b.masked.len()==first.length),"training update drew different row lengths or a partial tail");
            drawn.insert(first.length);
        }
    }
    ensure!(drawn==[4usize,8].into_iter().collect(),"variable training did not exercise configured lengths");
    reports.push(json!({"shared_update_length":true,"complete_blocks_only":true,"clean_tail_loss_retained":true,"drawn_lengths":drawn}));
    Ok(reports)
}

pub fn run(c: &Config, device: &Device) -> Result<Value> {
    let cpu = Device::ndarray().autodiff();
    let requested = device.clone().autodiff();
    let mut reports = Vec::new();
    for mixer in ["kronlstm"] {
        for cache in ["none", "surprise"] {
            for step7 in [false, true] {
                for reads in [InBlockReads::None, InBlockReads::HolaRead] {
                    if cache == "none" && !reads.is_none() { continue; }
                    for retrieval in [false, true] {
                        if retrieval && (cache == "none" || c.prefix_block_retrieval.is_none()) { continue; }
                        let (cfg, arm) = fixture_config(c, mixer, cache, step7, reads, retrieval);
                        cfg.validate()?;
                        cpu.seed(72);
                        let base = LanguageModel::new(&cfg, &arm, &cpu)?;
                        for requested_device in [false, true] {
                            if requested_device && c.backend == "cpu" { continue; }
                            let mut cfg = cfg.clone();
                            let mut model = if requested_device { base.clone().fork(&requested) } else { base.clone() };
                            let target = if requested_device { &requested } else { &cpu };
                            if requested_device { requested_backends(&mut model, &mut cfg, c, reads); }
                            cfg.validate()?;
                            for feedback in [SbdFeedback::SoftFeedback, SbdFeedback::HardMask] {
                                cfg.sbd.as_mut().unwrap().feedback = feedback;
                                let context = || format!("batching reference {mixer}/{cache}, step7={step7}, reads={reads:?}, retrieval={retrieval}, feedback={feedback:?}, device={}", cfg.backend);
                                let logits = logits_case(&model, &cfg, target).with_context(context)?;
                                let training = training_cases(&model, &cfg, target).with_context(context)?;
                                reports.push(json!({
                                    "mixer": mixer, "cache": cache, "step7": step7,
                                    "in_block_reads": reads, "feedback": feedback,
                                    "backend": cfg.backend, "device": crate::runtime::identity(target),
                                    "retrieval": cfg.prefix_block_retrieval,
                                    "fixture_shape": {"d_model":cfg.d_model,"layers":cfg.layers,"heads":cfg.heads,"key_dim":cfg.key_dim,"value_dim":cfg.value_dim,"chunk_size":cfg.chunk_size,"cache_capacity":cfg.cache_capacity},
                                    "logits": logits, "training": training,
                                }));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(json!({
        "passed": true,
        "absolute_tolerance": 2e-5, "relative_tolerance": 3e-4,
        "parameter_seed": 72,
        "requested_backend": c.backend,
        "requested_device_is_cpu": c.backend == "cpu",
        "retrieval_supplied": c.prefix_block_retrieval.is_some(),
        "cases": reports,
        "reference_scope": "Identical parameters and pre-change random draws; per-block versus block-axis batched execution independently on CPU and requested device. Both feedback modes, all valid mixer/HOLA/Step7/in-block-read combinations, padded mixed-length logits and detached teacher logits, complete training loss, and every element of every parameter gradient use the unchanged fixed tolerance. All-observed loss is bit-identical to clean NTP; length-one loss and gradients are independently compared with their causal clean-logit objective. Optional supplied retrieval is additionally checked with more completed earlier blocks than its selection budget; retrieval-absent cases always run.",
    }))
}
