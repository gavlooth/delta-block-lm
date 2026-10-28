use crate::{
    codpula::CircuitFlows,
    config::Config,
    data::{Batch, Example},
    model::{self, LanguageModel},
    sbd,
};
use anyhow::{Result, ensure};
use burn::optim::{GradientsAccumulator, GradientsParams};
use delta_mixers::profile::Sections;

pub(crate) struct Backward {
    pub loss: f64,
    pub gradients: GradientsParams,
    pub circuit_flows: Option<CircuitFlows>,
    pub passes: usize,
    pub rows_per_pass: usize,
    pub gradient_report: model::GradientReport,
}

pub(crate) fn backward(
    model: &LanguageModel,
    c: &Config,
    rows: &[Example],
    batch: &Batch,
    seed: u64,
) -> Result<Backward> {
    backward_profiled(model, c, rows, batch, seed, &mut delta_mixers::profile::Unprofiled)
}

pub(crate) fn backward_profiled<P: Sections>(
    model: &LanguageModel,
    c: &Config,
    rows: &[Example],
    batch: &Batch,
    seed: u64,
    profile: &mut P,
) -> Result<Backward> {
    ensure!(!rows.is_empty(), "SBD backward requires at least one row");
    let selected_config=sbd::training_config(c,seed);
    let c=&selected_config;
    let configured = c.sbd.as_ref().and_then(|s| s.rows_per_pass);
    let rows_per_pass = configured.unwrap_or(rows.len()).min(rows.len());
    if rows_per_pass == rows.len() {
        let result = sbd::training_loss_profiled(model, c, rows, batch, seed, profile)?;
        let loss = model::scalar(result.loss.clone())?;
        let (gradients, gradient_report) = model::checked_gradients_with_unused(model, result.loss, &result.unused_parameters, profile)?;
        return Ok(Backward {
            loss,
            gradients,
            circuit_flows: result.circuit_flows,
            passes: 1,
            rows_per_pass,
            gradient_report,
        });
    }

    let blocks = sbd::sample_training_blocks(c, rows, seed)?;
    let bundles = c.sbd.as_ref().and_then(|s| s.cocommit.as_ref()).map(|options| {
        crate::cocommit::sample_bundles(&blocks, options.training_bundle_max, seed)
    });
    let mut accumulator = GradientsAccumulator::<LanguageModel>::new();
    let mut circuit_flows = None::<CircuitFlows>;
    let mut loss = 0.0;
    let mut passes = 0;
    let mut unused_parameters: Option<Vec<&str>> = None;

    for row_start in (0..rows.len()).step_by(rows_per_pass) {
        let row_end = (row_start + rows_per_pass).min(rows.len());
        let pass_rows = &rows[row_start..row_end];
        let pass_batch = Batch::new(pass_rows, &batch.input.device());
        let mut pass_blocks = Vec::new();
        let mut pass_bundles = bundles.as_ref().map(|_| Vec::new());
        for (block_index, block) in blocks.iter().enumerate() {
            if (row_start..row_end).contains(&block.row) {
                let mut local = block.clone();
                local.row -= row_start;
                pass_blocks.push(local);
                if let (Some(all), Some(selected)) = (&bundles, &mut pass_bundles) {
                    selected.push(all[block_index].clone());
                }
            }
        }
        let mut pass_config = c.clone();
        pass_config.batch_size = pass_rows.len();
        let pass = sbd::training_loss_sampled_profiled(
            model,
            &pass_config,
            pass_rows,
            &pass_batch,
            &pass_blocks,
            pass_bundles.as_deref(),
            profile,
        )?;
        if let Some(unused) = &mut unused_parameters {
            unused.retain(|name| pass.unused_parameters.contains(name));
        } else {
            unused_parameters = Some(pass.unused_parameters);
        }
        let scale = pass_rows.len() as f64 / rows.len() as f64;
        loss += model::scalar(pass.loss.clone())? * scale;
        accumulator.accumulate(
            model,
            GradientsParams::from_grads((pass.loss * scale).backward(), model),
        );
        if let Some(part) = pass.circuit_flows {
            if let Some(total) = &mut circuit_flows {
                total.add(part);
            } else {
                circuit_flows = Some(part);
            }
        }
        passes += 1;
    }

    let gradients = accumulator.grads();
    let gradient_report = model::check_parameter_gradients_with_unused(model, &gradients, &unused_parameters.unwrap_or_default(), profile)?;
    Ok(Backward {
        loss,
        gradients,
        circuit_flows,
        passes,
        rows_per_pass,
        gradient_report,
    })
}
