//! One global neural gradient clip, paper AdamW groups, and a restartable schedule.
use anyhow::{Context, Result, ensure};
use burn::{
    module::{Module, ModuleVisitor, Param, ParamId},
    optim::{AdamWState, GradientsParams, ModuleOptimizer, Optimizer},
    prelude::*,
};
use std::{collections::HashSet, path::Path};

pub const NO_DECAY_PARAMETERS: &str = r"(^|\.)([^.]*norm)\.[^.]+$|(^|\.)(bias|[^.]*_bias|A_log|a_log|b_delta|decay_scale|decay_bias|gate)$";
pub const LNGRAM_TABLES: &str = r"(^|\.)lngram\.tables\.[0-9]+\.weight$";
pub const SOFT_CONFIDENCE: &str = r"(^|\.)soft_confidence\.(raw_scale|raw_centre_neg|raw_steep)$";
const RECIPE_KEY: &str = "neural_optimizer_recipe";
const STEP_KEY: &str = "neural_optimizer_completed_steps";

#[derive(Clone)]
pub struct AdamW {
    pub fused: bool,
    pub decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub lr_multiplier: f64,
}
impl AdamW {
    pub fn new(c: &crate::config::Config, decay: f32, lr_multiplier: f64) -> Self {
        Self { fused: !c.update_backend.is_tensor(), decay, beta1: c.optimizer_beta1,
            beta2: c.optimizer_beta2, epsilon: c.optimizer_epsilon, lr_multiplier }
    }
}
impl Optimizer for AdamW {
    type State<const D: usize> = AdamWState<D>;
    fn step<const D: usize>(&self, lr: f64, tensor: Tensor<D>, grad: Tensor<D>, state: Option<Self::State<D>>)
        -> (Tensor<D>, Option<Self::State<D>>) {
        let lr = lr*self.lr_multiplier;
        if !self.fused {
            return burn::optim::AdamWConfig::new().with_weight_decay(self.decay)
                .with_beta_1(self.beta1).with_beta_2(self.beta2).with_epsilon(self.epsilon)
                .build().step(lr, tensor, grad, state);
        }
        #[cfg(any(feature = "vulkan", feature = "cuda"))]
        {
            let (time, m, v, maximum) = match state {
                Some(s) => (s.momentum.time+1, Some(s.momentum.moment_1),
                    Some(s.momentum.moment_2), s.momentum.max_moment_2),
                None => (1, None, None, None),
            };
            let (tensor, moment_1, moment_2) = delta_kernels::adamw_update(
                tensor, grad, m, v, time, lr, self.decay, self.beta1, self.beta2, self.epsilon);
            (tensor, Some(AdamWState { momentum: burn::optim::AdaptiveMomentumState {
                time, moment_1, moment_2, max_moment_2: maximum,
            }}))
        }
        #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
        { let _ = (lr, tensor, grad, state); panic!("fused AdamW requires a GPU feature") }
    }
    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device) -> Self::State<D> {
        state.momentum = state.momentum.to_device(device);
        state
    }
}

/// Only present, trainable neural gradients participate. Frozen circuit probability state
/// is not a neural parameter; its separate flow optimizer is unchanged.
pub fn global_clip<M: Module>(model: &M, gradients: &mut GradientsParams, threshold: f32, fused: bool) -> Result<f64> {
    struct Norms<'a> { gradients: &'a GradientsParams, seen: HashSet<ParamId>, values: Vec<Tensor<1>>, fused: bool }
    impl ModuleVisitor for Norms<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            if !param.val().is_require_grad() || !self.seen.insert(param.id) { return; }
            if let Some(g) = self.gradients.get::<D>(param.id) {
                let count: usize = g.dims().iter().product();
                self.values.push(if self.fused { square_sum(g) }
                    else { g.square().reshape([count]).sum_dim(0) });
            }
        }
    }
    let mut norms = Norms { gradients, seen: HashSet::new(), values: Vec::new(), fused };
    model.visit(&mut norms);
    let norm = if norms.values.is_empty() { 0.0 } else {
        let values = Tensor::cat(norms.values, 0).into_data().try_to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("reading global gradient norm: {e:?}"))?;
        ensure!(values.iter().all(|v| v.is_finite() && *v >= 0.0), "nonfinite/overflow neural gradients before global clipping");
        values.into_iter().map(f64::from).sum::<f64>().sqrt()
    };
    ensure!(threshold.is_finite() && threshold > 0.0, "invalid global clipping threshold");
    let coefficient = (f64::from(threshold)/norm).min(1.0);
    if coefficient < 1.0 {
        struct Scale<'a> { gradients: &'a mut GradientsParams, seen: HashSet<ParamId>, coefficient: f64 }
        impl ModuleVisitor for Scale<'_> {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                if !param.val().is_require_grad() || !self.seen.insert(param.id) { return; }
                if let Some(g) = self.gradients.remove::<D>(param.id) {
                    self.gradients.register(param.id, g*self.coefficient);
                }
            }
        }
        model.visit(&mut Scale { gradients, seen: HashSet::new(), coefficient });
    }
    Ok(norm)
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct Recipe {
    revision: String,
    beta1: f32, beta2: f32, epsilon: f32, decay: f32, clip: f32,
    peak: f64, steps: usize, warmup_fraction: f64, minimum_ratio: f64,
    soft_peak: Option<f64>, soft_warmup_steps: usize,
}
/// Update indices are one-based: the last warmup update reaches peak, the last
/// budgeted update reaches the floor. Fractional warmup lengths are rounded up.
pub fn learning_rate_factor(completed: usize, steps: usize, warmup_fraction: f64, minimum: f64) -> f64 {
    let steps = steps.max(1);
    let warmup = ((steps as f64*warmup_fraction).ceil() as usize).min(steps.saturating_sub(1));
    let update = completed.saturating_add(1).min(steps);
    if warmup > 0 && update <= warmup { return update as f64/warmup as f64; }
    let progress = (update-warmup) as f64/(steps-warmup) as f64;
    minimum+(1.0-minimum)*0.5*(1.0+(std::f64::consts::PI*progress).cos())
}

pub fn soft_confidence_rate(completed: usize, peak: f64, warmup_steps: usize) -> f64 {
    peak*(completed.saturating_add(1) as f64/warmup_steps.max(1) as f64).min(1.0)
}

/// Module-level boundary: all callers get exactly one global clip and one schedule
/// advance, even when a draw legitimately omits a subset of neural gradients.
#[derive(Clone)]
pub struct NeuralOptimizer {
    inner: ModuleOptimizer,
    recipe: Recipe,
    completed_steps: usize,
    fused: bool,
    soft_group: Option<burn::module::ParamGroup>,
}
impl NeuralOptimizer {
    pub fn new(c: &crate::config::Config, inner: ModuleOptimizer) -> Self {
        let soft_group = c.sbd.as_ref().filter(|s| s.feedback == crate::config::SbdFeedback::SoftFeedback)
            .map(|_|burn::module::ParamGroup::from_regex(SOFT_CONFIDENCE).expect("soft confidence parameter paths"));
        let inner = if let Some(group) = &soft_group {
            inner.with_group(group.clone(), AdamW::new(c, 0.0, 1.0), None)
        } else {inner};
        Self { inner, recipe: Recipe { revision: "kron-global-adamw-soft-head-v1".into(),
            beta1: c.optimizer_beta1, beta2: c.optimizer_beta2, epsilon: c.optimizer_epsilon,
            decay: c.weight_decay, clip: c.gradient_clip, peak: c.learning_rate,
            steps: c.resolved_tokens().expect("resolved optimizer token budget")/c.tokens_per_step(),
            warmup_fraction: c.warmup_fraction, minimum_ratio: c.min_learning_rate_ratio,
            soft_peak: soft_group.as_ref().map(|_|c.soft_confidence_learning_rate),
            soft_warmup_steps: c.soft_confidence_warmup_steps },
            completed_steps: 0, fused: !c.update_backend.is_tensor(), soft_group }
    }
    pub fn completed_steps(&self) -> usize { self.completed_steps }
    pub fn schedule_steps(&self) -> usize { self.recipe.steps }
    pub fn step<M: Module>(&mut self, peak: f64, model: M, mut gradients: GradientsParams) -> M {
        global_clip(&model, &mut gradients, self.recipe.clip, self.fused)
            .expect("finite neural gradients required before optimizer update");
        let lr = peak*learning_rate_factor(self.completed_steps, self.recipe.steps,
            self.recipe.warmup_fraction, self.recipe.minimum_ratio);
        let model = if let Some(group) = &self.soft_group {
            // Burn's public policy builder owns its group list. Keep rates absolute:
            // the head schedule is not a multiplier of the decaying backbone rate.
            let head_rate = soft_confidence_rate(self.completed_steps,
                self.recipe.soft_peak.expect("soft confidence recipe"), self.recipe.soft_warmup_steps);
            let policy = burn::lr_scheduler::module_lr_scheduler::ModuleLrScheduler::new(lr)
                .with_group(group.clone(), head_rate).step();
            self.inner.step(policy, model, gradients)
        } else { self.inner.step(lr, model, gradients) };
        self.completed_steps += 1;
        model
    }
    fn writer(&self) -> Result<burn_pack::Writer> {
        let reader = burn_pack::Reader::from_bytes(self.inner.into_bytes()?)?;
        let scalars = reader.scalars().clone();
        let metadata = reader.metadata().clone();
        let mut writer = burn_pack::Writer::new(reader.into_tensors()?);
        for (key, value) in scalars { writer = writer.with_scalar(&key, value); }
        for (key, value) in metadata { writer = writer.with_metadata(&key, &value); }
        Ok(writer.with_metadata(RECIPE_KEY, &serde_json::to_string(&self.recipe)?)
            .with_metadata(STEP_KEY, &self.completed_steps.to_string()))
    }
    pub fn into_bytes(&self) -> Result<burn::tensor::Bytes> { Ok(self.writer()?.into_bytes()?) }
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> { Ok(self.writer()?.write_to_file(path)?) }
    pub fn load(self, path: impl AsRef<Path>) -> Result<Self> {
        self.load_recipe(path.as_ref(), false)
    }
    /// Import the original schedule horizon together with its moments and logical clock.
    pub fn load_for_continuation(self, path: impl AsRef<Path>) -> Result<Self> {
        self.load_recipe(path.as_ref(), true)
    }
    fn load_recipe(mut self, path: &Path, source_horizon: bool) -> Result<Self> {
        let reader = burn_pack::Reader::from_file(path)?;
        let recipe: Recipe = serde_json::from_str(reader.metadata().get(RECIPE_KEY)
            .context("optimizer predates global clipping/Kron recipe; restart from scratch")?)?;
        if source_horizon { self.recipe.steps = recipe.steps; }
        ensure!(recipe == self.recipe, "optimizer recipe/schedule mismatch; restart from scratch");
        self.completed_steps = reader.metadata().get(STEP_KEY)
            .context("optimizer is missing global schedule counter")?.parse()?;
        self.recipe = recipe;
        self.inner = self.inner.load(path)?;
        Ok(self)
    }
}

pub fn square_sum<const D: usize>(gradient: Tensor<D>) -> Tensor<1> {
    #[cfg(any(feature = "vulkan", feature = "cuda"))]
    { delta_kernels::gradient_square_sum(gradient) }
    #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
    { let _ = gradient; panic!("fused gradient reduction requires a GPU feature") }
}

pub fn clipping_checks(device: &Device) -> Result<serde_json::Value> {
    use burn::nn::LinearConfig;
    let model = LinearConfig::new(1, 1).init(&device.clone().autodiff());
    let bias = model.bias.as_ref().unwrap();
    let mut gradients = GradientsParams::new();
    gradients.register(model.weight.id, Tensor::<2>::from_data([[0.8f32]], device));
    gradients.register(bias.id, Tensor::<1>::from_data([0.8f32], device));
    let norm = global_clip(&model, &mut gradients, 1.0, false)?;
    let a = gradients.get::<2>(model.weight.id).unwrap().into_scalar::<f32>();
    let b = gradients.get::<1>(bias.id).unwrap().into_scalar::<f32>();
    let expected = 1.0f64/2.0f64.sqrt();
    ensure!((f64::from(a)-expected).abs() < 1e-7 && (f64::from(b)-expected).abs() < 1e-7,
        "two .8 tensors must share the global coefficient");
    ensure!(a.to_bits() == b.to_bits(), "global clip changed relative gradient direction");
    gradients.register(model.weight.id, Tensor::<2>::from_data([[4.0f32]], device));
    gradients.register(bias.id, Tensor::<1>::from_data([2.0f32], device));
    global_clip(&model, &mut gradients, 1.0, false)?;
    let weight = gradients.get::<2>(model.weight.id).unwrap().into_scalar::<f32>();
    let bias = gradients.get::<1>(bias.id).unwrap().into_scalar::<f32>();
    ensure!((f64::from(weight)-4.0/20.0f64.sqrt()).abs() < 1e-7
        && (f64::from(bias)-2.0/20.0f64.sqrt()).abs() < 1e-7,
        "unequal tensor gradients did not retain their direction");
    Ok(serde_json::json!({"passed":true,"two_tensor_norm":norm,"clipped":[a,b],"global_direction_preserved":true}))
}

/// Numeric group check with the production constructor and actual parameter paths.
pub fn optimizer_group_checks(c: &crate::config::Config, device: &Device) -> Result<serde_json::Value> {
    use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};
    // Make the no-decay and 5x-rate effects much larger than the unchanged oracle tolerance.
    let mut fixture = c.clone();
    fixture.learning_rate = 0.05;
    fixture.weight_decay = 0.1;
    fixture.warmup_fraction = 0.0;
    fixture.min_learning_rate_ratio = 1.0;
    fixture.sbd = Some(crate::config::SbdConfig::default());
    fixture.soft_confidence_learning_rate = 0.03;
    fixture.soft_confidence_warmup_steps = 0;
    let c = &fixture;
    #[derive(Module, Debug)]
    struct Table { weight: Param<Tensor<2>> }
    #[derive(Module, Debug)]
    struct Tables { tables: Vec<Table> }
    #[derive(Module, Debug)]
    struct SoftConfidence {
        raw_scale: Param<Tensor<1>>, raw_centre_neg: Param<Tensor<1>>, raw_steep: Param<Tensor<1>>,
    }
    #[derive(Module, Debug)]
    struct Parameters {
        norm: RmsNorm, erase: Linear, write: Linear, output_gate: Linear,
        decay_scale: Param<Tensor<1>>, decay_bias: Param<Tensor<1>>,
        a_log: Param<Tensor<1>>, b_delta: Param<Tensor<1>>, gate_bias: Param<Tensor<1>>,
        lngram: Tables, frozen: Param<Tensor<1>>,
        soft_confidence: SoftConfidence,
    }
    let device = device.clone().autodiff();
    let one = || Param::from_tensor(Tensor::<1>::ones([1], &device));
    let linear = || LinearConfig::new(1, 1).init(&device);
    let parameters = Parameters { norm: RmsNormConfig::new(1).init(&device),
        erase: linear(), write: linear(), output_gate: linear(),
        decay_scale: one(), decay_bias: one(), a_log: one(), b_delta: one(), gate_bias: one(),
        lngram: Tables { tables: vec![Table { weight: Param::from_tensor(Tensor::<2>::ones([1,1], &device)) }] },
        soft_confidence: SoftConfidence { raw_scale: one(), raw_centre_neg: one(), raw_steep: one() },
        frozen: Param::from_tensor(Tensor::<1>::ones([1], &device).set_require_grad(false)) };
    #[derive(Default)]
    struct Values { path: Vec<String>, values: std::collections::BTreeMap<String, Vec<f32>> }
    impl ModuleVisitor for Values {
        fn enter_module(&mut self, name: &str, _: &str) { self.path.push(name.to_owned()); }
        fn exit_module(&mut self, _: &str, _: &str) { self.path.pop(); }
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            self.values.insert(self.path.join("."), p.val().into_data().try_to_vec::<f32>().unwrap());
        }
    }
    struct Gradients { values: GradientsParams }
    impl ModuleVisitor for Gradients {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            if p.val().is_require_grad() {
                self.values.register(p.id, Tensor::<D>::ones(p.val().dims(), &p.val().device()));
            }
        }
    }
    let mut before = Values::default(); parameters.visit(&mut before);
    let mut gradients = Gradients { values: GradientsParams::new() }; parameters.visit(&mut gradients);
    let count = gradients.values.len();
    let g = (f64::from(c.gradient_clip)/(count as f64).sqrt()).min(1.0);
    let mut optimizer = crate::train::optimizer(c);
    let lr = c.learning_rate*learning_rate_factor(0, optimizer.recipe.steps, c.warmup_fraction, c.min_learning_rate_ratio);
    let updated = optimizer.step(c.learning_rate, parameters, gradients.values);
    let mut after = Values::default(); updated.visit(&mut after);
    let mut groups = serde_json::Map::new();
    let mut maximum = 0.0f64;
    for (name, values) in &before.values {
        let table = name == "lngram.tables.0.weight";
        let frozen = name == "frozen";
        // Explicit fixture classification, independent of the optimizer's regex.
        let decayed = matches!(name.as_str(), "erase.weight" | "write.weight" | "output_gate.weight");
        let head = name.starts_with("soft_confidence.");
        let rate = if head {soft_confidence_rate(0,c.soft_confidence_learning_rate,c.soft_confidence_warmup_steps)}
            else {lr*if table {5.0} else {1.0}};
        let m = g*f64::from(1.0-c.optimizer_beta1);
        let v = g*g*f64::from(1.0-c.optimizer_beta2);
        let delta = (m/f64::from(1.0-c.optimizer_beta1))
            / ((v/f64::from(1.0-c.optimizer_beta2)).sqrt()+f64::from(c.optimizer_epsilon));
        for (&a, &b) in values.iter().zip(&after.values[name]) {
            let expected = if frozen {f64::from(a)} else {
                f64::from(a)*(if decayed {1.0-rate*f64::from(c.weight_decay)} else {1.0})-rate*delta
            };
            let error = (f64::from(b)-expected).abs();
            ensure!(error <= 2e-5+3e-4*expected.abs(), "optimizer group {name}: {b} vs {expected}");
            maximum = maximum.max(error);
        }
        groups.insert(name.clone(), serde_json::json!({"decay":decayed,"lr_multiplier":if table {5}else{1},"frozen":frozen}));
    }
    let reader = burn_pack::Reader::from_bytes(optimizer.into_bytes()?)?;
    let mut checked_moments = 0;
    for tensor in reader.into_tensors()? {
        let (name, dtype, shape, _, bytes) = tensor.into_parts()?;
        let expected = if name.ends_with("moment_1") {g*f64::from(1.0-c.optimizer_beta1)}
            else if name.ends_with("moment_2") {g*g*f64::from(1.0-c.optimizer_beta2)}
            else {continue};
        let values = TensorData::from_bytes(bytes, shape, dtype).try_to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("reading optimizer moment: {e:?}"))?;
        ensure!(values.iter().all(|v| (f64::from(*v)-expected).abs() <= 2e-5+3e-4*expected.abs()),
            "global clipping moment mismatch: {name}");
        checked_moments += 1;
    }
    ensure!(checked_moments == count*2, "missing optimizer moment checks");
    Ok(serde_json::json!({"passed":true,"groups":groups,"global_gradient_norm":(count as f64).sqrt(),
        "global_coefficient":g,"checked_moments":checked_moments,"maximum_parameter_error":maximum}))
}

/// An absent head must keep its parameters and moments untouched while the global
/// schedule advances. The first later soft draw uses global LR time but Adam time one.
pub fn soft_head_checks(c: &crate::config::Config, device: &Device) -> Result<serde_json::Value> {
    #[derive(Module, Debug)]
    struct Head { raw_scale: Param<Tensor<1>>, raw_centre_neg: Param<Tensor<1>>, raw_steep: Param<Tensor<1>> }
    #[derive(Module, Debug)]
    struct Parameters { body: Param<Tensor<1>>, soft_confidence: Head }
    let mut cfg = c.clone();
    cfg.sbd = Some(crate::config::SbdConfig::default());
    cfg.budget_file = None;
    cfg.train_tokens = cfg.tokens_per_step()*20;
    cfg.soft_confidence_learning_rate = 0.01;
    cfg.soft_confidence_warmup_steps = 2500;
    let device = device.clone().autodiff();
    let one = || Param::from_tensor(Tensor::<1>::ones([1], &device));
    let mut model = Parameters {body:one(),soft_confidence:Head {raw_scale:one(),raw_centre_neg:one(),raw_steep:one()}};
    let heads = |m: &Parameters| [m.soft_confidence.raw_scale.clone(),m.soft_confidence.raw_centre_neg.clone(),m.soft_confidence.raw_steep.clone()];
    let gradient = |m: &Parameters, soft: bool| {
        let mut g = GradientsParams::new();
        g.register(m.body.id,Tensor::<1>::from_data([0.25f32],&device));
        if soft {
            for (p,value) in heads(m).iter().zip([0.8f32,-0.4,1.2]) {
                g.register(p.id,Tensor::<1>::from_data([value],&device));
            }
        }
        g
    };
    let mut optimizer = crate::train::optimizer(&cfg);
    for _ in 0..8 { let g=gradient(&model,false); model=optimizer.step(cfg.learning_rate,model,g); }
    for p in heads(&model) { ensure!(p.val().into_scalar::<f32>()==1.0,"absent head parameter changed"); }
    let path = std::env::temp_dir().join(format!("soft-head-schedule-{}-{}.burnpack",std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()));
    optimizer.save(&path)?;
    let mut restored_optimizer = crate::checkpoint::load_optimizer(&cfg,&path)?;
    let mut restored = model.clone();
    let mut expected = [1.0f64;3]; let mut first=[0.0f64;3]; let mut second=[0.0f64;3];
    let mut head_updates=0usize;
    let mut maximum=0.0f64;
    for completed in 8..12 {
        let soft=completed%2==0;
        let g=gradient(&model,soft); let h=gradient(&restored,soft);
        model=optimizer.step(cfg.learning_rate,model,g);
        restored=restored_optimizer.step(cfg.learning_rate,restored,h);
        if soft {
            head_updates+=1;
            let raw=[0.8f32,-0.4,1.2];
            let norm=(0.25f64.powi(2)+raw.iter().map(|v|f64::from(*v).powi(2)).sum::<f64>()).sqrt();
            let coefficient=(f64::from(cfg.gradient_clip)/norm).min(1.0);
            // Independent absolute schedule; not the head's sparse appearance counter.
            let lr=0.01*(completed+1) as f64/2500.0;
            for i in 0..3 {
                let g=f64::from(raw[i])*coefficient;
                first[i]=first[i]*f64::from(cfg.optimizer_beta1)+g*f64::from(1.0-cfg.optimizer_beta1);
                second[i]=second[i]*f64::from(cfg.optimizer_beta2)+g*g*f64::from(1.0-cfg.optimizer_beta2);
                let delta=(first[i]/f64::from(1.0-cfg.optimizer_beta1.powi(head_updates as i32)))
                    / ((second[i]/f64::from(1.0-cfg.optimizer_beta2.powi(head_updates as i32))).sqrt()+f64::from(cfg.optimizer_epsilon));
                expected[i]-=lr*delta;
            }
        }
        for (p,want) in heads(&model).iter().zip(expected) {
            let error=(f64::from(p.val().into_scalar::<f32>())-want).abs();
            ensure!(error<=2e-7,"sparse head LR/Adam counter/no-decay mismatch");
            maximum=maximum.max(error);
            let reader=burn_pack::Reader::from_bytes(optimizer.into_bytes()?)?;
            let key=format!("{}.momentum.time",p.id.val());
            ensure!(reader.scalars().get(&key).is_some_and(|v|usize::try_from(*v).ok()==Some(head_updates)),
                "head Adam counter must count present gradients, not global steps");
        }
        ensure!(crate::checkpoint::optimizer_signature(&cfg,optimizer.into_bytes()?)?
            ==crate::checkpoint::optimizer_signature(&cfg,restored_optimizer.into_bytes()?)?,
            "skip-draw to soft-draw restart changed optimizer state");
        for (a,b) in heads(&model).iter().zip(heads(&restored)) {
            ensure!(a.val().into_scalar::<f32>().to_bits()==b.val().into_scalar::<f32>().to_bits(),
                "skip-draw to soft-draw restart changed head parameters");
        }
    }
    std::fs::remove_file(path)?;
    Ok(serde_json::json!({"passed":true,"global_updates":12,"head_updates":head_updates,
        "first_head_rate":0.01*9.0/2500.0,"restart_exact":true,"maximum_parameter_error":maximum}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn global_clip_rescales_two_subthreshold_tensors() {
        clipping_checks(&Device::ndarray()).unwrap();
    }
    #[test]
    fn schedule_hits_warmup_peak_and_final_floor() {
        assert_eq!(learning_rate_factor(0, 100, 0.1, 0.1), 0.1);
        assert_eq!(learning_rate_factor(9, 100, 0.1, 0.1), 1.0);
        assert_eq!(learning_rate_factor(99, 100, 0.1, 0.1), 0.1);
        assert_eq!(learning_rate_factor(100, 100, 0.1, 0.1), 0.1);
    }
    #[test]
    fn config_uses_paper_defaults_and_preserves_explicit_overrides() {
        let defaults: crate::config::Config = serde_json::from_str("{}").unwrap();
        assert_eq!((defaults.optimizer_beta1, defaults.optimizer_beta2, defaults.optimizer_epsilon),
            (0.9, 0.95, 1e-10));
        assert_eq!((defaults.weight_decay, defaults.warmup_fraction, defaults.min_learning_rate_ratio),
            (0.1, 0.1, 0.1));
        assert_eq!((defaults.soft_confidence_learning_rate,defaults.soft_confidence_warmup_steps),(0.01,2500));
        let custom: crate::config::Config = serde_json::from_value(serde_json::json!({
            "optimizer_beta1":0.85,"optimizer_beta2":0.99,"optimizer_epsilon":1e-8,
            "weight_decay":0.02,"learning_rate":0.002,"warmup_fraction":0.2,
            "min_learning_rate_ratio":0.05})).unwrap();
        custom.validate().unwrap();
        assert_eq!((custom.optimizer_beta1,custom.optimizer_beta2,custom.optimizer_epsilon),
            (0.85,0.99,1e-8));
        assert_eq!((custom.weight_decay,custom.learning_rate,custom.warmup_fraction,custom.min_learning_rate_ratio),
            (0.02,0.002,0.2,0.05));
        assert_eq!(soft_confidence_rate(0,0.01,2500),0.01/2500.0);
        assert_eq!(soft_confidence_rate(2499,0.01,2500),0.01);
        assert_eq!(soft_confidence_rate(2500,0.01,2500),0.01);
    }
    #[test]
    fn absent_head_uses_global_schedule_and_local_adam_time_after_restart() {
        soft_head_checks(&crate::config::Config::default(),&Device::ndarray()).unwrap();
    }
    #[test]
    fn global_clip_rejects_nonfinite_or_overflow_without_writing_parameters() {
        let device = Device::ndarray().autodiff();
        let model = burn::nn::LinearConfig::new(1,1).init(&device);
        for value in [f32::NAN,f32::INFINITY,f32::NEG_INFINITY,f32::MAX] {
            let mut gradients = GradientsParams::new();
            gradients.register(model.weight.id,Tensor::<2>::from_data([[value]],&device));
            assert!(global_clip(&model,&mut gradients,1.0,false).is_err());
        }
    }
    #[test]
    fn paper_decay_exclusions_and_lngram_rate_match_numbers() {
        optimizer_group_checks(&crate::config::Config::default(), &Device::ndarray()).unwrap();
    }
    #[test]
    fn schedule_and_updates_resume_exactly_with_an_omitted_gradient() {
        let mut c = crate::config::Config::default();
        c.train_tokens = c.tokens_per_step()*20;
        let device = Device::ndarray().autodiff();
        let mut uninterrupted = burn::nn::LinearConfig::new(1, 1).init(&device);
        let mut optimizer = crate::train::optimizer(&c);
        let gradients = |m: &burn::nn::Linear, step: usize| {
            let mut g = GradientsParams::new();
            g.register(m.weight.id, Tensor::<2>::from_data([[if step%2==0 {0.8f32}else{-1.6}]], &device));
            if step%3 != 0 { g.register(m.bias.as_ref().unwrap().id, Tensor::<1>::from_data([0.8f32], &device)); }
            g
        };
        for step in 0..7 { let g = gradients(&uninterrupted, step); uninterrupted = optimizer.step(c.learning_rate, uninterrupted, g); }
        let path = std::env::temp_dir().join(format!("kron-schedule-{}-{}.burnpack", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        optimizer.save(&path).unwrap();
        let mut resumed_optimizer = crate::checkpoint::load_optimizer(&c, &path).unwrap();
        let mut resumed = uninterrupted.clone();
        assert_eq!(resumed_optimizer.completed_steps(), 7);
        let mut target = c.clone(); target.train_tokens *= 5;
        let mut imported_optimizer = crate::checkpoint::load_optimizer_for_continuation(&target,&path).unwrap();
        let mut imported = uninterrupted.clone();
        assert_eq!(imported_optimizer.schedule_steps(),20);
        for step in 7..25 {
            let a = gradients(&uninterrupted, step); let b = gradients(&resumed, step);
            uninterrupted = optimizer.step(c.learning_rate, uninterrupted, a);
            resumed = resumed_optimizer.step(c.learning_rate, resumed, b);
            let g = gradients(&imported,step);
            imported = imported_optimizer.step(target.learning_rate,imported,g);
            assert_eq!(uninterrupted.weight.val().into_data(),imported.weight.val().into_data());
            assert_eq!(crate::checkpoint::optimizer_signature(&c,optimizer.into_bytes().unwrap()).unwrap(),
                crate::checkpoint::optimizer_signature(&c,imported_optimizer.into_bytes().unwrap()).unwrap());
            assert_eq!(uninterrupted.weight.val().into_data(), resumed.weight.val().into_data());
            assert_eq!(uninterrupted.bias.as_ref().unwrap().val().into_data(), resumed.bias.as_ref().unwrap().val().into_data());
            assert_eq!(crate::checkpoint::optimizer_signature(&c, optimizer.into_bytes().unwrap()).unwrap(),
                crate::checkpoint::optimizer_signature(&c, resumed_optimizer.into_bytes().unwrap()).unwrap());
        }
        let mut changed = c.clone(); changed.optimizer_beta2 = 0.99;
        assert!(crate::train::optimizer(&changed).load(&path).is_err());
        assert!(crate::checkpoint::load_optimizer_for_continuation(&changed,&path).is_err());
        let mut changed = c; changed.train_tokens *= 2;
        assert!(crate::train::optimizer(&changed).load(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
