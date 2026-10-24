//! Reversible retained answers and one-call keep/remask outcome decisions.
use burn::nn::{Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{DType, activation::softmax};
use crate::learned_output::{LearnedOutputMlp, LEARNED_OUTPUT_WIDTH,
    output_row_logits, probability_entropy};
use crate::pointer::{PointerState, PointerStepOutput, SourcePointerContext};
use crate::joint_circuit::{CIRCUIT_FIELDS, CIRCUIT_FEATURES};

pub const REMASK_FIELDS: usize = 6;
pub const REFINEMENT_ROUNDS: usize = 8;
pub const REMASK_FACTORS: usize = 11;
pub const REMASK_FEATURES: usize = 112 + CIRCUIT_FEATURES;
pub const FACTOR_FEATURES: usize = 90;
pub const RETAINED_FIELDS: usize = 4;
pub const RETAINED_STATS: usize = 72;
const SUMMARY_WIDTH: usize = 6;
const FACTOR_FIELDS:[usize;REMASK_FACTORS]=[0,1,1,2,2,3,3,4,4,5,5];

/// Field order is relation, subject start/end, object start/end, evidence.
/// Relation has one factor; boundaries have token/byte factors; evidence has start/end factors.
#[derive(Debug, Clone)]
pub struct RemaskFrame {
    pub factors: [Tensor<3>; REMASK_FACTORS],
    choices: Tensor<3, Int>,
    none: [Option<usize>; REMASK_FACTORS],
}
impl RemaskFrame {
    /// Current head probabilities and joint circuit proposals, with supplied boundaries fixed exactly.
    /// Byte logits have already been recomputed at the circuit's selected tokens.
    pub fn from_output(output: &PointerStepOutput, observed: &PointerState,
        no_relation: usize, none_token: usize, valid: &Tensor<2>, decoded: &Tensor<3, Int>) -> Self {
        let [b, m, byte_classes] = output.pointers.byte[0].dims();
        let device = output.hidden.device();
        assert_eq!(valid.dims(), [b, m]);
        assert_eq!(decoded.dims(), [b, m, CIRCUIT_FIELDS]);
        let invalid = valid.clone().lower_elem(0.5).unsqueeze_dim(2);
        let normalize = |logits: &Tensor<3>| softmax(
            logits.clone().cast(DType::F32).mask_fill(invalid.clone(), 0.0), 2)
            .mask_fill(invalid.clone(), 0.0);
        let bytes = crate::plan::int((0..byte_classes).map(|offset|offset as i32).collect(),
            [1, 1, byte_classes], &device).expand([b, m, byte_classes]);
        let token: [Tensor<3>; 4] = std::array::from_fn(|field| {
            let fixed = observed.observed.clone().slice([0..b, 0..m, field..field + 1]).greater_elem(0.5);
            normalize(&output.pointers.token[field].clone().mask_fill(fixed.clone(),0.0))
                .mask_where(fixed.expand(observed.token[field].dims()), observed.token[field].clone()).mask_fill(invalid.clone(), 0.0)
        });
        let byte: [Tensor<3>; 4] = std::array::from_fn(|field| {
            let fixed = observed.observed.clone().slice([0..b, 0..m, field..field + 1]).greater_elem(0.5);
            let offsets = observed.observed_offsets.clone().slice([0..b, 0..m, field..field + 1])
                .expand([b, m, byte_classes]);
            let one_hot = bytes.clone().equal(offsets).float();
            normalize(&output.pointers.byte[field].clone().mask_fill(fixed.clone(),0.0))
                .mask_where(fixed.expand([b, m, byte_classes]), one_hot).mask_fill(invalid.clone(), 0.0)
        });
        let choice = |logits: &Tensor<3>| logits.clone().cast(DType::F32).mask_fill(invalid.clone(), 0.0).argmax(2);
        let token_choices: [Tensor<3, Int>; 4] = std::array::from_fn(|field| {
            let fixed = observed.observed.clone().slice([0..b, 0..m, field..field + 1]).greater_elem(0.5);
            decoded.clone().slice([0..b, 0..m, field + 1..field + 2])
                .mask_where(fixed, observed.token[field].clone().mask_fill(invalid.clone(),0.0).argmax(2))
        });
        let byte_choices: [Tensor<3, Int>; 4] = std::array::from_fn(|field| {
            let fixed = observed.observed.clone().slice([0..b, 0..m, field..field + 1]).greater_elem(0.5);
            choice(&output.pointers.byte[field].clone().mask_fill(fixed.clone(),0.0)).mask_where(fixed,
                observed.observed_offsets.clone().slice([0..b, 0..m, field..field + 1]))
        });
        let choices = Tensor::cat(vec![decoded.clone().slice([0..b, 0..m, 0..1]),
            token_choices[0].clone(), byte_choices[0].clone(), token_choices[1].clone(), byte_choices[1].clone(),
            token_choices[2].clone(), byte_choices[2].clone(), token_choices[3].clone(), byte_choices[3].clone(),
            choice(&output.annotations.evidence_start), choice(&output.annotations.evidence_end)], 2)
            .mask_fill(invalid.clone(), 0);
        Self {
            factors: [normalize(&output.annotations.relation),
                token[0].clone(), byte[0].clone(), token[1].clone(), byte[1].clone(),
                token[2].clone(), byte[2].clone(), token[3].clone(), byte[3].clone(),
                normalize(&output.annotations.evidence_start), normalize(&output.annotations.evidence_end)],
            choices,
            none: [Some(no_relation), Some(none_token), None, Some(none_token), None,
                Some(none_token), None, Some(none_token), None, None, None],
        }
    }

    /// One integer transfer can retain every round's exact categorical choices.
    /// Order: R, SS-token/byte, SE-token/byte, OS-token/byte, OE-token/byte, E-start/end.
    pub fn argmaxes(&self) -> Tensor<3, Int> {
        self.choices.clone()
    }


    pub fn detached(&self) -> Self {
        if !self.factors[0].device().is_autodiff() {return self.clone();}
        Self {factors: self.factors.each_ref().map(|factor| factor.clone().inner().autodiff()),
            choices: self.choices.clone(), none: self.none}
    }

    /// Six ordered fields × two factors × six statistics (72), twelve fact summaries,
    /// and six observed-field flags. Relation's absent second factor is exactly zero.
    pub fn features(&self, previous: Option<&Self>, observed: &Tensor<3>) -> Tensor<3> {
        let [b, m, _] = self.factors[0].dims();
        assert_eq!(observed.dims(), [b, m, REMASK_FIELDS]);
        let device = observed.device();
        let changed = previous.map(|previous| self.choices.clone().equal(previous.choices.clone()).bool_not().float())
            .unwrap_or_else(|| Tensor::zeros([b, m, REMASK_FACTORS], &device));
        let supplied=Tensor::<1>::from_floats([1.0,1.0,0.0,0.0,0.0,0.0],&device)
            .reshape([1,1,SUMMARY_WIDTH]).expand([b,m,SUMMARY_WIDTH]);
        let summaries: Vec<_> = self.factors.iter().enumerate().map(|(factor, probability)| {
            let summary=factor_summary(probability, previous.map(|frame| &frame.factors[factor]), self.none[factor],
                changed.clone().slice([0..b, 0..m, factor..factor + 1]));
            let field=FACTOR_FIELDS[factor];
            let fixed=observed.clone().slice([0..b,0..m,field..field+1]).greater_elem(0.5);
            summary.mask_where(fixed.expand([b,m,SUMMARY_WIDTH]),supplied.clone())
        }).collect();
        let mut ordered = Vec::with_capacity(REMASK_FIELDS * 2);
        ordered.push(summaries[0].clone());
        ordered.push(Tensor::zeros([b, m, SUMMARY_WIDTH], &device));
        ordered.extend(summaries[1..].iter().cloned());
        let ordered = Tensor::cat(ordered, 2);
        let all: Tensor<4> = Tensor::cat(summaries.clone(), 2).reshape([b, m, REMASK_FACTORS, SUMMARY_WIDTH]);
        let mean = all.clone().mean_dim(2).reshape([b, m, SUMMARY_WIDTH]);
        let minimum = all.clone().min_dim(2).reshape([b, m, SUMMARY_WIDTH]);
        let maximum = all.max_dim(2).reshape([b, m, SUMMARY_WIDTH]);
        let column = |values: &Tensor<3>, index: usize| values.clone().slice([0..b, 0..m, index..index + 1]);
        let product = summaries.iter().fold(Tensor::ones([b, m, 1], &device),
            |product, values| product * column(values, 0));
        let fact = Tensor::cat(vec![product, column(&minimum, 0), column(&minimum, 1),
            column(&mean, 2), column(&maximum, 2), column(&maximum, 3),
            column(&mean, 3).mul_scalar(REMASK_FACTORS as f64), column(&mean, 4), column(&maximum, 4),
            column(&mean, 5), column(&maximum, 5), observed.clone().mean_dim(2)], 2);
        Tensor::cat(vec![ordered, fact, observed.clone()], 2)
    }
}

fn factor_summary(probability: &Tensor<3>, previous: Option<&Tensor<3>>, none: Option<usize>, changed: Tensor<3>) -> Tensor<3> {
    let [b, m, width] = probability.dims();
    let device = probability.device();
    let (top, margin) = if width == 1 {
        (probability.clone(), probability.clone())
    } else {
        let top = probability.clone().topk(2, 2);
        let first = top.clone().slice([0..b, 0..m, 0..1]);
        (first.clone(), first - top.slice([0..b, 0..m, 1..2]))
    };
    let entropy = probability_entropy(probability.clone().reshape([b * m, width])).reshape([b, m, 1]);
    let none = none.map(|index| {
        assert!(index < width);
        probability.clone().slice([0..b, 0..m, index..index + 1])
    }).unwrap_or_else(|| Tensor::zeros([b, m, 1], &device));
    let distance = match previous {
        Some(previous) => {
            assert_eq!(probability.dims(), previous.dims());
            (probability.clone() - previous.clone()).abs().sum_dim(2)
        }
        None => Tensor::zeros([b, m, 1], &device),
    };
    Tensor::cat(vec![top, margin, entropy, none, distance, changed], 2)
}

/// Only the four supplied boundary fields can be observed in the current production contract.
pub fn observed_fields(pointers: &PointerState) -> Tensor<3> {
    let [b, m, _] = pointers.observed.dims();
    let zero = Tensor::zeros([b, m, 1], &pointers.observed.device());
    Tensor::cat(vec![zero.clone(), pointers.observed.clone(), zero], 2)
}

/// Compact per-call state: source-sized posterior caches are deliberately not copied here.
#[derive(Debug, Clone)]
pub struct RetainedSnapshot {
    pub filled: Tensor<3, Bool>,
    pub value: Tensor<3, Int>,
    pub age: Tensor<3>,
    pub factor_stats: Tensor<3>,
    pub relation_posterior: Tensor<3>,
}
impl RetainedSnapshot {
    pub fn answer(&self)->RetainedAnswer {
        RetainedAnswer {filled:self.filled.clone(),value:self.value.clone(),age:self.age.clone(),
            factor_stats:self.factor_stats.clone(),relation_posterior:self.relation_posterior.clone()}
    }
    pub fn slot_view(&self, row: usize, slots: std::ops::Range<usize>) -> Self {
        Self {filled:self.filled.clone().slice([row..row+1,slots.clone()]),
            value:self.value.clone().slice([row..row+1,slots.clone()]),
            age:self.age.clone().slice([row..row+1,slots.clone()]),
            factor_stats:self.factor_stats.clone().slice([row..row+1,slots.clone()]),
            relation_posterior:self.relation_posterior.clone().slice([row..row+1,slots])}
    }
}

#[derive(Debug, Clone)]
pub struct RetainedAnswer {
    pub filled: Tensor<3, Bool>,
    /// R, SS token/byte, SE token/byte, OS token/byte, OE token/byte, evidence start/end.
    pub value: Tensor<3, Int>,
    pub age: Tensor<3>,
    pub factor_stats: Tensor<3>,
    pub relation_posterior: Tensor<3>,
}
const VALUE_RANGES: [(usize,usize);RETAINED_FIELDS] = [(0,1),(1,5),(5,9),(9,11)];
const STAT_RANGES: [(usize,usize);RETAINED_FIELDS] = [(0,12),(12,36),(36,60),(60,72)];
impl RetainedAnswer {
    pub fn empty(b:usize,m:usize,relations:usize,device:&Device)->Self {
        Self {filled:Tensor::<3,Bool>::from_data(TensorData::new(vec![false;b*m*RETAINED_FIELDS],[b,m,RETAINED_FIELDS]),device),
            value:Tensor::zeros([b,m,REMASK_FACTORS],device),age:Tensor::zeros([b,m,RETAINED_FIELDS],device),
            factor_stats:Tensor::zeros([b,m,RETAINED_STATS],device),relation_posterior:Tensor::zeros([b,m,relations],device)}
    }
    pub fn snapshot(&self)->RetainedSnapshot {
        RetainedSnapshot {filled:self.filled.clone(),value:self.value.clone(),age:self.age.clone(),factor_stats:self.factor_stats.clone(),
            relation_posterior:self.relation_posterior.clone()}
    }
    pub fn slot_view(&self,row:usize,slots:std::ops::Range<usize>)->Self {
        let compact=self.snapshot().slot_view(row,slots.clone());
        Self {filled:compact.filled,value:compact.value,age:compact.age,factor_stats:compact.factor_stats,
            relation_posterior:self.relation_posterior.clone().slice([row..row+1,slots])}
    }
    pub(crate) fn age_one_call(&mut self,horizon:usize) {
        assert!(horizon>0);
        let autodiff=self.age.device().is_autodiff();
        self.age=(self.age.clone().inner()+1.0/horizon as f64)
            .mask_fill(self.filled.clone().inner().bool_not(),0.0);
        if autodiff {self.age=self.age.clone().autodiff();}
    }
    /// Hard writes and caches are constants. In particular no unreachable update graph is recorded.
    pub fn update(&mut self,frame:&RemaskFrame,features:&Tensor<3>,logits:&Tensor<3>,
        eligible:&Tensor<3>)->(Tensor<3>,Tensor<3>) {
        let [b,m,_]=self.filled.dims();
        let autodiff=self.age.device().is_autodiff();
        assert_eq!(logits.dims(),[b,m,RETAINED_FIELDS*2]);
        let choices=frame.argmaxes().inner();
        let old=self.value.clone().inner();
        let mut accepted=Vec::with_capacity(RETAINED_FIELDS);let mut replaced=Vec::with_capacity(RETAINED_FIELDS);
        let mut values=Vec::with_capacity(RETAINED_FIELDS);let mut stats=Vec::with_capacity(RETAINED_FIELDS);
        let mut fills=Vec::with_capacity(RETAINED_FIELDS);let mut ages=Vec::with_capacity(RETAINED_FIELDS);
        let mut relation_written=None;
        for field in 0..RETAINED_FIELDS {
            let filled=self.filled.clone().inner().slice([0..b,0..m,field..field+1]);
            let (first,last)=VALUE_RANGES[field];let (sf,sl)=STAT_RANGES[field];
            let proposal=choices.clone().slice([0..b,0..m,first..last]);
            let previous=old.clone().slice([0..b,0..m,first..last]);
            let differs=proposal.clone().equal(previous.clone()).int().sum_dim(2).lower_elem((last-first) as i64);
            let eligible=eligible.clone().inner().slice([0..b,0..m,field..field+1]).greater_elem(0.5);
            let accept=logits.clone().inner().slice([0..b,0..m,field..field+1]).greater_equal_elem(0.0)
                .bool_and(filled.clone().bool_not()).bool_and(eligible.clone());
            let replace=logits.clone().inner().slice([0..b,0..m,field+RETAINED_FIELDS..field+RETAINED_FIELDS+1])
                .greater_equal_elem(0.0).bool_and(filled.clone()).bool_and(differs).bool_and(eligible);
            let write=accept.clone().bool_or(replace.clone());
            values.push(previous.mask_where(write.clone().expand([b,m,last-first]),proposal));
            stats.push(self.factor_stats.clone().inner().slice([0..b,0..m,sf..sl])
                .mask_where(write.clone().expand([b,m,sl-sf]),features.clone().inner().slice([0..b,0..m,sf..sl])));
            ages.push(self.age.clone().inner().slice([0..b,0..m,field..field+1]).mask_fill(write.clone(),0.0));
            fills.push(filled.bool_or(write.clone()));
            if field==0 {relation_written=Some(write);}
            accepted.push(accept.float());replaced.push(replace.float());
        }
        self.value=Tensor::cat(values,2);self.filled=Tensor::cat(fills,2);
        self.age=Tensor::cat(ages,2);self.factor_stats=Tensor::cat(stats,2);
        self.relation_posterior=self.relation_posterior.clone().inner().mask_where(
            relation_written.expect("relation field").expand(self.relation_posterior.dims()),frame.factors[0].clone().inner());
        let (mut accepted,mut replaced)=(Tensor::cat(accepted,2),Tensor::cat(replaced,2));
        if autodiff {
            self.value=self.value.clone().autodiff();self.filled=self.filled.clone().autodiff();
            self.age=self.age.clone().autodiff();self.factor_stats=self.factor_stats.clone().autodiff();
            self.relation_posterior=self.relation_posterior.clone().autodiff();
            accepted=accepted.autodiff();replaced=replaced.autodiff();
        }
        (accepted,replaced)
    }
    /// Actual pre/post-buffer features. Span support is endpoint-token mean, not a joint probability.
    pub fn features(&self,frame:&RemaskFrame,base:Tensor<3>,visibility:Tensor<3>)->Tensor<3> {
        let [b,m,_]=self.filled.dims();assert_eq!(base.dims(),[b,m,FACTOR_FEATURES]);
        assert_eq!(visibility.dims(),[b,m,REMASK_FIELDS]);
        let mut retained=Vec::with_capacity(RETAINED_FIELDS);
        let argmaxes=frame.argmaxes();
        for field in 0..RETAINED_FIELDS {
            let (first,last)=VALUE_RANGES[field];
            let filled=self.filled.clone().slice([0..b,0..m,field..field+1]);
            let agreement=argmaxes.clone().slice([0..b,0..m,first..last])
                .equal(self.value.clone().slice([0..b,0..m,first..last])).int().sum_dim(2).equal_elem((last-first) as i64).float();
            let probability=|factor:usize|frame.factors[factor].clone().gather(2,
                self.value.clone().slice([0..b,0..m,factor..factor+1]));
            let support=match field {0=>probability(0),1=>(probability(1)+probability(3))*0.5,
                2=>(probability(5)+probability(7))*0.5,_=>(probability(9)+probability(10))*0.5};
            retained.push(Tensor::cat(vec![filled.clone().float(),agreement,
                self.age.clone().slice([0..b,0..m,field..field+1]),support],2)
                .mask_fill(filled.bool_not().expand([b,m,4]),0.0));
        }
        Tensor::cat(vec![base,visibility,Tensor::cat(retained,2)],2)
    }
}

/// Raw source-byte coherence; token numbering alone does not establish span ordering.
pub fn coherent_fields(frame:&RemaskFrame,pointers:&PointerState,context:&SourcePointerContext,
    valid:&Tensor<2>)->Tensor<3> {
    context.coherent_choices(&frame.argmaxes(),&pointers.observed,valid)
}

#[derive(Module, Debug)]
pub struct RefinementDecider {
    pub mlp: LearnedOutputMlp,
    pub acceptance_output: Linear,
    pub outcome_output: Linear,
    pub round_embedding: Embedding,
    hidden: usize,
}
impl RefinementDecider {
    /// Append two independent copies of the last A4 row under its original parameter ID.
    pub fn grow_a4_rounds(mut self)->Self {
        assert_eq!(self.round_embedding.weight.dims()[0],8);
        let value=self.round_embedding.weight.val();
        let last=value.clone().slice([7..8]);
        self.round_embedding.weight=burn::module::Param::initialized(self.round_embedding.weight.id,
            Tensor::cat(vec![value,last.clone(),last],0).detach()).set_require_grad(true);
        self
    }
    pub fn new(hidden:usize,rounds:usize,device:&Device)->Self {
        assert!(hidden>0);assert!(rounds>0,"refinement needs a positive horizon");
        Self {mlp:LearnedOutputMlp::new(hidden+REMASK_FEATURES+LEARNED_OUTPUT_WIDTH,device),
            acceptance_output:LinearConfig::new(LEARNED_OUTPUT_WIDTH,RETAINED_FIELDS*2).init(device),
            outcome_output:LinearConfig::new(LEARNED_OUTPUT_WIDTH,REMASK_FIELDS*2).with_initializer(Initializer::Zeros).init(device),
            round_embedding:EmbeddingConfig::new(rounds,LEARNED_OUTPUT_WIDTH).init(device),hidden}
    }
    fn project(&self,hidden:Tensor<3>,features:Tensor<3>,round:usize,projection:&Linear,outputs:usize)->Tensor<3> {
        assert!(round<self.round_embedding.weight.dims()[0]);let [b,m,h]=hidden.dims();assert_eq!(h,self.hidden);
        assert_eq!(features.dims(),[b,m,REMASK_FEATURES]);
        let rounds=Tensor::<2,Int>::full([b,m],round as i32,&hidden.device());
        let embedded=self.round_embedding.forward(rounds);
        let width=h+REMASK_FEATURES+LEARNED_OUTPUT_WIDTH;
        output_row_logits(&self.mlp,projection,Tensor::cat(vec![hidden.cast(DType::F32),features.cast(DType::F32),embedded],2)
            .reshape([b*m,width])).reshape([b,m,outputs])
    }
    pub fn acceptance(&self,hidden:Tensor<3>,features:Tensor<3>,round:usize)->Tensor<3> {
        self.project(hidden,features,round,&self.acceptance_output,RETAINED_FIELDS*2)
    }
    pub fn outcomes(&self,hidden:Tensor<3>,features:Tensor<3>,round:usize)->Tensor<3> {
        assert!(round+1<self.round_embedding.weight.dims()[0],"the last call has no next-proposal action");
        self.project(hidden,features,round,&self.outcome_output,REMASK_FIELDS*2)
    }
}

/// Strict comparison gives keep on ties; independent precomputed flips apply only to editable fields.
pub fn outcome_actions(logits:&Tensor<3>,observed:&Tensor<3>,valid:&Tensor<2>,
    flips:Option<&Tensor<3>>)->(Tensor<3>,Tensor<3>,Tensor<3>) {
    let [b,m,_]=logits.dims();assert_eq!(logits.dims(),[b,m,REMASK_FIELDS*2]);
    let eligible=observed.clone().lower_elem(0.5).bool_and(valid.clone().greater_elem(0.5).unsqueeze_dim(2));
    let policy=Tensor::cat((0..REMASK_FIELDS).map(|field|logits.clone().slice([0..b,0..m,field*2+1..field*2+2])
        .greater(logits.clone().slice([0..b,0..m,field*2..field*2+1]))).collect(),2).bool_and(eligible.clone());
    let flipped=flips.map(|flips| {assert_eq!(flips.dims(),[b,m,REMASK_FIELDS]);flips.clone().greater_elem(0.5).bool_and(eligible.clone())})
        .unwrap_or_else(||eligible.clone().bool_and(eligible.clone().bool_not()));
    let taken=policy.clone().bool_and(flipped.clone().bool_not()).bool_or(policy.clone().bool_not().bool_and(flipped.clone()));
    (policy.float(),taken.float(),flipped.float())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learned_output::{KnownBinaryTargets,binary_bce};
    fn host<const D:usize>(tensor:&Tensor<D>)->Vec<f32> {tensor.to_data().try_to_vec_as::<f32>().unwrap()}

    #[test]
    fn exp004_refinement_factor_summaries_match_f64_cpu() {
        let device=Device::ndarray();
        let probability=Tensor::from_data(TensorData::new(vec![0.7f32,0.3],[1,1,2]),&device);
        let previous=Tensor::from_data(TensorData::new(vec![0.6f32,0.4],[1,1,2]),&device);
        let frame=RemaskFrame {factors:std::array::from_fn(|_|probability.clone()),
            choices:Tensor::zeros([1,1,REMASK_FACTORS],&device),none:[Some(1);REMASK_FACTORS]};
        let old=RemaskFrame {factors:std::array::from_fn(|_|previous.clone()),choices:frame.choices.clone(),none:frame.none};
        let features=host(&frame.features(Some(&old),&Tensor::zeros([1,1,REMASK_FIELDS],&device)));
        let expected=[0.7,0.4,-0.7f64*0.7f64.ln()-0.3f64*0.3f64.ln(),0.3,0.2,0.0];
        for (actual,expected) in features[..6].iter().zip(expected) {assert!((f64::from(*actual)-expected).abs()<1e-6);}
        assert_eq!(&features[6..12],&[0.0;6]);assert_eq!(features.len(),FACTOR_FEATURES);
        assert!((f64::from(features[72])-0.7f64.powi(11)).abs()<1e-6);
    }

    #[test]
    fn exp004_refinement_outcomes_tie_keep_and_flips_respect_observations_cpu() {
        let device=Device::ndarray();let head=RefinementDecider::new(2,REFINEMENT_ROUNDS,&device);
        let output=head.outcomes(Tensor::zeros([1,2,2],&device),Tensor::zeros([1,2,REMASK_FEATURES],&device),0);
        assert_eq!(host(&output),vec![0.0;24]);
        let observed=Tensor::from_data(TensorData::new(vec![0.0f32,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[1,2,6]),&device);
        let valid=Tensor::from_data([[1.0f32,0.0]],&device);let flips=Tensor::ones([1,2,6],&device);
        let (policy,taken,flipped)=outcome_actions(&output,&observed,&valid,Some(&flips));
        assert_eq!(host(&policy),vec![0.0;12]);assert_eq!(host(&taken),vec![1.0,0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]);
        assert_eq!(host(&taken),host(&flipped));
    }


    #[test]
    fn exp004_refinement_unknown_acceptance_has_no_parameter_gradients_cpu() {
        let device=Device::ndarray().autodiff();let head=RefinementDecider::new(2,REFINEMENT_ROUNDS,&device);
        let output=head.acceptance(Tensor::zeros([1,1,2],&device),Tensor::zeros([1,1,REMASK_FEATURES],&device),0);
        let loss=binary_bce(output.reshape([8]),&KnownBinaryTargets {labels:vec![f32::NAN;8],weights:vec![0.0;8]});
        assert_eq!(host(&loss),vec![0.0]);let anchor=Tensor::<1>::ones([1],&device).require_grad();
        let gradients=(loss+anchor.sum()).backward();
        assert!(head.acceptance_output.weight.val().grad(&gradients).is_none());
        assert!(head.round_embedding.weight.val().grad(&gradients).is_none());
    }

    #[test]
    fn exp004_refinement_statistics_have_finite_subnormal_gradients_cpu() {
        let device=Device::ndarray().autodiff();
        let logits=Tensor::from_data(TensorData::new(vec![0.0f32,-90.0,f32::NEG_INFINITY],[1,1,3]),&device).require_grad();
        let probability=softmax(logits.clone(),2);
        let features=factor_summary(&probability,None,Some(2),Tensor::zeros([1,1,1],&device));
        assert!(host(&features).iter().all(|value|value.is_finite()));
        let gradients=features.sum().backward();
        assert!(host(&logits.grad(&gradients).unwrap()).iter().all(|value|value.is_finite()));
    }

    #[test]
    fn exp004_retained_writes_hold_statistics_and_replace_complete_fields_cpu() {
        let device=Device::ndarray();
        let probability=Tensor::from_data(TensorData::new(vec![0.7f32,0.3],[1,1,2]),&device);
        let mut frame=RemaskFrame {factors:std::array::from_fn(|_|probability.clone()),
            choices:Tensor::from_data(TensorData::new(vec![0i32,0,0,0,1,1,0,1,1,0,1],[1,1,11]),&device),none:[Some(1);11]};
        let mut retained=RetainedAnswer::empty(1,1,2,&device);
        let stats=Tensor::from_data(TensorData::new((0..90).map(|index|index as f32/100.0).collect::<Vec<_>>(),[1,1,90]),&device);
        let accept=Tensor::from_data(TensorData::new(vec![0.0f32;4].into_iter().chain([-1.0;4]).collect::<Vec<_>>(),[1,1,8]),&device);
        let eligible=Tensor::ones([1,1,4],&device);
        let (accepted,replaced)=retained.update(&frame,&stats,&accept,&eligible);
        assert_eq!(host(&accepted),vec![1.0;4]);assert_eq!(host(&replaced),vec![0.0;4]);
        assert_eq!(host(&retained.relation_posterior),vec![0.7,0.3]);
        let original=retained.snapshot();retained.age_one_call(REFINEMENT_ROUNDS);
        frame.choices=Tensor::from_data(TensorData::new(vec![1i32,0,1,1,1,0,0,1,1,1,1],[1,1,11]),&device);
        let (accepted,replaced)=retained.update(&frame,&Tensor::ones([1,1,90],&device),&accept,&eligible);
        assert_eq!(host(&accepted),vec![0.0;4]);assert_eq!(host(&replaced),vec![0.0;4]);
        assert_eq!(original.value.to_data(),retained.value.to_data());assert_eq!(host(&original.factor_stats),host(&retained.factor_stats));
        assert_eq!(host(&retained.age),vec![0.125;4]);
        let (accepted,replaced)=retained.update(&frame,&Tensor::ones([1,1,90],&device),&Tensor::ones([1,1,8],&device),&eligible);
        assert_eq!(host(&accepted),vec![0.0;4]);assert_eq!(host(&replaced),vec![1.0;4]);
        assert_eq!(frame.argmaxes().to_data(),retained.value.to_data());assert_eq!(host(&retained.factor_stats),vec![1.0;72]);
        assert_eq!(host(&retained.age),vec![0.0;4]);assert_eq!(host(&original.age),vec![0.0;4]);
    }

    #[test]
    fn exp004_retained_coherence_uses_raw_bytes_utf8_and_owning_document_cpu() {
        use crate::{SlotPlan,SourcePointerDocument};
        let device=Device::ndarray();
        let plan=SlotPlan::queries(&dre_core::DocumentLayout::one_document_per_row(&[2],2),&[vec![3]]).unwrap();
        let context=SourcePointerContext::new(&plan,Tensor::zeros([1,2,2],&device),&[
            SourcePointerDocument {row:0,document:0,first_token:0,text:"éx",token_offsets:&[(2,3),(0,2)]},
        ],Tensor::zeros([2],&device)).unwrap();
        let mut pointers=PointerState::masked(&context);
        pointers.observed=Tensor::from_data(TensorData::new(vec![1.0f32,0.0,0.0,0.0, 1.0,1.0,0.0,0.0, 0.0,0.0,0.0,0.0],[1,3,4]),&device);
        let frame=RemaskFrame {factors:std::array::from_fn(|_|Tensor::ones([1,3,3],&device)),
            choices:Tensor::from_data(TensorData::new(vec![
                0i32,1,0,0,1,2,0,2,0,0,0,
                0,0,0,1,0,2,0,1,2,1,0,
                0,1,1,0,1,2,0,2,0,0,0],[1,3,11]),&device),none:[None;11]};
        let eligible=coherent_fields(&frame,&pointers,&context,&Tensor::ones([1,3],&device));
        assert_eq!(host(&eligible),vec![1.0,1.0,1.0,1.0, 1.0,0.0,0.0,0.0, 1.0,0.0,1.0,1.0]);
    }

    #[test]
    fn exp004_refinement_supplied_nan_and_padding_have_finite_actual_gradients_cpu() {
        use crate::{PointerLogits,StepOutput};
        let device=Device::ndarray().autodiff();let head=RefinementDecider::new(2,8,&device);
        let logits=||Tensor::from_data(TensorData::new(vec![0.2f32,-0.2,f32::NEG_INFINITY,f32::NEG_INFINITY],[1,2,2]),&device).require_grad();
        let dirty=||Tensor::full([1,2,2],f32::NAN,&device).require_grad();
        let hidden=Tensor::from_data(TensorData::new(vec![0.3f32,-0.4,f32::NAN,f32::NAN],[1,2,2]),&device).require_grad();
        let output=PointerStepOutput {annotations:StepOutput {relation:logits(),evidence_start:logits(),evidence_end:logits()},
            pointers:PointerLogits {token:std::array::from_fn(|_|dirty()),byte:std::array::from_fn(|_|dirty()),
                byte_queries:std::array::from_fn(|_|Tensor::zeros([1,2,2],&device))},hidden:hidden.clone()};
        let pointers=PointerState {token:std::array::from_fn(|_|Tensor::from_data(TensorData::new(vec![1.0f32,0.0,f32::NAN,f32::NAN],[1,2,2]),&device)),
            visibility:Tensor::ones([1,2,4],&device),observed:Tensor::from_data(TensorData::new(
                vec![1.0f32,1.0,1.0,1.0,f32::NAN,f32::NAN,f32::NAN,f32::NAN],[1,2,4]),&device),
            observed_offsets:Tensor::zeros([1,2,4],&device)};
        let valid=Tensor::from_data([[1.0f32,0.0]],&device);let invalid=valid.clone().lower_elem(0.5).unsqueeze_dim(2);
        let observed=observed_fields(&pointers).mask_fill(invalid.clone(),0.0);
        let decoded=crate::plan::int(vec![0;10],[1,2,5],&device);
        let frame=RemaskFrame::from_output(&output,&pointers,0,1,&valid,&decoded);
        let base=frame.features(None,&observed).mask_fill(invalid.clone(),0.0);
        let base_features=RetainedAnswer::empty(1,2,2,&device).features(&frame,base,Tensor::zeros([1,2,6],&device));
        let features=Tensor::cat(vec![base_features,Tensor::zeros([1,2,CIRCUIT_FEATURES],&device)],2);
        assert!(host(&features).iter().all(|value|value.is_finite()));
        assert_eq!(&host(&features)[REMASK_FEATURES..],&[0.0;REMASK_FEATURES]);
        let prediction=head.acceptance(hidden.clone().mask_fill(invalid,0.0),features,0);
        let loss=binary_bce(prediction.reshape([16]),&KnownBinaryTargets {labels:vec![1.0;16],weights:[vec![1.0;8],vec![0.0;8]].concat()});
        let gradients=loss.backward();
        for tensor in output.pointers.token.iter().chain(output.pointers.byte.iter()) {
            assert!(host(&tensor.grad(&gradients).unwrap()).iter().all(|&value|value==0.0));
        }
        for tensor in [&output.annotations.relation,&output.annotations.evidence_start,&output.annotations.evidence_end,&hidden] {
            let gradient=host(&tensor.grad(&gradients).unwrap());assert!(gradient.iter().all(|value|value.is_finite()));
            assert!(gradient[2..].iter().all(|&value|value==0.0));
        }
    }
}

#[cfg(test)]
mod horizon_tests {
    use super::*;
    #[test]
    fn a5_round_rows_keep_original_ids_copy_last_values_and_train_both_new_rows_cpu() {
        let device=Device::ndarray().autodiff();device.seed(20261023);
        let original=RefinementDecider::new(2,8,&device);
        let old_id=original.round_embedding.weight.id;
        let old=original.round_embedding.weight.val().to_data().try_to_vec_as::<f32>().unwrap();
        let mut grown=original.grow_a4_rounds();
        grown.mlp.first_dropout.prob=0.0;grown.mlp.second_dropout.prob=0.0;
        assert_eq!(grown.round_embedding.weight.id,old_id);
        let value=grown.round_embedding.weight.val().to_data().try_to_vec_as::<f32>().unwrap();
        assert_eq!(&value[..8*512],old.as_slice());
        assert_eq!(&value[8*512..9*512],&old[7*512..8*512]);
        assert_eq!(&value[9*512..10*512],&old[7*512..8*512]);
        let output=|round|grown.acceptance(Tensor::full([1,1,2],0.3,&device),Tensor::full([1,1,REMASK_FEATURES],0.2,&device),round).square().sum();
        let gradients=(output(8)+output(9)).backward();
        let gradient=grown.round_embedding.weight.val().grad(&gradients).unwrap().to_data().try_to_vec_as::<f32>().unwrap();
        assert!(gradient[..8*512].iter().all(|&value|value==0.0));
        assert!(gradient[8*512..9*512].iter().any(|value|value.abs()>1e-8));
        assert!(gradient[9*512..10*512].iter().any(|value|value.abs()>1e-8));
        for (horizon,expected) in [(8,0.125f32),(10,0.1)] {
            let mut retained=RetainedAnswer::empty(1,1,2,&device);
            retained.filled=Tensor::from_data(TensorData::new(vec![true;4],[1,1,4]),&device);
            retained.age_one_call(horizon);
            assert_eq!(retained.age.to_data().try_to_vec_as::<f32>().unwrap(),vec![expected;4]);
        }
    }
}
