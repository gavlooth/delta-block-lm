//! Independent f64 circuit inference, exhaustive enumeration, and fixed-tolerance checks.
//! This module intentionally does not reuse tensor recurrences for expected values.
use anyhow::{Context, Result, bail, ensure};
use burn::prelude::*;
use serde_json::{Value, json};

const ABSOLUTE_TOLERANCE: f64 = 2e-5;
const RELATIVE_TOLERANCE: f64 = 3e-4;

#[derive(Default)]
struct Errors {
    count: usize,
    maximum_absolute: f64,
    maximum_ratio: f64,
    worst_actual: f64,
    worst_reference: f64,
    absolute_tolerance: f64,
    relative_tolerance: f64,
}
impl Errors {
    fn add(&mut self, actual: &[f64], expected: &[f64], absolute: f64, relative: f64, name: &str) -> Result<()> {
        ensure!(actual.len() == expected.len(), "{name}: lengths {} != {}", actual.len(), expected.len());
        self.absolute_tolerance = absolute;
        self.relative_tolerance = relative;
        for (i, (&a, &r)) in actual.iter().zip(expected).enumerate() {
            let error = (a - r).abs();
            let ratio = error / (absolute + relative * r.abs());
            ensure!(a.is_finite() && r.is_finite() && ratio <= 1.0,
                "{name}[{i}]: actual={a:.17e}, reference={r:.17e}, abs_error={error:.9e}, tolerance_ratio={ratio:.9e}");
            self.count += 1;
            self.maximum_absolute = self.maximum_absolute.max(error);
            if ratio >= self.maximum_ratio {
                self.maximum_ratio = ratio;
                self.worst_actual = a;
                self.worst_reference = r;
            }
        }
        Ok(())
    }
    fn strict(&mut self, actual: &[f64], expected: &[f64], name: &str) -> Result<()> {
        self.add(actual, expected, 2e-10, 2e-9, name)
    }
    fn tensor(&mut self, actual: &[f64], expected: &[f64], name: &str) -> Result<()> {
        self.add(actual, expected, ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE, name)
    }
    fn report(&self) -> Value {
        json!({"passed":true,"compared_elements":self.count,"max_abs":self.maximum_absolute,
            "maximum_tolerance_ratio":self.maximum_ratio,"worst_actual":self.worst_actual,
            "worst_reference":self.worst_reference,"absolute_tolerance":self.absolute_tolerance,
            "relative_tolerance":self.relative_tolerance})
    }
}

fn log_sum(values: &[f64]) -> f64 {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if maximum == f64::NEG_INFINITY { return maximum; }
    maximum + values.iter().map(|x| (x - maximum).exp()).sum::<f64>().ln()
}
fn normalized(values: &[f64]) -> Result<Vec<f64>> {
    ensure!(values.iter().all(|x| x.is_finite() || *x == f64::NEG_INFINITY), "invalid log row");
    let z = log_sum(values);
    ensure!(z.is_finite(), "empty probability row");
    Ok(values.iter().map(|x| x - z).collect())
}
fn digits(mut index: usize, radix: usize, length: usize) -> Vec<usize> {
    let mut result = vec![0; length];
    for x in &mut result { *x = index % radix; index /= radix; }
    result
}
fn tensor_values<const D: usize>(tensor: Tensor<D>) -> Vec<f64> {
    tensor.into_data().try_to_vec::<f32>().expect("f32 numerical fixture").into_iter().map(f64::from).collect()
}
fn float_tensor<const D: usize>(values: &[f64], shape: [usize; D], device: &Device) -> Tensor<D> {
    Tensor::from_data(TensorData::new(values.iter().map(|&x| x as f32).collect::<Vec<_>>(), shape), device)
}
fn int_tensor(values: &[usize], shape: [usize; 2], device: &Device) -> Tensor<2, Int> {
    Tensor::from_data(TensorData::new(values.iter().map(|&x| x as i32).collect::<Vec<_>>(), shape), device)
}
fn bool_tensor(values: &[bool], shape: [usize; 2], device: &Device) -> Tensor<2, Bool> {
    Tensor::from_data(TensorData::new(values.to_vec(), shape), device)
}

#[derive(Clone)]
struct Circuit {
    states: usize,
    vocab: usize,
    group: usize,
    length: usize,
    log_pi: Vec<f64>,
    log_a: Vec<f64>,
    log_root: Vec<f64>,
    logits: Vec<f64>,
    gates: Vec<f64>,
}
#[derive(Clone)]
struct Prepared {
    states: usize,
    vocab: usize,
    length: usize,
    leaf: Vec<f64>,
    transition: Vec<f64>,
    root: Vec<f64>,
    normalizers: Vec<f64>,
}
#[derive(Clone)]
struct Query {
    log_z: f64,
    state: Vec<f64>,
    edge: Vec<f64>,
    token: Vec<f64>,
    counts: Vec<f64>,
    incoming: Vec<f64>,
    backward: Vec<f64>,
}
#[derive(Clone)]
struct Flow {
    pi_plus: Vec<f64>, pi_minus: Vec<f64>,
    a_plus: Vec<f64>, a_minus: Vec<f64>,
    root_plus: Vec<f64>, root_minus: Vec<f64>,
    pi_reach: Vec<f64>, a_reach: Vec<f64>, root_reach: f64,
}
#[derive(Clone)]
struct Derivative {
    logits: Vec<f64>, gates: Vec<f64>, pi: Vec<f64>, a: Vec<f64>, root: Vec<f64>,
}
impl Circuit {
    fn fixture(states: usize, vocab: usize, length: usize, group: usize, seed: u64) -> Self {
        let mut rng = crate::data::Rng(seed);
        let mut random = |n| (0..n).map(|_| ((rng.next() as f64 / u64::MAX as f64) * 2.4 - 1.2) as f32 as f64).collect::<Vec<_>>();
        let mut log_pi = random(states * vocab);
        let mut log_a = random(states * states);
        let root = random(states);
        for row in log_pi.chunks_exact_mut(vocab) { row.copy_from_slice(&normalized(row).unwrap()); }
        for row in log_a.chunks_exact_mut(states) { row.copy_from_slice(&normalized(row).unwrap()); }
        Self { states, vocab, group, length, log_pi, log_a, log_root: normalized(&root).unwrap(),
            logits: random(length * vocab), gates: random((states / group).pow(2)) }
    }
    fn prepare(&self) -> Result<Prepared> {
        let (k, v, l) = (self.states, self.vocab, self.length);
        ensure!(k > 0 && v > 0 && l > 0 && l <= 16 && self.group > 0 && k % self.group == 0, "invalid reference dimensions");
        ensure!(self.log_pi.len() == k*v && self.log_a.len() == k*k && self.log_root.len() == k
            && self.logits.len() == l*v && self.gates.len() == (k/self.group).pow(2), "reference shape mismatch");
        let groups = k / self.group;
        let mut transition = vec![0.0; k*k];
        for s in 0..k {
            let row = (0..k).map(|t| self.log_a[s*k+t] + self.gates[(s/self.group)*groups+t/self.group]).collect::<Vec<_>>();
            transition[s*k..(s+1)*k].copy_from_slice(&normalized(&row)?);
        }
        let mut leaf = vec![0.0; l*k*v];
        let mut normalizers = vec![0.0; l*k];
        for i in 0..l { for s in 0..k {
            let row = (0..v).map(|x| self.log_pi[s*v+x] + self.logits[i*v+x]).collect::<Vec<_>>();
            normalizers[i*k+s] = log_sum(&row);
            leaf[(i*k+s)*v..(i*k+s+1)*v].copy_from_slice(&normalized(&row)?);
        }}
        Ok(Prepared { states:k, vocab:v, length:l, leaf, transition,
            root:normalized(&self.log_root)?, normalizers })
    }
    fn field(&mut self, field: usize) -> &mut Vec<f64> {
        match field { 0=>&mut self.logits, 1=>&mut self.gates, 2=>&mut self.log_pi, 3=>&mut self.log_a, 4=>&mut self.log_root, _=>unreachable!() }
    }
}
impl Prepared {
    fn selected(&self, evidence: &[Option<usize>]) -> Result<Vec<f64>> {
        ensure!(evidence.len() == self.length, "evidence length mismatch");
        let mut result = vec![0.0; self.length*self.states];
        for (i, value) in evidence.iter().enumerate() {
            if let Some(x) = *value {
                ensure!(x < self.vocab, "evidence token outside vocabulary");
                for s in 0..self.states { result[i*self.states+s] = self.leaf[(i*self.states+s)*self.vocab+x]; }
            }
        }
        Ok(result)
    }
    fn query(&self, evidence: &[Option<usize>]) -> Result<Query> {
        let (k, v, l) = (self.states, self.vocab, self.length);
        let selected = self.selected(evidence)?;
        let mut incoming = vec![0.0; l*k];
        let mut forward = vec![0.0; l*k];
        let mut backward = vec![0.0; l*k];
        incoming[..k].copy_from_slice(&self.root);
        for i in 0..l {
            if i > 0 { for t in 0..k {
                incoming[i*k+t] = log_sum(&(0..k).map(|s| forward[(i-1)*k+s]+self.transition[s*k+t]).collect::<Vec<_>>());
            }}
            for s in 0..k { forward[i*k+s] = incoming[i*k+s]+selected[i*k+s]; }
        }
        for i in (0..l-1).rev() { for s in 0..k {
            backward[i*k+s] = log_sum(&(0..k).map(|t| self.transition[s*k+t]+selected[(i+1)*k+t]+backward[(i+1)*k+t]).collect::<Vec<_>>());
        }}
        let log_z = log_sum(&forward[(l-1)*k..]);
        ensure!(log_z.is_finite(), "evidence has zero probability");
        let mut state = vec![0.0; l*k];
        let mut edge = vec![0.0; (l-1)*k*k];
        let mut token = vec![0.0; l*v];
        let mut counts = vec![0.0; l*k*v];
        for i in 0..l { for s in 0..k {
            state[i*k+s] = (forward[i*k+s]+backward[i*k+s]-log_z).exp();
            for x in 0..v {
                let p = if let Some(observed) = evidence[i] { if x == observed {1.0} else {0.0} } else { self.leaf[(i*k+s)*v+x].exp() };
                counts[(i*k+s)*v+x] = state[i*k+s]*p;
                token[i*v+x] += counts[(i*k+s)*v+x];
            }
            if i > 0 { for r in 0..k {
                edge[((i-1)*k+r)*k+s] = (forward[(i-1)*k+r]+self.transition[r*k+s]+selected[i*k+s]+backward[i*k+s]-log_z).exp();
            }}
        }}
        Ok(Query {log_z,state,edge,token,counts,incoming,backward})
    }
    fn flows(&self, positive: &Query, negative: &Query) -> Flow {
        let (k,v,l) = (self.states,self.vocab,self.length);
        let mut result = Flow {pi_plus:vec![0.0;k*v],pi_minus:vec![0.0;k*v],a_plus:vec![0.0;k*k],a_minus:vec![0.0;k*k],
            root_plus:vec![0.0;k],root_minus:vec![0.0;k],pi_reach:vec![0.0;k],a_reach:vec![0.0;k],root_reach:1.0};
        let mut reach = self.root.iter().map(|x|x.exp()).collect::<Vec<_>>();
        for i in 0..l {
            for s in 0..k {
                result.pi_reach[s] += reach[s];
                for x in 0..v {
                    let g = self.leaf[(i*k+s)*v+x].exp();
                    result.pi_plus[s*v+x] += positive.counts[(i*k+s)*v+x]+negative.state[i*k+s]*g;
                    result.pi_minus[s*v+x] += negative.counts[(i*k+s)*v+x]+positive.state[i*k+s]*g;
                }
                if i+1 < l {
                    result.a_reach[s] += reach[s];
                    let rp = positive.edge[i*k*k+s*k..i*k*k+(s+1)*k].iter().sum::<f64>();
                    let rn = negative.edge[i*k*k+s*k..i*k*k+(s+1)*k].iter().sum::<f64>();
                    for t in 0..k {
                        let a = self.transition[s*k+t].exp();
                        result.a_plus[s*k+t] += positive.edge[(i*k+s)*k+t]+rn*a;
                        result.a_minus[s*k+t] += negative.edge[(i*k+s)*k+t]+rp*a;
                    }
                }
            }
            if i+1 < l {
                reach = (0..k).map(|t|(0..k).map(|s|reach[s]*self.transition[s*k+t].exp()).sum()).collect();
            }
        }
        for s in 0..k {
            result.root_plus[s] = positive.state[s]+self.root[s].exp();
            result.root_minus[s] = negative.state[s]+self.root[s].exp();
        }
        result
    }
    fn derivative(&self, positive: &Query, negative: &Query, group: usize) -> Derivative {
        let (k,v,l) = (self.states,self.vocab,self.length);
        let flows = self.flows(positive,negative);
        let mut result = Derivative {logits:vec![0.0;l*v],gates:vec![0.0;(k/group).pow(2)],
            pi:flows.pi_plus.iter().zip(&flows.pi_minus).map(|(p,n)|p-n).collect(),
            a:flows.a_plus.iter().zip(&flows.a_minus).map(|(p,n)|p-n).collect(),
            root:flows.root_plus.iter().zip(&flows.root_minus).map(|(p,n)|p-n).collect()};
        for i in 0..l { for s in 0..k { for x in 0..v {
            result.logits[i*v+x] += positive.counts[(i*k+s)*v+x]-negative.counts[(i*k+s)*v+x]
                -(positive.state[i*k+s]-negative.state[i*k+s])*self.leaf[(i*k+s)*v+x].exp();
        }}}
        for s in 0..k { for t in 0..k {
            result.gates[(s/group)*(k/group)+t/group] += result.a[s*k+t];
        }}
        result
    }
    fn query_scores(&self, evidence: &[Option<usize>], proposal: &[usize]) -> Result<Vec<f64>> {
        ensure!(proposal.len() == self.length, "proposal length mismatch");
        let base = self.query(evidence)?.log_z;
        let full = proposal.iter().enumerate().map(|(i,&x)|evidence[i].or(Some(x))).collect::<Vec<_>>();
        let complete = self.query(&full)?.log_z;
        let mut result = vec![0.0; self.length];
        for i in 0..self.length {
            if evidence[i].is_some() {continue;}
            let mut single = evidence.to_vec(); single[i] = Some(proposal[i]);
            let mut without = full.clone(); without[i] = None;
            result[i] = (complete-self.query(&without)?.log_z-self.query(&single)?.log_z+base).abs();
        }
        Ok(result)
    }
    fn cavity_scores(&self, evidence: &[Option<usize>], proposal: &[usize]) -> Result<Vec<f64>> {
        let base = self.query(evidence)?;
        let full = proposal.iter().enumerate().map(|(i,&x)|evidence[i].or(Some(x))).collect::<Vec<_>>();
        let complete = self.query(&full)?;
        let mut result = vec![0.0; self.length];
        for i in 0..self.length {
            if evidence[i].is_some() {continue;}
            let without = log_sum(&(0..self.states).map(|s|complete.incoming[i*self.states+s]+complete.backward[i*self.states+s]).collect::<Vec<_>>());
            result[i] = (complete.log_z-without-base.token[i*self.vocab+proposal[i]].ln()).abs();
        }
        Ok(result)
    }
}

struct Assignment { output: Vec<usize>, latent: Vec<usize>, mass: f64 }
struct Enumeration {states:usize,vocab:usize,length:usize,assignments:Vec<Assignment>}
impl Enumeration {
    fn new(prepared: &Prepared) -> Self {
        let (k,v,l) = (prepared.states,prepared.vocab,prepared.length);
        let mut assignments = Vec::with_capacity(k.pow(l as u32)*v.pow(l as u32));
        for x in 0..v.pow(l as u32) {
            let output = digits(x,v,l);
            for z in 0..k.pow(l as u32) {
                let latent = digits(z,k,l);
                let mut mass = prepared.root[latent[0]].exp();
                for i in 0..l {
                    mass *= prepared.leaf[(i*k+latent[i])*v+output[i]].exp();
                    if i > 0 {mass *= prepared.transition[latent[i-1]*k+latent[i]].exp();}
                }
                assignments.push(Assignment {output:output.clone(),latent,mass});
            }
        }
        Self {states:k,vocab:v,length:l,assignments}
    }
    fn query(&self, evidence: &[Option<usize>]) -> Result<Query> {
        let (k,v,l) = (self.states,self.vocab,self.length);
        let mut state = vec![0.0;l*k];
        let mut edge = vec![0.0;(l-1)*k*k];
        let mut token = vec![0.0;l*v];
        let mut counts = vec![0.0;l*k*v];
        let mut mass = 0.0;
        for assignment in &self.assignments {
            if evidence.iter().enumerate().any(|(i,x)|x.is_some_and(|x|x != assignment.output[i])) {continue;}
            mass += assignment.mass;
            for i in 0..l {
                let (s,x) = (assignment.latent[i],assignment.output[i]);
                state[i*k+s] += assignment.mass;
                token[i*v+x] += assignment.mass;
                counts[(i*k+s)*v+x] += assignment.mass;
                if i > 0 {edge[((i-1)*k+assignment.latent[i-1])*k+s] += assignment.mass;}
            }
        }
        ensure!(mass > 0.0 && mass.is_finite(), "enumerated evidence has zero probability");
        for values in [&mut state,&mut edge,&mut token,&mut counts] {for x in values {*x /= mass;}}
        Ok(Query {log_z:mass.ln(),state,edge,token,counts,incoming:Vec::new(),backward:Vec::new()})
    }
    fn output_probabilities(&self, evidence: &[Option<usize>]) -> Result<Vec<f64>> {
        let normalizer = self.query(evidence)?.log_z.exp();
        let mut result = vec![0.0;self.vocab.pow(self.length as u32)];
        for assignment in &self.assignments {
            if evidence.iter().enumerate().any(|(i,x)|x.is_some_and(|x|x!=assignment.output[i])) {continue;}
            let index = assignment.output.iter().rev().fold(0,|index,&x|index*self.vocab+x);
            result[index] += assignment.mass/normalizer;
        }
        Ok(result)
    }
}

fn finite_differences(circuit: &Circuit, evidence: &[Option<usize>], clean: &[usize]) -> Result<Value> {
    let prepared = circuit.prepare()?;
    let complete = clean.iter().copied().map(Some).collect::<Vec<_>>();
    let positive = prepared.query(&complete)?;
    let negative = prepared.query(evidence)?;
    let derivative = prepared.derivative(&positive,&negative,circuit.group);
    let expected = [&derivative.logits,&derivative.gates,&derivative.pi,&derivative.a,&derivative.root];
    let mut reports = Vec::new();
    for (field,name) in ["token_potentials","group_gates","base_emission_log_coordinates","base_transition_log_coordinates","normalized_root_log_coordinates"].into_iter().enumerate() {
        let mut actual = Vec::with_capacity(expected[field].len());
        for coordinate in 0..expected[field].len() {
            let step = 1e-5;
            let mut plus = circuit.clone(); plus.field(field)[coordinate] += step;
            let mut minus = circuit.clone(); minus.field(field)[coordinate] -= step;
            let plus = plus.prepare()?; let minus = minus.prepare()?;
            actual.push(((plus.query(&complete)?.log_z-plus.query(evidence)?.log_z)
                -(minus.query(&complete)?.log_z-minus.query(evidence)?.log_z))/(2.0*step));
        }
        let mut errors = Errors::default();
        errors.add(&actual,expected[field],2e-7,2e-6,name)?;
        reports.push(json!({"quantity":name,"errors":errors.report(),"all_coordinates":true}));
    }
    let flow = prepared.flows(&positive,&negative);
    for values in [&flow.pi_plus,&flow.pi_minus,&flow.a_plus,&flow.a_minus,&flow.root_plus,&flow.root_minus] {
        ensure!(values.iter().all(|x|x.is_finite() && *x >= 0.0), "negative or invalid separated flow");
    }
    Ok(json!({"objective":"full conditional joint log likelihood","difference_step":1e-5,"coordinate_convention":"independent log entries with each row normalization differentiated","quantities":reports}))
}

fn enumeration_checks() -> Result<Value> {
    let mut reports = Vec::new();
    for (index,(k,v,l,group)) in [(2,2,1,1),(2,3,2,1),(3,2,3,1),(3,3,4,1),(2,2,4,2)].into_iter().enumerate() {
        let circuit = Circuit::fixture(k,v,l,group,4901+index as u64);
        let prepared = circuit.prepare()?;
        let enumerated = Enumeration::new(&prepared);
        let mut partition = Errors::default();
        let mut masses = Errors::default();
        let mut states = Errors::default();
        let mut edges = Errors::default();
        let mut tokens = Errors::default();
        let mut counts = Errors::default();
        let mut conditional = Errors::default();
        let mut deltas = Errors::default();
        partition.strict(&[enumerated.assignments.iter().map(|x|x.mass).sum()],&[1.0],"unconditional partition")?;
        partition.strict(&[prepared.query(&vec![None;l])?.log_z],&[0.0],"unconditional log partition")?;
        let mut evidence_queries = 0;
        for assignment in 0..(v+1).pow(l as u32) {
            let evidence = digits(assignment,v+1,l).into_iter().map(|x|if x==v {None} else {Some(x)}).collect::<Vec<_>>();
            let actual = prepared.query(&evidence)?;
            let expected = enumerated.query(&evidence)?;
            masses.strict(&[actual.log_z],&[expected.log_z],"evidence log mass")?;
            states.strict(&actual.state,&expected.state,"latent posterior")?;
            edges.strict(&actual.edge,&expected.edge,"edge posterior")?;
            tokens.strict(&actual.token,&expected.token,"conditional token marginal")?;
            counts.strict(&actual.counts,&expected.counts,"emission expected count")?;
            let probabilities = enumerated.output_probabilities(&evidence)?;
            conditional.strict(&[probabilities.iter().sum()],&[1.0],"conditional partition")?;
            for (x,&p) in probabilities.iter().enumerate() {
                if p == 0.0 {continue;}
                let output = digits(x,v,l);
                let complete = output.iter().copied().map(Some).collect::<Vec<_>>();
                conditional.strict(&[prepared.query(&complete)?.log_z-actual.log_z],&[p.ln()],"conditional joint log probability")?;
            }
            let proposal = (0..l).map(|i|(i+assignment)%v).collect::<Vec<_>>();
            let query = prepared.query_scores(&evidence,&proposal)?;
            let cavity = prepared.cavity_scores(&evidence,&proposal)?;
            deltas.strict(&query,&cavity,"query versus cavity dependence")?;
            let full = (0..l).map(|i|evidence[i].or(Some(proposal[i]))).collect::<Vec<_>>();
            let full_mass = enumerated.query(&full)?.log_z;
            for i in 0..l {
                if evidence[i].is_some() {continue;}
                let mut without = full.clone(); without[i] = None;
                let direct = (full_mass-enumerated.query(&without)?.log_z-expected.token[i*v+proposal[i]].ln()).abs();
                deltas.strict(&[cavity[i]],&[direct],"enumerated dependence definition")?;
            }
            evidence_queries += 1;
        }
        let clean = (0..l).map(|i|(i+1)%v).collect::<Vec<_>>();
        let evidence = clean.iter().enumerate().map(|(i,&x)|if i%2 == 1 {Some(x)} else {None}).collect::<Vec<_>>();
        let complete = clean.iter().copied().map(Some).collect::<Vec<_>>();
        let flow = prepared.flows(&prepared.query(&complete)?,&prepared.query(&evidence)?);
        let enumerated_flow = prepared.flows(&enumerated.query(&complete)?,&enumerated.query(&evidence)?);
        let mut separate_flows = Errors::default();
        for (actual,expected) in [
            (&flow.pi_plus,&enumerated_flow.pi_plus),(&flow.pi_minus,&enumerated_flow.pi_minus),
            (&flow.a_plus,&enumerated_flow.a_plus),(&flow.a_minus,&enumerated_flow.a_minus),
            (&flow.root_plus,&enumerated_flow.root_plus),(&flow.root_minus,&enumerated_flow.root_minus),
        ] {separate_flows.strict(actual,expected,"enumerated separate nonnegative flows")?;}
        reports.push(json!({"states":k,"vocabulary":v,"length":l,"group_size":group,
            "enumerated_joint_assignments":enumerated.assignments.len(),"evidence_queries":evidence_queries,
            "partition":partition.report(),"evidence_log_mass":masses.report(),"state_posterior":states.report(),
            "edge_posterior":edges.report(),"token_marginals":tokens.report(),"emission_counts":counts.report(),
            "conditional_joint":conditional.report(),"dependence":deltas.report(),"separate_flows":separate_flows.report(),
            "full_finite_differences":finite_differences(&circuit,&evidence,&clean)?}));
    }
    Ok(json!({"passed":true,"f64_tolerance":{"absolute":2e-10,"relative":2e-9},"cases":reports}))
}

fn tensor_case(device: &Device, states: usize, vocab: usize, lengths: &[usize], group: usize, execution: crate::codpula::CodpulaExecution) -> Result<Value> {
    use crate::codpula::PreparedCircuit;
    let device = device.clone().autodiff();
    let (n,k,v,l) = (lengths.len(),states,vocab,*lengths.iter().max().unwrap());
    let base = Circuit::fixture(k,v,l,group,8903+l as u64);
    let mut logits = vec![83.0;n*l*v];
    let mut gates = Vec::new();
    let mut clean = vec![v+199;n*l];
    let mut observed = vec![false;n*l];
    let mut expected_z = vec![0.0;n];
    let mut expected_state = vec![0.0;n*l*k];
    let mut expected_edge = vec![0.0;n*l*k*k];
    let mut expected_token = vec![0.0;n*l*v];
    let mut expected_counts = vec![0.0;n*l*k*v];
    let mut expected_logits = vec![0.0;n*l*v];
    let mut expected_gates = vec![0.0;n*(k/group).pow(2)];
    let mut expected_pi = vec![0.0;k*v];
    let mut expected_a = vec![0.0;k*k];
    let mut expected_root = vec![0.0;k];
    let mut expected_flows = Flow {pi_plus:vec![0.0;k*v],pi_minus:vec![0.0;k*v],
        a_plus:vec![0.0;k*k],a_minus:vec![0.0;k*k],root_plus:vec![0.0;k],root_minus:vec![0.0;k],
        pi_reach:vec![0.0;k],a_reach:vec![0.0;k],root_reach:0.0};
    let denominator = 7.0;
    let mut expected_loss = 0.0;
    let mut active_blocks = 0;
    let mut proposals = Vec::new();
    let mut expected_scores = Vec::new();
    for (row,&length) in lengths.iter().enumerate() {
        let mut circuit = base.clone();
        circuit.length = length.max(1);
        circuit.logits = (0..circuit.length*v).map(|i|(((i*31+row*17)%29) as f64/9.0-1.4) as f32 as f64).collect();
        circuit.gates = (0..(k/group).pow(2)).map(|i|(((i*13+row*7)%31) as f64/17.0-0.9) as f32 as f64).collect();
        gates.extend_from_slice(&circuit.gates);
        if length == 0 { proposals.push(Vec::new()); expected_scores.push(Vec::new()); continue; }
        logits[row*l*v..row*l*v+length*v].copy_from_slice(&circuit.logits);
        let target = (0..length).map(|i|(i*37+row*11+3)%v).collect::<Vec<_>>();
        let evidence = target.iter().enumerate().map(|(i,&x)| {
            let fixed = if row == 2 {true} else {i%3 == 1};
            clean[row*l+i] = x;
            observed[row*l+i] = fixed;
            if fixed {Some(x)} else {None}
        }).collect::<Vec<_>>();
        let prepared = circuit.prepare()?;
        let negative = prepared.query(&evidence)?;
        let positive = prepared.query(&target.iter().copied().map(Some).collect::<Vec<_>>())?;
        expected_z[row] = negative.log_z;
        expected_state[row*l*k..row*l*k+length*k].copy_from_slice(&negative.state);
        expected_token[row*l*v..row*l*v+length*v].copy_from_slice(&negative.token);
        expected_counts[row*l*k*v..row*l*k*v+length*k*v].copy_from_slice(&negative.counts);
        for i in 1..length {
            expected_edge[(row*l+i)*k*k..(row*l+i+1)*k*k].copy_from_slice(&negative.edge[(i-1)*k*k..i*k*k]);
        }
        let proposal = (0..length).map(|i|evidence[i].unwrap_or((i+row+1)%v)).collect::<Vec<_>>();
        expected_scores.push(prepared.query_scores(&evidence,&proposal)?);
        proposals.push(proposal);
        if evidence.iter().all(Option::is_some) {continue;}
        active_blocks += 1;
        expected_loss += (negative.log_z-positive.log_z)/denominator;
        let derivative = prepared.derivative(&positive,&negative,group);
        for i in 0..length*v {expected_logits[row*l*v+i] = -derivative.logits[i]/denominator;}
        for (i,&x) in derivative.gates.iter().enumerate() {expected_gates[row*(k/group).pow(2)+i] = -x/denominator;}
        for (out,gradient) in [(&mut expected_pi,&derivative.pi),(&mut expected_a,&derivative.a),(&mut expected_root,&derivative.root)] {
            for (out,&g) in out.iter_mut().zip(gradient) {*out -= g/denominator;}
        }
        let flows = prepared.flows(&positive,&negative);
        for (out,input) in [
            (&mut expected_flows.pi_plus,&flows.pi_plus),(&mut expected_flows.pi_minus,&flows.pi_minus),
            (&mut expected_flows.a_plus,&flows.a_plus),(&mut expected_flows.a_minus,&flows.a_minus),
            (&mut expected_flows.root_plus,&flows.root_plus),(&mut expected_flows.root_minus,&flows.root_minus),
            (&mut expected_flows.pi_reach,&flows.pi_reach),(&mut expected_flows.a_reach,&flows.a_reach),
        ] {for (out,&x) in out.iter_mut().zip(input) {*out += x;}}
        expected_flows.root_reach += flows.root_reach;
    }
    let pi = float_tensor(&base.log_pi,[k,v],&device).require_grad();
    let a = float_tensor(&base.log_a,[k,k],&device).require_grad();
    let root = float_tensor(&base.log_root,[k],&device).require_grad();
    let transposed_logits = (0..n).flat_map(|row|(0..v).flat_map({
        let logits = &logits;
        move |token|(0..l).map(move |position|logits[(row*l+position)*v+token])
    })).collect::<Vec<_>>();
    let potentials = float_tensor(&transposed_logits,[n,v,l],&device).swap_dims(1,2).require_grad();
    let gate = float_tensor(&gates,[n,k/group,k/group],&device).require_grad();
    let prepared = PreparedCircuit::prepare_with_execution(pi.clone(),a.clone(),root.clone(),potentials.clone(),gate.clone(),lengths.to_vec(),group,execution)?;
    let values = int_tensor(&clean,[n,l],&device);
    let fixed = bool_tensor(&observed,[n,l],&device);
    let query = prepared.query(values.clone(),fixed.clone())?;
    let mut forward = Vec::new();
    for (name,actual,expected) in [
        ("evidence_log_mass",tensor_values(query.log_z),expected_z),
        ("state_posterior",tensor_values(query.state),expected_state),
        ("edge_posterior",tensor_values(query.edge),expected_edge),
        ("token_marginals",tensor_values(query.token),expected_token),
        ("emission_counts",tensor_values(query.emission_counts),expected_counts),
    ] {
        let mut errors = Errors::default(); errors.tensor(&actual,&expected,name)?;
        forward.push(json!({"quantity":name,"errors":errors.report()}));
    }
    let loss = prepared.conditional_nll(values.clone(),fixed.clone(),denominator)?;
    let mut loss_errors = Errors::default(); loss_errors.tensor(&tensor_values(loss.loss.clone()),&[expected_loss],"conditional NLL")?;
    let gradients = loss.loss.backward();
    let mut derivatives = Vec::new();
    for (name,actual,expected) in [
        ("token_potentials",tensor_values(potentials.grad(&gradients).context("token potential derivative missing")?),expected_logits),
        ("group_gates",tensor_values(gate.grad(&gradients).context("group gate derivative missing")?),expected_gates),
        ("base_emission",tensor_values(pi.grad(&gradients).context("emission derivative missing")?),expected_pi),
        ("base_transition",tensor_values(a.grad(&gradients).context("transition derivative missing")?),expected_a),
        ("normalized_root",tensor_values(root.grad(&gradients).context("root derivative missing")?),expected_root),
    ] {
        let mut errors = Errors::default(); errors.tensor(&actual,&expected,name)?;
        derivatives.push(json!({"quantity":name,"errors":errors.report(),"all_coordinates":true}));
    }
    ensure!(loss.flows.active_blocks == active_blocks, "active block count differs");
    let mut flows = Vec::new();
    for (name,actual,expected) in [
        ("emission_positive",tensor_values(loss.flows.emission_positive),expected_flows.pi_plus),
        ("emission_negative",tensor_values(loss.flows.emission_negative),expected_flows.pi_minus),
        ("transition_positive",tensor_values(loss.flows.transition_positive),expected_flows.a_plus),
        ("transition_negative",tensor_values(loss.flows.transition_negative),expected_flows.a_minus),
        ("root_positive",tensor_values(loss.flows.root_positive),expected_flows.root_plus),
        ("root_negative",tensor_values(loss.flows.root_negative),expected_flows.root_minus),
        ("emission_traversal",tensor_values(loss.flows.emission_reach),expected_flows.pi_reach),
        ("transition_traversal",tensor_values(loss.flows.transition_reach),expected_flows.a_reach),
        ("root_traversal",tensor_values(loss.flows.root_reach),vec![expected_flows.root_reach]),
    ] {
        ensure!(actual.iter().all(|x|*x >= 0.0 && x.is_finite()), "{name}: invalid separated flow");
        let mut errors = Errors::default(); errors.tensor(&actual,&expected,name)?;
        flows.push(json!({"quantity":name,"errors":errors.report()}));
    }
    let scores = prepared.dependence_scores(values,fixed,&proposals)?;
    let mut score_errors = Errors::default();
    for (actual,expected) in scores.iter().zip(&expected_scores) {score_errors.tensor(actual,expected,"dependence score")?;}
    ensure!(scores.len() == expected_scores.len(), "dependence score row count differs");
    Ok(json!({"states":k,"vocabulary":v,"lengths":lengths,"group_size":group,"execution":format!("{execution:?}"),
        "dirty_padding":true,"noncontiguous_token_potentials":true,
        "active_blocks":active_blocks,"forward":forward,"conditional_nll":loss_errors.report(),
        "full_nll_vjp":derivatives,"separate_flows":flows,"dependence":score_errors.report()}))
}

fn marginal_vjp(device: &Device, execution: crate::codpula::CodpulaExecution) -> Result<Value> {
    use crate::codpula::PreparedCircuit;
    let device = device.clone().autodiff();
    let circuit = Circuit::fixture(2,3,3,1,5117);
    let evidence = vec![None,Some(2),None];
    let weights = (0..9).map(|i|((i*13%19) as f64/7.0-1.0) as f32 as f64).collect::<Vec<_>>();
    let objective = |c: &Circuit| -> Result<f64> {
        let query = c.prepare()?.query(&evidence)?;
        Ok(query.token.iter().zip(&weights).map(|(p,w)|p*w).sum::<f64>()+0.37*query.log_z)
    };
    let pi = float_tensor(&circuit.log_pi,[2,3],&device).require_grad();
    let a = float_tensor(&circuit.log_a,[2,2],&device).require_grad();
    let root = float_tensor(&circuit.log_root,[2],&device).require_grad();
    let logits = float_tensor(&circuit.logits,[1,3,3],&device).require_grad();
    let gates = float_tensor(&circuit.gates,[1,2,2],&device).require_grad();
    let prepared = PreparedCircuit::prepare_with_execution(pi.clone(),a.clone(),root.clone(),logits.clone(),gates.clone(),vec![3],1,execution)?;
    let query = prepared.query(int_tensor(&[0,2,0],[1,3],&device),bool_tensor(&[false,true,false],[1,3],&device))?;
    let upstream = float_tensor(&weights,[1,3,3],&device);
    let gradients = ((query.token*upstream).sum()+query.log_z.sum()*0.37).backward();
    let actual = [
        tensor_values(logits.grad(&gradients).context("marginal logits derivative")?),
        tensor_values(gates.grad(&gradients).context("marginal gates derivative")?),
        tensor_values(pi.grad(&gradients).context("marginal emission derivative")?),
        tensor_values(a.grad(&gradients).context("marginal transition derivative")?),
        tensor_values(root.grad(&gradients).context("marginal root derivative")?),
    ];
    let mut reports = Vec::new();
    for field in 0..5 {
        let mut expected = Vec::new();
        for i in 0..actual[field].len() {
            let mut plus = circuit.clone(); plus.field(field)[i] += 1e-5;
            let mut minus = circuit.clone(); minus.field(field)[i] -= 1e-5;
            expected.push((objective(&plus)?-objective(&minus)?)/2e-5);
        }
        let mut errors = Errors::default(); errors.tensor(&actual[field],&expected,"marginal full VJP")?;
        reports.push(json!({"input":(["token_potentials","group_gates","base_emission","base_transition","normalized_root"][field]),
            "errors":errors.report(),"all_coordinates":true}));
    }
    Ok(json!({"objective":"sum(token_marginal * signed_upstream) + 0.37*evidence_log_mass",
        "observations":[null,2,null],"full_finite_difference_vjp":reports}))
}

fn categorical(probabilities: &[f64], rng: &mut crate::data::Rng) -> Result<usize> {
    let sum = probabilities.iter().sum::<f64>();
    ensure!(sum.is_finite() && sum > 0.0 && probabilities.iter().all(|x|*x >= 0.0), "invalid sampling categorical row");
    let draw = ((rng.next() >> 11) as f64)*(1.0/9007199254740992.0)*sum;
    let mut cumulative = 0.0;
    for (i,&p) in probabilities.iter().enumerate() {
        cumulative += p;
        if draw < cumulative { return Ok(i); }
    }
    bail!("sampling cumulative mass did not cover draw")
}
fn sample_reference(prepared: &Prepared, evidence: &[Option<usize>], query: &Query, rng: &mut crate::data::Rng) -> Result<Vec<usize>> {
    let (k,v,l) = (prepared.states,prepared.vocab,prepared.length);
    let mut latent = vec![0;l];
    latent[l-1] = categorical(&query.state[(l-1)*k..],rng)?;
    for i in (0..l-1).rev() {
        let probabilities = (0..k).map(|s|query.edge[(i*k+s)*k+latent[i+1]]).collect::<Vec<_>>();
        latent[i] = categorical(&probabilities,rng)?;
    }
    let mut result = vec![0;l];
    for i in 0..l {
        result[i] = if let Some(x) = evidence[i] {x} else {
            categorical(&prepared.leaf[(i*k+latent[i])*v..(i*k+latent[i]+1)*v].iter().map(|x|x.exp()).collect::<Vec<_>>(),rng)?
        };
    }
    Ok(result)
}
fn sampling_report(samples: &[Vec<usize>], probabilities: &[f64], evidence: &[Option<usize>], vocab: usize, seed: u64) -> Result<Value> {
    let mut counts = vec![0usize;probabilities.len()];
    let length = evidence.len();
    let mut equal = 0usize;
    for sample in samples {
        ensure!(sample.len() == length && sample.iter().all(|&x|x < vocab), "sample shape/support differs");
        ensure!(evidence.iter().enumerate().all(|(i,x)|x.is_none_or(|x|sample[i] == x)), "sampler changed an observed token");
        let index = sample.iter().rev().fold(0,|index,&x|index*vocab+x);
        counts[index] += 1;
        equal += usize::from(sample[0] == sample[1]);
    }
    let alpha = 1e-6f64;
    let events = probabilities.len()+1;
    let radius = ((2.0*events as f64/alpha).ln()/(2.0*samples.len() as f64)).sqrt();
    let mut maximum_error = 0.0f64;
    for (i,(&count,&p)) in counts.iter().zip(probabilities).enumerate() {
        let actual = count as f64/samples.len() as f64;
        let error = (actual-p).abs();
        ensure!(error <= radius, "sample frequency[{i}]={actual}, expected={p}, Hoeffding radius={radius}, family alpha={alpha}");
        maximum_error = maximum_error.max(error);
        if p == 0.0 {ensure!(count == 0, "sample outside conditional support");}
    }
    let expected_equal = probabilities.iter().enumerate().filter(|(i,_)|i%vocab == (i/vocab)%vocab).map(|(_,p)|p).sum::<f64>();
    let actual_equal = equal as f64/samples.len() as f64;
    ensure!((actual_equal-expected_equal).abs() <= radius, "joint sampler cross-position equality mismatch");
    let p0 = probabilities.iter().enumerate().filter(|(i,_)|i%vocab == 1).map(|(_,p)|p).sum::<f64>();
    let p1 = probabilities.iter().enumerate().filter(|(i,_)|(i/vocab)%vocab == 1).map(|(_,p)|p).sum::<f64>();
    let both = probabilities.iter().enumerate().filter(|(i,_)|i%vocab == 1 && (i/vocab)%vocab == 1).map(|(_,p)|p).sum::<f64>();
    Ok(json!({"passed":true,"seed":seed,"samples":samples.len(),"confidence_rule":"two-sided Hoeffding for every output-sequence indicator and the cross-position equality indicator, with a union bound",
        "family_false_rejection_probability_at_most":alpha,"events":events,"absolute_frequency_radius":radius,
        "maximum_frequency_error":maximum_error,"joint_equality":{"actual":actual_equal,"reference":expected_equal},
        "reference_binary_covariance":both-p0*p1,"counts":counts,"reference_probabilities":probabilities}))
}
fn sampling_checks(device: &Device) -> Result<Value> {
    use crate::codpula::PreparedCircuit;
    let mut circuit = Circuit::fixture(2,2,3,1,6121);
    circuit.log_pi = [0.96f64,0.04,0.04,0.96].map(f64::ln).to_vec();
    circuit.log_a = [0.97f64,0.03,0.03,0.97].map(f64::ln).to_vec();
    circuit.log_root = vec![-2.0f64.ln();2];
    circuit.logits.fill(0.0); circuit.gates.fill(0.0);
    let prepared = circuit.prepare()?;
    let evidence = vec![None,None,Some(1)];
    let query = prepared.query(&evidence)?;
    let enumeration = Enumeration::new(&prepared);
    let probabilities = enumeration.output_probabilities(&evidence)?;
    let mut rows = Errors::default();
    let enumerated = enumeration.query(&evidence)?;
    rows.strict(&query.state[4..6],&enumerated.state[4..6],"terminal latent categorical")?;
    for i in 0..2 { for next in 0..2 {
        let actual = (0..2).map(|s|query.edge[(i*2+s)*2+next]/query.state[(i+1)*2+next]).collect::<Vec<_>>();
        let expected = (0..2).map(|s|enumerated.edge[(i*2+s)*2+next]/enumerated.state[(i+1)*2+next]).collect::<Vec<_>>();
        rows.strict(&actual,&expected,"backward sampling categorical")?;
    }}
    let seed = 0x4c1a5d23;
    let mut rng = crate::data::Rng(seed);
    let samples = (0..60000).map(|_|sample_reference(&prepared,&evidence,&query,&mut rng)).collect::<Result<Vec<_>>>()?;
    let reference = sampling_report(&samples,&probabilities,&evidence,2,seed)?;
    // Identical independent rows amortize the tensor readback without changing the joint draw within a row.
    let batch = 4096;
    let tensor = PreparedCircuit::from_log_potentials(
        float_tensor(&circuit.log_pi,[2,2],device),float_tensor(&circuit.log_a,[2,2],device),
        float_tensor(&circuit.log_root,[2],device),float_tensor(&circuit.logits.repeat(batch),[batch,3,2],device),
        float_tensor(&circuit.gates.repeat(batch),[batch,2,2],device),vec![3;batch],1)?;
    let values = int_tensor(&[0,0,1].repeat(batch),[batch,3],device);
    let fixed = bool_tensor(&[false,false,true].repeat(batch),[batch,3],device);
    let actual = tensor.sample(values,fixed,seed)?;
    ensure!(actual.len() == batch, "joint sampler row count differs");
    let tensor = sampling_report(&actual,&probabilities,&evidence,2,seed)?;
    Ok(json!({"categorical_rows":rows.report(),"f64_ffbs":reference,"tensor_ffbs":tensor}))
}

fn row_update(weights: &[f64], positive: &[f64], negative: &[f64], reach: f64, alpha: f64) -> Result<Vec<f64>> {
    ensure!(weights.len() == positive.len() && weights.len() == negative.len(), "row update dimensions");
    ensure!(alpha.is_finite() && alpha >= 0.0 && reach.is_finite() && reach >= 0.0, "row update step/traversal");
    ensure!(weights.iter().all(|x|x.is_finite() && *x > 0.0), "row update nonpositive weight");
    ensure!(positive.iter().chain(negative).all(|x|x.is_finite() && *x >= 0.0), "row update invalid flow");
    if alpha == 0.0 || reach == 0.0 || positive.iter().chain(negative).all(|x|*x == 0.0) {return Ok(weights.to_vec());}
    // Dividing numerator and denominator by eta avoids forming alpha/(reach*weight).
    let log_u = weights.iter().zip(positive).zip(negative).map(|((&w,&p),&n)| {
        w.ln()+(reach*w+alpha*p).ln()-(reach*w+alpha*n).ln()
    }).collect::<Vec<_>>();
    Ok(normalized(&log_u)?.iter().map(|x|x.exp()).collect())
}
fn update_checks() -> Result<Value> {
    let mut equations = Errors::default();
    let mut first_order = Vec::new();
    for (weights,positive,negative,reach) in [
        (vec![0.2,0.3,0.5],vec![0.1,0.8,0.2],vec![0.7,0.2,0.3],1.3),
        (vec![1e-180,0.1,0.9-1e-180],vec![0.1,0.0,2.0],vec![0.0,0.8,0.3],0.07),
        (vec![0.7,0.2,0.1],vec![0.0;3],vec![0.0;3],2.0),
        (vec![0.7,0.2,0.1],vec![0.2,0.6,0.3],vec![0.7,0.1,0.4],0.0),
    ] {
        for alpha in [0.0,0.05,7.0] {
            let result = row_update(&weights,&positive,&negative,reach,alpha)?;
            ensure!(result.iter().all(|x|x.is_finite() && *x > 0.0), "Eq9 lost positive support");
            equations.strict(&[result.iter().sum()],&[1.0],"Eq9 row normalization")?;
            if alpha == 0.0 || reach == 0.0 || positive.iter().chain(&negative).all(|x|*x == 0.0) {
                equations.strict(&result,&weights,"Eq9 unchanged inactive row")?;
            } else {
                let raw = weights.iter().zip(&positive).zip(&negative).map(|((&w,&p),&n)|w*(reach*w+alpha*p)/(reach*w+alpha*n)).collect::<Vec<_>>();
                let sum = raw.iter().sum::<f64>();
                equations.strict(&result,&raw.iter().map(|x|x/sum).collect::<Vec<_>>(),"Algorithm1 equivalence")?;
            }
        }
        if reach > 0.0 && weights.iter().all(|x|*x > 1e-4) {
            let signed = positive.iter().zip(&negative).map(|(p,n)|(p-n)/reach).collect::<Vec<_>>();
            let total = signed.iter().sum::<f64>();
            let expected = weights.iter().zip(&signed).map(|(w,g)|g-w*total).collect::<Vec<_>>();
            let mut previous = f64::INFINITY;
            for alpha in [1e-4,1e-5,1e-6] {
                let result = row_update(&weights,&positive,&negative,reach,alpha)?;
                let error = result.iter().zip(&weights).zip(&expected).map(|((&u,&w),&g)|((u-w)/alpha-g).abs()).fold(0.0,f64::max);
                ensure!(error <= previous*1.1+1e-9, "Eq9 first-order error did not decrease");
                previous = error;
                first_order.push(json!({"alpha":alpha,"maximum_derivative_error":error}));
            }
            ensure!(previous < 5e-4, "Eq9 additive first-order limit differs");
        }
    }
    Ok(json!({"equations":equations.report(),"first_order_limit":first_order,
        "includes_negative_signed_statistics":true,"includes_unreachable_rows":true,"includes_tiny_positive_weights":true}))
}

fn structural_checks(device: &Device) -> Result<Value> {
    use crate::codpula::PreparedCircuit;
    let mut normalization = Errors::default();
    let mut grouping = Errors::default();
    let mut lengths = Errors::default();
    for length in 1..=16 {
        let circuit = Circuit::fixture(3,3,length,1,7001+length as u64);
        let prepared = circuit.prepare()?;
        let unresolved = prepared.query(&vec![None;length])?;
        lengths.strict(&[unresolved.log_z],&[0.0],"actual length unconditional mass")?;
        for row in prepared.leaf.chunks_exact(3) {
            normalization.strict(&[row.iter().map(|x|x.exp()).sum()],&[1.0],"local leaf row")?;
        }
        for row in prepared.transition.chunks_exact(3) {
            normalization.strict(&[row.iter().map(|x|x.exp()).sum()],&[1.0],"modulated transition row")?;
        }
        normalization.strict(&[prepared.root.iter().map(|x|x.exp()).sum()],&[1.0],"root row")?;
    }
    for (k,group) in [(4,2),(32,4),(32,32)] {
        let mut circuit = Circuit::fixture(k,3,2,group,7193+k as u64);
        let original = circuit.prepare()?;
        circuit.gates.fill(0.0);
        let zero = circuit.prepare()?;
        grouping.strict(&zero.transition,&circuit.log_a,"zero gates recover base transition")?;
        grouping.strict(&original.root,&zero.root,"root is never modulated")?;
        let groups = k/group;
        for source in 0..k {
            for destination in 0..k {
                if destination/group == (destination+1).min(k-1)/group {
                    let next = (destination+1).min(k-1);
                    grouping.strict(&[original.transition[source*k+destination]-original.transition[source*k+next]],
                        &[circuit.log_a[source*k+destination]-circuit.log_a[source*k+next]],"same destination group shares gate")?;
                }
            }
        }
        for source_group in 0..groups {
            for destination_group in 0..groups {
                circuit.gates[source_group*groups+destination_group] = source_group as f64*0.7-1.1;
            }
        }
        grouping.strict(&circuit.prepare()?.transition,&zero.transition,"source-only gates cancel")?;
        if groups == 1 {
            grouping.strict(&original.transition,&zero.transition,"one-group gates cancel")?;
        }
    }
    let mut independent = Circuit::fixture(2,3,4,1,7211);
    let common_emission = independent.log_pi[..3].to_vec();
    independent.log_pi[3..6].copy_from_slice(&common_emission);
    let independence = independent.prepare()?;
    grouping.strict(&independence.query_scores(&[None,None,None,None],&[2,1,0,2])?,&[0.0;4],"independent outputs have zero dependence")?;
    grouping.strict(&independence.cavity_scores(&[Some(2),None,Some(0),Some(2)],&[2,1,0,2])?,&[0.0;4],"single candidate dependence is zero")?;
    let mut impossible = Circuit::fixture(2,2,2,1,7237);
    impossible.log_pi = vec![0.0,f64::NEG_INFINITY,0.0,f64::NEG_INFINITY];
    ensure!(impossible.prepare()?.query(&[None,Some(1)]).is_err(), "f64 impossible evidence was accepted");
    ensure!(impossible.prepare()?.query(&[Some(99),None]).is_err(), "f64 invalid token was accepted");
    let invalid = PreparedCircuit::from_log_potentials(
        float_tensor(&impossible.log_pi,[2,2],device),float_tensor(&impossible.log_a,[2,2],device),
        float_tensor(&impossible.log_root,[2],device),float_tensor(&impossible.logits,[1,2,2],device),
        float_tensor(&impossible.gates,[1,2,2],device),vec![2],1);
    let impossible_rejected = match invalid {
        Err(_) => true,
        Ok(prepared) => prepared.query(int_tensor(&[0,1],[1,2],device),bool_tensor(&[false,true],[1,2],device)).is_err(),
    };
    ensure!(impossible_rejected, "tensor impossible evidence was accepted");
    let ordinary = Circuit::fixture(2,2,2,1,7253);
    let prepared = PreparedCircuit::from_log_potentials(
        float_tensor(&ordinary.log_pi,[2,2],device),float_tensor(&ordinary.log_a,[2,2],device),
        float_tensor(&ordinary.log_root,[2],device),float_tensor(&ordinary.logits,[1,2,2],device),
        float_tensor(&ordinary.gates,[1,2,2],device),vec![2],1)?;
    ensure!(prepared.query(int_tensor(&[99,0],[1,2],device),bool_tensor(&[true,false],[1,2],device)).is_err(), "tensor invalid evidence token was accepted");
    let all_observed = prepared.conditional_nll(int_tensor(&[1,0],[1,2],device),bool_tensor(&[true,true],[1,2],device),1.0)?;
    let mut skipped = Errors::default();
    skipped.strict(&tensor_values(all_observed.loss),&[0.0],"all-observed loss")?;
    ensure!(all_observed.flows.active_blocks == 0, "all-observed row contributes circuit update");
    for values in [
        tensor_values(all_observed.flows.emission_positive),tensor_values(all_observed.flows.emission_negative),
        tensor_values(all_observed.flows.transition_positive),tensor_values(all_observed.flows.transition_negative),
        tensor_values(all_observed.flows.root_positive),tensor_values(all_observed.flows.root_negative),
        tensor_values(all_observed.flows.emission_reach),tensor_values(all_observed.flows.transition_reach),
        tensor_values(all_observed.flows.root_reach),
    ] {skipped.strict(&values,&vec![0.0;values.len()],"all-observed flow and traversal")?;}
    Ok(json!({"normalization":normalization.report(),"gate_grouping_and_root":grouping.report(),
        "lengths_1_through_16":lengths.report(),"all_observed":skipped.report(),
        "impossible_evidence_rejected":impossible_rejected,"invalid_token_rejected":true,
        "singleton_scope":"generic one-position chain math only; the model clean-logit singleton bypass belongs to model integration"}))
}

#[cfg(any(feature = "vulkan", feature = "cuda"))]
fn fused_checks(device: &Device) -> Result<Value> {
    use burn::tensor::activation::log_softmax;
    use delta_kernels::{
        codpula_chain_backward, codpula_chain_statistics, codpula_emission_flows, codpula_leaf_backward,
        codpula_leaf_normalizers, codpula_log_partition, codpula_token_marginals,
    };
    if !matches!(device.clone().inner().as_dispatch(),burn::backend::DispatchDevice::Cube(_)) {
        return Ok(json!({"executed":false,"reason":"fused circuit kernels require a CubeCL device; CPU uses the tensor path"}));
    }
    let device = device.clone().autodiff();
    let (n,l,k,v,group) = (3,3,32,257,4);
    let lengths = [3,2,0];
    let base = Circuit::fixture(k,v,l,group,8209);
    let mut logits = vec![51.0;n*l*v];
    let mut gates = Vec::new();
    let mut clean = vec![0;n*l];
    let mut observed = vec![0.0;n*l];
    let mut valid = vec![0.0;n*l];
    let mut expected_norm = vec![0.0;n*l*k];
    let mut expected_z = vec![0.0;n];
    let mut expected_gamma = vec![0.0;n*l*k];
    let mut expected_xi = vec![0.0;n*k*k];
    let mut expected_emission_counts = vec![0.0;k*v];
    let mut expected_marginal = vec![0.0;n*l*v];
    let mut expected_logits = vec![0.0;n*l*v];
    let mut expected_gates = vec![0.0;n*64];
    let mut expected_pi = vec![0.0;k*v];
    let mut expected_a = vec![0.0;k*k];
    let mut expected_root = vec![0.0;k];
    let mut expected_loss = 0.0;
    for row in 0..n {
        let mut circuit = base.clone();
        circuit.length = lengths[row].max(1);
        circuit.logits = (0..circuit.length*v).map(|i|(((i*7+row*11)%31) as f64/11.0-1.1) as f32 as f64).collect();
        circuit.gates = (0..64).map(|i|(((i*3+row*17)%23) as f64/19.0-0.5) as f32 as f64).collect();
        gates.extend_from_slice(&circuit.gates);
        if lengths[row] == 0 {continue;}
        let length = lengths[row];
        logits[row*l*v..row*l*v+length*v].copy_from_slice(&circuit.logits);
        let target = (0..length).map(|i|if i==0 {256} else {(i*83+row*17)%v}).collect::<Vec<_>>();
        let evidence = target.iter().enumerate().map(|(i,&x)| {
            clean[row*l+i] = x; valid[row*l+i] = 1.0;
            if i==1 {observed[row*l+i]=1.0;Some(x)} else {None}
        }).collect::<Vec<_>>();
        let prepared = circuit.prepare()?;
        let negative = prepared.query(&evidence)?;
        let positive = prepared.query(&target.iter().copied().map(Some).collect::<Vec<_>>())?;
        expected_norm[row*l*k..row*l*k+length*k].copy_from_slice(&prepared.normalizers);
        expected_z[row] = negative.log_z;
        for i in 0..length { for s in 0..k { for x in 0..v {
            expected_emission_counts[s*v+x] += negative.counts[(i*k+s)*v+x];
        }}}
        expected_gamma[row*l*k..row*l*k+length*k].copy_from_slice(&negative.state);
        expected_marginal[row*l*v..row*l*v+length*v].copy_from_slice(&negative.token);
        for i in 0..length-1 {for edge in 0..k*k {expected_xi[row*k*k+edge] += negative.edge[i*k*k+edge];}}
        expected_loss += negative.log_z-positive.log_z;
        let derivative = prepared.derivative(&positive,&negative,group);
        for i in 0..length*v {expected_logits[row*l*v+i] = -derivative.logits[i];}
        for i in 0..64 {expected_gates[row*64+i] = -derivative.gates[i];}
        for (out,input) in [(&mut expected_pi,&derivative.pi),(&mut expected_a,&derivative.a),(&mut expected_root,&derivative.root)] {
            for (out,&x) in out.iter_mut().zip(input) {*out -= x;}
        }
    }
    let pi = float_tensor(&base.log_pi,[k,v],&device).require_grad();
    let a = float_tensor(&base.log_a,[k,k],&device).require_grad();
    let root = float_tensor(&base.log_root,[k],&device).require_grad();
    let potentials = float_tensor(&logits,[n,l,v],&device).require_grad();
    let gate = float_tensor(&gates,[n,8,8],&device).require_grad();
    let valid_tensor = float_tensor(&valid,[n,l],&device);
    let observed_tensor = float_tensor(&observed,[n,l],&device);
    let normalizers = codpula_leaf_normalizers(potentials.clone(),pi.clone(),valid_tensor.clone());
    let log_a = log_softmax((a.clone().reshape([1,8,4,8,4])+gate.clone().reshape([n,8,1,8,1])).reshape([n,k,k]),2);
    let log_root = log_softmax(root.clone(),0);
    let token_index = int_tensor(&clean,[n,l],&device);
    let selected_logits = potentials.clone().gather(2,token_index.clone().unsqueeze_dim::<3>(2)).reshape([n,l,1]);
    let selected_pi = pi.clone().select(1,token_index.clone().reshape([n*l])).transpose().reshape([n,l,k]);
    let selected = selected_logits+selected_pi-normalizers.clone();
    let negative = selected.clone()*observed_tensor.clone().reshape([n,l,1]);
    let positive = selected*valid_tensor.clone().reshape([n,l,1]);
    let z_minus = codpula_log_partition(negative.clone(),log_a.clone(),log_root.clone(),valid_tensor.clone());
    let z_plus = codpula_log_partition(positive.clone(),log_a.clone(),log_root.clone(),valid_tensor.clone());
    let loss = (z_minus.clone()-z_plus).sum();
    let mut reports = Vec::new();
    for (name,actual,expected) in [
        ("leaf_log_normalizers",tensor_values(normalizers.clone()),expected_norm),
        ("evidence_log_mass",tensor_values(z_minus),expected_z.clone()),
        ("conditional_nll",tensor_values(loss.clone()),vec![expected_loss]),
    ] {
        let mut errors = Errors::default(); errors.tensor(&actual,&expected,name)?;
        reports.push(json!({"quantity":name,"errors":errors.report()}));
        if name == "evidence_log_mass" {
            let untracked = codpula_log_partition(negative.clone().detach(),log_a.clone().detach(),
                log_root.clone().detach(),valid_tensor.clone());
            let detached_loss = (untracked.clone()-codpula_log_partition(positive.clone().detach(),
                log_a.clone().detach(),log_root.clone().detach(),valid_tensor.clone())).sum();
            let mut detached_loss_errors = Errors::default();
            detached_loss_errors.tensor(&tensor_values(detached_loss),&[expected_loss],"untracked conditional NLL")?;
            reports.push(json!({"quantity":"untracked_conditional_nll","errors":detached_loss_errors.report()}));
            let single = codpula_log_partition(negative.clone().slice([0..1,0..l,0..k]).detach(),
                log_a.clone().slice([0..1,0..k,0..k]).detach(),log_root.clone().detach(),
                valid_tensor.clone().slice([0..1,0..l]));
            let mut single_errors = Errors::default();
            single_errors.tensor(&tensor_values(single),&[expected[0]],"untracked single-chain partition")?;
            reports.push(json!({"quantity":"untracked_single_chain_log_mass","errors":single_errors.report()}));
            let mut detached_errors = Errors::default();
            detached_errors.tensor(&tensor_values(untracked),&expected,"untracked evidence partition")?;
            reports.push(json!({"quantity":"untracked_evidence_log_mass","errors":detached_errors.report()}));
        }
    }
    let gradients = loss.backward();
    let mut derivatives = Vec::new();
    for (name,actual,expected) in [
        ("token_potentials",tensor_values(potentials.grad(&gradients).context("fused potential VJP")?),expected_logits),
        ("group_gates",tensor_values(gate.grad(&gradients).context("fused grouped gate VJP")?),expected_gates),
        ("base_emission",tensor_values(pi.grad(&gradients).context("fused base emission VJP")?),expected_pi),
        ("base_transition",tensor_values(a.grad(&gradients).context("fused base transition VJP")?),expected_a),
        ("normalized_root",tensor_values(root.grad(&gradients).context("fused root VJP")?),expected_root),
    ] {
        let mut errors = Errors::default(); errors.tensor(&actual,&expected,name)?;
        derivatives.push(json!({"quantity":name,"errors":errors.report(),"all_coordinates":true}));
    }
    let (z,gamma,xi) = codpula_chain_statistics(negative,log_a,log_root,valid_tensor.clone());
    let tokens = float_tensor(&clean.iter().map(|&x|x as f64).collect::<Vec<_>>(),[n,l],&device);
    let emission_counts = codpula_emission_flows(potentials.clone().detach(),pi.clone().detach(),normalizers.clone().detach(),
        gamma.clone(),tokens.clone(),observed_tensor.clone(),valid_tensor.clone());
    let marginals = codpula_token_marginals(potentials.clone().detach(),pi.clone().detach(),normalizers.clone().detach(),
        gamma.clone(),tokens,observed_tensor,valid_tensor.clone());
    for (name,actual,expected) in [
        ("statistics_log_mass",tensor_values(z),expected_z),
        ("state_posterior",tensor_values(gamma.clone()),expected_gamma.clone()),
        ("summed_edge_posterior",tensor_values(xi.clone()),expected_xi.clone()),
        ("full_vocabulary_marginals",tensor_values(marginals),expected_marginal),
        ("tied_emission_expected_counts",tensor_values(emission_counts),expected_emission_counts),
    ] {
        let mut errors = Errors::default(); errors.tensor(&actual,&expected,name)?;
        reports.push(json!({"quantity":name,"errors":errors.report()}));
    }
    let upstream = [0.7,-1.3,0.2];
    let (de,da,dr) = codpula_chain_backward(gamma,xi,float_tensor(&upstream,[n],&device));
    let expected_de = expected_gamma.iter().enumerate().map(|(i,x)|x*upstream[i/(l*k)]).collect::<Vec<_>>();
    let expected_da = expected_xi.iter().enumerate().map(|(i,x)|x*upstream[i/(k*k)]).collect::<Vec<_>>();
    let expected_dr = (0..k).map(|s|(0..n).map(|row|expected_gamma[row*l*k+s]*upstream[row]).sum()).collect::<Vec<_>>();
    for (name,actual,expected) in [
        ("explicit_chain_evidence_vjp",tensor_values(de),expected_de),
        ("explicit_chain_transition_vjp",tensor_values(da),expected_da),
        ("explicit_chain_root_vjp",tensor_values(dr),expected_dr),
    ] {
        let mut errors = Errors::default(); errors.tensor(&actual,&expected,name)?;
        derivatives.push(json!({"quantity":name,"errors":errors.report(),"all_coordinates":true}));
    }
    let norm_upstream = (0..n*l*k).map(|i|((i*17%29) as f64/13.0-1.0) as f32 as f64).collect::<Vec<_>>();
    let (dl,dp) = codpula_leaf_backward(potentials.detach(),pi.detach(),valid_tensor,normalizers.detach(),
        float_tensor(&norm_upstream,[n,l,k],&device));
    let mut expected_dl = vec![0.0;n*l*v];
    let mut expected_dp = vec![0.0;k*v];
    for row in 0..n { for i in 0..lengths[row] { for s in 0..k {
        let entries = (0..v).map(|x|base.log_pi[s*v+x]+logits[(row*l+i)*v+x]).collect::<Vec<_>>();
        let norm = log_sum(&entries);
        for x in 0..v {
            let adjoint = (entries[x]-norm).exp()*norm_upstream[(row*l+i)*k+s];
            expected_dl[(row*l+i)*v+x] += adjoint; expected_dp[s*v+x] += adjoint;
        }
    }}}
    for (name,actual,expected) in [
        ("explicit_leaf_potential_vjp",tensor_values(dl),expected_dl),
        ("explicit_leaf_base_vjp",tensor_values(dp),expected_dp),
    ] {
        let mut errors = Errors::default(); errors.tensor(&actual,&expected,name)?;
        derivatives.push(json!({"quantity":name,"errors":errors.report(),"all_coordinates":true}));
    }
    let prepared = tensor_case(&device,32,257,&[3,2,1,0],4,crate::codpula::CodpulaExecution::FusedGpu)?;
    let marginal = marginal_vjp(&device,crate::codpula::CodpulaExecution::FusedGpu)?;
    Ok(json!({"executed":true,"states":k,"vocabulary":v,"lengths":lengths,"group_size":group,
        "includes_bos_column":true,"forward":reports,"full_vjp":derivatives,
        "prepared_circuit":prepared,"conditional_marginal_vjp":marginal}))
}

#[cfg(not(any(feature = "vulkan", feature = "cuda")))]
fn fused_checks(_: &Device) -> Result<Value> {
    Ok(json!({"executed":false,"reason":"binary has no GPU kernel feature; CPU uses the tensor path"}))
}

fn gate_and_observed_gradients(device: &Device) -> Result<Value> {
    use crate::codpula::{CircuitState,Codpula};
    let device = device.clone().autodiff();
    let (n,l,d,v,k) = (2,3,3,257,32);
    let mut module = Codpula::new(d,9811,&device);
    let mut zero = Errors::default();
    let repeated = Codpula::new(d,9811,&device);
    let another = Codpula::new(d,9812,&device);
    let mut initialization = Errors::default();
    for (actual,expected,columns) in [
        (tensor_values(module.probability.log_emission.val()),tensor_values(repeated.probability.log_emission.val()),v),
        (tensor_values(module.probability.log_transition.val()),tensor_values(repeated.probability.log_transition.val()),k),
        (tensor_values(module.probability.log_root.val()),tensor_values(repeated.probability.log_root.val()),k),
    ] {
        initialization.strict(&actual,&expected,"same seed circuit initialization")?;
        for row in actual.chunks_exact(columns) {
            ensure!(row.iter().all(|x|x.is_finite() && (x+(columns as f64).ln()).abs() < 0.021), "circuit initialization is not a small positive near-uniform perturbation");
            initialization.tensor(&[row.iter().map(|x|x.exp()).sum()],&[1.0],"initialized normalized log row")?;
        }
    }
    ensure!(tensor_values(module.probability.log_emission.val()) != tensor_values(another.probability.log_emission.val()), "different circuit seeds produced identical rows");
    zero.strict(&tensor_values(module.gate_head.weight.val()),&vec![0.0;d*64],"zero gate weights")?;
    zero.strict(&tensor_values(module.gate_head.bias.as_ref().context("gate bias missing")?.val()),&vec![0.0;64],"zero gate bias")?;
    let initial_features = float_tensor(&[1.0,2.0,3.0,f64::NAN,f64::NAN,f64::NAN],[1,2,d],&device);
    zero.strict(&tensor_values(module.gates(initial_features,&[1])?),&vec![0.0;64],"zero gates ignore dirty padding")?;
    let weights = (0..d*64).map(|i|(((i*11%29) as f64/53.0)-0.2) as f32 as f64).collect::<Vec<_>>();
    let bias = (0..64).map(|i|(((i*7%17) as f64/41.0)-0.15) as f32 as f64).collect::<Vec<_>>();
    module.gate_head.weight = module.gate_head.weight.clone().map(|_|float_tensor(&weights,[d,64],&device).require_grad());
    let bias_parameter = module.gate_head.bias.take().context("gate bias missing")?;
    module.gate_head.bias = Some(bias_parameter.map(|_|float_tensor(&bias,[64],&device).require_grad()));
    let base = Circuit::fixture(k,v,l,4,9829);
    module.probability = CircuitState::from_logs(
        float_tensor(&base.log_pi,[k,v],&device),float_tensor(&base.log_a,[k,k],&device),float_tensor(&base.log_root,[k],&device));
    let lengths = [3,2];
    let mut features = (0..n*l*d).map(|i|(((i*19%31) as f64/13.0)-0.8) as f32 as f64).collect::<Vec<_>>();
    features[(l+2)*d..].fill(f64::NAN);
    let mut potentials = (0..n*l*v).map(|i|(((i*7%23) as f64/11.0)-1.1) as f32 as f64).collect::<Vec<_>>();
    potentials[(l+2)*v..].fill(f64::NAN);
    let clean = [256,11,93,17,145,999];
    let observed = [false,true,false,true,false,false];
    let temperature = 0.7;
    let denominator = 8.0;
    let mut expected_gates = vec![0.0;n*64];
    let mut expected_potential_gradient = vec![0.0;n*l*v];
    let mut expected_feature_gradient = vec![0.0;n*l*d];
    let mut expected_weight_gradient = vec![0.0;d*64];
    let mut expected_bias_gradient = vec![0.0;64];
    let mut expected_loss = 0.0;
    for row in 0..n {
        let length = lengths[row];
        let pooled = (0..d).map(|channel|(0..length).map(|i|features[(row*l+i)*d+channel]).sum::<f64>()/length as f64).collect::<Vec<_>>();
        for h in 0..64 {
            expected_gates[row*64+h] = bias[h]+(0..d).map(|channel|pooled[channel]*weights[channel*64+h]).sum::<f64>();
        }
        let mut circuit = base.clone();
        circuit.length = length;
        circuit.logits = potentials[row*l*v..row*l*v+length*v].iter().map(|x|x/temperature).collect();
        circuit.gates = expected_gates[row*64..(row+1)*64].to_vec();
        let prepared = circuit.prepare()?;
        let evidence = (0..length).map(|i|if observed[row*l+i] {Some(clean[row*l+i])} else {None}).collect::<Vec<_>>();
        let complete = (0..length).map(|i|Some(clean[row*l+i])).collect::<Vec<_>>();
        let positive = prepared.query(&complete)?;
        let negative = prepared.query(&evidence)?;
        expected_loss += (negative.log_z-positive.log_z)/denominator;
        let derivative = prepared.derivative(&positive,&negative,4);
        for i in 0..length { for x in 0..v {
            if !observed[row*l+i] {
                expected_potential_gradient[(row*l+i)*v+x] = -derivative.logits[i*v+x]/(denominator*temperature);
            }
        }}
        for h in 0..64 {
            let g = -derivative.gates[h]/denominator;
            expected_bias_gradient[h] += g;
            for channel in 0..d {
                expected_weight_gradient[channel*64+h] += pooled[channel]*g;
                for i in 0..length {expected_feature_gradient[(row*l+i)*d+channel] += weights[channel*64+h]*g/length as f64;}
            }
        }
    }
    let features = float_tensor(&features,[n,l,d],&device).require_grad();
    let potentials = float_tensor(&potentials,[n,l,v],&device).require_grad();
    let gates = module.gates(features.clone(),&lengths)?;
    let mut forward = Errors::default(); forward.tensor(&tensor_values(gates),&expected_gates,"valid mean pooled biased linear gates")?;
    let prepared = module.prepare(potentials.clone(),features.clone(),&lengths,bool_tensor(&observed,[n,l],&device),temperature)?;
    let result = prepared.conditional_nll(int_tensor(&clean,[n,l],&device),bool_tensor(&observed,[n,l],&device),denominator)?;
    let mut loss = Errors::default(); loss.tensor(&tensor_values(result.loss.clone()),&[expected_loss],"temperature applied before leaf normalization")?;
    let gradients = result.loss.backward();
    let feature_gradient = tensor_values(features.grad(&gradients).context("feature gate gradient missing")?);
    let observed_feature_maximum = observed.iter().enumerate().filter(|(_,fixed)|**fixed)
        .flat_map(|(i,_)|feature_gradient[i*d..(i+1)*d].iter().copied()).map(f64::abs).fold(0.0,f64::max);
    ensure!(observed_feature_maximum > 1e-9, "observed features stopped training the gate path");
    let mut reports = Vec::new();
    for (name,actual,expected) in [
        ("observed_masked_token_potentials",tensor_values(potentials.grad(&gradients).context("production potential gradient missing")?),expected_potential_gradient),
        ("valid_features_including_observed",feature_gradient,expected_feature_gradient),
        ("gate_head_weight",tensor_values(module.gate_head.weight.val().grad(&gradients).context("gate weight gradient missing")?),expected_weight_gradient),
        ("gate_head_bias",tensor_values(module.gate_head.bias.as_ref().unwrap().val().grad(&gradients).context("gate bias gradient missing")?),expected_bias_gradient),
    ] {
        let mut errors = Errors::default(); errors.tensor(&actual,&expected,name)?;
        reports.push(json!({"quantity":name,"errors":errors.report()}));
    }
    Ok(json!({"zero_initialization":zero.report(),"probability_initialization":initialization.report(),
        "pooled_gate_outputs":forward.report(),"conditional_nll":loss.report(),
        "temperature":temperature,"observed_feature_maximum_gradient":observed_feature_maximum,
        "dirty_padding":"NaN features/potentials and invalid padded clean id","full_vjp":reports}))
}

fn tensor_update_checks(device: &Device) -> Result<Value> {
    use crate::codpula::{CircuitFlows,CircuitState};
    let base = Circuit::fixture(3,3,2,1,10103);
    let mut state = CircuitState::from_logs(float_tensor(&base.log_pi,[3,3],device),
        float_tensor(&base.log_a,[3,3],device),float_tensor(&base.log_root,[3],device));
    let mut probability = [
        state.log_emission.val().into_data().try_to_vec::<f32>().unwrap().into_iter().map(|x|f64::from(x).exp()).collect::<Vec<_>>(),
        state.log_transition.val().into_data().try_to_vec::<f32>().unwrap().into_iter().map(|x|f64::from(x).exp()).collect::<Vec<_>>(),
        state.log_root.val().into_data().try_to_vec::<f32>().unwrap().into_iter().map(|x|f64::from(x).exp()).collect::<Vec<_>>(),
    ];
    let mut positive_ema = [vec![0.0;9],vec![0.0;9],vec![0.0;3]];
    let mut negative_ema = [vec![0.0;9],vec![0.0;9],vec![0.0;3]];
    let mut reports = Vec::new();
    for step in 1..=4 {
        let active = if step%2 == 0 {3} else {2};
        let positive: [Vec<f64>;3] = std::array::from_fn(|field| (0..probability[field].len()).map(|i| {
            if field == 0 && i < 3 {0.0} else {(((i*7+step*3+field)%19) as f64/11.0) as f32 as f64}
        }).collect());
        let negative: [Vec<f64>;3] = std::array::from_fn(|field| (0..probability[field].len()).map(|i| {
            if field == 0 && i < 3 {0.0} else {(((i*13+step+field*7)%23) as f64/13.0) as f32 as f64}
        }).collect());
        let reach = [vec![0.4,0.0,1.7],vec![0.7,0.9,0.0],vec![1.0]];
        for field in 0..3 {
            for i in 0..probability[field].len() {
                positive_ema[field][i] = 0.9*positive_ema[field][i]+0.1*positive[field][i]/active as f64;
                negative_ema[field][i] = 0.9*negative_ema[field][i]+0.1*negative[field][i]/active as f64;
            }
            let correction = 1.0-0.9f64.powi(step as i32);
            for row in 0..reach[field].len() {
                let range = row*3..(row+1)*3;
                let p = positive_ema[field][range.clone()].iter().map(|x|x/correction).collect::<Vec<_>>();
                let m = negative_ema[field][range.clone()].iter().map(|x|x/correction).collect::<Vec<_>>();
                let updated = row_update(&probability[field][range.clone()],&p,&m,reach[field][row]/active as f64,0.05)?;
                probability[field][range].copy_from_slice(&updated);
            }
        }
        let flows = CircuitFlows {
            emission_positive:float_tensor(&positive[0],[3,3],device),emission_negative:float_tensor(&negative[0],[3,3],device),
            transition_positive:float_tensor(&positive[1],[3,3],device),transition_negative:float_tensor(&negative[1],[3,3],device),
            root_positive:float_tensor(&positive[2],[3],device),root_negative:float_tensor(&negative[2],[3],device),
            emission_reach:float_tensor(&reach[0],[3],device),transition_reach:float_tensor(&reach[1],[3],device),
            root_reach:float_tensor(&reach[2],[1],device),active_blocks:active,
        };
        state.update(flows)?;
        let actual_rows = [tensor_values(state.log_emission.val()),tensor_values(state.log_transition.val()),tensor_values(state.log_root.val())];
        let actual_positive = [tensor_values(state.emission_positive_ema.val()),tensor_values(state.transition_positive_ema.val()),tensor_values(state.root_positive_ema.val())];
        let actual_negative = [tensor_values(state.emission_negative_ema.val()),tensor_values(state.transition_negative_ema.val()),tensor_values(state.root_negative_ema.val())];
        let mut rows = Errors::default();
        let mut plus = Errors::default();
        let mut minus = Errors::default();
        for field in 0..3 {
            rows.tensor(&actual_rows[field],&probability[field].iter().map(|x|x.ln()).collect::<Vec<_>>(),"updated normalized log rows")?;
            plus.tensor(&actual_positive[field],&positive_ema[field],"positive EMA")?;
            minus.tensor(&actual_negative[field],&negative_ema[field],"negative EMA")?;
            for row in actual_rows[field].chunks_exact(3) {
                ensure!(row.iter().all(|x|x.is_finite() && x.exp() > 0.0), "updated row support lost");
                rows.tensor(&[row.iter().map(|x|x.exp()).sum()],&[1.0],"updated row sum")?;
            }
        }
        let time = state.step.val().into_data().try_to_vec_as::<i32>().unwrap();
        ensure!(time == vec![step as i32], "circuit update time differs");
        reports.push(json!({"step":step,"active_blocks":active,"rows":rows.report(),"positive_ema":plus.report(),"negative_ema":minus.report()}));
    }
    let before = [tensor_values(state.log_emission.val()),tensor_values(state.log_transition.val()),tensor_values(state.log_root.val())];
    state.update(CircuitFlows {
        emission_positive:Tensor::zeros([3,3],device),emission_negative:Tensor::zeros([3,3],device),
        transition_positive:Tensor::zeros([3,3],device),transition_negative:Tensor::zeros([3,3],device),
        root_positive:Tensor::zeros([3],device),root_negative:Tensor::zeros([3],device),
        emission_reach:Tensor::zeros([3],device),transition_reach:Tensor::zeros([3],device),
        root_reach:Tensor::zeros([1],device),active_blocks:0,
    })?;
    ensure!(before == [tensor_values(state.log_emission.val()),tensor_values(state.log_transition.val()),tensor_values(state.log_root.val())], "inactive update changed circuit rows");
    ensure!(state.step.val().into_data().try_to_vec_as::<i32>().unwrap() == vec![4], "inactive update advanced circuit time");
    Ok(json!({"momentum":0.9,"bias_correction":"1 - 0.9^step","alpha":0.05,
        "flows_averaged_over_active_blocks":true,"traversal_sums_include_tied_occurrences":true,
        "includes_zero_statistics_and_unreachable_rows":true,"steps":reports}))
}

fn inverse_cdf(logs: &[f64], uniform: f64) -> Result<usize> {
    ensure!(uniform.is_finite() && (0.0..1.0).contains(&uniform), "invalid reference sampling uniform");
    ensure!(logs.iter().all(|x|x.is_finite() || *x == f64::NEG_INFINITY), "invalid reference sampling log mass");
    let mut cumulative = 0.0;
    let mut last = None;
    for (value,&log) in logs.iter().enumerate() {
        let probability = log.exp();
        if probability > 0.0 {
            last = Some(value);
            cumulative += probability;
            if cumulative > uniform {return Ok(value);}
        }
    }
    last.context("reference categorical has no supported class")
}

fn enumerated_uniform_sample(circuit: &Circuit, evidence: &[Option<usize>], uniforms: &[f64]) -> Result<Vec<usize>> {
    let (k,v,l) = (circuit.states,circuit.vocab,circuit.length);
    ensure!(uniforms.len() >= 2*l, "reference indexed uniform shape");
    let prepared = circuit.prepare()?;
    let enumerated = Enumeration::new(&prepared);
    let mut latent = vec![0;l];
    for position in (0..l).rev() {
        let mut masses = vec![0.0;k];
        for assignment in &enumerated.assignments {
            if evidence.iter().enumerate().any(|(i,x)|x.is_some_and(|x|x != assignment.output[i])) {continue;}
            if (position+1..l).any(|i|assignment.latent[i] != latent[i]) {continue;}
            masses[assignment.latent[position]] += assignment.mass;
        }
        let total = masses.iter().sum::<f64>();
        ensure!(total > 0.0, "enumerated FFBS suffix has zero mass");
        latent[position] = inverse_cdf(&masses.iter().map(|x|(x/total).ln()).collect::<Vec<_>>(),uniforms[position*2])?;
    }
    (0..l).map(|position| {
        if let Some(value) = evidence[position] {Ok(value)} else {
            inverse_cdf(&prepared.leaf[(position*k+latent[position])*v..(position*k+latent[position]+1)*v],uniforms[position*2+1])
        }
    }).collect()
}

fn indexed_uniform_regressions(device: &Device) -> Result<Value> {
    use crate::codpula::{PreparedCircuit,sample_log_categorical};
    let mut categorical_cases = Vec::new();
    for (logs,uniform,expected) in [
        (vec![0.25f64.ln(),f64::NEG_INFINITY,0.75f64.ln()],0.25,2),
        (vec![0.25f64.ln(),f64::NEG_INFINITY,0.75f64.ln()],0.0,0),
        (vec![f64::NEG_INFINITY,0.2f64.ln(),0.7f64.ln()],0.99,2),
        (vec![f64::NEG_INFINITY,0.0,f64::NEG_INFINITY],0.0,1),
        (vec![f64::NEG_INFINITY,0.0,f64::NEG_INFINITY],1.0-f64::EPSILON,1),
    ] {
        let actual = sample_log_categorical(&logs,uniform)?;
        ensure!(actual == expected && actual == inverse_cdf(&logs,uniform)?,
            "categorical strict CDF/zero-support regression: u={uniform}, actual={actual}, expected={expected}");
        categorical_cases.push(json!({"uniform":uniform,"selected":actual,"reference":expected}));
    }
    for uniform in [-0.01,1.0,f64::NAN,f64::INFINITY] {
        ensure!(sample_log_categorical(&[0.0],uniform).is_err(), "categorical helper accepted invalid uniform");
    }
    let zero_probabilities = [0.0f64;3];
    let zero_log_weights = zero_probabilities.map(f64::ln);
    ensure!(zero_log_weights.iter().all(|x|*x == f64::NEG_INFINITY), "zero probabilities must map to negative-infinity log weights");
    let zero_mass_error = sample_log_categorical(&zero_log_weights,0.5)
        .expect_err("all-zero posterior must not produce a token or NaN-normalized distribution").to_string();
    ensure!(zero_mass_error.contains("zero support"), "all-zero posterior sampling error is unclear: {zero_mass_error}");
    for logs in [vec![],vec![f64::NAN,0.0],vec![f64::INFINITY,0.0]] {
        ensure!(sample_log_categorical(&logs,0.5).is_err(), "categorical helper fabricated a choice from invalid support");
    }
    let (n,width,k,v) = (3,4,2,3);
    let lengths = [3,2,0];
    let base = Circuit::fixture(k,v,3,1,12007);
    let mut logits = vec![f64::NAN;n*width*v];
    let mut gates = vec![f64::NAN;n*k*k];
    let values = [999,2,997,996,0,995,994,993,992,991,990,989];
    let fixed = [false,true,false,true,true,false,true,true,true,true,true,true];
    let uniforms = [
        0.0,0.15,0.33,f64::NAN,0.99,0.67,f64::NAN,f64::NAN,
        0.95,f64::NAN,0.5,0.2,f64::NAN,f64::NAN,f64::NAN,f64::NAN,
        f64::NAN,f64::NAN,f64::NAN,f64::NAN,f64::NAN,f64::NAN,f64::NAN,f64::NAN,
    ].map(|x|x as f32 as f64);
    let mut expected = Vec::new();
    for (row,&length) in lengths.iter().enumerate() {
        if length == 0 {expected.push(Vec::new());continue;}
        let mut circuit = base.clone();
        circuit.length = length;
        circuit.logits = (0..length*v).map(|i|(((i*17+row*11)%23) as f64/9.0-1.0) as f32 as f64).collect();
        if row == 1 {circuit.logits[1] = f64::NEG_INFINITY;}
        circuit.gates = (0..k*k).map(|i|(((i*7+row*3)%13) as f64/11.0-0.5) as f32 as f64).collect();
        logits[row*width*v..row*width*v+length*v].copy_from_slice(&circuit.logits);
        gates[row*k*k..(row+1)*k*k].copy_from_slice(&circuit.gates);
        let evidence = (0..length).map(|i|if fixed[row*width+i] {Some(values[row*width+i])} else {None}).collect::<Vec<_>>();
        expected.push(enumerated_uniform_sample(&circuit,&evidence,&uniforms[row*width*2..(row+1)*width*2])?);
    }
    let prepared = PreparedCircuit::from_log_potentials(
        float_tensor(&base.log_pi,[k,v],device),float_tensor(&base.log_a,[k,k],device),
        float_tensor(&base.log_root,[k],device),float_tensor(&logits,[n,width,v],device),
        float_tensor(&gates,[n,k,k],device),lengths.to_vec(),1)?;
    let supplied = int_tensor(&values,[n,width],device);
    let known = bool_tensor(&fixed,[n,width],device);
    let explicit = float_tensor(&uniforms,[n,width,2],device);
    let actual = prepared.sample_with_uniforms(supplied.clone(),known.clone(),explicit.clone())?;
    ensure!(actual == expected, "explicit FFBS uniform mapping differs: actual={actual:?}, enumerated={expected:?}");
    ensure!(prepared.sample_with_uniforms(supplied.clone(),known.clone(),explicit)? == actual, "explicit uniform sampler is not reproducible");
    for row in 0..2 {
        let one = PreparedCircuit::from_log_potentials(
            float_tensor(&base.log_pi,[k,v],device),float_tensor(&base.log_a,[k,k],device),
            float_tensor(&base.log_root,[k],device),float_tensor(&logits[row*width*v..(row+1)*width*v],[1,width,v],device),
            float_tensor(&gates[row*k*k..(row+1)*k*k],[1,k,k],device),vec![lengths[row]],1)?;
        let sampled = one.sample_with_uniforms(
            int_tensor(&values[row*width..(row+1)*width],[1,width],device),
            bool_tensor(&fixed[row*width..(row+1)*width],[1,width],device),
            float_tensor(&uniforms[row*width*2..(row+1)*width*2],[1,width,2],device))?;
        ensure!(sampled == vec![actual[row].clone()], "indexed uniforms changed when another batch row was omitted");
    }
    for seed in [0,19,u64::MAX] {
        ensure!(prepared.sample(supplied.clone(),known.clone(),seed)? == prepared.sample(supplied.clone(),known.clone(),seed)?,
            "seeded joint sampler is not reproducible for seed {seed}");
    }
    for invalid in [f64::NAN,f64::INFINITY,-0.01,1.0] {
        for channel in 0..2 {
            let mut bad = uniforms.to_vec();
            bad[channel] = invalid;
            ensure!(prepared.sample_with_uniforms(supplied.clone(),known.clone(),float_tensor(&bad,[n,width,2],device)).is_err(),
                "invalid actual-position sampling uniform was accepted");
        }
    }
    Ok(json!({"passed":true,"categorical_cases":categorical_cases,"lengths":lengths,"actual":actual,"enumerated_ffbs_reference":expected,
        "uniform_channels":["latent_state","token"],"strict_uniform_range":"0 <= u < 1",
        "zero_mass_rejection":{"input_probabilities":zero_probabilities,"all_log_weights_negative_infinity":true,"error":zero_mass_error},
        "same_uniforms_same_sample":true,"omitted_batch_rows_preserve_samples":true,
        "unknown_ids_and_padded_nan_uniforms_ignored":true,"observed_token_uniform_unused":true,
        "seed_reproducibility":[0,19,u64::MAX],"invalid_uniforms_rejected":true}))
}

fn support_and_dependence_regressions(device: &Device) -> Result<Value> {
    use crate::codpula::PreparedCircuit;
    let build = |circuit: &Circuit| {
        PreparedCircuit::from_log_potentials(
            float_tensor(&circuit.log_pi,[circuit.states,circuit.vocab],device),
            float_tensor(&circuit.log_a,[circuit.states,circuit.states],device),
            float_tensor(&circuit.log_root,[circuit.states],device),
            float_tensor(&circuit.logits,[1,circuit.length,circuit.vocab],device),
            float_tensor(&circuit.gates,[1,circuit.states/circuit.group,circuit.states/circuit.group],device),
            vec![circuit.length],circuit.group)
    };
    let mut singleton = Circuit::fixture(2,3,3,1,11903);
    singleton.logits = vec![f64::NEG_INFINITY,0.0,f64::NEG_INFINITY,
        f64::NEG_INFINITY,f64::NEG_INFINITY,0.0,0.0,f64::NEG_INFINITY,f64::NEG_INFINITY];
    let prepared = build(&singleton).context("single supported token at each valid position")?;
    let fixed = bool_tensor(&[false;3],[1,3],device);
    let unknown = int_tensor(&[999,998,997],[1,3],device);
    let complete = int_tensor(&[1,2,0],[1,3],device);
    let reference = singleton.prepare()?;
    let enumerated = Enumeration::new(&reference);
    let mut support = Errors::default();
    support.tensor(&tensor_values(prepared.query(unknown.clone(),fixed.clone())?.log_z),&[0.0],"singleton-support unconditional mass")?;
    support.tensor(&tensor_values(prepared.query(complete.clone(),bool_tensor(&[true;3],[1,3],device))?.log_z),
        &[enumerated.query(&[Some(1),Some(2),Some(0)])?.log_z],"singleton-support full assignment mass")?;
    support.tensor(&tensor_values(prepared.conditional_nll(complete.clone(),fixed.clone(),1.0)?.loss),&[0.0],"singleton-support joint NLL")?;
    let marginals = prepared.marginals(unknown.clone(),fixed.clone())?;
    ensure!(!marginals.is_tracked(), "support readout unexpectedly retains an autodiff graph");
    support.tensor(&tensor_values(marginals),&[0.0,1.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0],"singleton-support point masses")?;
    for seed in [0,1,9,719,u64::MAX] {
        ensure!(prepared.sample(unknown.clone(),fixed.clone(),seed)? == vec![vec![1,2,0]], "singleton-support sampler changed a deterministic token");
    }
    let mut impossible = singleton.clone();
    impossible.logits = [0.0,f64::NEG_INFINITY,-1.0].repeat(3);
    let prepared = build(&impossible)?;
    let impossible_values = int_tensor(&[1,0,0],[1,3],device);
    let impossible_fixed = bool_tensor(&[true,false,false],[1,3],device);
    ensure!(prepared.query(impossible_values.clone(),impossible_fixed.clone()).is_err(), "zero-mass evidence query fabricated a distribution");
    ensure!(prepared.marginals(impossible_values.clone(),impossible_fixed.clone()).is_err(), "zero-mass evidence fabricated a point mass");
    ensure!(prepared.sample(impossible_values.clone(),impossible_fixed.clone(),17).is_err(), "zero-mass evidence produced a sample");
    ensure!(prepared.dependence_scores(impossible_values.clone(),impossible_fixed.clone(),&[vec![1,0,0]]).is_err(), "zero-mass evidence produced a dependence score");
    ensure!(prepared.conditional_nll(impossible_values.clone(),impossible_fixed,1.0).is_err(), "zero-mass observed evidence produced successful loss");
    ensure!(prepared.conditional_nll(impossible_values,fixed.clone(),1.0).is_err(), "zero-mass gold assignment produced successful loss");
    let mut empty = impossible.clone();
    empty.logits[3..6].fill(f64::NEG_INFINITY);
    ensure!(build(&empty).is_err(), "empty valid support was normalized into a distribution");
    let bad_values = int_tensor(&[i32::MAX as usize,0,0],[1,3],device);
    let known = bool_tensor(&[true,false,false],[1,3],device);
    ensure!(prepared.query(bad_values.clone(),known.clone()).is_err(), "out-of-axis known evidence reached gather");
    ensure!(prepared.sample(bad_values,known,17).is_err(), "out-of-axis known evidence reached sampler");
    let mut dependent = Circuit::fixture(2,2,2,1,11927);
    dependent.log_pi = [0.96f64,0.04,0.04,0.96].map(f64::ln).to_vec();
    dependent.log_a = [0.97f64,0.03,0.03,0.97].map(f64::ln).to_vec();
    dependent.log_root = vec![-2.0f64.ln();2];
    dependent.logits.fill(0.0);
    dependent.gates.fill(0.0);
    let reference = Enumeration::new(&dependent.prepare()?);
    let base = reference.query(&[None,None])?.log_z;
    let signed = reference.query(&[Some(0),Some(1)])?.log_z
        -reference.query(&[Some(0),None])?.log_z-reference.query(&[None,Some(1)])?.log_z+base;
    ensure!(signed < -0.01, "dependence fixture does not exercise a negative signed log change");
    let prepared = build(&dependent)?;
    let actual = prepared.dependence_scores(int_tensor(&[999,999],[1,2],device),
        bool_tensor(&[false,false],[1,2],device),&[vec![0,1]])?;
    let mut dependence = Errors::default();
    dependence.tensor(&actual[0],&[-signed,-signed],"absolute rather than signed dependence")?;
    let singleton_score = prepared.dependence_scores(int_tensor(&[0,1],[1,2],device),
        bool_tensor(&[false,true],[1,2],device),&[vec![0,1]])?;
    ensure!(singleton_score == vec![vec![0.0,0.0]], "positive-mass singleton dependence is not exactly zero");
    Ok(json!({"support":support.report(),"singleton_support_seeds":[0,1,9,719,u64::MAX],
        "impossible_evidence_rejected_by":["query","marginals","sample","dependence_scores","conditional_nll"],
        "impossible_gold_rejected":true,"empty_valid_support_rejected":true,"unknown_placeholder_ids_ignored":true,
        "known_invalid_ids_rejected":true,"negative_signed_log_change":signed,
        "absolute_dependence":dependence.report(),"singleton_dependence_exact_zero":true}))
}

/// Numerical acceptance checks only; this function performs no training or timing experiment.
/// A failed quantity returns its actual/reference values, absolute error and tolerance ratio.
pub fn numerical_checks(device: &Device) -> Result<Value> {
    let tensor_execution = crate::codpula::CodpulaExecution::Tensor;
    let enumeration = enumeration_checks().context("independent exhaustive f64 checks")?;
    let structure = structural_checks(device).context("normalization, lengths, support and grouping")?;
    let support_regressions = support_and_dependence_regressions(device).context("adapted zero-support, evidence and signed-dependence regressions")?;
    let uniform_regressions = indexed_uniform_regressions(device).context("adapted explicit-uniform FFBS sampler regressions")?;
    let mut tensor = vec![
        tensor_case(device,2,3,&[4,2,3,0],1,tensor_execution)?,
        tensor_case(device,4,3,&[3,2,1,0],2,tensor_execution)?,
        tensor_case(device,4,3,&[3,2,1,0],4,tensor_execution)?,
        tensor_case(device,32,257,&[3,2,1,0],4,tensor_execution)?,
    ];
    for length in 2..=16 {
        tensor.push(tensor_case(device,3,3,&[length,1,0],1,tensor_execution)
            .with_context(||format!("actual-length tensor check L={length}"))?);
    }
    let marginal = marginal_vjp(device,tensor_execution).context("conditional marginal full finite-difference VJP")?;
    let sampling = sampling_checks(device).context("joint FFBS distribution")?;
    let update = update_checks().context("independent Eq9 rows")?;
    let state_update = tensor_update_checks(device).context("persistent separate-flow circuit update")?;
    let gates = gate_and_observed_gradients(device).context("valid-state gate pooling and observed derivative paths")?;
    let fused = fused_checks(device).context("real K32 V257 fused forward and full VJP")?;
    Ok(json!({"passed":true,"device":crate::runtime::identity(device),
        "absolute_tolerance":ABSOLUTE_TOLERANCE,"relative_tolerance":RELATIVE_TOLERANCE,
        "enumeration":enumeration,"structure":structure,"tensor":tensor,
        "marginal_vjp":marginal,"sampling":sampling,"eq9":update,"state_update":state_update,
        "neural_gates":gates,"fused":fused,
        "support_and_dependence_regressions":support_regressions,
        "indexed_uniform_regressions":uniform_regressions,
        "integration_scope":"model final-RMSNorm provenance, clean singleton bypass, soft-feedback detach/embedding, decoder commitment ties and checkpoint/model optimizer integration belong to the model integration checks"}))
}
