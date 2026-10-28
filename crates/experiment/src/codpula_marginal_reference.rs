//! Independent exhaustive f64 oracle for the weighted singleton objective.
use super::*;
use crate::codpula::CircuitState;

fn reference(c: &Circuit, evidence: &[Option<usize>], gold: &[usize], weights: &[f64], denominator: f64) -> Result<(f64, Flow, Derivative)> {
    let prepared = c.prepare()?;
    let enumeration = Enumeration::new(&prepared);
    let base = enumeration.query(evidence)?;
    let mut flow = prepared.flows(&base, &base);
    for values in [&mut flow.pi_plus, &mut flow.pi_minus, &mut flow.a_plus, &mut flow.a_minus, &mut flow.root_plus, &mut flow.root_minus] { values.fill(0.0); }
    let mut derivative = prepared.derivative(&base, &base, c.group);
    let mut loss = 0.0;
    for i in 0..c.length {
        if evidence[i].is_some() || weights[i] == 0.0 { continue; }
        let mut selected = evidence.to_vec();
        selected[i] = Some(gold[i]);
        let selected = enumeration.query(&selected)?;
        loss += weights[i] * (base.log_z - selected.log_z) / denominator;
        let term = prepared.flows(&selected, &base);
        for (total, part) in [(&mut flow.pi_plus,&term.pi_plus),(&mut flow.pi_minus,&term.pi_minus),
            (&mut flow.a_plus,&term.a_plus),(&mut flow.a_minus,&term.a_minus),
            (&mut flow.root_plus,&term.root_plus),(&mut flow.root_minus,&term.root_minus)] {
            for (sum, value) in total.iter_mut().zip(part) { *sum += weights[i] * value; }
        }
        let term = prepared.derivative(&selected, &base, c.group);
        for (total, part) in [(&mut derivative.logits,&term.logits),(&mut derivative.gates,&term.gates),
            (&mut derivative.pi,&term.pi),(&mut derivative.a,&term.a),(&mut derivative.root,&term.root)] {
            for (sum, value) in total.iter_mut().zip(part) { *sum -= weights[i] * value / denominator; }
        }
    }
    Ok((loss, flow, derivative))
}

pub(super) fn checks(device: &Device) -> Result<Value> {
    let device = device.clone().autodiff();
    let (rows,width,k,v) = (3,3,2,3);
    let denominator = 2.75;
    let mut first = Circuit::fixture(k,v,3,1,6021);
    first.root_gates = vec![0.3,-0.2];
    let mut second = Circuit::fixture(k,v,2,1,6022);
    second.log_pi = first.log_pi.clone(); second.log_a = first.log_a.clone(); second.log_root = first.log_root.clone();
    second.root_gates = vec![-0.4,0.1];
    let circuits = [first,second];
    let evidence = [vec![Some(1),None,None],vec![None,Some(2)]];
    let gold = [vec![0,2,1],vec![1,0]];
    let weights = [vec![2.0,0.25,1.5],vec![0.75,4.0]];
    let mut expected_loss = 0.0;
    let mut expected_flows = vec![vec![0.0;k*v],vec![0.0;k*v],vec![0.0;k*k],vec![0.0;k*k],vec![0.0;k],vec![0.0;k],vec![0.0;k],vec![0.0;k],vec![0.0;1]];
    let mut expected_gradients = vec![vec![0.0;rows*width*v],vec![0.0;rows*k*k],vec![0.0;k*v],vec![0.0;k*k],vec![0.0;k],vec![0.0;rows*k]];
    let mut raw_logits = vec![f64::NAN;rows*width*v];
    let mut raw_gates = vec![f64::NAN;rows*k*k];
    let mut raw_roots = vec![f64::NAN;rows*k];
    let mut fixed = vec![false;rows*width];
    let mut supplied = vec![99;rows*width];
    let mut labels = vec![99;rows*width];
    let mut row_weights = vec![f64::NAN;rows*width];
    let mut differences = Errors::default();
    for (row,c) in circuits.iter().enumerate() {
        raw_logits[row*width*v..row*width*v+c.length*v].copy_from_slice(&c.logits);
        raw_gates[row*k*k..(row+1)*k*k].copy_from_slice(&c.gates);
        raw_roots[row*k..(row+1)*k].copy_from_slice(&c.root_gates);
        for i in 0..c.length {
            fixed[row*width+i] = evidence[row][i].is_some();
            supplied[row*width+i] = evidence[row][i].unwrap_or(0);
            labels[row*width+i] = gold[row][i];
            row_weights[row*width+i] = weights[row][i];
        }
        let (loss, flow, derivative) = reference(c,&evidence[row],&gold[row],&weights[row],denominator)?;
        expected_loss += loss;
        let root_reach = [flow.root_reach];
        let parts: [&[f64];9] = [&flow.pi_plus,&flow.pi_minus,&flow.a_plus,&flow.a_minus,&flow.root_plus,&flow.root_minus,&flow.pi_reach,&flow.a_reach,&root_reach];
        for (total, part) in expected_flows.iter_mut().zip(parts) {
            for (sum,value) in total.iter_mut().zip(part) { *sum += value; }
        }
        expected_gradients[0][row*width*v..row*width*v+c.length*v].copy_from_slice(&derivative.logits);
        expected_gradients[1][row*k*k..(row+1)*k*k].copy_from_slice(&derivative.gates);
        expected_gradients[5][row*k..(row+1)*k].copy_from_slice(&derivative.root);
        for (total, part) in expected_gradients[2..5].iter_mut().zip([&derivative.pi,&derivative.a,&derivative.root]) {
            for (sum,value) in total.iter_mut().zip(part) { *sum += value; }
        }
        for field in 0..6 {
            let analytic = match field {0=>&derivative.logits,1=>&derivative.gates,2=>&derivative.pi,3=>&derivative.a,_=>&derivative.root};
            let mut finite = Vec::new();
            for coordinate in 0..analytic.len() {
                let mut plus=c.clone(); plus.field(field)[coordinate] += 1e-5;
                let mut minus=c.clone(); minus.field(field)[coordinate] -= 1e-5;
                finite.push((reference(&plus,&evidence[row],&gold[row],&weights[row],denominator)?.0-reference(&minus,&evidence[row],&gold[row],&weights[row],denominator)?.0)/2e-5);
            }
            differences.add(&finite,analytic,2e-7,2e-6,"enumerated marginal CE derivative")?;
        }
    }
    let pi=float_tensor(&circuits[0].log_pi,[k,v],&device).require_grad();
    let a=float_tensor(&circuits[0].log_a,[k,k],&device).require_grad();
    let rho=float_tensor(&circuits[0].log_root,[k],&device).require_grad();
    let logits=float_tensor(&raw_logits,[rows,width,v],&device).require_grad();
    let gates=float_tensor(&raw_gates,[rows,k,k],&device).require_grad();
    let roots=float_tensor(&raw_roots,[rows,k],&device).require_grad();
    let prepared=PreparedCircuit::prepare_contextual(pi.clone(),a.clone(),rho.clone(),logits.clone(),gates.clone(),roots.clone(),vec![3,2,0],1)?;
    let result=prepared.marginal_ce(int_tensor(&labels,[rows,width],&device),int_tensor(&supplied,[rows,width],&device),
        bool_tensor(&fixed,[rows,width],&device),float_tensor(&row_weights,[rows,width],&device),denominator)?;
    let mut errors=Errors::default();
    errors.tensor(&tensor_values(result.loss.clone()),&[expected_loss],"weighted marginal CE")?;
    ensure!(result.flows.active_blocks==2,"marginal objective counted targets instead of logical chains");
    for (actual,expected) in [&result.flows.emission_positive,&result.flows.emission_negative,&result.flows.transition_positive,&result.flows.transition_negative,&result.flows.root_positive,&result.flows.root_negative,&result.flows.emission_reach,&result.flows.transition_reach,&result.flows.root_reach].into_iter().zip(&expected_flows) {
        errors.tensor(&tensor_values(actual.clone()),expected,"weighted marginal flow/reach")?;
    }
    let gradients=result.loss.backward();
    for (actual,expected) in [tensor_values(logits.grad(&gradients).unwrap()),tensor_values(gates.grad(&gradients).unwrap()),tensor_values(pi.grad(&gradients).unwrap()),tensor_values(a.grad(&gradients).unwrap()),tensor_values(rho.grad(&gradients).unwrap()),tensor_values(roots.grad(&gradients).unwrap())].into_iter().zip(&expected_gradients) {
        errors.tensor(&actual,expected,"marginal CE neural/probability gradient")?;
    }
    let mut state=CircuitState::from_logs(float_tensor(&circuits[0].log_pi,[k,v],&device),float_tensor(&circuits[0].log_a,[k,k],&device),float_tensor(&circuits[0].log_root,[k],&device));
    state.update(result.flows)?;
    for (kind,base,columns,positive,negative,reach,actual) in [
        ("emission",&circuits[0].log_pi,v,&expected_flows[0],&expected_flows[1],&expected_flows[6],tensor_values(state.log_emission.val().exp())),
        ("transition",&circuits[0].log_a,k,&expected_flows[2],&expected_flows[3],&expected_flows[7],tensor_values(state.log_transition.val().exp())),
        ("root",&circuits[0].log_root,k,&expected_flows[4],&expected_flows[5],&expected_flows[8],tensor_values(state.log_root.val().exp()))] {
        let mut expected=Vec::new();
        for row in 0..base.len()/columns {
            let offset=row*columns;
            expected.extend(row_update(&base[offset..offset+columns].iter().map(|x|x.exp()).collect::<Vec<_>>(),
                &positive[offset..offset+columns].iter().map(|x|x/2.0).collect::<Vec<_>>(),
                &negative[offset..offset+columns].iter().map(|x|x/2.0).collect::<Vec<_>>(),reach[row]/2.0,0.05)?);
        }
        errors.tensor(&actual,&expected,&format!("marginal first finite {kind} update"))?;
    }
    Ok(json!({"passed":true,"lengths":[3,2,0],"weights":weights,"logical_active_blocks":2,"evidence_differs_from_gold":true,
        "scalar_oracle":"exhaustive latent/output enumeration, singleton events conditioned only on O","all_six_parameter_fields_finite_differenced":true,
        "errors":errors.report(),"f64_finite_differences":differences.report(),"first_bias_corrected_ema_update":true}))
}

#[cfg(all(test,feature="cpu"))]
mod tests {
    use super::*;

    #[test]
    fn marginal_zero_target_update_is_inactive_and_active_bad_weights_error() {
        let device=Device::ndarray();
        let c=Circuit::fixture(2,3,2,1,6030);
        let prepared=PreparedCircuit::from_log_potentials(float_tensor(&c.log_pi,[2,3],&device),
            float_tensor(&c.log_a,[2,2],&device),float_tensor(&c.log_root,[2],&device),
            float_tensor(&c.logits,[1,2,3],&device),float_tensor(&c.gates,[1,2,2],&device),vec![2],1).unwrap();
        let labels=int_tensor(&[0,1],[1,2],&device);
        let values=int_tensor(&[1,2],[1,2],&device);
        let result=prepared.marginal_ce(labels.clone(),values.clone(),bool_tensor(&[true,true],[1,2],&device),
            float_tensor(&[f64::NAN,-1.0],[1,2],&device),1.0).unwrap();
        assert_eq!(tensor_values(result.loss),vec![0.0]);
        assert_eq!(result.flows.active_blocks,0);
        assert_eq!(tensor_values(result.flows.root_reach.clone()),vec![0.0]);
        let mut state=CircuitState::from_logs(float_tensor(&c.log_pi,[2,3],&device),
            float_tensor(&c.log_a,[2,2],&device),float_tensor(&c.log_root,[2],&device));
        state.update(result.flows).unwrap();
        assert_eq!(state.step.val().into_data().try_to_vec::<i32>().unwrap(),vec![0]);
        for bad in [-0.1,f64::NAN,f64::INFINITY] {
            assert!(prepared.marginal_ce(labels.clone(),values.clone(),bool_tensor(&[false,false],[1,2],&device),
                float_tensor(&[bad,0.0],[1,2],&device),1.0).is_err());
        }
    }
    #[test]
    fn marginal_token_loss_matches_enumeration_gradients_and_finite_update() {
        super::checks(&burn::prelude::Device::ndarray()).unwrap();
    }
}
