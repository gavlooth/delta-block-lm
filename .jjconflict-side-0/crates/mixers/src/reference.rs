//! Plain f64 sequential KronLSTM oracle for arXiv:2610.26834 Eq. (10),
//! transposed to key-major state storage.
use crate::MixerKind;
#[derive(Clone, Debug)]
pub struct Token {
    pub q: Vec<f64>,
    pub k: Vec<f64>,
    pub v: Vec<f64>,
    pub alpha: Vec<f64>,
    pub erase: Vec<f64>,
    pub write: Vec<f64>,
}
impl Token {
    pub fn zeros_like(&self) -> Self {
        Self {
            q: vec![0.; self.q.len()],
            k: vec![0.; self.k.len()],
            v: vec![0.; self.v.len()],
            alpha: vec![0.; self.alpha.len()],
            erase: vec![0.; self.erase.len()],
            write: vec![0.; self.write.len()],
        }
    }
    pub fn fields(&self) -> [&[f64]; 6] {
        [
            &self.q,
            &self.k,
            &self.v,
            &self.alpha,
            &self.erase,
            &self.write,
        ]
    }
}
pub struct Scan {
    pub output: Vec<Vec<f64>>,
    pub states: Vec<Vec<f64>>,
    pub scores: Vec<f64>,
    pub final_state: Vec<f64>,
}
pub fn scan(
    _kind: MixerKind,
    tokens: &[Token],
    valid: &[bool],
    reset: &[bool],
    initial: Option<&[f64]>,
) -> Scan {
    assert!(!tokens.is_empty());
    assert_eq!(tokens.len(), valid.len());
    assert_eq!(tokens.len(), reset.len());
    let k = tokens[0].k.len();
    let v = tokens[0].v.len();
    let mut state = initial.map_or_else(|| vec![0.; k * v], |x| x.to_vec());
    let mut output = Vec::with_capacity(tokens.len());
    let mut states = Vec::with_capacity(tokens.len());
    let mut scores = Vec::with_capacity(tokens.len());
    for (t, x) in tokens.iter().enumerate() {
        assert!(
            x.fields().into_iter().flatten().all(|x| x.is_finite()),
            "nonfinite recurrence input"
        );
        if !valid[t] {
            output.push(vec![0.; v]);
            states.push(state.clone());
            scores.push(0.);
            continue;
        }
        if reset[t] {
            state.fill(0.);
        }
        for i in 0..k {
            for j in 0..v {
                state[i * v + j] *= x.alpha[i];
            }
        }
        let mut edit = vec![0.; v];
        let mut bilinear = 0.;
        for i in 0..k {
            for j in 0..v {
                bilinear += x.k[i] * state[i * v + j] * x.v[j];
            }
        }
        let correction = x.write[0] - x.erase[0] * bilinear;
        for j in 0..v {
            edit[j] = correction * x.v[j];
        }
        let score = (x.k.iter().map(|x| x * x).sum::<f64>()
            * edit.iter().map(|x| x * x).sum::<f64>())
        .sqrt();
        let mut out = vec![0.; v];
        for i in 0..k {
            for j in 0..v {
                state[i * v + j] += x.k[i] * edit[j];
                out[j] += x.q[i] * state[i * v + j];
            }
        }
        output.push(out);
        states.push(state.clone());
        scores.push(score);
    }
    Scan {
        output,
        states,
        scores,
        final_state: state,
    }
}

pub struct Backward {
    pub tokens: Vec<Token>,
    pub initial_state: Vec<f64>,
}

/// Analytic VJP of outputs, returned state and physical edit scores, including initial state.
pub fn backward(
    kind: MixerKind,
    tokens: &[Token],
    valid: &[bool],
    reset: &[bool],
    upstream: &[Vec<f64>],
    final_adjoint: &[f64],
    initial: Option<&[f64]>,
    score_adjoint: &[f64],
) -> Backward {
    let forward = scan(kind, tokens, valid, reset, initial);
    let k = tokens[0].k.len();
    let v = tokens[0].v.len();
    let initial = initial.map_or_else(|| vec![0.; k * v], |x| x.to_vec());
    assert_eq!(score_adjoint.len(), tokens.len());
    let mut adj = final_adjoint.to_vec();
    assert_eq!(adj.len(), k * v);
    let mut result = tokens.iter().map(Token::zeros_like).collect::<Vec<_>>();
    for t in (0..tokens.len()).rev() {
        if !valid[t] {
            continue;
        }
        let x = &tokens[t];
        let g = &mut result[t];
        let zero = vec![0.; k * v];
        let previous = if reset[t] {
            &zero
        } else if t == 0 {
            &initial
        } else {
            &forward.states[t - 1]
        };
        let mut bar = vec![0.; k * v];
        for i in 0..k {
            for j in 0..v {
                g.q[i] += forward.states[t][i * v + j] * upstream[t][j];
                adj[i * v + j] += x.q[i] * upstream[t][j];
                bar[i * v + j] = x.alpha[i] * previous[i * v + j];
            }
        }
        let mut dbar = adj.clone();
        let mut bilinear = 0.;
        let mut dz = 0.;
        for i in 0..k {
            for j in 0..v {
                bilinear += bar[i * v + j] * x.k[i] * x.v[j];
                dz += adj[i * v + j] * x.k[i] * x.v[j];
            }
        }
        let z = x.write[0] - x.erase[0] * bilinear;
        let nk = x.k.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nv = x.v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nk > 0. && nv > 0. {
            let sign = if z > 0. {
                1.
            } else if z < 0. {
                -1.
            } else {
                0.
            };
            dz += score_adjoint[t] * sign * nk * nv;
            for i in 0..k {
                g.k[i] += score_adjoint[t] * z.abs() * nv / nk * x.k[i];
            }
            for j in 0..v {
                g.v[j] += score_adjoint[t] * z.abs() * nk / nv * x.v[j];
            }
        }
        g.write[0] = dz;
        g.erase[0] = -bilinear * dz;
        for i in 0..k {
            for j in 0..v {
                g.k[i] +=
                    adj[i * v + j] * z * x.v[j] - x.erase[0] * dz * bar[i * v + j] * x.v[j];
                g.v[j] +=
                    adj[i * v + j] * z * x.k[i] - x.erase[0] * dz * bar[i * v + j] * x.k[i];
                dbar[i * v + j] -= x.erase[0] * dz * x.k[i] * x.v[j];
            }
        }
        for i in 0..k {
            for j in 0..v {
                g.alpha[i] += dbar[i * v + j] * previous[i * v + j];
                adj[i * v + j] = if reset[t] {
                    0.
                } else {
                    x.alpha[i] * dbar[i * v + j]
                };
            }
        }
    }
    Backward {
        tokens: result,
        initial_state: adj,
    }
}
