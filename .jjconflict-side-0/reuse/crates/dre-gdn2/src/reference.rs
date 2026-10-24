//! CPU f64 oracle for the Gated Delta Rule-2 token recurrence.

pub const HEADS: usize = 4;
pub const KEY_DIM: usize = 32;
pub const VALUE_DIM: usize = 64;

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
    pub fn new(
        q: Vec<f64>,
        k: Vec<f64>,
        v: Vec<f64>,
        alpha: Vec<f64>,
        erase: Vec<f64>,
        write: Vec<f64>,
    ) -> Self {
        assert_eq!(q.len(), KEY_DIM);
        assert_eq!(k.len(), KEY_DIM);
        assert_eq!(alpha.len(), KEY_DIM);
        assert_eq!(erase.len(), KEY_DIM);
        assert_eq!(v.len(), VALUE_DIM);
        assert_eq!(write.len(), VALUE_DIM);
        Self { q, k, v, alpha, erase, write }
    }
}

#[derive(Clone, Debug)]
pub struct ScanResult {
    /// Token-major outputs, each of width `VALUE_DIM`.
    pub output: Vec<Vec<f64>>,
    /// State immediately after each valid token; zero at padding positions.
    pub states: Vec<Vec<f64>>,
}

/// Apply one GDN-2 recurrence step using f64 arithmetic.
pub fn step(state: &mut [f64], token: &Token) -> Vec<f64> {
    assert_eq!(state.len(), KEY_DIM * VALUE_DIM);
    let mut s_bar = vec![0.0; KEY_DIM * VALUE_DIM];
    for i in 0..KEY_DIM {
        for j in 0..VALUE_DIM {
            let idx = i * VALUE_DIM + j;
            s_bar[idx] = token.alpha[i] * state[idx];
        }
    }
    let mut read = vec![0.0; VALUE_DIM];
    for j in 0..VALUE_DIM {
        for i in 0..KEY_DIM {
            read[j] += s_bar[i * VALUE_DIM + j] * token.erase[i] * token.k[i];
        }
    }
    let mut delta = vec![0.0; VALUE_DIM];
    for j in 0..VALUE_DIM {
        delta[j] = token.write[j] * token.v[j] - read[j];
    }
    for i in 0..KEY_DIM {
        for j in 0..VALUE_DIM {
            state[i * VALUE_DIM + j] = s_bar[i * VALUE_DIM + j] + token.k[i] * delta[j];
        }
    }
    let mut output = vec![0.0; VALUE_DIM];
    for j in 0..VALUE_DIM {
        for i in 0..KEY_DIM {
            output[j] += state[i * VALUE_DIM + j] * token.q[i];
        }
    }
    output
}

/// Scan a token sequence, resetting state at each document start and ignoring padding.
/// `doc_start` and `doc_end` are inclusive position limits; `end < start` marks padding.
pub fn scan(tokens: &[Token], doc_start: &[i32], doc_end: &[i32]) -> ScanResult {
    assert_eq!(tokens.len(), doc_start.len());
    assert_eq!(tokens.len(), doc_end.len());
    let mut state = vec![0.0; KEY_DIM * VALUE_DIM];
    let mut output = vec![vec![0.0; VALUE_DIM]; tokens.len()];
    let mut states = vec![vec![0.0; KEY_DIM * VALUE_DIM]; tokens.len()];
    for t in 0..tokens.len() {
        if doc_end[t] < doc_start[t] {
            continue;
        }
        if doc_start[t] == t as i32 {
            state.fill(0.0);
        }
        output[t] = step(&mut state, &tokens[t]);
        states[t].copy_from_slice(&state);
    }
    ScanResult { output, states }
}

/// Analytic gradient of the sum of token outputs dotted with `output_grad`.
/// Gradients are returned in the same token order as the inputs.
pub fn backward(tokens: &[Token], doc_start: &[i32], doc_end: &[i32], output_grad: &[Vec<f64>]) -> Vec<Token> {
    assert_eq!(tokens.len(), doc_start.len());
    assert_eq!(tokens.len(), doc_end.len());
    assert_eq!(tokens.len(), output_grad.len());
    assert!(output_grad.iter().all(|g| g.len() == VALUE_DIM));
    let forward = scan(tokens, doc_start, doc_end);
    let mut result = tokens.iter().map(|_| Token::new(
        vec![0.0; KEY_DIM], vec![0.0; KEY_DIM], vec![0.0; VALUE_DIM],
        vec![0.0; KEY_DIM], vec![0.0; KEY_DIM], vec![0.0; VALUE_DIM],
    )).collect::<Vec<_>>();
    let mut adjoint = vec![0.0; KEY_DIM * VALUE_DIM];
    for t in (0..tokens.len()).rev() {
        if doc_end[t] < doc_start[t] {
            continue;
        }
        if doc_end[t] == t as i32 {
            adjoint.fill(0.0);
        }
        let token = &tokens[t];
        let s_prev = if doc_start[t] == t as i32 {
            vec![0.0; KEY_DIM * VALUE_DIM]
        } else {
            forward.states[t - 1].clone()
        };
        let mut s_bar = vec![0.0; KEY_DIM * VALUE_DIM];
        for i in 0..KEY_DIM {
            for j in 0..VALUE_DIM {
                let idx = i * VALUE_DIM + j;
                s_bar[idx] = token.alpha[i] * s_prev[idx];
            }
        }
        let mut read = vec![0.0; VALUE_DIM];
        for j in 0..VALUE_DIM {
            for i in 0..KEY_DIM {
                read[j] += s_bar[i * VALUE_DIM + j] * token.erase[i] * token.k[i];
            }
        }
        let mut delta = vec![0.0; VALUE_DIM];
        for j in 0..VALUE_DIM {
            delta[j] = token.write[j] * token.v[j] - read[j];
        }

        // Output o = S^T q.
        let mut d_q = vec![0.0; KEY_DIM];
        for i in 0..KEY_DIM {
            for j in 0..VALUE_DIM {
                d_q[i] += forward.states[t][i * VALUE_DIM + j] * output_grad[t][j];
                adjoint[i * VALUE_DIM + j] += token.q[i] * output_grad[t][j];
            }
        }
        result[t].q = d_q;

        // S = S_bar + k delta^T; delta = z - S_bar^T e; e = b*k.
        let mut d_delta = vec![0.0; VALUE_DIM];
        let mut d_s_bar = adjoint.clone();
        let mut d_k = vec![0.0; KEY_DIM];
        for i in 0..KEY_DIM {
            for j in 0..VALUE_DIM {
                let a = adjoint[i * VALUE_DIM + j];
                d_k[i] += a * delta[j];
                d_delta[j] += a * token.k[i];
            }
        }
        for i in 0..KEY_DIM {
            for j in 0..VALUE_DIM {
                d_s_bar[i * VALUE_DIM + j] -= token.erase[i] * token.k[i] * d_delta[j];
            }
        }
        let mut d_e = vec![0.0; KEY_DIM];
        for i in 0..KEY_DIM {
            for j in 0..VALUE_DIM {
                d_e[i] -= s_bar[i * VALUE_DIM + j] * d_delta[j];
            }
        }
        for i in 0..KEY_DIM {
            result[t].erase[i] = d_e[i] * token.k[i];
            d_k[i] += d_e[i] * token.erase[i];
            for j in 0..VALUE_DIM {
                adjoint[i * VALUE_DIM + j] = token.alpha[i] * d_s_bar[i * VALUE_DIM + j];
                result[t].alpha[i] += d_s_bar[i * VALUE_DIM + j] * s_prev[i * VALUE_DIM + j];
            }
        }
        for j in 0..VALUE_DIM {
            result[t].write[j] = d_delta[j] * token.v[j];
            result[t].v[j] = d_delta[j] * token.write[j];
        }
        result[t].k = d_k;
    }
    result
}
