//! Independent scalar exact-attention/cache-membership and gradient oracle.
use crate::CachePolicy;
pub struct Result {
    pub output: Vec<f64>,
    pub dq: Vec<f64>,
    pub dk: Vec<f64>,
    pub dv: Vec<f64>,
    pub final_positions: Vec<f64>,
    pub pairs: usize,
}
/// Flattened input layout [batch,time,head,channel], metadata [batch,time].
pub fn evaluate(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    score: &[f64],
    valid: &[bool],
    docs: &[usize],
    dims: [usize; 5],
    capacity: usize,
    chunk: usize,
    policy: CachePolicy,
    upstream: &[f64],
) -> Result {
    let [b, t, h, dk, dv] = dims;
    let mut result = Result {
        output: vec![0.; b * t * h * dv],
        dq: vec![0.; q.len()],
        dk: vec![0.; k.len()],
        dv: vec![0.; v.len()],
        final_positions: vec![-1.; b * h * capacity],
        pairs: 0,
    };
    let scale = (dk as f64).sqrt();
    for row in 0..b {
        for head in 0..h {
            let mut cache = Vec::<usize>::new();
            for first in (0..t).step_by(chunk) {
                let end = (first + chunk).min(t);
                for pos in first..end {
                    if !valid[row * t + pos] {
                        continue;
                    }
                    let visible = cache
                        .iter()
                        .copied()
                        .chain(first..=pos)
                        .filter(|&s| valid[row * t + s] && docs[row * t + s] == docs[row * t + pos])
                        .collect::<Vec<_>>();
                    let qi = ((row * t + pos) * h + head) * dk;
                    let oi = ((row * t + pos) * h + head) * dv;
                    let mut logits = visible
                        .iter()
                        .map(|&s| {
                            let ki = ((row * t + s) * h + head) * dk;
                            (0..dk).map(|i| q[qi + i] * k[ki + i]).sum::<f64>() / scale
                        })
                        .collect::<Vec<_>>();
                    logits.push(0.); // learned sink initialized to zero, with an exactly zero value.
                    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let total = logits.iter().map(|x| (x - max).exp()).sum::<f64>();
                    let probabilities = logits
                        .iter()
                        .map(|x| (x - max).exp() / total)
                        .collect::<Vec<_>>();
                    for (j, &s) in visible.iter().enumerate() {
                        let vi = ((row * t + s) * h + head) * dv;
                        for c in 0..dv {
                            result.output[oi + c] += probabilities[j] * v[vi + c];
                        }
                    }
                    for (j, &s) in visible.iter().enumerate() {
                        let ki = ((row * t + s) * h + head) * dk;
                        let vi = ((row * t + s) * h + head) * dv;
                        let dlogit = probabilities[j]
                            * (0..dv)
                                .map(|c| upstream[oi + c] * (v[vi + c] - result.output[oi + c]))
                                .sum::<f64>();
                        for c in 0..dv {
                            result.dv[vi + c] += probabilities[j] * upstream[oi + c];
                        }
                        for c in 0..dk {
                            result.dq[qi + c] += dlogit * k[ki + c] / scale;
                            result.dk[ki + c] += dlogit * q[qi + c] / scale;
                        }
                    }
                }
                let last_doc = docs[row * t + end - 1];
                cache.extend(first..end);
                cache.retain(|&s| valid[row * t + s] && docs[row * t + s] == last_doc);
                cache.sort_by(|&left, &right| {
                    let value = |s| {
                        if policy == CachePolicy::Recency {
                            s as f64
                        } else {
                            score[(row * t + s) * h + head]
                        }
                    };
                    value(right)
                        .total_cmp(&value(left))
                        .then_with(|| left.cmp(&right))
                });
                cache.truncate(capacity);
            }
            for (slot, &pos) in cache.iter().enumerate() {
                result.final_positions[(row * h + head) * capacity + slot] = pos as f64;
            }
        }
    }
    result.pairs = (0..t)
        .step_by(chunk)
        .map(|first| {
            let n = (t - first).min(chunk);
            b * h * n * (capacity + n + 1)
        })
        .sum();
    result
}
