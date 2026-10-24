//! Cooperative key/value state partitioning: 128 or 256 lanes per batch/head.
//! Each lane owns 16 or 8 key entries of one value column. Time remains exact
//! and sequential; no transition summaries, floating atomics or alpha division.
use super::kron::ix;
use cubecl::frontend::Tensor as View;
use cubecl::prelude::*;
const K: usize = 32;
const V: usize = 64;

#[cube(launch)]
pub(crate) fn forward_kernel(
    q: &View<f32>,
    k: &View<f32>,
    v: &View<f32>,
    alpha: &View<f32>,
    erase: &View<f32>,
    write: &View<f32>,
    valid: &View<f32>,
    reset: &View<f32>,
    initial: &View<f32>,
    packed: &mut View<f32>,
    checkpoint: &mut View<f32>,
    auxiliary: &mut View<f32>,
    #[comptime] chunk: usize,
    #[comptime] partitions: usize,
) {
    let lane = UNIT_POS as usize;
    let j = lane % V;
    let shard = lane / V;
    let head = CUBE_POS as usize;
    let h = head % q.shape(2);
    let b = head / q.shape(2);
    if b >= q.shape(0) {
        terminate!();
    }
    let time = q.shape(1);
    let heads = q.shape(2);
    let chunks = (time + chunk - 1) / chunk;
    let output_len = q.shape(0) * time * heads * V;
    let state_len = q.shape(0) * heads * K * V;
    let mut state = Array::<f32>::new(K / partitions);
    let mut reduction = Shared::<[f32]>::new_slice(V * partitions * 3);
    let mut a = 0usize;
    while a < K / partitions {
        let i = shard * (K / partitions) + a;
        state[a] = initial[b * initial.stride(0)
            + h * initial.stride(1)
            + i * initial.stride(2)
            + j * initial.stride(3)];
        a += 1;
    }
    let mut t = 0usize;
    while t < time {
        if t % chunk == 0 {
            a = 0;
            while a < K / partitions {
                let i = shard * (K / partitions) + a;
                checkpoint[((head * chunks + t / chunk) * K + i) * V + j] = state[a];
                a += 1;
            }
        }
        let active = valid[b * valid.stride(0) + t * valid.stride(1)] != 0.0f32;
        let restarting = reset[b * reset.stride(0) + t * reset.stride(1)] != 0.0f32;
        let mut value = 0.0f32;
        let mut bilinear = 0.0f32;
        let mut key_square = 0.0f32;
        if active {
            value = v[ix(v, b, t, h, j)];
            a = 0;
            while a < K / partitions {
                let i = shard * (K / partitions) + a;
                let key = k[ix(k, b, t, h, i)];
                state[a] = if restarting {
                    0.0f32
                } else {
                    state[a] * alpha[ix(alpha, b, t, h, i)]
                };
                bilinear += key * state[a] * value;
                key_square += key * key;
                a += 1;
            }
        }
        reduction[lane] = bilinear;
        reduction[V * partitions + lane] = if shard == 0 { value * value } else { 0.0f32 };
        reduction[2 * V * partitions + lane] = if j == 0 { key_square } else { 0.0f32 };
        sync_cube();
        let mut stride = 32usize;
        stride *= partitions;
        while stride > 0 {
            if lane < stride {
                reduction[lane] += reduction[lane + stride];
                reduction[V * partitions + lane] += reduction[V * partitions + lane + stride];
                reduction[2 * V * partitions + lane] +=
                    reduction[2 * V * partitions + lane + stride];
            }
            sync_cube();
            stride /= 2;
        }
        let p = reduction[0];
        let key_norm = f32::sqrt(reduction[2 * V * partitions]);
        let value_norm = f32::sqrt(reduction[V * partitions]);
        let mut correction = 0.0f32;
        if active {
            correction = write[ix(write, b, t, h, 0)] - erase[ix(erase, b, t, h, 0)] * p;
        }
        sync_cube();
        let mut output = 0.0f32;
        if active {
            a = 0;
            while a < K / partitions {
                let i = shard * (K / partitions) + a;
                state[a] += correction * k[ix(k, b, t, h, i)] * value;
                output += q[ix(q, b, t, h, i)] * state[a];
                a += 1;
            }
        }
        reduction[lane] = output;
        sync_cube();
        stride = partitions / 2;
        while stride > 0 {
            if shard < stride {
                reduction[lane] += reduction[lane + stride * V];
            }
            sync_cube();
            stride /= 2;
        }
        let token = (b * time + t) * heads + h;
        if shard == 0 {
            packed[token * V + j] = reduction[j];
        }
        if lane == 0 {
            auxiliary[token * 4] = correction;
            auxiliary[token * 4 + 1] = p;
            auxiliary[token * 4 + 2] = key_norm;
            auxiliary[token * 4 + 3] = value_norm;
            packed[output_len + state_len + token] = f32::abs(correction) * key_norm * value_norm;
        }
        sync_cube();
        t += 1;
    }
    a = 0;
    while a < K / partitions {
        let i = shard * (K / partitions) + a;
        packed[output_len + (head * K + i) * V + j] = state[a];
        a += 1;
    }
}

#[cube(launch)]
pub(crate) fn backward_kernel(
    q: &View<f32>,
    k: &View<f32>,
    v: &View<f32>,
    alpha: &View<f32>,
    erase: &View<f32>,
    valid: &View<f32>,
    reset: &View<f32>,
    checkpoint: &View<f32>,
    auxiliary: &View<f32>,
    upstream: &View<f32>,
    replay: &mut View<f32>,
    dq: &mut View<f32>,
    dk: &mut View<f32>,
    dv: &mut View<f32>,
    da: &mut View<f32>,
    de: &mut View<f32>,
    dw: &mut View<f32>,
    ds: &mut View<f32>,
    #[comptime] chunk: usize,
    #[comptime] partitions: usize,
) {
    let lane = UNIT_POS as usize;
    let j = lane % V;
    let shard = lane / V;
    let head = CUBE_POS as usize;
    let h = head % q.shape(2);
    let b = head / q.shape(2);
    if b >= q.shape(0) {
        terminate!();
    }
    let time = q.shape(1);
    let heads = q.shape(2);
    let chunks = (time + chunk - 1) / chunk;
    let output_len = q.shape(0) * time * heads * V;
    let state_len = q.shape(0) * heads * K * V;
    let mut state = Array::<f32>::new(K / partitions);
    let mut adjoint = Array::<f32>::new(K / partitions);
    let mut scalar = Shared::<[f32]>::new_slice(V * partitions);
    let mut channels = Shared::<[f32]>::new_slice(3 * K * V);
    let mut a = 0usize;
    while a < K / partitions {
        let i = shard * (K / partitions) + a;
        adjoint[a] = upstream[(output_len + (head * K + i) * V + j) * upstream.stride(0)];
        a += 1;
    }
    let mut c = chunks;
    while c > 0 {
        c -= 1;
        a = 0;
        while a < K / partitions {
            let i = shard * (K / partitions) + a;
            state[a] = checkpoint[((head * chunks + c) * K + i) * V + j];
            a += 1;
        }
        let mut r = 0usize;
        while r < chunk && c * chunk + r < time {
            let t = c * chunk + r;
            let token = (b * time + t) * heads + h;
            let active = valid[b * valid.stride(0) + t * valid.stride(1)] != 0.0f32;
            let restarting = reset[b * reset.stride(0) + t * reset.stride(1)] != 0.0f32;
            let correction = auxiliary[token * 4];
            a = 0;
            while a < K / partitions {
                let i = shard * (K / partitions) + a;
                replay[((head * chunk + r) * K + i) * V + j] = state[a];
                if active {
                    let bar = if restarting {
                        0.0f32
                    } else {
                        alpha[ix(alpha, b, t, h, i)] * state[a]
                    };
                    state[a] = bar + correction * k[ix(k, b, t, h, i)] * v[ix(v, b, t, h, j)];
                }
                a += 1;
            }
            r += 1;
        }
        while r > 0 {
            r -= 1;
            let t = c * chunk + r;
            let token = (b * time + t) * heads + h;
            let active = valid[b * valid.stride(0) + t * valid.stride(1)] != 0.0f32;
            let restarting = reset[b * reset.stride(0) + t * reset.stride(1)] != 0.0f32;
            let correction = auxiliary[token * 4];
            let p = auxiliary[token * 4 + 1];
            let key_norm = auxiliary[token * 4 + 2];
            let value_norm = auxiliary[token * 4 + 3];
            let mut value = 0.0f32;
            let mut dy = 0.0f32;
            let mut beta = 0.0f32;
            let mut score_grad = 0.0f32;
            let mut dr = 0.0f32;
            if active {
                value = v[ix(v, b, t, h, j)];
                beta = erase[ix(erase, b, t, h, 0)];
                dy = upstream[(token * V + j) * upstream.stride(0)];
                score_grad = upstream[(output_len + state_len + token) * upstream.stride(0)];
                a = 0;
                while a < K / partitions {
                    let i = shard * (K / partitions) + a;
                    adjoint[a] += q[ix(q, b, t, h, i)] * dy;
                    dr += adjoint[a] * k[ix(k, b, t, h, i)] * value;
                    a += 1;
                }
            }
            scalar[lane] = dr;
            sync_cube();
            let mut stride = 32usize;
            stride *= partitions;
            while stride > 0 {
                if lane < stride {
                    scalar[lane] += scalar[lane + stride];
                }
                sync_cube();
                stride /= 2;
            }
            let mut sign = 0.0f32;
            if correction > 0.0f32 {
                sign = 1.0f32;
            } else if correction < 0.0f32 {
                sign = -1.0f32;
            }
            dr = scalar[0] + score_grad * sign * key_norm * value_norm;
            let mut key_score_scale = 0.0f32;
            let mut value_score_scale = 0.0f32;
            if key_norm > 0.0f32 {
                key_score_scale = score_grad * f32::abs(correction) * value_norm / key_norm;
            }
            if value_norm > 0.0f32 {
                value_score_scale = score_grad * f32::abs(correction) * key_norm / value_norm;
            }
            let mut value_grad = 0.0f32;
            a = 0;
            while a < K / partitions {
                let i = shard * (K / partitions) + a;
                let mut query_grad = 0.0f32;
                let mut key_grad = 0.0f32;
                let mut decay_grad = 0.0f32;
                if active {
                    let key = k[ix(k, b, t, h, i)];
                    let decay = alpha[ix(alpha, b, t, h, i)];
                    let previous = replay[((head * chunk + r) * K + i) * V + j];
                    let bar = if restarting { 0.0f32 } else { decay * previous };
                    query_grad = (bar + correction * key * value) * dy;
                    key_grad = correction * adjoint[a] * value - beta * dr * bar * value;
                    value_grad += correction * adjoint[a] * key - beta * dr * bar * key;
                    let dbar = adjoint[a] - beta * dr * key * value;
                    if restarting {
                        adjoint[a] = 0.0f32;
                    } else {
                        decay_grad = dbar * previous;
                        adjoint[a] = decay * dbar;
                    }
                }
                channels[i * V + j] = query_grad;
                channels[(K + i) * V + j] = key_grad;
                channels[(2 * K + i) * V + j] = decay_grad;
                a += 1;
            }
            // Every lane has consumed dr before reusing scalar for the value VJP.
            sync_cube();
            scalar[lane] = value_grad;
            sync_cube();
            stride = V / 2;
            while stride > 0 {
                if j < stride {
                    a = 0;
                    while a < K / partitions {
                        let i = shard * (K / partitions) + a;
                        channels[i * V + j] += channels[i * V + j + stride];
                        channels[(K + i) * V + j] += channels[(K + i) * V + j + stride];
                        channels[(2 * K + i) * V + j] += channels[(2 * K + i) * V + j + stride];
                        a += 1;
                    }
                }
                if stride < partitions && shard < stride {
                    scalar[lane] += scalar[lane + stride * V];
                }
                sync_cube();
                stride /= 2;
            }
            if j < K / partitions {
                let i = shard * (K / partitions) + j;
                dq[token * K + i] = channels[i * V];
                let mut direct = 0.0f32;
                if active {
                    direct = key_score_scale * k[ix(k, b, t, h, i)];
                }
                dk[token * K + i] = channels[(K + i) * V] + direct;
                da[token * K + i] = channels[(2 * K + i) * V];
            }
            if shard == 0 {
                dv[token * V + j] = scalar[j] + value_score_scale * value;
            }
            if lane == 0 {
                de[token] = -dr * p;
                dw[token] = dr;
            }
            // Protect value-VJP readers before the next token overwrites scalar.
            sync_cube();
        }
    }
    a = 0;
    while a < K / partitions {
        let i = shard * (K / partitions) + a;
        ds[(head * K + i) * V + j] = adjoint[a];
        a += 1;
    }
}
