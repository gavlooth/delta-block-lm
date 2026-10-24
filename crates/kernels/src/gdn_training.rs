//! Stateful GDN-2 training recurrence. One value column per lane, bounded replay.
use burn::backend::{TensorMetadata, ops::FloatTensorOps};
use burn::{prelude::Shape, tensor::DType};
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use cubecl::frontend::Tensor as View;
use cubecl::prelude::*;
use crate::kron::ix;
const K: usize = 32;
const V: usize = 64;
const AUX: usize = V + 2;

#[cube(launch)]
fn forward_kernel(
    q: &View<f32>, k: &View<f32>, v: &View<f32>, alpha: &View<f32>,
    erase: &View<f32>, write: &View<f32>, valid: &View<f32>, reset: &View<f32>,
    initial: &View<f32>, packed: &mut View<f32>, checkpoint: &mut View<f32>,
    auxiliary: &mut View<f32>, #[comptime] chunk: usize,
) {
    let j = UNIT_POS as usize;
    let head = CUBE_POS as usize;
    let h = head % q.shape(2);
    let b = head / q.shape(2);
    if b >= q.shape(0) { terminate!(); }
    let time = q.shape(1);
    let heads = q.shape(2);
    let chunks = (time + chunk - 1) / chunk;
    let output_len = q.shape(0) * time * heads * V;
    let state_len = q.shape(0) * heads * K * V;
    let mut state = Array::<f32>::new(K);
    let mut reduction = Shared::<[f32]>::new_slice(V);
    let mut i = 0usize;
    while i < K {
        state[i] = initial[b * initial.stride(0) + h * initial.stride(1) + i * initial.stride(2) + j * initial.stride(3)];
        i += 1;
    }
    let mut t = 0usize;
    while t < time {
        if t % chunk == 0 {
            i = 0;
            while i < K {
                checkpoint[((head * chunks + t / chunk) * K + i) * V + j] = state[i];
                i += 1;
            }
        }
        let token = (b * time + t) * heads + h;
        let active = valid[b * valid.stride(0) + t * valid.stride(1)] != 0.0f32;
        let restarting = reset[b * reset.stride(0) + t * reset.stride(1)] != 0.0f32;
        let mut delta = 0.0f32;
        let mut key_square = 0.0f32;
        if active {
            delta = write[ix(write, b, t, h, j)] * v[ix(v, b, t, h, j)];
            i = 0;
            while i < K {
                let key = k[ix(k, b, t, h, i)];
                state[i] = if restarting { 0.0f32 } else { state[i] * alpha[ix(alpha, b, t, h, i)] };
                delta -= erase[ix(erase, b, t, h, i)] * key * state[i];
                key_square += key * key;
                i += 1;
            }
        }
        auxiliary[token * AUX + j] = delta;
        reduction[j] = delta * delta;
        sync_cube();
        let mut stride = 32usize;
        while stride > 0 {
            if j < stride { reduction[j] += reduction[j + stride]; }
            sync_cube();
            stride /= 2;
        }
        if j == 0 {
            let key_norm = f32::sqrt(key_square);
            let edit_norm = f32::sqrt(reduction[0]);
            auxiliary[token * AUX + V] = key_norm;
            auxiliary[token * AUX + V + 1] = edit_norm;
            packed[output_len + state_len + token] = key_norm * edit_norm;
        }
        sync_cube();
        let mut output = 0.0f32;
        if active {
            i = 0;
            while i < K {
                state[i] += k[ix(k, b, t, h, i)] * delta;
                output += q[ix(q, b, t, h, i)] * state[i];
                i += 1;
            }
        }
        packed[token * V + j] = output;
        t += 1;
    }
    i = 0;
    while i < K {
        packed[output_len + (head * K + i) * V + j] = state[i];
        i += 1;
    }
}

#[cube(launch)]
fn backward_kernel(
    q: &View<f32>, k: &View<f32>, v: &View<f32>, alpha: &View<f32>,
    erase: &View<f32>, write: &View<f32>, valid: &View<f32>, reset: &View<f32>,
    checkpoint: &View<f32>, auxiliary: &View<f32>, upstream: &View<f32>,
    replay: &mut View<f32>, dq: &mut View<f32>, dk: &mut View<f32>,
    dv: &mut View<f32>, da: &mut View<f32>, de: &mut View<f32>,
    dw: &mut View<f32>, ds: &mut View<f32>, #[comptime] chunk: usize,
) {
    let j = UNIT_POS as usize;
    let head = CUBE_POS as usize;
    let h = head % q.shape(2);
    let b = head / q.shape(2);
    if b >= q.shape(0) { terminate!(); }
    let time = q.shape(1);
    let heads = q.shape(2);
    let chunks = (time + chunk - 1) / chunk;
    let output_len = q.shape(0) * time * heads * V;
    let state_len = q.shape(0) * heads * K * V;
    let mut state = Array::<f32>::new(K);
    let mut adjoint = Array::<f32>::new(K);
    // Q/K/alpha/erase gradients reduce over columns; exactly 32 KiB shared.
    let mut channels = Shared::<[f32]>::new_slice(4 * K * V);
    let mut i = 0usize;
    while i < K {
        adjoint[i] = upstream[(output_len + (head * K + i) * V + j) * upstream.stride(0)];
        i += 1;
    }
    let mut c = chunks;
    while c > 0 {
        c -= 1;
        i = 0;
        while i < K {
            state[i] = checkpoint[((head * chunks + c) * K + i) * V + j];
            i += 1;
        }
        let mut r = 0usize;
        while r < chunk && c * chunk + r < time {
            let t = c * chunk + r;
            let token = (b * time + t) * heads + h;
            let active = valid[b * valid.stride(0) + t * valid.stride(1)] != 0.0f32;
            let restarting = reset[b * reset.stride(0) + t * reset.stride(1)] != 0.0f32;
            i = 0;
            while i < K {
                replay[((head * chunk + r) * K + i) * V + j] = state[i];
                if active {
                    let bar = if restarting { 0.0f32 } else { alpha[ix(alpha, b, t, h, i)] * state[i] };
                    state[i] = bar + k[ix(k, b, t, h, i)] * auxiliary[token * AUX + j];
                }
                i += 1;
            }
            r += 1;
        }
        while r > 0 {
            r -= 1;
            let t = c * chunk + r;
            let token = (b * time + t) * heads + h;
            let active = valid[b * valid.stride(0) + t * valid.stride(1)] != 0.0f32;
            let restarting = reset[b * reset.stride(0) + t * reset.stride(1)] != 0.0f32;
            let delta = auxiliary[token * AUX + j];
            let key_norm = auxiliary[token * AUX + V];
            let edit_norm = auxiliary[token * AUX + V + 1];
            let mut dy = 0.0f32;
            let mut dd = 0.0f32;
            let mut key_scale = 0.0f32;
            if active {
                dy = upstream[(token * V + j) * upstream.stride(0)];
                let score_grad = upstream[(output_len + state_len + token) * upstream.stride(0)];
                if key_norm > 0.0f32 && edit_norm > 0.0f32 {
                    dd = score_grad * key_norm * delta / edit_norm;
                    key_scale = score_grad * edit_norm / key_norm;
                }
                i = 0;
                while i < K {
                    adjoint[i] += q[ix(q, b, t, h, i)] * dy;
                    dd += k[ix(k, b, t, h, i)] * adjoint[i];
                    i += 1;
                }
            }
            i = 0;
            while i < K {
                let mut query_grad = 0.0f32;
                let mut key_grad = 0.0f32;
                let mut decay_grad = 0.0f32;
                let mut erase_grad = 0.0f32;
                if active {
                    let key = k[ix(k, b, t, h, i)];
                    let decay = alpha[ix(alpha, b, t, h, i)];
                    let erase_gate = erase[ix(erase, b, t, h, i)];
                    let previous = replay[((head * chunk + r) * K + i) * V + j];
                    let bar = if restarting { 0.0f32 } else { decay * previous };
                    query_grad = (bar + key * delta) * dy;
                    key_grad = adjoint[i] * delta - erase_gate * bar * dd;
                    erase_grad = -key * bar * dd;
                    let dbar = adjoint[i] - erase_gate * key * dd;
                    if restarting { adjoint[i] = 0.0f32; } else {
                        decay_grad = dbar * previous;
                        adjoint[i] = decay * dbar;
                    }
                }
                channels[i * V + j] = query_grad;
                channels[(K + i) * V + j] = key_grad;
                channels[(2 * K + i) * V + j] = decay_grad;
                channels[(3 * K + i) * V + j] = erase_grad;
                i += 1;
            }
            sync_cube();
            let mut stride = 32usize;
            while stride > 0 {
                if j < stride {
                    i = 0;
                    while i < 4 * K {
                        channels[i * V + j] += channels[i * V + j + stride];
                        i += 1;
                    }
                }
                sync_cube();
                stride /= 2;
            }
            if j < K {
                dq[token * K + j] = channels[j * V];
                let mut direct = 0.0f32;
                if active { direct = key_scale * k[ix(k, b, t, h, j)]; }
                dk[token * K + j] = channels[(K + j) * V] + direct;
                da[token * K + j] = channels[(2 * K + j) * V];
                de[token * K + j] = channels[(3 * K + j) * V];
            }
            let mut value_grad = 0.0f32;
            let mut write_grad = 0.0f32;
            if active {
                value_grad = dd * write[ix(write, b, t, h, j)];
                write_grad = dd * v[ix(v, b, t, h, j)];
            }
            dv[token * V + j] = value_grad;
            dw[token * V + j] = write_grad;
            sync_cube();
        }
    }
    i = 0;
    while i < K {
        ds[(head * K + i) * V + j] = adjoint[i];
        i += 1;
    }
}

fn checked_elements(dims: &[usize]) -> usize {
    let elements = dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d))
        .expect("GDN-2 element count overflows usize");
    assert!(elements <= u32::MAX as usize, "GDN-2 exceeds 32-bit addressing");
    elements
}

fn assert_u32_tensor(tensor: &CubeTensor) {
    checked_elements(tensor.meta.shape().as_slice());
    let mut span = 1usize;
    for (&dim, &stride) in tensor.meta.shape().as_slice().iter().zip(tensor.meta.strides().iter()) {
        assert!(stride <= u32::MAX as usize, "GDN-2 stride exceeds 32-bit addressing");
        span = dim.saturating_sub(1).checked_mul(stride)
            .and_then(|offset| span.checked_add(offset)).expect("GDN-2 address span overflow");
    }
    assert!(span <= u32::MAX as usize, "GDN-2 span exceeds 32-bit addressing");
    assert!(tensor.handle.size() / tensor.elem_size() as u64 <= u32::MAX as u64,
        "GDN-2 backing buffer exceeds 32-bit addressing");
}

fn empty(reference: &CubeTensor, dims: impl Into<Shape>) -> CubeTensor {
    let shape = dims.into();
    checked_elements(shape.as_slice());
    let tensor = CubeBackend::float_empty(shape, &reference.device, DType::F32.into());
    assert_u32_tensor(&tensor);
    tensor
}

fn count(reference: &CubeTensor, heads: usize) -> CubeCount {
    let (max_x, max_y, _) = reference.client.properties().hardware.max_cube_count;
    let x = heads.min(max_x.min(65535) as usize).max(1);
    let y = heads.div_ceil(x);
    assert!(y <= max_y as usize, "GDN-2 exceeds workgroup limits");
    CubeCount::Static(x as u32, y as u32, 1)
}

pub(crate) fn run_scan(
    q: CubeTensor, k: CubeTensor, v: CubeTensor, alpha: CubeTensor,
    erase: CubeTensor, write: CubeTensor, valid: CubeTensor, reset: CubeTensor,
    initial: CubeTensor, chunk: usize,
) -> (CubeTensor, CubeTensor, CubeTensor) {
    let [batch, time, heads, keys] = q.shape().dims::<4>();
    assert!(batch > 0 && time > 0 && heads > 0, "GDN-2 requires nonempty axes");
    assert_eq!(keys, K, "fused GDN-2 requires key width 32");
    assert_eq!(k.shape().dims::<4>(), [batch, time, heads, K]);
    assert_eq!(v.shape().dims::<4>(), [batch, time, heads, V]);
    assert_eq!(alpha.shape().dims::<4>(), [batch, time, heads, K]);
    assert_eq!(erase.shape().dims::<4>(), [batch, time, heads, K]);
    assert_eq!(write.shape().dims::<4>(), [batch, time, heads, V]);
    assert_eq!(valid.shape().dims::<2>(), [batch, time]);
    assert_eq!(reset.shape().dims::<2>(), [batch, time]);
    assert_eq!(initial.shape().dims::<4>(), [batch, heads, K, V]);
    assert!((1..=16).contains(&chunk), "GDN-2 checkpoint chunk must be 1..=16");
    for tensor in [&q, &k, &v, &alpha, &erase, &write, &valid, &reset, &initial] {
        assert_eq!(tensor.dtype, DType::F32, "fused GDN-2 supports f32 tensors only");
        assert_eq!(tensor.device, q.device, "GDN-2 inputs must share a device");
        assert_u32_tensor(tensor);
    }
    checked_elements(&[batch, heads, chunk, K, V]);
    checked_elements(&[batch, heads, time.div_ceil(chunk), K, V]);
    checked_elements(&[batch, time, heads, AUX]);
    let state_len = checked_elements(&[batch, heads, K, V]);
    let packed_len = checked_elements(&[batch, time, heads, V + 1])
        .checked_add(state_len).expect("GDN-2 packed length overflow");
    checked_elements(&[packed_len]);
    let packed = empty(&q, [packed_len]);
    let checkpoint = empty(&q, [batch, heads, time.div_ceil(chunk), K, V]);
    let auxiliary = empty(&q, [batch, time, heads, AUX]);
    forward_kernel::launch(
        &q.client, count(&q, batch * heads), CubeDim::new_1d(V as u32),
        q.clone().into_tensor_arg(), k.into_tensor_arg(), v.into_tensor_arg(),
        alpha.into_tensor_arg(), erase.into_tensor_arg(), write.into_tensor_arg(),
        valid.into_tensor_arg(), reset.into_tensor_arg(), initial.into_tensor_arg(),
        packed.clone().into_tensor_arg(), checkpoint.clone().into_tensor_arg(),
        auxiliary.clone().into_tensor_arg(), chunk,
    );
    (packed, checkpoint, auxiliary)
}

pub(crate) fn run_backward(
    q: CubeTensor, k: CubeTensor, v: CubeTensor, alpha: CubeTensor,
    erase: CubeTensor, write: CubeTensor, upstream: CubeTensor, checkpoint: CubeTensor,
    auxiliary: CubeTensor, valid: CubeTensor, reset: CubeTensor, chunk: usize,
) -> (CubeTensor, CubeTensor, CubeTensor, CubeTensor, CubeTensor, CubeTensor, CubeTensor) {
    let [batch, time, heads, _] = q.shape().dims::<4>();
    for tensor in [&q, &k, &v, &alpha, &erase, &write, &upstream, &checkpoint, &auxiliary, &valid, &reset] {
        assert_u32_tensor(tensor);
    }
    let replay = empty(&q, [batch, heads, chunk, K, V]);
    let dq = empty(&q, q.shape());
    let dk = empty(&q, k.shape());
    let dv = empty(&q, v.shape());
    let da = empty(&q, alpha.shape());
    let de = empty(&q, erase.shape());
    let dw = empty(&q, write.shape());
    let ds = empty(&q, [batch, heads, K, V]);
    backward_kernel::launch(
        &q.client, count(&q, batch * heads), CubeDim::new_1d(V as u32),
        q.clone().into_tensor_arg(), k.into_tensor_arg(), v.into_tensor_arg(),
        alpha.into_tensor_arg(), erase.into_tensor_arg(), write.into_tensor_arg(),
        valid.into_tensor_arg(), reset.into_tensor_arg(), checkpoint.into_tensor_arg(),
        auxiliary.into_tensor_arg(), upstream.into_tensor_arg(), replay.into_tensor_arg(),
        dq.clone().into_tensor_arg(), dk.clone().into_tensor_arg(), dv.clone().into_tensor_arg(),
        da.clone().into_tensor_arg(), de.clone().into_tensor_arg(), dw.clone().into_tensor_arg(),
        ds.clone().into_tensor_arg(), chunk,
    );
    (dq, dk, dv, da, de, dw, ds)
}
