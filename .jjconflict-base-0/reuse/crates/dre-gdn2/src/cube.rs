use burn::backend::{TensorMetadata, cubecl::dtype_to_storage_type, ops::FloatTensorOps, wgpu::{CubeBackend, CubeTensor}};
use burn::cubecl::prelude::*;
use burn::prelude::Shape;
use cubecl::frontend::Tensor as CubeTensorView;

const DK: usize = 32;
const DV: usize = 64;

/// Workgroup count for `cubes` cubes, one logical task each. X holds at most 65,535 cubes (the per-dimension limit
/// of WGPU adapters, and lower if the device says so) and the rest goes to Y; kernels read the flattened `CUBE_POS`
/// and stop cubes past the end. `CubeDim::new(&client, 1)` still yields a full plane, so every kernel also stops
/// all units except unit 0, which alone owns the task's writes.
fn cube_count(client: &Client, cubes: usize) -> CubeCount {
    let (max_x, max_y, _) = client.properties().hardware.max_cube_count;
    let x = cubes.min(max_x.min(u16::MAX as u32) as usize);
    let y = if x == 0 { 1 } else { cubes.div_ceil(x) };
    assert!(y <= max_y as usize, "GDN-2 dispatch of {cubes} cubes exceeds the device workgroup limits");
    CubeCount::Static(x as u32, y as u32, 1)
}

#[cube(launch)]
fn scan_kernel<F: Float, I: Int>(
    q: &CubeTensorView<F>, k: &CubeTensorView<F>, v: &CubeTensorView<F>,
    alpha: &CubeTensorView<f32>, erase: &CubeTensorView<f32>, write: &CubeTensorView<f32>,
    doc_start: &CubeTensorView<I>, doc_end: &CubeTensorView<I>,
    state_in: &CubeTensorView<f32>, out: &mut CubeTensorView<F>,
    checkpoints: &mut CubeTensorView<f32>, state_out: &mut CubeTensorView<f32>,
    #[comptime] interval: usize,
    token_start: u32, token_end: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let item = CUBE_POS;
    let j = item % DV;
    let group = item / DV;
    let b = group / q.shape(2);
    let h = group - b * q.shape(2);
    if b >= q.shape(0) || UNIT_POS != 0 { terminate!(); }
    let out_s0 = out.stride(0); let out_s1 = out.stride(1); let out_s2 = out.stride(2); let out_s3 = out.stride(3);
    let cp_s0 = checkpoints.stride(0); let cp_s1 = checkpoints.stride(1); let cp_s2 = checkpoints.stride(2); let cp_s3 = checkpoints.stride(3); let cp_s4 = checkpoints.stride(4);
    let mut state = Array::<f32>::new(DK);
    let mut i = 0usize;
    while i < DK {
        let index = b * state_in.stride(0) + h * state_in.stride(1) + i * state_in.stride(2) + j * state_in.stride(3);
        state[i] = if token_start == 0 { f32::cast_from(0.0f32) } else { state_in[index] };
        i += 1;
    }
    let mut t = token_start as usize;
    while t < token_end as usize {
        if t % interval == 0 {
            let cp = t / interval;
            let mut i = 0usize;
            while i < DK {
                let ci = b * cp_s0 + h * cp_s1 + cp * cp_s2 + i * cp_s3 + j * cp_s4;
                checkpoints[ci] = state[i];
                i += 1;
            }
        }
        let si = b * doc_start.stride(0) + t * doc_start.stride(1);
        let start = doc_start[si];
        let end = doc_end[b * doc_end.stride(0) + t * doc_end.stride(1)];
        let valid = end >= start;
        let begins = start == I::cast_from(t);
        if !valid {
            out[b * out_s0 + t * out_s1 + h * out_s2 + j * out_s3] = F::cast_from(f32::cast_from(0.0f32));
        } else {
            if begins {
                let mut i = 0usize;
                while i < DK { state[i] = f32::cast_from(0.0f32); i += 1; }
            }
            let mut read = f32::cast_from(0.0f32);
            let mut i = 0usize;
            while i < DK {
                let ai = b * alpha.stride(0) + t * alpha.stride(1) + h * alpha.stride(2) + i * alpha.stride(3);
                let ei = b * erase.stride(0) + t * erase.stride(1) + h * erase.stride(2) + i * erase.stride(3);
                let ki = b * k.stride(0) + t * k.stride(1) + h * k.stride(2) + i * k.stride(3);
                state[i] = state[i] * alpha[ai];
                read += state[i] * erase[ei] * f32::cast_from(k[ki]);
                i += 1;
            }
            let vi = b * v.stride(0) + t * v.stride(1) + h * v.stride(2) + j * v.stride(3);
            let wi = b * write.stride(0) + t * write.stride(1) + h * write.stride(2) + j * write.stride(3);
            let delta = write[wi] * f32::cast_from(v[vi]) - read;
            i = 0usize;
            let mut output = f32::cast_from(0.0f32);
            while i < DK {
                let ki = b * k.stride(0) + t * k.stride(1) + h * k.stride(2) + i * k.stride(3);
                let qi = b * q.stride(0) + t * q.stride(1) + h * q.stride(2) + i * q.stride(3);
                state[i] += f32::cast_from(k[ki]) * delta;
                output += state[i] * f32::cast_from(q[qi]);
                i += 1;
            }
            out[b * out_s0 + t * out_s1 + h * out_s2 + j * out_s3] = F::cast_from(output);
        }
        t += 1;
    }
    let mut i = 0usize;
    while i < DK {
        let index = b * state_out.stride(0) + h * state_out.stride(1) + i * state_out.stride(2) + j * state_out.stride(3);
        state_out[index] = state[i];
        i += 1;
    }
}

#[cube(launch)]
fn backward_kernel<F: Float, I: Int>(
    q: &CubeTensorView<F>, k: &CubeTensorView<F>, v: &CubeTensorView<F>,
    alpha: &CubeTensorView<f32>, erase: &CubeTensorView<f32>, write: &CubeTensorView<f32>,
    output_grad: &CubeTensorView<F>, checkpoints: &CubeTensorView<f32>,
    doc_start: &CubeTensorView<I>, doc_end: &CubeTensorView<I>,
    adjoint_in: &CubeTensorView<f32>, adjoint_out: &mut CubeTensorView<f32>,
    dq_part: &mut CubeTensorView<f32>, dk_part: &mut CubeTensorView<f32>,
    dv: &mut CubeTensorView<F>, da_part: &mut CubeTensorView<f32>,
    de_part: &mut CubeTensorView<f32>, dw: &mut CubeTensorView<f32>,
    scratch: &mut CubeTensorView<f32>,
    #[comptime] interval: usize,
    checkpoint_start: u32, checkpoint_end: u32, token_start: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let item = CUBE_POS;
    let j = item % DV;
    let group = item / DV;
    let b = group / q.shape(2);
    let h = group - b * q.shape(2);
    if b >= q.shape(0) || UNIT_POS != 0 { terminate!(); }
    let sequence = q.shape(1);
    let segment_count = (sequence + interval - 1) / interval;
    let cp_s0 = checkpoints.stride(0); let cp_s1 = checkpoints.stride(1); let cp_s2 = checkpoints.stride(2); let cp_s3 = checkpoints.stride(3); let cp_s4 = checkpoints.stride(4);
    let sc_s0 = scratch.stride(0); let sc_s1 = scratch.stride(1); let sc_s2 = scratch.stride(2); let sc_s3 = scratch.stride(3); let sc_s4 = scratch.stride(4);
    let dq_s0 = dq_part.stride(0); let dq_s1 = dq_part.stride(1); let dq_s2 = dq_part.stride(2); let dq_s3 = dq_part.stride(3); let dq_s4 = dq_part.stride(4);
    let dk_s0 = dk_part.stride(0); let dk_s1 = dk_part.stride(1); let dk_s2 = dk_part.stride(2); let dk_s3 = dk_part.stride(3); let dk_s4 = dk_part.stride(4);
    let da_s0 = da_part.stride(0); let da_s1 = da_part.stride(1); let da_s2 = da_part.stride(2); let da_s3 = da_part.stride(3); let da_s4 = da_part.stride(4);
    let de_s0 = de_part.stride(0); let de_s1 = de_part.stride(1); let de_s2 = de_part.stride(2); let de_s3 = de_part.stride(3); let de_s4 = de_part.stride(4);
    let dv_s0 = dv.stride(0); let dv_s1 = dv.stride(1); let dv_s2 = dv.stride(2); let dv_s3 = dv.stride(3);
    let dw_s0 = dw.stride(0); let dw_s1 = dw.stride(1); let dw_s2 = dw.stride(2); let dw_s3 = dw.stride(3);
    let mut adjoint = Array::<f32>::new(DK);
    let mut state = Array::<f32>::new(DK);
    let mut s_prev = Array::<f32>::new(DK);
    let mut s_bar = Array::<f32>::new(DK);
    let adj_s0 = adjoint_in.stride(0); let adj_s1 = adjoint_in.stride(1); let adj_s2 = adjoint_in.stride(2); let adj_s3 = adjoint_in.stride(3);
    let mut i = 0usize;
    while i < DK {
        let ix = b * adj_s0 + h * adj_s1 + j * adj_s2 + i * adj_s3;
        adjoint[i] = if checkpoint_end as usize == segment_count { f32::cast_from(0.0f32) } else { adjoint_in[ix] };
        i += 1;
    }
    let mut seg = checkpoint_end as usize;
    while seg > checkpoint_start as usize {
        seg -= 1;
            let first = seg * interval;
            let last = if first + interval < sequence { first + interval } else { sequence };
            let count = last - first;
            i = 0usize;
            while i < DK {
                state[i] = checkpoints[b * cp_s0 + h * cp_s1 + seg * cp_s2 + i * cp_s3 + j * cp_s4];
                i += 1;
            }

            // Replay this checkpoint segment and keep each state before its token.
            let mut local_t = 0usize;
            while local_t < count {
                let t = first + local_t;
                let start = doc_start[b * doc_start.stride(0) + t * doc_start.stride(1)];
                let end = doc_end[b * doc_end.stride(0) + t * doc_end.stride(1)];
                if end >= start {
                    if start == I::cast_from(t) {
                        i = 0usize;
                        while i < DK { state[i] = f32::cast_from(0.0f32); i += 1; }
                    }
                    let mut read = f32::cast_from(0.0f32);
                    i = 0usize;
                    while i < DK {
                        scratch[b * sc_s0 + h * sc_s1 + j * sc_s2 + local_t * sc_s3 + i * sc_s4] = state[i];
                        let ai = b * alpha.stride(0) + t * alpha.stride(1) + h * alpha.stride(2) + i * alpha.stride(3);
                        let ei = b * erase.stride(0) + t * erase.stride(1) + h * erase.stride(2) + i * erase.stride(3);
                        let ki = b * k.stride(0) + t * k.stride(1) + h * k.stride(2) + i * k.stride(3);
                        state[i] *= alpha[ai];
                        read += state[i] * erase[ei] * f32::cast_from(k[ki]);
                        i += 1;
                    }
                    let vi = b * v.stride(0) + t * v.stride(1) + h * v.stride(2) + j * v.stride(3);
                    let wi = b * write.stride(0) + t * write.stride(1) + h * write.stride(2) + j * write.stride(3);
                    let delta = write[wi] * f32::cast_from(v[vi]) - read;
                    i = 0usize;
                    while i < DK {
                        let ki = b * k.stride(0) + t * k.stride(1) + h * k.stride(2) + i * k.stride(3);
                        state[i] += f32::cast_from(k[ki]) * delta;
                        i += 1;
                    }
                }
                local_t += 1;
            }

            // Reverse the replayed states with the analytic recurrence adjoint.
            let mut rev = count;
            while rev > 0 {
                rev -= 1;
                let t = first + rev;
                let start = doc_start[b * doc_start.stride(0) + t * doc_start.stride(1)];
                let end = doc_end[b * doc_end.stride(0) + t * doc_end.stride(1)];
                if end >= start {
                    if end == I::cast_from(t) {
                        i = 0usize;
                        while i < DK { adjoint[i] = f32::cast_from(0.0f32); i += 1; }
                    }
                    let mut read = f32::cast_from(0.0f32);
                    i = 0usize;
                    while i < DK {
                        s_prev[i] = scratch[b * sc_s0 + h * sc_s1 + j * sc_s2 + rev * sc_s3 + i * sc_s4];
                        let ai = b * alpha.stride(0) + t * alpha.stride(1) + h * alpha.stride(2) + i * alpha.stride(3);
                        let ei = b * erase.stride(0) + t * erase.stride(1) + h * erase.stride(2) + i * erase.stride(3);
                        let ki = b * k.stride(0) + t * k.stride(1) + h * k.stride(2) + i * k.stride(3);
                        s_bar[i] = alpha[ai] * s_prev[i];
                        read += s_bar[i] * erase[ei] * f32::cast_from(k[ki]);
                        let og = f32::cast_from(output_grad[b * output_grad.stride(0) + t * output_grad.stride(1) + h * output_grad.stride(2) + j * output_grad.stride(3)]);
                        let qi = f32::cast_from(q[b * q.stride(0) + t * q.stride(1) + h * q.stride(2) + i * q.stride(3)]);
                        adjoint[i] += qi * og;
                        i += 1;
                    }
                    let vi = b * v.stride(0) + t * v.stride(1) + h * v.stride(2) + j * v.stride(3);
                    let wi = b * write.stride(0) + t * write.stride(1) + h * write.stride(2) + j * write.stride(3);
                    let delta = write[wi] * f32::cast_from(v[vi]) - read;
                    let og = f32::cast_from(output_grad[b * output_grad.stride(0) + t * output_grad.stride(1) + h * output_grad.stride(2) + j * output_grad.stride(3)]);
                    let mut d_delta = f32::cast_from(0.0f32);
                    i = 0usize;
                    while i < DK {
                        let ki = b * k.stride(0) + t * k.stride(1) + h * k.stride(2) + i * k.stride(3);
                        d_delta += adjoint[i] * f32::cast_from(k[ki]);
                        i += 1;
                    }
                    let v_idx = b * dv_s0 + t * dv_s1 + h * dv_s2 + j * dv_s3;
                    let w_idx = b * dw_s0 + t * dw_s1 + h * dw_s2 + j * dw_s3;
                    dv[v_idx] = F::cast_from(d_delta * write[wi]);
                    dw[w_idx] = d_delta * f32::cast_from(v[vi]);
                    i = 0usize;
                    while i < DK {
                        let ai = b * alpha.stride(0) + t * alpha.stride(1) + h * alpha.stride(2) + i * alpha.stride(3);
                        let ei = b * erase.stride(0) + t * erase.stride(1) + h * erase.stride(2) + i * erase.stride(3);
                        let ki = b * k.stride(0) + t * k.stride(1) + h * k.stride(2) + i * k.stride(3);
                        let state_t = s_bar[i] + f32::cast_from(k[ki]) * delta;
                        let local_t = t - token_start as usize;
                        let q_idx = b * dq_s0 + local_t * dq_s1 + h * dq_s2 + j * dq_s3 + i * dq_s4;
                        let k_idx = b * dk_s0 + local_t * dk_s1 + h * dk_s2 + j * dk_s3 + i * dk_s4;
                        let e_idx = b * de_s0 + local_t * de_s1 + h * de_s2 + j * de_s3 + i * de_s4;
                        let a_idx = b * da_s0 + local_t * da_s1 + h * da_s2 + j * da_s3 + i * da_s4;
                        let d_e = -s_bar[i] * d_delta;
                        let ds = adjoint[i] - erase[ei] * f32::cast_from(k[ki]) * d_delta;
                        dq_part[q_idx] = state_t * og;
                        dk_part[k_idx] = adjoint[i] * delta + d_e * erase[ei];
                        de_part[e_idx] = d_e * f32::cast_from(k[ki]);
                        da_part[a_idx] = ds * s_prev[i];
                        adjoint[i] = alpha[ai] * ds;
                        i += 1;
                    }
                } else {
                    // Padding contributes nothing; the partial buffers are reused across segments.
                    let local_t = t - token_start as usize;
                    i = 0usize;
                    while i < DK {
                        dq_part[b * dq_s0 + local_t * dq_s1 + h * dq_s2 + j * dq_s3 + i * dq_s4] = f32::cast_from(0.0f32);
                        dk_part[b * dk_s0 + local_t * dk_s1 + h * dk_s2 + j * dk_s3 + i * dk_s4] = f32::cast_from(0.0f32);
                        de_part[b * de_s0 + local_t * de_s1 + h * de_s2 + j * de_s3 + i * de_s4] = f32::cast_from(0.0f32);
                        da_part[b * da_s0 + local_t * da_s1 + h * da_s2 + j * da_s3 + i * da_s4] = f32::cast_from(0.0f32);
                        i += 1;
                    }
                }
                }
            }
    let mut i = 0usize;
    while i < DK {
        let ix = b * adjoint_out.stride(0) + h * adjoint_out.stride(1) + j * adjoint_out.stride(2) + i * adjoint_out.stride(3);
        adjoint_out[ix] = adjoint[i];
        i += 1;
    }
}

#[cube(launch)]
fn reduce_gradient_kernel<F: Float>(
    dq_part: &CubeTensorView<f32>, dk_part: &CubeTensorView<f32>,
    da_part: &CubeTensorView<f32>, de_part: &CubeTensorView<f32>,
    dq: &mut CubeTensorView<F>, dk: &mut CubeTensorView<F>,
    da: &mut CubeTensorView<f32>, de: &mut CubeTensorView<f32>,
    token_start: u32, chunk_tokens: u32, tokens: u32,
    #[define(F)] _dtype: ElemType,
) {
    let index = CUBE_POS;
    let key_width = dq.shape(3);
    let values = dq_part.shape(3);
    // The final dispatch segment can be shorter than the checkpoint interval.
    let local_t = index / (key_width * dq.shape(2)) % chunk_tokens as usize;
    let t = local_t + token_start as usize;
    let h = index / key_width % dq.shape(2);
    let b = index / (key_width * dq.shape(2) * chunk_tokens as usize);
    if b >= dq.shape(0) || t >= tokens as usize || UNIT_POS != 0 { terminate!(); }
    let i = index % key_width;
    let mut q_sum = f32::cast_from(0.0f32);
    let mut k_sum = f32::cast_from(0.0f32);
    let mut a_sum = f32::cast_from(0.0f32);
    let mut e_sum = f32::cast_from(0.0f32);
    let mut j = 0usize;
    while j < values {
        let pi = b * dq_part.stride(0) + local_t * dq_part.stride(1) + h * dq_part.stride(2) + j * dq_part.stride(3) + i * dq_part.stride(4);
        q_sum += dq_part[pi];
        k_sum += dk_part[pi];
        a_sum += da_part[pi];
        e_sum += de_part[pi];
        j += 1;
    }
    let oi = b * dq.stride(0) + t * dq.stride(1) + h * dq.stride(2) + i * dq.stride(3);
    dq[oi] = F::cast_from(q_sum);
    dk[oi] = F::cast_from(k_sum);
    da[oi] = a_sum;
    de[oi] = e_sum;
}

pub(crate) fn run_scan(q: CubeTensor, k: CubeTensor, v: CubeTensor, alpha: CubeTensor, erase: CubeTensor, write: CubeTensor, doc_start: CubeTensor, doc_end: CubeTensor, interval: usize) -> (CubeTensor, CubeTensor) {
    let [batch, tokens, heads, value_width] = v.shape().dims::<4>();
    let [_, _, _, key_width] = q.shape().dims::<4>();
    assert_eq!(key_width, DK);
    assert_eq!(value_width, DV);
    assert!(interval > 0);
    let client = q.client.clone();
    let device = q.device.clone();
    let dtype = q.dtype;
    let int_dtype = doc_start.dtype;
    let output = CubeBackend::float_zeros(v.shape(), &device, dtype.into());
    let checkpoint_count = (tokens + interval - 1) / interval;
    let checkpoint = CubeBackend::float_zeros(Shape::new([batch, heads, checkpoint_count, DK, DV]), &device, burn::tensor::DType::F32.into());
    let dim = CubeDim::new(&client, 1);
    let count = cube_count(&client, batch * heads * DV);
    let state_shape = Shape::new([batch, heads, DK, DV]);
    let mut state_in = CubeBackend::float_zeros(state_shape.clone(), &device, burn::tensor::DType::F32.into());
    let mut state_out = CubeBackend::float_zeros(state_shape, &device, burn::tensor::DType::F32.into());
    let mut token_start = 0usize;
    while token_start < tokens {
        let token_end = (token_start + interval).min(tokens);
        scan_kernel::launch(
            &client, count.clone(), dim, q.clone().into_tensor_arg(), k.clone().into_tensor_arg(),
            v.clone().into_tensor_arg(), alpha.clone().into_tensor_arg(), erase.clone().into_tensor_arg(),
            write.clone().into_tensor_arg(), doc_start.clone().into_tensor_arg(), doc_end.clone().into_tensor_arg(),
            state_in.clone().into_tensor_arg(), output.clone().into_tensor_arg(), checkpoint.clone().into_tensor_arg(),
            state_out.clone().into_tensor_arg(), interval, token_start as u32, token_end as u32,
            [dtype_to_storage_type(dtype), dtype_to_storage_type(int_dtype)],
        );
        core::mem::swap(&mut state_in, &mut state_out);
        token_start = token_end;
    }
    (output, checkpoint)
}

pub(crate) fn run_backward(q: CubeTensor, k: CubeTensor, v: CubeTensor, alpha: CubeTensor, erase: CubeTensor, write: CubeTensor, output_grad: CubeTensor, checkpoint: CubeTensor, doc_start: CubeTensor, doc_end: CubeTensor, interval: usize) -> (CubeTensor, CubeTensor, CubeTensor, CubeTensor, CubeTensor, CubeTensor) {
    let [batch, tokens, heads, value_width] = v.shape().dims::<4>();
    let key_width = q.shape().dims::<4>()[3];
    let client = q.client.clone();
    let device = q.device.clone();
    let dtype = q.dtype;
    let int_dtype = doc_start.dtype;
    let dq = CubeBackend::float_zeros(q.shape(), &device, dtype.into());
    let dk = CubeBackend::float_zeros(k.shape(), &device, dtype.into());
    let dv = CubeBackend::float_zeros(v.shape(), &device, dtype.into());
    let da = CubeBackend::float_zeros(alpha.shape(), &device, burn::tensor::DType::F32.into());
    let de = CubeBackend::float_zeros(erase.shape(), &device, burn::tensor::DType::F32.into());
    let dw = CubeBackend::float_zeros(write.shape(), &device, burn::tensor::DType::F32.into());
    let partial_shape = Shape::new([batch, interval, heads, value_width, key_width]);
    let dq_part = CubeBackend::float_zeros(partial_shape.clone(), &device, burn::tensor::DType::F32.into());
    let dk_part = CubeBackend::float_zeros(partial_shape.clone(), &device, burn::tensor::DType::F32.into());
    let da_part = CubeBackend::float_zeros(partial_shape.clone(), &device, burn::tensor::DType::F32.into());
    let de_part = CubeBackend::float_zeros(partial_shape, &device, burn::tensor::DType::F32.into());
    let scratch = CubeBackend::float_zeros(Shape::new([batch, heads, value_width, interval, key_width]), &device, burn::tensor::DType::F32.into());
    let dim = CubeDim::new(&client, 1);
    let count = cube_count(&client, batch * heads * value_width);
    let adjoint_shape = Shape::new([batch, heads, value_width, key_width]);
    let mut adjoint_in = CubeBackend::float_zeros(adjoint_shape.clone(), &device, burn::tensor::DType::F32.into());
    let mut adjoint_out = CubeBackend::float_zeros(adjoint_shape, &device, burn::tensor::DType::F32.into());
    let checkpoint_count = (tokens + interval - 1) / interval;
    let mut checkpoint_end = checkpoint_count;
    while checkpoint_end > 0 {
        let checkpoint_start = checkpoint_end - 1;
        let token_start = checkpoint_start * interval;
        let chunk_tokens = (tokens - token_start).min(interval);
        backward_kernel::launch(
            &client, count.clone(), dim, q.clone().into_tensor_arg(), k.clone().into_tensor_arg(),
            v.clone().into_tensor_arg(), alpha.clone().into_tensor_arg(), erase.clone().into_tensor_arg(),
            write.clone().into_tensor_arg(), output_grad.clone().into_tensor_arg(), checkpoint.clone().into_tensor_arg(),
            doc_start.clone().into_tensor_arg(), doc_end.clone().into_tensor_arg(), adjoint_in.clone().into_tensor_arg(),
            adjoint_out.clone().into_tensor_arg(), dq_part.clone().into_tensor_arg(), dk_part.clone().into_tensor_arg(),
            dv.clone().into_tensor_arg(), da_part.clone().into_tensor_arg(), de_part.clone().into_tensor_arg(),
            dw.clone().into_tensor_arg(), scratch.clone().into_tensor_arg(), interval,
            checkpoint_start as u32, checkpoint_end as u32, token_start as u32,
            [dtype_to_storage_type(dtype), dtype_to_storage_type(int_dtype)],
        );
        let reduce_count = cube_count(&client, batch * chunk_tokens * heads * key_width);
        reduce_gradient_kernel::launch(&client, reduce_count, dim, dq_part.clone().into_tensor_arg(), dk_part.clone().into_tensor_arg(), da_part.clone().into_tensor_arg(), de_part.clone().into_tensor_arg(), dq.clone().into_tensor_arg(), dk.clone().into_tensor_arg(), da.clone().into_tensor_arg(), de.clone().into_tensor_arg(), token_start as u32, chunk_tokens as u32, tokens as u32, dtype_to_storage_type(dtype));
        core::mem::swap(&mut adjoint_in, &mut adjoint_out);
        checkpoint_end = checkpoint_start;
    }
    (dq, dk, dv, da, de, dw)
}
