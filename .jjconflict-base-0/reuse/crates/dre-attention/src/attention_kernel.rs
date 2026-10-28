use burn::backend::{Backend, AutodiffBackend, backend_extension, autodiff::{Autodiff, checkpoint::{base::Checkpointer, strategy::CheckpointStrategy}, grads::Gradients, ops::{Backward, Ops, OpsKind}}, cubecl::dtype_to_storage_type, fusion::custom::TensorSpec, ops::FloatTensorOps, tensor::{FloatTensor, IntTensor}, wgpu::{CubeBackend, CubeTensor}};
use burn::backend::TensorMetadata;
use burn::cubecl::prelude::*;
use burn::prelude::Shape;
use cubecl::frontend::{Array, Shared, Tensor as CubeTensorView};

fn attention_cube_count_with_limits(cubes: usize, max_x: u32, max_y: u32) -> CubeCount {
    let x = cubes.min(max_x.min(u16::MAX as u32) as usize);
    let y = if x == 0 { 1 } else { cubes.div_ceil(x) };
    assert!(y <= max_y as usize, "attention dispatch of {cubes} cubes exceeds the device workgroup limits");
    CubeCount::Static(x as u32, y as u32, 1)
}

#[cfg(test)]
static TEST_MAX_X: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
#[cfg(test)]
static TEST_SPLIT_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static TEST_LAST_LOGICAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static TEST_LAST_X: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static TEST_LAST_Y: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static TEST_LAST_EXCESS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TestDispatchGeometry {
    pub(crate) logical_groups: usize,
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) excess_groups: usize,
}

#[cfg(test)]
pub(crate) struct TestDispatchLimitGuard;

#[cfg(test)]
impl TestDispatchLimitGuard {
    pub(crate) fn new(limit: u32) -> Self {
        TEST_SPLIT_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
        TEST_LAST_LOGICAL.store(0, std::sync::atomic::Ordering::Relaxed);
        TEST_LAST_X.store(0, std::sync::atomic::Ordering::Relaxed);
        TEST_LAST_Y.store(0, std::sync::atomic::Ordering::Relaxed);
        TEST_LAST_EXCESS.store(0, std::sync::atomic::Ordering::Relaxed);
        TEST_MAX_X.store(limit, std::sync::atomic::Ordering::Relaxed);
        Self
    }

    pub(crate) fn dispatch_stats(&self) -> (usize, TestDispatchGeometry) {
        (
            TEST_SPLIT_CALLS.load(std::sync::atomic::Ordering::Relaxed),
            TestDispatchGeometry {
                logical_groups: TEST_LAST_LOGICAL.load(std::sync::atomic::Ordering::Relaxed),
                x: TEST_LAST_X.load(std::sync::atomic::Ordering::Relaxed),
                y: TEST_LAST_Y.load(std::sync::atomic::Ordering::Relaxed),
                excess_groups: TEST_LAST_EXCESS.load(std::sync::atomic::Ordering::Relaxed),
            },
        )
    }
}

#[cfg(test)]
impl Drop for TestDispatchLimitGuard {
    fn drop(&mut self) {
        TEST_MAX_X.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

pub(super) fn attention_cube_count(client: &Client, cubes: usize) -> CubeCount {
    let (max_x, max_y, _) = client.properties().hardware.max_cube_count;
    assert!(client.properties().hardware.max_shared_memory_size >= 3 * 32 * 64 * 4,
        "attention backward needs 24 KiB of workgroup shared memory");
    #[cfg(test)]
    let max_x = {
        let test_limit = TEST_MAX_X.load(std::sync::atomic::Ordering::Relaxed);
        if test_limit == 0 { max_x } else { max_x.min(test_limit) }
    };
    let count = attention_cube_count_with_limits(cubes, max_x, max_y);
    #[cfg(test)]
    {
        let x = cubes.min(max_x.min(u16::MAX as u32) as usize);
        let y = if x == 0 { 1 } else { cubes.div_ceil(x) };
        let excess = x * y - cubes;
        if excess > 0 {
            TEST_LAST_LOGICAL.store(cubes, std::sync::atomic::Ordering::Relaxed);
            TEST_LAST_X.store(x, std::sync::atomic::Ordering::Relaxed);
            TEST_LAST_Y.store(y, std::sync::atomic::Ordering::Relaxed);
            TEST_LAST_EXCESS.store(excess, std::sync::atomic::Ordering::Relaxed);
            TEST_SPLIT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    count
}


fn forward_meta(q: &TensorSpec, _k: &TensorSpec, _v: &TensorSpec, _start: &TensorSpec, _end: &TensorSpec) -> (TensorSpec, TensorSpec) {
    (q.clone(), TensorSpec::new(Shape::new([q.shape[0], q.shape[1], q.shape[2], 2]), burn::tensor::DType::F32))
}
fn backward_meta(q: &TensorSpec, k: &TensorSpec, v: &TensorSpec, _start: &TensorSpec, _end: &TensorSpec, _o: &TensorSpec, _normalization: &TensorSpec, _do: &TensorSpec) -> (TensorSpec, TensorSpec, TensorSpec) {
    (q.clone(), k.clone(), v.clone())
}

#[backend_extension(Cube, Autodiff, Fusion)]
pub(crate) trait AttentionKernelOps: Backend {
    #[fusion(meta = forward_meta)]
    fn attention_forward(q: FloatTensor<Self>, k: FloatTensor<Self>, v: FloatTensor<Self>, doc_start: IntTensor<Self>, doc_end: IntTensor<Self>) -> (FloatTensor<Self>, FloatTensor<Self>);
    #[fusion(meta = backward_meta)]
    fn attention_backward(q: FloatTensor<Self>, k: FloatTensor<Self>, v: FloatTensor<Self>, doc_start: IntTensor<Self>, doc_end: IntTensor<Self>, o: FloatTensor<Self>, normalization: FloatTensor<Self>, grad: FloatTensor<Self>) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);
}


#[cube(launch)]
fn forward_kernel<F: Float, I: Int>(q: &CubeTensorView<F>, k: &CubeTensorView<F>, v: &CubeTensorView<F>, doc_start: &CubeTensorView<I>, doc_end: &CubeTensorView<I>, o: &mut CubeTensorView<F>, normalization: &mut CubeTensorView<f32>, #[define(F, I)] _dtypes: [ElemType; 2]) {
    let heads = q.shape(1); let tokens = q.shape(2);
    let tiles = (tokens + 31usize) / 32usize;
    let cube = CUBE_POS as usize;
    let groups = q.shape(0) * heads * tiles;
    if cube >= groups { terminate!(); }
    let b = cube / (heads * tiles);
    let head = (cube / tiles) % heads;
    let tile = cube % tiles;
    let query_base = tile * 32usize;
    let query = query_base + UNIT_POS_X as usize;
    let query_valid = query < tokens;
    let mut start = I::cast_from(0u32);
    let mut end = I::cast_from(0u32);
    if query_valid {
        start = I::cast_from(doc_start[b * doc_start.stride(0) + query * doc_start.stride(1)]);
        end = I::cast_from(doc_end[b * doc_end.stride(0) + query * doc_end.stride(1)]);
    }
    let mut shared_k = Shared::<[f32]>::new_slice(32usize * 64usize);
    let mut shared_v = Shared::<[f32]>::new_slice(32usize * 64usize);
    let window_lo = if query_base > 128usize { query_base - 128usize } else { 0usize };
    let last_query = if query_base + 31usize < tokens { query_base + 31usize } else { tokens - 1usize };
    let window_hi = if last_query + 128usize < tokens { last_query + 128usize } else { tokens - 1usize };
    let mut acc = Array::<f32>::new(64usize);
    let mut dim = 0usize;
    while dim < 64usize { acc[dim] = f32::cast_from(0.0_f32); dim += 1usize; }
    // The first eligible score initializes the maximum; no finite score floor is valid.
    let mut maximum = f32::cast_from(0.0_f32);
    let mut total = f32::cast_from(0.0_f32);
    let mut key_base = window_lo;
    while key_base <= window_hi {
        let mut load = UNIT_POS_X as usize;
        while load < 32usize * 64usize {
            let key = key_base + load / 64usize;
            let component = load % 64usize;
            let offset = b * k.stride(0) + head * k.stride(1) + key * k.stride(2) + component * k.stride(3);
            let value_offset = b * v.stride(0) + head * v.stride(1) + key * v.stride(2) + component * v.stride(3);
            if key < tokens {
                shared_k[load] = f32::cast_from(k[offset]);
                shared_v[load] = f32::cast_from(v[value_offset]);
            } else {
                shared_k[load] = f32::cast_from(0.0_f32);
                shared_v[load] = f32::cast_from(0.0_f32);
            }
            load += CUBE_DIM_X as usize;
        }
        sync_cube();
        let mut tile_key = 0usize;
        while tile_key < 32usize {
            let key = key_base + tile_key;
            if query_valid && end >= start && key < tokens && I::cast_from(key as u32) >= start && I::cast_from(key as u32) <= end && key + 128usize >= query && query + 128usize >= key {
                let query_offset = b * q.stride(0) + head * q.stride(1) + query * q.stride(2);
                let mut dot = f32::cast_from(0.0_f32);
                dim = 0usize;
                while dim < 64usize { dot += f32::cast_from(q[query_offset + dim * q.stride(3)]) * shared_k[tile_key * 64usize + dim]; dim += 1usize; }
                let score = dot * f32::cast_from(0.125_f32);
                let next_max = if total == f32::cast_from(0.0_f32) || score > maximum { score } else { maximum };
                let alpha = if total == f32::cast_from(0.0_f32) { f32::cast_from(0.0_f32) } else { (maximum - next_max).exp() };
                let probability = (score - next_max).exp();
                total = total * alpha + probability;
                dim = 0usize;
                while dim < 64usize { acc[dim] = acc[dim] * alpha + probability * shared_v[tile_key * 64usize + dim]; dim += 1usize; }
                maximum = next_max;
            }
            tile_key += 1usize;
        }
        sync_cube();
        key_base += 32usize;
    }
    if query_valid {
        let row = (b * heads + head) * tokens + query;
        let normalization_offset = b * normalization.stride(0) + head * normalization.stride(1) + query * normalization.stride(2);
        let total_offset = normalization_offset + normalization.stride(3);
        let output = row * 64usize;
        if end < start {
            dim = 0usize; while dim < 64usize { o[output + dim] = F::cast_from(f32::cast_from(0.0_f32)); dim += 1usize; }
            normalization[normalization_offset] = f32::cast_from(0.0_f32);
            normalization[total_offset] = f32::cast_from(1.0_f32);
        } else {
            dim = 0usize; while dim < 64usize { o[output + dim] = F::cast_from(acc[dim] / total); dim += 1usize; }
            // Keep the small normalizer separate: maximum + ln(total) loses it at extreme scores.
            normalization[normalization_offset] = maximum;
            normalization[total_offset] = total;
        }
    }
}

#[cube(launch)]
fn query_backward_kernel<F: Float, I: Int>(q: &CubeTensorView<F>, k: &CubeTensorView<F>, v: &CubeTensorView<F>, doc_start: &CubeTensorView<I>, doc_end: &CubeTensorView<I>, o: &CubeTensorView<F>, normalization: &CubeTensorView<f32>, grad: &CubeTensorView<F>, dq: &mut CubeTensorView<F>, #[define(F, I)] _dtypes: [ElemType; 2]) {
    let heads = q.shape(1); let tokens = q.shape(2);
    let tiles = (tokens + 31usize) / 32usize;
    let cube = CUBE_POS as usize;
    let groups = q.shape(0) * heads * tiles;
    if cube >= groups { terminate!(); }
    let b = cube / (heads * tiles);
    let head = (cube / tiles) % heads;
    let query_tile = cube % tiles;
    let query_base = query_tile * 32usize;
    let query = query_base + UNIT_POS_X as usize;
    let query_valid = query < tokens;
    let mut start = I::cast_from(0u32);
    let mut end = I::cast_from(0u32);
    if query_valid {
        start = I::cast_from(doc_start[b * doc_start.stride(0) + query * doc_start.stride(1)]);
        end = I::cast_from(doc_end[b * doc_end.stride(0) + query * doc_end.stride(1)]);
    }
    let mut shared_k = Shared::<[f32]>::new_slice(32usize * 64usize);
    let mut shared_v = Shared::<[f32]>::new_slice(32usize * 64usize);
    let row = (b * heads + head) * tokens + query;
    let normalization_offset = b * normalization.stride(0) + head * normalization.stride(1) + query * normalization.stride(2);
    let go = b * grad.stride(0) + head * grad.stride(1) + query * grad.stride(2);
    let oo = row * 64usize;
    let mut correction = f32::cast_from(0.0_f32);
    let mut dim = 0usize;
    if query_valid {
        while dim < 64usize { correction += f32::cast_from(grad[go + dim * grad.stride(3)]) * f32::cast_from(o[oo + dim]); dim += 1usize; }
    }
    let window_lo = if query_base > 128usize { query_base - 128usize } else { 0usize };
    let last_query = if query_base + 31usize < tokens { query_base + 31usize } else { tokens - 1usize };
    let window_hi = if last_query + 128usize < tokens { last_query + 128usize } else { tokens - 1usize };
    let mut accum = Array::<f32>::new(64usize);
    dim = 0usize; while dim < 64usize { accum[dim] = f32::cast_from(0.0_f32); dim += 1usize; }
    let mut key_base = window_lo;
    while key_base <= window_hi {
        let mut load = UNIT_POS_X as usize;
        while load < 32usize * 64usize {
            let key = key_base + load / 64usize;
            let component = load % 64usize;
            let offset = b * k.stride(0) + head * k.stride(1) + key * k.stride(2) + component * k.stride(3);
            let value_offset = b * v.stride(0) + head * v.stride(1) + key * v.stride(2) + component * v.stride(3);
            if key < tokens { shared_k[load] = f32::cast_from(k[offset]); shared_v[load] = f32::cast_from(v[value_offset]); }
            else { shared_k[load] = f32::cast_from(0.0_f32); shared_v[load] = f32::cast_from(0.0_f32); }
            load += CUBE_DIM_X as usize;
        }
        sync_cube();
        let mut tile_key = 0usize;
        while tile_key < 32usize {
            let key = key_base + tile_key;
            if query_valid && end >= start && key < tokens && I::cast_from(key as u32) >= start && I::cast_from(key as u32) <= end && key + 128usize >= query && query + 128usize >= key {
                let qo = b * q.stride(0) + head * q.stride(1) + query * q.stride(2);
                let mut dot = f32::cast_from(0.0_f32); dim = 0usize;
                while dim < 64usize { dot += f32::cast_from(q[qo + dim * q.stride(3)]) * shared_k[tile_key * 64usize + dim]; dim += 1usize; }
                let probability = (dot * f32::cast_from(0.125_f32) - normalization[normalization_offset]).exp() / normalization[normalization_offset + normalization.stride(3)];
                let mut dv = f32::cast_from(0.0_f32); dim = 0usize;
                while dim < 64usize { dv += f32::cast_from(grad[go + dim * grad.stride(3)]) * shared_v[tile_key * 64usize + dim]; dim += 1usize; }
                let ds = probability * (dv - correction) * f32::cast_from(0.125_f32);
                dim = 0usize;
                while dim < 64usize { accum[dim] += ds * shared_k[tile_key * 64usize + dim]; dim += 1usize; }
            }
            tile_key += 1usize;
        }
        sync_cube();
        key_base += 32usize;
    }
    if query_valid { dim = 0usize; while dim < 64usize { dq[row * 64usize + dim] = F::cast_from(accum[dim]); dim += 1usize; } }
}

#[cube(launch)]
pub(super) fn key_backward_kernel<F: Float, I: Int>(q: &CubeTensorView<F>, k: &CubeTensorView<F>, v: &CubeTensorView<F>, doc_start: &CubeTensorView<I>, doc_end: &CubeTensorView<I>, o: &CubeTensorView<F>, normalization: &CubeTensorView<f32>, grad: &CubeTensorView<F>, dk: &mut CubeTensorView<F>, dv: &mut CubeTensorView<F>, #[define(F, I)] _dtypes: [ElemType; 2]) {
    let heads = q.shape(1); let tokens = q.shape(2);
    let tiles = (tokens + 31usize) / 32usize;
    let cube = CUBE_POS as usize;
    let groups = q.shape(0) * heads * tiles;
    if cube >= groups { terminate!(); }
    let b = cube / (heads * tiles);
    let head = (cube / tiles) % heads;
    let key_tile = cube % tiles;
    let key_base = key_tile * 32usize;
    let token = key_base + UNIT_POS_X as usize;
    let key_valid = token < tokens;
    let mut key_vector = Array::<f32>::new(64usize);
    let mut value_vector = Array::<f32>::new(64usize);
    let mut dim = 0usize;
    while dim < 64usize {
        key_vector[dim] = f32::cast_from(0.0_f32);
        value_vector[dim] = f32::cast_from(0.0_f32);
        if key_valid {
            let offset = b * k.stride(0) + head * k.stride(1) + token * k.stride(2) + dim * k.stride(3);
            let value_offset = b * v.stride(0) + head * v.stride(1) + token * v.stride(2) + dim * v.stride(3);
            key_vector[dim] = f32::cast_from(k[offset]);
            value_vector[dim] = f32::cast_from(v[value_offset]);
        }
        dim += 1usize;
    }
    let mut shared_q = Shared::<[f32]>::new_slice(32usize * 64usize);
    let mut shared_o = Shared::<[f32]>::new_slice(32usize * 64usize);
    let mut shared_grad = Shared::<[f32]>::new_slice(32usize * 64usize);
    let mut accum_k = Array::<f32>::new(64usize);
    let mut accum_v = Array::<f32>::new(64usize);
    dim = 0usize;
    while dim < 64usize { accum_k[dim] = f32::cast_from(0.0_f32); accum_v[dim] = f32::cast_from(0.0_f32); dim += 1usize; }
    let query_lo = if key_base > 128usize { key_base - 128usize } else { 0usize };
    let last_key = if key_base + 31usize < tokens { key_base + 31usize } else { tokens - 1usize };
    let query_hi = if last_key + 128usize < tokens { last_key + 128usize } else { tokens - 1usize };
    let mut query_base = query_lo;
    while query_base <= query_hi {
        let mut load = UNIT_POS_X as usize;
        while load < 32usize * 64usize {
            let query = query_base + load / 64usize;
            let component = load % 64usize;
            if query < tokens {
                let row = (b * heads + head) * tokens + query;
                let q_offset = b * q.stride(0) + head * q.stride(1) + query * q.stride(2) + component * q.stride(3);
                let grad_base = b * grad.stride(0) + head * grad.stride(1) + query * grad.stride(2);
                shared_q[load] = f32::cast_from(q[q_offset]);
                shared_o[load] = f32::cast_from(o[row * 64usize + component]);
                shared_grad[load] = f32::cast_from(grad[grad_base + component * grad.stride(3)]);
            } else {
                shared_q[load] = f32::cast_from(0.0_f32);
                shared_o[load] = f32::cast_from(0.0_f32);
                shared_grad[load] = f32::cast_from(0.0_f32);
            }
            load += CUBE_DIM_X as usize;
        }
        sync_cube();
        let mut query_lane = 0usize;
        while query_lane < 32usize {
            let query = query_base + query_lane;
            if key_valid && query < tokens && token + 128usize >= query && query + 128usize >= token && I::cast_from(token as u32) >= I::cast_from(doc_start[b * doc_start.stride(0) + query * doc_start.stride(1)]) && I::cast_from(token as u32) <= I::cast_from(doc_end[b * doc_end.stride(0) + query * doc_end.stride(1)]) {
                let normalization_offset = b * normalization.stride(0) + head * normalization.stride(1) + query * normalization.stride(2);
                let mut dot = f32::cast_from(0.0_f32);
                let mut correction = f32::cast_from(0.0_f32);
                let mut grad_value = f32::cast_from(0.0_f32);
                dim = 0usize;
                while dim < 64usize {
                    dot += shared_q[query_lane * 64usize + dim] * key_vector[dim];
                    correction += shared_grad[query_lane * 64usize + dim] * shared_o[query_lane * 64usize + dim];
                    grad_value += shared_grad[query_lane * 64usize + dim] * value_vector[dim];
                    dim += 1usize;
                }
                let probability = (dot * f32::cast_from(0.125_f32) - normalization[normalization_offset]).exp() / normalization[normalization_offset + normalization.stride(3)];
                let ds = probability * (grad_value - correction) * f32::cast_from(0.125_f32);
                dim = 0usize;
                while dim < 64usize {
                    accum_k[dim] += ds * shared_q[query_lane * 64usize + dim];
                    accum_v[dim] += probability * shared_grad[query_lane * 64usize + dim];
                    dim += 1usize;
                }
            }
            query_lane += 1usize;
        }
        sync_cube();
        query_base += 32usize;
    }
    if key_valid {
        let output = ((b * heads + head) * tokens + token) * 64usize;
        dim = 0usize;
        while dim < 64usize { dk[output + dim] = F::cast_from(accum_k[dim]); dv[output + dim] = F::cast_from(accum_v[dim]); dim += 1usize; }
    }
}

fn forward_cube(q: CubeTensor, k: CubeTensor, v: CubeTensor, starts: CubeTensor, ends: CubeTensor) -> (CubeTensor, CubeTensor) {
    let [batch, heads, tokens, width] = q.shape().dims::<4>();
    assert_eq!(width, 64);
    let groups = batch * heads * tokens.div_ceil(32);
    let client = q.client.clone(); let device = q.device.clone(); let dtype = q.dtype;
    let int_dtype = starts.dtype;
    let output = CubeBackend::float_zeros(q.shape(), &device, dtype.into());
    let normalization = CubeBackend::float_zeros(Shape::new([batch, heads, tokens, 2]), &device, burn::tensor::DType::F32.into());
    let count = attention_cube_count(&client, groups); let dim = CubeDim::new_1d(32);
    forward_kernel::launch(&client, count, dim, q.into_tensor_arg(), k.into_tensor_arg(), v.into_tensor_arg(), starts.into_tensor_arg(), ends.into_tensor_arg(), output.clone().into_tensor_arg(), normalization.clone().into_tensor_arg(), [dtype_to_storage_type(dtype), dtype_to_storage_type(int_dtype)]);
    (output, normalization)
}
fn backward_cube(q: CubeTensor, k: CubeTensor, v: CubeTensor, starts: CubeTensor, ends: CubeTensor, o: CubeTensor, normalization: CubeTensor, grad: CubeTensor) -> (CubeTensor, CubeTensor, CubeTensor) {
    let [batch, heads, tokens, width] = q.shape().dims::<4>(); assert_eq!(width, 64);
    let client = q.client.clone(); let device = q.device.clone(); let dtype = q.dtype;
    let dq = CubeBackend::float_zeros(q.shape(), &device, dtype.into()); let dk = CubeBackend::float_zeros(k.shape(), &device, dtype.into()); let dv = CubeBackend::float_zeros(v.shape(), &device, dtype.into());
    let groups = batch * heads * tokens.div_ceil(32);
    let count = attention_cube_count(&client, groups); let dim = CubeDim::new_1d(32); let types = [dtype_to_storage_type(dtype), dtype_to_storage_type(starts.dtype)];
    query_backward_kernel::launch(&client, count.clone(), dim, q.clone().into_tensor_arg(), k.clone().into_tensor_arg(), v.clone().into_tensor_arg(), starts.clone().into_tensor_arg(), ends.clone().into_tensor_arg(), o.clone().into_tensor_arg(), normalization.clone().into_tensor_arg(), grad.clone().into_tensor_arg(), dq.clone().into_tensor_arg(), types);
    key_backward_kernel::launch(&client, count, dim, q.into_tensor_arg(), k.into_tensor_arg(), v.into_tensor_arg(), starts.into_tensor_arg(), ends.into_tensor_arg(), o.into_tensor_arg(), normalization.into_tensor_arg(), grad.into_tensor_arg(), dk.clone().into_tensor_arg(), dv.clone().into_tensor_arg(), types);
    (dq, dk, dv)
}

impl AttentionKernelOps for CubeBackend {
    fn attention_forward(q: CubeTensor, k: CubeTensor, v: CubeTensor, doc_start: CubeTensor, doc_end: CubeTensor) -> (CubeTensor, CubeTensor) { forward_cube(q, k, v, doc_start, doc_end) }
    fn attention_backward(q: CubeTensor, k: CubeTensor, v: CubeTensor, doc_start: CubeTensor, doc_end: CubeTensor, o: CubeTensor, normalization: CubeTensor, grad: CubeTensor) -> (CubeTensor, CubeTensor, CubeTensor) { backward_cube(q, k, v, doc_start, doc_end, o, normalization, grad) }
}

impl<B: Backend + AttentionKernelOps, C: CheckpointStrategy> AttentionKernelOps for Autodiff<B, C> {
    fn attention_forward(q: FloatTensor<Self>, k: FloatTensor<Self>, v: FloatTensor<Self>, doc_start: IntTensor<Self>, doc_end: IntTensor<Self>) -> (FloatTensor<Self>, FloatTensor<Self>) {
        #[derive(Debug)] struct AttentionBackward;
        impl<B: Backend + AttentionKernelOps> Backward<B, 3> for AttentionBackward {
            type State = (B::FloatTensorPrimitive, B::FloatTensorPrimitive, B::FloatTensorPrimitive, B::IntTensorPrimitive, B::IntTensorPrimitive, B::FloatTensorPrimitive, B::FloatTensorPrimitive);
            fn backward(self, ops: Ops<Self::State, 3>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
                let [nq, nk, nv] = ops.parents; let grad = grads.consume::<B>(&ops.node);
                let (q, k, v, starts, ends, o, normalization) = ops.state;
                let (dq, dk, dv) = B::attention_backward(q, k, v, starts, ends, o, normalization, grad);
                if let Some(n) = nq { grads.register::<B>(n.id, dq); }
                if let Some(n) = nk { grads.register::<B>(n.id, dk); }
                if let Some(n) = nv { grads.register::<B>(n.id, dv); }
            }
        }
        let nq = q.node(); let nk = k.node(); let nv = v.node();
        let (q, k, v) = (q.into_primitive(), k.into_primitive(), v.into_primitive());
        let (o, normalization) = B::attention_forward(q.clone(), k.clone(), v.clone(), doc_start.clone(), doc_end.clone());
        let state = (q, k, v, doc_start, doc_end, o.clone(), normalization.clone());
        match AttentionBackward.prepare::<C>([nq, nk, nv]).compute_bound().stateful() {
            OpsKind::Tracked(prep) => (prep.finish(state, o), <Autodiff<B, C> as AutodiffBackend>::from_inner(normalization)),
            OpsKind::UnTracked(prep) => (prep.finish(o), <Autodiff<B, C> as AutodiffBackend>::from_inner(normalization)),
        }
    }
    fn attention_backward(q: FloatTensor<Self>, k: FloatTensor<Self>, v: FloatTensor<Self>, doc_start: IntTensor<Self>, doc_end: IntTensor<Self>, o: FloatTensor<Self>, normalization: FloatTensor<Self>, grad: FloatTensor<Self>) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let q = <Self as AutodiffBackend>::inner(q); let k = <Self as AutodiffBackend>::inner(k); let v = <Self as AutodiffBackend>::inner(v);
        let o = <Self as AutodiffBackend>::inner(o); let normalization = <Self as AutodiffBackend>::inner(normalization); let grad = <Self as AutodiffBackend>::inner(grad);
        let (dq, dk, dv) = B::attention_backward(q, k, v, doc_start, doc_end, o, normalization, grad);
        (<Self as AutodiffBackend>::from_inner(dq), <Self as AutodiffBackend>::from_inner(dk), <Self as AutodiffBackend>::from_inner(dv))
    }
}

/// Returns output and FP32 `[row maximum, sum(exp(score - maximum))]` per query.
pub(crate) fn run_attention(q: burn::Tensor<4>, k: burn::Tensor<4>, v: burn::Tensor<4>, layout: &dre_core::BatchLayout, radius: usize) -> (burn::Tensor<4>, burn::Tensor<4>) {
    assert_eq!(radius, 128, "the local attention kernel uses radius 128");
    let (output, normalization) = <burn::backend::Dispatch as AttentionKernelOps>::attention_forward(q.into_dispatch(), k.into_dispatch(), v.into_dispatch(), layout.doc_start.clone().into_dispatch(), layout.doc_end.clone().into_dispatch());
    (burn::Tensor::from_dispatch(output), burn::Tensor::from_dispatch(normalization))
}
#[cfg(test)]
mod dispatch_tests {
    use burn::cubecl::prelude::CubeCount;
    use super::attention_cube_count_with_limits;

    #[test]
    fn source_attention_dispatch_splits_at_65535_workgroups() {
        let logical_workgroups = 65_536usize;
        let count = attention_cube_count_with_limits(logical_workgroups, 65_535, 65_535);
        assert!(matches!(count, CubeCount::Static(65_535, 2, 1)));
        let launched = 65_535usize * 2;
        assert!(launched > logical_workgroups);
        assert_eq!(launched - logical_workgroups, 65_534);
    }
}
