use burn::backend::{Backend, AutodiffBackend, backend_extension, autodiff::{Autodiff, checkpoint::{base::Checkpointer, strategy::CheckpointStrategy}, grads::Gradients, ops::{Backward, Ops, OpsKind}}, cubecl::dtype_to_storage_type, fusion::custom::TensorSpec, ops::FloatTensorOps, tensor::{FloatTensor, IntTensor}, wgpu::{CubeBackend, CubeTensor}};
use burn::backend::TensorMetadata;
use burn::cubecl::prelude::*;
use burn::prelude::Shape;
use cubecl::frontend::{Array, Tensor as CubeTensorView};

fn relay_forward_meta(q: &TensorSpec, _k: &TensorSpec, _v: &TensorSpec, _rk: &TensorSpec, _rv: &TensorSpec, _ids: &TensorSpec, _start: &TensorSpec, _end: &TensorSpec, _rows: &TensorSpec, _starts: &TensorSpec, _ends: &TensorSpec) -> (TensorSpec, TensorSpec) {
    (q.clone(), TensorSpec::new(Shape::new([q.shape[0], q.shape[1], q.shape[2], 2]), burn::tensor::DType::F32))
}
fn relay_backward_meta(q: &TensorSpec, k: &TensorSpec, v: &TensorSpec, rk: &TensorSpec, rv: &TensorSpec, _ids: &TensorSpec, _start: &TensorSpec, _end: &TensorSpec, _rows: &TensorSpec, _starts: &TensorSpec, _ends: &TensorSpec, _o: &TensorSpec, _normalization: &TensorSpec, _grad: &TensorSpec) -> (TensorSpec, TensorSpec, TensorSpec, TensorSpec, TensorSpec) {
    (q.clone(), k.clone(), v.clone(), rk.clone(), rv.clone())
}

#[backend_extension(Cube, Autodiff, Fusion)]
pub(crate) trait RelayAttentionKernelOps: Backend {
    #[fusion(meta = relay_forward_meta)]
    fn relay_attention_forward(q: FloatTensor<Self>, k: FloatTensor<Self>, v: FloatTensor<Self>, relay_k: FloatTensor<Self>, relay_v: FloatTensor<Self>, doc_id: IntTensor<Self>, doc_start: IntTensor<Self>, doc_end: IntTensor<Self>, doc_row: IntTensor<Self>, relay_start: IntTensor<Self>, relay_end: IntTensor<Self>) -> (FloatTensor<Self>, FloatTensor<Self>);
    #[fusion(meta = relay_backward_meta)]
    fn relay_attention_backward(q: FloatTensor<Self>, k: FloatTensor<Self>, v: FloatTensor<Self>, relay_k: FloatTensor<Self>, relay_v: FloatTensor<Self>, doc_id: IntTensor<Self>, doc_start: IntTensor<Self>, doc_end: IntTensor<Self>, doc_row: IntTensor<Self>, relay_start: IntTensor<Self>, relay_end: IntTensor<Self>, o: FloatTensor<Self>, normalization: FloatTensor<Self>, grad: FloatTensor<Self>) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);
}
#[derive(Clone, Debug)]
struct RelayBackwardState<B: Backend> {
    q: B::FloatTensorPrimitive,
    k: B::FloatTensorPrimitive,
    v: B::FloatTensorPrimitive,
    relay_k: B::FloatTensorPrimitive,
    relay_v: B::FloatTensorPrimitive,
    doc_id: B::IntTensorPrimitive,
    doc_start: B::IntTensorPrimitive,
    doc_end: B::IntTensorPrimitive,
    doc_row: B::IntTensorPrimitive,
    relay_start: B::IntTensorPrimitive,
    relay_end: B::IntTensorPrimitive,
    output: B::FloatTensorPrimitive,
    normalization: B::FloatTensorPrimitive,
}

#[cube(launch)]
fn relay_forward_kernel<F: Float, I: Int>(
    q: &CubeTensorView<F>, k: &CubeTensorView<F>, v: &CubeTensorView<F>,
    rk: &CubeTensorView<F>, rv: &CubeTensorView<F>,
    doc_id: &CubeTensorView<I>, doc_start: &CubeTensorView<I>, doc_end: &CubeTensorView<I>,
    o: &mut CubeTensorView<F>, normalization: &mut CubeTensorView<f32>, #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let heads = q.shape(1);
    let tokens = q.shape(2);
    let tiles = (tokens + 31usize) / 32usize;
    let cube = CUBE_POS as usize;
    let groups = q.shape(0) * heads * tiles;
    if cube >= groups { terminate!(); }
    let b = cube / (heads * tiles);
    let head = (cube / tiles) % heads;
    let query = (cube % tiles) * 32usize + UNIT_POS_X as usize;
    let valid = query < tokens;
    let mut doc = 0usize;
    let mut start = I::cast_from(0u32);
    let mut end = I::cast_from(0u32);
    if valid {
        doc = u32::cast_from(doc_id[b * doc_id.stride(0) + query * doc_id.stride(1)]) as usize;
        start = I::cast_from(doc_start[b * doc_start.stride(0) + query * doc_start.stride(1)]);
        end = I::cast_from(doc_end[b * doc_end.stride(0) + query * doc_end.stride(1)]);
    }
    let qbase = b * q.stride(0) + head * q.stride(1) + query * q.stride(2);
    let mut accum = Array::<f32>::new(64usize);
    let mut d = 0usize;
    while d < 64usize { accum[d] = f32::cast_from(0.0_f32); d += 1usize; }
    // The first eligible score initializes the maximum, even below any finite sentinel.
    let mut maximum = f32::cast_from(0.0_f32);
    let mut total = f32::cast_from(0.0_f32);
    if valid && end >= start {
        let lo = if query > 128usize { query - 128usize } else { 0usize };
        let hi = if query + 128usize < tokens { query + 128usize } else { tokens - 1usize };
        let mut key = lo;
        while key <= hi {
            if I::cast_from(key as u32) >= start && I::cast_from(key as u32) <= end {
                let kb = b * k.stride(0) + head * k.stride(1) + key * k.stride(2);
                let mut dot = f32::cast_from(0.0_f32);
                d = 0usize;
                while d < 64usize { dot += f32::cast_from(q[qbase + d * q.stride(3)]) * f32::cast_from(k[kb + d * k.stride(3)]); d += 1usize; }
                let score = dot * f32::cast_from(0.125_f32);
                let next = if total == f32::cast_from(0.0_f32) || score > maximum { score } else { maximum };
                let alpha = if total == f32::cast_from(0.0_f32) { f32::cast_from(0.0_f32) } else { (maximum - next).exp() };
                let p = (score - next).exp();
                total = total * alpha + p;
                d = 0usize;
                while d < 64usize {
                    let vb = b * v.stride(0) + head * v.stride(1) + key * v.stride(2) + d * v.stride(3);
                    accum[d] = accum[d] * alpha + p * f32::cast_from(v[vb]);
                    d += 1usize;
                }
                maximum = next;
            }
            key += 1usize;
        }
        let mut slot = 0usize;
        while slot < 16usize {
            let kb = doc * rk.stride(0) + head * rk.stride(1) + slot * rk.stride(2);
            let vb = doc * rv.stride(0) + head * rv.stride(1) + slot * rv.stride(2);
            let mut dot = f32::cast_from(0.0_f32);
            d = 0usize;
            while d < 64usize { dot += f32::cast_from(q[qbase + d * q.stride(3)]) * f32::cast_from(rk[kb + d * rk.stride(3)]); d += 1usize; }
            let score = dot * f32::cast_from(0.125_f32);
            let next = if total == f32::cast_from(0.0_f32) || score > maximum { score } else { maximum };
            let alpha = if total == f32::cast_from(0.0_f32) { f32::cast_from(0.0_f32) } else { (maximum - next).exp() };
            let p = (score - next).exp();
            total = total * alpha + p;
            d = 0usize;
            while d < 64usize { accum[d] = accum[d] * alpha + p * f32::cast_from(rv[vb + d * rv.stride(3)]); d += 1usize; }
            maximum = next;
            slot += 1usize;
        }
    }
    if valid {
        let row = (b * heads + head) * tokens + query;
        let normalization_offset = b * normalization.stride(0) + head * normalization.stride(1) + query * normalization.stride(2);
        let total_offset = normalization_offset + normalization.stride(3);
        let out = row * 64usize;
        if end < start {
            d = 0usize;
            while d < 64usize { o[out + d] = F::cast_from(f32::cast_from(0.0_f32)); d += 1usize; }
            normalization[normalization_offset] = f32::cast_from(0.0_f32);
            normalization[total_offset] = f32::cast_from(1.0_f32);
        } else {
            d = 0usize;
            while d < 64usize { o[out + d] = F::cast_from(accum[d] / total); d += 1usize; }
            normalization[normalization_offset] = maximum;
            normalization[total_offset] = total;
        }
    }
}

#[cube(launch)]
fn relay_query_backward_kernel<F: Float, I: Int>(q: &CubeTensorView<F>, k: &CubeTensorView<F>, v: &CubeTensorView<F>, rk: &CubeTensorView<F>, rv: &CubeTensorView<F>, doc_id: &CubeTensorView<I>, doc_start: &CubeTensorView<I>, doc_end: &CubeTensorView<I>, o: &CubeTensorView<F>, normalization: &CubeTensorView<f32>, grad: &CubeTensorView<F>, dq: &mut CubeTensorView<F>, #[define(F, I)] _types: [ElemType; 2]) {
    let heads = q.shape(1); let tokens = q.shape(2); let tiles = (tokens + 31usize) / 32usize; let cube = CUBE_POS as usize;
    let groups = q.shape(0) * heads * tiles;
    if cube >= groups { terminate!(); }
    let b = cube / (heads * tiles); let head = (cube / tiles) % heads; let query = (cube % tiles) * 32usize + UNIT_POS_X as usize;
    if query < tokens {
        let doc = u32::cast_from(doc_id[b * doc_id.stride(0) + query * doc_id.stride(1)]) as usize; let start = I::cast_from(doc_start[b * doc_start.stride(0) + query * doc_start.stride(1)]); let end = I::cast_from(doc_end[b * doc_end.stride(0) + query * doc_end.stride(1)]);
        let row = (b * heads + head) * tokens + query; let go = b * grad.stride(0) + head * grad.stride(1) + query * grad.stride(2); let qo = b * q.stride(0) + head * q.stride(1) + query * q.stride(2);
        let normalization_offset = b * normalization.stride(0) + head * normalization.stride(1) + query * normalization.stride(2);
        let mut correction = f32::cast_from(0.0_f32); let mut d = 0usize; while d < 64usize { correction += f32::cast_from(grad[go + d * grad.stride(3)]) * f32::cast_from(o[row * 64usize + d]); d += 1usize; }
        let mut accum = Array::<f32>::new(64usize); d = 0usize; while d < 64usize { accum[d] = f32::cast_from(0.0_f32); d += 1usize; }
        if end >= start {
            let lo = if query > 128usize { query - 128usize } else { 0usize }; let hi = if query + 128usize < tokens { query + 128usize } else { tokens - 1usize }; let mut key = lo;
            while key <= hi {
                if I::cast_from(key as u32) >= start && I::cast_from(key as u32) <= end {
                    let kb = b * k.stride(0) + head * k.stride(1) + key * k.stride(2); let vb = b * v.stride(0) + head * v.stride(1) + key * v.stride(2); let mut dot = f32::cast_from(0.0_f32); let mut gv = f32::cast_from(0.0_f32); d = 0usize;
                    while d < 64usize { dot += f32::cast_from(q[qo + d * q.stride(3)]) * f32::cast_from(k[kb + d * k.stride(3)]); gv += f32::cast_from(grad[go + d * grad.stride(3)]) * f32::cast_from(v[vb + d * v.stride(3)]); d += 1usize; }
                    let p = (dot * f32::cast_from(0.125_f32) - normalization[normalization_offset]).exp() / normalization[normalization_offset + normalization.stride(3)]; let ds = p * (gv - correction) * f32::cast_from(0.125_f32); d = 0usize;
                    while d < 64usize { accum[d] += ds * f32::cast_from(k[kb + d * k.stride(3)]); d += 1usize; }
                }
                key += 1usize;
            }
            let mut slot = 0usize;
            while slot < 16usize {
                let kb = doc * rk.stride(0) + head * rk.stride(1) + slot * rk.stride(2); let vb = doc * rv.stride(0) + head * rv.stride(1) + slot * rv.stride(2); let mut dot = f32::cast_from(0.0_f32); let mut gv = f32::cast_from(0.0_f32); d = 0usize;
                while d < 64usize { dot += f32::cast_from(q[qo + d * q.stride(3)]) * f32::cast_from(rk[kb + d * rk.stride(3)]); gv += f32::cast_from(grad[go + d * grad.stride(3)]) * f32::cast_from(rv[vb + d * rv.stride(3)]); d += 1usize; }
                let p = (dot * f32::cast_from(0.125_f32) - normalization[normalization_offset]).exp() / normalization[normalization_offset + normalization.stride(3)]; let ds = p * (gv - correction) * f32::cast_from(0.125_f32); d = 0usize;
                while d < 64usize { accum[d] += ds * f32::cast_from(rk[kb + d * rk.stride(3)]); d += 1usize; }
                slot += 1usize;
            }
        }
        d = 0usize; while d < 64usize { dq[row * 64usize + d] = F::cast_from(accum[d]); d += 1usize; }
    }
}

#[cube(launch)]
fn relay_key_backward_kernel<F: Float, I: Int>(q: &CubeTensorView<F>, o: &CubeTensorView<F>, grad: &CubeTensorView<F>, rk: &CubeTensorView<F>, rv: &CubeTensorView<F>, doc_row: &CubeTensorView<I>, relay_start: &CubeTensorView<I>, relay_end: &CubeTensorView<I>, normalization: &CubeTensorView<f32>, drk: &mut CubeTensorView<F>, drv: &mut CubeTensorView<F>, #[define(F, I)] _types: [ElemType; 2]) {
    let heads = q.shape(1); let tokens = q.shape(2); let group = CUBE_POS as usize;
    let groups = rk.shape(0) * heads * 16usize;
    if group >= groups { terminate!(); }
    let slot = group % 16usize; let head = (group / 16usize) % heads; let document = group / (16usize * heads);
    let row = u32::cast_from(doc_row[document * doc_row.stride(0)]) as usize;
    let start = u32::cast_from(relay_start[document * relay_start.stride(0)]) as usize;
    let end = u32::cast_from(relay_end[document * relay_end.stride(0)]) as usize;
    let lane = UNIT_POS_X as usize;
    let mut dk0 = f32::cast_from(0.0_f32); let mut dk1 = f32::cast_from(0.0_f32);
    let mut dv0 = f32::cast_from(0.0_f32); let mut dv1 = f32::cast_from(0.0_f32);
    let kb = document * rk.stride(0) + head * rk.stride(1) + slot * rk.stride(2);
    let vb = document * rv.stride(0) + head * rv.stride(1) + slot * rv.stride(2);
    let mut query = start;
    while query <= end {
        if query < tokens {
            let qbase = row * q.stride(0) + head * q.stride(1) + query * q.stride(2);
            let outrow = (row * heads + head) * tokens + query;
            let normalization_offset = row * normalization.stride(0) + head * normalization.stride(1) + query * normalization.stride(2);
            let go = outrow * 64usize;
            let grad_base = row * grad.stride(0) + head * grad.stride(1) + query * grad.stride(2);
            let mut dot = f32::cast_from(0.0_f32); let mut correction = f32::cast_from(0.0_f32); let mut gv = f32::cast_from(0.0_f32);
            let mut d = 0usize;
            while d < 64usize {
                dot += f32::cast_from(q[qbase + d * q.stride(3)]) * f32::cast_from(rk[kb + d * rk.stride(3)]);
                correction += f32::cast_from(grad[grad_base + d * grad.stride(3)]) * f32::cast_from(o[go + d]);
                gv += f32::cast_from(grad[grad_base + d * grad.stride(3)]) * f32::cast_from(rv[vb + d * rv.stride(3)]);
                d += 1usize;
            }
            let p = (dot * f32::cast_from(0.125_f32) - normalization[normalization_offset]).exp() / normalization[normalization_offset + normalization.stride(3)];
            let ds = p * (gv - correction) * f32::cast_from(0.125_f32);
            dk0 += ds * f32::cast_from(q[qbase + lane * q.stride(3)]);
            dk1 += ds * f32::cast_from(q[qbase + (lane + 32usize) * q.stride(3)]);
            dv0 += p * f32::cast_from(grad[grad_base + lane * grad.stride(3)]);
            dv1 += p * f32::cast_from(grad[grad_base + (lane + 32usize) * grad.stride(3)]);
        }
        query += 1usize;
    }
    let drk_stride = drk.stride(3);
    let drv_stride = drv.stride(3);
    let drk_base = document * drk.stride(0) + head * drk.stride(1) + slot * drk.stride(2);
    let drv_base = document * drv.stride(0) + head * drv.stride(1) + slot * drv.stride(2);
    drk[drk_base + lane * drk_stride] = F::cast_from(dk0); drk[drk_base + (lane + 32usize) * drk_stride] = F::cast_from(dk1);
    drv[drv_base + lane * drv_stride] = F::cast_from(dv0); drv[drv_base + (lane + 32usize) * drv_stride] = F::cast_from(dv1);
}
fn relay_forward_cube(q: CubeTensor, k: CubeTensor, v: CubeTensor, rk: CubeTensor, rv: CubeTensor, ids: CubeTensor, starts: CubeTensor, ends: CubeTensor, _rows: CubeTensor, _relay_starts: CubeTensor, _relay_ends: CubeTensor) -> (CubeTensor, CubeTensor) {
    let [batch, heads, tokens, width] = q.shape().dims::<4>();
    assert_eq!(width, 64);
    let groups = batch * heads * tokens.div_ceil(32);
    let client = q.client.clone(); let device = q.device.clone(); let dtype = q.dtype; let int_dtype = ids.dtype;
    let output = CubeBackend::float_zeros(q.shape(), &device, dtype.into());
    let normalization = CubeBackend::float_zeros(Shape::new([batch, heads, tokens, 2]), &device, burn::tensor::DType::F32.into());
    relay_forward_kernel::launch(
        &client, super::attention_kernel::attention_cube_count(&client, groups), CubeDim::new_1d(32),
        q.into_tensor_arg(), k.into_tensor_arg(), v.into_tensor_arg(), rk.into_tensor_arg(), rv.into_tensor_arg(),
        ids.into_tensor_arg(), starts.into_tensor_arg(), ends.into_tensor_arg(),
        output.clone().into_tensor_arg(), normalization.clone().into_tensor_arg(),
        [dtype_to_storage_type(dtype), dtype_to_storage_type(int_dtype)],
    );
    (output, normalization)
}

fn relay_backward_cube(
    q: CubeTensor, k: CubeTensor, v: CubeTensor, rk: CubeTensor, rv: CubeTensor,
    ids: CubeTensor, starts: CubeTensor, ends: CubeTensor, rows: CubeTensor, relay_starts: CubeTensor, relay_ends: CubeTensor,
    o: CubeTensor, normalization: CubeTensor, grad: CubeTensor,
) -> (CubeTensor, CubeTensor, CubeTensor, CubeTensor, CubeTensor) {
    let [batch, heads, tokens, width] = q.shape().dims::<4>();
    let [documents, _, _, _] = rk.shape().dims::<4>();
    assert_eq!(width, 64);
    let client = q.client.clone(); let device = q.device.clone(); let dtype = q.dtype; let int_dtype = ids.dtype;
    let dq = CubeBackend::float_zeros(q.shape(), &device, dtype.into());
    let dk = CubeBackend::float_zeros(k.shape(), &device, dtype.into());
    let dv = CubeBackend::float_zeros(v.shape(), &device, dtype.into());
    let drk = CubeBackend::float_zeros(rk.shape(), &device, dtype.into());
    let drv = CubeBackend::float_zeros(rv.shape(), &device, dtype.into());
    let groups = batch * heads * tokens.div_ceil(32);
    let count = super::attention_kernel::attention_cube_count(&client, groups); let dim = CubeDim::new_1d(32);
    let types = [dtype_to_storage_type(dtype), dtype_to_storage_type(int_dtype)];
    relay_query_backward_kernel::launch(
        &client, count.clone(), dim,
        q.clone().into_tensor_arg(), k.clone().into_tensor_arg(), v.clone().into_tensor_arg(), rk.clone().into_tensor_arg(), rv.clone().into_tensor_arg(),
        ids.clone().into_tensor_arg(), starts.clone().into_tensor_arg(), ends.clone().into_tensor_arg(),
        o.clone().into_tensor_arg(), normalization.clone().into_tensor_arg(), grad.clone().into_tensor_arg(), dq.clone().into_tensor_arg(), types,
    );
    super::attention_kernel::key_backward_kernel::launch(
        &client, count, dim,
        q.clone().into_tensor_arg(), k.into_tensor_arg(), v.into_tensor_arg(), starts.into_tensor_arg(), ends.into_tensor_arg(),
        o.clone().into_tensor_arg(), normalization.clone().into_tensor_arg(), grad.clone().into_tensor_arg(), dk.clone().into_tensor_arg(), dv.clone().into_tensor_arg(), types,
    );
    relay_key_backward_kernel::launch(
        &client, super::attention_kernel::attention_cube_count(&client, documents * heads * 16), dim,
        q.into_tensor_arg(), o.into_tensor_arg(), grad.into_tensor_arg(), rk.into_tensor_arg(), rv.into_tensor_arg(),
        rows.into_tensor_arg(), relay_starts.into_tensor_arg(), relay_ends.into_tensor_arg(), normalization.into_tensor_arg(),
        drk.clone().into_tensor_arg(), drv.clone().into_tensor_arg(), types,
    );
    (dq, dk, dv, drk, drv)
}

impl RelayAttentionKernelOps for CubeBackend {
    fn relay_attention_forward(q: CubeTensor, k: CubeTensor, v: CubeTensor, relay_k: CubeTensor, relay_v: CubeTensor, doc_id: CubeTensor, doc_start: CubeTensor, doc_end: CubeTensor, doc_row: CubeTensor, relay_start: CubeTensor, relay_end: CubeTensor) -> (CubeTensor, CubeTensor) {
        relay_forward_cube(q, k, v, relay_k, relay_v, doc_id, doc_start, doc_end, doc_row, relay_start, relay_end)
    }
    fn relay_attention_backward(q: CubeTensor, k: CubeTensor, v: CubeTensor, relay_k: CubeTensor, relay_v: CubeTensor, doc_id: CubeTensor, doc_start: CubeTensor, doc_end: CubeTensor, doc_row: CubeTensor, relay_start: CubeTensor, relay_end: CubeTensor, o: CubeTensor, normalization: CubeTensor, grad: CubeTensor) -> (CubeTensor, CubeTensor, CubeTensor, CubeTensor, CubeTensor) {
        relay_backward_cube(q, k, v, relay_k, relay_v, doc_id, doc_start, doc_end, doc_row, relay_start, relay_end, o, normalization, grad)
    }
}

impl<B: Backend + RelayAttentionKernelOps, C: CheckpointStrategy> RelayAttentionKernelOps for Autodiff<B, C> {
    fn relay_attention_forward(q: FloatTensor<Self>, k: FloatTensor<Self>, v: FloatTensor<Self>, rk: FloatTensor<Self>, rv: FloatTensor<Self>, ids: IntTensor<Self>, starts: IntTensor<Self>, ends: IntTensor<Self>, rows: IntTensor<Self>, relay_starts: IntTensor<Self>, relay_ends: IntTensor<Self>) -> (FloatTensor<Self>, FloatTensor<Self>) {
        #[derive(Debug)] struct RelayBackward;
        impl<B: Backend + RelayAttentionKernelOps> Backward<B, 5> for RelayBackward {
            type State = RelayBackwardState<B>;
            fn backward(self, ops: Ops<Self::State, 5>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
                let [nq, nk, nv, nrk, nrv] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);
                let RelayBackwardState { q, k, v, relay_k, relay_v, doc_id, doc_start, doc_end, doc_row, relay_start, relay_end, output, normalization } = ops.state;
                let (dq, dk, dv, drk, drv) = B::relay_attention_backward(q, k, v, relay_k, relay_v, doc_id, doc_start, doc_end, doc_row, relay_start, relay_end, output, normalization, grad);
                if let Some(n) = nq { grads.register::<B>(n.id, dq); }
                if let Some(n) = nk { grads.register::<B>(n.id, dk); }
                if let Some(n) = nv { grads.register::<B>(n.id, dv); }
                if let Some(n) = nrk { grads.register::<B>(n.id, drk); }
                if let Some(n) = nrv { grads.register::<B>(n.id, drv); }
            }
        }
        let nodes = [q.node(), k.node(), v.node(), rk.node(), rv.node()];
        let (q, k, v, rk, rv) = (q.into_primitive(), k.into_primitive(), v.into_primitive(), rk.into_primitive(), rv.into_primitive());
        let (o, normalization) = B::relay_attention_forward(q.clone(), k.clone(), v.clone(), rk.clone(), rv.clone(), ids.clone(), starts.clone(), ends.clone(), rows.clone(), relay_starts.clone(), relay_ends.clone());
        let state = RelayBackwardState { q, k, v, relay_k: rk, relay_v: rv, doc_id: ids, doc_start: starts, doc_end: ends, doc_row: rows, relay_start: relay_starts, relay_end: relay_ends, output: o.clone(), normalization: normalization.clone() };
        match RelayBackward.prepare::<C>(nodes).compute_bound().stateful() {
            OpsKind::Tracked(prep) => (prep.finish(state, o), <Autodiff<B, C> as AutodiffBackend>::from_inner(normalization)),
            OpsKind::UnTracked(prep) => (prep.finish(o), <Autodiff<B, C> as AutodiffBackend>::from_inner(normalization)),
        }
    }
    fn relay_attention_backward(q: FloatTensor<Self>, k: FloatTensor<Self>, v: FloatTensor<Self>, rk: FloatTensor<Self>, rv: FloatTensor<Self>, ids: IntTensor<Self>, starts: IntTensor<Self>, ends: IntTensor<Self>, rows: IntTensor<Self>, relay_starts: IntTensor<Self>, relay_ends: IntTensor<Self>, o: FloatTensor<Self>, normalization: FloatTensor<Self>, grad: FloatTensor<Self>) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let q = <Self as AutodiffBackend>::inner(q); let k = <Self as AutodiffBackend>::inner(k); let v = <Self as AutodiffBackend>::inner(v);
        let rk = <Self as AutodiffBackend>::inner(rk); let rv = <Self as AutodiffBackend>::inner(rv);
        let o = <Self as AutodiffBackend>::inner(o); let normalization = <Self as AutodiffBackend>::inner(normalization); let grad = <Self as AutodiffBackend>::inner(grad);
        let (dq, dk, dv, drk, drv) = B::relay_attention_backward(q, k, v, rk, rv, ids, starts, ends, rows, relay_starts, relay_ends, o, normalization, grad);
        (<Self as AutodiffBackend>::from_inner(dq), <Self as AutodiffBackend>::from_inner(dk), <Self as AutodiffBackend>::from_inner(dv), <Self as AutodiffBackend>::from_inner(drk), <Self as AutodiffBackend>::from_inner(drv))
    }
}
/// Returns output and FP32 `[row maximum, sum(exp(score - maximum))]` per query.
pub(crate) fn run_relay_windowed_attention(
    q: burn::Tensor<4>,
    k: burn::Tensor<4>,
    v: burn::Tensor<4>,
    relay_k: burn::Tensor<4>,
    relay_v: burn::Tensor<4>,
    layout: &dre_core::BatchLayout,
) -> (burn::Tensor<4>, burn::Tensor<4>) {
    let [batch, heads, tokens, width] = q.dims();
    let documents = layout.host.document_capacity();
    assert_eq!([batch,tokens],[layout.host.batch(),layout.host.tokens()]);
    assert_eq!(width, 64);
    assert_eq!(relay_k.dims(), [documents, heads, 16, 64]);
    assert_eq!(relay_v.dims(), [documents, heads, 16, 64]);
    let (output, normalization) = <burn::backend::Dispatch as RelayAttentionKernelOps>::relay_attention_forward(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        relay_k.into_dispatch(),
        relay_v.into_dispatch(),
        layout.document_id.clone().into_dispatch(),
        layout.doc_start.clone().into_dispatch(),
        layout.doc_end.clone().into_dispatch(),
        layout.document_rows.clone().into_dispatch(),
        layout.document_start.clone().into_dispatch(),
        layout.document_end.clone().into_dispatch(),
    );
    (burn::Tensor::from_dispatch(output), burn::Tensor::from_dispatch(normalization))
}
