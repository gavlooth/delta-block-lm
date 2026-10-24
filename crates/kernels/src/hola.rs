//! Position-only exact-memory schedule and direct-original-KV attention.
//! Selection is O(chunks*(capacity*chunk + chunk^2)), with no rank matrix,
//! sorting operator, KV cache copies, host transfers, or floating-point atomics.
//! A selected key has one uninterrupted cache lifetime. The inverse VJP walks
//! that lifetime, so total key/query visits are O(time*(capacity+chunk)), not T².
use burn::backend::{TensorMetadata, ops::FloatTensorOps};
use burn::{prelude::Shape, tensor::DType};
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use cubecl::frontend::Tensor as View;
use cubecl::prelude::*;

const K: usize = 32;
const V: usize = 64;
const SCALE: f32 = 0.1767766952966369;

#[cube]
fn ix(x: &View<f32>, b: usize, t: usize, h: usize, c: usize) -> usize {
    b * x.stride(0) + t * x.stride(1) + h * x.stride(2) + c * x.stride(3)
}

#[cube]
fn metadata(x: &View<f32>, b: usize, t: usize) -> f32 {
    x[b * x.stride(0) + t * x.stride(1)]
}

#[cube]
fn earlier(score: f32, position: f32, other_score: f32, other_position: f32) -> bool {
    score > other_score || (score == other_score && position < other_position)
}

#[cube(launch)]
fn selection_kernel(
    scores: &View<f32>,
    valid: &View<f32>,
    documents: &View<f32>,
    schedule: &mut View<f32>,
    expiry: &mut View<f32>,
    #[comptime] capacity: usize,
    #[comptime] chunk: usize,
    #[comptime] recency: bool,
) {
    let lane = UNIT_POS as usize;
    let head = CUBE_POS as usize;
    let heads = scores.shape(2);
    let h = head % heads;
    let b = head / heads;
    if b >= scores.shape(0) {
        terminate!();
    }
    let time = scores.shape(1);
    let chunks = (time + chunk - 1) / chunk;
    let mut cache = Shared::<[f32]>::new_slice(capacity);
    let mut next = Shared::<[f32]>::new_slice(capacity);
    let mut positions = Shared::<[f32]>::new_slice(capacity + chunk);
    let mut priorities = Shared::<[f32]>::new_slice(capacity + chunk);
    if lane < capacity {
        cache[lane] = -1.0f32;
        schedule[head * (chunks + 1) * capacity + lane] = -1.0f32;
    }
    sync_cube();
    let mut c = 0usize;
    while c < chunks {
        let first = c * chunk;
        let end = usize::min(first + chunk, time);
        // Deliberately use the PHYSICAL last slot, including padding. This is
        // the reference's cache reset rule, not the last valid token's document.
        let last_doc = metadata(documents, b, end - 1);
        if lane < capacity {
            next[lane] = -1.0f32;
        }
        if lane < capacity + chunk {
            let mut p = 0.0f32;
            p -= 1.0f32;
            if lane < capacity {
                p = cache[lane];
            } else {
                let t = first + lane - capacity;
                if t < end {
                    expiry[(b * time + t) * heads + h] = end as f32;
                    if metadata(valid, b, t) >= 0.5f32 {
                        p = t as f32;
                    }
                }
            }
            let mut score = 0.0f32;
            if p >= 0.0f32 {
                let t = p as usize;
                if metadata(documents, b, t) == last_doc {
                    score = if recency {
                        p
                    } else {
                        scores[ix(scores, b, t, h, 0)]
                    };
                } else {
                    p = -1.0f32;
                }
            }
            positions[lane] = p;
            priorities[lane] = score;
        }
        sync_cube();
        if lane < capacity + chunk {
            let p = positions[lane];
            if p >= 0.0f32 {
                // Previous cache entries are already sorted and either ALL
                // match last_doc or NONE do. No capacity² reranking is needed.
                let mut rank = 0usize;
                if lane < capacity {
                    rank = lane;
                } else {
                    let mut s = 0usize;
                    while s < capacity {
                        if positions[s] >= 0.0f32
                            && earlier(priorities[s], positions[s], priorities[lane], p)
                        {
                            rank += 1;
                        }
                        s += 1;
                    }
                }
                let mut s = 0usize;
                s += capacity;
                while s < capacity + chunk {
                    if positions[s] >= 0.0f32
                        && earlier(priorities[s], positions[s], priorities[lane], p)
                    {
                        rank += 1;
                    }
                    s += 1;
                }
                if rank < capacity {
                    next[rank] = p;
                }
            }
        }
        sync_cube();
        if lane < capacity {
            let p = next[lane];
            cache[lane] = p;
            schedule[(head * (chunks + 1) + c + 1) * capacity + lane] = p;
            if p >= 0.0f32 {
                expiry[(b * time + p as usize) * heads + h] = usize::min(end + chunk, time) as f32;
            }
        }
        sync_cube();
        c += 1;
    }
}

#[cube]
fn source(
    schedule: &View<f32>,
    valid: &View<f32>,
    documents: &View<f32>,
    b: usize,
    t: usize,
    h: usize,
    heads: usize,
    time: usize,
    slot: usize,
    #[comptime] capacity: usize,
    #[comptime] chunk: usize,
) -> f32 {
    let chunks = (time + chunk - 1) / chunk;
    let mut p = if slot < capacity {
        schedule[(((b * heads + h) * (chunks + 1) + t / chunk) * capacity) + slot]
    } else {
        (t / chunk * chunk + slot - capacity) as f32
    };
    if p >= 0.0f32 {
        let s = p as usize;
        if s >= time || s > t {
            p = -1.0f32;
        } else if metadata(valid, b, s) < 0.5f32
            || metadata(documents, b, s) != metadata(documents, b, t)
        {
            p = -1.0f32;
        }
    }
    p
}

#[cube]
fn logit(q: &View<f32>, k: &View<f32>, b: usize, t: usize, s: usize, h: usize) -> f32 {
    let mut result = 0.0f32;
    let mut i = 0usize;
    while i < K {
        result += q[ix(q, b, t, h, i)] * k[ix(k, b, s, h, i)];
        i += 1;
    }
    result * SCALE
}

#[cube(launch)]
fn forward_kernel(
    q: &View<f32>,
    k: &View<f32>,
    v: &View<f32>,
    valid: &View<f32>,
    documents: &View<f32>,
    sink: &View<f32>,
    schedule: &View<f32>,
    output: &mut View<f32>,
    stats: &mut View<f32>,
    #[comptime] capacity: usize,
    #[comptime] chunk: usize,
    #[comptime] lanes: usize,
) {
    let lane = UNIT_POS as usize;
    let token = CUBE_POS as usize;
    let heads = q.shape(2);
    let time = q.shape(1);
    let h = token % heads;
    let t = token / heads % time;
    let b = token / (heads * time);
    if b >= q.shape(0) {
        terminate!();
    }
    let active = metadata(valid, b, t) >= 0.5f32;
    let mut probabilities = Shared::<[f32]>::new_slice(capacity + chunk);
    let mut positions = Shared::<[f32]>::new_slice(capacity + chunk);
    let mut reduction = Shared::<[f32]>::new_slice(lanes);
    let sink_logit = sink[h * sink.stride(0)];
    let mut value = sink_logit;
    if lane < capacity + chunk {
        let p = source(
            schedule, valid, documents, b, t, h, heads, time, lane, capacity, chunk,
        );
        positions[lane] = p;
        let mut score = 0.0f32;
        score -= 3.402823466e38f32;
        if active && p >= 0.0f32 {
            score = logit(q, k, b, t, p as usize, h);
        }
        probabilities[lane] = score;
        value = f32::max(value, score);
    }
    reduction[lane] = value;
    sync_cube();
    let mut stride = 32usize;
    if lanes == 128 {
        stride = 64usize;
    }
    while stride > 0 {
        if lane < stride {
            reduction[lane] = f32::max(reduction[lane], reduction[lane + stride]);
        }
        sync_cube();
        stride /= 2;
    }
    let maximum = reduction[0];
    // All lanes must consume maximum before reduction is overwritten.
    sync_cube();
    let mut weight = 0.0f32;
    if lane < capacity + chunk {
        if active && positions[lane] >= 0.0f32 {
            weight = f32::exp(probabilities[lane] - maximum);
        }
        probabilities[lane] = weight;
    }
    reduction[lane] = weight;
    if lane == 0 {
        reduction[0] += f32::exp(sink_logit - maximum);
    }
    sync_cube();
    stride = 32usize;
    if lanes == 128 {
        stride = 64usize;
    }
    while stride > 0 {
        if lane < stride {
            reduction[lane] += reduction[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    let inverse = 1.0f32 / reduction[0];
    // Normalize before the value product, like softmax(...).matmul(V).
    // Accumulating unnormalized weights can overflow for finite original V.
    if lane < capacity + chunk {
        probabilities[lane] *= inverse;
    }
    sync_cube();
    if lane == 0 {
        stats[token * 2] = maximum;
        stats[token * 2 + 1] = inverse;
    }
    if lane < V {
        let mut result = 0.0f32;
        if active {
            let mut s = 0usize;
            while s < capacity + chunk {
                if positions[s] >= 0.0f32 {
                    result += probabilities[s] * v[ix(v, b, positions[s] as usize, h, lane)];
                }
                s += 1;
            }
        }
        output[token * V + lane] = result;
    }
}

#[cube(launch)]
fn query_backward_kernel(
    q: &View<f32>,
    k: &View<f32>,
    v: &View<f32>,
    valid: &View<f32>,
    documents: &View<f32>,
    sink: &View<f32>,
    schedule: &View<f32>,
    output: &View<f32>,
    stats: &View<f32>,
    upstream: &View<f32>,
    dq: &mut View<f32>,
    auxiliary: &mut View<f32>,
    #[comptime] capacity: usize,
    #[comptime] chunk: usize,
    #[comptime] lanes: usize,
) {
    let lane = UNIT_POS as usize;
    let token = CUBE_POS as usize;
    let heads = q.shape(2);
    let time = q.shape(1);
    let h = token % heads;
    let t = token / heads % time;
    let b = token / (heads * time);
    if b >= q.shape(0) {
        terminate!();
    }
    let active = metadata(valid, b, t) >= 0.5f32;
    let mut reduction = Shared::<[f32]>::new_slice(lanes);
    let mut derivatives = Shared::<[f32]>::new_slice(capacity + chunk);
    let mut positions = Shared::<[f32]>::new_slice(capacity + chunk);
    let mut delta = 0.0f32;
    if lane < V && active {
        delta = upstream[ix(upstream, b, t, h, lane)] * output[token * V + lane];
    }
    reduction[lane] = delta;
    sync_cube();
    let mut stride = 32usize;
    if lanes == 128 {
        stride = 64usize;
    }
    while stride > 0 {
        if lane < stride {
            reduction[lane] += reduction[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    delta = reduction[0];
    let maximum = stats[token * 2];
    let inverse = stats[token * 2 + 1];
    if lane == 0 {
        auxiliary[token * 2] = delta;
        auxiliary[token * 2 + 1] = if active {
            -delta * f32::exp(sink[h * sink.stride(0)] - maximum) * inverse
        } else {
            0.0f32
        };
    }
    if lane < capacity + chunk {
        let p = source(
            schedule, valid, documents, b, t, h, heads, time, lane, capacity, chunk,
        );
        positions[lane] = p;
        let mut derivative = 0.0f32;
        if active && p >= 0.0f32 {
            let s = p as usize;
            let probability = f32::exp(logit(q, k, b, t, s, h) - maximum) * inverse;
            let mut dot = 0.0f32;
            let mut j = 0usize;
            while j < V {
                dot += upstream[ix(upstream, b, t, h, j)] * v[ix(v, b, s, h, j)];
                j += 1;
            }
            derivative = probability * (dot - delta) * SCALE;
        }
        derivatives[lane] = derivative;
    }
    sync_cube();
    if lane < K {
        let mut result = 0.0f32;
        if active {
            let mut s = 0usize;
            while s < capacity + chunk {
                if positions[s] >= 0.0f32 {
                    result += derivatives[s] * k[ix(k, b, positions[s] as usize, h, lane)];
                }
                s += 1;
            }
        }
        dq[token * K + lane] = result;
    }
}

#[cube(launch)]
fn key_value_backward_kernel(
    q: &View<f32>,
    k: &View<f32>,
    v: &View<f32>,
    valid: &View<f32>,
    documents: &View<f32>,
    expiry: &View<f32>,
    stats: &View<f32>,
    auxiliary: &View<f32>,
    upstream: &View<f32>,
    dk: &mut View<f32>,
    dv: &mut View<f32>,
) {
    let lane = UNIT_POS as usize;
    let token = CUBE_POS as usize;
    let heads = q.shape(2);
    let time = q.shape(1);
    let h = token % heads;
    let s = token / heads % time;
    let b = token / (heads * time);
    if b >= q.shape(0) {
        terminate!();
    }
    let mut key_gradient = 0.0f32;
    let mut value_gradient = 0.0f32;
    let mut pair = Shared::<[f32]>::new_slice(2usize);
    if metadata(valid, b, s) >= 0.5f32 {
        let end = expiry[token] as usize;
        let document = metadata(documents, b, s);
        let mut t = s;
        while t < end {
            let active = metadata(valid, b, t) >= 0.5f32 && metadata(documents, b, t) == document;
            if active {
                let query_token = (b * time + t) * heads + h;
                let dy = upstream[ix(upstream, b, t, h, lane)];
                let mut query = 0.0f32;
                if lane < K {
                    query = q[ix(q, b, t, h, lane)];
                }
                if lane == 0 {
                    // Use exactly the forward/query-VJP arithmetic, rather than
                    // a different tree-reduced dot with the saved normalizer.
                    let probability = f32::exp(logit(q, k, b, t, s, h) - stats[query_token * 2])
                        * stats[query_token * 2 + 1];
                    let mut dot = 0.0f32;
                    let mut j = 0usize;
                    while j < V {
                        dot += upstream[ix(upstream, b, t, h, j)] * v[ix(v, b, s, h, j)];
                        j += 1;
                    }
                    pair[0] = probability;
                    pair[1] = probability * (dot - auxiliary[query_token * 2]) * SCALE;
                }
                sync_cube();
                key_gradient += pair[1] * query;
                value_gradient += pair[0] * dy;
                // Every lane consumes the scalars before the next pair reuses them.
                sync_cube();
            }
            t += 1;
        }
    }
    if lane < K {
        dk[token * K + lane] = key_gradient;
    }
    dv[token * V + lane] = value_gradient;
}

#[cube(launch)]
fn sink_backward_kernel(auxiliary: &View<f32>, sink_gradient: &mut View<f32>) {
    let lane = UNIT_POS as usize;
    let h = CUBE_POS as usize;
    let heads = auxiliary.shape(2);
    if h >= heads {
        terminate!();
    }
    let mut result = 0.0f32;
    let mut row = lane;
    let rows = auxiliary.shape(0) * auxiliary.shape(1);
    while row < rows {
        result += auxiliary[(row * heads + h) * 2 + 1];
        row += V;
    }
    let mut reduction = Shared::<[f32]>::new_slice(V);
    reduction[lane] = result;
    sync_cube();
    let mut stride = 32usize;
    while stride > 0 {
        if lane < stride {
            reduction[lane] += reduction[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 {
        sink_gradient[h] = reduction[0];
    }
}

fn checked_elements(dims: &[usize]) -> usize {
    let elements = dims
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
        .expect("HOLA tensor element count overflows usize");
    assert!(
        elements <= u32::MAX as usize,
        "HOLA tensor exceeds 32-bit addressing"
    );
    elements
}

fn checked_span(dims: &[usize], strides: &[usize], backing_elements: u64) {
    checked_elements(dims);
    assert_eq!(dims.len(), strides.len(), "HOLA shape/stride rank mismatch");
    let mut span = 1usize;
    for (&dim, &stride) in dims.iter().zip(strides) {
        assert!(
            stride <= u32::MAX as usize,
            "HOLA stride exceeds 32-bit addressing"
        );
        span = dim
            .saturating_sub(1)
            .checked_mul(stride)
            .and_then(|offset| span.checked_add(offset))
            .expect("HOLA tensor address span overflows usize");
    }
    assert!(
        span <= u32::MAX as usize,
        "HOLA span exceeds 32-bit addressing"
    );
    assert!(
        backing_elements <= u32::MAX as u64,
        "HOLA backing buffer exceeds 32-bit addressing"
    );
}

fn assert_u32_tensor(tensor: &CubeTensor) {
    checked_span(
        tensor.meta.shape().as_slice(),
        tensor.meta.strides(),
        tensor.handle.size() / tensor.elem_size() as u64,
    );
}

fn empty(reference: &CubeTensor, dims: impl Into<Shape>) -> CubeTensor {
    let shape = dims.into();
    checked_elements(shape.as_slice());
    let tensor = CubeBackend::float_empty(shape, &reference.device, DType::F32.into());
    assert_u32_tensor(&tensor);
    tensor
}

fn count(reference: &CubeTensor, groups: usize) -> CubeCount {
    let (max_x, max_y, _) = reference.client.properties().hardware.max_cube_count;
    let x = groups.min(max_x.min(65535) as usize).max(1);
    let y = groups.div_ceil(x);
    assert!(y <= max_y as usize, "HOLA exceeds device workgroup limits");
    CubeCount::Static(x as u32, y as u32, 1)
}

pub(crate) fn run_read(
    q: CubeTensor,
    k: CubeTensor,
    v: CubeTensor,
    scores: CubeTensor,
    valid: CubeTensor,
    documents: CubeTensor,
    sink: CubeTensor,
    capacity: usize,
    chunk: usize,
    recency: bool,
) -> (CubeTensor, CubeTensor, CubeTensor, CubeTensor) {
    let [batch, time, heads, keys] = q.shape().dims::<4>();
    assert!(
        batch > 0 && time > 0 && heads > 0,
        "HOLA requires nonempty axes"
    );
    assert!(
        time <= (1 << 24),
        "HOLA float position schedule requires exactly representable positions"
    );
    assert_eq!(keys, K, "fused HOLA requires key width 32");
    assert_eq!(k.shape().dims::<4>(), [batch, time, heads, K]);
    assert_eq!(v.shape().dims::<4>(), [batch, time, heads, V]);
    assert_eq!(scores.shape().dims::<4>(), [batch, time, heads, 1]);
    assert_eq!(valid.shape().dims::<2>(), [batch, time]);
    assert_eq!(documents.shape().dims::<2>(), [batch, time]);
    assert_eq!(sink.shape().dims::<1>(), [heads]);
    assert!((1..=64).contains(&capacity), "HOLA capacity must be 1..=64");
    assert!((1..=16).contains(&chunk), "HOLA chunk must be 1..=16");
    for tensor in [&q, &k, &v, &scores, &valid, &documents, &sink] {
        assert_eq!(
            tensor.dtype,
            DType::F32,
            "fused HOLA supports f32 tensors only"
        );
        assert_eq!(tensor.device, q.device, "HOLA inputs must share a device");
        assert_u32_tensor(tensor);
    }
    let lanes = (capacity + chunk).next_power_of_two().max(V);
    let schedule = empty(&q, [batch, heads, time.div_ceil(chunk) + 1, capacity]);
    let expiry = empty(&q, [batch, time, heads]);
    let output = empty(&q, [batch, time, heads, V]);
    let stats = empty(&q, [batch, time, heads, 2]);
    selection_kernel::launch(
        &q.client,
        count(&q, batch * heads),
        CubeDim::new_1d(lanes as u32),
        scores.into_tensor_arg(),
        valid.clone().into_tensor_arg(),
        documents.clone().into_tensor_arg(),
        schedule.clone().into_tensor_arg(),
        expiry.clone().into_tensor_arg(),
        capacity,
        chunk,
        recency,
    );
    forward_kernel::launch(
        &q.client,
        count(&q, batch * time * heads),
        CubeDim::new_1d(lanes as u32),
        q.clone().into_tensor_arg(),
        k.into_tensor_arg(),
        v.into_tensor_arg(),
        valid.into_tensor_arg(),
        documents.into_tensor_arg(),
        sink.into_tensor_arg(),
        schedule.clone().into_tensor_arg(),
        output.clone().into_tensor_arg(),
        stats.clone().into_tensor_arg(),
        capacity,
        chunk,
        lanes,
    );
    (output, schedule, expiry, stats)
}

pub(crate) fn run_backward(
    q: CubeTensor,
    k: CubeTensor,
    v: CubeTensor,
    valid: CubeTensor,
    documents: CubeTensor,
    sink: CubeTensor,
    schedule: CubeTensor,
    expiry: CubeTensor,
    output: CubeTensor,
    stats: CubeTensor,
    upstream: CubeTensor,
    capacity: usize,
    chunk: usize,
) -> (CubeTensor, CubeTensor, CubeTensor, CubeTensor) {
    let [batch, time, heads, _] = q.shape().dims::<4>();
    for tensor in [
        &q, &k, &v, &valid, &documents, &sink, &schedule, &expiry, &output, &stats, &upstream,
    ] {
        assert_u32_tensor(tensor);
    }
    let dq = empty(&q, q.shape());
    let dk = empty(&q, k.shape());
    let dv = empty(&q, v.shape());
    let ds = empty(&q, [heads]);
    let auxiliary = empty(&q, [batch, time, heads, 2]);
    let lanes = (capacity + chunk).next_power_of_two().max(V);
    query_backward_kernel::launch(
        &q.client,
        count(&q, batch * time * heads),
        CubeDim::new_1d(lanes as u32),
        q.clone().into_tensor_arg(),
        k.clone().into_tensor_arg(),
        v.clone().into_tensor_arg(),
        valid.clone().into_tensor_arg(),
        documents.clone().into_tensor_arg(),
        sink.into_tensor_arg(),
        schedule.into_tensor_arg(),
        output.into_tensor_arg(),
        stats.clone().into_tensor_arg(),
        upstream.clone().into_tensor_arg(),
        dq.clone().into_tensor_arg(),
        auxiliary.clone().into_tensor_arg(),
        capacity,
        chunk,
        lanes,
    );
    key_value_backward_kernel::launch(
        &q.client,
        count(&q, batch * time * heads),
        CubeDim::new_1d(V as u32),
        q.clone().into_tensor_arg(),
        k.into_tensor_arg(),
        v.into_tensor_arg(),
        valid.into_tensor_arg(),
        documents.into_tensor_arg(),
        expiry.into_tensor_arg(),
        stats.into_tensor_arg(),
        auxiliary.clone().into_tensor_arg(),
        upstream.into_tensor_arg(),
        dk.clone().into_tensor_arg(),
        dv.clone().into_tensor_arg(),
    );
    sink_backward_kernel::launch(
        &q.client,
        count(&q, heads),
        CubeDim::new_1d(V as u32),
        auxiliary.into_tensor_arg(),
        ds.clone().into_tensor_arg(),
    );
    (dq, dk, dv, ds)
}

#[cfg(test)]
mod address_tests {
    use super::{checked_elements, checked_span};

    #[test]
    fn accepts_the_largest_representable_element_span() {
        checked_span(&[u32::MAX as usize], &[1], u32::MAX as u64);
    }

    #[test]
    #[should_panic(expected = "HOLA span exceeds 32-bit addressing")]
    fn rejects_strided_spans_that_do_not_fit_kernel_metadata() {
        checked_span(&[2, 2], &[u32::MAX as usize, 1], u32::MAX as u64);
    }

    #[test]
    #[should_panic(expected = "HOLA tensor element count overflows usize")]
    fn rejects_allocation_count_overflow_before_allocating() {
        checked_elements(&[usize::MAX, 2]);
    }

    #[test]
    #[should_panic(expected = "HOLA backing buffer exceeds 32-bit addressing")]
    fn rejects_oversized_backing_storage_even_for_small_views() {
        checked_span(&[1], &[1], u32::MAX as u64 + 1);
    }
}
