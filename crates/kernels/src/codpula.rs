//! Short-chain log-space inference without a chain/position/state/vocabulary tensor.
use std::sync::atomic::{AtomicU64, Ordering};

use burn::backend::{TensorMetadata, cubecl::dtype_to_storage_type, ops::FloatTensorOps};
use burn::{prelude::Shape, tensor::DType};
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use cubecl::{features::TypeUsage, frontend::Tensor as View, prelude::*};

static LAUNCHES: AtomicU64 = AtomicU64::new(0);
static ALLOCATED: AtomicU64 = AtomicU64::new(0);
static LEAF_FORWARD: AtomicU64 = AtomicU64::new(0);
static LEAF_BACKWARD: AtomicU64 = AtomicU64::new(0);
static CHAIN_FORWARD: AtomicU64 = AtomicU64::new(0);
static CHAIN_BACKWARD: AtomicU64 = AtomicU64::new(0);
static MARGINALS: AtomicU64 = AtomicU64::new(0);
static EMISSION_FLOWS: AtomicU64 = AtomicU64::new(0);

/// Cumulative submitted work, not synchronized timings or peak device memory.
#[derive(Clone, Copy, Debug, Default)]
pub struct CodpulaKernelCounters {
    pub launches: u64,
    pub allocated_bytes: u64,
    pub leaf_forward_calls: u64,
    pub leaf_backward_calls: u64,
    pub chain_forward_calls: u64,
    pub chain_backward_calls: u64,
    pub marginal_calls: u64,
    pub emission_flow_calls: u64,
}

pub fn codpula_kernel_counters() -> CodpulaKernelCounters {
    CodpulaKernelCounters {
        launches: LAUNCHES.load(Ordering::Relaxed),
        allocated_bytes: ALLOCATED.load(Ordering::Relaxed),
        leaf_forward_calls: LEAF_FORWARD.load(Ordering::Relaxed),
        leaf_backward_calls: LEAF_BACKWARD.load(Ordering::Relaxed),
        chain_forward_calls: CHAIN_FORWARD.load(Ordering::Relaxed),
        chain_backward_calls: CHAIN_BACKWARD.load(Ordering::Relaxed),
        marginal_calls: MARGINALS.load(Ordering::Relaxed),
        emission_flow_calls: EMISSION_FLOWS.load(Ordering::Relaxed),
    }
}

#[cube]
fn ix2<F: Float>(x: &View<F>, a: usize, b: usize) -> usize {
    a * x.stride(0) + b * x.stride(1)
}
#[cube]
fn ix3<F: Float>(x: &View<F>, a: usize, b: usize, c: usize) -> usize {
    a * x.stride(0) + b * x.stride(1) + c * x.stride(2)
}

#[cube]
fn valid_token<F: Float>(token: F, vocabulary: usize) -> bool {
    token >= F::new(0.0f32) && token < F::cast_from(vocabulary) && F::floor(token) == token
}

#[cube(launch)]
fn leaf_kernel<F: Float>(
    logits: &View<F>, pi: &View<F>, valid: &View<F>, out: &mut View<F>,
    #[define(F)] _dtype: ElemType,
) {
    let task = CUBE_POS as usize;
    let lane = UNIT_POS as usize;
    let k = task % pi.shape(0);
    let row = task / pi.shape(0);
    let t = row % logits.shape(1);
    let c = row / logits.shape(1);
    if c >= logits.shape(0) { terminate!(); }
    let active = valid[ix2(valid, c, t)] == F::new(1.0f32);
    let mut maximum = F::new(f32::NEG_INFINITY);
    let mut v = lane;
    if active {
        while v < logits.shape(2) {
            maximum = F::max(maximum, logits[ix3(logits, c, t, v)] + pi[ix2(pi, k, v)]);
            v += 128;
        }
    }
    let mut reduction = Shared::<[F]>::new_slice(128usize);
    reduction[lane] = maximum;
    sync_cube();
    let mut offset = 64usize;
    while offset > 0 {
        if lane < offset { reduction[lane] = F::max(reduction[lane], reduction[lane + offset]); }
        sync_cube();
        offset /= 2;
    }
    maximum = reduction[0];
    let mut sum = F::new(0.0f32);
    v = lane;
    if active && maximum != F::new(f32::NEG_INFINITY) {
        while v < logits.shape(2) {
            sum += F::exp(logits[ix3(logits, c, t, v)] + pi[ix2(pi, k, v)] - maximum);
            v += 128;
        }
    }
    sync_cube();
    reduction[lane] = sum;
    sync_cube();
    offset = 64;
    while offset > 0 {
        if lane < offset { reduction[lane] = reduction[lane] + reduction[lane + offset]; }
        sync_cube();
        offset /= 2;
    }
    if lane == 0 {
        out[task] = if active {
            maximum + F::ln(reduction[0])
        } else if valid[ix2(valid, c, t)] == F::new(0.0f32) {
            F::new(0.0f32)
        } else {
            F::new(f32::NAN)
        };
    }
}

#[cube(launch)]
fn leaf_input_backward_kernel<F: Float>(
    logits: &View<F>, pi: &View<F>, valid: &View<F>, normalizers: &View<F>,
    upstream: &View<F>, out: &mut View<F>, #[define(F)] _dtype: ElemType,
) {
    let task = CUBE_POS as usize * 128 + UNIT_POS as usize;
    let v = task % logits.shape(2);
    let row = task / logits.shape(2);
    let t = row % logits.shape(1);
    let c = row / logits.shape(1);
    if c >= logits.shape(0) { terminate!(); }
    let mut sum = F::new(0.0f32);
    if valid[ix2(valid, c, t)] == F::new(1.0f32) {
        let mut k = 0usize;
        while k < pi.shape(0) {
            let weight = upstream[ix3(upstream, c, t, k)];
            if weight != F::new(0.0f32) {
                sum += weight * F::exp(logits[ix3(logits, c, t, v)] + pi[ix2(pi, k, v)]
                    - normalizers[ix3(normalizers, c, t, k)]);
            }
            k += 1;
        }
    }
    if valid[ix2(valid, c, t)] != F::new(0.0f32) && valid[ix2(valid, c, t)] != F::new(1.0f32) {
        sum = F::new(f32::NAN);
    }
    out[task] = sum;
}

// The same reduction computes tied leaf-normalizer VJPs and posterior emission counts.
#[cube(launch)]
fn emission_reduce_kernel<F: Float>(
    logits: &View<F>, pi: &View<F>, valid: &View<F>, normalizers: &View<F>,
    weights: &View<F>, tokens: &View<F>, observed: &View<F>, out: &mut View<F>,
    #[comptime] counts: bool, #[define(F)] _dtype: ElemType,
) {
    let task = CUBE_POS as usize;
    let lane = UNIT_POS as usize;
    let v = task % pi.shape(1);
    let k = task / pi.shape(1);
    if k >= pi.shape(0) { terminate!(); }
    let mut row = lane;
    let mut sum = F::new(0.0f32);
    while row < logits.shape(0) * logits.shape(1) {
        let t = row % logits.shape(1);
        let c = row / logits.shape(1);
        let mask = valid[ix2(valid, c, t)];
        if mask == F::new(1.0f32) {
            let weight = weights[ix3(weights, c, t, k)];
            if counts && observed[ix2(observed, c, t)] == F::new(1.0f32) {
                let token = tokens[ix2(tokens, c, t)];
                if !valid_token::<F>(token, pi.shape(1)) {
                    sum = F::new(f32::NAN);
                } else if token == F::cast_from(v) {
                    sum += weight;
                }
            } else if counts && observed[ix2(observed, c, t)] != F::new(0.0f32) {
                sum = F::new(f32::NAN);
            } else if weight != F::new(0.0f32) {
                sum += weight * F::exp(logits[ix3(logits, c, t, v)] + pi[ix2(pi, k, v)]
                    - normalizers[ix3(normalizers, c, t, k)]);
            }
        } else if mask != F::new(0.0f32) {
            sum = F::new(f32::NAN);
        }
        row += 64;
    }
    let mut reduction = Shared::<[F]>::new_slice(64usize);
    reduction[lane] = sum;
    sync_cube();
    let mut offset = 32usize;
    while offset > 0 {
        if lane < offset { reduction[lane] = reduction[lane] + reduction[lane + offset]; }
        sync_cube();
        offset /= 2;
    }
    if lane == 0 { out[task] = reduction[0]; }
}

#[cube(launch)]
fn chain_kernel<F: Float>(
    evidence: &View<F>, transition: &View<F>, root: &View<F>, valid: &View<F>,
    log_z: &mut View<F>, gamma: &mut View<F>, edges: &mut View<F>,
    #[comptime] posterior: bool,
    #[define(F)] _dtype: ElemType,
) {
    let c = CUBE_POS as usize;
    let s = UNIT_POS as usize;
    if c >= evidence.shape(0) { terminate!(); }
    let states = evidence.shape(2);
    let time = evidence.shape(1);
    let mut length = 0usize;
    let mut invalid = false;
    let mut t = 0usize;
    while t < time {
        let mask = valid[ix2(valid, c, t)];
        if mask == F::new(1.0f32) {
            if length != t { invalid = true; }
            length += 1;
        } else if mask != F::new(0.0f32) { invalid = true; }
        if posterior && s < states { gamma[(c * time + t) * states + s] = F::new(0.0f32); }
        t += 1;
    }
    let mut alpha = Shared::<[F]>::new_slice(16usize * 32usize);
    let mut matrix = Shared::<[F]>::new_slice(32usize * 32usize);
    let mut beta = Shared::<[F]>::new_slice(32usize);
    let mut next = Shared::<[F]>::new_slice(32usize);
    let mut total = Shared::<[F]>::new_slice(1usize);
    let mut flows = Array::<F>::new(32usize);
    let mut u = 0usize;
    while u < 32 {
        if posterior { flows[u] = F::new(0.0f32); }
        if length > 1 && s < states && u < states { matrix[s * 32 + u] = transition[ix3(transition, c, s, u)]; }
        u += 1;
    }
    if posterior { beta[s] = F::new(0.0f32); }
    if s < states && length > 0 {
        let root_row = if root.shape(0) == states { 0usize } else { c * states };
        alpha[s] = root[(root_row + s) * root.stride(0)] + evidence[ix3(evidence, c, 0, s)];
    }
    sync_cube();
    t = 1;
    while t < length {
        if s < states {
            let mut maximum = F::new(f32::NEG_INFINITY);
            u = 0;
            while u < states {
                maximum = F::max(maximum, alpha[(t - 1) * 32 + u] + matrix[u * 32 + s]);
                u += 1;
            }
            let mut sum = F::new(0.0f32);
            u = 0;
            if maximum != F::new(f32::NEG_INFINITY) {
                while u < states {
                    sum += F::exp(alpha[(t - 1) * 32 + u] + matrix[u * 32 + s] - maximum);
                    u += 1;
                }
            }
            alpha[t * 32 + s] = evidence[ix3(evidence, c, t, s)] + maximum + F::ln(sum);
        }
        sync_cube();
        t += 1;
    }
    if s == 0 {
        let mut z = if invalid { F::new(f32::NAN) } else { F::new(0.0f32) };
        if length > 0 && !invalid {
            let mut maximum = F::new(f32::NEG_INFINITY);
            u = 0;
            while u < states { maximum = F::max(maximum, alpha[(length - 1) * 32 + u]); u += 1; }
            let mut sum = F::new(0.0f32);
            u = 0;
            if maximum != F::new(f32::NEG_INFINITY) {
                while u < states { sum += F::exp(alpha[(length - 1) * 32 + u] - maximum); u += 1; }
            }
            z = maximum + F::ln(sum);
        }
        total[0] = z;
        log_z[c] = z;
    }
    sync_cube();
    // A comptime terminate!() does not exclude subsequent CubeCL expansion.
    if posterior {
        t = length;
        while t > 0 {
            t -= 1;
            if s < states {
                gamma[(c * time + t) * states + s] = F::exp(alpha[t * 32 + s] + beta[s] - total[0]);
                if t > 0 {
                    let mut maximum = F::new(f32::NEG_INFINITY);
                    u = 0;
                    while u < states {
                        let value = matrix[s * 32 + u] + evidence[ix3(evidence, c, t, u)] + beta[u];
                        maximum = F::max(maximum, value);
                        flows[u] += F::exp(alpha[(t - 1) * 32 + s] + value - total[0]);
                        u += 1;
                    }
                    let mut sum = F::new(0.0f32);
                    u = 0;
                    if maximum != F::new(f32::NEG_INFINITY) {
                        while u < states {
                            sum += F::exp(matrix[s * 32 + u] + evidence[ix3(evidence, c, t, u)] + beta[u] - maximum);
                            u += 1;
                        }
                    }
                    next[s] = maximum + F::ln(sum);
                }
            }
            sync_cube();
            if t > 0 && s < states { beta[s] = next[s]; }
            sync_cube();
        }
        if s < states {
            u = 0;
            while u < states {
                edges[(c * states + s) * states + u] = if invalid { F::new(f32::NAN) } else { flows[u] };
                u += 1;
            }
        }
    }
}

#[cube(launch)]
fn chain_backward_kernel<F: Float>(
    gamma: &View<F>, edges: &View<F>, upstream: &View<F>,
    de: &mut View<F>, da: &mut View<F>, #[define(F)] _dtype: ElemType,
) {
    let c = CUBE_POS as usize;
    let s = UNIT_POS as usize;
    if c >= gamma.shape(0) { terminate!(); }
    let states = gamma.shape(2);
    if s < states {
        let weight = upstream[c * upstream.stride(0)];
        let mut t = 0usize;
        while t < gamma.shape(1) {
            de[(c * gamma.shape(1) + t) * states + s] = weight * gamma[ix3(gamma, c, t, s)];
            t += 1;
        }
        let mut u = 0usize;
        while u < states {
            da[(c * states + s) * states + u] = weight * edges[ix3(edges, c, s, u)];
            u += 1;
        }
    }
}

#[cube(launch)]
fn root_backward_kernel<F: Float>(
    gamma: &View<F>, upstream: &View<F>, out: &mut View<F>, #[define(F)] _dtype: ElemType,
) {
    let s = UNIT_POS as usize;
    let row = CUBE_POS as usize;
    if s < gamma.shape(2) {
        if out.shape(0) == gamma.shape(2) {
            let mut c = 0usize;
            let mut sum = F::new(0.0f32);
            while c < gamma.shape(0) {
                sum += upstream[c * upstream.stride(0)] * gamma[ix3(gamma, c, 0, s)];
                c += 1;
            }
            out[s] = sum;
        } else if row < gamma.shape(0) {
            out[row * gamma.shape(2) + s] = upstream[row * upstream.stride(0)] * gamma[ix3(gamma, row, 0, s)];
        }
    }
}

#[cube(launch)]
fn marginal_kernel<F: Float>(
    logits: &View<F>, pi: &View<F>, normalizers: &View<F>, gamma: &View<F>,
    tokens: &View<F>, observed: &View<F>, valid: &View<F>, out: &mut View<F>,
    #[define(F)] _dtype: ElemType,
) {
    let task = CUBE_POS as usize * 128 + UNIT_POS as usize;
    let v = task % logits.shape(2);
    let row = task / logits.shape(2);
    let t = row % logits.shape(1);
    let c = row / logits.shape(1);
    if c >= logits.shape(0) { terminate!(); }
    let mut sum = F::new(0.0f32);
    if valid[ix2(valid, c, t)] == F::new(1.0f32) {
        if observed[ix2(observed, c, t)] == F::new(1.0f32) {
            let token = tokens[ix2(tokens, c, t)];
            if !valid_token::<F>(token, pi.shape(1)) {
                sum = F::new(f32::NAN);
            } else if token == F::cast_from(v) {
                sum = F::new(1.0f32);
            }
        } else if observed[ix2(observed, c, t)] != F::new(0.0f32) {
            sum = F::new(f32::NAN);
        } else {
            let mut k = 0usize;
            while k < pi.shape(0) {
                sum += gamma[ix3(gamma, c, t, k)] * F::exp(logits[ix3(logits, c, t, v)]
                    + pi[ix2(pi, k, v)] - normalizers[ix3(normalizers, c, t, k)]);
                k += 1;
            }
        }
    }
    if valid[ix2(valid, c, t)] != F::new(0.0f32) && valid[ix2(valid, c, t)] != F::new(1.0f32) {
        sum = F::new(f32::NAN);
    }
    out[task] = sum;
}

fn validate(tensors: &[&CubeTensor]) {
    let first = tensors[0];
    assert!(matches!(first.dtype, DType::F32 | DType::F64), "Codpula fused execution requires f32 or f64");
    let usage = first.client.properties().type_usage(dtype_to_storage_type(first.dtype));
    assert!(usage.is_superset(TypeUsage::Buffer | TypeUsage::Conversion | TypeUsage::Arithmetic),
        "Codpula fused dtype is unsupported on this device");
    let bytes = first.dtype.size();
    for tensor in tensors {
        assert_eq!(tensor.dtype, first.dtype, "Codpula fused dtype mismatch");
        assert_eq!(tensor.device, first.device, "Codpula fused device mismatch");
        let mut span = 1usize;
        for (&dim, &stride) in tensor.meta.shape().as_slice().iter().zip(tensor.meta.strides().iter()) {
            assert!(dim > 0 && stride <= u32::MAX as usize, "Codpula fused invalid shape or stride");
            span = (dim - 1).checked_mul(stride).and_then(|n| span.checked_add(n))
                .expect("Codpula fused address overflow");
        }
        assert!(span <= u32::MAX as usize && tensor.handle.size() / bytes as u64 <= u32::MAX as u64,
            "Codpula fused exceeds 32-bit element addressing");
    }
}
fn empty(x: &CubeTensor, shape: impl Into<Shape>) -> CubeTensor {
    let shape = shape.into();
    assert!(shape.num_elements() <= u32::MAX as usize, "Codpula fused allocation exceeds 32-bit addressing");
    ALLOCATED.fetch_add((shape.num_elements() * x.dtype.size()) as u64, Ordering::Relaxed);
    CubeBackend::float_empty(shape, &x.device, x.dtype.into())
}
fn count(x: &CubeTensor, groups: usize) -> CubeCount {
    let (max_x, max_y, _) = x.client.properties().hardware.max_cube_count;
    let nx = groups.min(max_x.min(65535) as usize).max(1);
    assert!(groups.div_ceil(nx) <= max_y as usize, "Codpula fused dispatch exceeds device grid");
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    CubeCount::Static(nx as u32, groups.div_ceil(nx) as u32, 1)
}
fn leaf_shapes(logits: &CubeTensor, pi: &CubeTensor, valid: &CubeTensor) -> [usize; 4] {
    let [chains, time, vocabulary] = logits.shape().dims::<3>();
    let [states, pv] = pi.shape().dims::<2>();
    assert!((1..=16).contains(&time) && (1..=32).contains(&states) && (1..=257).contains(&vocabulary),
        "Codpula fused requires L<=16, K<=32, V<=257");
    assert_eq!(pv, vocabulary);
    assert_eq!(valid.shape().dims::<2>(), [chains, time]);
    [chains, time, states, vocabulary]
}

pub(crate) fn leaf_forward(logits: CubeTensor, pi: CubeTensor, valid: CubeTensor) -> CubeTensor {
    validate(&[&logits, &pi, &valid]);
    let [c, l, k, _] = leaf_shapes(&logits, &pi, &valid);
    let out = empty(&logits, [c, l, k]);
    LEAF_FORWARD.fetch_add(1, Ordering::Relaxed);
    leaf_kernel::launch(&logits.client, count(&logits, c * l * k), CubeDim::new_1d(128),
        logits.clone().into_tensor_arg(), pi.into_tensor_arg(), valid.into_tensor_arg(),
        out.clone().into_tensor_arg(), dtype_to_storage_type(logits.dtype));
    out
}

pub(crate) fn leaf_backward(
    logits: CubeTensor, pi: CubeTensor, valid: CubeTensor, normalizers: CubeTensor, upstream: CubeTensor,
    need_logits: bool, need_pi: bool,
) -> (CubeTensor, CubeTensor) {
    validate(&[&logits, &pi, &valid, &normalizers, &upstream]);
    let [c, l, k, v] = leaf_shapes(&logits, &pi, &valid);
    assert_eq!(normalizers.shape().dims::<3>(), [c, l, k]);
    assert_eq!(upstream.shape(), normalizers.shape());
    let dx = if need_logits { empty(&logits, logits.shape()) } else { logits.clone() };
    let dp = if need_pi { empty(&logits, pi.shape()) } else { pi.clone() };
    LEAF_BACKWARD.fetch_add(1, Ordering::Relaxed);
    let dtype = dtype_to_storage_type(logits.dtype);
    if need_logits {
        leaf_input_backward_kernel::launch(&logits.client, count(&logits, (c * l * v).div_ceil(128)), CubeDim::new_1d(128),
            logits.clone().into_tensor_arg(), pi.clone().into_tensor_arg(), valid.clone().into_tensor_arg(),
            normalizers.clone().into_tensor_arg(), upstream.clone().into_tensor_arg(), dx.clone().into_tensor_arg(), dtype);
    }
    if need_pi {
        emission_reduce_kernel::launch(&logits.client, count(&logits, k * v), CubeDim::new_1d(64),
            logits.clone().into_tensor_arg(), pi.into_tensor_arg(), valid.clone().into_tensor_arg(),
            normalizers.into_tensor_arg(), upstream.into_tensor_arg(), valid.clone().into_tensor_arg(),
            valid.into_tensor_arg(), dp.clone().into_tensor_arg(), false, dtype);
    }
    (dx, dp)
}

pub(crate) fn chain_forward(
    evidence: CubeTensor, transition: CubeTensor, root: CubeTensor, valid: CubeTensor,
    posterior: bool,
) -> (CubeTensor, CubeTensor, CubeTensor) {
    validate(&[&evidence, &transition, &root, &valid]);
    let [c, l, k] = evidence.shape().dims::<3>();
    assert!((1..=16).contains(&l) && (1..=32).contains(&k), "Codpula fused chain requires L<=16 and K<=32");
    assert_eq!(transition.shape().dims::<3>(), [c, k, k]);
    assert!(root.shape().dims::<1>() == [k] || root.shape().dims::<1>() == [c * k], "Codpula roots must be shared [K] or per-row [C*K]");
    assert_eq!(valid.shape().dims::<2>(), [c, l]);
    let z = empty(&evidence, [c]);
    let gamma = if posterior { empty(&evidence, [c, l, k]) } else { evidence.clone() };
    let edges = if posterior { empty(&evidence, [c, k, k]) } else { transition.clone() };
    CHAIN_FORWARD.fetch_add(1, Ordering::Relaxed);
    chain_kernel::launch(&evidence.client, count(&evidence, c), CubeDim::new_1d(32),
        evidence.clone().into_tensor_arg(), transition.into_tensor_arg(), root.into_tensor_arg(),
        valid.into_tensor_arg(), z.clone().into_tensor_arg(), gamma.clone().into_tensor_arg(),
        edges.clone().into_tensor_arg(), posterior, dtype_to_storage_type(evidence.dtype));
    (z, gamma, edges)
}

pub(crate) fn chain_backward(gamma: CubeTensor, edges: CubeTensor, upstream: CubeTensor, root_count: usize) -> (CubeTensor, CubeTensor, CubeTensor) {
    validate(&[&gamma, &edges, &upstream]);
    let [c, l, k] = gamma.shape().dims::<3>();
    assert!((1..=16).contains(&l) && (1..=32).contains(&k));
    assert_eq!(edges.shape().dims::<3>(), [c, k, k]);
    assert_eq!(upstream.shape().dims::<1>(), [c]);
    let de = empty(&gamma, gamma.shape());
    let da = empty(&gamma, edges.shape());
    assert!(root_count == k || root_count == c * k, "invalid Codpula root derivative shape");
    let dr = empty(&gamma, [root_count]);
    CHAIN_BACKWARD.fetch_add(1, Ordering::Relaxed);
    let dtype = dtype_to_storage_type(gamma.dtype);
    chain_backward_kernel::launch(&gamma.client, count(&gamma, c), CubeDim::new_1d(32),
        gamma.clone().into_tensor_arg(), edges.into_tensor_arg(), upstream.clone().into_tensor_arg(),
        de.clone().into_tensor_arg(), da.clone().into_tensor_arg(), dtype);
    root_backward_kernel::launch(&gamma.client, count(&gamma, if root_count == k { 1 } else { c }), CubeDim::new_1d(32),
        gamma.clone().into_tensor_arg(), upstream.into_tensor_arg(), dr.clone().into_tensor_arg(), dtype);
    (de, da, dr)
}

pub(crate) fn readout(
    logits: CubeTensor, pi: CubeTensor, normalizers: CubeTensor, gamma: CubeTensor,
    tokens: CubeTensor, observed: CubeTensor, valid: CubeTensor, counts: bool,
) -> CubeTensor {
    validate(&[&logits, &pi, &normalizers, &gamma, &tokens, &observed, &valid]);
    let [c, l, k, v] = leaf_shapes(&logits, &pi, &valid);
    assert_eq!(normalizers.shape().dims::<3>(), [c, l, k]);
    assert_eq!(gamma.shape(), normalizers.shape());
    assert_eq!(tokens.shape().dims::<2>(), [c, l]);
    assert_eq!(observed.shape(), tokens.shape());
    let dtype = dtype_to_storage_type(logits.dtype);
    if counts {
        EMISSION_FLOWS.fetch_add(1, Ordering::Relaxed);
        let out = empty(&logits, [k, v]);
        emission_reduce_kernel::launch(&logits.client, count(&logits, k * v), CubeDim::new_1d(64),
            logits.clone().into_tensor_arg(), pi.into_tensor_arg(), valid.into_tensor_arg(),
            normalizers.into_tensor_arg(), gamma.into_tensor_arg(), tokens.into_tensor_arg(),
            observed.into_tensor_arg(), out.clone().into_tensor_arg(), true, dtype);
        out
    } else {
        MARGINALS.fetch_add(1, Ordering::Relaxed);
        let out = empty(&logits, [c, l, v]);
        marginal_kernel::launch(&logits.client, count(&logits, (c * l * v).div_ceil(128)), CubeDim::new_1d(128),
            logits.clone().into_tensor_arg(), pi.into_tensor_arg(), normalizers.into_tensor_arg(),
            gamma.into_tensor_arg(), tokens.into_tensor_arg(), observed.into_tensor_arg(),
            valid.into_tensor_arg(), out.clone().into_tensor_arg(), dtype);
        out
    }
}
