//! Original CubeCL implementation of paper §3.3 (18–25), §3.4 and Appendix B.
//! Causal products are formed directly (never inverse cumulative decay), including
//! reset factors of zero and padding identity factors. Only chunk boundaries recur.
use burn::backend::{TensorMetadata, cubecl::dtype_to_storage_type, ops::FloatTensorOps};
use burn::prelude::Shape;
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use cubecl::frontend::Tensor as View;
use cubecl::prelude::*;

pub(crate) const C: usize = 16;
const K: usize = 32;
const V: usize = 64;

#[cube]
fn ix<F: Numeric>(x: &View<F>, b: usize, t: usize, h: usize, i: usize) -> usize {
    b * x.stride(0) + t * x.stride(1) + h * x.stride(2) + i * x.stride(3)
}
#[cube]
fn di(group: usize, r: usize, s: usize, i: usize) -> usize {
    ((group * (C + 1) + r) * (C + 1) + s) * K + i
}

#[cube(launch)]
fn decay_kernel<I: Int>(
    a: &View<f32>,
    start: &View<I>,
    end: &View<I>,
    d: &mut View<f32>,
    #[define(I)] _dtype: ElemType,
) {
    let task = CUBE_POS * 64 + UNIT_POS as usize;
    let i = task % K;
    let group = task / K;
    let chunks = (a.shape(1) + C - 1) / C;
    let c = group % chunks;
    let h = group / chunks % a.shape(2);
    let b = group / chunks / a.shape(2);
    if b >= a.shape(0) {
        terminate!();
    }
    let mut s = 0usize;
    while s <= C {
        let mut product = 1.0f32;
        let mut r = s;
        while r <= C {
            d[di(group, r, s, i)] = product;
            if r < C {
                let t = c * C + r;
                if t < a.shape(1) {
                    let st = start[b * start.stride(0) + t * start.stride(1)];
                    let en = end[b * end.stride(0) + t * end.stride(1)];
                    if en >= st {
                        if st == I::cast_from(t) {
                            product = 0.0f32;
                        }
                        product *= a[ix(a, b, t, h, i)];
                    }
                }
            }
            r += 1;
        }
        s += 1;
    }
}

#[cube(launch)]
fn score_kernel<F: Float, I: Int>(
    q: &View<F>,
    k: &View<F>,
    e: &View<f32>,
    start: &View<I>,
    end: &View<I>,
    d: &View<f32>,
    lower: &mut View<f32>,
    m: &mut View<f32>,
    #[define(F, I)] _dtype: [ElemType; 2],
) {
    let task = CUBE_POS * 64 + UNIT_POS as usize;
    let s = task % C;
    let r = task / C % C;
    let group = task / (C * C);
    let chunks = (q.shape(1) + C - 1) / C;
    let c = group % chunks;
    let h = group / chunks % q.shape(2);
    let b = group / chunks / q.shape(2);
    if b >= q.shape(0) {
        terminate!();
    }
    let t = c * C + r;
    let u = c * C + s;
    let mut ts = 0.0f32;
    let mut ms = 0.0f32;
    if s <= r && t < q.shape(1) && u < q.shape(1) {
        let valid_t = end[b * end.stride(0) + t * end.stride(1)]
            >= start[b * start.stride(0) + t * start.stride(1)];
        let valid_u = end[b * end.stride(0) + u * end.stride(1)]
            >= start[b * start.stride(0) + u * start.stride(1)];
        if valid_t && valid_u {
            let mut i = 0usize;
            while i < K {
                let factor = d[di(group, r + 1, s + 1, i)] * f32::cast_from(k[ix(k, b, u, h, i)]);
                ms += f32::cast_from(q[ix(q, b, t, h, i)]) * factor;
                if s < r {
                    ts += e[ix(e, b, t, h, i)] * f32::cast_from(k[ix(k, b, t, h, i)]) * factor;
                }
                i += 1;
            }
        }
    }
    lower[task] = ts;
    m[task] = ms;
}

#[cube(launch)]
fn inverse_kernel(lower: &View<f32>, inverse: &mut View<f32>) {
    let task = CUBE_POS * 64 + UNIT_POS as usize;
    if task >= inverse.len() / C {
        terminate!();
    }
    let r = task % C;
    let group = task / C;
    let mut row = Array::<f32>::new(C);
    let mut s = 0usize;
    while s < C {
        row[s] = if s == r { 1.0f32 } else { 0.0f32 };
        s += 1;
    }
    s = r;
    while s > 0 {
        s -= 1;
        let mut value = 0.0f32;
        let mut p = s + 1;
        while p <= r {
            value -= row[p] * lower[(group * C + p) * C + s];
            p += 1;
        }
        row[s] = value;
    }
    s = 0usize;
    while s < C {
        inverse[task * C + s] = row[s];
        s += 1;
    }
}

// Eq. (22–23): R=A(Z-E_gamma S0), S1=gamma_C S0+K_tail^T R.
#[cube(launch)]
fn forward_boundary_kernel<F: Float, I: Int>(
    k: &View<F>,
    v: &View<F>,
    e: &View<f32>,
    w: &View<f32>,
    start: &View<I>,
    end: &View<I>,
    d: &View<f32>,
    inverse: &View<f32>,
    saved: &mut View<f32>,
    #[define(F, I)] _dtype: [ElemType; 2],
) {
    let task = CUBE_POS * 64 + UNIT_POS as usize;
    let j = task % V;
    let h = task / V % k.shape(2);
    let b = task / V / k.shape(2);
    if b >= k.shape(0) {
        terminate!();
    }
    let chunks = (k.shape(1) + C - 1) / C;
    let mut state = Array::<f32>::new(K);
    let mut rhs = Array::<f32>::new(C);
    let mut delta = Array::<f32>::new(C);
    let mut i = 0usize;
    while i < K {
        state[i] = 0.0f32;
        i += 1;
    }
    let mut c = 0usize;
    while c < chunks {
        let group = (b * k.shape(2) + h) * chunks + c;
        i = 0usize;
        while i < K {
            saved[(group * (K + C) + i) * V + j] = state[i];
            i += 1;
        }
        let mut r = 0usize;
        while r < C {
            let t = c * C + r;
            let mut value = 0.0f32;
            if t < k.shape(1) {
                if end[b * end.stride(0) + t * end.stride(1)]
                    >= start[b * start.stride(0) + t * start.stride(1)]
                {
                    value = w[ix(w, b, t, h, j)] * f32::cast_from(v[ix(v, b, t, h, j)]);
                    i = 0usize;
                    while i < K {
                        value -= e[ix(e, b, t, h, i)]
                            * f32::cast_from(k[ix(k, b, t, h, i)])
                            * d[di(group, r + 1, 0, i)]
                            * state[i];
                        i += 1;
                    }
                }
            }
            rhs[r] = value;
            r += 1;
        }
        r = 0usize;
        while r < C {
            let mut value = 0.0f32;
            let mut s = 0usize;
            while s <= r {
                value += inverse[(group * C + r) * C + s] * rhs[s];
                s += 1;
            }
            delta[r] = value;
            saved[(group * (K + C) + K + r) * V + j] = value;
            r += 1;
        }
        i = 0usize;
        while i < K {
            let mut value = d[di(group, C, 0, i)] * state[i];
            r = 0usize;
            while r < C {
                let t = c * C + r;
                if t < k.shape(1) {
                    if end[b * end.stride(0) + t * end.stride(1)]
                        >= start[b * start.stride(0) + t * start.stride(1)]
                    {
                        value += d[di(group, C, r + 1, i)]
                            * f32::cast_from(k[ix(k, b, t, h, i)])
                            * delta[r];
                    }
                }
                r += 1;
            }
            state[i] = value;
            i += 1;
        }
        c += 1;
    }
}

#[cube(launch)]
fn output_kernel<F: Float, I: Int>(
    q: &View<F>,
    start: &View<I>,
    end: &View<I>,
    d: &View<f32>,
    m: &View<f32>,
    saved: &View<f32>,
    out: &mut View<F>,
    #[define(F, I)] _dtype: [ElemType; 2],
) {
    let task = CUBE_POS * 64 + UNIT_POS as usize;
    if task >= out.len() {
        terminate!();
    }
    let j = task % V;
    let h = task / V % q.shape(2);
    let t = task / V / q.shape(2) % q.shape(1);
    let b = task / V / q.shape(2) / q.shape(1);
    if end[b * end.stride(0) + t * end.stride(1)] < start[b * start.stride(0) + t * start.stride(1)]
    {
        out[task] = F::cast_from(0.0f32);
        terminate!();
    }
    let chunks = (q.shape(1) + C - 1) / C;
    let group = (b * q.shape(2) + h) * chunks + t / C;
    let r = t % C;
    let mut value = 0.0f32;
    let mut i = 0usize;
    while i < K {
        value += f32::cast_from(q[ix(q, b, t, h, i)])
            * d[di(group, r + 1, 0, i)]
            * saved[(group * (K + C) + i) * V + j];
        i += 1;
    }
    let mut s = 0usize;
    while s <= r {
        value += m[(group * C + r) * C + s] * saved[(group * (K + C) + K + s) * V + j];
        s += 1;
    }
    out[task] = F::cast_from(value);
}

// Transposed triangular solve: dZ=A^T(M^T dO+K_tail dS1).
// dS0=gamma_C dS1+sum gamma_r(q_r dO_r-e_r dZ_r).
#[cube(launch)]
fn backward_boundary_kernel<F: Float, I: Int>(
    q: &View<F>,
    k: &View<F>,
    e: &View<f32>,
    go: &View<F>,
    start: &View<I>,
    end: &View<I>,
    d: &View<f32>,
    m: &View<f32>,
    inverse: &View<f32>,
    adj: &mut View<f32>,
    dz: &mut View<f32>,
    #[define(F, I)] _dtype: [ElemType; 2],
) {
    let task = CUBE_POS * 64 + UNIT_POS as usize;
    let j = task % V;
    let h = task / V % q.shape(2);
    let b = task / V / q.shape(2);
    if b >= q.shape(0) {
        terminate!();
    }
    let chunks = (q.shape(1) + C - 1) / C;
    let mut state = Array::<f32>::new(K);
    let mut dr = Array::<f32>::new(C);
    let mut z = Array::<f32>::new(C);
    let mut i = 0usize;
    while i < K {
        state[i] = 0.0f32;
        i += 1;
    }
    let mut c = chunks;
    while c > 0 {
        c -= 1;
        let group = (b * q.shape(2) + h) * chunks + c;
        i = 0usize;
        while i < K {
            adj[(group * K + i) * V + j] = state[i];
            i += 1;
        }
        let mut r = 0usize;
        while r < C {
            let t = c * C + r;
            let mut value = 0.0f32;
            if t < q.shape(1) {
                if end[b * end.stride(0) + t * end.stride(1)]
                    >= start[b * start.stride(0) + t * start.stride(1)]
                {
                    i = 0usize;
                    while i < K {
                        value += d[di(group, C, r + 1, i)]
                            * f32::cast_from(k[ix(k, b, t, h, i)])
                            * state[i];
                        i += 1;
                    }
                    let mut s = r;
                    while s < C {
                        let u = c * C + s;
                        if u < q.shape(1) {
                            value +=
                                m[(group * C + s) * C + r] * f32::cast_from(go[ix(go, b, u, h, j)]);
                        }
                        s += 1;
                    }
                }
            }
            dr[r] = value;
            r += 1;
        }
        r = 0usize;
        while r < C {
            let mut value = 0.0f32;
            let mut s = r;
            while s < C {
                value += inverse[(group * C + s) * C + r] * dr[s];
                s += 1;
            }
            z[r] = value;
            dz[(group * C + r) * V + j] = value;
            r += 1;
        }
        i = 0usize;
        while i < K {
            let mut value = d[di(group, C, 0, i)] * state[i];
            r = 0usize;
            while r < C {
                let t = c * C + r;
                if t < q.shape(1) {
                    if end[b * end.stride(0) + t * end.stride(1)]
                        >= start[b * start.stride(0) + t * start.stride(1)]
                    {
                        value += d[di(group, r + 1, 0, i)]
                            * (f32::cast_from(q[ix(q, b, t, h, i)])
                                * f32::cast_from(go[ix(go, b, t, h, j)])
                                - e[ix(e, b, t, h, i)]
                                    * f32::cast_from(k[ix(k, b, t, h, i)])
                                    * z[r]);
                    }
                }
                r += 1;
            }
            state[i] = value;
            i += 1;
        }
    }
}

// Independent token/key-channel VJPs. Reconstruct just this state coordinate
// and adjoint; reduce value columns here rather than allocate four T*K*V buffers.
#[cube(launch)]
fn gradient_kernel<F: Float, I: Int>(
    q: &View<F>,
    k: &View<F>,
    a: &View<f32>,
    e: &View<f32>,
    go: &View<F>,
    start: &View<I>,
    end: &View<I>,
    d: &View<f32>,
    saved: &View<f32>,
    adj: &View<f32>,
    dz: &View<f32>,
    dq: &mut View<F>,
    dk: &mut View<F>,
    da: &mut View<f32>,
    de: &mut View<f32>,
    #[define(F, I)] _dtype: [ElemType; 2],
) {
    let task = CUBE_POS * 64 + UNIT_POS as usize;
    if task >= dq.len() {
        terminate!();
    }
    let i = task % K;
    let h = task / K % q.shape(2);
    let t = task / K / q.shape(2) % q.shape(1);
    let b = task / K / q.shape(2) / q.shape(1);
    let st = start[b * start.stride(0) + t * start.stride(1)];
    let en = end[b * end.stride(0) + t * end.stride(1)];
    let mut gq = 0.0f32;
    let mut gk = 0.0f32;
    let mut ga = 0.0f32;
    let mut ge = 0.0f32;
    if en >= st {
        let chunks = (q.shape(1) + C - 1) / C;
        let c = t / C;
        let r = t % C;
        let group = (b * q.shape(2) + h) * chunks + c;
        let ki = f32::cast_from(k[ix(k, b, t, h, i)]);
        let ei = e[ix(e, b, t, h, i)] * ki;
        let mut j = 0usize;
        while j < V {
            let mut prev = d[di(group, r, 0, i)] * saved[(group * (K + C) + i) * V + j];
            let mut s = 0usize;
            while s < r {
                let u = c * C + s;
                if end[b * end.stride(0) + u * end.stride(1)]
                    >= start[b * start.stride(0) + u * start.stride(1)]
                {
                    prev += d[di(group, r, s + 1, i)]
                        * f32::cast_from(k[ix(k, b, u, h, i)])
                        * saved[(group * (K + C) + K + s) * V + j];
                }
                s += 1;
            }
            if st == I::cast_from(t) {
                prev = 0.0f32;
            }
            let bar = a[ix(a, b, t, h, i)] * prev;
            let delta = saved[(group * (K + C) + K + r) * V + j];
            let og = f32::cast_from(go[ix(go, b, t, h, j)]);
            let mut g = d[di(group, C, r + 1, i)] * adj[(group * K + i) * V + j]
                + f32::cast_from(q[ix(q, b, t, h, i)]) * og;
            s = r + 1;
            while s < C {
                let u = c * C + s;
                if u < q.shape(1) {
                    if end[b * end.stride(0) + u * end.stride(1)]
                        >= start[b * start.stride(0) + u * start.stride(1)]
                    {
                        g += d[di(group, s + 1, r + 1, i)]
                            * (f32::cast_from(q[ix(q, b, u, h, i)])
                                * f32::cast_from(go[ix(go, b, u, h, j)])
                                - e[ix(e, b, u, h, i)]
                                    * f32::cast_from(k[ix(k, b, u, h, i)])
                                    * dz[(group * C + s) * V + j]);
                    }
                }
                s += 1;
            }
            let z = dz[(group * C + r) * V + j];
            gq += (bar + ki * delta) * og;
            gk += g * delta - e[ix(e, b, t, h, i)] * bar * z;
            ge -= ki * bar * z;
            ga += prev * (g - ei * z);
            j += 1;
        }
    }
    dq[task] = F::cast_from(gq);
    dk[task] = F::cast_from(gk);
    da[task] = ga;
    de[task] = ge;
}

#[cube(launch)]
fn value_gradient_kernel<F: Float, I: Int>(
    v: &View<F>,
    w: &View<f32>,
    start: &View<I>,
    end: &View<I>,
    dz: &View<f32>,
    dv: &mut View<F>,
    dw: &mut View<f32>,
    #[define(F, I)] _dtype: [ElemType; 2],
) {
    let task = CUBE_POS * 64 + UNIT_POS as usize;
    if task >= dv.len() {
        terminate!();
    }
    let j = task % V;
    let h = task / V % v.shape(2);
    let t = task / V / v.shape(2) % v.shape(1);
    let b = task / V / v.shape(2) / v.shape(1);
    let chunks = (v.shape(1) + C - 1) / C;
    let group = (b * v.shape(2) + h) * chunks + t / C;
    let mut z = 0.0f32;
    if end[b * end.stride(0) + t * end.stride(1)]
        >= start[b * start.stride(0) + t * start.stride(1)]
    {
        z = dz[(group * C + t % C) * V + j];
    }
    dv[task] = F::cast_from(z * w[ix(w, b, t, h, j)]);
    dw[task] = z * f32::cast_from(v[ix(v, b, t, h, j)]);
}

fn count(client: &Client, tasks: usize) -> CubeCount {
    let cubes = tasks.div_ceil(64);
    let x = cubes
        .min(client.properties().hardware.max_cube_count.0.min(65535) as usize)
        .max(1);
    CubeCount::Static(x as u32, cubes.div_ceil(x) as u32, 1)
}
fn zeros(q: &CubeTensor, dims: impl Into<Shape>) -> CubeTensor {
    CubeBackend::float_zeros(dims.into(), &q.device, burn::tensor::DType::F32.into())
}
fn prepare(
    q: &CubeTensor,
    k: &CubeTensor,
    a: &CubeTensor,
    e: &CubeTensor,
    start: &CubeTensor,
    end: &CubeTensor,
) -> (CubeTensor, CubeTensor, CubeTensor) {
    let [b, t, h, _] = q.shape().dims::<4>();
    let groups = b * h * t.div_ceil(C);
    let d = zeros(q, [groups, C + 1, C + 1, K]);
    let lower = zeros(q, [groups, C, C]);
    let m = zeros(q, [groups, C, C]);
    let inverse = zeros(q, [groups, C, C]);
    let cl = &q.client;
    // Explicit width: CubeDim::new is allowed to return fewer working units.
    let dim = CubeDim::new_1d(64);
    let ty = [
        dtype_to_storage_type(q.dtype),
        dtype_to_storage_type(start.dtype),
    ];
    decay_kernel::launch(
        cl,
        count(cl, groups * K),
        dim,
        a.clone().into_tensor_arg(),
        start.clone().into_tensor_arg(),
        end.clone().into_tensor_arg(),
        d.clone().into_tensor_arg(),
        ty[1],
    );
    score_kernel::launch(
        cl,
        count(cl, groups * C * C),
        dim,
        q.clone().into_tensor_arg(),
        k.clone().into_tensor_arg(),
        e.clone().into_tensor_arg(),
        start.clone().into_tensor_arg(),
        end.clone().into_tensor_arg(),
        d.clone().into_tensor_arg(),
        lower.clone().into_tensor_arg(),
        m.clone().into_tensor_arg(),
        ty,
    );
    inverse_kernel::launch(
        cl,
        count(cl, groups * C),
        dim,
        lower.into_tensor_arg(),
        inverse.clone().into_tensor_arg(),
    );
    (d, m, inverse)
}

pub(crate) fn run_scan(
    q: CubeTensor,
    k: CubeTensor,
    v: CubeTensor,
    a: CubeTensor,
    e: CubeTensor,
    w: CubeTensor,
    start: CubeTensor,
    end: CubeTensor,
) -> (CubeTensor, CubeTensor) {
    let [b, t, h, width] = v.shape().dims::<4>();
    assert_eq!(width, V);
    assert_eq!(q.shape().dims::<4>()[3], K);
    let (d, m, inverse) = prepare(&q, &k, &a, &e, &start, &end);
    let saved = zeros(&q, [b, h, t.div_ceil(C), K + C, V]);
    let out = CubeBackend::float_zeros(v.shape(), &q.device, q.dtype.into());
    let cl = &q.client;
    let dim = CubeDim::new_1d(64);
    let ty = [
        dtype_to_storage_type(q.dtype),
        dtype_to_storage_type(start.dtype),
    ];
    forward_boundary_kernel::launch(
        cl,
        count(cl, b * h * V),
        dim,
        k.into_tensor_arg(),
        v.into_tensor_arg(),
        e.into_tensor_arg(),
        w.into_tensor_arg(),
        start.clone().into_tensor_arg(),
        end.clone().into_tensor_arg(),
        d.clone().into_tensor_arg(),
        inverse.into_tensor_arg(),
        saved.clone().into_tensor_arg(),
        ty,
    );
    output_kernel::launch(
        cl,
        count(cl, b * t * h * V),
        dim,
        q.clone().into_tensor_arg(),
        start.into_tensor_arg(),
        end.into_tensor_arg(),
        d.into_tensor_arg(),
        m.into_tensor_arg(),
        saved.clone().into_tensor_arg(),
        out.clone().into_tensor_arg(),
        ty,
    );
    (out, saved)
}

pub(crate) fn run_backward(
    q: CubeTensor,
    k: CubeTensor,
    v: CubeTensor,
    a: CubeTensor,
    e: CubeTensor,
    w: CubeTensor,
    go: CubeTensor,
    saved: CubeTensor,
    start: CubeTensor,
    end: CubeTensor,
) -> (
    CubeTensor,
    CubeTensor,
    CubeTensor,
    CubeTensor,
    CubeTensor,
    CubeTensor,
) {
    let [b, t, h, _] = v.shape().dims::<4>();
    let (d, m, inverse) = prepare(&q, &k, &a, &e, &start, &end);
    let adj = zeros(&q, [b * h * t.div_ceil(C), K, V]);
    let dz = zeros(&q, [b * h * t.div_ceil(C), C, V]);
    let dq = CubeBackend::float_zeros(q.shape(), &q.device, q.dtype.into());
    let dk = CubeBackend::float_zeros(k.shape(), &q.device, k.dtype.into());
    let dv = CubeBackend::float_zeros(v.shape(), &q.device, v.dtype.into());
    let da = zeros(&q, a.shape());
    let de = zeros(&q, e.shape());
    let dw = zeros(&q, w.shape());
    let cl = &q.client;
    let dim = CubeDim::new_1d(64);
    let ty = [
        dtype_to_storage_type(q.dtype),
        dtype_to_storage_type(start.dtype),
    ];
    backward_boundary_kernel::launch(
        cl,
        count(cl, b * h * V),
        dim,
        q.clone().into_tensor_arg(),
        k.clone().into_tensor_arg(),
        e.clone().into_tensor_arg(),
        go.clone().into_tensor_arg(),
        start.clone().into_tensor_arg(),
        end.clone().into_tensor_arg(),
        d.clone().into_tensor_arg(),
        m.into_tensor_arg(),
        inverse.into_tensor_arg(),
        adj.clone().into_tensor_arg(),
        dz.clone().into_tensor_arg(),
        ty,
    );
    gradient_kernel::launch(
        cl,
        count(cl, b * t * h * K),
        dim,
        q.clone().into_tensor_arg(),
        k.into_tensor_arg(),
        a.into_tensor_arg(),
        e.into_tensor_arg(),
        go.into_tensor_arg(),
        start.clone().into_tensor_arg(),
        end.clone().into_tensor_arg(),
        d.into_tensor_arg(),
        saved.into_tensor_arg(),
        adj.into_tensor_arg(),
        dz.clone().into_tensor_arg(),
        dq.clone().into_tensor_arg(),
        dk.clone().into_tensor_arg(),
        da.clone().into_tensor_arg(),
        de.clone().into_tensor_arg(),
        ty,
    );
    value_gradient_kernel::launch(
        cl,
        count(cl, b * t * h * V),
        dim,
        v.into_tensor_arg(),
        w.into_tensor_arg(),
        start.into_tensor_arg(),
        end.into_tensor_arg(),
        dz.into_tensor_arg(),
        dv.clone().into_tensor_arg(),
        dw.clone().into_tensor_arg(),
        ty,
    );
    (dq, dk, dv, da, de, dw)
}
