//! Exact causal n-gram lookup with on-demand counterfactual VJPs.
//! Table-gradient ownership is exclusive per (route,column), with chronological accumulation.
use burn::backend::{TensorMetadata, ops::{FloatTensorOps, IntTensorOps}};
use burn::tensor::DType;
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use cubecl::{frontend::Tensor as View, prelude::*};

#[cube]
fn index3(x: &View<f32>, b: usize, t: usize, c: usize) -> usize {
    b * x.stride(0) + t * x.stride(1) + c * x.stride(2)
}
#[cube]
fn complete(valid: &View<f32>, docs: &View<f32>, b: usize, t: usize, order: usize) -> bool {
    let mut active = t + 1 >= order;
    if active {
        let document = docs[b * docs.stride(0) + t * docs.stride(1)];
        for lag in 0..order {
            let source = t - lag;
            if valid[b * valid.stride(0) + source * valid.stride(1)] <= 0.0
                || docs[b * docs.stride(0) + source * docs.stride(1)] != document
            {
                active = false;
            }
        }
    }
    active
}
#[cube]
fn symbol(z: &View<f32>, b: usize, t: usize, route: usize, bits: usize) -> usize {
    let mut code = 0usize;
    for bit in 0..bits {
        if z[index3(z, b, t, route * bits + bit)] > 0.0 {
            code += 1usize << bit;
        }
    }
    code
}
#[cube]
fn address(
    codes: &View<f32>,
    b: usize,
    t: usize,
    route: usize,
    bits: usize,
    order: usize,
) -> usize {
    let mut local = 0usize;
    for slot in 0..order {
        local += (codes[index3(codes, b, t + 1 - order + slot, route)] as usize) << (bits * slot);
    }
    (route << (bits * order)) + local
}
#[cube]
fn probability(x: f32, temperature: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-temperature * x))
}

#[cube(launch)]
fn codes_kernel(z: &View<f32>, codes: &mut View<f32>, bits: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < codes.len() {
        let r = i % codes.shape(2);
        let t = i / codes.shape(2) % codes.shape(1);
        let b = i / (codes.shape(2) * codes.shape(1));
        codes[i] = symbol(z, b, t, r, bits) as f32;
    }
}

#[cube(launch)]
fn forward_kernel(
    codes: &View<f32>,
    table: &View<f32>,
    valid: &View<f32>,
    docs: &View<f32>,
    y: &mut View<f32>,
    bits: usize,
    order: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i >= y.len() {
        terminate!();
    }
    let dm = table.shape(1);
    let routes = codes.shape(2);
    let c = i % dm;
    let r = i / dm % routes;
    let t = i / (dm * routes) % codes.shape(1);
    let b = i / (dm * routes * codes.shape(1));
    let mut value = 0.0f32;
    if complete(valid, docs, b, t, order) {
        value = table[address(codes, b, t, r, bits, order) * table.stride(0) + c * table.stride(1)];
    }
    y[i] = value;
}

#[cube(launch)]
fn logits_backward_kernel(
    z: &View<f32>,
    codes: &View<f32>,
    table: &View<f32>,
    valid: &View<f32>,
    docs: &View<f32>,
    dy: &View<f32>,
    dz: &mut View<f32>,
    bits: usize,
    order: usize,
    mode: usize,
    temperature: f32,
    scale: f32,
) {
    let i = ABSOLUTE_POS as usize;
    if i >= dz.len() {
        terminate!();
    }
    let bit = i % bits;
    let route = i / bits % (z.shape(2) / bits);
    let s = i / z.shape(2) % z.shape(1);
    let b = i / (z.shape(2) * z.shape(1));
    let dm = table.shape(1);
    let pj = probability(z[index3(z, b, s, route * bits + bit)], temperature);
    let current_symbol = codes[index3(codes, b, s, route)] as usize;
    let mut total = 0.0f32;
    for lag in 0..order {
        let t = s + lag;
        if t < z.shape(1) {
            if complete(valid, docs, b, t, order) {
                let slot = order - 1 - lag;
                let row = address(codes, b, t, route, bits, order);
                if mode == 0 {
                    let bit_set = (current_symbol & (1usize << bit)) != 0;
                    let flip = 1usize << (bits * slot + bit);
                    let other = if bit_set { row - flip } else { row + flip };
                    let sign = if bit_set { 1.0f32 } else { -1.0f32 };
                    let mut dot = 0.0f32;
                    for c in 0..dm {
                        let difference = (table[row * table.stride(0) + c * table.stride(1)]
                            - table[other * table.stride(0) + c * table.stride(1)])
                            * sign;
                        dot += dy[index3(dy, b, t, route * dm + c)] * difference;
                    }
                    total += temperature * pj * (1.0 - pj) * dot;
                } else {
                    let base = row - (current_symbol << (bits * slot));
                    for code in 0..(1usize << bits) {
                        let candidate = base + (code << (bits * slot));
                        let mut prob = 1.0f32;
                        for j in 0..bits {
                            let p = probability(z[index3(z, b, s, route * bits + j)], temperature);
                            prob *= if (code & (1usize << j)) != 0 {
                                p
                            } else {
                                1.0 - p
                            };
                        }
                        let mut dot = 0.0f32;
                        for c in 0..dm {
                            dot += dy[index3(dy, b, t, route * dm + c)]
                                * table[candidate * table.stride(0) + c * table.stride(1)];
                        }
                        let beta = if (code & (1usize << bit)) != 0 {
                            1.0f32
                        } else {
                            0.0f32
                        };
                        total += temperature * prob * (beta - pj) * dot;
                    }
                }
            }
        }
    }
    dz[i] = scale * total;
}

#[cube(launch)]
fn table_backward_kernel(
    codes: &View<f32>,
    valid: &View<f32>,
    docs: &View<f32>,
    dy: &View<f32>,
    dt: &mut View<f32>,
    bits: usize,
    order: usize,
) {
    let owner = ABSOLUTE_POS as usize;
    let dm = dt.shape(1);
    let routes = codes.shape(2);
    if owner >= routes * dm {
        terminate!();
    }
    let route = owner / dm;
    let c = owner % dm;
    // Every address in this route/column is exclusively owned by this invocation.
    // This sums repeated hits in (batch,time) order, without floating-point atomics.
    for b in 0..codes.shape(0) {
        for t in 0..codes.shape(1) {
            if complete(valid, docs, b, t, order) {
                let row = address(codes, b, t, route, bits, order);
                let target = row * dm + c;
                dt[target] = dt[target] + dy[index3(dy, b, t, route * dm + c)];
            }
        }
    }
}

#[cube(launch)]
fn lookup_backward_kernel(
    indices: &View<i32>,
    dy: &View<f32>,
    dt: &mut View<f32>,
) {
    let owner = ABSOLUTE_POS as usize;
    let routes = dt.shape(0);
    let rows = dt.shape(1);
    let mem = dt.shape(2);
    if owner < routes * mem {
        let route = owner / mem;
        let column = owner % mem;
        let dt_s0 = dt.stride(0);
        let dt_s1 = dt.stride(1);
        let dt_s2 = dt.stride(2);
        let index_s0 = indices.stride(0);
        let index_s1 = indices.stride(1);
        let dy_s0 = dy.stride(0);
        let dy_s1 = dy.stride(1);
        let dy_s2 = dy.stride(2);
        for row in 0..rows {
            dt[route * dt_s0 + row * dt_s1 + column * dt_s2] = 0.0;
        }
        // Each (route,column) owns every corresponding row cell, so repeated
        // IDs accumulate serially in ascending position order without atomics.
        for position in 0..indices.shape(1) {
            let row = indices[route * index_s0 + position * index_s1] as usize;
            let target = route * dt_s0 + row * dt_s1 + column * dt_s2;
            let source = route * dy_s0 + position * dy_s1 + column * dy_s2;
            dt[target] = dt[target] + dy[source];
        }
    }
}
pub(crate) fn table_lookup(table: CubeTensor, indices: CubeTensor) -> CubeTensor {
    let [routes, rows, mem] = table.shape().dims::<3>();
    let [index_routes, positions] = indices.shape().dims::<2>();
    assert!(routes > 0 && rows > 0 && mem > 0 && positions > 0);
    assert_eq!(routes, index_routes);
    validate_lookup_views(&table, &indices);
    let indices = CubeBackend::int_expand(
        CubeBackend::int_reshape(indices, [routes, positions, 1].into()),
        [routes, positions, mem].into(),
    );
    CubeBackend::float_gather(1, table, indices)
}

pub(crate) fn table_lookup_backward(
    table: CubeTensor,
    indices: CubeTensor,
    dy: CubeTensor,
) -> CubeTensor {
    let [routes, _, mem] = table.shape().dims::<3>();
    let [index_routes, positions] = indices.shape().dims::<2>();
    assert_eq!([routes, positions, mem], dy.shape().dims::<3>());
    assert_eq!(routes, index_routes);
    validate_lookup_views(&table, &indices);
    validate_lookup_views(&dy, &indices);
    let dt = CubeBackend::float_empty(table.shape(), &table.device, table.dtype.into());
    lookup_backward_kernel::launch(
        &table.client,
        count(&table, routes * mem),
        CubeDim::new_1d(64),
        indices.into_tensor_arg(),
        dy.into_tensor_arg(),
        dt.clone().into_tensor_arg(),
    );
    dt
}

fn validate_lookup_views(table: &CubeTensor, indices: &CubeTensor) {
    assert_eq!(table.dtype, DType::F32, "Lngram table lookup requires f32");
    assert_eq!(indices.dtype, DType::I32, "Lngram lookup indices require i32");
    assert_eq!(table.device, indices.device);
    for (x, element_size) in [(table, 4u64), (indices, 4u64)] {
        let mut span = 1usize;
        for (&dim, &stride) in x
            .meta
            .shape()
            .as_slice()
            .iter()
            .zip(x.meta.strides().iter())
        {
            span = (dim - 1)
                .checked_mul(stride)
                .and_then(|n| span.checked_add(n))
                .expect("Lngram table lookup address overflow");
        }
        assert!(span <= u32::MAX as usize && x.handle.size() / element_size <= u32::MAX as u64);
    }
}
fn count(x: &CubeTensor, elements: usize) -> CubeCount {
    let groups = elements.div_ceil(64);
    let (mx, my, _) = x.client.properties().hardware.max_cube_count;
    let gx = groups.min(mx.min(65535) as usize).max(1);
    assert!(groups.div_ceil(gx) <= my as usize);
    CubeCount::Static(gx as u32, groups.div_ceil(gx) as u32, 1)
}
fn validate(
    z: &CubeTensor,
    table: &CubeTensor,
    valid: &CubeTensor,
    docs: &CubeTensor,
    bits: usize,
    order: usize,
) {
    let [b, t, width] = z.shape().dims::<3>();
    assert!(b > 0 && t > 0 && (1..=16).contains(&bits) && order > 0 && width % bits == 0);
    assert!(bits.checked_mul(order).is_some_and(|n| n < 31));
    let [rows, dm] = table.shape().dims::<2>();
    assert!(dm > 0);
    assert_eq!(rows, (width / bits) * (1usize << (bits * order)));
    assert_eq!(valid.shape().dims::<2>(), [b, t]);
    assert_eq!(docs.shape().dims::<2>(), [b, t]);
    validate_views(z, &[z, table, valid, docs]);
}
fn validate_views(z: &CubeTensor, views: &[&CubeTensor]) {
    for x in views {
        assert_eq!(x.dtype, DType::F32);
        assert_eq!(x.device, z.device);
        let mut span = 1usize;
        for (&d, &s) in x
            .meta
            .shape()
            .as_slice()
            .iter()
            .zip(x.meta.strides().iter())
        {
            span = (d - 1)
                .checked_mul(s)
                .and_then(|n| span.checked_add(n))
                .expect("Lngram address overflow");
        }
        assert!(span <= u32::MAX as usize && x.handle.size() / 4 <= u32::MAX as u64);
    }
}
pub(crate) fn codes(z: CubeTensor, bits: usize) -> CubeTensor {
    let [b, t, width] = z.shape().dims::<3>();
    assert!(b > 0 && t > 0 && (1..=16).contains(&bits) && width % bits == 0);
    validate_views(&z, &[&z]);
    // Symbols fit exactly in FP32 for all supported bit widths.
    let codes = CubeBackend::float_empty([b, t, width / bits].into(), &z.device, DType::F32.into());
    codes_kernel::launch(
        &z.client,
        count(&z, codes.shape().num_elements()),
        CubeDim::new_1d(64),
        z.clone().into_tensor_arg(),
        codes.clone().into_tensor_arg(),
        bits,
    );
    codes
}
pub(crate) fn forward(
    z: CubeTensor,
    codes: CubeTensor,
    table: CubeTensor,
    valid: CubeTensor,
    docs: CubeTensor,
    bits: usize,
    order: usize,
) -> CubeTensor {
    validate(&z, &table, &valid, &docs, bits, order);
    let [b, t, width] = z.shape().dims::<3>();
    assert_eq!(codes.shape().dims::<3>(), [b, t, width / bits]);
    validate_views(&z, &[&codes]);
    let y = CubeBackend::float_empty(
        [b, t, width / bits * table.shape().dims::<2>()[1]].into(),
        &z.device,
        DType::F32.into(),
    );
    forward_kernel::launch(
        &z.client,
        count(&z, y.shape().num_elements()),
        CubeDim::new_1d(64),
        codes.clone().into_tensor_arg(),
        table.into_tensor_arg(),
        valid.into_tensor_arg(),
        docs.into_tensor_arg(),
        y.clone().into_tensor_arg(),
        bits,
        order,
    );
    y
}
pub(crate) fn backward(
    z: CubeTensor,
    table: CubeTensor,
    valid: CubeTensor,
    docs: CubeTensor,
    codes: CubeTensor,
    dy: CubeTensor,
    bits: usize,
    order: usize,
    mode: usize,
    temperature: f32,
    scale: f32,
) -> (CubeTensor, CubeTensor) {
    validate(&z, &table, &valid, &docs, bits, order);
    let [b, t, width] = z.shape().dims::<3>();
    assert_eq!(codes.shape().dims::<3>(), [b, t, width / bits]);
    assert_eq!(codes.dtype, DType::F32);
    let dz = CubeBackend::float_empty(z.shape(), &z.device, DType::F32.into());
    let dt = CubeBackend::float_zeros(table.shape(), &z.device, DType::F32.into());
    logits_backward_kernel::launch(
        &z.client,
        count(&z, dz.shape().num_elements()),
        CubeDim::new_1d(64),
        z.clone().into_tensor_arg(),
        codes.clone().into_tensor_arg(),
        table.clone().into_tensor_arg(),
        valid.clone().into_tensor_arg(),
        docs.clone().into_tensor_arg(),
        dy.clone().into_tensor_arg(),
        dz.clone().into_tensor_arg(),
        bits,
        order,
        mode,
        temperature,
        scale,
    );
    table_backward_kernel::launch(
        &z.client,
        count(&z, width / bits * table.shape().dims::<2>()[1]),
        CubeDim::new_1d(64),
        codes.into_tensor_arg(),
        valid.into_tensor_arg(),
        docs.into_tensor_arg(),
        dy.into_tensor_arg(),
        dt.clone().into_tensor_arg(),
        bits,
        order,
    );
    (dz, dt)
}
