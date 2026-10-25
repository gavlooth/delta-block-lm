//! Temporary noisy HOLA read and deterministic analytic VJP, K32/V64, f32.
use burn::backend::{TensorMetadata, ops::FloatTensorOps};
use burn::{prelude::Shape, tensor::DType};
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use cubecl::{frontend::Tensor as View, prelude::*};

#[cube]
fn ix(x: &View<f32>, b: usize, h: usize, t: usize, c: usize) -> usize {
    b*x.stride(0)+h*x.stride(1)+t*x.stride(2)+c*x.stride(3)
}
#[cube(launch)]
fn read(q: &View<f32>, k: &View<f32>, v: &View<f32>, visible: &View<f32>, valid: &View<f32>, sink: &View<f32>, y: &mut View<f32>, p: &mut View<f32>, #[comptime] lanes: usize) {
    let lane = UNIT_POS as usize;
    let row = CUBE_POS as usize;
    let n = q.shape(2);
    let h = row/n%q.shape(1);
    let b = row/(n*q.shape(1));
    let t = row%n;
    let m = k.shape(2);
    let active = valid[b*valid.stride(0)+t*valid.stride(1)] > 0.5f32;
    let mut weights = Shared::<[f32]>::new_slice(lanes);
    let mut sums = Shared::<[f32]>::new_slice(lanes);
    let mut score = 0.0f32;
    score -= 1.0e30f32;
    if lane == m { score = sink[h*sink.stride(0)]; }
    if lane < m && active && visible[ix(visible,b,h,t,lane)] > 0.5f32 {
        score = 0.0f32;
        for c in 0..32 { score += q[ix(q,b,h,t,c)]*k[ix(k,b,h,lane,c)]; }
        score *= 0.1767766952966369f32;
    }
    weights[lane] = score;
    sums[lane] = score;
    sync_cube();
    let mut offset = 32usize;
    if lanes == 128 { offset = 64usize; }
    while offset > 0 {
        if lane < offset { sums[lane] = f32::max(sums[lane],sums[lane+offset]); }
        sync_cube();
        offset /= 2;
    }
    let maximum = sums[0];
    sync_cube();
    let mut weight = 0.0f32;
    if lane <= m { weight = f32::exp(weights[lane]-maximum); }
    weights[lane] = weight;
    sums[lane] = weight;
    sync_cube();
    offset = 32usize;
    if lanes == 128 { offset = 64usize; }
    while offset > 0 {
        if lane < offset { sums[lane] += sums[lane+offset]; }
        sync_cube();
        offset /= 2;
    }
    let denominator = sums[0];
    if lane <= m {
        weights[lane] /= denominator;
        p[row*(m+1)+lane] = weights[lane];
    }
    sync_cube();
    if lane < 64 {
        let mut value = 0.0f32;
        if active {
            for s in 0..m { value += weights[s]*v[ix(v,b,h,s,lane)]; }
        }
        y[row*64+lane] = value;
    }
}
#[cube(launch)]
fn query_vjp(q: &View<f32>, k: &View<f32>, v: &View<f32>, valid: &View<f32>, p: &View<f32>, y: &View<f32>, dy: &View<f32>, dq: &mut View<f32>, dp: &mut View<f32>, ds: &mut View<f32>, #[comptime] lanes: usize) {
    let lane = UNIT_POS as usize;
    let row = CUBE_POS as usize;
    let n = q.shape(2);
    let h = row/n%q.shape(1);
    let b = row/(n*q.shape(1));
    let t = row%n;
    let m = k.shape(2);
    let active = valid[b*valid.stride(0)+t*valid.stride(1)] > 0.5f32;
    let mut adj = Shared::<[f32]>::new_slice(lanes);
    let mut dot = 0.0f32;
    if active { for c in 0..64 { dot += dy[ix(dy,b,h,t,c)]*y[ix(y,b,h,t,c)]; } }
    let mut value = 0.0f32;
    if lane < m && active {
        let mut term = 0.0f32;
        for c in 0..64 { term += dy[ix(dy,b,h,t,c)]*v[ix(v,b,h,lane,c)]; }
        value = p[ix(p,b,h,t,lane)]*(term-dot);
    }
    adj[lane] = value;
    if lane < m { dp[row*m+lane] = value; }
    if lane == 0 { ds[row] = -p[ix(p,b,h,t,m)]*dot; }
    sync_cube();
    if lane < 32 {
        let mut value = 0.0f32;
        for s in 0..m { value += adj[s]*k[ix(k,b,h,s,lane)]; }
        dq[row*32+lane] = value*0.1767766952966369f32;
    }
}
#[cube(launch)]
fn key_vjp(q: &View<f32>, k: &View<f32>, valid: &View<f32>, p: &View<f32>, dp: &View<f32>, dy: &View<f32>, dk: &mut View<f32>, dv: &mut View<f32>) {
    let lane = UNIT_POS as usize;
    let row = CUBE_POS as usize;
    let m = k.shape(2);
    let s = row%m;
    let h = row/m%q.shape(1);
    let b = row/(m*q.shape(1));
    let mut key = 0.0f32;
    let mut value = 0.0f32;
    for t in 0..q.shape(2) {
        if valid[b*valid.stride(0)+t*valid.stride(1)] > 0.5f32 {
            if lane < 32 { key += dp[ix(dp,b,h,t,s)]*q[ix(q,b,h,t,lane)]; }
            value += p[ix(p,b,h,t,s)]*dy[ix(dy,b,h,t,lane)];
        }
    }
    if lane < 32 { dk[row*32+lane] = key*0.1767766952966369f32; }
    dv[row*64+lane] = value;
}
#[cube(launch)]
fn sink_vjp(partial: &View<f32>, sink: &mut View<f32>) {
    let h = ABSOLUTE_POS as usize;
    if h < sink.shape(0) {
        let mut value = 0.0f32;
        for b in 0..partial.shape(0) { for t in 0..partial.shape(2) { value += partial[b*partial.stride(0)+h*partial.stride(1)+t*partial.stride(2)]; } }
        sink[h] = value;
    }
}
fn empty(x: &CubeTensor, shape: impl Into<Shape>) -> CubeTensor { CubeBackend::float_empty(shape.into(),&x.device,DType::F32.into()) }
fn count(n: usize) -> CubeCount { assert!(n <= 65535); CubeCount::Static(n as u32,1,1) }
fn validate(q: &CubeTensor, k: &CubeTensor, v: &CubeTensor, visible: &CubeTensor, valid: &CubeTensor, sink: &CubeTensor) -> (usize,usize,usize,usize) {
    let [b,h,n,d] = q.shape().dims::<4>();
    let m = k.shape().dims::<4>()[2];
    assert!(b>0 && h>0 && n>0 && n<=16 && m>0 && m<=127,"fused noisy HOLA requires 1..=16 queries and 1..=127 keys");
    assert_eq!(d,32,"fused noisy HOLA requires key32/value64");
    assert_eq!(k.shape().dims::<4>(),[b,h,m,32]);
    assert_eq!(v.shape().dims::<4>(),[b,h,m,64]);
    assert_eq!(visible.shape().dims::<4>(),[b,h,n,m]);
    assert_eq!(valid.shape().dims::<2>(),[b,n]);
    assert_eq!(sink.shape().dims::<1>(),[h]);
    for x in [q,k,v,visible,valid,sink] { assert_eq!(x.dtype,DType::F32); assert_eq!(x.device,q.device); }
    (b,h,n,m)
}
pub(crate) fn forward(q: CubeTensor,k: CubeTensor,v: CubeTensor,visible: CubeTensor,valid: CubeTensor,sink: CubeTensor) -> (CubeTensor,CubeTensor) {
    let (b,h,n,m) = validate(&q,&k,&v,&visible,&valid,&sink);
    let y = empty(&q,[b,h,n,64]);
    let p = empty(&q,[b,h,n,m+1]);
    let lanes = (m+1).next_power_of_two().max(64);
    read::launch(&q.client,count(b*h*n),CubeDim::new_1d(lanes as u32),q.clone().into_tensor_arg(),k.into_tensor_arg(),v.into_tensor_arg(),visible.into_tensor_arg(),valid.into_tensor_arg(),sink.into_tensor_arg(),y.clone().into_tensor_arg(),p.clone().into_tensor_arg(),lanes);
    (y,p)
}
pub(crate) fn backward(q: CubeTensor,k: CubeTensor,v: CubeTensor,valid: CubeTensor,p: CubeTensor,y: CubeTensor,dy: CubeTensor) -> (CubeTensor,CubeTensor,CubeTensor,CubeTensor) {
    let [b,h,n,_] = q.shape().dims::<4>();
    let m = k.shape().dims::<4>()[2];
    let dq = empty(&q,[b,h,n,32]);
    let dk = empty(&q,[b,h,m,32]);
    let dv = empty(&q,[b,h,m,64]);
    let ds = empty(&q,[h]);
    let partial = empty(&q,[b,h,n]);
    let dp = empty(&q,[b,h,n,m]);
    let lanes = (m+1).next_power_of_two().max(64);
    query_vjp::launch(&q.client,count(b*h*n),CubeDim::new_1d(lanes as u32),q.clone().into_tensor_arg(),k.clone().into_tensor_arg(),v.into_tensor_arg(),valid.clone().into_tensor_arg(),p.clone().into_tensor_arg(),y.into_tensor_arg(),dy.clone().into_tensor_arg(),dq.clone().into_tensor_arg(),dp.clone().into_tensor_arg(),partial.clone().into_tensor_arg(),lanes);
    key_vjp::launch(&q.client,count(b*h*m),CubeDim::new_1d(64),q.clone().into_tensor_arg(),k.into_tensor_arg(),valid.into_tensor_arg(),p.into_tensor_arg(),dp.into_tensor_arg(),dy.into_tensor_arg(),dk.clone().into_tensor_arg(),dv.clone().into_tensor_arg());
    sink_vjp::launch(&q.client,CubeCount::Static(h.div_ceil(64) as u32,1,1),CubeDim::new_1d(64),partial.into_tensor_arg(),ds.clone().into_tensor_arg());
    (dq,dk,dv,ds)
}
