//! Temporary primitive localization for the retained T7 tensor-solver failure.
use burn::prelude::*;
use serde_json::json;

const B: usize = 2;
const H: usize = 4;
const T: usize = 7;
const K: usize = 32;
const ATOL: f64 = 2e-5;
const RTOL: f64 = 3e-4;

fn data(n: usize, phase: usize) -> Vec<f32> {
    (0..n).map(|i| 0.1 * (((i * 13 + phase * 19 + 1) as f64 * 0.037).sin() as f32)).collect()
}
fn tensor<const D: usize>(data: Vec<f32>, shape: [usize; D], device: &Device) -> Tensor<D> {
    Tensor::from_data(TensorData::new(data, shape), device)
}
fn values<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.into_data().try_to_vec::<f32>().unwrap()
}
fn report(name: &str, actual: Vec<f32>, expected: &[f64]) -> bool {
    assert_eq!(actual.len(), expected.len());
    let bad = actual.iter().zip(expected).enumerate().filter(|(_, (a, e))| {
        !a.is_finite() || !e.is_finite() || (f64::from(**a) - **e).abs() > ATOL + RTOL * e.abs()
    }).map(|(i, (a, e))| json!({"index":i,"actual":a,"expected":e})).collect::<Vec<_>>();
    let maximum = actual.iter().zip(expected).map(|(a, e)| (f64::from(*a) - e).abs()).fold(0., f64::max);
    println!("{}", json!({"probe":name,"passed":bad.is_empty(),"maximum_absolute_error":maximum,
        "failure_count":bad.len(),"first_failures":bad.iter().take(8).collect::<Vec<_>>() }));
    bad.is_empty()
}
// Physical [batch,row,head,column] upload exposes the solver's noncontiguous head/time view.
fn upload4(data: Vec<f32>, shape: [usize; 4], strided: bool, device: &Device) -> Tensor<4> {
    if !strided { return tensor(data, shape, device); }
    let [b, h, m, n] = shape;
    let mut physical = vec![0.; data.len()];
    for bi in 0..b { for hi in 0..h { for mi in 0..m { for ni in 0..n {
        physical[((bi*m+mi)*h+hi)*n+ni] = data[((bi*h+hi)*m+mi)*n+ni];
    }}}}
    tensor(physical, [b,m,h,n], device).swap_dims(1,2)
}

fn gemm(device: &Device, m: usize, k: usize, n: usize, broadcast: bool, strided: bool) -> bool {
    let a = data(B*H*m*k, 0);
    let rhs_batches = if broadcast { 1 } else { B*H };
    let b = data(rhs_batches*k*n, 1);
    let u = data(B*H*m*n, 2);
    let mut y = vec![0.; u.len()];
    let mut da = vec![0.; a.len()];
    let mut db = vec![0.; b.len()];
    for bh in 0..B*H { for i in 0..m { for j in 0..n { for z in 0..k {
        let ai = (bh*m+i)*k+z;
        let bi = ((if broadcast {0} else {bh})*k+z)*n+j;
        let yi = (bh*m+i)*n+j;
        y[yi] += f64::from(a[ai])*f64::from(b[bi]);
        da[ai] += f64::from(u[yi])*f64::from(b[bi]);
        db[bi] += f64::from(a[ai])*f64::from(u[yi]);
    }}}}
    let a = upload4(a, [B,H,m,k], strided, device).require_grad();
    let b = upload4(b, if broadcast {[1,1,k,n]} else {[B,H,k,n]}, strided, device).require_grad();
    let out = a.clone().matmul(b.clone());
    let grads = (out.clone()*tensor(u,[B,H,m,n],device)).sum().backward();
    let name = format!("gemm_{m}x{k}x{n}_broadcast{broadcast}_strided{strided}");
    let mut ok = report(&format!("{name}/forward"), values(out), &y);
    ok &= report(&format!("{name}/lhs_vjp"), values(a.grad(&grads).unwrap()), &da);
    ok &= report(&format!("{name}/rhs_vjp"), values(b.grad(&grads).unwrap()), &db);
    ok
}

fn metadata() -> (Vec<f32>,Vec<f32>,Vec<f32>) {
    let mut alpha = vec![0.; B*H*T*K];
    for bh in 0..B*H { for t in 0..T { for k in 0..K {
        alpha[(bh*T+t)*K+k] = if t == 2 && k == 0 {0.} else {0.91 + 0.03 * (((bh*T+t)*K+k) as f32 * 0.13).sin()};
    }}}
    let mut valid = vec![1.;B*T];
    let mut reset = vec![0.;B*T];
    valid[T-1] = 0.;
    valid[2*T-2..].fill(0.);
    reset[T] = 1.;
    reset[5] = 1.;
    reset[T+5] = 1.; // reset on padding must be ignored
    (alpha,valid,reset)
}
fn effective_alpha(alpha: &[f32], valid: &[f32], reset: &[f32]) -> Vec<f64> {
    alpha.iter().enumerate().map(|(i,a)| {
        let b = i/(H*T*K);
        let t = (i/K)%T;
        if valid[b*T+t] == 0. {1.} else {f64::from(*a)*(1.-f64::from(reset[b*T+t]))}
    }).collect()
}
fn decay(device: &Device, pairwise: bool) -> bool {
    let (raw,valid,reset) = metadata();
    let effective = effective_alpha(&raw,&valid,&reset);
    let count = B*H*T*(if pairwise {T} else {1})*K;
    let seed = data(count,3);
    let mut expected = vec![0.;count];
    let mut da = vec![0.;raw.len()];
    for bh in 0..B*H { for r in 0..T { for s in 0..if pairwise {T} else {1} { for k in 0..K {
        let out_i = if pairwise {((bh*T+r)*T+s)*K+k} else {(bh*T+r)*K+k};
        // slice_assign construction has unused upper entries prod(alpha[0..=r]).
        let start = if pairwise && s <= r {s+1} else {0};
        expected[out_i] = (start..=r).map(|t|effective[(bh*T+t)*K+k]).product();
        for omitted in start..=r {
            let index = (bh*T+omitted)*K+k;
            let meta = bh/H*T+omitted;
            da[index] += f64::from(seed[out_i]) * (start..=r).filter(|&t|t!=omitted)
                .map(|t|effective[(bh*T+t)*K+k]).product::<f64>() * f64::from(valid[meta]*(1.-reset[meta]));
        }
    }}}}
    let raw = upload4(raw,[B,H,T,K],true,device).require_grad();
    let active = tensor(valid,[B,1,T,1],device);
    let alpha = raw.clone()*(1.0f32-tensor(reset,[B,1,T,1],device))*active.clone()+(1.0f32-active);
    let (actual, grads) = if pairwise {
        let mut product = Tensor::ones([B,H,T,K],device);
        let mut rows = Vec::new();
        for r in 0..T {
            product = (product*alpha.clone().slice([0..B,0..H,r..r+1,0..K]))
                .slice_assign([0..B,0..H,r..r+1,0..K],Tensor::ones([B,H,1,K],device));
            rows.push(product.clone().unsqueeze_dim::<5>(2));
        }
        let out = Tensor::cat(rows,2);
        let grads = (out.clone()*tensor(seed,[B,H,T,T,K],device)).sum().backward();
        (values(out),grads)
    } else {
        let out = alpha.cumprod(2);
        let grads = (out.clone()*tensor(seed,[B,H,T,K],device)).sum().backward();
        (values(out),grads)
    };
    let name = if pairwise {"decay_slice_assign"} else {"decay_prefix"};
    let mut ok = report(&format!("{name}/forward"),actual,&expected);
    ok &= report(&format!("{name}/alpha_vjp"),values(raw.grad(&grads).unwrap()),&da);
    ok
}

fn overlap(device: &Device, shared: bool) -> bool {
    let (raw,valid,reset) = metadata();
    let alpha = effective_alpha(&raw,&valid,&reset);
    let q = data(B*H*T*K,0);
    let k = if shared {q.clone()} else {data(B*H*T*K,1)};
    let seed = data(B*H*T*T,2);
    let mut expected = vec![0.;seed.len()];
    let mut dq = vec![0.;q.len()];
    let mut dk = vec![0.;k.len()];
    let mut decay = vec![0.; B*H*T*T*K];
    for bh in 0..B*H { for r in 0..T { for s in 0..T { for z in 0..K {
        let di = ((bh*T+r)*T+s)*K+z;
        decay[di] = ((s+1)..=r).map(|t|alpha[(bh*T+t)*K+z]).product::<f64>() as f32;
        if s>r {continue;}
        let qi = (bh*T+r)*K+z;
        let ki = (bh*T+s)*K+z;
        let yi = (bh*T+r)*T+s;
        let d = f64::from(decay[di])*f64::from(valid[bh/H*T+r]*valid[bh/H*T+s]);
        expected[yi] += f64::from(q[qi])*f64::from(k[ki])*d;
        dq[qi] += f64::from(seed[yi])*f64::from(k[ki])*d;
        dk[ki] += f64::from(seed[yi])*f64::from(q[qi])*d;
    }}}}
    let q = upload4(q,[B,H,T,K],true,device).require_grad();
    let k = if shared {q.clone()} else {upload4(k,[B,H,T,K],true,device).require_grad()};
    let mask = tensor(valid,[B,1,T,1],device);
    let left = q.clone()*mask.clone();
    let right = k.clone()*mask;
    let factors = tensor(decay,[B,H,T,T,K],device)*right.unsqueeze_dim::<5>(2);
    let out = (left.unsqueeze_dim::<5>(3)*factors).sum_dim(4).reshape([B,H,T,T]).tril(0);
    let grads = (out.clone()*tensor(seed,[B,H,T,T],device)).sum().backward();
    let name = format!("overlap_shared{shared}");
    let mut ok = report(&format!("{name}/forward"),values(out),&expected);
    if shared {
        for (a,b) in dq.iter_mut().zip(dk) {*a+=b;}
        ok &= report(&format!("{name}/shared_key_vjp"),values(q.grad(&grads).unwrap()),&dq);
    } else {
        ok &= report(&format!("{name}/query_vjp"),values(q.grad(&grads).unwrap()),&dq);
        ok &= report(&format!("{name}/key_vjp"),values(k.grad(&grads).unwrap()),&dk);
    }
    ok
}

fn main() {
    assert!(!cfg!(feature="fusion"),"diagnostic must match the non-fusion failing runtime");
    let backend = std::env::args().nth(1).expect("cpu|vulkan");
    #[cfg(feature="vulkan")]
    {
        use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
        let mut config = CubeClRuntimeConfig::default();
        config.compilation.cache = true;
        config.environment.name = "delta-block-local".into();
        config.environment.path = cubecl::config::cache::CacheConfig::Directory(".cache/cubecl".into());
        CubeClRuntimeConfig::set(config);
    }
    let device = match backend.as_str() {
        "cpu" => Device::ndarray(),
        #[cfg(feature="vulkan")]
        "vulkan" => Device::vulkan(Default::default()),
        _ => panic!("backend unavailable"),
    }.autodiff();
    println!("{}",json!({"device":format!("{device:?}"),"identity":format!("{:?}",device.identity()),"absolute_tolerance":ATOL,"relative_tolerance":RTOL}));
    let mut ok = true;
    for strided in [false,true] {
        for (m,k,n,broadcast) in [(T,K,64,false),(T,64,T,false),(T,T,T,false),(T,T,T,true),(T,T,1,false)] {
            ok &= gemm(&device,m,k,n,broadcast,strided);
        }
    }
    ok &= decay(&device,false);
    ok &= decay(&device,true);
    ok &= overlap(&device,false);
    ok &= overlap(&device,true);
    println!("{}",json!({"all_probes_passed":ok,"scope":"primitive localization, not recurrence acceptance"}));
    if !ok {std::process::exit(1);}
}
