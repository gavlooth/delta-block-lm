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

struct ExactFixture {
    fields: Vec<Vec<f32>>,
    initial: Vec<f32>,
    upstream: Vec<f32>,
    valid: Vec<bool>,
    reset: Vec<bool>,
    expected_key: Vec<f64>,
    expected_rhs_key: Vec<f64>,
    expected_write: Vec<f64>,
}
fn exact_fixture() -> ExactFixture {
    use delta_mixers::{MixerKind, reference::{self, Token}};
    let widths = [K,K,64,K,1,1];
    let mut fields: Vec<Vec<f32>> = widths.iter().enumerate().map(|(field,&width)| {
        (0..B*T*H*width).map(|i| {
            let signal = ((i*13+field*19+1) as f64 * 0.037).sin() as f32;
            if field == 3 {0.91+0.03*signal} else if field >= 4 {0.45+0.1*signal} else {0.1*signal}
        }).collect()
    }).collect();
    let valid: Vec<bool> = (0..B*T).map(|i| i%T < T-(i/T+1)).collect();
    let mut reset: Vec<bool> = (0..B*T).map(|i| i%T==0 || i%T==5).collect();
    reset[0] = false;
    for row in 0..B { for p in 0..T { for h in 0..H {
        let scalar = (row*T+p)*H+h;
        for field in [1,2] {
            let width = widths[field];
            let norm = fields[field][scalar*width..(scalar+1)*width].iter().map(|x|x*x).sum::<f32>().sqrt();
            for x in &mut fields[field][scalar*width..(scalar+1)*width] {*x/=norm;}
        }
        if p==1 {fields[4][scalar]=0.;fields[5][scalar]=0.;}
        else if p==3 {fields[5][scalar]=-0.2;}
        else if p==4 || p==6 {
            let field = if p==4 {1} else {2};
            let width = widths[field];
            fields[field][scalar*width..(scalar+1)*width].fill(0.);
        }
        if p==2 {fields[3][scalar*K]=0.;}
    }}}
    let initial: Vec<f32> = (0..B*H*K*64).map(|i|0.03*((i*11+7) as f32*0.017).sin()).collect();
    let upstream: Vec<f32> = (0..B*T*H*64).map(|i|((i*7+3) as f64*0.051).cos() as f32).collect();
    let mut expected_key = vec![0.;B*T*H*K];
    let mut expected_rhs_key = vec![0.;B*T*H*K];
    let mut expected_write = vec![0.;B*T*H];
    for b in 0..B {for h in 0..H {
        let tokens: Vec<Token> = (0..T).map(|t| {
            let x: Vec<Vec<f64>> = fields.iter().zip(widths).map(|(f,w)| {
                let start = ((b*T+t)*H+h)*w;
                f[start..start+w].iter().map(|&x|f64::from(x)).collect()
            }).collect();
            Token {q:x[0].clone(),k:x[1].clone(),v:x[2].clone(),alpha:x[3].clone(),erase:x[4].clone(),write:x[5].clone()}
        }).collect();
        let state: Vec<f64> = initial[(b*H+h)*K*64..(b*H+h+1)*K*64].iter().map(|&x|f64::from(x)).collect();
        let u: Vec<Vec<f64>> = (0..T).map(|t| {
            let first = ((b*T+t)*H+h)*64;
            upstream[first..first+64].iter().map(|&x|f64::from(x)).collect()
        }).collect();
        let back = reference::backward(MixerKind::KronLstm,&tokens,&valid[b*T..(b+1)*T],&reset[b*T..(b+1)*T],
            &u,&vec![0.;K*64],Some(&state),&[0.;T]);
        let mut prefix = [1.;K];
        for t in 0..T {
            expected_write[(b*T+t)*H+h] = back.tokens[t].write[0];
            for k in 0..K {
                if valid[b*T+t] {prefix[k] *= if reset[b*T+t] {0.} else {tokens[t].alpha[k]};}
                let i = ((b*T+t)*H+h)*K+k;
                expected_key[i] = back.tokens[t].k[k];
                if valid[b*T+t] {
                    expected_rhs_key[i] = -tokens[t].erase[0]*back.tokens[t].write[0]*prefix[k]*
                        (0..64).map(|v|state[k*64+v]*tokens[t].v[v]).sum::<f64>();
                }
            }
        }
    }}
    ExactFixture {fields,initial,upstream,valid,reset,expected_key,expected_rhs_key,expected_write}
}
fn fixture_inputs(f: &ExactFixture, device: &Device) -> (Vec<Tensor<4>>,Tensor<4>,Tensor<2>,Tensor<2>) {
    let x = f.fields.iter().zip([K,K,64,K,1,1]).map(|(data,width)|
        tensor(data.clone(),[B,T,H,width],device).require_grad()).collect();
    (x,tensor(f.initial.clone(),[B,H,K,64],device).require_grad(),
        tensor(f.valid.iter().map(|&v|if v {1.} else {0.}).collect(),[B,T],device),
        tensor(f.reset.iter().map(|&v|if v {1.} else {0.}).collect(),[B,T],device))
}
fn fixture_scan(x: &[Tensor<4>], initial: Tensor<4>, valid: Tensor<2>, reset: Tensor<2>) -> delta_mixers::chunk::ScanOutput {
    delta_mixers::chunk::scan(delta_mixers::MixerKind::KronLstm,delta_mixers::chunk::ScanInput {
        q:x[0].clone(),k:x[1].clone(),v:x[2].clone(),alpha:x[3].clone(),erase:x[4].clone(),write:x[5].clone()
    },valid,reset,16,Some(initial))
}
fn exact_chain(device: &Device, cuts_only: bool) {
    let f = exact_fixture();
    let focus = K;
    println!("{}",json!({"probe":"independent_f64_decomposition","key_index":focus,
        "full_key_vjp":f.expected_key[focus],"rhs_key_vjp":f.expected_rhs_key[focus],
        "other_key_vjp":f.expected_key[focus]-f.expected_rhs_key[focus]}));
    if !cuts_only {
    // This uses the actual production graph; unlike the saved acceptance, it records the
    // focused gradient and forward finite difference instead of stopping at the first mismatch.
    let (x,initial,valid,reset) = fixture_inputs(&f,device);
    let out = fixture_scan(&x,initial,valid,reset);
    let _ = (values(out.output.clone()),values(out.final_state),values(out.scores));
    let grads = (out.output*tensor(f.upstream.clone(),[B,T,H,64],device)).sum().backward();
    let actual = values(x[1].grad(&grads).unwrap());
    println!("{}",json!({"probe":"production_graph_focused_key_vjp","actual":actual[focus],"expected":f.expected_key[focus]}));
    report("production_graph/all_key_vjp",actual,&f.expected_key);
    for epsilon in [0.001f32,0.003] {
        let mut objectives = Vec::new();
        let mut changed = Vec::new();
        for sign in [-1.,1.] {
            let (mut x,initial,valid,reset) = fixture_inputs(&f,device);
            let mut keys = f.fields[1].clone();
            keys[focus] += sign*epsilon;
            changed.push(f64::from(keys[focus]));
            x[1] = tensor(keys,[B,T,H,K],device);
            let output = values(fixture_scan(&x,initial,valid,reset).output);
            objectives.push(output.iter().zip(&f.upstream).map(|(&y,&u)|f64::from(y)*f64::from(u)).sum::<f64>());
        }
        println!("{}",json!({"probe":"production_graph_forward_finite_difference","epsilon":epsilon,
            "finite_difference":(objectives[1]-objectives[0])/(changed[1]-changed[0]),"expected":f.expected_key[focus]}));
    }
    }
    let modes: &[&str] = if cuts_only {
        &["masked_key","key_prefix","state_projection","inner","erased","rhs"]
    } else { &["full","rhs_only","without_rhs"] };
    for &mode in modes {
        let (x,initial,valid,reset) = fixture_inputs(&f,device);
        let mut retained = Vec::<(&str,Tensor<4>)>::new();
        let mut keep = |name,t:Tensor<4>| {
            if cuts_only && mode==name {let t=t.detach().require_grad();retained.push((name,t.clone()));t} else {t}
        };
        let active = valid.reshape([B,1,T,1]);
        let q=x[0].clone().swap_dims(1,2)*active.clone();
        let k=keep("masked_key",x[1].clone().swap_dims(1,2)*active.clone());
        let v=x[2].clone().swap_dims(1,2)*active.clone();
        let erase=x[4].clone().swap_dims(1,2)*active.clone();
        let write=x[5].clone().swap_dims(1,2)*active.clone();
        let alpha=x[3].clone().swap_dims(1,2)*(1.0f32-reset.reshape([B,1,T,1]))*active.clone()+(1.0f32-active);
        let prefix=keep("prefix",alpha.clone().cumprod(2));
        let mut product=Tensor::ones([B,H,T,K],device);
        let mut rows=Vec::new();
        for r in 0..T {
            product=(product*alpha.clone().slice([0..B,0..H,r..r+1,0..K]))
                .slice_assign([0..B,0..H,r..r+1,0..K],Tensor::ones([B,H,1,K],device));
            rows.push(product.clone().unsqueeze_dim::<5>(2));
        }
        let decay=Tensor::cat(rows,2);
        let rhs_k=if mode=="without_rhs" {k.clone().detach()} else {k.clone()};
        let other_k=if mode=="rhs_only" || cuts_only {k.clone().detach()} else {k.clone()};
        let factors=decay.clone()*other_k.clone().unsqueeze_dim::<5>(2);
        let overlap=|left:Tensor<4>| (left.unsqueeze_dim::<5>(3)*factors.clone()).sum_dim(4).reshape([B,H,T,T]);
        let lower=overlap(other_k.clone())*v.clone().matmul(v.clone().swap_dims(2,3))*erase.clone();
        let kprefix=keep("key_prefix",rhs_k*prefix.clone());
        let projection=keep("state_projection",kprefix.matmul(initial.clone()));
        let inner=keep("inner", (projection*v.clone()).sum_dim(3));
        let erased=keep("erased", inner*erase);
        let rhs=keep("rhs",write-erased);
        let mut power=lower.tril(-1).neg();
        let mut inverse=Tensor::<2>::eye(T,device).reshape([1,1,T,T]);
        let mut covered=1;
        while covered<T {
            inverse=inverse.clone()+power.clone().matmul(inverse);
            covered*=2;
            if covered<T {power=power.clone().matmul(power);}
        }
        let correction=keep("correction",inverse.matmul(rhs));
        let edits=correction*v;
        let out=(q.clone()*prefix.clone()).matmul(initial.clone())+overlap(q).tril(0).matmul(edits.clone());
        let squared=other_k.clone().square().sum_dim(3)*edits.clone().square().sum_dim(3);
        let zero=squared.clone().equal_elem(0.);
        let scores=squared.mask_fill(zero.clone(),1.).sqrt().mask_fill(zero,0.);
        let tail=decay.slice([0..B,0..H,T-1..T,0..T,0..K]).reshape([B,H,T,K]);
        let last=prefix.slice([0..B,0..H,T-1..T,0..K]).reshape([B,H,K,1]);
        let final_state=last*initial+(tail*other_k).swap_dims(2,3).matmul(edits);
        let _ = (values(out.clone()),values(final_state),values(scores));
        let grads=(out.swap_dims(1,2)*tensor(f.upstream.clone(),[B,T,H,64],device)).sum().backward();
        if !cuts_only {
        let actual=values(x[1].grad(&grads).unwrap());
        let expected:Vec<f64>=f.expected_key.iter().zip(&f.expected_rhs_key).map(|(&all,&rhs)|
            if mode=="rhs_only" {rhs} else if mode=="without_rhs" {all-rhs} else {all}).collect();
        println!("{}",json!({"probe":format!("graph_{mode}/focused_key_vjp"),"actual":actual[focus],"expected":expected[focus]}));
        report(&format!("graph_{mode}/all_key_vjp"),actual,&expected);
        }
        for (name,t) in retained {
            let shape=t.dims();
            let g=t.grad(&grads).map(values).unwrap_or_default();
            let offset=T*shape[3];
            let seed = f.expected_write[1];
            let beta = f64::from(f.fields[4][1]);
            let expected:Vec<f64> = match name {
                "masked_key" => f.expected_rhs_key[K..2*K].to_vec(),
                "key_prefix" => (0..K).map(|k| -beta*seed*(0..64).map(|v|
                    f64::from(f.initial[(K+k)*64+v])*f64::from(f.fields[2][64+v])).sum::<f64>()).collect(),
                "state_projection" => (0..64).map(|v| -beta*seed*f64::from(f.fields[2][64+v])).collect(),
                "inner" => vec![-beta*seed],
                "erased" => vec![-seed],
                "rhs" => vec![seed],
                _ => unreachable!(),
            };
            println!("{}",json!({"probe":format!("cut/{name}"),"shape":shape,
                "b0_t0_h1_vjp":g.get(offset..offset+shape[3]),"expected_b0_t0_h1":expected,
                "vjp_max_abs":g.iter().map(|x|x.abs()).fold(0.,f32::max)}));
            report(&format!("cut/{name}/focused_row"),g[offset..offset+shape[3]].to_vec(),&expected);
        }
    }
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
    if let Some(mode @ ("chain" | "cut")) = std::env::args().nth(2).as_deref() {
        exact_chain(&device, mode=="cut");
        return;
    }
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
