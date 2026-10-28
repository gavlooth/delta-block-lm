//! Original-position sparse HOLA reads, including detached prefix block ranking.
use burn::{backend::{TensorMetadata, ops::FloatTensorOps}, prelude::Shape, tensor::DType};
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use cubecl::{frontend::Tensor as View, prelude::*};
const SCALE: f32 = 0.1767766952966369;
#[cube]
fn ix(x: &View<f32>, b: usize, t: usize, h: usize, c: usize) -> usize {
    b*x.stride(0)+t*x.stride(1)+h*x.stride(2)+c*x.stride(3)
}
#[cube]
fn meta(x: &View<f32>, b: usize, t: usize) -> f32 { x[b*x.stride(0)+t*x.stride(1)] }
#[cube]
fn element(clean: &View<f32>, active: &View<f32>, b: usize, s: usize, h: usize, c: usize,
    #[comptime] enabled: bool) -> f32 {
    if enabled && s>=clean.shape(1) {active[ix(active,b,s-clean.shape(1),h,c)]}
    else {clean[ix(clean,b,s,h,c)]}
}
#[cube]
fn dot(q: &View<f32>, k: &View<f32>, ak: &View<f32>, b: usize, t: usize, s: usize, h: usize,
    #[comptime] active: bool) -> f32 {
    let mut result = 0.0f32;
    for c in 0..32usize { result += q[ix(q,b,t,h,c)]*element(k,ak,b,s,h,c,active); }
    result*SCALE
}
#[cube]
fn cached(schedule: &View<f32>, b: usize, h: usize, snapshot: usize, s: usize,
    #[comptime] capacity: usize) -> bool {
    let mut present=false;
    for slot in 0..capacity {
        let p=schedule[b*schedule.stride(0)+h*schedule.stride(1)+snapshot*schedule.stride(2)+slot*schedule.stride(3)];
        if p==s as f32 {present=true;}
    }
    present
}


#[cube(launch)]
fn summary_kernel(k: &View<f32>, valid: &View<f32>, documents: &View<f32>,
    summaries: &mut View<f32>, #[comptime] block: usize, #[comptime] blocks: usize) {
    let lane = UNIT_POS as usize;
    let row = CUBE_POS as usize;
    let h = row%k.shape(2);
    let j = row/k.shape(2)%blocks;
    let b = row/(k.shape(2)*blocks);
    if b>=k.shape(0) {terminate!();}
    let mut count = 0.0f32;
    let mut sum = 0.0f32;
    let mut document = 0.0f32;
    let mut uniform = true;
    for offset in 0..block {
        let s = j*block+offset;
        if s<k.shape(1) && meta(valid,b,s)>=0.5f32 {
            let doc = meta(documents,b,s);
            if count==0.0f32 {document=doc;}
            else if doc!=document {uniform=false;}
            if lane<32usize {sum+=k[ix(k,b,s,h,lane)];}
            count+=1.0f32;
        }
    }
    if lane<32usize {
        summaries[row*35usize+lane]=if count>0.0f32 {sum/count} else {0.0f32};
    }
    if lane==0 {
        summaries[row*35usize+32usize]=count;
        summaries[row*35usize+33usize]=if uniform {1.0f32} else {0.0f32};
        summaries[row*35usize+34usize]=document;
    }
}

#[cube(launch)]
fn select_kernel(
    q: &View<f32>, k: &View<f32>, query_valid: &View<f32>, valid: &View<f32>,
    documents: &View<f32>, query_documents: &View<f32>, schedule: &View<f32>,
    summaries: &View<f32>, starts: &View<f32>, ids: &mut View<f32>,
    #[comptime] capacity: usize, #[comptime] chunk: usize,
    #[comptime] block: usize, #[comptime] top: usize, #[comptime] own: usize, #[comptime] blocks: usize,
    #[comptime] frozen: bool, #[comptime] active_size: usize,
) {
    let lane = UNIT_POS as usize;
    let row = CUBE_POS as usize;
    let heads = q.shape(2);
    let n = q.shape(1);
    let h = row%heads;
    let t = row/heads%n;
    let b = row/(heads*n);
    if b >= q.shape(0) { terminate!(); }
    let start = starts[b*starts.stride(0)] as usize;
    let absolute = start+t;
    let mut cut = absolute/chunk*chunk;
    let mut end = absolute+1;
    let mut snapshot = absolute/chunk;
    if frozen { cut = start/chunk*chunk; end = start; snapshot = 0usize; }
    let mut own_start=absolute/block*block;
    if frozen {own_start=start/block*block;}
    let doc = meta(query_documents,b,t);
    let active = meta(query_valid,b,t)>=0.5f32;
    let slots = capacity+chunk+top*block+own+active_size;
    let mut scores = Shared::<[f32]>::new_slice(blocks);
    let mut eligible = Shared::<[u32]>::new_slice(blocks);
    let mut j = lane;
    while j < blocks {
        let mut count = 0.0f32;
        let mut score = 0.0f32;
        if active && (j+1)*block <= own_start {
            let summary=((b*blocks+j)*heads+h)*35usize;
            if summaries[summary+33usize]>0.5f32 {
                if summaries[summary+34usize]==doc {
                    count=summaries[summary+32usize];
                    for c in 0..32usize {score+=q[ix(q,b,t,h,c)]*summaries[summary+c];}
                }
            } else {
                // A document transition requires the query's own valid subset.
                for c in 0..32usize {
                    let mut sum = 0.0f32;
                    let mut token_count = 0.0f32;
                    for offset in 0..block {
                        let s = j*block+offset;
                        if meta(valid,b,s)>=0.5f32 && meta(documents,b,s)==doc {
                            sum += k[ix(k,b,s,h,c)];
                            token_count += 1.0f32;
                        }
                    }
                    if token_count>0.0f32 { score += q[ix(q,b,t,h,c)]*(sum/token_count); }
                    count = token_count;
                }
            }
        }
        scores[j] = score*SCALE;
        eligible[j] = if count>0.0f32 {1u32} else {0u32};
        j += 64usize;
    }
    sync_cube();
    let mut slot = lane;
    while slot < slots {
        ids[row*slots+slot] = -1.0f32;
        slot += 64usize;
    }
    sync_cube();
    slot = lane;
    while slot < capacity+chunk {
        let mut p = 0.0f32;
        p -= 1.0f32;
        if active {
            if slot < capacity {
                p = schedule[b*schedule.stride(0)+h*schedule.stride(1)+snapshot*schedule.stride(2)+slot*schedule.stride(3)];
            } else {
                let s = cut+slot-capacity;
                if s<end {p=s as f32;}
            }
            if p>=0.0f32 {
                let s = p as usize;
                if s>=k.shape(1) || meta(valid,b,s)<0.5f32 || meta(documents,b,s)!=doc {p = -1.0f32;}
            }
        }
        ids[row*slots+slot] = p;
        slot += 64usize;
    }
    j = lane;
    while j < blocks {
        if eligible[j]>0u32 {
            let mut rank = 0usize;
            for other in 0..blocks {
                if eligible[other]>0u32 && (scores[other]>scores[j] || (scores[other]==scores[j] && other<j)) {rank+=1usize;}
            }
            if rank<top {
                for offset in 0..block {
                    let s = j*block+offset;
                    if !cached(schedule,b,h,snapshot,s,capacity) && meta(valid,b,s)>=0.5f32 && meta(documents,b,s)==doc {
                        ids[row*slots+capacity+chunk+rank*block+offset] = s as f32;
                    }
                }
            }
        }
        j += 64usize;
    }
    let mut offset=lane;
    while offset<own {
        let s=own_start+offset;
        if active && s<cut && meta(valid,b,s)>=0.5f32 && meta(documents,b,s)==doc
            && !cached(schedule,b,h,snapshot,s,capacity) {
            ids[row*slots+capacity+chunk+top*block+offset]=s as f32;
        }
        offset+=64usize;
    }
    if active_size>0 {
        let mut offset=lane;
        while offset<active_size {
            if active && meta(query_valid,b,offset)>=0.5f32 {
                ids[row*slots+capacity+chunk+top*block+own+offset]=(k.shape(1)+offset) as f32;
            }
            offset+=64usize;
        }
    }
}

#[cube(launch)]
fn read_kernel(q: &View<f32>, k: &View<f32>, v: &View<f32>, ids: &View<f32>,
    sink: &View<f32>, ak: &View<f32>, av: &View<f32>, output: &mut View<f32>,
    stats: &mut View<f32>, #[comptime] slots: usize, #[comptime] active: bool) {
    let lane = UNIT_POS as usize;
    let row = CUBE_POS as usize;
    let heads = q.shape(2);
    let n = q.shape(1);
    let h = row%heads;
    let t = row/heads%n;
    let b = row/(heads*n);
    if b>=q.shape(0) {terminate!();}
    let mut logits = Shared::<[f32]>::new_slice(slots);
    let mut reduction = Shared::<[f32]>::new_slice(64usize);
    let mut maximum = sink[h*sink.stride(0)];
    let mut slot = lane;
    while slot<slots {
        let p = ids[row*slots+slot];
        let mut value = 0.0f32;
        value -= 3.402823466e38f32;
        if p>=0.0f32 {value=dot(q,k,ak,b,t,p as usize,h,active);}
        logits[slot]=value;
        maximum=f32::max(maximum,value);
        slot+=64usize;
    }
    reduction[lane]=maximum;
    sync_cube();
    let mut stride=32usize;
    while stride>0 {
        if lane<stride {reduction[lane]=f32::max(reduction[lane],reduction[lane+stride]);}
        sync_cube(); stride/=2usize;
    }
    maximum=reduction[0];
    let mut sum=0.0f32;
    slot=lane;
    while slot<slots {
        let probability=f32::exp(logits[slot]-maximum);
        logits[slot]=probability;
        sum+=probability;
        slot+=64usize;
    }
    if lane==0 {sum+=f32::exp(sink[h*sink.stride(0)]-maximum);}
    reduction[lane]=sum;
    sync_cube(); stride=32usize;
    while stride>0 {
        if lane<stride {reduction[lane]+=reduction[lane+stride];}
        sync_cube(); stride/=2usize;
    }
    let inverse=1.0f32/reduction[0];
    if lane==0 {stats[row*2usize]=maximum; stats[row*2usize+1usize]=inverse;}
    let mut result=0.0f32;
    for s in 0..slots {
        let p=ids[row*slots+s];
        if p>=0.0f32 {result+=logits[s]*inverse*element(v,av,b,p as usize,h,lane,active);}
    }
    output[row*64usize+lane]=result;
}

#[cube(launch)]
fn query_backward(q: &View<f32>, k: &View<f32>, v: &View<f32>, ids: &View<f32>,
    sink: &View<f32>, ak: &View<f32>, av: &View<f32>, output: &View<f32>,
    stats: &View<f32>, upstream: &View<f32>, dq: &mut View<f32>, pairs: &mut View<f32>,
    ds_rows: &mut View<f32>, #[comptime] slots: usize, #[comptime] active: bool) {
    let lane=UNIT_POS as usize;
    let row=CUBE_POS as usize;
    let heads=q.shape(2); let n=q.shape(1);
    let h=row%heads; let t=row/heads%n; let b=row/(heads*n);
    if b>=q.shape(0) {terminate!();}
    let mut reduction=Shared::<[f32]>::new_slice(64usize);
    let mut derivatives=Shared::<[f32]>::new_slice(slots);
    reduction[lane]=upstream[ix(upstream,b,t,h,lane)]*output[row*64usize+lane];
    sync_cube(); let mut stride=32usize;
    while stride>0 {
        if lane<stride {reduction[lane]+=reduction[lane+stride];}
        sync_cube(); stride/=2usize;
    }
    let delta=reduction[0];
    let maximum=stats[row*2usize]; let inverse=stats[row*2usize+1usize];
    if lane==0 {ds_rows[row]=-delta*f32::exp(sink[h*sink.stride(0)]-maximum)*inverse;}
    let mut slot=lane;
    while slot<slots {
        let p=ids[row*slots+slot]; let mut probability=0.0f32; let mut derivative=0.0f32;
        if p>=0.0f32 {
            let s=p as usize;
            probability=f32::exp(dot(q,k,ak,b,t,s,h,active)-maximum)*inverse;
            let mut value=0.0f32;
            for c in 0..64usize {value+=upstream[ix(upstream,b,t,h,c)]*element(v,av,b,s,h,c,active);}
            derivative=probability*(value-delta)*SCALE;
        }
        derivatives[slot]=derivative;
        pairs[(row*slots+slot)*2usize]=probability;
        pairs[(row*slots+slot)*2usize+1usize]=derivative;
        slot+=64usize;
    }
    sync_cube();
    if lane<32usize {
        let mut result=0.0f32;
        for s in 0..slots {
            let p=ids[row*slots+s];
            if p>=0.0f32 {result+=derivatives[s]*element(k,ak,b,p as usize,h,lane,active);}
        }
        dq[row*32usize+lane]=result;
    }
}

#[cube(launch)]
fn kv_backward(q: &View<f32>, ids: &View<f32>, pairs: &View<f32>, upstream: &View<f32>,
    starts: &View<f32>, dk: &mut View<f32>, dv: &mut View<f32>, #[comptime] capacity: usize,
    #[comptime] chunk: usize, #[comptime] block: usize, #[comptime] top: usize, #[comptime] own: usize,
    #[comptime] frozen: bool, #[comptime] active_size: usize) {
    let lane=UNIT_POS as usize; let token=CUBE_POS as usize;
    let heads=q.shape(2); let time=dk.shape(1); let n=q.shape(1);
    let h=token%heads; let s=token/heads%time; let b=token/(heads*time);
    if b>=q.shape(0) {terminate!();}
    let start = starts[b*starts.stride(0)] as usize;
    let slots=capacity+chunk+top*block+own+active_size;
    let mut key_gradient=0.0f32; let mut value_gradient=0.0f32;
    for t in 0..n {
        let row=(b*n+t)*heads+h;
        let mut cut=(start+t)/chunk*chunk;
        if frozen {cut=start/chunk*chunk;}
        let mut own_start=(start+t)/block*block;
        if frozen {own_start=start/block*block;}
        let mut slot=lane+slots;
        slot-=lane;
        for cache in 0..capacity {
            if ids[row*slots+cache]==s as f32 {slot=cache;}
        }
        if s>=cut && s<cut+chunk {
            let local=capacity+s-cut;
            if ids[row*slots+local]==s as f32 {slot=local;}
        }
        for rank in 0..top {
            let extra=capacity+chunk+rank*block+s%block;
            if ids[row*slots+extra]==s as f32 {slot=extra;}
        }
        if s>=own_start && s<cut && s-own_start<own {
            let own_slot=capacity+chunk+top*block+s-own_start;
            if ids[row*slots+own_slot]==s as f32 {slot=own_slot;}
        }
        if slot<slots {
            let pair=(row*slots+slot)*2usize;
            value_gradient+=pairs[pair]*upstream[ix(upstream,b,t,h,lane)];
            if lane<32usize {key_gradient+=pairs[pair+1usize]*q[ix(q,b,t,h,lane)];}
        }
    }
    dv[token*64usize+lane]=value_gradient;
    if lane<32usize {dk[token*32usize+lane]=key_gradient;}
}
#[cube(launch)]
fn active_backward(q: &View<f32>, ids: &View<f32>, pairs: &View<f32>, upstream: &View<f32>,
    dak: &mut View<f32>, dav: &mut View<f32>, #[comptime] prefix_slots: usize) {
    let lane=UNIT_POS as usize;
    let token=CUBE_POS as usize;
    let heads=q.shape(2); let n=q.shape(1);
    let h=token%heads; let s=token/heads%n; let b=token/(heads*n);
    if b>=q.shape(0) {terminate!();}
    let slots=prefix_slots+n;
    let mut key_gradient=0.0f32; let mut value_gradient=0.0f32;
    for t in 0..n {
        let row=(b*n+t)*heads+h;
        let slot=prefix_slots+s;
        if ids[row*slots+slot]>=0.0f32 {
            let pair=(row*slots+slot)*2usize;
            value_gradient+=pairs[pair]*upstream[ix(upstream,b,t,h,lane)];
            if lane<32usize {key_gradient+=pairs[pair+1usize]*q[ix(q,b,t,h,lane)];}
        }
    }
    dav[token*64usize+lane]=value_gradient;
    if lane<32usize {dak[token*32usize+lane]=key_gradient;}
}

#[cube(launch)]
fn sink_backward(rows: &View<f32>, ds: &mut View<f32>) {
    let h=ABSOLUTE_POS as usize;
    if h>=ds.shape(0) {terminate!();}
    let mut sum=0.0f32;
    let mut row=h;
    while row<rows.len() {sum+=rows[row]; row+=ds.shape(0);}
    ds[h]=sum;
}

fn empty(reference: &CubeTensor, shape: impl Into<Shape>) -> CubeTensor {
    let shape=shape.into();
    crate::hola::checked_elements(shape.as_slice());
    CubeBackend::float_empty(shape,&reference.device,DType::F32.into())
}
fn count(reference: &CubeTensor, groups: usize) -> CubeCount {
    let (max_x,max_y,_)=reference.client.properties().hardware.max_cube_count;
    let x=groups.min(max_x as usize).min(65535).max(1);
    let y=groups.div_ceil(x); assert!(y<=max_y as usize);
    CubeCount::Static(x as u32,y as u32,1)
}
pub(crate) fn run_summaries(k: CubeTensor,valid: CubeTensor,documents: CubeTensor,block: usize) -> CubeTensor {
    let [b,time,h,width]=k.shape().dims::<4>();
    assert!(b>0 && time>0 && h>0 && width==32 && block>0 && block<=time);
    assert_eq!(valid.shape().dims::<2>(),[b,time]);
    assert_eq!(documents.shape().dims::<2>(),[b,time]);
    for x in [&k,&valid,&documents] {
        assert_eq!(x.dtype,DType::F32);
        assert_eq!(x.device,k.device);
        crate::hola::assert_u32_tensor(x);
    }
    let blocks=(time/block).max(1);
    let summaries=empty(&k,[b,blocks,h,35]);
    summary_kernel::launch(&k.client,count(&k,b*blocks*h),CubeDim::new_1d(64),
        k.clone().into_tensor_arg(),valid.into_tensor_arg(),documents.into_tensor_arg(),
        summaries.clone().into_tensor_arg(),block,blocks);
    summaries
}

pub(crate) fn run_read(q: CubeTensor,k: CubeTensor,v: CubeTensor,query_valid: CubeTensor,
    valid: CubeTensor,documents: CubeTensor,query_documents: CubeTensor,schedule: CubeTensor,
    summaries: CubeTensor,sink: CubeTensor,ak: CubeTensor,av: CubeTensor,starts: CubeTensor,
    chunk: usize,block: usize,top: usize,own: usize,frozen: bool,active: bool)
    -> (CubeTensor,CubeTensor,CubeTensor) {
    let [b,n,h,dk]=q.shape().dims::<4>(); let time=k.shape().dims::<4>()[1];
    let capacity=schedule.shape().dims::<4>()[3];
    assert!(b>0 && n>0 && h>0 && dk==32 && (1..=16).contains(&chunk));
    assert!(block>0 && block%chunk==0 && capacity>0);
    assert!(own<=block-chunk);
    assert!(time<=1<<24);
    assert!(time.checked_add(if active {n}else{0}).is_some_and(|positions|positions<=1<<24),
        "prefix retrieval positions must be exactly representable as f32");
    if active {
        assert!(frozen && (1..=16).contains(&n),"active prefix read requires a frozen query block of 1..16 tokens");
        assert_eq!(ak.shape().dims::<4>(),[b,n,h,32]);
        assert_eq!(av.shape().dims::<4>(),[b,n,h,64]);
    }
    assert_eq!(k.shape().dims::<4>(),[b,time,h,32]);
    assert_eq!(v.shape().dims::<4>(),[b,time,h,64]);
    assert_eq!(valid.shape().dims::<2>(),[b,time]);
    assert_eq!(documents.shape().dims::<2>(),[b,time]);
    assert_eq!(query_valid.shape().dims::<2>(),[b,n]);
    assert_eq!(query_documents.shape().dims::<2>(),[b,n]);
    assert_eq!(sink.shape().dims::<1>(),[h]);
    assert_eq!(starts.shape().dims::<1>(),[b]);
    assert_eq!(schedule.shape().dims::<4>(),[b,h,if frozen {1} else {time.div_ceil(chunk)+1},capacity]);
    for x in [&q,&k,&v,&query_valid,&valid,&documents,&query_documents,&schedule,&summaries,&sink,&ak,&av,&starts] {
        assert_eq!(x.dtype,DType::F32); assert_eq!(x.device,q.device);
        crate::hola::assert_u32_tensor(x);
    }
    let active_size=if active {n}else{0};
    let slots=capacity+chunk+top*block+own+active_size;
    assert!(slots<=4096,"prefix retrieval read exceeds shared-memory shape");
    let ids=empty(&q,[b,n,h,slots]); let output=empty(&q,[b,n,h,64]); let stats=empty(&q,[b,n,h,2]);
    let blocks=(time/block).max(1);
    assert!(blocks<=4096,"prefix retrieval summary count exceeds shared-memory shape");
    if top>0 {assert_eq!(summaries.shape().dims::<4>(),[b,blocks,h,35]);}
    select_kernel::launch(&q.client,count(&q,b*n*h),CubeDim::new_1d(64),
        q.clone().into_tensor_arg(),k.clone().into_tensor_arg(),query_valid.into_tensor_arg(),
        valid.into_tensor_arg(),documents.into_tensor_arg(),query_documents.into_tensor_arg(),
        schedule.into_tensor_arg(),summaries.into_tensor_arg(),starts.into_tensor_arg(),
        ids.clone().into_tensor_arg(),capacity,chunk,block,top,own,blocks,frozen,active_size);
    read_kernel::launch(&q.client,count(&q,b*n*h),CubeDim::new_1d(64),q.clone().into_tensor_arg(),
        k.into_tensor_arg(),v.into_tensor_arg(),ids.clone().into_tensor_arg(),sink.into_tensor_arg(),
        ak.into_tensor_arg(),av.into_tensor_arg(),output.clone().into_tensor_arg(),stats.clone().into_tensor_arg(),slots,active);
    (output,ids,stats)
}
pub(crate) fn run_backward(q: CubeTensor,k: CubeTensor,v: CubeTensor,ids: CubeTensor,sink: CubeTensor,
    ak: CubeTensor,av: CubeTensor,output: CubeTensor,stats: CubeTensor,upstream: CubeTensor,
    starts: CubeTensor,capacity: usize,chunk: usize,block: usize,top: usize,own: usize,frozen: bool,active: bool)
    -> (CubeTensor,CubeTensor,CubeTensor,CubeTensor,CubeTensor,CubeTensor) {
    let [b,n,h,_]=q.shape().dims::<4>(); let time=k.shape().dims::<4>()[1];
    let active_size=if active {n}else{0};
    let slots=capacity+chunk+top*block+own+active_size;
    let dq=empty(&q,q.shape()); let dk=empty(&q,k.shape()); let dv=empty(&q,v.shape());
    let ds=empty(&q,[h]); let rows=empty(&q,[b*n*h]); let pairs=empty(&q,[b,n,h,slots*2]);
    let dak=if active {empty(&q,ak.shape())}else{dk.clone()};
    let dav=if active {empty(&q,av.shape())}else{dv.clone()};
    query_backward::launch(&q.client,count(&q,b*n*h),CubeDim::new_1d(64),
        q.clone().into_tensor_arg(),k.into_tensor_arg(),v.into_tensor_arg(),ids.clone().into_tensor_arg(),
        sink.into_tensor_arg(),ak.into_tensor_arg(),av.into_tensor_arg(),output.into_tensor_arg(),
        stats.into_tensor_arg(),upstream.clone().into_tensor_arg(),dq.clone().into_tensor_arg(),
        pairs.clone().into_tensor_arg(),rows.clone().into_tensor_arg(),slots,active);
    kv_backward::launch(&q.client,count(&q,b*time*h),CubeDim::new_1d(64),q.clone().into_tensor_arg(),
        ids.clone().into_tensor_arg(),pairs.clone().into_tensor_arg(),upstream.clone().into_tensor_arg(),
        starts.into_tensor_arg(),dk.clone().into_tensor_arg(),dv.clone().into_tensor_arg(),capacity,chunk,block,top,own,frozen,active_size);
    if active {
        active_backward::launch(&q.client,count(&q,b*n*h),CubeDim::new_1d(64),q.clone().into_tensor_arg(),
            ids.into_tensor_arg(),pairs.into_tensor_arg(),upstream.into_tensor_arg(),
            dak.clone().into_tensor_arg(),dav.clone().into_tensor_arg(),capacity+chunk+top*block+own);
    }
    sink_backward::launch(&q.client,count(&q,h.div_ceil(64)),CubeDim::new_1d(64),rows.into_tensor_arg(),ds.clone().into_tensor_arg());
    (dq,dk,dv,ds,dak,dav)
}
