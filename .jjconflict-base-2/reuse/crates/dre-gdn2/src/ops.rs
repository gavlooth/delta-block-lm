use burn::backend::{
    AutodiffBackend, Backend, Dispatch, backend_extension,
    autodiff::{Autodiff, checkpoint::{base::Checkpointer, strategy::CheckpointStrategy}, grads::Gradients, ops::{Backward, Ops, OpsKind}},
    fusion::custom::TensorSpec, tensor::{FloatTensor, IntTensor}, wgpu::CubeBackend,
};
use burn::prelude::{Shape, Tensor};

fn scan_meta(_q: &TensorSpec, _k: &TensorSpec, v: &TensorSpec, _alpha: &TensorSpec, _erase: &TensorSpec, _write: &TensorSpec, _doc_start: &TensorSpec, _doc_end: &TensorSpec, interval: &usize, chunked: &bool) -> (TensorSpec, TensorSpec) {
    let [batch, tokens, heads, _] = v.shape.dims::<4>();
    let (segments, rows) = if *chunked {
        (tokens.div_ceil(crate::chunked::C), 32 + crate::chunked::C)
    } else { (tokens.div_ceil(*interval), 32) };
    (v.clone(), TensorSpec::new(Shape::new([batch, heads, segments, rows, 64]), burn::tensor::DType::F32))
}

fn backward_meta(q: &TensorSpec, k: &TensorSpec, v: &TensorSpec, alpha: &TensorSpec, erase: &TensorSpec, write: &TensorSpec, _out_grad: &TensorSpec, _checkpoint: &TensorSpec, _doc_start: &TensorSpec, _doc_end: &TensorSpec, _interval: &usize, _chunked: &bool) -> (TensorSpec, TensorSpec, TensorSpec, TensorSpec, TensorSpec, TensorSpec) {
    (q.clone(), k.clone(), v.clone(), alpha.clone(), erase.clone(), write.clone())
}

#[backend_extension(Cube, Autodiff, Fusion)]
pub trait Gdn2Ops: Backend {
    #[fusion(meta = scan_meta)]
    fn gdn2_scan(q: FloatTensor<Self>, k: FloatTensor<Self>, v: FloatTensor<Self>, alpha: FloatTensor<Self>, erase: FloatTensor<Self>, write: FloatTensor<Self>, doc_start: IntTensor<Self>, doc_end: IntTensor<Self>, interval: usize, chunked: bool) -> (FloatTensor<Self>, FloatTensor<Self>);

    #[fusion(meta = backward_meta)]
    fn gdn2_backward(q: FloatTensor<Self>, k: FloatTensor<Self>, v: FloatTensor<Self>, alpha: FloatTensor<Self>, erase: FloatTensor<Self>, write: FloatTensor<Self>, output_grad: FloatTensor<Self>, checkpoint: FloatTensor<Self>, doc_start: IntTensor<Self>, doc_end: IntTensor<Self>, interval: usize, chunked: bool) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);
}

impl Gdn2Ops for CubeBackend {
    fn gdn2_scan(q: FloatTensor<Self>, k: FloatTensor<Self>, v: FloatTensor<Self>, alpha: FloatTensor<Self>, erase: FloatTensor<Self>, write: FloatTensor<Self>, doc_start: IntTensor<Self>, doc_end: IntTensor<Self>, interval: usize, chunked: bool) -> (FloatTensor<Self>, FloatTensor<Self>) {
        if chunked { crate::chunked::run_scan(q, k, v, alpha, erase, write, doc_start, doc_end) }
        else { crate::cube::run_scan(q, k, v, alpha, erase, write, doc_start, doc_end, interval) }
    }

    fn gdn2_backward(q: FloatTensor<Self>, k: FloatTensor<Self>, v: FloatTensor<Self>, alpha: FloatTensor<Self>, erase: FloatTensor<Self>, write: FloatTensor<Self>, output_grad: FloatTensor<Self>, checkpoint: FloatTensor<Self>, doc_start: IntTensor<Self>, doc_end: IntTensor<Self>, interval: usize, chunked: bool) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        if chunked { crate::chunked::run_backward(q, k, v, alpha, erase, write, output_grad, checkpoint, doc_start, doc_end) }
        else { crate::cube::run_backward(q, k, v, alpha, erase, write, output_grad, checkpoint, doc_start, doc_end, interval) }
    }
}

impl<B: Backend + Gdn2Ops, C: CheckpointStrategy> Gdn2Ops for Autodiff<B, C> {
    fn gdn2_scan(q: FloatTensor<Self>, k: FloatTensor<Self>, v: FloatTensor<Self>, alpha: FloatTensor<Self>, erase: FloatTensor<Self>, write: FloatTensor<Self>, doc_start: IntTensor<Self>, doc_end: IntTensor<Self>, interval: usize, chunked: bool) -> (FloatTensor<Self>, FloatTensor<Self>) {
        #[derive(Debug)]
        struct ScanBackward;
        impl<B: Backend + Gdn2Ops> Backward<B, 6> for ScanBackward {
            type State = (FloatTensor<B>, FloatTensor<B>, FloatTensor<B>, FloatTensor<B>, FloatTensor<B>, FloatTensor<B>, FloatTensor<B>, IntTensor<B>, IntTensor<B>, usize, bool);
            fn backward(self, ops: Ops<Self::State, 6>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
                let [nq, nk, nv, na, ne, nw] = ops.parents;
                let output_grad = grads.consume::<B>(&ops.node);
                let (q, k, v, alpha, erase, write, checkpoint, doc_start, doc_end, interval, chunked) = ops.state;
                let (dq, dk, dv, da, de, dw) = B::gdn2_backward(q, k, v, alpha, erase, write, output_grad, checkpoint, doc_start, doc_end, interval, chunked);
                if let Some(node) = nq { grads.register::<B>(node.id, dq); }
                if let Some(node) = nk { grads.register::<B>(node.id, dk); }
                if let Some(node) = nv { grads.register::<B>(node.id, dv); }
                if let Some(node) = na { grads.register::<B>(node.id, da); }
                if let Some(node) = ne { grads.register::<B>(node.id, de); }
                if let Some(node) = nw { grads.register::<B>(node.id, dw); }
            }
        }
        let guards = [q.node(), k.node(), v.node(), alpha.node(), erase.node(), write.node()];
        let (q, k, v, alpha, erase, write) = (q.into_primitive(), k.into_primitive(), v.into_primitive(), alpha.into_primitive(), erase.into_primitive(), write.into_primitive());
        let (out, checkpoint) = B::gdn2_scan(q.clone(), k.clone(), v.clone(), alpha.clone(), erase.clone(), write.clone(), doc_start.clone(), doc_end.clone(), interval, chunked);
        let checkpoint_ad = <Autodiff<B, C> as AutodiffBackend>::from_inner(checkpoint.clone());
        let out = match ScanBackward.prepare::<C>(guards).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish((q, k, v, alpha, erase, write, checkpoint, doc_start, doc_end, interval, chunked), out),
            OpsKind::UnTracked(prep) => prep.finish(out),
        };
        (out, checkpoint_ad)
    }

    fn gdn2_backward(q: FloatTensor<Self>, k: FloatTensor<Self>, v: FloatTensor<Self>, alpha: FloatTensor<Self>, erase: FloatTensor<Self>, write: FloatTensor<Self>, output_grad: FloatTensor<Self>, checkpoint: FloatTensor<Self>, doc_start: IntTensor<Self>, doc_end: IntTensor<Self>, interval: usize, chunked: bool) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let (dq, dk, dv, da, de, dw) = B::gdn2_backward(q.into_primitive(), k.into_primitive(), v.into_primitive(), alpha.into_primitive(), erase.into_primitive(), write.into_primitive(), output_grad.into_primitive(), checkpoint.into_primitive(), doc_start, doc_end, interval, chunked);
        (<Autodiff<B, C> as AutodiffBackend>::from_inner(dq), <Autodiff<B, C> as AutodiffBackend>::from_inner(dk), <Autodiff<B, C> as AutodiffBackend>::from_inner(dv), <Autodiff<B, C> as AutodiffBackend>::from_inner(da), <Autodiff<B, C> as AutodiffBackend>::from_inner(de), <Autodiff<B, C> as AutodiffBackend>::from_inner(dw))
    }
}

pub fn gdn2_scan(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, alpha: Tensor<4>, erase: Tensor<4>, write: Tensor<4>, doc_start: Tensor<2, burn::tensor::Int>, doc_end: Tensor<2, burn::tensor::Int>, interval: usize, chunked: bool) -> Tensor<4> {
    // Recurrence gates never use low-precision storage or decay arithmetic.
    let alpha = alpha.cast(burn::tensor::DType::F32);
    let erase = erase.cast(burn::tensor::DType::F32);
    let write = write.cast(burn::tensor::DType::F32);
    let (output, _checkpoint) = <Dispatch as Gdn2Ops>::gdn2_scan(q.into_dispatch(), k.into_dispatch(), v.into_dispatch(), alpha.into_dispatch(), erase.into_dispatch(), write.into_dispatch(), doc_start.into_dispatch(), doc_end.into_dispatch(), interval, chunked);
    Tensor::from_dispatch(output)
}
