#[cfg(feature = "fusion")]
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::{
    AutodiffBackend, Backend, Dispatch,
    autodiff::{
        Autodiff,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    backend_extension,
    tensor::{FloatTensor, IntTensor},
};
use burn::prelude::Tensor;
use burn_cubecl::CubeBackend;

#[backend_extension(Cube, Autodiff, Fusion: cfg(feature = "fusion"))]
pub trait Gdn2Ops: Backend {
    #[fusion(meta = |q: &TensorSpec, _k, v: &TensorSpec, _alpha, _erase, _write, _doc_start, _doc_end| {
        let [batch, time, heads, _] = q.shape.dims::<4>();
        (v.clone(), TensorSpec::new([batch, heads, time.div_ceil(16), 48, 64].into(), q.dtype))
    })]
    fn gdn2_scan(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        alpha: FloatTensor<Self>,
        erase: FloatTensor<Self>,
        write: FloatTensor<Self>,
        doc_start: IntTensor<Self>,
        doc_end: IntTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);

    #[fusion(meta = |q: &TensorSpec, k: &TensorSpec, v: &TensorSpec, alpha: &TensorSpec, erase: &TensorSpec, write: &TensorSpec, _output_grad, _checkpoint, _doc_start, _doc_end| {
        (q.clone(), k.clone(), v.clone(), alpha.clone(), erase.clone(), write.clone())
    })]
    fn gdn2_backward(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        alpha: FloatTensor<Self>,
        erase: FloatTensor<Self>,
        write: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        checkpoint: FloatTensor<Self>,
        doc_start: IntTensor<Self>,
        doc_end: IntTensor<Self>,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    );
}

impl Gdn2Ops for CubeBackend {
    fn gdn2_scan(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        alpha: FloatTensor<Self>,
        erase: FloatTensor<Self>,
        write: FloatTensor<Self>,
        doc_start: IntTensor<Self>,
        doc_end: IntTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        crate::gdn2::run_scan(q, k, v, alpha, erase, write, doc_start, doc_end)
    }

    fn gdn2_backward(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        alpha: FloatTensor<Self>,
        erase: FloatTensor<Self>,
        write: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        checkpoint: FloatTensor<Self>,
        doc_start: IntTensor<Self>,
        doc_end: IntTensor<Self>,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        crate::gdn2::run_backward(
            q,
            k,
            v,
            alpha,
            erase,
            write,
            output_grad,
            checkpoint,
            doc_start,
            doc_end,
        )
    }
}

impl<B: Backend + Gdn2Ops, C: CheckpointStrategy> Gdn2Ops for Autodiff<B, C> {
    fn gdn2_scan(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        alpha: FloatTensor<Self>,
        erase: FloatTensor<Self>,
        write: FloatTensor<Self>,
        doc_start: IntTensor<Self>,
        doc_end: IntTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        #[derive(Debug)]
        struct ScanBackward;
        impl<B: Backend + Gdn2Ops> Backward<B, 6> for ScanBackward {
            type State = (
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                IntTensor<B>,
                IntTensor<B>,
            );
            fn backward(
                self,
                ops: Ops<Self::State, 6>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [nq, nk, nv, na, ne, nw] = ops.parents;
                let output_grad = grads.consume::<B>(&ops.node);
                let (q, k, v, alpha, erase, write, checkpoint, doc_start, doc_end) = ops.state;
                let (dq, dk, dv, da, de, dw) = B::gdn2_backward(
                    q,
                    k,
                    v,
                    alpha,
                    erase,
                    write,
                    output_grad,
                    checkpoint,
                    doc_start,
                    doc_end,
                );
                if let Some(node) = nq {
                    grads.register::<B>(node.id, dq);
                }
                if let Some(node) = nk {
                    grads.register::<B>(node.id, dk);
                }
                if let Some(node) = nv {
                    grads.register::<B>(node.id, dv);
                }
                if let Some(node) = na {
                    grads.register::<B>(node.id, da);
                }
                if let Some(node) = ne {
                    grads.register::<B>(node.id, de);
                }
                if let Some(node) = nw {
                    grads.register::<B>(node.id, dw);
                }
            }
        }
        let guards = [
            q.node(),
            k.node(),
            v.node(),
            alpha.node(),
            erase.node(),
            write.node(),
        ];
        let (q, k, v, alpha, erase, write) = (
            q.into_primitive(),
            k.into_primitive(),
            v.into_primitive(),
            alpha.into_primitive(),
            erase.into_primitive(),
            write.into_primitive(),
        );
        let (out, checkpoint) = B::gdn2_scan(
            q.clone(),
            k.clone(),
            v.clone(),
            alpha.clone(),
            erase.clone(),
            write.clone(),
            doc_start.clone(),
            doc_end.clone(),
        );
        let checkpoint_ad = <Autodiff<B, C> as AutodiffBackend>::from_inner(checkpoint.clone());
        let out = match ScanBackward.prepare::<C>(guards).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish(
                (q, k, v, alpha, erase, write, checkpoint, doc_start, doc_end),
                out,
            ),
            OpsKind::UnTracked(prep) => prep.finish(out),
        };
        (out, checkpoint_ad)
    }

    fn gdn2_backward(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        alpha: FloatTensor<Self>,
        erase: FloatTensor<Self>,
        write: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
        checkpoint: FloatTensor<Self>,
        doc_start: IntTensor<Self>,
        doc_end: IntTensor<Self>,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        let (dq, dk, dv, da, de, dw) = B::gdn2_backward(
            q.into_primitive(),
            k.into_primitive(),
            v.into_primitive(),
            alpha.into_primitive(),
            erase.into_primitive(),
            write.into_primitive(),
            output_grad.into_primitive(),
            checkpoint.into_primitive(),
            doc_start,
            doc_end,
        );
        (
            <Autodiff<B, C> as AutodiffBackend>::from_inner(dq),
            <Autodiff<B, C> as AutodiffBackend>::from_inner(dk),
            <Autodiff<B, C> as AutodiffBackend>::from_inner(dv),
            <Autodiff<B, C> as AutodiffBackend>::from_inner(da),
            <Autodiff<B, C> as AutodiffBackend>::from_inner(de),
            <Autodiff<B, C> as AutodiffBackend>::from_inner(dw),
        )
    }
}

pub fn gdn2_scan(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    alpha: Tensor<4>,
    erase: Tensor<4>,
    write: Tensor<4>,
    doc_start: Tensor<2, burn::tensor::Int>,
    doc_end: Tensor<2, burn::tensor::Int>,
) -> Tensor<4> {
    // Recurrence gates never use low-precision storage or decay arithmetic.
    let alpha = alpha.cast(burn::tensor::DType::F32);
    let erase = erase.cast(burn::tensor::DType::F32);
    let write = write.cast(burn::tensor::DType::F32);
    let (output, _checkpoint) = <Dispatch as Gdn2Ops>::gdn2_scan(
        q.into_dispatch(),
        k.into_dispatch(),
        v.into_dispatch(),
        alpha.into_dispatch(),
        erase.into_dispatch(),
        write.into_dispatch(),
        doc_start.into_dispatch(),
        doc_end.into_dispatch(),
    );
    Tensor::from_dispatch(output)
}
