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
    tensor::FloatTensor,
};
use burn::prelude::Tensor;
use burn_cubecl::CubeBackend;
#[backend_extension(Cube, Autodiff, Fusion: cfg(feature = "fusion"))]
pub trait OhcOps: Backend {
    #[fusion(meta = |x: &TensorSpec,_q,_w,_f| x.clone())]
    fn ohc_write(
        x: FloatTensor<Self>,
        q: FloatTensor<Self>,
        w: FloatTensor<Self>,
        f: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
    #[fusion(meta = |x: &TensorSpec,q: &TensorSpec,w: &TensorSpec,f: &TensorSpec,_dy| (x.clone(),q.clone(),w.clone(),f.clone()))]
    fn ohc_vjp(
        x: FloatTensor<Self>,
        q: FloatTensor<Self>,
        w: FloatTensor<Self>,
        f: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    );
}
impl OhcOps for CubeBackend {
    fn ohc_write(
        x: FloatTensor<Self>,
        q: FloatTensor<Self>,
        w: FloatTensor<Self>,
        f: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        crate::ohc::forward(x, q, w, f)
    }
    fn ohc_vjp(
        x: FloatTensor<Self>,
        q: FloatTensor<Self>,
        w: FloatTensor<Self>,
        f: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        crate::ohc::backward(x, q, w, f, dy)
    }
}
impl<B: Backend + OhcOps, C: CheckpointStrategy> OhcOps for Autodiff<B, C> {
    fn ohc_write(
        x: FloatTensor<Self>,
        q: FloatTensor<Self>,
        w: FloatTensor<Self>,
        f: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct WriteBackward;
        impl<B: Backend + OhcOps> Backward<B, 4> for WriteBackward {
            type State = (
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
            );
            fn backward(
                self,
                ops: Ops<Self::State, 4>,
                grads: &mut Gradients,
                _: &mut Checkpointer,
            ) {
                let dy = grads.consume::<B>(&ops.node);
                let (x, q, w, f) = ops.state;
                let (dx, dq, dw, df) = B::ohc_vjp(x, q, w, f, dy);
                for (node, g) in ops.parents.into_iter().zip([dx, dq, dw, df]) {
                    if let Some(node) = node {
                        grads.register::<B>(node.id, g);
                    }
                }
            }
        }
        let parents = [x.node(), q.node(), w.node(), f.node()];
        let (x, q, w, f) = (
            x.into_primitive(),
            q.into_primitive(),
            w.into_primitive(),
            f.into_primitive(),
        );
        let y = B::ohc_write(x.clone(), q.clone(), w.clone(), f.clone());
        match WriteBackward
            .prepare::<C>(parents)
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish((x, q, w, f), y),
            OpsKind::UnTracked(prep) => prep.finish(y),
        }
    }
    fn ohc_vjp(
        x: FloatTensor<Self>,
        q: FloatTensor<Self>,
        w: FloatTensor<Self>,
        f: FloatTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        let (dx, dq, dw, df) = B::ohc_vjp(
            x.into_primitive(),
            q.into_primitive(),
            w.into_primitive(),
            f.into_primitive(),
            dy.into_primitive(),
        );
        (
            Self::from_inner(dx),
            Self::from_inner(dq),
            Self::from_inner(dw),
            Self::from_inner(df),
        )
    }
}
/// QX + w fᵀ for f32 per-token inputs, with a first-order analytic VJP.
pub fn ohc_write(x: Tensor<4>, q: Tensor<4>, w: Tensor<3>, f: Tensor<3>) -> Tensor<4> {
    Tensor::from_dispatch(<Dispatch as OhcOps>::ohc_write(
        x.into_dispatch(),
        q.into_dispatch(),
        w.into_dispatch(),
        f.into_dispatch(),
    ))
}
