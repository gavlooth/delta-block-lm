//! Explicit non-autotuned matmul for numerically gated exceptions to the global autotuner.
#[cfg(feature = "fusion")]
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::{
    Backend, Dispatch,
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
use burn_cubecl::{
    CubeBackend,
    kernel::matmul::{MatmulStrategy, matmul},
};

#[backend_extension(Cube, Autodiff, Fusion: cfg(feature = "fusion"))]
pub trait HeuristicMatmulOps: Backend {
    #[fusion(meta = |a: &TensorSpec, b: &TensorSpec| {
        let [m,_] = a.shape.dims::<2>();
        let [_,n] = b.shape.dims::<2>();
        TensorSpec::new([m,n].into(),a.dtype)
    })]
    fn heuristic_matmul(a: FloatTensor<Self>, b: FloatTensor<Self>) -> FloatTensor<Self>;
}
impl HeuristicMatmulOps for CubeBackend {
    fn heuristic_matmul(a: FloatTensor<Self>, b: FloatTensor<Self>) -> FloatTensor<Self> {
        let dtype = a.dtype;
        matmul(a, b, None, MatmulStrategy::Cube, dtype).expect("explicit heuristic GEMM setup")
    }
}
impl<B: Backend + HeuristicMatmulOps, C: CheckpointStrategy> HeuristicMatmulOps for Autodiff<B, C> {
    fn heuristic_matmul(a: FloatTensor<Self>, b: FloatTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct MatmulBackward;
        impl<B: Backend + HeuristicMatmulOps> Backward<B, 2> for MatmulBackward {
            type State = (FloatTensor<B>, FloatTensor<B>);
            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                _: &mut Checkpointer,
            ) {
                let dy = grads.consume::<B>(&ops.node);
                let (a, b) = ops.state;
                let [left, right] = ops.parents;
                if let Some(node) = left {
                    let dx = B::heuristic_matmul(dy.clone(), B::float_swap_dims(b, 0, 1));
                    grads.register::<B>(node.id, dx);
                }
                if let Some(node) = right {
                    let dw = B::heuristic_matmul(B::float_swap_dims(a, 0, 1), dy);
                    grads.register::<B>(node.id, dw);
                }
            }
        }
        let parents = [a.node(), b.node()];
        let (a, b) = (a.into_primitive(), b.into_primitive());
        let y = B::heuristic_matmul(a.clone(), b.clone());
        match MatmulBackward
            .prepare::<C>(parents)
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish((a, b), y),
            OpsKind::UnTracked(prep) => prep.finish(y),
        }
    }
}
/// Two-dimensional first-order GEMM, retaining transposed views and the library's f32 heuristic.
pub fn heuristic_matmul(a: Tensor<2>, b: Tensor<2>) -> Tensor<2> {
    Tensor::from_dispatch(<Dispatch as HeuristicMatmulOps>::heuristic_matmul(
        a.into_dispatch(),
        b.into_dispatch(),
    ))
}
