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
    ops::{ActivationOps, FloatTensorOps},
    tensor::FloatTensor,
};
use burn::prelude::Tensor;
use burn_cubecl::CubeBackend;

#[backend_extension(Cube, Autodiff, Fusion: cfg(feature = "fusion"))]
pub trait EpilogueOps: Backend {
    #[fusion(meta = |x: &TensorSpec, _fused_forward| x.clone())]
    fn silu_analytic(x: FloatTensor<Self>, fused_forward: bool) -> FloatTensor<Self>;
    #[fusion(meta = |x: &TensorSpec, _dy| x.clone())]
    fn silu_vjp(x: FloatTensor<Self>, dy: FloatTensor<Self>) -> FloatTensor<Self>;
}
impl EpilogueOps for CubeBackend {
    fn silu_analytic(x: FloatTensor<Self>, fused_forward: bool) -> FloatTensor<Self> {
        if fused_forward {
            crate::epilogue::launch(x, None)
        } else {
            Self::float_mul(x.clone(), Self::sigmoid(x))
        }
    }
    fn silu_vjp(x: FloatTensor<Self>, dy: FloatTensor<Self>) -> FloatTensor<Self> {
        crate::epilogue::launch(x, Some(dy))
    }
}
impl<B: Backend + EpilogueOps, C: CheckpointStrategy> EpilogueOps for Autodiff<B, C> {
    fn silu_analytic(x: FloatTensor<Self>, fused_forward: bool) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct SiluBackward;
        impl<B: Backend + EpilogueOps> Backward<B, 1> for SiluBackward {
            type State = FloatTensor<B>;
            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _: &mut Checkpointer,
            ) {
                let dy = grads.consume::<B>(&ops.node);
                if let Some(parent) = &ops.parents[0] {
                    grads.register::<B>(parent.id, B::silu_vjp(ops.state, dy));
                }
            }
        }
        let parents = [x.node()];
        let x = x.into_primitive();
        match SiluBackward
            .prepare::<C>(parents)
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let y = B::silu_analytic(x.clone(), fused_forward);
                prep.finish(x, y)
            }
            OpsKind::UnTracked(prep) => prep.finish(B::silu_analytic(x, fused_forward)),
        }
    }
    fn silu_vjp(x: FloatTensor<Self>, dy: FloatTensor<Self>) -> FloatTensor<Self> {
        Self::from_inner(B::silu_vjp(x.into_primitive(), dy.into_primitive()))
    }
}
/// First-order f32 GPU SiLU; retains the input only, recomputing sigmoid in one VJP kernel.
/// `fused_forward=false` keeps CubeBackend's library sigmoid/multiply forward.
pub fn silu_analytic(x: Tensor<3>, fused_forward: bool) -> Tensor<3> {
    Tensor::from_dispatch(<Dispatch as EpilogueOps>::silu_analytic(
        x.into_dispatch(),
        fused_forward,
    ))
}
/// Explicit VJP entry point for noncontiguous upstream reference checks.
pub fn silu_vjp(x: Tensor<3>, dy: Tensor<3>) -> Tensor<3> {
    Tensor::from_dispatch(<Dispatch as EpilogueOps>::silu_vjp(
        x.into_dispatch(),
        dy.into_dispatch(),
    ))
}
