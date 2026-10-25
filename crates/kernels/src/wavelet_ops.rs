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
use burn_cubecl::CubeBackend;
#[backend_extension(Cube,Autodiff,Fusion: cfg(feature="fusion"))]
pub trait WaveletOps: Backend {
    #[fusion(meta=|x:&TensorSpec,_valid,_docs,_dilation,_transpose|x.clone())]
    fn wavelet(
        x: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        docs: FloatTensor<Self>,
        dilation: usize,
        transpose: bool,
    ) -> FloatTensor<Self>;
}
impl WaveletOps for CubeBackend {
    fn wavelet(
        x: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        docs: FloatTensor<Self>,
        dilation: usize,
        transpose: bool,
    ) -> FloatTensor<Self> {
        crate::wavelet::run(x, valid, docs, dilation, transpose)
    }
}
impl<B: Backend + WaveletOps, C: CheckpointStrategy> WaveletOps for Autodiff<B, C> {
    fn wavelet(
        x: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        docs: FloatTensor<Self>,
        dilation: usize,
        transpose: bool,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct WaveletBackward;
        impl<B: Backend + WaveletOps> Backward<B, 1> for WaveletBackward {
            type State = (FloatTensor<B>, FloatTensor<B>, usize, bool);
            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _: &mut Checkpointer,
            ) {
                let dy = grads.consume::<B>(&ops.node);
                let (valid, docs, dilation, transpose) = ops.state;
                let dx = B::wavelet(dy, valid, docs, dilation, !transpose);
                if let Some(node) = ops.parents[0].as_ref() {
                    grads.register::<B>(node.id, dx);
                }
            }
        }
        let parents = [x.node()];
        let (x, valid, docs) = (
            x.into_primitive(),
            valid.into_primitive(),
            docs.into_primitive(),
        );
        let y = B::wavelet(x, valid.clone(), docs.clone(), dilation, transpose);
        match WaveletBackward
            .prepare::<C>(parents)
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish((valid, docs, dilation, transpose), y),
            OpsKind::UnTracked(prep) => prep.finish(y),
        }
    }
}
/// First-order f32 GPU transform. Fixed coefficients; retains metadata only, not x.
pub fn wavelet_smooth(
    x: Tensor<3>,
    valid: Tensor<2>,
    docs: Tensor<2>,
    dilation: usize,
) -> Tensor<3> {
    assert!(!valid.is_require_grad() && !docs.is_require_grad());
    Tensor::from_dispatch(<Dispatch as WaveletOps>::wavelet(
        x.into_dispatch(),
        valid.into_dispatch(),
        docs.into_dispatch(),
        dilation,
        false,
    ))
}
