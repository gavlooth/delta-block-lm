#[cfg(feature = "fusion")]
use burn::backend::fusion::custom::TensorSpec;
use burn::{
    backend::{
        Backend, Dispatch,
        autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy},
        backend_extension,
        tensor::FloatTensor,
    },
    prelude::Tensor,
};
use burn_cubecl::CubeBackend;

#[backend_extension(Cube, Autodiff, Fusion: cfg(feature = "fusion"))]
pub trait UpdateOps: Backend {
    #[fusion(meta = |x: &TensorSpec| TensorSpec::new([1].into(), x.dtype))]
    fn square_sum(x: FloatTensor<Self>) -> FloatTensor<Self>;
    #[fusion(meta = |x: &TensorSpec, _g, _m, _v, _initialized, _time, _lr, _decay, _beta1, _beta2, _epsilon| (x.clone(), x.clone(), x.clone()))]
    fn adam(
        x: FloatTensor<Self>,
        g: FloatTensor<Self>,
        m: FloatTensor<Self>,
        v: FloatTensor<Self>,
        initialized: bool,
        time: usize,
        lr: f64,
        decay: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);
}
impl UpdateOps for CubeBackend {
    fn square_sum(x: FloatTensor<Self>) -> FloatTensor<Self> {
        crate::update::square_sum(x)
    }
    fn adam(
        x: FloatTensor<Self>,
        g: FloatTensor<Self>,
        m: FloatTensor<Self>,
        v: FloatTensor<Self>,
        initialized: bool,
        time: usize,
        lr: f64,
        decay: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        crate::update::adam(x, g, m, v, initialized, time, lr, decay, beta1, beta2, epsilon)
    }
}
impl<B: Backend + UpdateOps, C: CheckpointStrategy> UpdateOps for Autodiff<B, C> {
    fn square_sum(x: FloatTensor<Self>) -> FloatTensor<Self> {
        <Self as burn::backend::AutodiffBackend>::from_inner(B::square_sum(x.into_primitive()))
    }
    fn adam(
        x: FloatTensor<Self>,
        g: FloatTensor<Self>,
        m: FloatTensor<Self>,
        v: FloatTensor<Self>,
        initialized: bool,
        time: usize,
        lr: f64,
        decay: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let (x, m, v) = B::adam(
            x.into_primitive(),
            g.into_primitive(),
            m.into_primitive(),
            v.into_primitive(),
            initialized,
            time,
            lr,
            decay,
            beta1,
            beta2,
            epsilon,
        );
        use burn::backend::AutodiffBackend;
        (
            Self::from_inner(x),
            Self::from_inner(m),
            Self::from_inner(v),
        )
    }
}
/// Non-autodiff gradient-safety reduction. Does not materialize a squared gradient.
pub fn gradient_square_sum<const D: usize>(gradient: Tensor<D>) -> Tensor<1> {
    Tensor::from_dispatch(<Dispatch as UpdateOps>::square_sum(
        gradient.into_dispatch(),
    ))
}
/// First-order optimizer operation on globally clipped gradients, with explicit AdamW constants.
pub fn adamw_update<const D: usize>(
    x: Tensor<D>,
    g: Tensor<D>,
    m: Option<Tensor<D>>,
    v: Option<Tensor<D>>,
    time: usize,
    lr: f64,
    decay: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
) -> (Tensor<D>, Tensor<D>, Tensor<D>) {
    let initialized = m.is_some();
    assert_eq!(initialized, v.is_some());
    let raw = x.into_dispatch();
    let (x, m, v) = <Dispatch as UpdateOps>::adam(
        raw.clone(),
        g.into_dispatch(),
        m.map(Tensor::into_dispatch).unwrap_or_else(|| raw.clone()),
        v.map(Tensor::into_dispatch).unwrap_or(raw),
        initialized,
        time,
        lr,
        decay,
        beta1,
        beta2,
        epsilon,
    );
    (
        Tensor::from_dispatch(x),
        Tensor::from_dispatch(m),
        Tensor::from_dispatch(v),
    )
}
