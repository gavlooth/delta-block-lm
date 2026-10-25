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
pub trait CodpulaOps: Backend {
    #[fusion(meta = |x: &TensorSpec, pi: &TensorSpec, _valid| {
        let [c,l,_] = x.shape.dims::<3>();
        TensorSpec::new([c,l,pi.shape.dims::<2>()[0]].into(), x.dtype)
    })]
    fn codpula_leaf_normalizers(
        logits: FloatTensor<Self>, log_pi: FloatTensor<Self>, valid: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    #[fusion(meta = |x: &TensorSpec, pi: &TensorSpec, _valid, _normalizers, _upstream, _need_logits, _need_pi| (x.clone(), pi.clone()))]
    fn codpula_leaf_backward(
        logits: FloatTensor<Self>, log_pi: FloatTensor<Self>, valid: FloatTensor<Self>,
        normalizers: FloatTensor<Self>, upstream: FloatTensor<Self>,
        need_logits: bool, need_pi: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);

    #[fusion(meta = |e: &TensorSpec, _a, _root, _valid| {
        TensorSpec::new([e.shape.dims::<3>()[0]].into(), e.dtype)
    })]
    fn codpula_partition(
        log_evidence: FloatTensor<Self>, log_transition: FloatTensor<Self>,
        log_root: FloatTensor<Self>, valid: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    #[fusion(meta = |e: &TensorSpec, a: &TensorSpec, _root, _valid| {
        let [c,_,_] = e.shape.dims::<3>();
        (TensorSpec::new([c].into(), e.dtype), e.clone(), a.clone())
    })]
    fn codpula_chain(
        log_evidence: FloatTensor<Self>, log_transition: FloatTensor<Self>,
        log_root: FloatTensor<Self>, valid: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);

    #[fusion(meta = |g: &TensorSpec, a: &TensorSpec, _upstream| {
        (g.clone(), a.clone(), TensorSpec::new([g.shape.dims::<3>()[2]].into(), g.dtype))
    })]
    fn codpula_chain_backward(
        gamma: FloatTensor<Self>, summed_xi: FloatTensor<Self>, upstream: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);

    #[fusion(meta = |x: &TensorSpec, pi: &TensorSpec, _normalizers, _gamma, _tokens, _observed, _valid, counts: &bool| {
        if *counts { pi.clone() } else { x.clone() }
    })]
    fn codpula_readout(
        logits: FloatTensor<Self>, log_pi: FloatTensor<Self>, normalizers: FloatTensor<Self>,
        gamma: FloatTensor<Self>, tokens: FloatTensor<Self>, observed: FloatTensor<Self>,
        valid: FloatTensor<Self>, counts: bool,
    ) -> FloatTensor<Self>;
}

impl CodpulaOps for CubeBackend {
    fn codpula_leaf_normalizers(
        logits: FloatTensor<Self>, log_pi: FloatTensor<Self>, valid: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        crate::codpula::leaf_forward(logits, log_pi, valid)
    }
    fn codpula_leaf_backward(
        logits: FloatTensor<Self>, log_pi: FloatTensor<Self>, valid: FloatTensor<Self>,
        normalizers: FloatTensor<Self>, upstream: FloatTensor<Self>,
        need_logits: bool, need_pi: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        crate::codpula::leaf_backward(logits, log_pi, valid, normalizers, upstream, need_logits, need_pi)
    }
    fn codpula_partition(
        log_evidence: FloatTensor<Self>, log_transition: FloatTensor<Self>,
        log_root: FloatTensor<Self>, valid: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        crate::codpula::chain_forward(log_evidence, log_transition, log_root, valid, false).0
    }
    fn codpula_chain(
        log_evidence: FloatTensor<Self>, log_transition: FloatTensor<Self>,
        log_root: FloatTensor<Self>, valid: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        crate::codpula::chain_forward(log_evidence, log_transition, log_root, valid, true)
    }
    fn codpula_chain_backward(
        gamma: FloatTensor<Self>, summed_xi: FloatTensor<Self>, upstream: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        crate::codpula::chain_backward(gamma, summed_xi, upstream)
    }
    fn codpula_readout(
        logits: FloatTensor<Self>, log_pi: FloatTensor<Self>, normalizers: FloatTensor<Self>,
        gamma: FloatTensor<Self>, tokens: FloatTensor<Self>, observed: FloatTensor<Self>,
        valid: FloatTensor<Self>, counts: bool,
    ) -> FloatTensor<Self> {
        crate::codpula::readout(logits, log_pi, normalizers, gamma, tokens, observed, valid, counts)
    }
}

#[derive(Debug)]
struct ChainBackward;
impl<B: Backend + CodpulaOps> Backward<B, 3> for ChainBackward {
    type State = (FloatTensor<B>, FloatTensor<B>);
    fn backward(self, ops: Ops<Self::State, 3>, grads: &mut Gradients, _: &mut Checkpointer) {
        let upstream = grads.consume::<B>(&ops.node);
        let (gamma, edges) = ops.state;
        let (de, da, dr) = B::codpula_chain_backward(gamma, edges, upstream);
        for (node, gradient) in ops.parents.into_iter().zip([de, da, dr]) {
            if let Some(node) = node { grads.register::<B>(node.id, gradient); }
        }
    }
}

impl<B: Backend + CodpulaOps, C: CheckpointStrategy> CodpulaOps for Autodiff<B, C> {
    fn codpula_leaf_normalizers(
        logits: FloatTensor<Self>, log_pi: FloatTensor<Self>, valid: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct LeafBackward;
        impl<B: Backend + CodpulaOps> Backward<B, 2> for LeafBackward {
            type State = (FloatTensor<B>, FloatTensor<B>, FloatTensor<B>, FloatTensor<B>);
            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, _: &mut Checkpointer) {
                let upstream = grads.consume::<B>(&ops.node);
                let (logits, pi, valid, normalizers) = ops.state;
                let (dx, dp) = B::codpula_leaf_backward(logits, pi, valid, normalizers, upstream,
                    ops.parents[0].is_some(), ops.parents[1].is_some());
                for (node, gradient) in ops.parents.into_iter().zip([dx, dp]) {
                    if let Some(node) = node { grads.register::<B>(node.id, gradient); }
                }
            }
        }
        let parents = [logits.node(), log_pi.node()];
        let (logits, pi, valid) = (logits.into_primitive(), log_pi.into_primitive(), valid.into_primitive());
        let out = B::codpula_leaf_normalizers(logits.clone(), pi.clone(), valid.clone());
        match LeafBackward.prepare::<C>(parents).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish((logits, pi, valid, out.clone()), out),
            OpsKind::UnTracked(prep) => prep.finish(out),
        }
    }
    fn codpula_leaf_backward(
        logits: FloatTensor<Self>, log_pi: FloatTensor<Self>, valid: FloatTensor<Self>,
        normalizers: FloatTensor<Self>, upstream: FloatTensor<Self>,
        need_logits: bool, need_pi: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (dx, dp) = B::codpula_leaf_backward(logits.into_primitive(), log_pi.into_primitive(),
            valid.into_primitive(), normalizers.into_primitive(), upstream.into_primitive(), need_logits, need_pi);
        (Self::from_inner(dx), Self::from_inner(dp))
    }
    fn codpula_partition(
        log_evidence: FloatTensor<Self>, log_transition: FloatTensor<Self>,
        log_root: FloatTensor<Self>, valid: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let parents = [log_evidence.node(), log_transition.node(), log_root.node()];
        let (e, a, r, valid) = (log_evidence.into_primitive(), log_transition.into_primitive(),
            log_root.into_primitive(), valid.into_primitive());
        match ChainBackward.prepare::<C>(parents).compute_bound().stateful() {
            OpsKind::Tracked(prep) => {
                let (z, gamma, edges) = B::codpula_chain(e, a, r, valid);
                prep.finish((gamma, edges), z)
            }
            OpsKind::UnTracked(prep) => prep.finish(B::codpula_partition(e, a, r, valid)),
        }
    }
    fn codpula_chain(
        log_evidence: FloatTensor<Self>, log_transition: FloatTensor<Self>,
        log_root: FloatTensor<Self>, valid: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let parents = [log_evidence.node(), log_transition.node(), log_root.node()];
        let (z, gamma, edges) = B::codpula_chain(log_evidence.into_primitive(), log_transition.into_primitive(),
            log_root.into_primitive(), valid.into_primitive());
        let gamma_ad = Self::from_inner(gamma.clone());
        let edges_ad = Self::from_inner(edges.clone());
        let z = match ChainBackward.prepare::<C>(parents).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish((gamma, edges), z),
            OpsKind::UnTracked(prep) => prep.finish(z),
        };
        (z, gamma_ad, edges_ad)
    }
    fn codpula_chain_backward(
        gamma: FloatTensor<Self>, summed_xi: FloatTensor<Self>, upstream: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let (de, da, dr) = B::codpula_chain_backward(gamma.into_primitive(), summed_xi.into_primitive(), upstream.into_primitive());
        (Self::from_inner(de), Self::from_inner(da), Self::from_inner(dr))
    }
    fn codpula_readout(
        logits: FloatTensor<Self>, log_pi: FloatTensor<Self>, normalizers: FloatTensor<Self>,
        gamma: FloatTensor<Self>, tokens: FloatTensor<Self>, observed: FloatTensor<Self>,
        valid: FloatTensor<Self>, counts: bool,
    ) -> FloatTensor<Self> {
        Self::from_inner(B::codpula_readout(logits.into_primitive(), log_pi.into_primitive(),
            normalizers.into_primitive(), gamma.into_primitive(), tokens.into_primitive(),
            observed.into_primitive(), valid.into_primitive(), counts))
    }
}

/// State-specific log-sum-exp over the complete vocabulary, [C,L,V] + [K,V] -> [C,L,K].
/// Positive finite rows are the caller's responsibility. Padding is not read and yields zero.
/// First-order VJPs cover logits and shared log_pi; valid is nondifferentiable.
pub fn codpula_leaf_normalizers(logits: Tensor<3>, log_pi: Tensor<2>, valid: Tensor<2>) -> Tensor<3> {
    Tensor::from_dispatch(<Dispatch as CodpulaOps>::codpula_leaf_normalizers(
        logits.into_dispatch(), log_pi.into_dispatch(), valid.into_dispatch()))
}

/// Direct analytic VJP for arbitrary [C,L,K] upstream weights; output [C,L,V], [K,V].
/// Returned derivatives are detached; higher-order differentiation is not implemented.
pub fn codpula_leaf_backward(
    logits: Tensor<3>, log_pi: Tensor<2>, valid: Tensor<2>, normalizers: Tensor<3>, upstream: Tensor<3>,
) -> (Tensor<3>, Tensor<2>) {
    let (dx, dp) = <Dispatch as CodpulaOps>::codpula_leaf_backward(logits.into_dispatch(), log_pi.into_dispatch(),
        valid.into_dispatch(), normalizers.into_dispatch(), upstream.into_dispatch(), true, true);
    (Tensor::from_dispatch(dx), Tensor::from_dispatch(dp))
}

/// Log evidence mass for each chain. Shapes: evidence [C,L,K], transition [C,K,K], root [K], valid [C,L].
/// The root is not modulated; transitions are shared across every active edge of a chain.
/// VJPs cover all three log-weight tensors, including the sum over tied transition/root occurrences.
/// K<=32, L<=16. Valid must be a prefix of binary ones; all-zero rows are inactive, with logZ=0.
pub fn codpula_log_partition(
    log_evidence: Tensor<3>, log_transition: Tensor<3>, log_root: Tensor<1>, valid: Tensor<2>,
) -> Tensor<1> {
    Tensor::from_dispatch(<Dispatch as CodpulaOps>::codpula_partition(log_evidence.into_dispatch(),
        log_transition.into_dispatch(), log_root.into_dispatch(), valid.into_dispatch()))
}

/// Detached logZ [C], state posterior [C,L,K], and posterior edge counts summed over time [C,K,K].
/// The same log-space forward/backward kernel implements this and the differentiable log partition.
/// Invalid evidence/masks produce nonfinite logZ, never a substitute conditional distribution.
pub fn codpula_chain_statistics(
    log_evidence: Tensor<3>, log_transition: Tensor<3>, log_root: Tensor<1>, valid: Tensor<2>,
) -> (Tensor<1>, Tensor<3>, Tensor<3>) {
    let (z, gamma, edges) = <Dispatch as CodpulaOps>::codpula_chain(log_evidence.detach().into_dispatch(),
        log_transition.detach().into_dispatch(), log_root.detach().into_dispatch(), valid.detach().into_dispatch());
    (Tensor::from_dispatch(z), Tensor::from_dispatch(gamma), Tensor::from_dispatch(edges))
}

/// Direct log-partition VJP: gamma, summed_xi and upstream [C] -> evidence, transition and shared root derivatives.
pub fn codpula_chain_backward(
    gamma: Tensor<3>, summed_xi: Tensor<3>, upstream: Tensor<1>,
) -> (Tensor<3>, Tensor<3>, Tensor<1>) {
    let (de, da, dr) = <Dispatch as CodpulaOps>::codpula_chain_backward(gamma.into_dispatch(),
        summed_xi.into_dispatch(), upstream.into_dispatch());
    (Tensor::from_dispatch(de), Tensor::from_dispatch(da), Tensor::from_dispatch(dr))
}

/// Detached token marginals [C,L,V], including observed point masses and zero padding.
/// All tensors share the float dtype; tokens [C,L] contain integral IDs, masks are binary.
pub fn codpula_token_marginals(
    logits: Tensor<3>, log_pi: Tensor<2>, normalizers: Tensor<3>, gamma: Tensor<3>,
    tokens: Tensor<2>, observed: Tensor<2>, valid: Tensor<2>,
) -> Tensor<3> {
    Tensor::from_dispatch(<Dispatch as CodpulaOps>::codpula_readout(logits.into_dispatch(), log_pi.into_dispatch(),
        normalizers.into_dispatch(), gamma.into_dispatch(), tokens.into_dispatch(), observed.into_dispatch(), valid.into_dispatch(), false))
}

/// Detached posterior emission counts summed over tied occurrences, [K,V], without [C,L,K,V] storage.
/// Fixed positions contribute gamma at their token; unresolved positions contribute gamma times the normalized leaf.
pub fn codpula_emission_flows(
    logits: Tensor<3>, log_pi: Tensor<2>, normalizers: Tensor<3>, gamma: Tensor<3>,
    tokens: Tensor<2>, observed: Tensor<2>, valid: Tensor<2>,
) -> Tensor<2> {
    Tensor::from_dispatch(<Dispatch as CodpulaOps>::codpula_readout(logits.into_dispatch(), log_pi.into_dispatch(),
        normalizers.into_dispatch(), gamma.into_dispatch(), tokens.into_dispatch(), observed.into_dispatch(), valid.into_dispatch(), true))
}
