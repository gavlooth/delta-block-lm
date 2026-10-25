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

#[backend_extension(Cube, Autodiff)]
pub trait LngramOps: Backend {
    fn lngram_codes(z: FloatTensor<Self>, bits: usize) -> FloatTensor<Self>;
    fn lngram_table_lookup(table: FloatTensor<Self>, indices: IntTensor<Self>) -> FloatTensor<Self>;
    fn lngram_table_lookup_backward(
        table: FloatTensor<Self>,
        indices: IntTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
    fn lngram_memory(
        z: FloatTensor<Self>,
        codes: FloatTensor<Self>,
        table: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        docs: FloatTensor<Self>,
        bits: usize,
        order: usize,
        mode: usize,
        temperature: f32,
        scale: f32,
    ) -> FloatTensor<Self>;
    fn lngram_memory_backward(
        z: FloatTensor<Self>,
        table: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        docs: FloatTensor<Self>,
        codes: FloatTensor<Self>,
        dy: FloatTensor<Self>,
        bits: usize,
        order: usize,
        mode: usize,
        temperature: f32,
        scale: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);
}
impl LngramOps for CubeBackend {
    fn lngram_codes(z: FloatTensor<Self>, bits: usize) -> FloatTensor<Self> {
        crate::lngram::codes(z, bits)
    }
    fn lngram_table_lookup(table: FloatTensor<Self>, indices: IntTensor<Self>) -> FloatTensor<Self> {
        crate::lngram::table_lookup(table, indices)
    }
    fn lngram_table_lookup_backward(
        table: FloatTensor<Self>,
        indices: IntTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        crate::lngram::table_lookup_backward(table, indices, dy)
    }
    fn lngram_memory(
        z: FloatTensor<Self>,
        codes: FloatTensor<Self>,
        table: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        docs: FloatTensor<Self>,
        bits: usize,
        order: usize,
        _mode: usize,
        _temperature: f32,
        _scale: f32,
    ) -> FloatTensor<Self> {
        crate::lngram::forward(z, codes, table, valid, docs, bits, order)
    }
    fn lngram_memory_backward(
        z: FloatTensor<Self>,
        table: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        docs: FloatTensor<Self>,
        codes: FloatTensor<Self>,
        dy: FloatTensor<Self>,
        bits: usize,
        order: usize,
        mode: usize,
        temperature: f32,
        scale: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        crate::lngram::backward(
            z, table, valid, docs, codes, dy, bits, order, mode, temperature, scale,
        )
    }
}
impl<B: Backend + LngramOps, C: CheckpointStrategy> LngramOps for Autodiff<B, C> {
    fn lngram_table_lookup(
        table: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct TableLookupBackward;
        impl<B: Backend + LngramOps> Backward<B, 1> for TableLookupBackward {
            type State = (FloatTensor<B>, IntTensor<B>);
            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _: &mut Checkpointer,
            ) {
                let dy = grads.consume::<B>(&ops.node);
                let (table, indices) = ops.state;
                let [parent] = ops.parents;
                if let Some(parent) = parent {
                    let gradient = B::lngram_table_lookup_backward(table, indices, dy);
                    grads.register::<B>(parent.id, gradient);
                }
            }
        }
        let parents = [table.node()];
        let table = table.into_primitive();
        let y = B::lngram_table_lookup(table.clone(), indices.clone());
        match TableLookupBackward
            .prepare::<C>(parents)
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish((table, indices), y),
            OpsKind::UnTracked(prep) => prep.finish(y),
        }
    }
    fn lngram_table_lookup_backward(
        table: FloatTensor<Self>,
        indices: IntTensor<Self>,
        dy: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        Self::from_inner(B::lngram_table_lookup_backward(
            table.into_primitive(),
            indices,
            dy.into_primitive(),
        ))
    }
    fn lngram_codes(z: FloatTensor<Self>, bits: usize) -> FloatTensor<Self> {
        Self::from_inner(B::lngram_codes(z.into_primitive(), bits))
    }
    fn lngram_memory(
        z: FloatTensor<Self>,
        codes: FloatTensor<Self>,
        table: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        docs: FloatTensor<Self>,
        bits: usize,
        order: usize,
        mode: usize,
        temperature: f32,
        scale: f32,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct MemoryBackward;
        impl<B: Backend + LngramOps> Backward<B, 2> for MemoryBackward {
            type State = (
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                usize,
                usize,
                usize,
                f32,
                f32,
            );
            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                _: &mut Checkpointer,
            ) {
                let dy = grads.consume::<B>(&ops.node);
                let (z, table, valid, docs, codes, bits, order, mode, temperature, scale) =
                    ops.state;
                let (dz, dt) = B::lngram_memory_backward(
                    z,
                    table,
                    valid,
                    docs,
                    codes,
                    dy,
                    bits,
                    order,
                    mode,
                    temperature,
                    scale,
                );
                for (node, gradient) in ops.parents.into_iter().zip([dz, dt]) {
                    if let Some(node) = node {
                        grads.register::<B>(node.id, gradient);
                    }
                }
            }
        }
        let parents = [z.node(), table.node()];
        let (z, codes, table, valid, docs) = (
            z.into_primitive(),
            codes.into_primitive(),
            table.into_primitive(),
            valid.into_primitive(),
            docs.into_primitive(),
        );
        let y = B::lngram_memory(
            z.clone(),
            codes.clone(),
            table.clone(),
            valid.clone(),
            docs.clone(),
            bits,
            order,
            mode,
            temperature,
            scale,
        );
        match MemoryBackward
            .prepare::<C>(parents)
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                (
                    z,
                    table,
                    valid,
                    docs,
                    codes,
                    bits,
                    order,
                    mode,
                    temperature,
                    scale,
                ),
                y,
            ),
            OpsKind::UnTracked(prep) => prep.finish(y),
        }
    }
    fn lngram_memory_backward(
        z: FloatTensor<Self>,
        table: FloatTensor<Self>,
        valid: FloatTensor<Self>,
        docs: FloatTensor<Self>,
        codes: FloatTensor<Self>,
        dy: FloatTensor<Self>,
        bits: usize,
        order: usize,
        mode: usize,
        temperature: f32,
        scale: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (dz, dt) = B::lngram_memory_backward(
            z.into_primitive(),
            table.into_primitive(),
            valid.into_primitive(),
            docs.into_primitive(),
            codes.into_primitive(),
            dy.into_primitive(),
            bits,
            order,
            mode,
            temperature,
            scale,
        );
        (Self::from_inner(dz), Self::from_inner(dt))
    }
}
/// One order's exact causal memory [B,T,R*memory_dim], with analytic counterfactual logit VJP.
/// Metadata is nondifferentiable. Table VJPs contain only actual indexed rows.
/// `codes = lngram_codes(z, bits)` is nondifferentiable and shared across orders.
pub fn lngram_memory(
    z: Tensor<3>,
    codes: Tensor<3>,
    table: Tensor<2>,
    valid: Tensor<2>,
    docs: Tensor<2>,
    bits: usize,
    order: usize,
    local_expectation: bool,
    temperature: f64,
    scale: f64,
) -> Tensor<3> {
    assert!(
        !codes.is_require_grad() && !valid.is_require_grad() && !docs.is_require_grad(),
        "Lngram codes and masks are metadata"
    );
    Tensor::from_dispatch(<Dispatch as LngramOps>::lngram_memory(
        z.into_dispatch(),
        codes.into_dispatch(),
        table.into_dispatch(),
        valid.into_dispatch(),
        docs.into_dispatch(),
        bits,
        order,
        usize::from(local_expectation),
        temperature as f32,
        scale as f32,
    ))
}

/// Exact hard route symbols, shared by all orders; no derivative through the hard bits.
pub fn lngram_codes(z: Tensor<3>, bits: usize) -> Tensor<3> {
    Tensor::from_dispatch(<Dispatch as LngramOps>::lngram_codes(
        z.into_dispatch(),
        bits,
    ))
}
/// Route-wise integer table lookup: `table [R,rows,M]`, `indices [R,P]`, output `[R,P,M]`.
/// Every index must be in `0..rows`. GPU backward uses one owner per route/channel, scans
/// positions in ascending order, and writes the dense table gradient without atomics. This
/// costs O(R*rows*M + P*R*M) work and no expanded `[P,rows,M]` temporary.
pub fn lngram_table_lookup(
    table: Tensor<3>,
    indices: Tensor<2, burn::prelude::Int>,
) -> Tensor<3> {
    Tensor::from_dispatch(<Dispatch as LngramOps>::lngram_table_lookup(
        table.into_dispatch(),
        indices.into_dispatch(),
    ))
}
