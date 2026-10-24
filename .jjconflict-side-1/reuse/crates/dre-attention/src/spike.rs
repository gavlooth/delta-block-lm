use burn::backend::{Backend, backend_extension, autodiff::{Autodiff, checkpoint::{base::Checkpointer, strategy::CheckpointStrategy}, grads::Gradients, ops::{Backward, Ops, OpsKind}}, cubecl::dtype_to_storage_type, fusion::custom::TensorSpec, ops::FloatTensorOps, tensor::{FloatTensor, IntTensor}, wgpu::{CubeBackend, CubeTensor}};
use burn::cubecl::prelude::*;
use cubecl::frontend::Tensor as CubeTensorView;
use burn::prelude::Shape;
use burn::backend::TensorMetadata;

fn spike_meta(a: &TensorSpec, _b: &TensorSpec, rows: &TensorSpec) -> (TensorSpec, TensorSpec) {
    (a.clone(), TensorSpec::new(rows.shape.clone(), a.dtype))
}

#[backend_extension(Cube, Autodiff, Fusion)]
pub trait SpikeOps: Backend {
    #[fusion(meta = spike_meta)]
    fn spike(a: FloatTensor<Self>, b: FloatTensor<Self>, scale: IntTensor<Self>) -> (FloatTensor<Self>, FloatTensor<Self>);
}

#[cube(launch)]
fn spike_kernel<F: Float, I: Int>(a: &CubeTensorView<F>, b: &CubeTensorView<F>, scale: &CubeTensorView<I>, out: &mut CubeTensorView<F>, sums: &mut CubeTensorView<F>, #[define(F, I)] _dtypes: [ElemType; 2]) {
    let row = ABSOLUTE_POS;
    if row >= a.shape(0) { terminate!(); }
    let out_stride_0 = out.stride(0);
    let width = a.shape(1);
    let out_stride_1 = out.stride(1);
    let factor = F::cast_from(scale[row * scale.stride(0)]);
    let mut total = F::new(0.0f32);
    let mut col = 0usize;
    while col < width {
        let ai = row * a.stride(0) + col * a.stride(1);
        let bi = row * b.stride(0) + col * b.stride(1);
        let value = a[ai] + b[bi] * factor;
        out[row * out_stride_0 + col * out_stride_1] = value;
        total += value;
        col += 1;
    }
    sums[row] = total;
}

fn run_spike(a: CubeTensor, b: CubeTensor, scale: CubeTensor) -> (CubeTensor, CubeTensor) {
    let [rows, _width] = a.shape().dims::<2>();
    let client = a.client.clone();
    let device = a.device.clone();
    let dtype = a.dtype;
    let int_dtype = scale.dtype;
    let output = CubeBackend::float_zeros(a.shape(), &device, dtype.into());
    let sums = CubeBackend::float_zeros(Shape::new([rows]), &device, dtype.into());
    let count = CubeCount::new_1d(rows as u32);
    let dim = CubeDim::new(&client, rows);
    spike_kernel::launch(&client, count, dim, a.into_tensor_arg(), b.into_tensor_arg(), scale.into_tensor_arg(), output.clone().into_tensor_arg(), sums.clone().into_tensor_arg(), [dtype_to_storage_type(dtype), dtype_to_storage_type(int_dtype)]);
    (output, sums)
}

impl SpikeOps for CubeBackend {
    fn spike(a: CubeTensor, b: CubeTensor, scale: CubeTensor) -> (CubeTensor, CubeTensor) { run_spike(a, b, scale) }
}

impl<B: Backend + SpikeOps, C: CheckpointStrategy> SpikeOps for Autodiff<B, C> {
    fn spike(a: FloatTensor<Self>, b: FloatTensor<Self>, scale: IntTensor<Self>) -> (FloatTensor<Self>, FloatTensor<Self>) {
        #[derive(Debug)] struct SpikeBackward;
        impl<B: Backend> Backward<B, 2> for SpikeBackward {
            type State = (B::FloatTensorPrimitive, B::IntTensorPrimitive);
            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
                let [na, nb] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);
                let (a, scale) = ops.state;
                let dims = a.shape().dims::<2>();
                let dtype = burn::backend::get_or_init_device_settings::<B>(&a.device()).float_dtype;
                let factor = B::float_expand(B::float_reshape(B::int_into_float(scale, dtype), Shape::new([dims[0], 1])), Shape::new([dims[0], dims[1]]));
                if let Some(node) = na { grads.register::<B>(node.id, grad.clone()); }
                if let Some(node) = nb { grads.register::<B>(node.id, B::float_mul(grad, factor)); }
            }
        }
        #[derive(Debug)] struct SpikeSumBackward;
        impl<B: Backend> Backward<B, 2> for SpikeSumBackward {
            type State = (B::FloatTensorPrimitive, B::IntTensorPrimitive);
            fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
                let [na, nb] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);
                let (a, scale) = ops.state;
                let dims = a.shape().dims::<2>();
                let dtype = burn::backend::get_or_init_device_settings::<B>(&a.device()).float_dtype;
                let grad = B::float_expand(B::float_reshape(grad, Shape::new([dims[0], 1])), Shape::new([dims[0], dims[1]]));
                let factor = B::float_expand(B::float_reshape(B::int_into_float(scale, dtype), Shape::new([dims[0], 1])), Shape::new([dims[0], dims[1]]));
                if let Some(node) = na { grads.register::<B>(node.id, grad.clone()); }
                if let Some(node) = nb { grads.register::<B>(node.id, B::float_mul(grad, factor)); }
            }
        }
        let ga = a.node(); let gb = b.node(); let ga_sum = a.node(); let gb_sum = b.node();
        let (a, b) = (a.into_primitive(), b.into_primitive());
        let (out, row_sum) = B::spike(a.clone(), b, scale.clone());
        let output = match SpikeBackward.prepare::<C>([ga, gb]).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish((a.clone(), scale.clone()), out),
            OpsKind::UnTracked(prep) => prep.finish(out),
        };
        let sums = match SpikeSumBackward.prepare::<C>([ga_sum, gb_sum]).compute_bound().stateful() {
            OpsKind::Tracked(prep) => prep.finish((a, scale), row_sum),
            OpsKind::UnTracked(prep) => prep.finish(row_sum),
        };
        (output, sums)
    }
}

/// Runs a scaled row addition and also returns its row sums.
pub fn spike(a: burn::Tensor<2>, b: burn::Tensor<2>, scale: burn::Tensor<1, burn::tensor::Int>) -> (burn::Tensor<2>, burn::Tensor<1>) {
    let (output, sums) = <burn::backend::Dispatch as SpikeOps>::spike(a.into_dispatch(), b.into_dispatch(), scale.into_dispatch());
    (burn::Tensor::from_dispatch(output), burn::Tensor::from_dispatch(sums))
}
