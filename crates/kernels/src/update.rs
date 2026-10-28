//! Allocation-bounded square reduction and exact (unreassociated) AdamW element update.
use burn::{
    backend::{TensorMetadata, ops::FloatTensorOps},
    prelude::Shape,
    tensor::DType,
};
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use cubecl::{frontend::Tensor as View, prelude::*};

#[cube]
fn index(x: &View<f32>, linear: usize) -> usize {
    let mut rest = linear;
    let mut offset = 0usize;
    for reverse in 0..x.rank() {
        let axis = x.rank() - reverse - 1;
        offset += (rest % x.shape(axis)) * x.stride(axis);
        rest /= x.shape(axis);
    }
    offset
}
#[cube(launch)]
fn square_kernel(x: &View<f32>, partial: &mut View<f32>) {
    let group = CUBE_POS as usize;
    let lane = UNIT_POS as usize;
    let mut sum = 0.0f32;
    let mut i = group * 256 + lane;
    while i < x.len() {
        let value = x[index(x, i)];
        sum += value * value;
        i += partial.len() * 256;
    }
    let mut shared = Shared::<[f32]>::new_slice(256usize);
    shared[lane] = sum;
    sync_cube();
    let mut stride = 128usize;
    while stride > 0 {
        if lane < stride {
            shared[lane] += shared[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 {
        partial[group] = shared[0];
    }
}
#[cube(launch)]
fn adam_kernel(
    parameter: &View<f32>,
    gradient: &View<f32>,
    first: &View<f32>,
    second: &View<f32>,
    output: &mut View<f32>,
    next_first: &mut View<f32>,
    next_second: &mut View<f32>,
    #[comptime] initialized: bool,
    correction1: f32,
    correction2: f32,
    decay: f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < parameter.len() {
        // The module boundary already applied the one global clipping coefficient.
        let g = gradient[index(gradient, i)];
        let mut m = g * (1.0f32 - beta1);
        let mut v = (g * g) * (1.0f32 - beta2);
        if initialized {
            m = first[index(first, i)] * beta1 + m;
            v = second[index(second, i)] * beta2 + v;
        }
        next_first[i] = m;
        next_second[i] = v;
        let delta = (m / correction1) / (f32::sqrt(v / correction2) + epsilon);
        output[i] = parameter[index(parameter, i)] * decay - delta * lr;
    }
}
pub(crate) fn validate(tensors: &[&CubeTensor]) {
    for x in tensors {
        assert_eq!(x.dtype, DType::F32, "update/loss kernels require f32");
        assert_eq!(x.device, tensors[0].device, "device mismatch");
        let mut span = 1usize;
        for (&dim, &stride) in x
            .meta
            .shape()
            .as_slice()
            .iter()
            .zip(x.meta.strides().iter())
        {
            assert!(dim > 0 && stride <= u32::MAX as usize);
            span = (dim - 1)
                .checked_mul(stride)
                .and_then(|n| span.checked_add(n))
                .expect("address overflow");
        }
        assert!(
            span <= u32::MAX as usize && x.handle.size() / 4 <= u32::MAX as u64,
            "32-bit addressing exceeded"
        );
    }
}
pub(crate) fn empty(x: &CubeTensor, shape: impl Into<Shape>) -> CubeTensor {
    let shape = shape.into();
    assert!(shape.num_elements() <= u32::MAX as usize);
    CubeBackend::float_empty(shape, &x.device, DType::F32.into())
}
pub(crate) fn square_sum(x: CubeTensor) -> CubeTensor {
    validate(&[&x]);
    let groups = x.shape().num_elements().div_ceil(256).clamp(1, 64);
    let partial = empty(&x, [groups]);
    square_kernel::launch(
        &x.client,
        CubeCount::Static(groups as u32, 1, 1),
        CubeDim::new_1d(256),
        x.clone().into_tensor_arg(),
        partial.clone().into_tensor_arg(),
    );
    CubeBackend::float_sum(partial)
}
pub(crate) fn adam(
    parameter: CubeTensor,
    gradient: CubeTensor,
    first: CubeTensor,
    second: CubeTensor,
    initialized: bool,
    time: usize,
    lr: f64,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
) -> (CubeTensor, CubeTensor, CubeTensor) {
    validate(&[&parameter, &gradient, &first, &second]);
    assert_eq!(parameter.shape(), gradient.shape());
    if initialized {
        assert_eq!(parameter.shape(), first.shape());
        assert_eq!(parameter.shape(), second.shape());
    }
    // Reuse only uniquely owned, contiguous buffers: each lane reads/writes its own element.
    // Shared model clones, record views and strided inputs retain independent output storage.
    let reuse_parameter = parameter.can_mut() && parameter.is_contiguous();
    let reuse_first = initialized && first.can_mut() && first.is_contiguous();
    let reuse_second = initialized && second.can_mut() && second.is_contiguous();
    let out = if reuse_parameter {
        parameter.clone()
    } else {
        empty(&parameter, parameter.shape())
    };
    let m = if reuse_first {
        first.clone()
    } else {
        empty(&parameter, parameter.shape())
    };
    let v = if reuse_second {
        second.clone()
    } else {
        empty(&parameter, parameter.shape())
    };
    let out_arg = if reuse_parameter {
        parameter.as_tensor_alias(0)
    } else {
        out.clone().into_tensor_arg()
    };
    let m_arg = if reuse_first {
        first.as_tensor_alias(2)
    } else {
        m.clone().into_tensor_arg()
    };
    let v_arg = if reuse_second {
        second.as_tensor_alias(3)
    } else {
        v.clone().into_tensor_arg()
    };
    adam_kernel::launch(
        &parameter.client,
        CubeCount::Static(parameter.shape().num_elements().div_ceil(256) as u32, 1, 1),
        CubeDim::new_1d(256),
        parameter.clone().into_tensor_arg(),
        gradient.into_tensor_arg(),
        first.into_tensor_arg(),
        second.into_tensor_arg(),
        out_arg,
        m_arg,
        v_arg,
        initialized,
        1f32 - beta1.powi(time as i32),
        1f32 - beta2.powi(time as i32),
        (1.0 - lr * f64::from(weight_decay)) as f32,
        lr as f32,
        beta1,
        beta2,
        epsilon,
    );
    (out, m, v)
}
