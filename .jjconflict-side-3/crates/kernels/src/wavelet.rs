//! Fixed-filter causal dilated depthwise smoothing; metadata carries no gradients.
use burn::{
    backend::{TensorMetadata, ops::FloatTensorOps},
    tensor::DType,
};
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use cubecl::{frontend::Tensor as View, prelude::*};
#[cube]
fn coefficient(k: usize) -> f32 {
    if k == 0 {
        1.88585787961206876e-02f32
    } else if k == 1 {
        1.33061091396920905e-01f32
    } else if k == 2 {
        3.72787535743233367e-01f32
    } else if k == 3 {
        4.86814055366819876e-01f32
    } else if k == 4 {
        1.98818870884508697e-01f32
    } else if k == 5 {
        -1.76668100897056285e-01f32
    } else if k == 6 {
        -1.38554939360483159e-01f32
    } else if k == 7 {
        9.00637242666966487e-02f32
    } else if k == 8 {
        6.58014935505349952e-02f32
    } else if k == 9 {
        -5.04832855983897086e-02f32
    } else if k == 10 {
        -2.08296240438008047e-02f32
    } else if k == 11 {
        2.34849070486985584e-02f32
    } else if k == 12 {
        2.55021848390723843e-03f32
    } else if k == 13 {
        -7.58950116792824880e-03f32
    } else if k == 14 {
        9.86662682481602394e-04f32
    } else if k == 15 {
        1.40884329509733796e-03f32
    } else if k == 16 {
        -4.84973919928205423e-04f32
    } else if k == 17 {
        -8.23545030453889725e-05f32
    } else if k == 18 {
        6.61771834255533786e-05f32
    } else {
        -9.37920781375020288e-06f32
    }
}

#[cube(launch)]
fn kernel(
    x: &View<f32>,
    valid: &View<f32>,
    docs: &View<f32>,
    y: &mut View<f32>,
    #[comptime] dilation: usize,
    #[comptime] transpose: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i >= y.len() {
        terminate!();
    }
    let c = i % x.shape(2);
    let t = i / x.shape(2) % x.shape(1);
    let b = i / (x.shape(2) * x.shape(1));
    let mut sum = 0.0f32;
    let dest = b * docs.stride(0) + t * docs.stride(1);
    if valid[b * valid.stride(0) + t * valid.stride(1)] > 0.5f32 {
        #[unroll]
        for k in 0..20 {
            let lag = k * dilation;
            let mut source = t;
            let mut inside = false;
            if transpose {
                if lag < x.shape(1) - t {
                    source = t + lag;
                    inside = true;
                }
            } else {
                if lag <= t {
                    source = t - lag;
                    inside = true;
                }
            }
            if inside {
                if docs[b * docs.stride(0) + source * docs.stride(1)] == docs[dest]
                    && valid[b * valid.stride(0) + source * valid.stride(1)] > 0.5f32
                {
                    sum += x[b * x.stride(0) + source * x.stride(1) + c * x.stride(2)]
                        * coefficient(k);
                }
            }
        }
    }
    y[i] = sum;
}
pub(crate) fn run(
    x: CubeTensor,
    valid: CubeTensor,
    docs: CubeTensor,
    dilation: usize,
    transpose: bool,
) -> CubeTensor {
    let [b, t, c] = x.shape().dims::<3>();
    assert!(b > 0 && t > 0 && c > 0 && dilation > 0 && dilation <= 128);
    assert_eq!(valid.shape().dims::<2>(), [b, t]);
    assert_eq!(docs.shape().dims::<2>(), [b, t]);
    for a in [&x, &valid, &docs] {
        assert_eq!(a.dtype, DType::F32);
        assert_eq!(a.device, x.device);
        let mut span = 1usize;
        for (&d, &s) in a
            .meta
            .shape()
            .as_slice()
            .iter()
            .zip(a.meta.strides().iter())
        {
            span = (d - 1)
                .checked_mul(s)
                .and_then(|n| span.checked_add(n))
                .expect("wavelet address overflow");
        }
        assert!(span <= u32::MAX as usize && a.handle.size() / 4 <= u32::MAX as u64);
    }
    let output = CubeBackend::float_empty(x.shape(), &x.device, DType::F32.into());
    let groups = x.shape().num_elements().div_ceil(64);
    let (mx, my, _) = x.client.properties().hardware.max_cube_count;
    let nx = groups.min(mx.min(65535) as usize).max(1);
    assert!(groups.div_ceil(nx) <= my as usize);
    kernel::launch(
        &x.client,
        CubeCount::Static(nx as u32, groups.div_ceil(nx) as u32, 1),
        CubeDim::new_1d(64),
        x.clone().into_tensor_arg(),
        valid.into_tensor_arg(),
        docs.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        dilation,
        transpose,
    );
    output
}
