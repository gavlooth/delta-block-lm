//! Causal WNet-Dyn port of dre-wavelet. See docs/optimization/wavelet.md.
use crate::profile::{Sections, Unprofiled};
use burn::{
    module::Param,
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::{TensorData, activation::sigmoid},
};
use serde::{Deserialize, Serialize};
mod checks;
mod reference;
pub use checks::numerical_checks;
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WaveletBackend {
    #[default]
    Tensor,
    FusedGpu,
}
impl WaveletBackend {
    pub fn is_tensor(&self) -> bool {
        *self == Self::Tensor
    }
}
pub const FILTER_TAPS: usize = 20;

/// Daubechies db10 low-pass filter `h[0..20]` (sum `√2`), copied from PyWavelets
/// `pywt/_extensions/c/wavelets_coeffs.template.h` (`db10_`) at commit `c9b542b2d2c37d28ed1d9bd49d178d2769608905`.
pub const DB10_LOW_PASS: [f64; FILTER_TAPS] = [
    2.667_005_790_055_555_358_661_744_877_130_858e-2,
    1.881_768_000_776_914_890_208_929_736_790_939e-1,
    5.272_011_889_317_255_864_817_448_279_595_081e-1,
    6.884_590_394_536_035_657_418_717_825_492_358e-1,
    2.811_723_436_605_774_607_487_269_984_455_892e-1,
    -2.498_464_243_273_153_794_161_018_979_207_791e-1,
    -1.959_462_743_773_770_435_042_992_543_190_981e-1,
    1.273_693_403_357_932_600_826_772_332_014_009e-1,
    9.305_736_460_357_235_116_035_228_983_545_273e-2,
    -7.139_414_716_639_708_714_533_609_307_605_064e-2,
    -2.945_753_682_187_581_285_828_323_760_141_839e-2,
    3.321_267_405_934_100_173_976_365_318_215_912e-2,
    3.606_553_566_956_169_655_423_291_417_133_403e-3,
    -1.073_317_548_333_057_504_431_811_410_651_364e-2,
    1.395_351_747_052_901_165_789_318_447_957_707e-3,
    1.992_405_295_185_056_117_158_742_242_640_643e-3,
    -6.858_566_949_597_116_265_613_709_819_265_714e-4,
    -1.164_668_551_292_854_509_514_809_710_258_991e-4,
    9.358_867_032_006_959_133_405_013_034_222_854e-5,
    -1.326_420_289_452_124_481_243_667_531_226_683e-5,
];

/// The smoothing filter `s` of WNet App. B: the db10 low-pass filter normalised to unit sum.
pub fn smoothing_filter() -> [f64; FILTER_TAPS] {
    let sum: f64 = DB10_LOW_PASS.iter().sum();
    DB10_LOW_PASS.map(|tap| tap / sum)
}

/// Zero extension before each document: no future or cross-document reads.
pub fn smooth(
    x: Tensor<3>,
    valid: Tensor<2>,
    docs: Tensor<2>,
    dilation: usize,
    backend: WaveletBackend,
) -> Tensor<3> {
    if backend == WaveletBackend::FusedGpu {
        #[cfg(any(feature = "vulkan", feature = "cuda"))]
        {
            return delta_kernels::wavelet_smooth(x, valid, docs, dilation);
        }
        #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
        {
            panic!("fused wavelet requires GPU feature");
        }
    }
    let [b, t, c] = x.dims();
    let device = x.device();
    let filter = smoothing_filter();
    let positions = Tensor::<1, Int>::arange(0..t as i64, &device);
    let valid_positions = valid.greater_elem(0.5);
    let mut sum = None;
    for (k, weight) in filter.into_iter().enumerate() {
        let lag = k * dilation;
        if lag >= t {
            break;
        }
        let indices = (positions.clone() - lag as i64).clamp_min(0);
        let same = docs.clone().select(1, indices.clone()).equal(docs.clone());
        let in_range = positions
            .clone()
            .greater_equal_elem(lag as i64)
            .reshape([1, t])
            .expand([b, t]);
        let active = same
            .bool_and(in_range)
            .bool_and(valid_positions.clone().select(1, indices.clone()))
            .bool_and(valid_positions.clone());
        let term = x.clone().select(1, indices).mask_fill(
            active.bool_not().unsqueeze_dim::<3>(2).expand([b, t, c]),
            0.,
        ) * weight;
        sum = Some(match sum {
            None => term,
            Some(old) => old + term,
        });
    }
    sum.expect("nonempty time axis")
}
pub fn undecimated_bands(
    v: Tensor<3>,
    valid: Tensor<2>,
    docs: Tensor<2>,
    levels: usize,
    backend: WaveletBackend,
) -> Vec<Tensor<3>> {
    let mut approximation = v;
    let mut details = Vec::with_capacity(levels);
    for level in 0..levels {
        let next = smooth(
            approximation.clone(),
            valid.clone(),
            docs.clone(),
            1 << level,
            backend,
        );
        details.push(approximation - next.clone());
        approximation = next;
    }
    let mut bands = Vec::with_capacity(levels + 1);
    bands.push(approximation);
    bands.extend(details.into_iter().rev());
    bands
}
#[derive(Module, Debug)]
pub struct WaveletDyn {
    pub input: Linear,
    pub gate: Linear,
    pub value: Linear,
    pub output: Linear,
    pub levels: usize,
    pub groups: usize,
    #[module(skip)]
    pub backend: WaveletBackend,
}
impl WaveletDyn {
    pub fn new(
        d: usize,
        levels: usize,
        groups: usize,
        backend: WaveletBackend,
        device: &Device,
    ) -> Self {
        assert!(levels > 0 && levels <= 8 && groups > 0 && d % groups == 0);
        if !backend.is_tensor() {
            crate::chunk::assert_fused_device(device);
        }
        let projection = || LinearConfig::new(d, d).with_bias(false).init(device);
        let mut gate = LinearConfig::new(d, (levels + 1) * groups).init(device);
        let bias = (0..levels + 1)
            .flat_map(|band| {
                let p: f64 = if band == 0 { 0.88 } else { 0.12 };
                std::iter::repeat_n((p / (1. - p)).ln() as f32, groups)
            })
            .collect::<Vec<_>>();
        gate.bias = Some(Param::from_data(
            TensorData::new(bias, [(levels + 1) * groups]),
            device,
        ));
        Self {
            input: projection(),
            gate,
            value: projection(),
            output: projection(),
            levels,
            groups,
            backend,
        }
    }
    pub fn forward(&self, u: Tensor<3>, valid: Tensor<2>, reset: Tensor<2>) -> Tensor<3> {
        self.forward_profiled(u, valid, reset, &mut Unprofiled)
    }
    pub fn forward_profiled<P: Sections>(
        &self,
        u: Tensor<3>,
        valid: Tensor<2>,
        reset: Tensor<2>,
        profile: &mut P,
    ) -> Tensor<3> {
        let [b, t, d] = u.dims();
        let u = profile.section("Wavelet: metadata and padding", || {
            u.mask_fill(
                valid
                    .clone()
                    .lower_elem(0.5)
                    .unsqueeze_dim::<3>(2)
                    .expand([b, t, d]),
                0.,
            )
        });
        let docs = profile.section("Wavelet: document IDs", || {
            (reset * valid.clone()).cumsum(1)
        });
        let projected = profile.section("Wavelet: input projection", || {
            self.input.forward(u.clone())
        });
        let bands = profile.section("Wavelet: causal bands", || {
            undecimated_bands(projected, valid, docs, self.levels, self.backend)
        });
        let gates = profile.section("Wavelet: gates", || {
            sigmoid(self.gate.forward(u.clone())).reshape([b, t, self.levels + 1, self.groups])
        });
        let mixed = profile.section("Wavelet: weighted bands", || {
            bands
                .into_iter()
                .enumerate()
                .map(|(band, values)| {
                    values.reshape([b, t, self.groups, d / self.groups])
                        * gates
                            .clone()
                            .narrow(2, band, 1)
                            .reshape([b, t, self.groups, 1])
                })
                .reduce(|sum, term| sum + term)
                .unwrap()
                .reshape([b, t, d])
        });
        let value = profile.section("Wavelet: value projection", || self.value.forward(u));
        profile.section("Wavelet: output projection", || {
            self.output.forward(mixed * value)
        })
    }
}
