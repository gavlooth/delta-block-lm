//! Dynamic undecimated wavelet token mixer (WNet-Dyn), spec §3.2.
//!
//! Reference: WNet, Salama et al., arXiv:2610.04720v1 — §4.1–4.2 Eqs. (6–7) and App. B Eq. (8)
//! (<https://arxiv.org/html/2610.04720v1>). Equations, every adaptation and the boundary rule are recorded in
//! `docs/references/wavelet.md`.
//!
//! ```text
//! V = U W_i
//! A_0 = V,  A_j = s_{↑2^{j-1}} * A_{j-1},  D_j = A_{j-1} - A_j          (à trous, App. B Eq. 8)
//! bands B = [A_J, D_J, …, D_1], each [B, T, d]
//! G = σ(U W_g + b)  [B, T, J+1, groups], each group broadcast over d/groups channels   (Eq. 6)
//! out = ((Σ_j G_j ⊙ B_j) ⊙ U W_v) W_o                                     (Eq. 7)
//! ```
//!
//! The db10 filter is fixed, as in the paper's WNet-Dyn (a learned filter is only one of its ablations).
//! Every level is computed per document of a packed row with a symmetric boundary extension, so no information
//! crosses document edges and documents of any length (including shorter than the filter) are handled.

use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn::tensor::activation::sigmoid;
use dre_core::{BatchLayout, DocumentLayout};

/// Kind of wavelet transform (spec §15.2 `wavelet.transform`).
pub const TRANSFORM: &str = "undecimated";

/// Activation of the per-token scale gates (spec §15.2 `wavelet.scale_gate_activation`).
pub const SCALE_GATE_ACTIVATION: &str = "sigmoid";

/// Number of taps of the db10 filter.
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

/// `(left, right)` reach of one level with the given dilation: the level reads positions `t - left ..= t + right`.
/// Tap `k` reads `t + right - k * dilation`.
pub fn level_reach(dilation: usize) -> (usize, usize) {
    let span = dilation * (FILTER_TAPS - 1);
    let left = span / 2;
    (left, span - left)
}

/// Position that the symmetric (half-sample) extension of a document of `length` positions maps `offset` to:
/// `x[-1] = x[0]`, `x[length] = x[length - 1]`, repeated with period `2 * length` for offsets farther away.
pub fn symmetric_position(offset: i64, length: usize) -> usize {
    assert!(length > 0, "documents have at least one position");
    let length = length as i64;
    let period = 2 * length;
    let folded = offset.rem_euclid(period);
    (if folded < length { folded } else { period - 1 - folded }) as usize
}

/// Source row of every level and tap, as flat row indices into the `[B * T, d]` view of a batch.
///
/// Entry `(level * FILTER_TAPS + k) * B * T + b * T + t` is the row that tap `k` of level `level` (dilation
/// `2^level`) reads for output position `t` of row `b`. Valid positions read only their own document; padded
/// positions read themselves.
pub fn tap_sources(layout: &DocumentLayout, levels: usize) -> Vec<i32> {
    let rows = layout.batch() * layout.tokens();
    assert!(rows <= i32::MAX as usize, "batch of {rows} positions exceeds the i32 index range");
    let mut sources = vec![0i32; levels * FILTER_TAPS * rows];
    for level in 0..levels {
        let dilation = 1usize << level;
        let (_, right) = level_reach(dilation);
        for k in 0..FILTER_TAPS {
            let block = &mut sources[(level * FILTER_TAPS + k) * rows..][..rows];
            for (row, documents) in layout.documents_per_row().iter().enumerate() {
                let row_start = row * layout.tokens();
                let mut start = 0usize;
                for &length in documents {
                    for t in 0..length {
                        let offset = t as i64 + right as i64 - (k * dilation) as i64;
                        block[row_start + start + t] = (row_start + start + symmetric_position(offset, length)) as i32;
                    }
                    start += length;
                }
                for t in start..layout.tokens() {
                    block[row_start + t] = (row_start + t) as i32;
                }
            }
        }
    }
    sources
}

/// Undecimated (à trous) decomposition of `v: [B, T, d]` into `levels + 1` full-length bands
/// `[A_J, D_J, …, D_1]`, each `[B, T, d]`, computed separately inside every document of `layout`.
///
/// The bands add up to `v` (telescoping). At padded positions every band depends only on `v` at that position.
pub fn undecimated_bands(v: Tensor<3>, layout: &DocumentLayout, levels: usize) -> Vec<Tensor<3>> {
    let [batch, tokens, channels] = v.dims();
    assert_eq!(
        [batch, tokens],
        [layout.batch(), layout.tokens()],
        "input [B, T] does not match the document layout"
    );
    let rows = batch * tokens;
    let filter = smoothing_filter();
    let sources = Tensor::<1, Int>::from_data(
        TensorData::new(tap_sources(layout, levels), [levels * FILTER_TAPS * rows]),
        &v.device(),
    );
    let tap = |input: &Tensor<2>, level: usize, k: usize| {
        let indices = sources.clone().narrow(0, (level * FILTER_TAPS + k) * rows, rows);
        input.clone().select(0, indices).mul_scalar(filter[k])
    };

    let mut approximation = v.reshape([rows, channels]);
    let mut details = Vec::with_capacity(levels);
    for level in 0..levels {
        let smoothed = (1..FILTER_TAPS).fold(tap(&approximation, level, 0), |sum, k| sum + tap(&approximation, level, k));
        details.push(approximation - smoothed.clone());
        approximation = smoothed;
    }

    let mut bands = Vec::with_capacity(levels + 1);
    bands.push(approximation.reshape([batch, tokens, channels]));
    bands.extend(details.into_iter().rev().map(|detail| detail.reshape([batch, tokens, channels])));
    bands
}

/// Configuration of [`WaveletDyn`]; defaults are the spec §3.2 values.
#[derive(Config, Debug)]
pub struct WaveletDynConfig {
    #[config(default = "dre_core::Precision::Fp32")]
    pub precision: dre_core::Precision,
    /// Model width `d`.
    #[config(default = 512)]
    pub d_model: usize,
    /// Decomposition levels `J`; the mixer uses `J + 1` bands.
    #[config(default = 4)]
    pub levels: usize,
    /// Gate groups per band; each group covers `d_model / groups` channels.
    #[config(default = 8)]
    pub groups: usize,
    /// Initial gate value `σ(b_0)` of the coarse band `A_J` (WNet §4.2).
    #[config(default = 0.88)]
    pub coarse_gate_init: f64,
    /// Initial gate value `σ(b_j)` of every detail band (WNet §4.2).
    #[config(default = 0.12)]
    pub detail_gate_init: f64,
}

impl WaveletDynConfig {
    pub fn init(&self, device: &Device) -> WaveletDyn {
        assert!(self.levels >= 1, "the wavelet mixer needs at least one level");
        assert!(
            self.groups > 0 && self.d_model % self.groups == 0,
            "d_model {} is not divisible into {} gate groups",
            self.d_model,
            self.groups
        );
        for probability in [self.coarse_gate_init, self.detail_gate_init] {
            assert!(probability > 0.0 && probability < 1.0, "gate initial value {probability} is outside (0, 1)");
        }
        let bands = self.levels + 1;
        let projection = || LinearConfig::new(self.d_model, self.d_model).with_bias(false).init(device);

        let logit = |probability: f64| (probability / (1.0 - probability)).ln() as f32;
        let bias: Vec<f32> = (0..bands)
            .flat_map(|band| {
                let initial = if band == 0 { self.coarse_gate_init } else { self.detail_gate_init };
                std::iter::repeat_n(logit(initial), self.groups)
            })
            .collect();
        let mut gate = LinearConfig::new(self.d_model, bands * self.groups).init(device);
        gate.bias = Some(Param::from_data(TensorData::new(bias, [bands * self.groups]), device));

        WaveletDyn {
            mixed: self.precision == dre_core::Precision::Bf16Mixed,
            input: projection(),
            gate,
            value: projection(),
            output: projection(),
            levels: self.levels,
            groups: self.groups,
        }
    }
}

/// Dynamic undecimated wavelet mixer, `[B, T, d] -> [B, T, d]` (spec §3.2, WNet-Dyn Eqs. 6–7).
#[derive(Module, Debug)]
pub struct WaveletDyn {
    mixed: bool,
    /// `W_i`: projection read by the wavelet bands, no bias.
    pub input: Linear,
    /// `W_g` and `b`: gate logits, `d -> (levels + 1) * groups`, band-major.
    pub gate: Linear,
    /// `W_v`: value branch, no bias.
    pub value: Linear,
    /// `W_o`: output projection, no bias.
    pub output: Linear,
    pub levels: usize,
    pub groups: usize,
}

impl WaveletDyn {
    /// Scale gates `σ(U W_g + b)`, `[B, T, levels + 1, groups]`; band `0` is `A_J`, band `j > 0` is `D_{J+1-j}`.
    pub fn gates(&self, u: Tensor<3>) -> Tensor<4> {
        let [batch, tokens, _] = u.dims();
        sigmoid(dre_core::project(&self.gate, u, self.mixed)).reshape([batch, tokens, self.levels + 1, self.groups])
    }

    /// Mixes `u: [B, T, d]` within each document of `layout`; padded positions of the output are exactly 0.
    pub fn forward(&self, u: Tensor<3>, layout: &BatchLayout) -> Tensor<3> {
        let [batch, tokens, channels] = u.dims();
        let group_width = channels / self.groups;
        let padding = layout.valid.clone().lower_elem(0.5).expand([batch, tokens, channels]);
        let u = u.mask_fill(padding, 0.0);

        let bands = undecimated_bands(dre_core::project(&self.input, u.clone(), self.mixed), &layout.host, self.levels);
        let gates = self.gates(u.clone());
        let mixed = bands
            .into_iter()
            .enumerate()
            .map(|(band, values)| {
                let gate = gates.clone().narrow(2, band, 1).reshape([batch, tokens, self.groups, 1]);
                values.reshape([batch, tokens, self.groups, group_width]) * gate
            })
            .reduce(|sum, term| sum + term)
            .expect("at least one band")
            .reshape([batch, tokens, channels]);

        dre_core::project(&self.output, mixed * dre_core::project(&self.value, u, self.mixed), self.mixed)
    }
}
