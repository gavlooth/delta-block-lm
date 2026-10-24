# Causal recurrent mixers

Current local optimization results are in the root `RESULTS.md`. Fused KronLSTM and fused HOLA pass independent f64 operator checks, and the measured normalized models pass CPU parameter-gradient/update comparisons. A newly exposed legacy GPU TensorSolver gradient failure remains on the short nonzero-initial-state stress fixture; do not treat that path as an arbitrary-input numerical oracle. CPU remains the tensor reference. Explicit fused selection is recommended for new supported Vulkan experiments; historical defaults are unchanged.

`delta-mixers` supplies a trainable Burn 0.22 module, not a frozen feature map.
The default configuration preserves four heads, key width 32, value width 64,
and computational chunk length 16. Model width is independently configurable.
Supported computational chunk sizes are 1 through 16, checked before parameter
initialization and by the low-level scan. This is the migrated kernel contract,
not a maximum context length or a denoising-block limit. Larger computational
chunks need a different stable triangular solver and are not implemented:
unbounded nilpotent matrix powers in f32 can overflow on contractive inputs.

```rust,ignore
use burn::prelude::*;
use delta_mixers::{CachePolicy, MixerConfig, MixerKind};
let device = Device::ndarray().autodiff();
let config = MixerConfig {
    kind: MixerKind::KronLstm,
    cache_policy: CachePolicy::Surprise,
    ..MixerConfig::default()
};
let mixer = config.init(&device);
let x = Tensor::<3>::zeros([2, 128, config.d_model], &device);
let y = mixer.forward(x); // [batch, time, d_model]
```

`forward_masked(x, valid, reset)` accepts float masks `[batch,time]` (0 or 1).
Padding emits zero and freezes recurrence state. A valid reset token starts a
new document before its update. Short convolutions and exact cache reads cannot
cross a reset; invalid source positions never contribute. Every `forward` starts
from an empty prefix. The lower-level `chunk::scan` accepts an optional boundary
state and returns outputs, final state and committed-edit scores; cache and
convolution histories are not included in that low-level state. Consequently,
resuming `chunk::scan` is not a complete streaming-model/cache API.

## Recurrences and chunking

States use `[key,value]` storage per head. GDN-2 uses

- `B = diag(alpha) S`,
- `d = write * v - (erase * k)^T B`,
- `S' = B + k d^T`, `o = q^T S'`.

KronLSTM implements [Eq. (10)](https://arxiv.org/html/2610.26834#S2.E10),
with separately learned scalar gates per head:

- `B = diag(alpha) S`,
- `z = c - beta * k^T B v`,
- `S' = B + z k v^T`, `o = q^T S'`.

KronLSTM values and both mixers' recurrent queries/keys are L2-normalized.
Both have learned Q/K/V projections, document-aware short convolutions of width
four, channel-wise learned decay, output gating, head RMS normalization and an
output projection. GDN-2 erase/write gates are channel-wise; KronLSTM beta/c
are scalar per head. The wrapper uses identical output processing in both arms.

`chunk::scan` derives a strictly lower-triangular correction system per chunk.
For GDN-2 its unknowns are value-vector edits. For KronLSTM they are scalar
bilinear corrections; its lower matrix is the product of decayed key overlap,
value overlap and beta. A finite nilpotent polynomial solves each triangular
system using batched matrix multiplications. Only boundary states recur across
chunks. Direct decay products include reset zeros and padding identities;
there is no division by cumulative decay. Burn's zero-aware `cumprod` backward
handles the prefix products at resets.

The default `ScanBackend::TensorSolver` runs the same differentiable tensor
program on NdArray and Vulkan. `ScanBackend::FusedGpu` explicitly selects the
KronLSTM CubeCL forward/backward scan at key32/value64; requesting it for GDN-2,
a CPU device, or an unsupported shape fails instead of falling back. The
`chunk::scan_selected` API accepts the selector and otherwise preserves
`chunk::scan`'s masks, initial/final state and physical edit scores. All three
returned tensors retain derivatives, including the initial state and score
objectives. At a zero physical edit both paths choose the norm's zero
subgradient without changing nonzero scores or cache ordering.

HOLA independently selects `HolaBackend::Tensor` (the historical default) or
`HolaBackend::FusedGpu`. The latter requires a GPU, key32/value64, capacity
1..=64 and chunk length 1..=16. It retains exact causal membership, original
KV entries, earlier-position tie breaking, recency selection and gradients
through cache reads/normalization/gate/sink. These execution selectors change
no trainable parameter shapes or identities. The separate GDN-2 custom kernel
remains independently checked; GDN-2 recurrent model execution uses the tensor
solver. Implemented fused paths are not by themselves a measured speedup.

`projection_backend` is independent of the recurrence/HOLA selectors:
`separate` (the omitted historical default) executes the original seven input
linears; `grouped` concatenates Q/K/decay, V/output-gate and erase/write weights
into three temporary contractions; `merged` concatenates all seven into one.
The dependent output projection remains separate: its input is the recurrent
and optional cache result, not the common normalized block input. Packing uses
Burn's differentiable concatenate/linear/slice operations. Original parameter
names, shapes, IDs and AdamW state are unchanged; no packed parameter record,
initializer, cached stale weight or checkpoint migration is introduced.
Projection selection does not itself change activations or normalization.
Nondefault selectors serialize explicitly and require a fresh run identity;
the omitted default still serializes exactly as before. See `RESULTS.md` for
the measured adoption decision, not just the forward-only improvement.

`normalization_backend` independently selects `tensor` (omitted, historical),
`fused_norms` (custom L2 and all learned RMSNorm reductions), or
`fused_norms_gates` (also fold sigmoid/output multiplication into mixer
RMSNorm and fuse channel-bias/head-scale/softplus/exp decay). Original modules,
gamma IDs, separate HOLA Q/K parameters, valid masks and checkpoint storage
remain unchanged. L2 uses `sum(x²)+1e-6`; block/final RMSNorm uses
`mean(x²)+1e-6`; inherited mixer/HOLA RMSNorm uses `mean(x²)+1e-5`.
No epsilon or softplus threshold is retuned. The custom scopes require f32
GPU execution, final-axis widths 1..=1024 and first-order differentiation.

The remaining convolution SiLU, erase/write sigmoids, HOLA blend gate and
residual additions continue to use existing primitives. These small pieces
are not assumed to have the same payoff as the reduction/backward-heavy
scopes. The independent scalar-f64 normalization oracle checks output and
every input/gamma/decay-bias/head-scale/output-gate VJP, zeros, saturated
gates, strided inputs/adjoints, tail widths and internal/trailing validity
masking at unchanged 2e-5/3e-4 tolerances. Run
`normalization-reference` and `normalization-benchmark` as documented in
`../experiment/README.md`; use integrated paired results, not isolated
dispatch savings, to choose the explicit selector.

## HOLA and matched recency control

`None`, `Surprise`, and `Recency` are available for each recurrence. Cache-only
parameters are absent in `None` arms. The cache retains original convolutional
K/V representations, not erase/write factors or normalized KronLSTM write
values. Independent learned cache-path Q/K RMSNorm scales sharpen the exact
softmax read; a learned per-head gate (initialized at -4) adds its result to the
recurrent read, with a learned null-sink logit and exactly zero sink value.

The persistent cache holds the top-capacity committed edits in the causal
history. GDN-2 scores are `norm(k) * norm(d)` and KronLSTM scores are
`abs(z) * norm(k) * norm(v)`. Neither score includes passive decay. Scores are
fixed at the original update. Selection is discrete, with exact ties resolved
by earlier position; selected original K/V and cache-read parameters remain
differentiable. Recency changes only selection to position order, preserving
capacity, exact read and local attention.

A read sees the cache selected after the previous computational chunk plus
causally visible original KV in the current chunk, without duplicate entries.
A cache update occurs only after the chunk's causal recurrence has completed.
Packed-document resets also mask entries and local reads inside a chunk.
Current-chunk attention is counted: the check report records allocated
`batch * heads * chunk_tokens * (capacity + chunk_tokens + 1)` score pairs,
including masked work and the null sink. `exact_memory_bytes` is an f32
KV/score/position/document storage estimate including the current chunk; it is
not measured peak device allocation or training-graph memory.
`MixerConfig::attention_work(batch,tokens)` exposes separate history, current-chunk
and sink pair counts, plus stable-selection rank pairs and sort input elements.
The selection rank computes `(capacity + chunk_tokens)^2` pairs per batch/head;
the number of sort elements is not claimed to be a sort-comparison count.

This crate does not implement block-denoising temporary-state writes or
bidirectional noisy-token replay. Persistent inference cache commits must only
receive finalized causal tokens; no step-7 components are included.

## Numerical command

From the workspace root, the integration owner runs:

```sh
timeout 120s cargo run -p delta-experiment -- reference configs/reference-cpu.json
timeout 120s cargo run -p delta-experiment --features vulkan -- reference configs/reference-vulkan.json
timeout 60s cargo test -p delta-mixers --lib
```

No GPU features are enabled for the CPU command/test. The experiment runtime
selects and records the device and persistent kernel-cache location.
`numerical_checks(&Device)` returns a serializable report or the first numerical
failure, never a fabricated success. It covers:

- f32 chunked outputs, final states, six independent Q/K/V/decay/erase/write
  gradient fields, and active-edit scores against independent f64 sequential
  arithmetic and analytic reverse mode;
- lengths 7 and 35, four heads at 32/64 widths, multiple chunks, internal and
  trailing padding, packed resets at positions 5 and 19, and boundary-state
  continuation at position 3 inside the computational chunk, plus aligned unit
  keys/values with unit decay and half-strength gates at the supported maximum;
- both cache policies at capacities 3 and 16, exact ties, original KV gradients,
  causal cache membership, resets and padding against an independent scalar
  f64 exact-attention/cache oracle;
- full trainable-layer input directional derivatives and one Q-projection
  parameter finite difference in all six mixer/cache arms. Cache arms additionally
  span a chunk without reset and check learned cache Q/K scales, gate and sink;
- on GPU, actual custom GDN-2 forward/backward at unextended lengths 7 and 35,
  including the true shorter-than-chunk case. Separate final-state passes add
  thirty-two diagnostic zero-update identity-query tokens to read all state rows;
  `final_state_probe_tokens` records their timing cost explicitly.
- KronLSTM lengths 1/7/35 with nonzero initial state, channelwise zero decay,
  signed and exactly zero corrections, zero key/value norms, an all-padding
  chunk and reset-on-padding; output-only, final-state-only, score-only and
  combined analytic VJPs, including every float input and the initial state;
- GPU HOLA membership/output/gradient comparisons against the retained tensor
  path and independent f64 cache arithmetic, plus full-layer finite differences
  across tensor/fused recurrence and tensor/fused HOLA combinations.

Recurrence/cache f32 tolerances are `2e-5 + 3e-4 * abs(f64_reference)`.
Membership must agree exactly. Layer central differences use perturbation
`2e-3` and tolerance `3e-3 + 0.03 * abs(analytic_gradient)` because model
execution and perturbed objectives are f32. First-use and warm forward/backward
wall times include synchronized host readback. Kernel compilation itself is not
separately instrumented: the cold pass includes its cost, along with execution
and readback; the warm pass is separately reported. Any nonfinite compared
value fails the command.

Adding a check implementation does not mean it has passed. Execution results,
build-budget measurements and experiment outcomes belong in the run report.

## Source provenance

`src/gdn2_reference.rs` is ported from
`reuse/crates/dre-gdn2/src/reference.rs`, original SHA-256
`f35b1ac4b9cc756d9cb1c316e3ac496ca0fa46d8c52912fab1f5ca1f6cd43e45`.
The port additionally preserves state snapshots at internal padding positions,
so the following valid token's backward pass sees the correct previous state.
A regression checks the following alpha/erase/key derivatives by f64 differences.
The source snapshot provenance and inherited MIT OR Apache-2.0 license are
recorded in `reuse/PROVENANCE.json`. The new generic scalar oracle and common
chunk tensor solver preserve its causal GDN-2 equations while adding the
KronLSTM recurrence, boundary-state objectives and explicit padding freezes.
The HOLA adaptation follows [the original method](https://arxiv.org/html/2607.02303)
with mixer-specific committed-edit scores, not a claim of reproducing its
published training results.
