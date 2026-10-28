# Four-stream orthogonal residual connections

## Architecture and migration

The implementation originated in `reuse/crates/dre-ohc/src/{streams,stage,rotation,readout}.rs`. The active equations now follow [oHC v1, Eqs. (1), (4), (8), (12), (15)](https://arxiv.org/html/2609.02672v1), with omitted gate/expansion/contraction details from pinned [Megatron-LM `hyper_connection.py`](https://github.com/NVIDIA/Megatron-LM/blob/4c3ff2a78fc3417088f289f61a2fe36809d6bb86/megatron/core/transformer/hyper_connection.py) and the selected [hybrid stack](https://github.com/NVIDIA/Megatron-LM/blob/d7e06cffa6f37c845dfa35cbf0137191708d0a3c/megatron/core/models/hybrid/hybrid_block.py). The audit is static evidence; these primary sources determine the corrections. No archived migration input is modified.

`ohc_streams: true` changes the model architecture. Each active wavelet, Lngram, mixer and MLP sublayer has its own wrapper. The combined four-block profile has 14 wrappers. Each wrapper reads four streams, runs its existing operator (including the operator's normalization), and writes the result back. This replaces ordinary residual addition; it does not add another one. Final learned stream contraction is followed by exactly the existing model final norm and vocabulary head.

The default is `ohc_streams: false`, `ohc_backend: "tensor"`. Both fields are absent from default serialization. Enabled models require the current `paper-local-math-joint-plus-token-v3` model math identity: old checkpoints are incompatible, including old oHC records with affine stream expansion, stream-read gain, write scalar or readout norm. There is no conversion or compatibility shim. Backends retain identical parameter semantics. Fused selectors require Vulkan or CUDA; there is no CPU fallback. The central identity check covers the coordinated paper corrections and combined objective, not merely tensor dimensions.

### Current equations

For each token `X: [4,d]`, the bias-free gate projection has weight `[4d,16]`. Its output is `(vec(X) W) / (sqrt(mean(vec(X)²)) + 1e-6)`, preserving pinned upstream `native_proj_rms` epsilon placement. Four read gates are `sigmoid(alpha_read * p + beta_read) + 1e-6`; four write gates are `2 * sigmoid(alpha_write * p + beta_write)`. Eight rotation logits use fixed, nontrainable `alpha_mix = 0.01`. Read/write gate scales remain trainable from `0.01`. The zero norm is handled before the tensor square root; fused VJPs use the exact nonzero derivative and zero norm subgradient. In particular the composite projection derivative at zero is finite `W / 1e-6`.

The quaternion leading coordinates receive a constant one. Each quaternion is normalized with epsilon `1e-12` under the square root; `Q = L(q) R(conjugate(r))` acts as `q x conjugate(r)`. Ordinary construction zeros the eight rotation projection columns and rotation bias, so `Q` starts at the exact f32 identity. It does not disable the operator branch or freeze rotation learning. Read/write projection columns keep their upstream Xavier limit based on 24 original outputs; only the 16 consumed outputs are stored.

The operator reads **raw** streams, `u = sum_s a_s X_s`. Its output is written as `Q X + b F(u)` with no additional scalar or per-stream RMS gain. Stream expansion is exact parameter-free replication, not learned affine expansion. Learned readout retains its `[4d,4]` projection, unit initial gate scale, zero biases, RMS-conditioned sigmoid weights and `1e-5` gate floor. Its result is the **unnormalized** weighted sum; the model applies its single final RMSNorm. The contraction gate's epsilon remains under the square root, as in the selected upstream `learned_output_contract`; it is not the stage gate equation.

### Causal-LM adaptations

- Invalid features are selected to zero before exact stream replication, preventing inactive NaN/Inf inputs from contaminating outputs or input gradients.
- Each write is explicitly selected to zero at invalid tokens. No padded residual can persist between sublayers.
- Initialization, flattened gate projection, quaternion rotation, read, write, and readout operate independently at each token. They need no token history or document ID. `reset` remains with the causal mixer; it is not used to pool or route oHC streams. There is no bidirectional extension to remove.
- The predecessor's BF16 option is not copied: the active workload and acceptance gates are f32. There are no relay streams or early-bank readouts in this causal LM.
- Four streams, quaternion arithmetic, 16 consumed gate outputs, and full saved-state CoCommit markers are unchanged. The marker remains an independent additive `[4,d]` tensor on saved pre-tail streams; it is not broadcast, resized or moved to a work vector.

`identity_init()` now means identity **residual transport**, not a no-op wrapper. Normal construction uses it; read/write gates and branch writes remain active.

## Designs compared

Three plausible approaches were considered: dropping fused paths and relying entirely on tensor operations, retaining compatibility parameters fixed at constants, or correcting tensor and fused equations together. The last keeps existing execution choices without obsolete parameters or a second convention, and is the selected implementation.

1. **`tensor`** uses Burn projection with a compensated skinny-contraction VJP, stable flattened RMS, raw weighted sum, sigmoid, quaternion construction, batched `Q X` and broadcast write.
2. **`fused_gpu`** keeps tensor reads but fuses `Q X + b F(u)` and its existing analytic VJP.
3. **`fused_read_write`**, **`fused_gate`**, and **`transposed_gate`** preserve the same equations. Raw fused reads retain only streams and weights, with two adjoints; removed inverse norms and gain adjoints are not cached. Fused gate normalization/projection VJPs place epsilon outside the root and handle zero RMS explicitly. Quaternion and write VJPs are unchanged.

Strided streams, matrices, projections, gates, operator outputs and adjoints remain supported. Reductions are deterministic, without atomics.

## Correctness gates

`ohc-reference` runs an independent scalar f64 sequential oracle with its own reverse-mode tape and Hamilton-product rotations. It checks outputs, feature/operator gradients, all nine remaining stage/readout parameter tensors and the single model-final-norm gain. Tolerance remains `2e-5 + 3e-4 * abs(reference)`. Cases include zero and near-zero projection inputs with **nonzero weights**, noncontiguous inputs, raw reads from unequal streams, exact replication, ordinary exact-identity rotation with active writes, raw learned contraction and one final norm with non-unit gains. Lengths 1/5/17, batch two, widths 3/7, padding, empty rows, NaN/Inf padding and token locality remain covered. Separate primitive checks exercise every retained input gradient. Projection checks cover 129-token reduction tails and repeated deterministic gradients.

A complete tiny causal LM is also compared between CPU tensor execution and GPU fused-write execution: logits, every parameter gradient, and the first AdamW update are checked at the same tolerance. Internal document resets, padding, tail length nine, future-token perturbations and cross-document perturbations are exercised. Separate per-token checks ensure that changing one input position cannot change any other oHC position.
Current CPU and local Vulkan `ohc-reference` runs passed during resumed acceptance on 2026-10-28, with the independent f64 reference and its tolerance unchanged. Command/status/stdout/stderr evidence is under `artifacts/resume-acceptance/commands/` (`cpu-ohc-final` and `vulkan-ohc-shared-split`); numerical reports are in the backend subdirectories.
The first resumed CPU run exposed input-gradient cancellation at zero and near-zero gate inputs: the inverse RMS approaches `1e6`, amplifying the rounding of the 16-term `dy Wᵀ` contraction. The tensor path now normalizes before projection and uses product-residual plus TwoSum compensation for that contraction. The fused VJP contracts unscaled adjoints, compensates the same products/sums, and applies inverse RMS last. Register-only compensation failed on local Vulkan; explicit shared-memory rounding boundaries passed. All stage selectors use the corrected projection entry point. The tiny causal full-model fixture explicitly disables inherited SBD-only parameters instead of asking a causal loss for their gradients.

No current performance measurement was run for this precision repair. The added compensation and Vulkan synchronization invalidate any assumption that historical gate-projection timing remains unchanged.

## Historical measurements — superseded equations

Everything below records the **pre-correction** equations and their old parameter counts, normalization, initialization, caches and performance. It is retained as historical evidence, not current acceptance or performance evidence. Rerun the current checks before training; old numerical results do not establish parity with the corrected equations.


Measured Vulkan reference results on the local Intel adapter:

| Gate | Maximum absolute difference/error |
|---|---:|
| CPU vs GPU model logits | 1.416e-7 |
| Every model parameter gradient | 3.874e-7 |
| First AdamW parameter update | 2.570e-7 |
| Causal/document perturbation | 0 |
| `QᵀQ - I` | 3.293e-7 |
| `det(Q) - 1` | 4.815e-7 |
| Rotated squared norm | 3.586e-7 |
| Ordinary initial `Q - I` | 0.03042 |
| Exact-identity initialization | 0 |

The largest raw component-gradient difference is `0.001528`, in the zero-input cases where repeated RMS inverse factors make the reference derivative large; every individual result still satisfies the unchanged absolute-plus-relative tolerance. Raw maximum absolute error alone is not the pass criterion. Reports contain the individual cases and `passed: true`.

The first CPU-model transfer attempt used Burn's differentiable `to_device`, which creates intermediary parameters and produced missing-parameter-gradient errors. The gate now uses `Module::fork` to establish independent parameter leaves on the target device; no missing-gradient check is suppressed.

## Reproducibility

All saved inputs and reports are under `artifacts/step7/ohc/`. `lm512.json` copies `configs/lm512-optimized.json`, adding only this architecture/selector and private artifact location; its batch/context/width/layers are 8/512/256/4. `lm128.json` is the pilot context128 counterpart. The selected configuration is `lm512-readwrite.json`; the write-only comparison and its reports are preserved separately. Architecture comparisons clone the same ordinary model parameters, not merely the same seed: Burn parameter initialization is lazy, so shared parameters are materialized before attaching oHC. All enabled arms also share identical oHC parameters. Training data, seeds, step indices, optimizer settings and trial counts match across compared arms.

From the `s7-ohc` workspace root:

```sh
../build.sh s7-ohc
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 180s /tmp/delta-bins/s7-ohc/delta-experiment ohc-reference artifacts/step7/ohc/reference-cpu.json
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 180s /tmp/delta-bins/s7-ohc/delta-experiment ohc-reference artifacts/step7/ohc/reference-vulkan.json
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 300s /tmp/delta-bins/s7-ohc/delta-experiment ohc-benchmark artifacts/step7/ohc/lm512.json --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 300s /tmp/delta-bins/s7-ohc/delta-experiment ohc-benchmark artifacts/step7/ohc/lm128.json --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 300s /tmp/delta-bins/s7-ohc/delta-experiment ohc-learning-smoke artifacts/step7/ohc/learning-smoke.json
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 180s /tmp/delta-bins/s7-ohc/delta-experiment ohc-reference artifacts/step7/ohc/reference-vulkan-readwrite.json
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 300s /tmp/delta-bins/s7-ohc/delta-experiment ohc-benchmark artifacts/step7/ohc/lm512-readwrite.json --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 300s /tmp/delta-bins/s7-ohc/delta-experiment ohc-benchmark artifacts/step7/ohc/lm128-readwrite.json --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 180s /tmp/delta-bins/s7-ohc/delta-experiment ohc-attribution artifacts/step7/ohc/lm512-readwrite.json
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 300s /tmp/delta-bins/s7-ohc/delta-experiment ohc-learning-smoke artifacts/step7/ohc/learning-smoke-readwrite.json
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 600s env CARGO_TARGET_DIR=/home/heefoo/code/delta-block-lm-ws/target-shared cargo test --workspace --features delta-experiment/vulkan
```

## Initial write-only measurements

The first comparison uses two warmups and 11 paired, interleaved trials. Complete-step timing includes host preparation/upload, forward, scalar loss readback, all parameter gradient checks, clipping and AdamW. It excludes first-use compilation and the separate launch-count observation steps. The local device reports Intel Arc through Vulkan; dtype is f32 and seed is one.

| B8 / D256 / four blocks | Context128 | Context512 |
|---|---:|---:|
| Isolated tensor F, ms | 6.651 | 26.505 |
| Isolated write-fused F, ms | 5.261 | 22.219 |
| Isolated tensor F+B, ms | 38.315 | 134.198 |
| Isolated write-fused F+B, ms | 30.624 | 106.348 |
| Complete step, oHC off, ms | 184.327 | 853.605 |
| Complete step, tensor oHC, ms | 401.376 | 1620.660 |
| Complete step, write-fused oHC, ms | 339.499 | 1379.034 |
| Paired write-fusion step gain, ms | 63.859 | 236.925 |
| Paired gain MAD, ms | 8.749 | 8.734 |
| Candidate wins | 11/11 | 11/11 |

At context512, isolated tensor/write-fused dispatches are 89/87 forward and 312/306 forward+backward. Complete guarded-step launches are 1718 with oHC off, 3977 with tensor oHC, and 3929 with write-fused oHC. This is an execution improvement, not an equal end-to-end speedup or an architecture-quality result. Write-fused oHC still adds approximately 525 ms, or 62%, over the single-stream step.

## Attribution of the remaining overhead

Because the remaining cost greatly exceeds the byte traffic alone, `ohc-attribution` measures resident per-stage pieces independently, with two warmups and 11 trials at B8/T512/D256. Forward+backward includes a square-sum scalar objective for each piece; intermediates are independent gradient leaves. Complete read concatenates work, rotation, and write gates so all read paths receive adjoints. Pieces are **not additive** end-to-end times: parameter reductions, objectives and gradient traversal differ.

| Piece | F, ms | F+B, ms | F launches | F+B launches |
|---|---:|---:|---:|---:|
| Stream initialization, once per model | 2.064 | 12.341 | 5 | 21 |
| Gate GEMM, `4d → 16` | 5.727 | 22.985 | 1 | 13 |
| Gate RMS and sigmoid epilogue | 1.920 | 9.105 | 32 | 100 |
| Per-stream RMS and weighted read | 3.262 | 21.964 | 8 | 44 |
| Quaternion-to-SO(4) map | 1.157 | 2.006 | 15 | 68 |
| Tensor write | 6.819 | 38.774 | 4 | 22 |
| Fused write | 1.578 | 9.138 | 2 | 16 |
| Complete tensor read | 11.114 | 54.045 | 56 | 197 |
| Readout, once per model | 8.573 | 38.134 | 24 | 96 |
| Four-stream scale+add traffic proxy | 1.613 | 6.685 | 2 | 13 |

The streams occupy 16 MiB per stage. The scale+add proxy reads and writes these streams; it is not a physical-bandwidth measurement. Its time is much lower than the 54 ms complete read backward path, supporting optimization of the tensor read graph rather than attributing the entire overhead to four-stream traffic. Gate GEMM, stream-normalization gradients and the many small gate/quaternion launches dominate the measured read side.

The explicit `fused_read_write` design therefore adds analytic first-order kernels for flattened gate RMS scaling, per-stream RMS plus weighted contraction, and quaternion-to-SO(4), alongside the existing fused write. The gate GEMM and sigmoid bias/scale operations remain ordinary differentiable Burn operations. The weighted-read kernel retains four inverse norms per token rather than the full normalized stream tensor; gate-norm and quaternion VJPs recompute their small local quantities. The predecessor's configurable stream RMS epsilon remains supported. Width is explicitly validated at at most 1024 for the workgroup reduction.

## Parameter and activation memory

At B8/T512/D256/four blocks:

- Added parameters: **139,677**, or **558,708 bytes** in f32. There are no lookup tables.
- A single residual stream is 4 MiB; four streams are **16 MiB**, adding **12 MiB** of live residual storage.
- The stream input retained by each of eight wrappers is 16 MiB, totaling **128 MiB** before other saved values. This is a tensor-size sum, not a separate allocation peak.
- Each per-stage rotation tensor and projected-gate tensor is 256 KiB.
- The fused write's retained `X,Q,gamma*b,F` and its backward output tensors each total 20.3125 MiB per stage. Shared references can alias existing storage; these are not automatically additive allocation costs.
- The weighted fused read retains four inverse norms per token: 64 KiB per stage. It does not materialize a `[B,T,4,D]` normalized-stream temporary.
- The initial three-trajectory context512 experiment sampled **1,133,362,368 bytes** of allocator pool reservation. This includes all scratch models, optimizer states and allocator reuse, not physical peak VRAM or a single model's live allocations.

## Short learning smoke

The tiny byte LM runs 200 steps per arm, seed one, batch one, context16, width32, one block, 3200 tokens per arm. The common ordinary model parameters are identical initially. Both arms use identical data windows, learning rate `0.003`, and work budgets, and every step checks all parameter gradients before AdamW.

| Arm | First 20 mean losses | Last 20 mean losses |
|---|---:|---:|
| oHC off | 5.06137 | 2.86696 |
| Write-fused oHC | 4.75072 | 2.73247 |

Both losses decrease, with no nonfinite losses or gradients. This is a short optimization-stability check, not a quality comparison.

## Build and regression gates

The final workspace regression run passes **58 tests**, with zero failures and one pre-existing ignored GPU-sharing test: experiment 37, kernels seven, mixers 14. Test compilation takes 6.11 seconds, below the 60-second no-run budget. A clean-local-package development rebuild takes 6.15 seconds, below the 10-second model rebuild budget; its 59-second shared-lock wait is separate. The only dependency addition is reuse of the existing workspace `serde_json` dependency for structured reference reports.

## Read-fusion decision

The final experiment compares all four complete-step arms in one run, with two warmups and 11 rotating forward/reverse, interleaved trials. The adapter is **Intel(R) Arc(tm) Graphics (MTL)**, an integrated GPU at PCI `0000:00:02.0`, using Vulkan and **Mesa 26.2.4-arch1.1**. Fusion and autotune are off. Isolated inputs have `valid=1` everywhere; actual LM batches also have all-valid windows and reset at each row's first position.

| Final comparison, B8/D256/four blocks | Context128 | Context512 |
|---|---:|---:|
| Isolated tensor F / F+B, ms | 6.624 / 38.057 | 26.529 / 132.125 |
| Isolated write-fused F / F+B, ms | 5.120 / 30.444 | 21.289 / 103.032 |
| Isolated read+write-fused F / F+B, ms | 4.576 / 24.798 | 19.075 / 82.569 |
| Complete step, oHC off, ms | 183.592 | 858.725 |
| Complete step, tensor oHC, ms | 409.372 | 1615.660 |
| Complete step, write-fused oHC, ms | 344.831 | 1395.409 |
| Complete step, read+write-fused oHC, ms | 301.463 | 1198.177 |
| Paired write-only → read+write step gain, ms | 44.545 | 196.998 |
| Paired gain MAD, ms | 15.339 | 7.988 |
| Candidate wins | 11/11 | 11/11 |

At context512 the added read fusion saves approximately 14.1% of the write-only complete-step median; tensor-to-selected median reduction is approximately 25.8%. These are model-step results, not extrapolated isolated gains. Isolated selected F/F+B dispatches fall to **61/196**, versus write-only **87/306**. Complete-step dispatches fall from **3929** to **3049**. The selected model still costs approximately **339 ms**, or **39.5%**, more than oHC off at context512; at context128 the relative overhead remains approximately 64%.

Repeated attribution in the same final design:

| Read piece | Tensor F / F+B, ms | Fused F / F+B, ms | Tensor F / F+B launches | Fused F / F+B launches |
|---|---:|---:|---:|---:|
| Per-stream RMS + weighted read | 3.482 / 22.523 | 1.465 / 4.898 | 8 / 44 | 1 / 12 |
| Quaternion-to-SO(4) | 0.674 / 1.296 | 0.239 / 0.508 | 15 / 68 | 1 / 11 |
| Complete read, all output adjoints | 11.733 / 56.115 | 8.053 / 32.243 | 56 / 197 | 30 / 87 |

The fused gate-RMS primitive alone is 0.630 ms forward / 1.431 ms forward+backward, with 1/11 launches including its scalar objective and gradient traversal. The tensor gate-RMS-plus-sigmoid row is 1.736/9.472 ms; those scopes differ, so that pair is not a primitive speedup ratio. The remaining gate GEMM costs 5.454/22.820 ms and remains the largest individual read piece. The final readout remains 8.801/39.001 ms once per model. The four-stream traffic proxy is 1.675/7.583 ms; the improvement comes from eliminating normalization intermediates and small tensor-graph launches, not changing the number of residual streams.

The new independent f64 read primitives pass, including strided inputs/gains/adjoints and custom epsilon: maximum fused errors are `2.465e-7` contiguous and `1.509e-7` strided. The complete CPU-model versus selected GPU gate has logits/every-gradient/AdamW maximum errors **1.192e-7 / 4.694e-7 / 2.757e-7**, and causal/document perturbation error zero. The selected 200-step smoke decreases from **4.75072** to **2.73166** mean loss over the first/last 20 steps; off remains **5.06137 → 2.86696**. No quality conclusion follows from this smoke.

**Decision:** use explicit `ohc_backend: "fused_read_write"` for the B8/T512 optimized Step-7 configuration; retain the tensor port as the default and both measured alternatives for reproducibility. For this workload, set an oHC budget of at most **0.40 s additional guarded-step time**, **0.60 MB added f32 parameters**, and **128 MiB retained residual-stream inputs** across the eight wrappers. Measured additions are approximately 0.339 s, 0.559 MB and 128 MiB. This authorizes the small Step-7 workload only, not larger training or Step 8. Architecture-quality evidence must justify the remaining compute cost before scaling.

The four-trajectory final run samples **1,159,548,352 bytes** of pool reservation. Its scratch-model count differs from the initial three-trajectory run; the two totals must not be interpreted as a measured single-model memory delta. Added AdamW moments occupy 1,117,416 bytes at f32; added parameters, gradients and both moments total 2,234,832 bytes, excluding allocator metadata.

`backward-profile` is intentionally unchanged: its full-model runner includes enabled oHC, but its existing independent mixer/MLP component fixtures do not attribute oHC. Use `ohc-attribution` for the new pieces, or interpret that older profiler's residual coverage as including them.

The final selected reference report contains **49 cases**, all passing. Its largest raw component-gradient difference is `0.002126` in the zero-input pipeline; all elements satisfy the unchanged `2e-5 + 3e-4 * abs(reference)` criterion. Across all backends, the final SO(4) maxima remain `3.293e-7` for `QᵀQ-I`, `4.815e-7` for determinant error, and `3.586e-7` for squared norm error; exact identity initialization remains bit-exact.

## Combined Step-7 gate-projection candidates

Two additional execution selectors keep the parameter layout, four-stream architecture, existing selectors and tensor default unchanged:

- **`fused_gate`** replaces the skinny gate GEMM and flattened-stream RMS with one token-local CubeCL operation. Each 256-thread workgroup computes all 16 projection channels, sharing the tree reductions over 16 input partitions. It reads the existing `[4d,16]` weight directly; loading the entire D256 weight into shared memory would require 64 KiB per group, so this candidate does not do that. The analytic input VJP also computes the projected adjoint. Weight gradients use shared 32-token by 16-feature/16-channel tiles, one partial per 128 tokens, then a fixed-order second-pass sum. There are no float atomics. Forward uses one launch; backward uses three. Gate scale/bias/sigmoid, the existing weighted-read primitive, quaternion primitive and fused write retain their existing implementations.
- **`transposed_gate`** reuses the library GEMM as `[16,4d] × [4d,BT]`, then transposes the result to `[BT,16]` and uses the existing fused flattened RMS, weighted read, quaternion map and write. The long token dimension becomes GEMM N rather than M. Transposes are views where the library accepts them; any materialization required by library projection or reshape is included in the timings. No alternate weight parameter or checkpoint layout is introduced.

These deliberately test the largest attributed read-side cost without rewriting the entire read and its seven trainable gate inputs at once. Both retain the ordinary differentiable sigmoid epilogues, so their remaining launch cost is visible rather than attributed to projection fusion.

`ohc-reference` includes both candidates in the complete independent-f64 pipeline and adds a separate scalar-tape projection check over B2/T129/D7, with multiple partial-reduction tiles and a tail. It checks projection outputs, stream gradients and every weight gradient for contiguous and strided streams, weights and adjoints, including zeros. Each weight-gradient result must repeat bit-for-bit across three evaluations. The complete model gate uses the configuration's explicit candidate selector. `ohc-benchmark` selects a three-arm paired comparison against `fused_read_write` when either new selector is requested, and also checks three bitwise-identical gate-weight gradients at the measured B/T/D.

For B8/T512/D256 the custom projection retains the existing 256 KiB projected output and allocates a 256 KiB projected adjoint plus 2 MiB of temporary weight partials during backward. Neither new selector adds parameters. The combined model has 14 stages (eight mixer/MLP, three wavelet and three Lngram), so retained stream-input tensor sizes sum to 224 MiB; this is not an allocator peak. Readout and stream initialization are unchanged.

### Gate-projection correctness and reproduction

Both Vulkan reference commands pass **93 component cases** with the unchanged `2e-5 + 3e-4 * abs(reference)` tolerance. The dedicated projection cases have maximum error **4.511e-7** for `fused_gate` and **1.020e-6** for `transposed_gate`; all three repeated weight gradients are bitwise identical. Across the complete component suite the largest raw error remains **0.002126** in a zero-input gradient whose reference magnitude is large; every element satisfies the same tolerance.

The CPU complete-model gate and all 17 CPU component cases pass. Against the CPU model, `fused_gate` has logits/every-parameter-gradient/first-AdamW-update maximum errors **1.192e-7 / 4.917e-7 / 2.757e-7**; `transposed_gate` has **1.192e-7 / 4.694e-7 / 2.757e-7**. Both causal/document perturbation errors are zero, with internal resets, padding and tail length nine.

From the `s7-ohcgate` workspace root, the build and acceptance commands are:

```sh
../build.sh s7-ohcgate
timeout 180s /tmp/delta-bins/s7-ohcgate/delta-experiment ohc-reference artifacts/step7/ohcgate/reference-cpu.json
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 240s /tmp/delta-bins/s7-ohcgate/delta-experiment ohc-reference artifacts/step7/ohcgate/reference-fused.json
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 240s /tmp/delta-bins/s7-ohcgate/delta-experiment ohc-reference artifacts/step7/ohcgate/reference-transposed.json
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 600s /tmp/delta-bins/s7-ohcgate/delta-experiment ohc-benchmark artifacts/step7/ohcgate/paired-ohc.json --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 300s /tmp/delta-bins/s7-ohcgate/delta-experiment ohc-attribution artifacts/step7/ohcgate/paired-ohc.json
```

The measured local-package development build takes **7.61 seconds** (zero lock wait). There are no dependency, feature, profile or crate-layout changes. The only build warnings are the existing `Device::ndarray` deprecations.

### Combined measurement and selection

`artifacts/step7/ohcgate/paired-ohc.json` is a copy of `artifacts/integration/step7/paired-ohc.json`, changing only `artifact_dir` and the explicit oHC selector. Wavelet and Lngram remain `fused_gpu` on blocks 0–2, alongside all eight mixer/MLP oHC stages. The benchmark uses identical model parameters, training windows, seed one and optimizer settings across independent trajectories. There are two warmups and 11 paired, rotating/reversed trials per arm, on Intel Arc MTL / Vulkan / Mesa 26.2.4, f32, fusion and autotune off. Complete steps include all gradient checks, clipping and AdamW.

| B8/T512/D256/four blocks, combined configuration | `fused_read_write` | `fused_gate` | `transposed_gate` |
|---|---:|---:|---:|
| Complete guarded-step median, ms | 2026.663 | **1763.317** | 1835.388 |
| Paired baseline-minus-candidate median, ms | — | **263.346** | 198.485 |
| Paired difference MAD, ms | — | 15.492 | 13.738 |
| Candidate wins | — | 11/11 | 11/11 |
| Complete-step launches | 5592 | 5550 | 5578 |
| Isolated stage pipeline F, ms | 19.904 | 16.097 | **15.007** |
| Isolated stage pipeline F+B, ms | 83.111 | **64.317** | 67.458 |
| Isolated stage pipeline F / F+B launches | 61 / 196 | 60 / 193 | 61 / 195 |

The isolated pipeline includes stream initialization, one full read/write with `F(u)=0.2u`, and learned normalized readout; it is not the read primitive alone. At the actual B8/T512/D256 shape, all three gate-weight gradients repeat bit-for-bit across three evaluations. The complete benchmark finishes in 87.86 seconds. Sampled pool reservation is 1,609,421,376 bytes across all scratch trajectories and allocator reuse, not single-model peak memory.

The same two-warmup/11-trial attribution protocol gives:

| Resident per-stage piece | F, ms | F+B, ms | F / F+B launches |
|---|---:|---:|---:|
| Original gate GEMM only | 5.711 | 23.553 | 1 / 13 |
| Original fused flattened RMS only | 0.732 | 1.585 | 1 / 11 |
| `fused_gate` projection plus flattened RMS | 2.535 | **6.787** | 1 / 13 |
| Transposed library projection plus flattened RMS | **1.889** | 9.170 | 2 / 15 |
| `fused_read_write` complete read | 8.332 | 33.764 | 30 / 87 |
| `fused_gate` complete read | 4.307 | **14.587** | 29 / 84 |
| `transposed_gate` complete read | **3.482** | 17.543 | 30 / 86 |
| Existing fused per-stream RMS plus weighted read | 1.076 | 3.746 | 1 / 12 |
| Existing fused quaternion map | 0.289 | 0.632 | 1 / 11 |
| Readout, once per model | 9.044 | 38.913 | 24 / 96 |
| Tensor gate RMS and sigmoid epilogue | 1.715 | 9.156 | 32 / 100 |

Attribution F+B launch counts include each piece's own square-sum objective and gradient traversal, not just its custom kernels. The custom projection itself has one forward and three backward launches. Rows with different objectives or independent intermediates are not additive timings. Readout, gate sigmoid epilogues and stream initialization remain unchanged; the main measured gain comes from replacing the badly shaped projection and its backward, not from eliminating all elementwise launches.

**Decision:** select explicit **`ohc_backend: "fused_gate"`** for this combined B8/T512 workload. Its paired complete-step gain is **263.346 ms**, approximately **13.0%** of the same-run baseline median, with 11/11 wins and a median difference well above twice its MAD. It also beats the transposed candidate in all 11 paired trials: median transposed-minus-fused **76.373 ms**, MAD **16.725 ms**. Keep `transposed_gate` as the measured library alternative; it is faster forward-only but slower for training. Existing selectors and defaults are unchanged. The measured run does not include an oHC-off arm, so it does not establish a new architecture-overhead number; the historical 2514 ms combined baseline is not a paired comparison with this run.

Both enabled candidates have 7,082,167 parameters, equal to `fused_read_write`. Relative to the same combined architecture without oHC, the 14 stages plus initialization/readout add 239,631 parameters (958,524 f32 bytes); neither candidate changes that cost. This result selects an execution path for the existing small Step-7 workload, not a quality conclusion or authorization to scale it.
