# Step 7: causal WaveletDyn

## Source and equations

The implementation ports `dre-wavelet/src/lib.rs` and its independent sequential f64 test oracle from `/home/heefoo/Documents/code/diffusion-relation-extractor`. The migration inputs are also retained under `reuse/crates/dre-wavelet`. The source design is WNet-Dyn, [WNet, arXiv:2610.04720v1](https://arxiv.org/html/2610.04720v1), equations 6–7 and appendix B. The db10 coefficient table is from [PyWavelets](https://github.com/PyWavelets/pywt/blob/c9b542b2d2c37d28ed1d9bd49d178d2769608905/pywt/_extensions/c/wavelets_coeffs.template.h). Coefficients are fixed and normalized to unit sum, not trainable.

For masked input U, `V = U W_i`; `A_0 = V`; `A_j = smooth(A_{j-1}, dilation=2^(j-1))`; `D_j = A_{j-1} - A_j`. WNet-Dyn uses **J=3**, with bands `[A3,D3,D2,D1]` (Appendix C/Table 4; J=4 belongs to WNet-G). Independent sigmoid gates `G = sigmoid(U W_g + b)` have eight contiguous channel groups. Output is `((sum_j G_j * B_j) * (U W_v)) W_o`. The three square projections have no bias; the gate has a bias. Coarse/detail bias probabilities remain 0.88/0.12. Projection initialization stays Burn Linear's default: the paper does not specify a different initializer.

## Causal adaptations

- The predecessor is a bidirectional document encoder. Its centered convolution reads both past and future, with repeated symmetric extension at both document edges. That algorithm is not used here.
- At level j, tap k reads **only `t - k * 2^j`**. A tap before its document start contributes zero. This is zero extension, not reflection, wraparound, repetition of the first sample, or renormalization of the available taps. It provides a simple finite impulse response and an input VJP that is the corresponding masked transpose, without a special high-fan-in first-token gradient.
- This changes the filter phase versus the centered WNet/predecessor implementation. Constants have a startup transient: unlike symmetric extension, coarse bands are not constant and detail bands need not vanish at the beginning of a document. Telescoping reconstruction still holds, including at boundaries.
- There are three levels, dilations `[1,2,4]`, and four bands. Maximum causal history is 133 preceding tokens: `1 + 19*(1+2+4) = 134` positions including the current token. Short documents do not borrow history from adjacent documents.
- Document IDs are device tensors `cumsum(reset * valid)` along time. Both source and destination must be valid and have equal document IDs. Batch rows never interact. Padding is replaced with zero before projections, including NaN padding.
- This repository uses f32; no predecessor mixed-BF16 projection branch was introduced. There is no host `DocumentLayout`, host tap-index construction, or device-to-host layout transfer in forward.
- `wavelet_layers` selects block indices. Selected blocks use dedicated **affine LayerNorm**, `gamma*(X-mean(X))/sqrt(mean((X-mean(X))^2)+epsilon)+beta`, before WaveletDyn, matching Eq. (2). Gamma starts at one and beta at zero; the local numerical epsilon remains `1e-6`. Ordinary, reconstructed, cached, batched and marked-tail execution use the same shared layer operation, including when the recurrent normalization backend is fused. The combined model retains three injections `[0,1,2]` and wavelet → Lngram → KronLSTM/HOLA → MLP until the combined placement is resolved; this is not a claim to reproduce standalone WNet's wavelet → FFN block schedule. Each oHC-enabled added sublayer keeps its own read/write stage.
- Reusing Burn's LayerNorm was chosen over expressing centered normalization through RMSNorm or adding another custom reduction kernel: it directly supports mean, population variance, both affine parameters and native backend derivatives without another implementation to maintain.

## Configuration and execution

`wavelet_layers` defaults to `[]`; `wavelet_backend` defaults to `"tensor"`. Disabled blocks allocate no wavelet parameters or normalization parameters. The shared model identity is now `model_math_revision: "paper-local-math-v2"` in training/pilot identities, `state.json` and the model burnpack metadata. J4/pre-RMSNorm checkpoints are incompatible, as are older same-shape model equations without this revision; model loading requires exact saved parameter names/shapes and never inserts missing LayerNorm bias values. Use a fresh run, pilot and checkpoint. Duplicate/out-of-range layers and incompatible width are rejected. The fused selector requires enabled layers and an explicit non-fusion GPU backend.

Two execution designs retain exactly the same parameters:

1. **Tensor port:** gather each causal tap using index selection, mask invalid/cross-document taps, and accumulate weighted taps. Position and validity metadata are shared across taps at each level. It is the default and the CPU implementation.
2. **`fused_gpu`:** one fixed-filter causal dilated depthwise CubeCL kernel per level. Its analytic input VJP is one transpose kernel per level; it has no filter gradient and retains only valid/document metadata, not the signal. It honors signal, metadata, and upstream strides. Constants are compiled into the kernel, so there are no coefficient uploads, device filter tables, or trainable filter parameters. Projection, sigmoid, grouped gating and band accumulation stay library operations.

The fused operation is a first-order f32 GPU path; no CPU or higher-order fallback is hidden behind the selector.

## Correctness protocol

`wavelet-reference` checks both implementations on GPU and the tensor implementation on CPU. The independent f64 oracle is the predecessor's plain sequential matrix/wavelet forward and analytic backward, adapted independently to causal zero extension. Only the fixed coefficient table is shared. Its analytic VJP is itself checked by central differences along three directions covering the input and all parameter tensors.

Elementwise tolerance is **`2e-5 + 3e-4 * abs(reference)`**, without relaxation. Checks include every output, every input derivative, all four projection matrices and gate bias, all four J3 bands and telescoping reconstruction, the exact 134-position causal support, coarse/detail bias probabilities, zero signals, NaN padding, document resets, one-token/short documents, long 317-token tails, and noncontiguous inputs. Additional direct smoother tests exercise strided signal, metadata and upstream VJP at dilations 1, 2, 8 and 128. Causality tests perturb an interior token or later document and require exactly unchanged earlier/other-document outputs.

The CPU-model gate compares a `[B2,T17,D16]` one-block byte LM against the identical CPU tensor model: vocabulary logits, every parameter gradient and one AdamW update, using the same tolerance. This is separate from the independent f64 component oracle. Unit references additionally compare LayerNorm output/input/gamma/beta VJPs to scalar float64 equations (including a constant row), and compare ordinary, reconstructed, cached, batched and marked-tail shared-layer outputs and all parameter gradients against independent centered-variance equations with non-default affine parameters and active Lngram memory/smoothing.

## Memory accounting

At D256/J3/groups8, each WaveletDyn has `3*256^2 + 32*256 + 32 = 204,832` parameters (819,328 f32 bytes); its affine pre-LayerNorm adds 512 parameters. Three injections add 616,032 parameters including norms, or 2,464,128 f32 bytes. AdamW first and second moments add twice those parameter bytes; gradients are additional. Wavelet has no learned lookup tables and no device coefficient allocation.

At B8/T512/D256 one signal is 4 MiB. Four full-resolution band tensors occupy 16 MiB per injection; grouped gate outputs occupy 512 KiB. [INFERENCE] The main retained full-mixer signal payload is approximately eight signal-sized tensors (four bands plus masked input, mixed signal, value and output-projection input), plus gates and masks: about 32.5 MiB per injection before backend-specific matmul, LayerNorm, graph, allocator and optimizer storage. This is an analytic graph estimate, not an exact allocator measurement. Native tensor gather backward additionally retains gather indices/masks; broadcasting may use views or materialized backend storage.

## Reproduction

All experiment inputs and reports are under `artifacts/step7/wavelet/`, the only shared artifact subtree written by this slice. Commands run from the workspace root. Build with `../build.sh s7-wavelet`; it acquires GPU then BUILD locks and copies the executable to `/tmp/delta-bins/s7-wavelet/delta-experiment`.

For each command below, use `flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 300s /tmp/delta-bins/s7-wavelet/delta-experiment` followed by:

```text
wavelet-reference artifacts/step7/wavelet/reference-cpu.json
wavelet-reference artifacts/step7/wavelet/reference-vulkan.json
wavelet-benchmark artifacts/step7/wavelet/pilot.json --warmups 2 --trials 11
wavelet-benchmark artifacts/step7/wavelet/target.json --warmups 2 --trials 11
wavelet-smoke artifacts/step7/wavelet/smoke.json
```

Benchmarks interleave modes in forward/reverse order. Isolated samples time a resident full WaveletDyn forward or forward+backward; integrated samples use separate matched-initialization off/tensor/fused trajectories and identical data indices, with loss, all-gradient checks, clipping and AdamW included. Common model weights are identical across off/on modes; only the added component is removed for off. Kernel launches are counted in separate warm observations, outside timing samples. Two warmups per mode are excluded and eleven samples per mode are saved. Startup, cold kernel compilation, corpus loading and checkpoint I/O are not included in warm samples. No warm isolated speedup is treated as the full-model speedup.

## Target measurements and execution decision

**Historical measurements below use the earlier J4/pre-RMSNorm model.** They remain unchanged evidence for that parameterization, not acceptance or timing of the corrected J3/affine-LayerNorm model. The correction slice did not run builds, tests or benchmarks; final integrated acceptance belongs to the integrator.


Device: Intel Arc Graphics (MTL), integrated Vulkan adapter, Mesa 26.2.4-arch1.1; f32, no Burn fusion or matmul autotuning. Seed 1, actual byte-LM training batches, D256/L4/H4/K32/V64/chunk16/cache16. The target configuration is copied from `configs/lm512-optimized.json`, with only wavelet placement and experiment-output settings added. It retains the optimized KronLSTM/HOLA/projection/convolution/normalization/update paths. Results are in `target/wavelet-benchmark.json`.

| B8/T512 measurement | Off | Tensor wavelet | Fused wavelet |
|---|---:|---:|---:|
| Isolated one-module forward median, ms | — | 71.960 | 33.601 |
| Isolated one-module forward+backward median, ms | — | 161.883 | 61.425 |
| Complete guarded step, three injections, median ms | 848.338 | 1314.626 | 1019.056 |
| Complete-step issued kernels | 1718 | 6974 | 2210 |
| Isolated forward issued kernels | — | 1236 | 40 |
| Isolated forward+backward issued kernels | — | 1704 | 116 |

For paired tensor→fused samples, median savings are 37.823 ms forward (MAD 1.590), 97.835 ms forward+backward (MAD 11.192), and **293.228 ms per full step** (MAD 7.985). Fused wins all 11 pairs in each comparison. The recorded beyond-noise rule requires median savings greater than three paired-difference MADs and wins in at least 90% of pairs; all three comparisons pass. The complete-step median is about 22.5% lower than tensor, not the isolated 62% forward+backward reduction.

Adding the three fused injections still costs time: median paired overhead versus off is 167.339 ms (MAD 12.649), with off faster in every pair. Median step times differ by 20.1%. Pool reservations sampled across the three complete trajectories reach 725,440,832 bytes (691.8 MiB), not a physical peak or component-only allocation.

**Decision:** retain tensor as the historical/default selector and select `wavelet_backend: "fused_gpu"` explicitly for the optimized wavelet experiment. Its numerical gates and both isolated and integrated gains justify the fixed-filter kernel. For this three-injection B8/T512 architecture experiment, use an additional median-step budget of 250 ms and at most 30% over the matching off trajectory before scaling. The measured paired overhead meets that local budget. This is a compute decision, not evidence of improved LM quality. Combined oHC/Lngram cost needs a separate parent experiment.

The initial tensor port before sharing position/validity metadata across taps is preserved in `initial-port-benchmark.json`: isolated F/FB medians 83.716/164.884 ms; complete-step medians off/tensor/fused 980.441/1586.035/1237.934 ms. It also favored fused in all 11 pairs. These separate runs show overall timing drift, so their differences are not an attribution of speedup to metadata reuse. The table above is the final paired comparison.

The final wrapper rebuild compiled all three local crates in 5.70 s (56.24 s including 50 s waiting for locks), within the ten-second model-edit build budget despite cleaning all local packages. No dependency or feature change was made. Cold compilation is not mislabeled as warm execution: the first Vulkan reference took 5.68 s total, including initialization, component checks and the CPU-model gate; there is no isolated cold-kernel timing. The final expanded reference took 17.58 s wall time including lock waits.

## Pilot, numerical results and learning smoke

The B1/T128 pilot retains D256/L4 and the same three injections and execution bundle. It uses the same two-warmup/eleven-trial interleaved protocol:

| B1/T128 measurement | Off | Tensor wavelet | Fused wavelet |
|---|---:|---:|---:|
| Isolated one-module forward median, ms | — | 11.909 | 2.484 |
| Isolated one-module forward+backward median, ms | — | 23.914 | 4.431 |
| Complete guarded step median, ms | 66.780 | 159.357 | 80.645 |
| Complete-step issued kernels | 1701 | 6705 | 2181 |

Paired tensor→fused complete-step savings are 78.951 ms (MAD 3.400), 11/11 wins. Adding fused versus off has a paired median overhead of 13.268 ms (MAD 3.206). Sampled allocator reservation is 144,566,528 bytes across all three trajectories. Full samples and isolated launch counts are in `pilot/wavelet-benchmark.json`.

Final numerical results:

| Gate | Cases / elements | Maximum absolute error | Maximum fraction of allowed tolerance |
|---|---:|---:|---:|
| CPU tensor versus independent f64 | 8 / 55,000 | 3.332e-7 | 0.006447 |
| GPU tensor + fused versus independent f64 | 16 / 110,000 | 4.876e-7 | 0.006447 |

Causality differences are exactly zero for both reports. The f64 oracle's central-difference errors are at most 2.219e-11 in the observed directional checks. The GPU tensor/fused CPU-model gates cover 27 parameter tensors. Fused maxima versus the CPU model are 1.937e-7 for logits, 2.384e-7 for all parameter gradients and 8.941e-7 for the AdamW-updated parameters; all pass the unchanged tolerance. CPU self-comparison is exactly zero.

**Historical learning smoke — GDN-2 removed on 2026-10-28 by user decision.** The original training budget and loss measurements below are retained unchanged, not reassigned to KronLSTM.

The matched-budget tiny byte-LM smoke uses D16/L1/B2/T32, GDN-2, no HOLA, one wavelet injection, seed 1, and learning rate 0.001. Both off and fused-wavelet models run **200 steps / 12,800 training tokens**, with identical common initial weights and training row indices. Every step checks scalar loss and all trainable gradients before AdamW. Mean training loss over the first/last 20 steps:

| Mode | First 20 | Last 20 |
|---|---:|---:|
| Off | 5.451087 | 3.215369 |
| Fused wavelet | 5.426859 | 3.225014 |

Both decrease without divergence. This is only a learning/finite-gradient smoke, not an assessment of relative quality. Full loss traces are saved in `smoke/wavelet-smoke.json`. An initial invocation was rejected before training because context 32 inherited recall association counts `[4,8,16]`, which the existing validator checks even for LM tasks. The unused recall counts were set to `[4]`; model, seed, budget and tolerance were unchanged. The rejected invocation is recorded in `rejected-smoke-configuration.json`.

`provenance.json` records the corpus-manifest, binary and Cargo.lock SHA-256 values. The selected repeatable configuration is `optimized.json`; the paired target input deliberately selects tensor so the baseline stays explicit.

The full workspace test suite and the combined oHC/wavelet/Lngram model are the parent's post-merge checks, not reported as exercised here. Run:

```sh
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 600s env CARGO_TARGET_DIR=/home/heefoo/code/delta-block-lm-ws/target-shared cargo test --workspace --features delta-experiment/vulkan
```

The generic inherited `backward-profile` component list is intentionally unchanged; `wavelet-benchmark` supplies the isolated wavelet forward/VJP measurements, while ordinary profiled model forward includes all wavelet sections.
