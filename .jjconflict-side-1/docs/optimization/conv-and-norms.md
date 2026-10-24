# Short causal convolutions and batched normalization/gating

## Decisions and scope

**Adopt the explicit `convolution_backend: "fused_gpu"` at the intended B8 workload.** Its B8/context512 complete-step paired saving is 69.933 ms, positive in all 15 trials, with a 96.48% sign-order interval of [54.993, 81.710] ms. The padded-tensor reformulation is rejected as a recommendation: it removes slice assignments and some launches but is slower at B8. Both candidates remain explicit, reproducible comparison choices; omitted `convolution_backend` remains the original `tensor` loop, including its serialization/fingerprint behavior.

This evaluation changes neither parameter storage/IDs nor training arithmetic outside short convolution. SiLU remains on the existing Burn primitive. It does not enable Burn automatic fusion or matmul autotune, alter tolerances, implement later plan components, or demonstrate larger K/V/context support. Normalization measurements change only the existing normalization selector; convolution measurements keep tensor normalization. Their savings must not be added as a measured combined improvement.

## Baseline and designs

The saved baseline is the current merged-projection, fused-Kron/fused-HOLA byte LM from `configs/lm512-projections.json`: Intel Arc MTL integrated GPU (`spirv_32902_32085`), Vulkan, Burn 0.22/CubeCL 0.11, f32, seed 1, four D256 layers, H4/K32/V64, convolution width4, chunk16, HOLA capacity16, vocabulary257, pinned corpus, AdamW learning rate .001, weight decay .01 and norm clip1. Saved B1/B8 configs under `artifacts/conv-and-norms/` change only batch and measurement bookkeeping (`artifact_dir`, `budget_file:null`, `max_seconds:300`). Contexts are selected explicitly by CLI.

The historical B8 profile attributed 80.529 ms forward+backward to convolution (4.99% of a 1,613.667 ms guarded step). This is prioritization evidence from `RESULTS.md`, not an interchangeable control for this session's timings.

Three production paths are compared with identical weights and masks:

1. **`tensor`**: the historical per-lag source slice, document equality, two validity multiplications, weighted contribution, output slice, add and `slice_assign` graph.
2. **`padded_tensor`**: the same per-lag weighted/masked contribution, left zero-padding by concatenation, then a sum of full-length contributions. It eliminates the output read/slice/assignment chain without constructing a `[batch,time,lag,channel]` tensor or using a dense convolution across documents. The extra concatenation/backward traffic fails to pay for itself at B8.
3. **`fused_gpu`**: one linear-index depthwise forward kernel, one reverse-lag input-VJP kernel, and one deterministic weight-VJP reduction kernel. The latter uses one 64-lane workgroup per lag/channel and 256 bytes shared reduction memory; no atomics or global partial-gradient scratch. Both forward and backward test source/destination validity and equality of document IDs produced by the unchanged `cumsum(reset * valid)` convention. A reset at the destination therefore prevents reading the preceding document; a reset on padding does not advance the document. Source, weight, metadata and output-adjoint strides are respected. Metadata is explicitly nondifferentiable. f32/device/shape/logical-length/storage-span checks precede allocation and launch; no silent fallback.

The selector propagates through model construction, full-model reference-check candidate selection and component attribution. Existing Q/K/V convolution weights retain shape `[4,channels]`, names, initializers and parameter identities. Omitted/default selector is skipped in serialized configs; added identity regressions cover default round-trip and nondefault fingerprint differences.

## Measurement protocol

The reusable `convolution-benchmark` extends the existing projection/normalization comparison harness rather than introducing another training-step implementation. Each mode has independent identical-start ordinary and section-barrier model/optimizer trajectories. All use the same real batches and step indices. Six-order rotation interleaves the three modes, and ordinary/sectioned order alternates. Two warmups precede 15 trials. Ordinary samples include preparation/upload, forward, real loss/backward, every finite-gradient check/readback, clipping and AdamW. Initialization/JIT, oracle checks, reports and checkpoint I/O are excluded. Section barriers and launch observation are outside ordinary timings.

Intervals below are the fourth through twelfth sorted within-trial savings of 15 pairs: nominal 96.48% sign-order coverage under independent pairs. They are not a confidence interval on a ratio of marginal medians. Absolute results from different processes are not mixed as controls.

Resident isolated convolution fixtures execute all four layers' Q/K/V channel widths128/128/256, with fresh gradient-requiring input/weight leaves, sum roots in both forward and forward+backward modes, all input/weight VJPs extracted, and outer device synchronization. F/FB order alternates. These fixtures use all-valid/no-reset shape-cost inputs; independent numerical checks separately cover document/padding cases. Isolated scopes exclude projections, SiLU, recurrence, HOLA and optimizer/checks.

Every measured invocation uses the private copied executable and private workspace `.cache/cubecl`, after the fixed build helper cleans only the three local packages inside GPU→BUILD locks. All GPU commands use an exclusive GPU lock and a hard compute timeout at most300s **inside** the lock, so queue wait is not mistaken for GPU execution. No concurrent build or GPU run overlaps these trials.

## Convolution complete guarded steps

All times are milliseconds. Savings compare a mode with its own session's tensor control.

| Batch | Context | Mode | Step median | Paired saving | Faster pairs | Paired interval | Kernels/step |
|---:|---:|---|---:|---:|---:|---|---:|
| 8 | 512 | tensor | 1462.890 | — | — | — | 5833 |
| 8 | 512 | padded tensor | 1482.540 | -11.391 | 6/15 | [-34.341,14.410] | 5617 |
| 8 | 512 | fused GPU | 1402.466 | 69.933 | 15/15 | [54.993,81.710] | 4621 |
| 8 | 128 | tensor | 345.728 | — | — | — | 5833 |
| 8 | 128 | padded tensor | 348.257 | -6.301 | 5/15 | [-18.074,4.371] | 5617 |
| 8 | 128 | fused GPU | 334.002 | 12.003 | 14/15 | [4.690,21.851] | 4621 |
| 1 | 512 | tensor | 567.626 | — | — | — | 5780 |
| 1 | 512 | padded tensor | 560.255 | 12.633 | 10/15 | [-8.636,30.018] | 5600 |
| 1 | 512 | fused GPU | 546.101 | 11.784 | 11/15 | [-0.814,42.517] | 4604 |
| 1 | 128 | tensor | 184.773 | — | — | — | 5780 |
| 1 | 128 | padded tensor | 179.461 | 5.294 | 12/15 | [0.201,8.956] | 5600 |
| 1 | 128 | fused GPU | 174.668 | 8.842 | 12/15 | [0.356,15.239] | 4604 |

The B8/context512 fused paired point estimate is about4.8% of its control median, with an interval excluding zero. B8/context128 corroborates the direction. B1/context512 remains inconclusive under the same interval criterion; the decision is explicitly based on the intended B8 workload, not a claim of a proven B1/context512 speedup. Lower dispatch count alone does not justify padded tensor adoption.

### Isolated convolution costs

All rows sum actual execution of the four layers within each trial before the median. Entries are forward / forward+backward milliseconds.

| Batch | Context | Tensor | Padded tensor | Fused GPU |
|---:|---:|---|---|---|
| 8 | 512 | 18.421 /77.322 | 17.073 /86.399 | 5.199 /19.160 |
| 8 | 128 | 7.226 /19.548 | 8.180 /21.210 | 2.095 /5.249 |
| 1 | 512 | 8.416 /16.685 | 7.214 /14.503 | 1.657 /3.757 |
| 1 | 128 | 7.515 /15.631 | 7.941 /13.354 | 1.398 /2.449 |

The B8/context512 isolated FB reduction is4.04x, **not** a4.04x whole-model gain. The ordinary integrated comparison above is the adoption evidence.

## Convolution numerical checks and memory/work

- Independent scalar-f64 formulas and analytic VJPs pass **72 Vulkan cases**, across all three paths: outputs, input gradients and every lag/channel weight gradient. Largest absolute error3.6583e-6; unchanged criterion `2e-5 + 3e-4*abs(reference)`. Cases include B2, lengths1/2/7/17/19, width1/4, channels3/35/128/256, short/tail grids, internal/tail invalid positions, valid resets and resets on padding, zero inputs/adjoints and zero weight entries. Both contiguous and genuinely transposed input/weight/valid/document/adjoint storage are exercised. No CPU transfer is part of production convolution.
- Integrated benchmark preflights pass unchanged tolerances for logits, **every model parameter gradient**, and a complete AdamW update at B1/B8 and both contexts. Across these convolution reports, fused logits are identical to the tensor control; maximum fused gradient/update errors are1.371e-6/1.722e-6. These are GPU historical-path equivalence checks, not a substitute for the parent-owned CPU full-model gate.
- Each fused complete step issues12 convolution forwards,12 input VJPs and12 weight VJPs: **36 custom convolution launches**. Total B8 step launches fall5833→4621 (1212 fewer); B1 falls5780→4604 (1176 fewer). Histograms are separate warm steps, not timing instrumentation.
- Work is linear: each pass visits at most `B*T*C*width` taps, with no token-quadratic or expanded lag tensor. For all four layers, `Cq+Ck+Cv=512`; the aggregate convolution activation payload is `4 layers * B*T*512*4` bytes: **32MiB at B8/512**,8MiB at B8/128,4MiB at B1/512 and1MiB at B1/128. Weight and weight-gradient payloads across all convolutions are32KiB each. These are logical aggregate tensor sizes, not an assertion that all buffers are simultaneously live.
- Custom autodiff retains original input/weight/mask/document handles, not per-tap intermediates or another saved output. It allocates the real output/input-gradient/weight-gradient shapes and uses256bytes shared memory per weight-reduction workgroup. The following SiLU's storage is unchanged. Tensor/padded paths retain their primitive autodiff graphs and materialize per-lag contributions; no exact allocator-traffic count is claimed.
- Sampled shared-pool reservations include all six independent model/optimizer trajectories, isolated fixtures and allocator reuse. They are approximately861.10MB at B8/512,377.69MB at B8/128,284.88MB at B1/512 and221.97MB at B1/128, essentially shared across the modes in each case. They are **not** mode-specific live temporary memory or physical GPU peak; no measured peak-memory reduction is claimed.

## Build identity and reproducible evidence

Evidence root: `artifacts/conv-and-norms`. `build-proof.json` records the fixed helper, source hashes, private-cache condition and exact measured binary SHA256:

`0768818dac31d3aac7542e4a1fe837edb4a044bcac4cf7e4fed39f0cd1fc5841`

The successful fixed-helper build took3.77s Cargo time,225.30s process wall including221s lock wait. Initial failed attempts are retained as failures: the first new report implementation mistakenly used unavailable `serde_json` in mixers (replaced by existing typed `Serialize` reports), and a pre-fix shared-target build reused a sibling's stale local-crate artifact. The parent fixed the helper to clean local packages within both locks. **No pre-fix binary contributes measurement evidence.** No dependency, feature default, tolerance or historical config/checkpoint was changed.

Convolution reports:

- B8/512: `b8/convolution-benchmark-2782067-1792867593902356909/report.json`
- B8/128: `b8/convolution-benchmark-2799819-1792867763525592786/report.json`
- B1/512+128: `b1/convolution-benchmark-2805341-1792867808674100273/report.json`
- Independent oracle: `b1/convolution-reference.json`

Exact completed commands, run from the `conv` workspace root:

```sh
../build.sh conv
sha256sum /tmp/delta-bins/conv/delta-experiment crates/kernels/src/conv.rs crates/kernels/src/conv_ops.rs crates/mixers/src/convolution.rs crates/mixers/src/convolution_checks.rs crates/mixers/src/lib.rs crates/experiment/src/config.rs crates/experiment/src/profile.rs
flock /tmp/delta-block-lm-gpu.lock timeout 120s /tmp/delta-bins/conv/delta-experiment convolution-reference artifacts/conv-and-norms/b1.json
flock /tmp/delta-block-lm-gpu.lock timeout 300s /usr/bin/time -f 'GPU execution wall: %e s' /tmp/delta-bins/conv/delta-experiment convolution-benchmark artifacts/conv-and-norms/b8.json --contexts 512 --warmups 2 --trials 15
flock /tmp/delta-block-lm-gpu.lock timeout 240s /usr/bin/time -f 'GPU execution wall: %e s' /tmp/delta-bins/conv/delta-experiment convolution-benchmark artifacts/conv-and-norms/b8.json --contexts 128 --warmups 2 --trials 15
flock /tmp/delta-block-lm-gpu.lock timeout 240s /usr/bin/time -f 'GPU execution wall: %e s' /tmp/delta-bins/conv/delta-experiment convolution-benchmark artifacts/conv-and-norms/b1.json --contexts 512,128 --warmups 2 --trials 15
```

Corresponding convolution compute/process-wall seconds:169.62/340.16,45.14/379.61 and94.03/470.12. Differences primarily reflect lock queue wait; report-internal elapsed times169.44/44.95/93.84 exclude that queue. The oracle process took14.69s including process startup/any queue.

Broad existing recurrence/HOLA/CPU full-model and historical checkpoint gates belong to the parent's combined integration acceptance, by explicit coordination: redundant queued full matrices were cancelled rather than rerun. In particular, `kron-model-check` must cover CPU logits, every parameter gradient and AdamW; historical `artifacts/lm512-vulkan/kronlstm-hola-seed1/checkpoint-13` must load with original config/fingerprints/tolerances. The parent has the existing `--diagnostic-output` option to route those results to a fresh integration directory without altering checkpoint identity. This slice does not claim those unexercised parent checks passed. Parent should also run the combined workspace tests and formatting check once after all slices land; new unit tests cover the CPU oracle, no CPU fallback and selector identity.
