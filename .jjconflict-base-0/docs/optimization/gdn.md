# Stateful GDN-2 fused training candidate (Steps 1–6 only)

## Status and decision

Implemented and measured. The f64 reference gate, the CPU model oracle (with the
directional check holding HOLA cache membership fixed) and B8 full-model steps pass;
see "Measured results". At B8/T512 the complete step with HOLA falls
2,609.6 → 1,710.9 ms (1.53×). Use `scan_backend: "fused_gdn2"` when GDN-2 is trained
on a GPU. `TensorSolver` remains the omitted default; historical config/checkpoint
fingerprints and existing Kron selector meanings are unchanged. The selector is
explicit and GDN-2-only.

Three designs were considered:

| Design | Training contract | Decision |
| --- | --- | --- |
| Existing differentiable chunk tensor solver | Output, initial/final state, physical scores and all derivatives | Preserve as baseline and default |
| Existing copied GDN-2 chunk-WY CubeCL operator (`ops.rs`) | Output-only, zero initial state, integer document bounds, detached internal checkpoint | Keep its existing independent diagnostic, not a drop-in training replacement |
| Stateful fused recurrent GDN-2 with analytic VJP | Same seven float inputs and three public differentiable outputs as the tensor solver | Explicit candidate; measure against baseline before adoption |

Appending diagnostic identity queries to the WY operator can inspect state but
cannot supply arbitrary differentiable initial state or the physical edit-score
VJP. Recomputing scores/state using the tensor solver would duplicate recurrence
work, so it is not used as a purported fused implementation.

## Equations and implementation

For a valid token, reset first discards previous state, then

- `bar[i,j] = alpha[i] * previous[i,j]`
- `delta[j] = write[j] * v[j] - sum_i erase[i] * k[i] * bar[i,j]`
- `state[i,j] = bar[i,j] + k[i] * delta[j]`
- `output[j] = sum_i q[i] * state[i,j]`
- `score = norm(k) * norm(delta)`.

Padding emits zero output/score and preserves state, even if reset is set on that
padding position. Reset cuts the gradient to earlier state and decay at that token.
Zero decay is supported without division or reconstruction from an inverse gate.

The VJP seeds the state adjoint from the returned final-state objective, adds
`q * output_adjoint`, then differentiates the outer product and channel-wise read.
The score seed contributes `norm(k) * delta / norm(delta)` to the edit adjoint and
`norm(delta) * k / norm(k)` directly to the key adjoint. Both contributions choose
zero at a zero physical edit. The write gate is retained explicitly for the value
VJP; there is no division by a gate. Q/K/V/alpha/erase/write and initial state all
receive analytic derivatives. Masks are nondifferentiable; only first derivatives
are supported, matching the other custom operators.

`gdn_training_ops.rs` follows the existing Kron packed-output registration:
output, final state and scores are slices of one autodiff node, so independent and
combined losses reach one seven-input VJP. The checkpoint and edit/norm buffers
are private detached implementation storage, not substitute public state outputs.

One 64-lane workgroup handles each batch/head; each lane owns one value column.
The forward kernel checkpoints before each computational chunk. Backward replays
at most that chunk, then reverses it. Input strides are honored. Supported shape:
f32, nonempty B/T/H, K32/V64, computational chunk 1..=16. GPU-only selection and
kind/shape guards fail rather than falling back. Parameter shapes, IDs and model
architecture are unchanged. Mixer initialization, mutable selection, scan dispatch,
arm validation and training-algorithm labels all recognize the explicit selector.

## Work and memory (analytic, not measured peaks)

With C=chunk size, the fused path has one forward and one backward recurrence
dispatch, each with B*H workgroups. Tensor slicing/autodiff adds separate work and
is not included in that dispatch count. Work is O(B*T*H*K*V), sequential over time,
without a quadratic time-by-time intermediate.

F32 buffer sizes in bytes:

- checkpoint: `4 * B * H * ceil(T/C) * 32 * 64`;
- auxiliary edits/norms: `4 * B * T * H * 66`;
- bounded backward replay: `4 * B * H * C * 32 * 64`;
- packed public output: `4 * (B*T*H*65 + B*H*32*64)`;
- backward shared reductions: 32 KiB per workgroup, plus per-lane state/adjoint;
- seven gradient outputs and retained six recurrence inputs are additional.

For B8/T512/H4/C16 these four global temporary/output buffers are respectively
8,388,608; 4,325,376; 4,194,304; and 4,521,984 bytes. These are not peak-memory
measurements: allocator reuse, slice materialization, fusion and driver storage
must be sampled separately by the integration owner. Low B*H concurrency and
32 KiB shared reductions may limit throughput; this is a measurement question.

## Correctness coverage

`gdn-reference` reuses the existing independent sequential CPU f64 recurrence/VJP
in `reference.rs`, not the tensor solver as the reference. The common acceptance
fixture is parameterized by mixer kind; Kron fixture semantics remain unchanged.
Absolute tolerance stays `2e-5`, relative tolerance stays `3e-4`.

The default checks cover T1/T7/T35, B2/H4/K32/V64, all six channel-wise input
gradients plus initial-state gradients, output/state/score-only and combined
objectives, nonzero initial state, exact zero/partial-zero decay, negative edits,
zero key/value/edit cases, valid resets, reset-on-padding, internal padding,
entire padded chunks, tail chunks, genuine strided inputs and split continuation
outputs/state/scores and their gradients. The full reference suite also exercises
GDN layer derivatives for no cache, surprise and recency selection.

`gdn-benchmark` runs these strict gates before measuring. It times synchronized
isolated forward and forward/backward at B1 and B8, T128 and T512, then checks the
actual B8 model against CPU parameters, CPU logits, every parameter gradient,
and an AdamW step before timing model optimizer steps. CPU all-parameter
finite-difference directional checking is inherited from the existing model
oracle. Both no-cache and surprise-HOLA models are included; scan/HOLA pairs are
reported separately so a HOLA change cannot masquerade as a scan improvement.
Canonical CPU parameters are forked before timing; transfer/setup and shape-first
calls are separate from warm samples. Shape-first calls happen after numerical
gating and are not claimed to be cold compiler measurements. Per-case seed,
config, parameter/fixture hashes, tolerances and samples are retained in JSON.

## Integration-owner queue (reference commands)

Use the owner's single consistent CPU/Vulkan/fusion feature build, once after all
source branches land. Formatting and compilation belong to that integration pass.
For example, if the central feature set is CPU/Vulkan/fusion:

```sh
timeout 300s cargo build -p delta-experiment --features cpu,vulkan,fusion
```

Run correctness serially on the shared device; commands are individually bounded:

```sh
timeout --kill-after=5s 180s target/debug/delta-experiment gdn-reference configs/gdn-reference-cpu.json --scan-backend tensor_solver
timeout --kill-after=5s 180s target/debug/delta-experiment gdn-reference configs/gdn-reference-vulkan.json --scan-backend tensor_solver
timeout --kill-after=5s 180s target/debug/delta-experiment gdn-reference configs/gdn-reference-vulkan.json --scan-backend fused_gdn2
timeout --kill-after=5s 180s target/debug/delta-experiment gdn-reference configs/gdn-reference-vulkan.json --scan-backend fused_gdn2 --tokens 512 --objective combined
```

The full-model CPU oracle and B8 measurements are in one bounded harness:

```sh
bun scripts/benchmark-gdn.ts vulkan target/debug/delta-experiment 2 5
```

The harness uses the existing d_model256/layers4 benchmark template, generates
GDN comparison arms in an isolated `artifacts/gdn/benchmark-*` namespace, writes
its resolved config, preserves stdout/stderr and exit codes, and uses an outer
930-second timeout with a five-second kill grace. The executable also checks the
900-second config deadline between cases/trials and writes partial reports.
Do not run this concurrently with another GPU owner. A failed gate or deadline
is a retained failure, not a reason to relax tolerances or accept partial timing.

Final measurements are below.

## Measured results

### Correctness

- The f64 gate (`gdn-reference --scan-backend fused_gdn2`) passed on Vulkan.
- Full-model B8/T128 and B8/T512, HOLA off and on, pass the CPU oracle on all four
  paths (tensor/fused scan × tensor/fused HOLA): logits ≤ 4.72e-6, all parameter
  gradients ≤ 4.32e-7, AdamW ≤ 4.87e-6.

The first run stopped at the CPU directional finite-difference check for B8/T128 with
HOLA on (error 3.447e-3, limit 3.315e-3). `gdn-directional-diagnostic` showed every
tested ε (5e-4…4e-3) changing cache membership in 7–24 chunks, so the difference
crossed the hard top-k selection. The check now replays the unperturbed forward's kept
cache slots in both perturbed forwards; ε 2e-3 and the tolerance are unchanged. With
membership fixed the GDN-2 errors are 1.0e-5 / 1.0e-5 / 1.09e-4 / 5.0e-5 at the four ε
(limit 3.32e-3), and the full-model gate passes (1.09e-4 at T128, 3.16e-4 at T512).
Reports: `artifacts/integration/directional-diagnostic.stdout` (free selection) and
`artifacts/integration/directional-diagnostic-fixed.stdout` (both).

### Full-model steps

B8, five interleaved trials after two warmups, median complete step in ms
(`artifacts/gdn/benchmark-1792873067810-3535974/`):

| Context | HOLA | `tensor_solver` | `fused_gdn2` | `fused_gdn2` + fused HOLA |
| ---: | --- | ---: | ---: | ---: |
| 128 | off | 409.1 | 282.2 | — |
| 128 | on | 521.4 | 402.2 | 352.5 |
| 512 | off | 2,095.5 | 1,296.2 | — |
| 512 | on | 2,609.6 | 1,794.6 | 1,710.9 |

Every fused-scan trial was faster than every tensor-solver trial in the same case.
These are medians of five samples, not paired bootstrap intervals.

### Isolated recurrence timing

Median milliseconds, forward (F) and forward+backward (F+B), one layer:

| Shape | `tensor_solver` F | `tensor_solver` F+B | `fused_gdn2` F | `fused_gdn2` F+B |
| --- | ---: | ---: | ---: | ---: |
| B1/T128 | 19.59 | 55.40 | 5.76 | 24.26 |
| B8/T128 | 19.63 | 57.17 | 6.05 | 25.43 |
| B1/T512 | 74.82 | 261.84 | 20.12 | 87.88 |
| B8/T512 | 69.58 | 245.35 | 19.66 | 93.04 |

From the final run (`artifacts/gdn/benchmark-1792873067810-3535974/`). The fused
recurrence is about 3.4–3.7× faster forward and 2.3–3.0× faster forward+backward;
the complete step gains less (1.4–1.6×) because the rest of the model is unchanged.
