# Tensor-solver initial-state key-gradient investigation

## Scope and retained failure

The common differentiable triangular solver is `crates/mixers/src/chunk.rs`.
This investigation does not change scan selection, historical configuration defaults,
checkpoint parameters, or the numerical acceptance thresholds (`atol = 2e-5`,
`rtol = 3e-4`). The saved failure, not a rerun to confirm it, is the baseline:

`artifacts/kron-validation/solver-output_only/kron-reference.json`

It records non-fusion Vulkan on Intel Arc MTL, TensorSolver,
B2/T7/H4/K32/V64/chunk16, output-only VJP, nonzero initial state, zero alpha,
zero/signed corrections, padding and resets. Key field 1 element 32
(b0/t0/h1/k0) is `0.0020814966`, versus independent f64
`0.002235880982479412`. Its absolute error is `0.00015438434143956372`.
CPU and the custom fused scan passing do not certify the GPU tensor solver.

The accepted source correction is `e1cb69ce`: an equivalent rowwise dot GEMM
for the KronLSTM initial-state bilinear term. Its independent CPU/GPU regression
and historical/default Vulkan reference checks pass in the final integration
verification recorded below. The earlier rank-2 reduction attempt is retained
as a failed intermediate correction, not an accepted alternative.

## Algebraic localization

For a valid token, let `bar = diag(alpha) S_previous`,
`z = write - beta (k^T bar v)`, and `S = bar + z k v^T`.
With state adjoint `A`, the output-only key derivative is

```
dz = sum_{i,j} A[i,j] k[i] v[j]
dk[i] = sum_j A[i,j] z v[j] - beta dz sum_j bar[i,j] v[j]
```

A scalar reconstruction of the exact fixture at the failed coordinate separates:

| Contribution | Value |
| --- | ---: |
| Direct edit term | 0.0020814966478405863 |
| Initial-state erase term | 0.0001543843342999027 |
| Sum | 0.002235880982140489 |

The saved GPU value matches the direct term alone; the residual matches the
missing erase term. By itself this localized the failure but did not establish
the library cause. In the triangular formulation the corresponding branch is
`((k * prefix).matmul(initial) * v).sum_dim(3) * erase` in the RHS.
An incorrect alpha **backward** alone cannot explain a key derivative error:
alpha and key are independent inputs. Incorrect alpha/prefix **forward** values
or mutation of saved forward values could affect it.

## Primitive exclusion evidence

Frozen temporary diagnostic source: `b36e2026`.
The unified non-fusion build and Vulkan primitive harness ran serially.
Reported compile time was 3.02 s and full transaction time 5.16 s.
All probes passed the unchanged tolerances:

- Ten small GEMM geometry/layout/broadcast cases, including 7x32x64,
  7x64x7, 7x7x7, broadcast RHS, and 7x7x1. Both operand VJPs were checked
  independently against scalar f64. Largest absolute error: 1.29629e-7.
- Masked/reset prefix construction: forward 8.50448e-8, alpha VJP 1.35292e-8.
- Pairwise decay with repeated `slice_assign`: forward 8.50448e-8,
  alpha VJP 4.59085e-8.
- Strided masked key overlap with triangular masking: separate query/key VJPs
  1.42755e-9 / 4.07332e-9; shared-key VJP 5.88639e-9.

These rule out simple isolated primitive failures for the tested inputs.
They do not prove correctness of the composed recurrence graph or exclude
saved-value/index aliasing when that graph shares tensors.

## Exact-chain discriminator

Frozen temporary diagnostic source: `36db24d2` (child of the primitive probe).
It reconstructs the saved fixture without changing production recurrence code:

- The real production graph reports all key VJPs and forward finite differences
  at b0/t0/h1/k0 for two perturbation scales.
- A copied diagnostic graph partitions the key's dependencies into RHS-only
  and all-except-RHS paths using detached diagnostic inputs. This is not a
  production implementation or a proposed workaround.
- Independent f64 `grad(write)` is the RHS adjoint. Thus the expected direct RHS
  key contribution is `-beta * grad(write) * prefix * (S_initial v)`.
- The attempted non-leaf gradient-retention variant was invalid: Burn rejects
  `require_grad()` on a non-leaf. It panicked after the branch probes, and is
  not counted as a successful check.

The exact-chain run observed the production focused key gradient
`0.0020814966410398483`. Its forward finite differences were
`0.002217008899805466` at epsilon 0.001 and `0.0022392289984488624`
at epsilon 0.003, consistent with the correct forward dependency rather than
the missing backward contribution. The copied full graph had 368 failing key
elements (maximum absolute error about 7.466e-4). All-except-RHS passed every
key (maximum 2.94e-8); RHS-only returned exactly zero at the focused key versus
expected `0.00015438433429990284`, with 371 failing key elements.

The corrected suffix-cut discriminator is frozen at `f19cf74b`. It uses
`detach().require_grad()` to replace exactly one intermediate with a leaf,
then measures its suffix VJP. The cuts are masked key, key-prefix, state
projection, bilinear reduction, erase product, and RHS. Expected adjoints
come from the independent f64 RHS adjoint, initial state and value vectors.
It skips the previously exercised full-graph and finite-difference stages.

Historical reproduction commands at source `f19cf74b`, run serially under the
existing GPU then build locks, were:

```sh
timeout 180s cargo build --workspace --features delta-experiment/vulkan --example tensor_solver_diagnostic
timeout 120s target/debug/examples/tensor_solver_diagnostic vulkan cut
```

The executable also accepts `cpu cut` for a matched backend control.
The `cut` argument skips both broad primitive and earlier chain probes.
That checkpoint had no production math change. The temporary example has now
been removed; it remains available in the frozen diagnostic revisions, not
as a permanent extra executable.

## Proven failing boundary and correction

The cut run returned correct RHS, erased-product, and scalar-inner leaf VJPs.
The scalar-inner focused VJP was `0.04179788753` against f64
`0.04179788472`. Cutting at the state projection returned zero for all 64
focused channels, and consequently the key-prefix and masked-key cuts were
also zero.

The standalone reduction matrix at source `661e4e86` then passed all 72
forward/operand-VJP comparisons: T7/16, V32/64, contiguous and swapped value
inputs and scalar seeds; the T7/V64 cases also covered rank-2 reduction and
rowwise dot-matmul equivalents. Therefore this is **not** a general claim that
GPU multiplication or `sum_dim(3)` is incorrect in isolation.

The final context bisection at source `91c8ce19` established:

- A leaf substituted for the **product** `state_projection * v` also receives
  zero VJP through the original rank-4 `sum_dim(3)` in the recurrence graph.
  This excludes the multiplication's saved value, key GEMM, and alpha
  derivative as the point where this contribution disappears.
- Detaching only the erase coefficient or only the downstream inverse restores
  the projection VJP (maximum absolute error approximately 6.9e-10).
- Detaching write or the edit-value consumer, using a distinct/constant/fresh
  value for the inner product, and omitting unused final-state/score branches
  do not restore it.
- Both complete copied graphs with an equivalent rank-2 reduction or rowwise
  dot matmul pass **every** key VJP. Focused value:
  `0.0022358810529` versus f64 `0.0022358809825`; maximum key error 2.94e-8.
- Runtime checkpointing is **Disabled**. A static `BalancedCheckpointing`
  type name in one panic trace was not evidence of runtime recomputation.

The proven library defect boundary is the reduction's backward in the
erase/broadcast/triangular-solve graph. A layout/stride interaction in its
incoming singleton-width adjoint is the remaining low-level explanation to
investigate; the specific faulty CubeCL instruction has **not** been established.

The first correction (`c0a0dd9f`) flattened the product before reducing values.
It fixed the historical T7 acceptance but failed the independent chunk-size-1
regression, so it is not accepted as the complete correction (evidence below).
The successor uses the independently checked equivalent rowwise dot GEMM:

```rust
projected_initial
    .unsqueeze_dim::<5>(3)
    .matmul(v.unsqueeze_dim::<5>(4))
    .reshape([batch, heads, size, 1])
```

**Historical solver repair — GDN-2 removed on 2026-10-28 by user decision.** Comparisons to that former mixer below explain the original diagnosis only.

This computes the same row dot products, does not divide by decay, and preserves
all input gradients and existing reset/padding behavior. It is unconditional
for KronLSTM shapes, not a fixture-specific branch, detached-gradient workaround,
or dispatch to the custom fused scan. GDN-2 has no corresponding scalar
initial-state dot-product reduction and remains unchanged. The unrelated
key-overlap and norm reductions remain unchanged.

The dot GEMM removes this reduction backward from the RHS without detaching
any operand. Its tiny-matrix dispatch cost has not been benchmarked; no
performance improvement or memory-saving claim is made. No dependencies,
checkpoint parameters, public APIs, or numerical tolerances change.

## Upstream-report text

**Suggested title:** CubeCL Vulkan: composed rank-4 sum_dim VJP becomes zero
after a broadcast erase and batched triangular-solve consumer.

**Environment:** Burn/burn-autodiff/burn-cubecl 0.22.0, CubeCL 0.11.0,
cubek-reduce 0.3.0 from the lockfile; float32, Intel Arc MTL Vulkan,
non-fusion, non-autotuned, checkpointing Disabled.

**Reproduction:** Use frozen source `91c8ce19` and run the unified example build
followed by `timeout 120s target/debug/examples/tensor_solver_diagnostic vulkan context`.
The example is intentionally absent from the final production tree. Its exact
fixture is independently differentiated in f64 and the `context` mode skips
the already-run broad/finite-difference probes. The failing suffix is:

```
product_leaf: [2,4,7,64]
inner = product_leaf.sum_dim(3)
rhs = write - inner * erase                 // erase: [2,4,7,1], tracked
correction = inverse.matmul(rhs)            // inverse: [2,4,7,7], tracked
edits = correction * value
output = initial_output + overlap.matmul(edits)
loss = (output * independent_output_seed).sum()
```

**Expected:** Nonzero product-leaf gradient equal to the scalar inner adjoint
broadcast over 64 values. **Actual:** Zero product-leaf gradient, while a leaf
substituted immediately after `sum_dim(3)` has the correct nonzero adjoint.
Detaching erase or inverse restores it; removing value sharing does not.
Replacing only the reduction by either equivalent contraction restores the
historical T7 key VJP. However, rank-2 reduction still loses the initial erase
contribution at chunk size 1 in a second independent fixture. Standalone
multiply/reduce examples pass, so a report must retain the consumer graph
rather than claim a generic sum bug.

Submission destination: https://github.com/tracel-ai/burn/issues .
No upstream issue has been filed or assigned a number in this investigation.

## First correction verification: incomplete

The combined integration tree included frozen correction `c0a0dd9f`. A matched
before run reverted only its RHS reduction while keeping the rest of that tree
identical. Retained logs:

`artifacts/kron-validation/solver-fix-verification-2/`

| Check | Result | Recorded seconds |
| --- | --- | ---: |
| Before: independent Vulkan regression | FAIL | 8 |
| Rank-2 correction: independent CPU regression | PASS | 1 |
| Rank-2 correction: independent Vulkan regression | FAIL | 2 |
| Rank-2 correction: workspace CPU tests | PASS | 3 |
| Unified executable build | PASS | 5 |
| Historical T7 output/state/score objectives | all PASS | 7 |
| Default TensorSolver Kron reference | PASS | 5 |

**Historical diagnostic ordering — GDN-2 removed on 2026-10-28 by user decision.** The original failure and numeric decomposition below are retained unchanged.

The GPU regression fails identically before and after:
`key VJP[3]: -0.0014381682 != -0.0013434548440675893`.
The fixed test iteration order identifies the first case as
KronLSTM/B2/T5/H2/K3/V4/chunk1/output-only, before any GDN-2 case.
Assertions in the successor now include the caller, dimensions and chunk size.

Scalar decomposition of this independent fixture gives:

| Key-gradient term | Value |
| --- | ---: |
| Direct edit | -0.001438168312022801 |
| Initial-state erase | 0.0000947134679552117 |
| Sum | -0.0013434548440675893 |

The missing term is again exactly the initial-state erase contribution, not
a GDN-2-specific formula or a new tolerance issue. The rowwise dot-GEMM
successor changes only the same KronLSTM RHS contraction. Its production
acceptance must be recorded separately; the successful T7 tests do not excuse
the failed independent regression.

## Rowwise dot-GEMM acceptance

Frozen solver source: `e1cb69ce` (child of the incomplete rank-2 correction).
The final verification exercised the combined integration source
`53a684349a1a4cb3be5bda1f7ab2dcdf338529b9`, which includes that solver change.
Evidence is recorded in a child revision, without rewriting either tested
source snapshot.

Logs: `artifacts/integration/final/`. Tested experiment binary SHA-256:
`d94d0896930d2022b5db42ed0b9efd24b16094312fcee8c5d3e2c7e304856e10`.
The before-regression failure remains in
`artifacts/kron-validation/solver-fix-verification-2/before-vulkan-regression.stdout`.

| Check | Result | Recorded seconds |
| --- | --- | ---: |
| Unified executable build | PASS | 5 |
| CPU workspace tests | PASS: 33 + 7 + 13 tests | 8 |
| Independent Vulkan key/initial/reset/pad regression | PASS: 1 test | 3 |
| Historical T7 output/state/score objectives | all PASS | 7 |
| Default TensorSolver Kron reference | PASS: 13 cases | 4 |

The CPU workspace run includes
`tensor_solver_initial_key_reset_padding_cpu`. The explicitly run GPU test
is `tensor_solver_initial_key_reset_padding_vulkan`; its same independent
fixture fails before both corrections, fails after the rank-2 attempt, and
passes after the rowwise dot GEMM. Neither the fixture nor its tolerances were
weakened. The GPU test covers both solver callers and chunk sizes 1/2/5/16.

Historical objective JSON reports are preserved separately under
`artifacts/kron-validation/solver-final-1792870523144/`:

| T7 objective | Input VJP max abs | Initial-state VJP max abs | Continuation VJP max abs |
| --- | ---: | ---: | ---: |
| output_only | 6.85548e-8 | 1.64059e-8 | 8.07143e-8 |
| final_state_only | 2.93909e-9 | 0 | 3.46227e-9 |
| score_only | 6.78971e-9 | 3.53416e-10 | 8.96639e-9 |

The default reference additionally covers T1/T7/T35 and all four objectives,
plus strided T35 combined inputs and continuation gradients. Every case
reports `scan_backend: tensor_solver`; the largest reported input-gradient
absolute error is 1.43901e-7. The device reports Intel Arc MTL Vulkan with
fusion and autotune disabled.

This accepts the corrected solver behavior within the exercised contracts,
not a measured performance improvement or a general repair to Burn's
underlying reduction implementation. Unrelated integration checks are outside
this solver acceptance; the integration log also contains an IO-benchmark
failure, so the entire integration transaction must not be called all-pass.

## Independent regression behavior

`chunk_tests.rs` uses a separate small, non-normalized arithmetic fixture
(B2/T5/H2/K3/V4), not the saved trigonometric fixture. Current coverage uses
KronLSTM, chunk sizes 1/2/5/16, a reset on padding between active tokens,
a subsequent valid reset, channel-zero decay, zero write, and signed updates.
The original matrix also included the former reference mixer: GDN-2 removed on 2026-10-28 by user decision.
The independent sequential f64 oracle supplies output, key VJP and initial-state
VJP expectations. The fixture also asserts nonzero initial-state influence on
the first key gradient, zero input/output effect on padding, and discarded
initial-state gradients for a row that resets at its first active token.

Run the normal CPU test and the explicitly ignored GPU test using the same
unified workspace feature set:

```sh
timeout 180s cargo test --workspace --features delta-experiment/vulkan tensor_solver_initial_key_reset_padding_cpu -- --test-threads=1
timeout 180s cargo test --workspace --features delta-experiment/vulkan tensor_solver_initial_key_reset_padding_vulkan -- --ignored --test-threads=1
```

The first-correction outcomes are recorded above. The successor's checks use
the identical fixture and tolerances.
The GPU command must use the repository's GPU/build lock ordering.
Existing strict Kron acceptance, including all input fields, initial state,
continuation, and strided inputs, remains unchanged.

After the unified executable build, the following bounded command reruns the
three historical T7 objectives without overwriting any saved failure artifacts.
Only the artifact destination differs from each historical config:

```sh
timeout 180s cargo build --workspace --features delta-experiment/vulkan
timeout 180s bun -e '
import { resolve } from "node:path";
const root = resolve(`artifacts/kron-validation/solver-corrected-${Date.now()}`);
const executable = resolve(process.env.CARGO_TARGET_DIR ?? "target", "debug/delta-experiment");
let ok = true;
for (const objective of ["output_only", "final_state_only", "score_only"]) {
  const config = await Bun.file(`artifacts/kron-validation/solver-${objective}/config.json`).json();
  config.artifact_dir = `${root}/${objective}`;
  const file = `${config.artifact_dir}/config.json`;
  await Bun.write(file, JSON.stringify(config, null, 2) + "\n");
  const child = Bun.spawnSync(["timeout", "45s", executable, "kron-reference", file,
    "--scan-backend", "tensor_solver", "--tokens", "7", "--objective", objective],
    { stdout: "inherit", stderr: "inherit" });
  ok = ok && child.exitCode === 0;
}
console.log(JSON.stringify({ artifact_dir: root, all_objectives_passed: ok }));
process.exitCode = ok ? 0 : 1;
'
```

These after-change results must be recorded separately from the frozen
diagnostic evidence; the latter does not by itself certify the production edit.
