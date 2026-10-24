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
missing erase term. This is localization evidence, not yet a proven library
root cause. In the triangular formulation the corresponding branch is
`((k * prefix).matmul(initial) * v).sum_dim(3) * erase` in the RHS.
An incorrect alpha **backward** alone cannot explain a key derivative error:
alpha and key are independent inputs. Incorrect alpha/prefix **forward** values
or mutation of saved forward values could affect it.

## Primitive exclusion evidence

Frozen temporary diagnostic source: `b36e2026`.
The parent ran the unified non-fusion build and Vulkan primitive harness;
the agent did not run GPU, tests, builds, linters, or formatters.
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

The parent observed the production focused key gradient
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

The bounded commands, run serially by the parent under the existing GPU then
build locks, are:

```sh
timeout 180s cargo build --workspace --features delta-experiment/vulkan --example tensor_solver_diagnostic
timeout 120s target/debug/examples/tensor_solver_diagnostic vulkan cut
```

The executable also accepts `cpu cut` for a matched backend control.
The `cut` argument skips both broad primitive and earlier chain probes.
At this source checkpoint there is no claimed fix and no production math change.

## Independent regression behavior

`chunk_tests.rs` uses a separate small, non-normalized arithmetic fixture
(B2/T5/H2/K3/V4), not the saved trigonometric fixture. It covers both KronLSTM
and GDN-2, chunk sizes 1/2/5/16, a reset on padding between active tokens,
a subsequent valid reset, channel-zero decay, zero write, and signed updates.
The independent sequential f64 oracle supplies output, key VJP and initial-state
VJP expectations. The fixture also asserts nonzero initial-state influence on
the first key gradient, zero input/output effect on padding, and discarded
initial-state gradients for a row that resets at its first active token.

The parent can run the normal CPU test and the explicitly ignored GPU test
using the same unified workspace feature set:

```sh
timeout 180s cargo test --workspace --features delta-experiment/vulkan tensor_solver_initial_key_reset_padding_cpu -- --test-threads=1
timeout 180s cargo test --workspace --features delta-experiment/vulkan tensor_solver_initial_key_reset_padding_vulkan -- --ignored --test-threads=1
```

These tests have not been executed at this checkpoint. The GPU command must use
the repository's GPU/build lock ordering. Existing strict Kron acceptance,
including all input fields, initial state, continuation, and strided inputs,
remains unchanged and is still required for acceptance of a correction.
