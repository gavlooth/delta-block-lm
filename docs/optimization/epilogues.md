# MLP SiLU and residual epilogues

## Scope and status

This change provides three executable designs and paired attribution for the active
Step1–6 model. No benchmark, build, formatting, or numerical gate was run by the
implementation worker. All timings below must come from the parent's frozen-source
run; no gain is asserted. The assignment's B8/T512 MLP forward/backward attribution
of **358.64 ms** is context supplied to this work, not a measurement reproduced here.
It includes GEMMs and must not be presented as removable activation overhead.

The model remains `RMSNorm -> up linear -> SiLU -> down linear -> residual add`.
Both linears retain their original parameter tensors and parameter IDs. The new
module-skipped `epilogue_backend` field changes execution only. `native` remains the
historical default and is omitted from serialized config, preserving the legacy
fingerprint input. Explicit alternatives participate in the fingerprint. There are
no added learned tensors, reshaped stored weights, biases, or checkpoint shims.
The backward attribution command also honors the selected activation.

## Three actual implementations

| Selector | Forward | Input VJP | Saved activation |
| --- | --- | --- | --- |
| `native` | Original Burn `x * sigmoid(x)` | Original Burn autodiff | Library-managed |
| `analytic_backward` | CubeBackend library sigmoid and multiply | One custom pointwise kernel | Input x only |
| `fused_pointwise` | One custom pointwise SiLU kernel | Same one-kernel analytic VJP | Input x only |

The derivative is `dy * (s + x*s*(1-s))`, with `s = sigmoid(x)`. Recomputing sigmoid
avoids retaining its output or allocating a separate sigmoid intermediate in the
VJP. The custom kernels read arbitrary rank-3 input/upstream strides and write a
contiguous result. They support first-order f32 GPU execution and nonempty positive
dimensions; the zero tests mean zero-valued inputs, not empty tensors. No CPU
fallback or higher-order derivative is implied.

For `[B,T,2D]`, retained x and newly allocated dx are each `8*B*T*D` logical bytes:
for D=256, 256 KiB at B1/T128 and 8 MiB at B8/T512. Retaining x references the
existing allocation rather than copying it. Fused forward allocates y of the same
size. These are logical tensor sizes, not physical-device peaks.

### Library GEMM reuse decision

The installed Burn 0.22.0 tensor SiLU implementation is a multiply of x and sigmoid,
not a native SiLU GEMM epilogue (`burn-tensor/src/tensor/activation/base.rs`). The
public Cube matmul interface takes `(lhs, rhs, out, strategy, out_dtype)`
(`burn-cubecl/src/kernel/matmul/base.rs`); its optional output is a destination,
not an additive residual or activation selector. Cubek 0.3.0's public `launch_ref`
accepts strategy, client, lhs/rhs, output binding and dtypes, likewise without a
residual or SiLU argument. This is an API-compatibility observation, not a claim
that no internal fusion machinery exists. No library fork or new GEMM is added.

Burn graph fusion remains the reusable library alternative when enabled in an
explicit config. Custom backend-extension calls are fusion boundaries: in
particular `analytic_backward` preserves the Cube library forward expression,
not necessarily the cross-operation fusion opportunities of the native graph.
The supplied controlled launcher sets fusion and autotune false for all three
arms. A fusion-enabled run must be identified separately, not mixed with these
samples. GEMM algorithm/autotune work belongs to the separate GEMM evaluation.

Residual addition stays the native binary add in every design. It cannot legally
be fused with SiLU across the down matmul, or with the preceding mixer output
across RMSNorm. A replacement add-only kernel offers no algebraic dispatch saving;
we measure the existing add instead of introducing an equivalent custom kernel.
The fullmodel section named `Residual adds` includes both mixer and MLP residuals;
the isolated residual fixture is a single add.

## Independent correctness gates

`epilogue-reference` evaluates independently written scalar-f64 formulas for every
SiLU output and weighted input VJP, and for residual output and both input VJPs.
Shapes `[2,7,W]` with W=1,3,35,256,512 exercise short/tail dispatches. Each uses
contiguous and transposed storage, all-zero and mixed inputs, values ±1000, ±100,
±30, ±1e-6 and zero, and a nonuniform upstream containing zeros. Explicit custom
VJP calls test strided upstream storage even when ordinary autodiff would
materialize it. A shared-parent `x+x` residual tests gradient accumulation.

GPU reference invocations check all three SiLU designs. The CPU unit test checks
only the native design. Existing absolute/relative tolerances remain **2e-5 +
3e-4*abs(reference)**, unchanged. Config regressions cover default omission,
roundtrip, explicit fingerprint changes, and CPU rejection of custom selectors.

`epilogue-benchmark` reuses the existing fullmodel comparison machinery: clone the
same initialized model, check logits, every parameter gradient and one AdamW
update, then run three independent ordinary and three independent section-barrier
trajectories on identical data/step indices. It rotates forward/reverse execution
order and retains raw samples, paired deltas, fixture hash, device identity,
config, dispatch observations and sampled pool reservations. The existing
fullmodel equivalence tolerances are unchanged. Section-barrier sums are
attribution, not ordinary-step estimates.

## Parent-run commands

Use the centrally built common-feature binary; do not independently build while
another worker owns the GPU/build locks. From the repository root, the following
are bounded examples (the parent's common feature set takes precedence):

```sh
timeout 180s cargo test -p delta-experiment --features vulkan,fusion,autotune epilogue
bun scripts/locked-gpu.ts artifacts/epilogue-gates scalar-reference \
  target/debug/delta-experiment epilogue-reference configs/lm512-projections.json
bun scripts/benchmark-epilogues.ts target/debug/delta-experiment 1 128 2 5
bun scripts/benchmark-epilogues.ts target/debug/delta-experiment 1 512 2 5
bun scripts/benchmark-epilogues.ts target/debug/delta-experiment 8 128 2 5
bun scripts/benchmark-epilogues.ts target/debug/delta-experiment 8 512 2 5
```

Also run the existing config regression tests centrally (including
`legacy_selector_defaults_without_changing_serialized_identity` and
`pilot_identity_includes_cache_and_optimizer`). A test build does not necessarily
refresh `target/debug/delta-experiment`; the parent must build that binary once
with its common feature combination before invoking it.

The Bun launcher creates a timestamped artifact directory and an explicit config
from `lm512-projections.json`, changing only shape, deadlines/output paths and
explicit historical epilogue/fusion/autotune defaults. Each invocation uses the
existing GPU/build lock wrapper (80 s GPU queue, 15 s build queue, 180 s execution
cap); the benchmark's soft deadline is 150 s. A timeout/failure leaves captured
stdout/stderr and any incremental report, and exits nonzero. If five trials do
not fit, retain that result and rerun **all three designs** with three trials; do
not silently narrow only the slow arm.

Each shape's isolated report includes resident SiLU, native residual, and actual
up–SiLU–down–residual F and F/B samples with trainable input/up/down/residual tensors.
The F/B objective is square+sum; setup, random input construction and transfers
are outside the clocks. It reports the optimistic sum of native isolated SiLU
and residual F/B milliseconds and its ratio to native isolated MLP F/B. This is
an attribution estimate including extra objective/barrier costs, not a rigorous
integrated speedup bound. Fullmodel dispatch measurements and ordinary steps are
the deciding evidence. Report the unchanged GEMM contribution rather than
attributing a whole-MLP change to pointwise arithmetic.

## Decision rule and evidence history

Keep `native` by default. An opt-in custom path earns recommendation only after
all independent and fullmodel gates pass, ordinary-step paired savings are
repeatable at the requested shapes, and its memory/dispatch evidence justifies
the extra kernels. If gains are noise or regressions, recommend the library path
and publish the measured removable-cost estimate, not a hypothetical rejection.

Freeze the implementation in a described source change, create a child change,
and place subsequent measurement evidence in that child (or a separately named
artifact). Do not rewrite the described source to attach results. There are no
measurement results in this source document.
