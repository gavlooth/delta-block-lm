# HOLA norms and fused-read optimization

## Workload and acceptance

Local Intel Arc MTL integrated GPU, Vulkan, Burn 0.22/CubeCL 0.11, f32,
non-fusion backend, matmul autotune disabled. The baseline is
`configs/lm512-projections.json`: four D256 layers, H4/K32/V64, chunk16,
merged input projections, fused Kron scan and historical fused HOLA read,
tensor normalization. Realistic training batch8 is the adoption workload;
batch1 and contexts128/512 are also measured. Cache capacities16/64 are
attributed separately. All GPU invocations acquire the shared GPU lock.

The explicit `hola_backend` selector preserves omitted historical tensor
execution and the existing `fused_gpu` path. Historical configs/checkpoints
are not migrated. Parameters, cache membership, exact attention, numerical
tolerances, original-KV storage, learned Q/K RMS norms, sink and gate semantics
are unchanged.

The attribution and first paired/confirmation tables below were collected before
the parent corrected shared local-crate build isolation and made each workspace's
relative `.cache/cubecl` private. They are retained as **provisional** historical
measurements, not called false or silently replaced. Adoption and final numerical
acceptance are conditioned on a rebuilt private-cache validation/paired session.
Queue wait is separate from each command's at-most300s GPU execution bound.

## Baseline attribution and design

The bounded `hola-cost` command uses resident independent seeded
uniform[-.5,.5] leaves, valid=1/document=0, H4/K32/V64/chunk16/T512. Warm
ordinary forward+backward wall samples use outer synchronization; separate
serialized per-launch samples identify selection, forward, query VJP, key/value
VJP and sink VJP. Their timing method is recorded rather than mixing device
timestamps and host/system timing. These are per-layer measurements, not an
extrapolated complete-model step or hardware-utilization estimate.

The initial attribution (`artifacts/hola/initial-cost-shared-gamma.json`)
measured key/value VJP medians of8.851/19.005ms at B1/cap16/64 and
55.811/184.246ms at B8. Forward/query VJP at B8/cap16 were2.789/4.188ms;
selection/sink were0.210/0.040ms. The serial key-lifetime walk dominates.
This initial norm fixture shared one gamma between Q/K; the final attribution
uses independent Q/K gamma tensors, matching production. Both reports remain
available; the initial norm number is not used as the final norm measurement.

Final baseline breakdown (`artifacts/hola/hola-cost.json`), median ms per layer:

| Batch | Cap | Selection device | Forward attention device | Query VJP device | Key/value VJP device | Sink VJP device | Two RMS F wall | Two RMS FB wall | Read F wall | Read FB wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 0.224 | 0.417 | 0.609 | 9.058 | 0.014 | 0.493 | 1.776 | 1.083 | 22.724 |
| 1 | 64 | 0.606 | 0.785 | 1.072 | 21.472 | 0.014 | 0.506 | 1.740 | 1.961 | 32.572 |
| 8 | 16 | 0.214 | 2.923 | 4.489 | 60.145 | 0.041 | 1.061 | 9.887 | 3.754 | 67.594 |
| 8 | 64 | 0.631 | 5.943 | 8.488 | 175.378 | 0.042 | 0.982 | 9.076 | 7.864 | 228.494 |

All individual read timings above used **device timestamps**. Device medians
from serialized launches do not sum to the separately sampled ordinary wall
median: host scheduling/submission/barriers and device-clock variation remain,
particularly in B1. RMS ordinary timings also contain the synthetic combined
output root. Norm forward/backward launches are56 including that root; read
forward/backward launches are6 including one all-ones adjoint fill. Do not
interpret these differences as missing model components or attainable savings.

Compared designs:

1. **Historical recomputation** (`fused_gpu`): a workgroup per key walks its
   uninterrupted causal/cache lifetime. Lane0 recomputes the32-channel logit
   and64-channel upstream/value dot for every query; two full workgroup
   barriers communicate its scalars per visited query.
2. **Sparse edge reuse** (`fused_gpu_pairs`): the existing query VJP additionally
   writes its already-computed probability and scaled logit adjoint for each
   query/cache-or-local slot. The key/value VJP inverts the existing position
   schedule once per visited chunk, reads those scalars and accumulates the
   original Q/upstream in the original chronological order. It does not
   recompute attention, use atomics, copy KV or allocate a dense T-by-T matrix.
   The same sequential logit arithmetic is retained, including the existing
   cancellation regression, rather than changing it to a tree dot product.
3. **Sparse edge reuse plus cache-only custom RMS** (`fused_gpu_pairs_norms`):
   reuse the already-existing analytic RMS operator for only HOLA's two learned
   Q/K RMS norms. Other model normalizations/gates remain historical tensor
   operations. This compares reuse of an existing primitive with the sparse
   backward redesign without confounding all-model normalization changes.

`hola-tune` uses the existing paired complete-step comparison machinery:
three independent identical-start ordinary trajectories, AB/BA-style rotating
forward/reverse mode orders, the same real batches/step indices, two warmups,
all-parameter finite checks/clipping and AdamW. Separate section-barrier
trajectories and launch observations are outside ordinary timing samples.
Raw paired samples and equivalence preflights are saved in each report.

## Numerical coverage

Independent scalar-f64 output, exact final membership and all Q/K/V/sink VJPs
cover surprise/recency, ties, document changes/repeated labels, reset-on-padding,
internal/full-chunk/tail padding, chunks1/16, tail lengths7/35/83/145,
capacities1/16/64, finite large values, zero inputs/adjoints and genuinely
strided Q/K/V/scores/upstream views. Selection scores remain nondifferentiable.
The historical cancellation probe checks the analytic VJP of the actual f32
forward probability, not a falsely claimed exact f64 cancellation result.
All original absolute/relative tolerances are retained.

## Isolated candidate costs

Later synchronized ordinary per-layer F/FB medians (ms), with identical fixed
shapes and five samples. These sessions are **not interleaved with the earlier
baseline**, so use them to attribute the redesigned component, not to quote a
precise isolated speedup independent of clocks/host scheduling.

| Batch | Cap | Pairs read F | Pairs read FB | Pair key/value VJP device | Cache-only custom RMS F | Cache-only custom RMS FB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 1.375 | 3.206 | 0.893 | 0.771 | 1.125 |
| 1 | 64 | 2.149 | 6.004 | 1.828 | 0.371 | 0.769 |
| 8 | 16 | 3.766 | 12.438 | 3.984 | 1.061 | 2.770 |
| 8 | 64 | 7.269 | 27.355 | 11.562 | 1.130 | 2.839 |

`artifacts/hola/pairs/hola-cost.json` and
`artifacts/hola/pairs-norms/hola-cost.json` retain every raw sample, launch
name and timing method. Read F and the selection/forward/query/sink code are
unchanged by either backward selector; differences in their measured durations
between sessions are not algorithmic forward improvements. The candidate
query VJP additionally writes edge scalars, so its work increases slightly.
The learned-RMS fixture falls56→8 launches, but complete-step evidence below,
not that launch reduction or isolated gain, determines adoption.

## Memory and work

For N=B*T*H, S=capacity+chunk, the candidate adds `8*N*S` logical bytes during
backward only: at T512/H4, cap16,0.5MiB at B1 or4MiB at B8; at cap64,
1.25MiB/10MiB. No new retained forward activations or dispatches are required.
The original schedule is4*B*H*(ceil(T/chunk)+1)*capacity bytes, expiry4*N,
softmax stats8*N, auxiliary8*N, retained output256*N. Q/K/V/sink gradient
payload is unchanged. Pair writes are bounded sparse O(T*(capacity+chunk)),
not quadratic in context. Schedule inversion scans up to capacity positions
once per resident key/chunk; this is extra bounded lookup work, not a claim of
unchanged arithmetic work. The candidate removes each repeated32+64 scalar
reduction and replaces two barriers per query by two per chunk.

Sampled backend pool reservation includes all scratch trajectories and allocator
reuse on a shared-memory integrated GPU; it is not an exact physical peak.

## Paired complete guarded steps

Eleven interleaved paired samples per mode after two warmups. Times are
ordinary complete-step medians in ms, including forward/backward, data upload,
loss readback, all-parameter gradient checks/clipping and AdamW. No section
barriers or launch observer is present in those samples.

| Batch | Context | Cap | Historical | Sparse pairs | Pairs + cache RMS |
|---:|---:|---:|---:|---:|---:|
| 8 | 512 | 16 | 1535.782 | 1421.865 | 1246.994 |
| 8 | 128 | 16 | 359.188 | 292.575 | 297.615 |
| 1 | 512 | 16 | 550.149 | 508.455 | 504.695 |
| 1 | 128 | 16 | 191.771 | 182.726 | 181.281 |
| 8 | 512 | 64 | 1947.580 | 1313.044 | 1307.354 |
| 8 | 128 | 64 | 443.185 | 306.988 | 299.968 |
| 1 | 512 | 64 | 586.749 | 512.098 | 505.151 |
| 1 | 128 | 64 | 195.326 | 177.833 | 177.981 |

For cap16/context512, the median **within-trial** historical-minus-candidate
savings are194.930/305.360ms at B8 and29.063/41.961ms at B1 for pairs /
pairs+RMS. These are not differences of independently selected medians.
Seeded20,000-resample paired-median bootstrap95% intervals are[41.322,264.887] /
[227.488,394.604]ms at B8 and[20.661,53.323] /[35.216,47.006]ms at B1.
At B1/context128 the pairs-only interval[-0.726,15.537]ms includes zero:
no beyond-noise improvement is claimed there. The B8/context128 pairs-only
interval[57.399,72.394]ms excludes zero.

Warm kernels/complete-step are5833 at B8 and5780 at B1 for historical and
pairs; pairs+cache-RMS reduces each by192 to5641/5588. Counts are separate
observations of actual complete steps, not timed launches. B8 has53 additional
shape-dependent library launches compared with B1; do not substitute a B1
histogram for B8. Reducing launches alone is not the adoption criterion.

Maximum sampled six-trajectory pool reservations at cap16 are861,580,800 /
391,799,168bytes at B8/context512/128 and285,884,608 /222,446,976bytes at B1.
The variants share/reuse the same process pool; these are not separate
per-variant allocation peaks. Cap64 observations are similar, not evidence
that the logical sparse-edge buffer is free.

Complete raw reports:

- B8/cap16: `artifacts/hola/b8/hola-tune-2622738-1792866185260495688/report.json`
- B1/cap16: `artifacts/hola/b1/hola-tune-2641058-1792866355804767107/report.json`
- B8/cap64: `artifacts/hola/b8-cap64/hola-tune-2658208-1792866499871959746/report.json`
- B1/cap64: `artifacts/hola/b1-cap64/hola-tune-2678792-1792866681322867408/report.json`

## Confirmation and decision

A separate21-pair B8/context512/cap16 session, again with two warmups,
confirms the sparse-backward gain:

| HOLA selector | Complete step median, ms | Tokens/s | Median paired saving vs historical, ms | Paired-median bootstrap95% interval, ms |
|---|---:|---:|---:|---:|
| `fused_gpu` | 1498.015 | 2734.3 | 0 | — |
| `fused_gpu_pairs` | 1245.016 | 3289.9 | 259.483 | [212.109,296.436] |
| `fused_gpu_pairs_norms` | 1217.081 | 3365.4 | 272.503 | [242.828,304.227] |

Report:
`artifacts/hola/b8/hola-tune-2691796-1792866796647880881/report.json`.
The confirmation completed in209.33s internally, below its300s GPU bound.
Initial B8/B1/cap64-B8/cap64-B1 sessions completed in164.59/79.08/181.26/79.54s
internally; longer process wall times include GPU-lock queue wait, not compute.

**Adopt the explicit `hola_backend: \"fused_gpu_pairs\"` path.** On the realistic
B8/T512/cap16 workload it reduces complete-step median by16.89%, giving
**1.203x throughput**, with a clearly positive paired interval in both sessions.
At cap64 the11-pair B8 full-step median improves1947.580→1313.044ms (1.483x).
No B1/T128 beyond-noise pair-only gain is claimed. No isolated improvement
is substituted for these complete guarded steps.

**Do not recommend the cache-only RMS override as a separate adoption.**
It remains an explicitly named evaluated alternative for reproducibility,
reusing an existing kernel rather than introducing another norm implementation.
Its incremental paired median saving over pairs-only shrinks from180.575ms
in the first B8/T512 session to13.124ms in the confirmation, with a marginal
[0.085,42.039]ms bootstrap interval. At B1 and B8/T128 the incremental intervals
include zero; B8/T128's median even increases slightly. This is not robust
evidence for a general normalization recommendation, despite the combined
variant being much faster than historical HOLA. The robust result is sparse
read-backward reuse, not a192-launch reduction.

The measured HOLA component budget is now approximately12.4ms read FB plus
10.0ms tensor Q/K RMS FB per layer at B8/T512/cap16 (independent resident
synthetic-adjoint fixtures). Re-profile the actual whole model after combined
optimizations; do not sum isolated medians into a predicted step or extrapolate
this fixed-capacity result to larger contexts/precision/devices.

## Reproduction

Build from this workspace only:

```sh
../build.sh hola
flock /tmp/delta-block-lm-gpu.lock timeout 300s /tmp/delta-bins/hola/delta-experiment hola-cost artifacts/hola/baseline.json
flock /tmp/delta-block-lm-gpu.lock timeout 300s /tmp/delta-bins/hola/delta-experiment hola-reference artifacts/hola/pairs.json --hola-backend fused_gpu_pairs
```

Artifact configs under `artifacts/hola/` copy the recommended config, isolate
report directories, disable budget-file output and raise only the cooperative
report deadline to300s. B8/cap64/candidate configs explicitly change the named
workload or execution selector. No matmul autotune, backend feature, dtype or
optimizer change is included.
