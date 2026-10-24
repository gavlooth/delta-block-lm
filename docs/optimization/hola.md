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

## Provisional confirmation and decision

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

The provisional decision accepted `fused_gpu_pairs` (1.203x at B8/T512/cap16)
but held off recommending the cache-only RMS override independently. Its
incremental paired median saving over pairs-only shrank from180.575ms in the
first B8/T512 session to13.124ms in this confirmation, with a marginal
[0.085,42.039]ms bootstrap interval. B1 and B8/T128 incremental intervals
included zero. The final private-cache session below supersedes that
conservative intended-shape decision; this earlier evidence remains visible.

The measured HOLA component budget is now approximately12.4ms read FB plus
10.0ms tensor Q/K RMS FB per layer at B8/T512/cap16 (independent resident
synthetic-adjoint fixtures). Re-profile the actual whole model after combined
optimizations; do not sum isolated medians into a predicted step or extrapolate
this fixed-capacity result to larger contexts/precision/devices.

## Final validation and provenance

Production source is frozen in commit
`c4219152795f83168c277b8bbce91e9db5fb7aad`
(change `tznpnntyvswlwyqxvxokypyorxvxyvxu`); subsequent evidence-only updates
are in child change `nuptvnzqpptvylzoxryoowmonssporkw`.
The corrected `../build.sh hola` acquired GPU then build lock, cleaned only
the three local packages, rebuilt all three from this workspace and copied
the private executable. Cargo build took3.72s; queue wait358s and361.58s
process wall are reported separately. The existing ndarray deprecation
warnings remain visible. Build/profile/features/dependencies are unchanged.

`artifacts/hola/final-provenance.json` records source-file SHA256 values,
immutable commit, build protocol and binary SHA256
`c1931fa864f8c140a55c07801a318c614dbef5cbdb4a615333c1ae00b8fcc14e`.
`stat` confirms both relative `.cache` and `.cache/cubecl` are private
directories, not the former shared symlink. No source change follows this
build.

The final private-cache `hola-reference` passes **all47 checks**:
11 tensor and12 each historical fused / sparse pairs / pairs+norms read
selectors. It checks the expanded strided/zero and existing padding/reset/tie/
tail/large-value/cancellation fixtures with unchanged tolerances, including
all Q/K/V/sink adjoints and exact membership. Evidence:
`artifacts/hola/final-pairs/hola-reference.json`. Non-overflow maximum output
error is3.480e-7; maximum Q/K error4.747e-5 is in the analytic cancellation
probe and passes the original relative criterion. The large-finite-value
fixture is checked under the same original scale-aware tolerance, not
excluded from pass/fail. Selection-score gradients remain absent.

Recorded pre-isolation supplementary gates also pass:

- Historical `kron-model-check`:12 combinations; max CPU logits/parameter
  gradient/AdamW errors5.126e-6 /2.652e-6 /8.829e-6.
- Sparse-pairs configured `kron-model-check`:4 cases (contexts128/512, with/
  without HOLA); maxima5.126e-6 /2.652e-6 /8.985e-6.
- Pairs+RMS configured check:4 cases; maxima4.709e-6 /2.652e-6 /7.797e-6.
  All include every model parameter, finite checks, one AdamW update and
  the existing all-parameter CPU directional finite difference.
- Independent normalization116 cases; selected fused Kron12 cases; CPU
  reference6 recurrence/4 cache/6 full-layer cases.

Those full-model reports are respectively under
`artifacts/hola/b1/kron-model-check-2687569-1792866761049914904/`,
`artifacts/hola/pairs/kron-model-check-2757554-1792867368642807630/`,
and `artifacts/hola/pairs-norms/kron-model-check-2760741-1792867398027891828/`.
No tolerance, reset, padding, cache policy, optimizer or historical fingerprint
was relaxed. The known arbitrary-input legacy tensor-solver failure is not
rerun, suppressed or accepted.

Per parent coordination, the combined integration workspace owns the **final
shared** full-model, workspace-test and historical
`artifacts/lm512-vulkan/kronlstm-hola-seed1/checkpoint-13` gates; they are not
duplicated or falsely claimed exercised by this isolated item. The new
checkpoint-only `--diagnostic-output` option redirects report/scratch files
without modifying the original config/hash, allowing that shared gate to
write inside its assigned evidence namespace.
That integration also owns widening `backward-profile`'s historically strict
scan/read guards to `is_fused()` and propagating the combined convolution/
normalization selectors; this isolated source deliberately leaves that
shared profiling guard unchanged to avoid conflicting follow-on commits.

## Final paired result and adoption

The corrected-helper/private-cache B8 session completes both contexts with
all preflight output/every-parameter-gradient/AdamW comparisons passing the
original2e-5/3e-4 tolerances. Eleven paired trials follow two warmups:

| Context | Historical step, ms | Pairs step, ms | Pairs + cache RMS step, ms | Pairs paired saving95% interval, ms | Combined paired saving95% interval, ms |
|---:|---:|---:|---:|---:|---:|
| 512 | 1502.349 | 1274.329 | 1236.615 | [192.491,262.424] | [239.213,299.841] |
| 128 | 376.110 | 317.481 | 312.308 | [54.738,65.152] | [50.205,74.248] |

At the adoption workload B8/T512/cap16, pairs-only gives1.179x complete-step
throughput and the combined selector gives **1.215x** (2726.4→3312.3tokens/s,
17.69% lower step median). Median within-trial savings are232.262ms and
267.540ms respectively. The combined selector's incremental saving over
pairs-only is32.743ms with paired-median95% interval **[27.879,59.411]ms**.
That positive intended-batch result, together with both earlier positive
B8/T512 incremental intervals, supports the cache-only RMS addition at this
specific workload. This is not a claim that every norm/gate merits fusion or
that the prior rejected all-model/B1 normalization scope is now accepted.

**Adopt `hola_backend: \"fused_gpu_pairs_norms\"` for the measured B8/T512 path.**
The sparse-pairs-only selector is also individually accepted, isolating the
robust backward redesign. Neither historical default nor existing `fused_gpu`
is silently redirected. At T128 the combined variant is1.204x historical,
but the incremental RMS interval[-14.737,7.778]ms includes zero, so no separate
RMS-only improvement is claimed there or at B1. The provisional B1 paired
table remains reported rather than replaced with an extrapolation from B8.
No precision, membership, parameter identity, cache capacity or optimizer
change is folded into either comparison.

Final report:
`artifacts/hola/final-b8/hola-tune-2882850-1792868412207054173/report.json`;
derived paired intervals/protocol:
`artifacts/hola/final-summary.json` (20,000resamples, median, LCG seed7713).
Internal elapsed157.37s;165.00s process wall includes the separate queued
numerical gate. Kernels/step remain5833/5833/5641 and sampled pool maxima
861,580,288bytes at512 and391,799,680bytes at128, shared across scratch
trajectories with the limitations above. The numerical gate took7.43s
process wall. The fixed source/binary hashes above bind both final runs.

## Reproduction

Build from this workspace only:

```sh
../build.sh hola
flock /tmp/delta-block-lm-gpu.lock timeout 300s /tmp/delta-bins/hola/delta-experiment hola-cost artifacts/hola/baseline.json
flock /tmp/delta-block-lm-gpu.lock timeout 300s /tmp/delta-bins/hola/delta-experiment hola-reference artifacts/hola/pairs.json --hola-backend fused_gpu_pairs
flock /tmp/delta-block-lm-gpu.lock timeout 300s /tmp/delta-bins/hola/delta-experiment hola-cost artifacts/hola/pairs.json
flock /tmp/delta-block-lm-gpu.lock timeout 300s /tmp/delta-bins/hola/delta-experiment hola-cost artifacts/hola/pairs-norms.json
flock /tmp/delta-block-lm-gpu.lock timeout 300s /tmp/delta-bins/hola/delta-experiment hola-tune artifacts/hola/b8.json --contexts 512,128 --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock timeout 300s /tmp/delta-bins/hola/delta-experiment hola-tune artifacts/hola/b1.json --contexts 128,512 --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock timeout 300s /tmp/delta-bins/hola/delta-experiment hola-tune artifacts/hola/b8-cap64.json --contexts 512,128 --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock timeout 300s /tmp/delta-bins/hola/delta-experiment hola-tune artifacts/hola/b1-cap64.json --contexts 128,512 --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock timeout 300s /tmp/delta-bins/hola/delta-experiment hola-tune artifacts/hola/b8.json --contexts 512 --warmups 2 --trials 21
flock /tmp/delta-block-lm-gpu.lock timeout 300s /tmp/delta-bins/hola/delta-experiment kron-model-check artifacts/hola/b1.json
flock /tmp/delta-block-lm-gpu.lock timeout 300s /tmp/delta-bins/hola/delta-experiment kron-model-check artifacts/hola/pairs.json --model-scope configured
flock /tmp/delta-block-lm-gpu.lock timeout 300s /tmp/delta-bins/hola/delta-experiment kron-model-check artifacts/hola/pairs-norms.json --model-scope configured
flock /tmp/delta-block-lm-gpu.lock timeout 300s /tmp/delta-bins/hola/delta-experiment normalization-reference artifacts/hola/pairs-norms.json
flock /tmp/delta-block-lm-gpu.lock timeout 300s /tmp/delta-bins/hola/delta-experiment kron-reference artifacts/hola/b1.json --scan-backend fused_gpu
/tmp/delta-bins/hola/delta-experiment reference artifacts/hola/cpu.json
# Final corrected-helper/private-cache commands:
../build.sh hola
flock /tmp/delta-block-lm-gpu.lock timeout 300s /tmp/delta-bins/hola/delta-experiment hola-reference artifacts/hola/final-pairs.json
flock /tmp/delta-block-lm-gpu.lock timeout 300s /tmp/delta-bins/hola/delta-experiment hola-tune artifacts/hola/final-b8.json --contexts 512,128 --warmups 2 --trials 11
```

Artifact configs under `artifacts/hola/` copy the recommended config, isolate
report directories, disable budget-file output and raise only the cooperative
report deadline to300s. B8/cap64/candidate configs explicitly change the named
workload or execution selector. No matmul autotune, backend feature, dtype or
optimizer change is included.
