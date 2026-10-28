# Prefix block retrieval: MoBA own-block coverage

The optional model selector is:

```json
{"prefix_block_retrieval":{"layers":[3],"block_size":64,"blocks_per_query":2}}
```

`blocks_per_query` counts selected **earlier** blocks; the query's own block
is always extra. Block size must be positive and divisible by computational
chunk size; listed layers must exist and have HOLA; the requested number of
earlier blocks must be positive. Omission leaves historical configuration
serialization, initialization order and checkpoint fingerprints unchanged.
Enabled fingerprints include semantic revision `prefix-block-moba-own-v1`,
which distinguishes these rules from the superseded gapful retrieval rules.
Existing configurations are not enabled. The explicit fixture is
`configs/lm512-step8-retrieval.json`.

The separate execution selector `prefix_block_retrieval_backend` accepts
`tensor` (default) or `fused_gpu`. The GPU implementation requires explicit
non-fusion Vulkan, f32, K32/V64, computational chunks through 16 and supported
HOLA cache/address/shared-memory shapes. Unsupported shapes are rejected;
there is no silent CPU substitution.

## Exact read

For a clean query at absolute position `t`, let `a=floor(t/B)*B`.

1. Own-block keys are valid same-document tokens `[a,t]`, always included.
2. Select up to `k` eligible completed earlier blocks `j<floor(t/B)`.
   A block's summary is the mean of its valid same-document HOLA-normalized
   keys. Empty same-document subsets are ineligible. Ranking is normalized
   query dot summary divided by `sqrt(key_dim)`, descending, with ties going
   to the earlier block. Ranking and summaries used for ranking are detached.
3. Append original normalized keys and original convolved HOLA values.
   Exclude every appended token already in the HOLA cache **or** causal
   current-chunk keys, so no token appears twice.
4. Cache, causal chunk, retrieved earlier blocks, remaining own-block keys,
   and learned zero-valued sink share **one softmax**. The existing sigmoid
   gate and addition to the recurrent output stay outside the read.

Gradients reach the query, original keys/values and sink, not selection or
selection summaries. Requesting more earlier blocks than are eligible includes
all eligible blocks, without repetition. For a first retrieval block, own-block
coverage includes all its causal keys even after HOLA has evicted some of them.
There is no gap between the own-block start and current chunk.

### Frozen SBD snapshot

Every noisy query in one active block uses the **same** clean snapshot and
`a=floor(start/B)*B`, including noisy positions that cross a retrieval-block
boundary. Own-block clean keys are `[a,start)`; earlier candidates are
`j<floor(start/B)`. No noisy K/V becomes a prefix candidate. The existing
clean unfinished chunk `[cut,start)` is already in the read, so appended own
keys stop before `cut`. Cached duplicates are excluded explicitly.

Cached training and prefix reconstruction call the same isolated
`prefix_retrieval::noisy_read` helper. With `sbd.in_block_reads: "hola_read"`,
both paths supply its active tuple: HOLA-normalized raw noisy K and raw
convolved noisy V, shaped `[batch,length,head,width]`. All valid active
queries see all valid active keys, both directions including self, inside
the same softmax as cache, clean chunk, retrieved/own-block keys and sink.
Active K/V have separate analytic derivatives. The combined read uses
`prefix_block_retrieval_backend`; non-retrieval layers retain
`sbd.noisy_read_backend`. The explicit retrieval config enables both
in-block reads and fused execution. Unsupported execution rejects rather
than falling back. The length-one SBD convention remains the clean
next-token computation, selected by the existing outer length-one path.

## Execution and costs

The CPU/GPU tensor reference computes exact document-aware summaries and
visibility with Burn tensors. It appends the original prefix before the
current computational chunk and masks it to selected earlier blocks plus
own-block history. This transparent reference materializes quadratic prefix
read logits/visibility; it is not a sparse-memory implementation.

The CubeCL layout builds HOLA's original-position schedule, compact block
summaries and query-specific original-position lists on the GPU, then performs
one fused stable softmax/read. Homogeneous-document blocks reuse a detached
mean; mixed-document blocks compute the query's exact same-document subset.
Selected cached SBD layers retain summaries once, rather than recomputing them
for every noisy block. There are no per-chunk host selection transfers.

For batch `N`, time `T`, heads `H`, widths `K`/`V`, cache size `C`, chunk `Q`,
retrieval block `B`, and earlier-block count `k`:

- Selected-layer full-prefix K/V value footprint is `4*N*T*H*(K+V)` bytes.
  Existing training projections/SBD snapshots can already retain these buffers;
  this is not necessarily an incremental allocation.
- Complete-block summaries use `4*N*floor(T/B)*H*(K+3)` bytes, including count
  and document metadata. Unselected layers acquire no retrieval state.
- A clean read allocates `C+Q+min(k,floor((T-1)/B))*B+O` position slots per
  query, where `O=min(B-Q,floor((T-1)/Q)*Q)` covers own-block history before
  the current chunk. The canonical fixture has 208 slots.
- A noisy read uses `O=cut-floor(start/B)*B`, a smaller earlier-block count
  when appropriate, and adds active length only when explicitly supplied.
  The canonical dimensions need at most 224 slots with 16 active tokens.
- The fused read saves two f32 softmax statistics per query; its VJP saves two
  f32 scalars per allocated edge. It accumulates original K/V derivatives
  without floating-point atomics. The current inverse VJP scans query
  membership, `O(N*H*T*T*(C+k))` checks, and block ranking performs
  `O(N*H*T*M*M)` comparisons for `M=floor(T/B)`. These costs are not a
  linear-time backward result.

Only the indexed fused layout is implemented. A separate partial-log-sum-exp
layout is not implemented: HOLA exposes no reusable partial max/sum/value
outputs or shared-normalizer VJP, so it would require another kernel family
rather than reuse the existing read. Adoption compares tensor and indexed
fused layouts with paired complete-step evidence.

## Gates and measurement protocol

`prefix-retrieval-reference` uses an independent scalar f64 oracle for cache
selection, block selection, the joint read and analytic query/key/value/sink
VJPs. Fixtures include document transitions, repeated document labels,
padding, ties, cached/current-chunk duplicate exclusion, more requested
blocks than eligible, own-block coverage after eviction, lists exceeding 128
keys, frozen reads crossing a retrieval-block boundary, and the optional
active segment with padded queries and keys. Every element keeps tolerance
`2e-5 + 3e-4*abs(reference)`. The model directional finite-difference gate
replays both cache membership and prefix visibility from an unperturbed read.

From the retrieval workspace:

```sh
timeout 3600s ../build.sh retrieval
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 240s /tmp/delta-bins/retrieval/delta-experiment prefix-retrieval-reference configs/lm512-step8-retrieval.json
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 1800s /tmp/delta-bins/retrieval/delta-experiment kron-model-check configs/lm512-step8-retrieval.json --model-scope configured
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 1800s /tmp/delta-bins/retrieval/delta-experiment sbd-reference configs/lm512-step8-retrieval.json
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 3600s /tmp/delta-bins/retrieval/delta-experiment prefix-retrieval-benchmark configs/lm512-step8-retrieval.json --warmups 2 --trials 11
flock /tmp/delta-block-lm-gpu.lock flock /tmp/delta-block-lm-build.lock timeout 3000s /tmp/delta-bins/retrieval/delta-experiment prefix-retrieval-sbd-diagnostic configs/lm512-step8-retrieval.json
```

Primary cost/layout measurements use B8/T512 complete **clean causal** AdamW
steps: the command explicitly removes the SBD objective and records source
and effective configurations. It also measures a resident T2048 read and VJP.
Two warmups and at least eleven interleaved forward/reverse samples are
required; paired median differences use a deterministic 10,000-resample 95%
bootstrap interval. Launch observations and allocator-pool samples are
reported separately; pool reservations are not physical live-memory peaks.

The separate full-SBD diagnostic is one B1/T512 clean-plus-masked-block
optimizer step, including gradients, clipping and AdamW. It records launches,
noisy block model evaluations and selected-layer read launch share, not an
inferred runtime share. The full-SBD paired command uses B1/T512, or B1/T256
if the diagnostic exceeds 3000 seconds, with the same two-warmup/eleven-pair
minimum. No B8 SBD session or per-block loop optimization is included.

MoBA evidence belongs under `artifacts/retrieval/moba/`. Earlier gapful-rule
reports directly under `artifacts/retrieval/` are superseded and do not verify
this architecture. No training-quality study, held-out evaluation, alternative
mechanism or parameter search is included.

## MoBA numerical results

The independent oracle passes ten CPU tensor cases and twenty Vulkan
tensor/fused cases under the own-block rules, including active reads with
valid queries beyond a retrieval-block boundary. Maximum absolute errors:

| Execution | Output | Query VJP | Key VJP | Value VJP | Sink VJP |
|---|---:|---:|---:|---:|---:|
| CPU tensor | 2.416e-8 | 1.508e-8 | 2.858e-8 | 1.500e-7 | 7.047e-7 |
| Vulkan tensor/fused | 2.744e-8 | 2.893e-8 | 4.352e-8 | 2.789e-7 | 2.097e-6 |

All maxima are below the absolute tolerance alone; every element also passes
the unchanged absolute-plus-relative test. Evidence:
`artifacts/retrieval/moba/{cpu/,}prefix-retrieval-reference.json`.

The Vulkan-feature workspace suite passes 45 experiment, seven kernel and
sixteen mixer tests, with one GPU-sharing test ignored.

## Measured execution cost

Intel Arc Graphics (MTL), Mesa 26.2.4, Vulkan, f32, fusion/autotune off.
Each comparison uses two warmups, eleven interleaved pairs in one process and
the complete configurations recorded in the report.

### B8/T512 clean causal AdamW step

| Retrieval layout | Median ms | Observed launches |
|---|---:|---:|
| Off | 1863.765 | 5550 |
| Tensor | 2234.488 | 12473 |
| Indexed fused | 2015.634 | 5552 |

Tensor minus fused: paired median **211.244 ms saved**, 95% bootstrap interval
**[197.175, 218.854] ms**, fused faster in **11/11** pairs. The fused layout is
adopted for the explicit retrieval fixture.

Off minus fused: paired median **-139.102 ms**, interval
**[-185.987, -119.737] ms**, retrieval faster in **0/11** pairs. Retrieval adds
cost at T512; no saving is asserted.

Sampled allocator-pool reservation across this session peaked at
1,769,097,856 bytes. Selected-layer prefix K/V have a 6,291,456-byte value
footprint; summaries use 35,840 bytes. The reservation sample includes all
runners and is not a per-layout live-memory peak.

### Isolated B1/T2048/H4 read, K32/V64

| Layout | Forward median ms | Forward+VJP median ms | Forward launches | Forward+VJP launches |
|---|---:|---:|---:|---:|
| Tensor | 1713.428 | 2901.180 | 48723 | 56644 |
| Indexed fused | 18.417 | 271.405 | 4 | 19 |

Paired forward saving: 1693.043 ms, interval [1649.287, 1739.112], 11/11
faster. Paired forward+VJP saving: 2622.184 ms, interval
[2620.499, 2651.338], 11/11 faster. These are read measurements, not model
speedups. The fused inverse VJP's quadratic membership scan remains visible
in its larger backward cost.

Sampled pool reservation peaked at 649,704,192 bytes. Fused positions occupy
6,815,744 bytes, VJP edge scalars 13,631,488 bytes, statistics 65,536 bytes,
and summaries 17,920 bytes. The tensor reference allocates 8,323,072 appended
prefix logits per forward; logits plus probabilities account for 66,584,576
f32 value bytes, not a live-allocation sum. Evidence:
`artifacts/retrieval/moba/prefix-retrieval-benchmark.json`.

### B1/T512 full-SBD diagnostic

One complete diagnostic step took 63,920.520 ms and issued 372,835 kernels.
It evaluated 128 noisy model passes (teacher plus differentiable passes).
Selected-layer combined HOLA/retrieval kernels accounted for 455 launches,
or 0.1220% of issued kernels; this is **not** a measured time share.
The sampled pool reservation was 1,300,858,368 bytes. The diagnostic includes
possible first-use compilation and fits the 3000-second execution criterion.
Evidence: `artifacts/retrieval/moba/prefix-retrieval-sbd-diagnostic.json`.

## Full-model and compatibility gates

Configured-scope `kron-model-check` passes B8/T128 and B8/T512, both without
and with HOLA. Maximum CPU/GPU errors across all four cases are 6.915e-6
for logits, 1.304e-6 for every-parameter gradients, and 1.644e-5 for AdamW
parameters, all below the unchanged absolute tolerance alone. Every model
parameter has finite gradients. The CPU full-parameter directional check
passes with the existing finite-difference tolerances (0.003 absolute and
0.03 relative); maximum absolute discrepancy is 3.271e-4. Both cache slots
and retrieved-prefix visibility are replayed for perturbed passes.
Evidence: `artifacts/retrieval/moba/kron-model-check-711649-1792922745744110292/report.json`.

SBD reference passes eight CPU and eight Vulkan cases. Retrieval-enabled
fixtures use clean prefix length 83: one completed 64-token block, one
completed computational chunk inside the own block, and an unfinished clean
chunk. Vulkan-versus-CPU maxima are 3.502e-7 for logits and 2.282e-8 for all
parameter gradients. Cached/reconstructed Vulkan logits are identical and
parameter gradients differ by at most 1.397e-9. CPU cached/reconstructed
maxima are 1.491e-7 for logits and 8.615e-9 for gradients. Length-one,
future-locality and clean-replay checks pass. Evidence:
`artifacts/retrieval/moba/{cpu/,}sbd-reference.json`.

The omitted-selector historical Vulkan checkpoint check passes all six arms
at its original 1e-5 absolute plus 1e-4 relative tolerances. Maximum logit
error is zero, with bit-identical save/load/repeated-evaluation checks.
Generated diagnostic checkpoint payloads were removed; final reports and
hashes remain in `artifacts/retrieval/moba/checkpoint/`.

### B1/T512 full-SBD paired cost

Two warmups and eleven interleaved pairs completed on the full SBD objective:

| Retrieval | Median step ms | Observed launches |
|---|---:|---:|
| Off | 44258.335 | 216368 |
| Indexed fused on | 43128.916 | 213423 |

Paired off-minus-on median is **1869.235 ms**, 95% bootstrap interval
**[1129.419, 2438.515] ms**, on faster in **10/11** pairs. The sampled pool
reservation peaked at 1,637,456,064 bytes. Launch counts are a separate
matched-row observation after timing, not the diagnostic's first row.

This comparison also replaces the historical cached SBD frozen-read tensor
assembly with indexed fused execution in selected layers; the saving is not
an isolated arithmetic cost of adding retrieved keys, and is not a quality
result. Cross-session absolute medians are not compared. Evidence:
`artifacts/retrieval/moba/sbd512/prefix-retrieval-sbd-benchmark.json`.

The unchanged HOLA reference also passes 47 cases with zero membership
error. Its large-value stress has output absolute error 3.380e30
(`finite_large_original_values_normalized_before_matmul`); the cancellation
VJP stress has query/key absolute error 4.747e-5. These pass their existing
per-element `2e-5 + 3e-4*abs(reference)` tests; neither stress result is
compared to the absolute term alone. Maximum value/sink VJP errors are
1.127e-6 and 6.457e-6. Evidence:
`artifacts/retrieval/moba/hola-reference.json`.
