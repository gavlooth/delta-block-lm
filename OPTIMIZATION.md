# Optimization checklist

Each item follows the "Optimize components before scaling" rules in `AGENTS.md`. An item is done only when it has: a saved baseline, an independent numerical check, isolated and full-model timings, memory/work numbers, and an entry in `RESULTS.md`.

Status: `[ ]` not started, `[~]` in progress, `[x]` done.

## Done

- [x] KronLSTM recurrence: fused GPU scan forward/backward (context 512 model step 3.05x without HOLA).
- [x] HOLA cache selection and exact attention: fused GPU forward/backward (combined with fused KronLSTM 3.86x).
- [x] HOLA and KronLSTM 32-bit address limits checked before launch.
- [x] Profile the whole active model step: reusable bounded `profile` command, contexts 128/512, per-block forward attribution, total backward, transfers/checks/AdamW and launch counts (see `RESULTS.md`, Whole-model profile).
- [x] Burn automatic kernel fusion: implemented fusion-aware custom-op registration and explicit execution selection, then evaluated and **rejected for this workload**. Launches fall 5,712→2,579, but ordinary steps are 13.3%/4.0% slower at contexts 128/512; strict Kron continuation and full-model CPU gates fail unchanged tolerances. See `RESULTS.md`, Burn automatic kernel fusion.
- [x] Q/K/V, decay, erase, write, output-gate and output projections: compared grouped and single-merged input contractions with unchanged parameter storage/IDs. Adopt explicit `projection_backend: "merged"`; omitted = historical separate. Paired full-step gains repeat at contexts128/512 despite more total launches; dependent output stays separate. See `RESULTS.md`, Mixer projection contraction merging. Activations remain unchanged and are covered by normalization/gating below.
- [x] Normalization and gating: compared custom L2/all-RMS reductions against an expanded sigmoid-output/RMS + softplus/exp-decay scope, with analytic VJPs and explicit selectors. Numerical/model/checkpoint gates pass, but context512 full-step paired intervals include zero in both 21-trial sessions despite 5,780→4,836 kernels; **not adopted**. Keep merged projections with tensor normalization/gating. See `RESULTS.md`, Normalization and gating fusion.
- [x] Attribute backward cost: bounded `backward-profile` command at the exact recommended model shapes and contexts128/512; isolated warm forward/forward+backward for every component, all four layers, real unmodified guarded steps, signed coverage/overlap gaps and scan batch/head launch/workgroup sweep. Context512 backward coverage98.98%; scan dominates at324.61ms forward+backward (62.68% of the step). See `RESULTS.md`, Backward attribution.
- [x] Batch scaling at context512: allow positive configured training batches in `backward-profile` without relaxing other recommended shapes; attribute B1/4/8/16/32 complete guarded steps and report tokens/s/memory. B8 reaches2,538tokens/s vs916 at B1; scan share falls60.85%→21.06%, behind HOLA and MLP. B32 fits (2.76GB sampled pool) but reaches only2,433tokens/s. See `RESULTS.md`, Batch scaling.
- [x] MLP matrix-multiply efficiency, broadened to merged-input, mixer-output and vocabulary-head projections: bounded actual F/dX/dW GEMMs at B1/8/16/context512, packing/layout comparison and Vulkan/autotune attribution. **Adopt an explicit guarded autotune option**, not bare autotune: keep MLP/merged inputs tuned, pin dependent output/head to the correct Cube heuristic, and refuse training on failed per-geometry f64 probes. Paired B8 step1619.76→1188.30ms (26.64% less time), B1 560.21→510.47ms (8.88%); both95% intervals exclude zero. All47 active-shape probes and configured T128/T512 full-model gates pass; legacy solver GEMMs remain unsafe/forbidden for autotuned training. Historical defaults unchanged. See `docs/optimization/gemm-efficiency.md` and `RESULTS.md`, Matrix-multiply efficiency; SiLU/residual epilogues remain separate below.

## Implemented, not yet tuned

Ordered by measured **B8/context512 forward+backward component cost** on the
current recommended merged-projection/tensor-normalization path. B8 is a
realistic training batch, unlike the original B1 occupancy experiment.
Resident independent-leaf measurements sum all four layers within trials;
AdamW/checks include their complete non-autodiff work. Percentages use the
ordinary **1,613.667ms** complete guarded step. Costs are prioritization
evidence, not additive savings or hardware utilization: synthetic VJP roots,
barriers and overlap cause a signed gap. Completed/evaluated projection and
normalization scopes remain in Done; all remaining costs are in `RESULTS.md`.

**Decision: lower the more-parallel scan below HOLA and MLP, rather than
implementing a chunk-parallel recurrence next on B1 evidence alone.** At B8,
HOLA costs395.18ms (24.49%), MLP358.64ms (22.23%) and scan339.84ms (21.06%),
versus scan340.25ms (60.85%) at B1. Eight times the real batch work has almost
unchanged scan cost: batch parallelism already captures much of the unused
capacity. A hypothetical2x scan-only gain would save169.92ms, yielding about
**1.12x** complete-step speedup at B8, not the1.44x B1 estimate. These are
Amdahl-style estimates assuming unchanged other costs, not measured gains or
proof that a parallel scan would attain them.

At B16 the same ordering holds: HOLA825.27ms (25.89%), MLP706.74ms (22.17%),
scan600.68ms (18.84%). Doubling B8 to B16 improves full-step tokens/s only
1.25% while approximately doubling pool reservation. B32 also ranks
HOLA/MLP/scan first/second/third (22.44%/20.39%/16.74%), but its backward
coverage is only79.96%; do not assign its1,214.44ms signed gap to a component.
B32 fits with2,756,328,960bytes sampled step pool, yet is4.15% lower tokens/s
than B8. Prefer B8 as the cost/throughput baseline for the next comparison;
batch convergence and quality are not evaluated here.

- [ ] HOLA fused read backward: tune the existing read/normalization path without changing cache membership or exact attention.395.18ms forward+backward (24.49%) across four layers at B8/context512.
- [ ] MLP SiLU and residual-add epilogues: matrix-multiply efficiency is evaluated above; no activation/residual fusion is claimed. The accepted heuristic path still costs358.64ms forward+backward (22.23%; residual adds remain in their own measured group).
- [ ] KronLSTM: more-parallel scan/backward remains pending, now third rather than next. Preserve resets, exact state/output/score VJPs and tolerances. B8/context512 costs339.84ms forward+backward (21.06%); each scan kernel already launches32 workgroups of64 lanes, compared with4 at B1. Any chunk-summary/prefix or independent-state partition must earn its cost at the intended training batch, not only B1.
- [ ] Short causal convolutions: replace the per-tap slice/mask/assign loop with one fused depthwise kernel that respects document boundaries (80.53ms forward+backward,4.99%).
- [ ] AdamW and gradient checks: multi-tensor update and one combined finite/norm reduction, keeping the safety checks (37.97ms complete non-autodiff work,2.35%, including clipping and gradient-norm readback).
- [ ] Embedding, vocabulary head and loss: fused cross-entropy without materializing the full log-softmax (final norm/head/loss19.49ms forward+backward plus embedding3.11ms;1.40% combined).

Automatic fusion reduced dispatches in the rejected backend, but did not
win or pass its unchanged numerical gates. It does not complete the pending
multi-tensor optimizer/checks, more parallel scan, convolution or fused
cross-entropy work. These ranks use the accepted **non-fusion backend**.
Re-profile if any execution path changes; dispatch reduction alone is not
an integrated speedup.

Not ranked by this resident forward+backward comparison (inactive, outside the step, or not isolated here):

- [ ] Data pipeline: measure batch preparation and host-to-GPU transfer; add prefetch only if justified. Outside the new resident isolated component timings; the historical1.49ms forward/upload group belongs to a different session and is not mixed into these ranks.
- [ ] Soft top-k selection: move selection off the CPU (currently downloads probabilities, sorts every row, uploads IDs); keep candidates and feedback on the GPU. Needed before block diffusion training; inactive in the causal LM profile.
- [ ] Checkpoint save/load: reduce cost (~250 ms save, ~330 ms load at context 512) without weakening integrity checks; outside the training step.
- [ ] GDN-2 training: use the existing verified CubeCL operator in training if GDN-2 stays a comparison arm; inactive in the KronLSTM profile.

## Correctness issue (not an optimization)

- [ ] Legacy GPU tensor solver: key-gradient mismatch on the short nonzero-initial-state stress case. CPU and fused paths pass. Evidence: `artifacts/kron-validation/solver-output_only/kron-reference.json`.

## Not implemented yet (optimize when each is built)

- [ ] oHC stream mixing (Step 7)
- [ ] Wavelet mixer (Step 7)
- [ ] Three Lngram components (Step 7)
- [ ] SBD training and causal block replay (Step 8)
- [ ] Learned or adaptive block boundaries (Step 12)
- [ ] Codpula circuit inference, training and adaptive commitment (Step 9)
- [ ] CoCommit coordination passes (Step 10)
- [ ] R²D remasking and TCT training (Step 11)
- [ ] Energy-guided stochastic remasking, unary then pairwise (Step 11)

## Cross-cutting

- [ ] BF16 / mixed-precision kernels with numerical checks.
- [ ] CUDA benchmarks on the DGX Spark (needs an agreed GPU window).
- [x] Larger training batch sizes at current K32/V64/chunk16/context512: B1/4/8/16/32 measured, largest tested B32 fits; full steps, component attribution and memory in Batch scaling.
- [ ] Larger key/value widths and contexts beyond the current K32/V64/chunk16 pilot limits.
- [ ] Re-profile after each optimization, since removing one bottleneck exposes the next.
