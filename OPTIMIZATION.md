# Optimization checklist

Each item follows the "Optimize components before scaling" rules in `AGENTS.md`. An item is done only when it has: a saved baseline, an independent numerical check, isolated and full-model timings, memory/work numbers, and an entry in `RESULTS.md`.

Status: `[ ]` not started, `[~]` in progress, `[x]` done.

## Done

- [x] KronLSTM recurrence: fused GPU scan forward/backward (context 512 model step 3.05x without HOLA).
- [x] HOLA cache selection and exact attention: fused GPU forward/backward (combined with fused KronLSTM 3.86x).
- [x] HOLA and KronLSTM 32-bit address limits checked before launch.
- [x] Profile the whole active model step: reusable bounded `profile` command, contexts 128/512, per-block forward attribution, total backward, transfers/checks/AdamW and launch counts (see `RESULTS.md`, Whole-model profile).
- [x] Burn automatic kernel fusion: implemented fusion-aware custom-op registration and explicit execution selection, then evaluated and **rejected for this workload**. Launches fall 5,712→2,579, but ordinary steps are 13.3%/4.0% slower at contexts 128/512; strict Kron continuation and full-model CPU gates fail unchanged tolerances. See `RESULTS.md`, Burn automatic kernel fusion.
- Dependency patch update: Burn 0.22.1, CubeCL facade 0.11.3 and CubeK 0.3.1. Includes upstream fusion and matmul fixes; existing fusion/autotune restrictions remain unchanged. Vulkan and Vulkan+autotune builds and workspace tests passed. No new performance or numerical-fix conclusion is established by this update.
- [x] Q/K/V, decay, erase, write, output-gate and output projections: compared grouped and single-merged input contractions with unchanged parameter storage/IDs. Adopt explicit `projection_backend: "merged"`; omitted = historical separate. Paired full-step gains repeat at contexts128/512 despite more total launches; dependent output stays separate. See `RESULTS.md`, Mixer projection contraction merging. Activations remain unchanged and are covered by normalization/gating below.
- [x] Normalization and gating: compared custom L2/all-RMS reductions against an expanded sigmoid-output/RMS + softplus/exp-decay scope, with analytic VJPs and explicit selectors. At B1/context512 the paired intervals include zero (not adopted there); at the B8 training workload `fused_norms_gates` saves 115.26 ms per step alone and is **adopted** in `configs/lm512-optimized.json`. See `RESULTS.md`, Batch-8 component optimization round.
- [x] Attribute backward cost: bounded `backward-profile` command at the exact recommended model shapes and contexts128/512; isolated warm forward/forward+backward for every component, all four layers, real unmodified guarded steps, signed coverage/overlap gaps and scan batch/head launch/workgroup sweep. Context512 backward coverage98.98%; scan dominates at324.61ms forward+backward (62.68% of the step). See `RESULTS.md`, Backward attribution.
- [x] Batch scaling at context512: allow positive configured training batches in `backward-profile` without relaxing other recommended shapes; attribute B1/4/8/16/32 complete guarded steps and report tokens/s/memory. B8 reaches2,538tokens/s vs916 at B1; scan share falls60.85%→21.06%, behind HOLA and MLP. B32 fits (2.76GB sampled pool) but reaches only2,433tokens/s. See `RESULTS.md`, Batch scaling.
- [x] MLP matrix-multiply efficiency, broadened to merged-input, mixer-output and vocabulary-head projections: bounded actual F/dX/dW GEMMs at B1/8/16/context512, packing/layout comparison and Vulkan/autotune attribution. **Adopt an explicit guarded autotune option**, not bare autotune: keep MLP/merged inputs tuned, pin dependent output/head to the correct Cube heuristic, and refuse training on failed per-geometry f64 probes. Paired B8 step1619.76→1188.30ms (26.64% less time), B1 560.21→510.47ms (8.88%); both95% intervals exclude zero. All47 active-shape probes and configured T128/T512 full-model gates pass; legacy solver GEMMs remain unsafe/forbidden for autotuned training. Historical defaults unchanged. See `docs/optimization/gemm-efficiency.md` and `RESULTS.md`, Matrix-multiply efficiency; SiLU/residual epilogues remain separate below.

## Batch-8 component optimization round (done)

Adopted selectors are in `configs/lm512-optimized.json` (and
`configs/lm512-optimized-autotune.json` for a `vulkan,autotune` build). Combined
B8/context512 paired step 1,603.76 → 879.04 ms for scan/HOLA/convolution/
normalization; fused update −21.08 ms and analytic SiLU −22.23 ms on top; guarded
autotune 851.80 → 505.62 ms on top of `lm512-optimized`. See `RESULTS.md`,
Batch-8 component optimization round, and `docs/optimization/*.md`.

- [x] HOLA fused read backward: `fused_gpu_pairs` reuses query-VJP edge scalars; cache membership and exact attention unchanged. B8 full step −267.54 ms alone. `hola.md`.
- [x] MLP SiLU and residual epilogues: `epilogue_backend: analytic_backward` (one-kernel SiLU VJP) −22.23 ms on the optimized base; fused-pointwise forward smaller gain; residual adds stay native (no algebra across norms or matmuls). `epilogues.md`.
- [x] KronLSTM more-parallel scan: `fused_parallel256` key-shard partition, exact recurrence; B8 full step −17.10% alone. `scan.md`.
- [x] Short causal convolutions: `convolution_backend: fused_gpu`, document-boundary aware depthwise kernel with analytic VJPs; B8 −69.93 ms alone; padded tensor rejected. `conv-and-norms.md`.
- [x] AdamW and gradient checks: `update_backend: fused_gpu`, per-parameter fused clipping/AdamW with square-sum checks that keep at most 64 partials per tensor (Burn's per-parameter optimizer interface; not multi-tensor packing). −21.08 ms on the optimized base; AMSGrad state preserved through checkpoints. `update-and-loss.md`.
- [x] Embedding, vocabulary head and loss: fused cross-entropy (`fused64`/`fused256`) implemented and **rejected**: B8 paired intervals include zero. Tensor loss retained. `update-and-loss.md`.

Outside the resident step:

- [x] Data pipeline: corpus rows + host packing + upload ≈1.3 ms of a ≈950 ms B8/T512 step (<0.2%); a resident-batch run shows no measurable step gain, so **no prefetch**. Exact-capacity host packing (1.18 → 0.90 ms) kept. `io.md`.
- [x] Checkpoint save/load: measured with all integrity checks: 28.6 MB, atomic save ≈267 ms, validated model load ≈178 ms, validated optimizer load ≈187 ms at B8/T512. No code change: at `checkpoint_every: 4` and a 506 ms step the save adds ≈13% per step, so raise the cadence for long runs instead of adding an asynchronous writer. `io.md`.
- [x] Soft top-k selection: `soft_selection_backend: gpu` keeps selection and feedback state on device; B8×512/k16 feedback 8.64× (V257), 24.44× (V32k). Tiled variant not adopted. `soft-topk.md`.
- [x] GDN-2 training: `scan_backend: fused_gdn2` with full stateful analytic VJP passes independent f64 gates and the CPU model oracle; B8/T512 step with HOLA 2,609.6 → 1,710.9 ms (1.53×), without HOLA 2,095.5 → 1,296.2 ms. Comparison arm only. `gdn.md`, `RESULTS.md`.

## Correctness

- [x] Legacy GPU tensor solver key-gradient mismatch: **fixed**. The rank-4 row-dot `sum_dim(3)` VJP lost the initial-state erase adjoint on Vulkan; replaced by a row-wise dot `matmul`. New regression fails before and passes after; historical T7 fixtures pass. `tensor-solver.md`.
- [x] CPU full-parameter directional check with HOLA: the perturbed forwards now replay the unperturbed forward's kept cache slots, so the difference matches the fixed-membership gradient; ε and tolerance unchanged. All GDN-2/Kron HOLA cases pass at every tested ε (errors ≤ 6.2e-4 vs limits ≥ 3.3e-3). `RESULTS.md`.

## Later steps (optimize when each is built)

- [x] oHC stream mixing (Step 7): fused write, fused read, then gate projection fused with its RMS (`ohc_backend: fused_gate`, deterministic weight-gradient reduction); −263 ms per step over the previous fused path inside the full Step 7 model. Still the largest Step 7 cost (14 stages per step). `ohc.md`.
- [x] Wavelet mixer (Step 7): causal fixed-filter dilated kernel (`wavelet_backend: fused_gpu`); −756 ms per step against the tensor port inside the full Step 7 model. `wavelet.md`.
- [x] Three Lngram components (Step 7): fused symbol cache, addressing, gather and on-the-fly OneBit counterfactual (`lngram_backend: fused_gpu`); −68 ms [49, 75] against the corrected tensor port; no expanded counterfactual tensors retained. `lngram.md`.
- [x] SBD training and causal block replay (Step 8): O(T) clean preparation, immutable clean recurrence/cache snapshots and temporary bidirectional in-block reads through the existing HOLA softmax (`sbd.in_block_reads: hola_read`); no separate attention operator or new parameter. Adopted `sbd.noisy_read_backend: fused_gpu` from accepted B1/T512 complete-step pairs: median saving 2390 ms, 95% paired bootstrap [1668, 3176] ms, faster in 10/11 pairs; omitted selector stays tensor. The diagnostic counted 128 noisy block evaluations / 512 layer reads per B1/T512 step; noisy-HOLA forward sections were 9.18–11.74% of observed step time. B8/T512 reached its 3100s deadline with no paired result; no loop optimization was attempted. Independent f64 and complete CPU/Vulkan/cached/reconstructed gates pass at unchanged tolerance. Full cost, launch and sampled-reservation tables: `RESULTS.md`, `docs/sbd.md`.
- [ ] Learned or adaptive block boundaries (Step 12)
- [ ] Codpula circuit inference, training and adaptive commitment (Step 9)
- [ ] CoCommit coordination passes (Step 10)
- [ ] R²D remasking and TCT training (Step 11)
- [ ] Energy-guided stochastic remasking, unary then pairwise (Step 11)

## Cross-cutting

- [x] BF16 / mixed precision: this Vulkan backend reports F16/F32/F64 but no BF16; f32 kept. FP16-with-f32-state remains a separate unimplemented experiment. `precision-and-shapes.md`.
- [ ] CUDA benchmarks on the DGX Spark (needs an agreed GPU window).
- [x] Larger training batch sizes at current K32/V64/chunk16/context512: B1/4/8/16/32 measured, largest tested B32 fits; full steps, component attribution and memory in Batch scaling.
- [ ] Larger key/value widths and contexts beyond the current K32/V64/chunk16 pilot limits: custom kernels are specialized to K32/V64; storage scaling recorded in `precision-and-shapes.md`; no wider-state kernels or measurements yet.
- [x] Re-profile after the batch-8 round: combined paired benchmark and per-increment sessions recorded; next bottleneck attribution should use `backward-profile` on `configs/lm512-optimized.json`.
