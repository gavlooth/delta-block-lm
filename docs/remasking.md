# Step 11: stochastic remasking inside the active block

## Status, scope and sources

**Standalone profile implemented, 2026-10-25.** R²D, stochastic mask-set energy training and TCT are implemented behind explicit selectors. Section9 records the decisions; section14 records standalone numerical gates, command smokes and costs in `artifacts/remasking/`. Composition with contextual Codpula and CoCommit was published on main as `9e67b7ac` with partial acceptance. Its actual collection failed, and fitting/guided/TCT CLI/B8 checks were not executed; see [RESULTS: integrated11 publication checkpoint](../RESULTS.md#integrated11-publication-checkpoint) and `artifacts/integrated11/commands/`. Earlier arithmetic estimates are not measurements.

Training writes the source `metrics.json` before SBD diagnostics or generation.
End-of-training generation is explicitly skipped when refinement is selected:
the energy must first be fitted to that checkpoint. Run `remask-collect`,
`energy-fit`, then `remask-generate`; guided generation still rejects a missing
energy rather than silently using uniform masks.

The research question is whether **learning where to perturb, while still sampling**, improves refinement at equal work. It is not whether token confidence predicts an error. A confident wrong token remains eligible. An energy head predicts the usefulness of a *set to remask*, not the probability of the current text, and not a deterministic token replacement score.

**Authorized scope:** implement the selected mechanism, its actual training commands, correctness gates, tiny context-32 end-to-end smokes and execution optimization. Every earlier discussion below of quality comparisons, extra arms, sweeps, calibration grids, budget curves, hard-mask alternatives, separate paper-reference training, or policy-transfer studies is **not authorized**. Those passages explain source distinctions only; they are not an execution plan. Fixed-call trajectories are authorized solely as the equal-work energy-training recipe. Only execution-layout comparisons with unchanged mathematics are authorized for timing.

Required sources:

- Zhou, Wu and Zanette, [Random Remasking Scales Masked Diffusion Inference, arXiv:2610.30744v1](https://arxiv.org/html/2610.30744v1), full paper including Appendices A–E; [version page](https://arxiv.org/abs/2610.30744). Algorithm and equation numbers below refer to this version.
- [PLAN §6 and Step 11](../PLAN.md), [SBD](sbd.md), [Codpula](codpula.md), and [predecessor inventory §1.3](predecessor-inventory.md#13-step-11--energy-guided-stochastic-remasking--r²dtct).
- Predecessor root `/home/heefoo/Documents/code/diffusion-relation-extractor`: `crates/dre-relations/src/energy_guidance.rs` and `docs/references/guidance.md`. All predecessor line numbers below refer to the current tree read for this specification, not a guessed historical snapshot.

Do not change the backbone, HOLA reads, oHC, wavelet, Lngram, MoBA retrieval, existing SBD/Codpula defaults or Step 10 as part of this work. CoCommit is specified separately; this document neither adds its tail calls nor changes its design. This page is the Step-11 implementation document; PLAN remains the research requirement.

## 1. State and the two kinds of visible token

For one actual block of length `1 <= L <= 16`, keep these objects separate:

| Object | Meaning | May remasking change it? |
|---|---|---|
| Clean prefix | Prompt and completed earlier blocks, including existing recurrent/HOLA/history snapshots | No |
| `O` | Externally observed positions and **final commitments** in the current block, with their token values | No |
| `A` | Valid current-block positions not in `O`; `n=|A|` | Yes, regardless of confidence |
| `x[A]` | Complete provisional token assignment from the most recent denoising prediction | Yes |
| `R subset A` | Provisionally revealed positions used as token inputs on the next call | Yes; these are not final commitments |
| `M=A\R` | Positions receiving masked/soft inputs on the next call; `r=|M|` | Yes |
| Candidates | Original top-k IDs, probability weights and retained mass | Keep on remasking |
| Visibility | Weight with which candidate feedback enters a masked input | May reset on remasking without deleting candidates |

`SoftState.observed` currently means immutable evidence, and `prepare_circuit` derives evidence from it (`crates/experiment/src/sbd.rs:42–63`). **Never temporarily put provisional reveals in that field.** Add an explicit provisional-reveal mask/token input to the future remasking implementation. The caller must separately supply (a) the token/soft embedding to the backbone and (b) which tokens are evidence in a particular circuit query. This separation is essential for both remasking and TCT.

A four-position example: if position 0 is committed and positions 1–3 are provisional, a size-two remask can choose `{1,2}`, `{1,3}` or `{2,3}`. A very confident position 1 is not removed from this support. Position 0 never enters it. Re-revealing position 1 does not commit it.

Hard invariants are checked before sampling and before writing, outside the learned energy: membership in the actual block, active/nonpadding positions, exact cardinality, immutable `O`, unchanged prefix/cache/history, no cross-document effects and valid vocabulary IDs. Invalid state is an error, not a high energy. `n=0` performs no refinement; `L=1` retains the exact existing causal bypass and performs no remasking, energy scoring or TCT current-token input.

## 2. Uniform R²D: the paper and our exact adaptation

### 2.1 Kernel and limits of the theory

With clean assignment `x`, random masking channel `q[t|0]` and denoiser `f_theta`, §4 defines

\[
\widetilde K_t^\theta(x'\mid x)=\sum_y f_\theta(x'\mid y)q_{t\mid0}(y\mid x).
\]

Replacing `f_theta` by the true posterior gives `K_t`. Proposition 1 proves `p_data K_t=p_data`. Proposition 2 gives

\[
D_{KL}(p_{data}\Vert p\widetilde K_t^\theta)
\le \rho_t(p)D_{KL}(p_{data}\Vert p)+\epsilon_t,
\quad
\rho_t(p)=\frac{D_{KL}(q_t\Vert\widetilde q_t)}{D_{KL}(p_{data}\Vert p)},
\]
\[
\epsilon_t=\mathbb E_{y\sim q_t}D_{KL}(q_{0\mid t}(\cdot\mid y)\Vert f_\theta(\cdot\mid y)).
\]

Larger noise removes more old information but usually increases denoising error; the proposed decreasing schedule repeats each level until predictions stabilize. Appendix A.2 gives the equality with a nonnegative remainder. These results concern the stated stochastic posterior kernels. The paper's practical argmax algorithm, our restricted active-block support, learned state-dependent masking, soft feedback and Codpula-conditioned variants are not automatically exact samplers of the text distribution. Uniform exploration only guarantees mask-set access; it does not repair an incorrect stationary distribution.

The paper uses `alpha_0=1, alpha_1=0` for retained-token probability. This is the opposite convention from the masking probability called alpha in some Codpula formulas. Here `t` is noise, and `r` is an integer number of masked eligible positions; do not share names by assuming the two alphas agree.

### 2.2 Algorithm 1 without an off-by-one change

Paper Algorithm 1 initializes an all-masked input, then loops `i=K-1,...,0`, with `t=i/K`. At each level it makes **up to `M_paper+1` denoiser calls**, not `M_paper`. After each call it chooses `floor((1-t)L)` revealed positions uniformly from the **whole sequence**, not just the previously masked positions. The standard variant copies previously revealed inputs into the complete prediction; the correction variant predicts them again. Both can revise an earlier proposal by remasking it later.

At each level, reset `previous=None` and repetition counter `c=0`. Each identical complete prediction increments `c`; a different one resets it to zero. Exit when `c >= patience`. Thus `patience=2` means **three identical complete predictions**. Draw and apply the next remask **before** that exit check, as printed. Resetting `previous` per level is required. Do not compare marginal vectors or soft-state values for this test: compare the complete provisional token IDs on `A` (tokens in `O` are already fixed).

For our fixed eligible set `A`, replace the paper's free length by `n=|A|`:

\[
t=i/K,\qquad a_i=\lfloor(1-t)n\rfloor,
\qquad r_i=n-a_i.
\]

Use integer arithmetic `a_i=((K-i)*n)/K` to avoid rounded counts. Uniformly choose exactly `r_i` positions of `A` to mask. The complement is the paper's uniform reveal set. Do not select only low-confidence positions or only positions that were masked on the previous call. With `K=n=8`, the masks produced after successive levels have sizes `7,6,5,4,3,2,1,0`; the **first call** sees eight masks. Repeated inner calls at one level use that level's size. The last level still executes its first denoiser call before making the zero-mask input; returning the most recent complete prediction matches the paper.

Selected values: `K=n`, `M_paper=4`, `patience=2`, standard carry-over, retained soft masked inputs and current Codpula FFBS proposals. Five calls/level allow repeated correction and the three-identical-prediction exit test. No budget curves or alternative token operators are authorized. The paper's natural inference default is K=L; its language experiment uses patience2 and block-wise refinement that leaves earlier blocks and future masks unchanged. Our shorter L<=16, soft inputs and joint proposals are explicit model adaptations.

Future implementation pseudocode (one block, no intermediate final commitments):

```text
freeze clean prefix and O; A = valid positions minus O
if L == 1: use unchanged causal decoder
if A is empty: return O
input = O plus masks on A; candidates retained separately
for i in reverse(0..K):
    t = i/K; r = n - ((K-i)*n)/K
    previous = none; repetitions = 0
    repeat at most M_paper+1 times:
        output = denoise(frozen_prefix, input)
        complete = choose_tokens(output)
        copy O into complete
        if standard: copy provisional revealed input tokens into complete
        repetitions = repetitions+1 if complete[A] == previous else 0
        previous = complete[A]
        update detached candidates from this output, without changing O
        mask_set = sample_size_r(A, r, features_from_this_output)
        input = O + complete on A\mask_set + masked/soft inputs on mask_set
        if repetitions >= patience: break
return complete, retained candidates
```

`choose_tokens` calls the current Codpula exact latent-FFBS proposal operator. The existing factorized-output model keeps its existing per-position argmax with ascending-token-ID ties if that model is explicitly selected; no new factorized arm is added. Never label latent-path Viterbi or independent marginal argmax as exact joint-output MAP.

### 2.3 Decode budgets and comparable schedules

One denoising call means one backbone evaluation plus its selected output preparation. Count teacher calls, commitment calls and any diagnostic calls too. The maximum for an isolated R²D episode is `K*(M_paper+1)`, with no hidden initial prediction. Required prefix-cache preparation and clean-history appends are wall-time costs but not noisy-denoiser evaluations. No discarded clean forward is required after the final completed block.

Inference uses the selected adaptive exit and records actual denoiser calls, episode seconds and clean-history appends. Energy-training labels instead use exactly H fixed calls per fork, with identical remaining r sequence and supplied random values, because equal work is part of the training recipe. Pairwise mask scoring adds no denoiser call. No mask-policy quality comparison or operating curve is authorized.

## 3. The user's distribution over size-r mask sets

### 3.1 Normalize the mixture, not its unnormalized terms

For `Omega(A,r)={M subset A: |M|=r}`, define

\[
Z_r=\sum_{M\in\Omega(A,r)}\exp[-E_\phi(M;x,c)/T],
\qquad
P(M\mid x,c)=\frac{\varepsilon}{\binom nr}
 +(1-\varepsilon)\frac{\exp[-E_\phi(M;x,c)/T]}{Z_r}.
\]

This is PLAN §6's **normalized mixture**. Do not normalize `epsilon*U + (1-epsilon)*exp(-E/T)` only once: then energy offsets would change the exploration rate. Require finite `T>0` and `0<epsilon<=1`; pure uniform has `epsilon=1` and bypasses the energy head. Recommended pilot `epsilon=0.1`, `T=1`, margin 1 in training; calibrate on held-out training documents only. Both energies and `T` are in the same arbitrary learned units.

Draw a Bernoulli mixture component first, then draw uniformly or from the normalized energy law. This ensures every feasible set has probability at least `epsilon/binomial(n,r)` in real arithmetic. Energy offsets cancel; the all-equal-energy case is uniform. `r=0` and `r=n` have one set and need no head or random draw. With `r=1`, the selected-selected pair term below is zero; this is not evidence that pair energies are ineffective for groups.

### 3.2 Factorized energy, unary first and pairwise second

Let `m_i=1{i in M}`. Use

\[
E_\phi(M)=\sum_{i\in A}m_i u_i
 +\sum_{\{i,j\}\in G,\ i<j}m_i m_j v_{ij}
 +\sum_k w_k F_k(M),
\]
\[
z_i=\tanh(W_f f_i+b_f)\in\mathbb R^{32},
\qquad u_i=w_u^T z_i+b_u,
\qquad v_{ij}=z_i^T S z_j,
\quad S=(W_p+W_p^T)/2.
\]

The first experiment omits `S` and optional `F`. The second enables the same unary model plus symmetric bilinear pairs. `G` contains unique unordered eligible-position pairs, with no self edge. Recommend **all within-block pairs** at L<=16: 28 edges at L8 and 120 at L16; this catches distant jointly wrong bytes without learning a separate graph. This costs `O(L²)` scalar pair terms, explicitly acknowledged; never expand per-set hidden features or vocabulary-sized edge tensors. A local graph is an alternative in OPEN QUESTIONS.

The predecessor uses local **linear** context-to-label scores, despite the inventory's broader “unary MLP” wording. The small shared MLP above is a position-mask adaptation, not a verbatim source API. Applying a constant binary mask-label compatibility to all edges would give the same energy to every size-r set on a complete graph. The context-dependent `z_i^T S z_j` avoids that degeneracy while preserving the user's symmetric bilinear decision. Negative pair terms can favor jointly remasking two positions; neither pair scores nor optional learned rule weights are forced nonnegative.

`F_k` is absent initially. There are no relation ontology features in this byte LM. Any later language-specific feature needs an explicit formula, evidence of usefulness and independent checks; it may not encode prompt or commitment immutability as a soft penalty. No second Transformer, whole-block attention network, correction-value predictor, energy argmin decoder or gradient shift of token logits is part of this mechanism.

### 3.3 Exact features from this model

For `C` batched blocks padded to `Lmax`, reuse these existing values; none requires rerunning KronLSTM/HOLA:

| Value | Existing origin and shape | Use |
|---|---|---|
| `h` | `DenoisingOutput.features`, `[C,Lmax,D]`, final RMSNorm after oHC readout or ordinary residual path | Contextual representation; includes selected HOLA/in-block/retrieval reads and Step-7 features |
| `a` | `DenoisingOutput.logits`, `[C,Lmax,257]` | Factorized probabilities only when factorized output is selected |
| `p` | `PreparedCircuit.marginals(tokens,evidence)`, `[C,Lmax,257]`, or factorized softmax | Current model probabilities under the declared evidence/temperature |
| Candidate IDs/weights/mass | `SoftState.candidates`, `[C*Lmax,k]`, `[C*Lmax,k]`, `[C*Lmax,1]` | Candidate expectation, original mass and changes |
| `E`, mask | `LanguageModel.embedding.weight`, `[257,D]`; separate `sbd_mask`, `[D]` | Gather actual candidate embeddings without a dense vocabulary expansion |
| Visibility/provisional/validity | `SoftState.visibility`, `.provisional`, `.active`, plus new reveal mask and permanent `O` | Input history and legal positions |

Exact feature provenance is visible in `crates/experiment/src/sbd_training.rs:148–149,225–226`, `model.rs:40–42`, `sbd.rs:42–75`, and `soft.rs:12–17,282–287`. Use the output that produced `x` at the current call; do not mix old hidden states with refreshed probabilities.

Recommended `f_i` has `2D+12=524` entries at D256:

1. `h_i` (256 entries).
2. Candidate expectation at unit visibility, `b_i=sum_k w_ik E[id_ik] + (1-retained_mass_i)*mask` (256 entries), using original, **not renormalized**, selected probabilities.
3. Twelve scalars in this fixed order: `log p_i(x_i)`; full-support entropy `-sum_v p_iv log p_iv`; top-two log-probability gap; retained mass; input visibility; indicator that `x_i` changed from the preceding call; change in `log p_i(x_i)` measured for the **same current token** against the preceding probabilities; L1 change of sparse candidate distributions on their ID union; `i/max(L-1,1)`; provisional reveal indicator; `r/n`; and noise `t`.

Previous-call change features are zero on the first call, by definition, not read from uninitialized storage. Keep the previous full log probabilities only if using same-token log change (about 4.02 MiB at all C512/L8); do not substitute the previous top token's log probability. Sparse L1 compares original weights, not renormalized retained probabilities. For the standard carry-over variant `x_i` may be a retained input rather than the new top token; gather its actual probability. Impossible/nonfinite queried probabilities are errors; compute log values stably, not with an arbitrary floor. Entropy treats `0 log 0` as zero.

For Codpula the probability features are joint **conditional marginals**, not the standalone token-potential softmax. Optional dependence features can add the already-computed `delta_i` and proposal surprisal; optional leave-one-out features require extra evidence queries and their work must be charged. They are not included in the default 524 entries. Circuit log likelihood and best-alternative gain are descriptive features, never labels proving future benefit.

Freeze and detach the backbone, embedding/mask, output probabilities, all previous-call values and sampled decisions for energy training. Gradients reach only the feature MLP, unary head, pair matrix and any explicitly enabled feature weights. Store raw feature values, their definition/version, temperature and checkpoint hash; do not keep unrolled denoiser graphs. With TCT or another checkpoint, generate a separate feature/trajectory dataset or declare that distribution shift explicitly.

## 4. Exact sampling for every supported block

### 4.1 Uniform branch and reproducibility

Order eligible positions by ascending actual block index. Exact uniform size-r sampling can use a host combinatorial draw: draw a uniform integer in `[0,binomial(n,r))` using rejection to remove integer-modulo bias and unrank it in ascending bit-mask order. It is not repeated token draws with rejection. All membership checks use integer indices. `n<=16` makes binomial counts tiny. Empty/singleton supports consume no draws.

Use separate per-document/block/episode streams for mixture choice, uniform-set index, energy-set categorical choice, token proposals and training corruptions; identify them by saved seed and counters. The same base uniforms are supplied to paired branches, with one indexed value per decision even if a branch could skip it. Padding has no stream. Sampling precision/order and random mapping belong in the policy fingerprint, and resumable training saves the stream state. No branch gets extra proposal draws because its mask set is larger internally.

### 4.2 Unary dynamic programming

For unary energy define `w_i=exp(-u_i/T)`. The exact fixed-cardinality law is proportional to `product_{i in M} w_i`, **not** the Plackett–Luce law produced by successively drawing from a token softmax without replacement.

Compute suffix log partitions in host f64:

\[
D[n,0]=0,\quad D[n,k>0]=-\infty,
\quad D[j,k]=\operatorname{logaddexp}
 (D[j+1,k],-u_j/T+D[j+1,k-1]).
\]

Impossible `k<0` or `k>n-j` entries are `-infinity`. The normalizer is `log Z_r=D[0,r]`. Scanning ascending positions with `k` still to select, include position `j` with probability

\[
\exp[-u_j/T+D[j+1,k-1]-D[j,k]].
\]

When `k=0` exclude the rest; when `k=n-j` include the rest; these deterministic decisions use no new random value. Cost is O(nr) preparation and O(n) sampling. The implementation stores **two f64 values/entry**, a minimum-energy offset and residual log partition, so huge equal FP32 energies do not erase log multiplicities. At n16 this is at most289 entries/4,624 bytes. The recurrence above is mathematically equivalent; implement each logaddexp relative to the smaller branch energy, and compute inclusion probability from residuals and branch-energy differences, not by subtracting two enormous raw logZ values. Apply the mixture choice outside the DP. Enumeration verifies all small-set probabilities and normalizers; this is the exact mathematical law to floating precision, not integer-exact irrational probabilities.

### 4.3 Pairwise: exact enumeration, no mixing-time guess

For n<=16 use **complete enumeration** of eligible size-r sets, for production as well as the independent reference. Enumerate integer masks from 0 through `(1<<n)-1`, retaining popcount r. Download the once-computed FP32 unary and scalar pair terms, not all hidden states and not a tensor for every set. Evaluate each retained set in host f64: selected unary terms in ascending position order, then active pairs in lexicographic `(i,j)` order, then optional feature terms in a specified order. This preserves the predecessor's final-energy precision policy. Any diagnostic final energy ranking uses the same order and lowest-mask-index ties; sampling itself is not argmin ranking.

For `s_M=-E(M)/T`, let `b=max_M s_M`, `q_M=exp(s_M-b)` and `Q=sum_M q_M` in the same mask order. Then `log Z=b+log Q`. Sample with `U in [0,1)` using the first cumulative sum **strictly greater** than `U*Q`. The last positive entry absorbs only the residual rounding interval; reject invalid/nonpositive total rather than returning a fake uniform result. Underflowed learned weights do not remove the explicit uniform component. Validate arithmetic extremes and report when the learned branch numerically concentrates on one set.

For `Nsets=binomial(n,r)` and `e=|G|`, direct evaluation costs `O(Nsets*(n+e))` scalar checks/additions, storage `O(Nsets+n+e)`, not `O(Nsets*n*D)`. At L8/r4 there are 70 sets, each with four unary and six selected-pair additions; scanning all 28 edges requires 1,960 edge membership checks per block. At L16/r8 there are **12,870** sets, eight unary plus 28 selected-pair additions per set; scanning all 120 edges takes **1,544,400** membership checks/block. This exact cost is small enough to start at L8 but must be measured at L16 and larger batch counts.

Unary enumeration uses the same ordered f64 sums and remains the oracle and final diagnostic-ranking definition; the unary DP need only match probabilities/numerics to the declared criterion, not bit-identical association. For n>16 reject the exact-enumeration selector before allocation. No larger-block approximate sampler is silently selected. Two separately named later designs are (a) a width-w local pair graph with a cardinality DP, O(n*r*2^w) time, O(r*2^w) working storage, which is exact for that changed graph; or (b) fixed-cardinality Metropolis swaps with a declared number of sweeps, O(sweeps*n*degree) incremental work, which only approximates the dense graph distribution. Neither is part of the initial supported model; §9 records the decision needed before increasing L.

## 5. Composition with soft feedback and Codpula
### 5.1 Retained soft-feedback inputs

On M use the existing probability-weighted candidate-plus-mask embedding with visibility `1/soft_steps` after a fresh prediction. Candidate updates retain original top-k weights and mass and use existing soft_eta/soft_temperature, with Codpula temperature applied before local emission normalization and feedback update temperature one. Provisional reveals use E[x_i]. Neither kind of neural input changes permanent O. Initial visibility zero is the genuine all-masked start, not a hard-mask comparison.

Each noisy pass starts from the same clean-prefix recurrent/HOLA/history snapshots. Temporary noisy states and in-block K/V never enter the clean cache. After the episode, commit all remaining valid positions and append the completed block to clean history. Prepare the extended-prefix cache only if another multi-token block needs it; no standalone discarded final clean forward runs. Existing prefix reconstruction is the correctness baseline; reusable cached execution must match it numerically.

### 5.2 Codpula output during refinement

Prepare current neural potentials/gates once. Standard refinement draws seeded exact joint proposals on M, conditioned on O plus provisional reveals, and copies that call's revealed values. Feedback and energy features use O-only full-support marginals from the same prepared circuit, so the previous-probability features remain finite when a revealed token changes after remasking. This additional evidence query is charged, but does not rerun the backbone.

The existing factorized model keeps its existing per-token greedy operator if explicitly selected; no factorized comparison arm is added. TCT is a training mode, not an implicit switch to a different inference operator. HMM tables, state-specific local normalizers, grouped gates, full257 support and potential temperature remain unchanged. Messages are reusable only with identical prepared outputs/evidence/temperature.

### 5.3 Final commitments

Final commitments are delayed until the end of the refinement episode. Existing O is immutable throughout and excluded from eligible sets. This is an explicit decoder selector, not a silent change to the absent-selector dependence decoder. No dependence test creates intermediate permanent commitments during refinement. With explicitly enabled CoCommit, marks are only temporary announcements under §5.4. Provisional reveals must never be written to SoftState.observed; committing at the end is the only new immutable write.

### 5.4 Approved CoCommit composition contract

The explicit combination reuses the existing saved-prefix, unmarked A and
marked B tail operations; it adds no attention or other model mechanism.
Prefix/A/B neural work and circuit preparation remain batched across active
blocks. Host bundle choices may loop over metadata from a batched readback,
but production refinement/TCT must not call single-block assignment in a row
loop. B restarts from the same saved noisy-prefix representation as A, never
from A's tail state; neither branch mutates the clean prefix.

For standard R²D, A announces `S subset M`, where `M=A\R` and `A=valid\O`.
B supplies one coherent conditional joint proposal for M with temporary
proposal evidence `O+R`, carrying R unchanged. Correction mode announces
`S subset A` and B may revise R. Do not splice B samples on S into A samples
outside S: the complete proposal comes from one prepared B distribution.
If S is empty, use A unchanged, skip the marked pass and record that fact.
The selected output's O-only full marginals and final hidden states supply
feedback and mask-energy features. Marks never enter evidence and never
finalize positions inside the episode.

TCT's unmarked A objective remains the all-free clean correction joint NLL
given permanent O, including wrong provisional R. B adds the existing CoCommit
joint event on its announced `S subset A`, marginalizing every other free
position. R remains eligible for correction supervision; only O masks
token-potential gradients. No raw proposal is evidence in either correction
loss. Detached B proposals/marginals advance the existing confidence reveal
chain, retaining the chain's prescribed earlier provisional inputs.
Accumulate both branch gradients and normalized circuit statistics/counts
using CoCommit's weighting, then apply one neural/circuit update.

Ordinary SBD/CoCommit exact row accumulation uses `sbd.rows_per_pass`, with
full-batch masks/bundles sampled once. Persistent-chain TCT uses only
`tct.rows_per_pass`; never nest these accumulators. Both preserve full B*T
normalization and indexed RNG. A combination-specific algorithm revision must
enter training/pilot identity only for CoCommit with refinement or TCT;
historical standalone/omitted identities remain unchanged.

Endpoint scoring uses that same coordinated model rather than silently reading
A only. Its input remains retained soft candidates, never hard provisional R
or gold. Selection and evidence are O-only, S ranges over all free positions,
and gold enters only the conditional-NLL scorer. Paired fork endpoints share
their keys and an endpoint call tag for common announcement/proposal uniforms.
Score the coherent B distribution, or A only when S is empty, in a batched
readout; charge its actual prefix/tail/head/circuit work separately from the
one logical endpoint-scoring call.

Report logical R²D iterations separately from shared-prefix, A/B tail/head
calls, circuit preparations/evidence queries/FFBS, clean-cache preparations and history appends.
Full-backbone NFE alone hides a marked partial pass. Standalone Step11 costs
below are not measurements or same-cost comparisons of the combination.

## 6. Training the mask-set energy from outcomes

### 6.1 Frozen-denoiser paired trajectories

Start from one validated **frozen** SBD/Codpula checkpoint and training documents only. Freeze the feedback mode, token operator, temperatures, prefix, actual block, `O`, eligible positions, remask schedule and output definition. Collect decision states from uniform refinement, not from gold-filled active blocks masquerading as model errors. The clean prefix can be teacher-forced as in SBD training; record that limitation, and evaluate on generated prefixes too. Split train/validation/test by source document **before** producing trajectories; two states from the same document cannot straddle the split.

At a nontrivial decision `0<r<n`, retain a detached feature snapshot. Generate distinct same-size candidate sets and fork the **same full state**, including candidates/visibility, clean snapshot and indexed random draws. The first intervention is the mask set only. Run both branches for exactly `H` subsequent denoiser calls with the same remaining r schedule and uniform continuation policy. No final commitment is made within this comparison; `O` stays identical. Endpoints and losses cannot be compared across different `A`, r or budgets.

Recommended pilot: **512 decision states, four paired comparisons/state, H=2**, stratified by r, early/late block position and provisional error count; source strata use training targets only for analysis/collection, never as inference features. This yields 2,048 pairs and 8,192 branch denoiser-call/block units, plus shared prefix/decision-state collection. These are proposed sizes, not a reported dataset. Save seeds, clean targets, starting-state identities, selected sets, continuation schedule, output checkpoint, outcome definition and all actual work. Refresh or expand based on held-out ranking/quality uncertainty, not to fit a favored result.

### 6.2 Proposal-weighted corruption, adapted without a false gold mask

The predecessor changes one eligible categorical slot, excludes every gold alternative, and draws a replacement using detached model-proposal probabilities. We port **that corruption discipline**, not its relation labels. There is no known gold mask set in this LM; the better set is determined only by equal-work outcomes.

For each pair, draw base set `A0` uniformly from `Omega(A,r)`. Draw one member `i` uniformly from `A0`; choose an insertion `j` outside it with detached weights `q_j=1-p_j(x_j)`, normalized over the complement, and set `B0=(A0\{i}) union {j}`. As in the predecessor, if all allowed proposal weight is zero, draw uniformly over allowed insertions; nonfinite or negative proposal probabilities are invalid inputs, not a fallback. Alternate half the pairs with an independently uniform distinct `B0` instead of a one-swap corruption. The swap preserves exact r; the uniform half prevents the training distribution from omitting confident, jointly wrong groups. This proposal-weighted sampler is **only for collecting candidate interventions**: it is not the learned inference distribution and does not assign preference labels by confidence.

If there is only one feasible mask set, the pair has zero eligibility and consumes no trajectory work. Reject duplicate branches before rollout. Multi-swap or independent-set pairs are needed for the pairwise experiment; a one-swap-only dataset can miss improvements requiring a whole group. Preserve per-document RNG streams and detached proposals. The exact mixture, and the distinction between predecessor literal code and this position-swap adaptation, belong in the dataset fingerprint.

### 6.3 Outcome, orientation and the exact rank loss

Recommend endpoint loss `J=-log p_theta(gold_A | endpoint_soft_state,O)/n`, the mean conditional joint NLL of the paired clean free positions under one fresh frozen-denoiser endpoint readout. On the factorized arm use the corresponding mean clean-target cross-entropy. The endpoint input uses retained provisional candidate feedback, not clean targets, and the evidence indicators fix only permanent O. This measures whether the intervention makes subsequent recovery of real held-out text easier, using dense probability information rather than a sparse exact-byte match. Smaller J is preferred. A scoring call is identical for both branches, counts in collection cost, and does not update their states. This choice is expected to give more useful labels than scoring the model's own generated text, which can reward dull high-probability errors; this is reasoned speculation, not a measured advantage.

Exact-byte error and clean causal likelihood are alternative outcome proxies, **not authorized training objectives or quality diagnostics in this implementation**. Only the selected clean-target conditional NLL is collected. Never condition on incorrect provisional values and then request their gold likelihood: that can have zero support. Gold tokens occur only as scoring targets, never features or refinement inputs. TCT is not required to define finite clean-target likelihood from retained soft candidates with only O fixed.

Let `good` be the branch with smaller J and `bad` the other. Set pair weight `e=0` for duplicates, unavailable supervision, nonfinite outcomes or ties `|J_A-J_B| <= tie_tolerance`; record nonfinite cases as failures as well, not hidden dropped successes. Recommend `tie_tolerance=2e-5+3e-4*max(|J_A|,|J_B|)` for deterministic numerical ties; reward-noise uncertainty is a separate question. Otherwise `e=1` and train

\[
\mathcal L_{rank}
=\frac{\sum_b e_b\,\operatorname{softplus}
 (E_\phi(M_b^{good};f_b)-E_\phi(M_b^{bad};f_b)+m)}
 {\max(\sum_b e_b,1)},\qquad m\ge0.
\]

Port `energy_rank_loss` **semantics exactly** (`energy_guidance.rs:1106–1115`): stable softplus; finite nonnegative margin; replace inactive good/bad energies with finite zero **before** subtraction/softplus; mask the resulting inactive losses; multiply by eligibility; normalize by summed eligibility clamped to one. With zero eligible pairs return finite zero and zero gradients. A numerical tie supplies no ordering. Do not regress a scalar future gain, do not use cross-entropy against a guessed gold mask, and do not replace this loss by a deterministic selection policy.

Use FP32 differentiable sums for energy training, with gradients through selected unary and unique pair terms. Offline energy fitting does not require partition functions or differentiating through DP/sampling. Recommend existing AdamW implementation/settings for energy-only parameters, margin 1, paired batch 512, early stopping by held-out pair loss/ranking plus the actual mask-policy curve. A constant unary offset cancels at fixed r; do not interpret it as calibrated usefulness. Start unary, then initialize the pair matrix at zero and fit a pairwise arm from the same unary checkpoint/data plus explicit group fixtures. Compare held-out trajectories not used for fitting or selecting epsilon/T.

## 7. TCT is separate denoiser training, not energy supervision

### 7.1 Paper details

Paper §4.3 Proposition 3 follows Jensen's inequality:

\[
D_{KL}(p_{data}\Vert p\widetilde K_t^\theta)
\le\mathbb E_{y\sim\widetilde q_t,\ x_0\sim p_{data}}
[-\log f_\theta(x_0\mid y)]-H(p_{data}).
\]

This is not a guarantee that any paired finite training recipe reaches the true posterior. TCT approximates inference training by omitting R²D's correction inner loop, keeping a fixed K-round reveal trajectory and using confident reveals. The text calls this a “low-confidence reveal policy” once; **Algorithm 2, Appendix B and the Appendix-D.2 ablation explicitly reveal the highest-confidence masked positions**. Implement those concrete instructions, not the stray wording.

Algorithm 2 maintains independent streaming chains, clean targets and counters `n_j in [0,K_train]`. One update is:

1. Forward all current chain inputs and compute Eq. (2), cross-entropy against the clean target at **every free answer position, masked or provisionally revealed**. Update denoiser parameters.
2. Using detached outputs from that forward (pre-update outputs, no second forward), refill each chain whose counter already equals `K_train` with a fresh example, all masks and counter zero.
3. For every other chain, take per-position argmax predictions and `s_i=max_v p_i(v)` at still-masked positions, select top-k with thresholding, reveal the selected predictions, and increment its counter.

The `n_j=K_train` all-revealed endpoint is trained **before refill**. A K_train-round trajectory therefore supplies K_train+1 training inputs, including the initial all-masked and final input. Revealed values remain their earlier model choices during the training chain; the correction loss, not an overwrite in the chain update, teaches the model to disagree with those wrong values. Do not use teacher-forced clean tokens for reveals and do not backpropagate through discrete reveals or previous optimizer steps.

Paper Eq. (2) averages over chains and **sums** tokens; the model's existing SBD objective uses token normalization. These must be recorded as different reductions, not confused by reusing a mean loss. The paper leaves top-k/threshold values and many optimizer/mixing details unspecified. Its logical-task result 97.5% versus 85.5% for random-position TCT reveals (Appendix D.2) supports retaining confidence-based *training reveals*. It says nothing against stochastic learned **inference mask sets**; these are different mechanisms. Temperature-one stochastic TCT token reveals scored 96.8% versus 97.5% with argmax on Sudoku. Appendix D.1 reports weaker inference scaling from low-confidence remasking and worse stochastic inference on argmax-trained TCT, so keep operator changes explicit.

### 7.2 Our streaming-chain implementation contract

For B8/T512/L8 the block axis has C512 independent length-eight chains, each with a paired target block and its frozen clean prefix for that chain. Retain IDs, masks, candidate state, targets, round counter and RNG across optimizer steps, **not autograd tensors**. Recompute clean-prefix representations after every optimizer update; caches from old parameters cannot remain live. Group all current blocks into the existing batched noisy path. A chain refill comes from training documents; no chain crosses a document or carries a previous document's recurrent state. Save loader state and chain values/counters for exact restart.

Recommended K_train=4, confidence threshold 0.9, guaranteed scheduled reveals as follows: at counter n<K_train, let `u` be still-masked free positions and `q=ceil(u/(K_train-n))` (zero if u=0). Reveal all masked positions with s>=0.9, plus enough highest-confidence remaining positions to reach q. Ascending-position then ascending-token-ID ties; on the final reveal round all remaining positions are revealed. This is a **fully specified repository choice** for the paper's unspecified top-k/thresholding, not a numerical setting reported in the paper. An already fully revealed chain stays unchanged until its K_train endpoint update/refill.

The selected TCT mode is a joint adaptation: backbone inputs include provisional reveals, but circuit evidence contains **only permanent O**. Train conditional joint NLL of all gold free positions, with the existing local-normalizer derivatives and dual-flow update. Mask observed-potential gradients only at permanent O, never at provisional reveals; observed features/gates still differentiate normally. Incorrect provisional reveals therefore remain correctable rather than making the target event impossible. L1 keeps the causal bypass.

Use token-normalized joint TCT plus unchanged clean causal loss. There is no separate factorized TCT or hard-mask arm. TCT trains the denoiser/circuit, not the energy; offline energy fitting freezes its selected source model. Streaming chains revisit each target K_train+1 times, so record unique bytes, scored visits, optimizer steps, actual calls and total time rather than suggesting token counts alone measure work. No crossed training/policy study is authorized.

## 8. What to port and what to leave behind

All paths in this table are relative to the predecessor root named above. Port into active crates with provenance, not a dependency on the predecessor application.

| Source path:lines | Keep/adapt | Drop or change |
|---|---|---|
| `crates/dre-relations/src/energy_guidance.rs:186–210` | Local scorer + symmetric bilinear parameter split; optional feature-weight separation | Relation label count/output shape; local linear becomes the small shared mask-feature MLP here |
| same `:680–683,739–764` | Symmetrize once; project once; reuse attached local projection across better/worse sets; sum each factor once | Dense relation-assignment and entity-sharing graph API; use m_i and z_i for mask interactions |
| same `:224–299,340–365` | Validity zeroing **before** projections, unique edges, per-example ranges and deterministic segmented-sum pattern | `SlotPlan`, resolved entity endpoints and packed-relation metadata; never connect different blocks |
| same `:10–69,72–184` | Reference CSR incident sum/VJP pattern if profiling later needs it | Do not port the custom backend/CSR machinery for 28 scalar edges unless measured faster than ordinary batched reductions |
| same `:371–393,842–912` | FP32 model terms -> host f64 ordered energies; gather selected terms, prepare once, deterministic final diagnostic ranking | Two-to-eight final-graph argmin as the decoding policy; our output is a categorical **mask-set sample** over all feasible sets |
| same `:825–839` | Categorical gathers with attached parameter gradients, no edge-by-vocabulary one-hots | Relation soft-factor object and its four fixed feature kinds |
| same `:1106–1115` | Exact eligible-masked softplus margin-rank semantics, including zero-safe inactive energies | Names “gold graph”/“negative graph”; use outcome-ranked mask sets |
| same `:1117–1182` | Detached proposal-weighted alternative sampling, excluded original choice, uniform zero-weight case, one stream/document, zero eligibility with no alternative | Label/annotation/multi-label metadata; fixed-r mask corruption needs the swap adaptation in §6.2 |
| same `:395–481,483–537,565–677,1184–1317,1319–1540` | FP64 cancellation/order fixtures, dirty padding, categorical/dense energy and gradient equivalence, grouped sums/VJPs | Relation fixtures, ontology counts and predecessor's looser CSR-specific tolerance; use this repo's tolerance |
| same `:766–786,914–916` | Read as a reference for analytic derivatives only | Do not wire token-logit gradient correction into remasking; it changes another distribution and is outside the selected mechanism |
| same `:919–1104` | Sparse optional polynomial-factor idea if a concrete LM feature is later selected | All type, reverse-label, exclusion, functional-cardinality and entity-identity features; no invented byte ontology |
| `docs/references/guidance.md:7–20,24–46,63–73,173–212` | Factorized form, no second Transformer, ordered host f64 final energy, rank-training discipline and no huge edge expansions | Hard relation solver, Wikidata fetching, missing relation-label semantics, graph grouping/mention decoding |

Additional references, not wholesale ports: `diffusion_scheduler.rs:125–216,265–301` for same-snapshot multi-call execution; `training_query_chain.rs:61–114` for detached chain advancement; `training_query_batch_loss.rs` for document-local RNG continuation. No R²D or TCT implementation exists there. Already integrated soft-feedback and Codpula primitives must be reused, not copied into a second implementation.

## 9. Decisions

**Decided 2026-10-25 in the user's absence from the highest-ROI recommendation; reversible.** The director selected the following. This table supersedes conflicting proposals elsewhere on this page. Predecessor user decisions remain in force. No evidence is required before making a sensible engineering estimate; the reasons below distinguish speculation from paper observations.

| Question | Decided implementation | Reason |
|---|---|---|
| Q1 | M_paper=4 only; no budget curves | Five calls/level buy meaningful correction and an exit opportunity without importing a large logical-task budget (assumed). |
| Q2 | K=n | Every integer remask size remains reachable at short block lengths; this follows the paper's natural default. |
| Q3 | Patience2 adaptive inference; fixed calls only for paired training collection | The paper's exit saves wasted evaluations; equal-work branches are essential to the training label, not a quality study. |
| Q4 | Delay final commitments until episode end; never remask a final commitment | This preserves useful exploration and the model's immutable-token invariant. |
| Q5 | Current Codpula FFBS proposals; existing factorized operator only when the existing factorized model is selected | Preserve the selected joint model rather than introducing a new comparison. |
| Q6 | Retained soft feedback only | User-required candidate information is worth its small embedding work; no hard-mask arm. |
| Q7 | 524->32 contextual features at D256 | Context can identify useful groups that scalar confidence misses, for about 69M MAC/full block axis (assumed). |
| Q8 | Marginals; dependence scores only if already computed | No extra backbone queries for speculative features. Full-support O-only feature marginals are necessary for finite same-token log changes. |
| Q9 | Unary cardinality DP plus enumeration oracle | Exact law with O(nr) sampling preparation and an independent small-block check. |
| Q10 | All within-block pairs | Twenty-eight L8 edges cheaply capture distant joint corrections that a local graph misses (assumed). |
| Q11 | Exact pair enumeration through L16; reject n>16 | Correct probabilities outweigh approximate-chain complexity at the supported size (assumed). |
| Q12 | Keep L<=16 | Longer contexts do not require exponentially larger mask-set support. |
| Q13 | Clean-target conditional joint NLL from endpoint soft input | Dense text-recovery labels are expected to beat sparse byte hits or model self-preference for one scoring pass (assumed). |
| Q14 | Half proposal-weighted swaps, half independent uniform distinct sets | Preserve predecessor proposal weighting while covering whole-group alternatives at the same rollout count (assumed). |
| Q15 | 512 decision states, four pairs/state, H2 | Two-step outcomes and size coverage are expected to justify the collection cost without full long trajectories (assumed). |
| Q16 | Deterministic common-uniform pairs and numerical tie filter | Removes irrelevant branch randomness without repeated-seed collection. |
| Q17 | Frozen backbone; offline rank fit | Stable outcome labels, small fitting graphs and predecessor margin-ranking semantics. |
| Q18 | Epsilon0.1, T1 fixed; no grid | Keeps positive exploration with reproducible cost; no calibration work is authorized. |
| Q19 | No optional rule features | No byte-language rule has been selected; do not import relation ontology. |
| Q20 | K_train4, threshold0.9 plus scheduled minimum | Five visited inputs/chain give repeated self-error training; confident reveals follow the paper's useful training choice. |
| Q21 | Conditional joint all-free-position NLL only | Wrong revealed inputs must remain correctable without contradicting evidence support. |
| Q22 | Token-normalized TCT plus unchanged clean causal loss | Keeps optimizer scale and causal learning without duplicating the full ordinary SBD objective. |
| Q23 | Streaming chains; explicit TCT training mode | Actual self-error exposure without K-fold unrolling each optimizer step. |
| Q24 | Host f64 ordered sums; one batched readback | Preserves the predecessor precision decision while amortizing synchronization. |
| Q25 | Policy-transfer study dropped | Outside the authorized implementation scope. |
| Q26 | Numerical gates, exact-law enumeration and tiny command smokes | No quality study; smoke R²D generation, collection, fitting, guided generation and TCT training end to end. |

Later that day exact row-chunk gradient/flow accumulation was selected under the user's standing optimization authorization. The director requested the largest candidate that fits at B8/T512, trying4 then2 then1 only if needed. Four rows fit on the first corrected diagnostic, so the selected config records `tct.rows_per_pass:4`; no smaller candidate was run. The new contextual Codpula profile is implemented separately; this feature's joint dispatch uses `model.codpula`/`!is_factorized`, and its merged profile receives separate parent integration gates.

Implementation clarification: standard FFBS proposals condition on O plus provisional reveals, while feedback/energy features use the prepared circuit's **O-only full-support marginals**. Otherwise the same-current-token previous-log feature can become log zero after a provisional reveal changes. This adds a circuit marginal query, not a backbone call, and is charged in execution cost. It avoids arbitrary floors while retaining all 524 feature entries.

## 10. Cost at B8/T512/L8 and later

### 10.1 What is measured and what is estimated

The user's current batched SBD+Codpula complete step is approximately **8.0 seconds at B8/T512**. This is the supplied measured baseline, not a generation-forward time. Existing docs record separate B1 measurements and an older SBD-only B8 diagnostic; do not substitute them for a missing B8 generation benchmark. All additions below are **arithmetic/storage estimates or explicitly hypothetical timing scenarios**, not observed speedups or a promise of wall time.

At this training shape, `N=B*T=4096` positions and `C=N/L=512` block chains, D256, HMM K32 and V257. A “full block-axis pass” processes all 512 blocks as independent examples. Actual autoregressive generation at B8 has only **8 active blocks = 64 positions per frontier**, 64 frontiers to generate 512 bytes/row; those later blocks cannot be denoised in parallel with unavailable prefixes. Keep these two workloads separate.

### 10.2 Added work per denoising pass

| Addition | All C512 blocks / N4096 | Generation frontier C8 / N64 |
|---|---:|---:|
| Candidate expectation, k4/D256 | 4,194,304 weighted embedding entries | 65,536 |
| Full entropy/probability features, V257 | ~1,052,672 probability entries/reduction | 16,448 |
| Feature MLP 524->32 plus scalar unary head | ~68.81M MAC | ~1.075M MAC |
| Pair preparation: zS, then 28 dot products/block | ~4.65M MAC | ~72,704 MAC |
| Unary DP at worst L8 r4 | <=512*9*5=23,040 f64 cells | <=360 cells |
| Pair enumeration at L8/r4 | 35,840 sets; 358,400 selected factor additions; 1,003,520 pair membership checks | 560 sets; 5,600 selected additions; 15,680 pair checks |
| Downloaded unary+pair scalar terms | 512*(8+28)*4 = 73,728 bytes (72 KiB) | 1,152 bytes |

MAC means multiply-accumulate, not one floating-point operation and not a log/exp. Terms downloaded once serve all sets; no per-set device call is allowed. At 10–100 GMAC/s effective head throughput, 68.81M MAC would take **0.69–6.88 ms** for the full block axis or **0.011–0.108 ms** at one generation frontier, before launches, reductions and synchronization. These are hypothetical arithmetic scenarios; a one-kilobyte synchronization can dominate arithmetic on a small frontier. If an added readback/sampler costs 1 ms/frontier, L8/M1's 16 passes times 64 frontiers add about **1.024 s per B8/T512 generated batch**; the recommended M4 cap gives **2.56 s** before adaptive exit. At 10 ms these become 10.24 s and 25.6 s. Measure this; do not extrapolate 8.0s training time into a forward latency.

Pairwise L16/r8 has 12,870 sets/block, 463,320 selected additions and 1,544,400 pair checks/block. B8/T512 then has C256 blocks: ~118.6M selected additions and ~395.4M pair checks per full pass. Thus “blocks are short” is not a license to ignore host enumeration. At longer T with L held fixed, arithmetic/state counts scale approximately with B*T for batched training; current full-prefix reconstruction makes total generation work grow faster than the number of active bytes. Increasing L changes enumeration combinatorially and needs Q12 decided first.

### 10.3 Trainable parameters and working memory

At D256 the shared feature MLP and unary head have `524*32+32+32+1 = 16,833` parameters; symmetric pair storage uses another 1,024 unconstrained matrix entries. Their f32 weights are about 66 KiB unary / 70 KiB pairwise; optimizer histories and gradients multiply this small count. No new recurrent state or HOLA slot is introduced.

`[4096,524]` feature input is **8.1875 MiB** if materialized, `[4096,32]` hidden is 0.5 MiB; candidate expectations add 4 MiB if not fused with assembly. Existing final h is reused, not cloned to retain an unrolled graph. A previous `[4096,257]` probability table adds about 4.016 MiB; sparse candidates and scalar history are much smaller. Avoid retaining both raw f and a second identical concatenation solely for scoring. At a generation frontier these counts are 1/64. A pairwise enumerator needs only one block's energies/weights at a time: at L16 at most two 12,870-entry f64 arrays, about 201 KiB, plus O(L²) terms; do not multiply this scratch by every block without a measured reason.

For offline fitting, 512 stored feature snapshots of L8 contain 4,096 token feature rows, about **8.19 MiB**, before targets/state metadata. Four pairs/state refer to those features rather than copying them four times. One 512-pair optimizer update evaluates at most 512 unique states (4,096 positions), prepares ~68.81M MAC of unary features plus their backward work, and gathers two sets/state. Approximate MLP forward+backward is <=~0.21G MAC if all 512 states are distinct; shared states reduce it. A 10–100 GMAC/s hypothetical rate gives 2.1–21 ms arithmetic, **not** an end-to-end step estimate. Denoiser training can be stopped during this energy-only fit; it need not add 8 seconds to each energy update.

### 10.4 Full training steps, TCT and data collection

Ordinary SBD+Codpula training without energy data collection stays approximately **8.0 s/step** by supplied evidence; generating an offline policy dataset is not “free training overhead.” If feature scoring is additionally enabled on an ordinary step, the increment is one small head plus feature reductions/readback only; a measured 10/100/1000 ms increment would be 0.125%/1.25%/12.5% of 8 seconds. These are scale examples, not predictions.

Let `C_clean` include clean forward/state preparation and the clean-loss backward share, `F_noisy` one detached batched noisy forward/output preparation, and `G_noisy` one differentiable noisy forward/output/backward share. Roughly,

`8.0s = C_clean + F_noisy + G_noisy + C_optimizer`

for the current two-pass soft-feedback step. Streaming hard-input TCT replacing that noisy objective has

`C_TCT = C_clean + G_TCT + C_optimizer + C_chain_update`.

There is **no K_train-fold forward multiplier per optimizer step**. The all-free-position objective keeps all chains active, versus ordinary masking's expected 8/9 active L8 blocks; this can increase noisy work by up to about 12.5% before the objective's derivative differences. As a planning scenario only, if the removed teacher is 10–30% of the 8s baseline and TCT's all-chain/noisy-gradient changes add 0–15% of baseline, a step would be about **5.6–8.4 s plus chain-update/readback**, not a measured forecast. Keeping an extra detached soft-feedback teacher instead returns the teacher cost; adding TCT on top of all SBD objectives costs another noisy student. Measure both isolated and complete optimizer cost before choosing either.

K_train=4 revisits each block in five inputs across five optimizer steps. At 8s/step, the conditional baseline-priced five-visit cycle would be 40s, not 8s; compare unique training bytes and cumulative compute accordingly. TCT needs fresh prefixes after parameter updates, while a frozen-denoiser policy rollout can reuse prefix caches within an episode.

The proposed 2,048 pairs/H2 require **8,192 branch call/block units**, about 16 full C512 call-equivalents when perfectly batched, plus shared decision-state collection and endpoint scoring. Two endpoints/pair add **4,096 scoring calls/block units**, about eight full-axis equivalents of the chosen scorer. Overall collection time is

`16*F_branch + 8*F_endpoint + C_state_collection + C_prefix_preparation + transfers`,

where F values are separately measured frozen full-axis times. Do not multiply 24 by the **8-second optimizer step**: collection has no backward/update, and a clean scorer differs from the noisy scorer. Larger H, repeated seeds or commitment composition scale those explicit counts.

Generation work for pure L8/M1 at B8/T512 is 64 frontiers*16 calls = **1,024 B8 frontier evaluations**; the recommended M4 cap is 64*40 = **2,560**, before adaptive exit. Both add initial prefix-cache preparation and 63 extended-prefix cache preparations before subsequent frontiers; no unused final clean replay is required. For adaptive scheduling the maximum is `64*8*(M_paper+1)` and actual calls must be reported. Dependence composition adds one assignment call/commitment round and can repeat the full episode up to L times/block; never present its cost as the same 16- or 40-call decoder.

## 11. Correctness plan

The numerical contracts below are implemented in unit/CLI references, using the existing integration binary rather than another Burn-linked executable. Observed reports and maxima are in section14. Independent sequential host-f64 references remain separate from tensor/GPU code. The unchanged criterion is

`abs(actual-reference) <= 2e-5 + 3e-4*abs(reference)`

for f32/tensor/GPU comparisons, without relaxing it for a new kernel. Require much tighter f64 self-consistency before using a reference. Compare log probabilities in log space to expose hidden underflow. Empirical sampling frequencies need statistical confidence intervals, **not** this numerical tolerance.

1. **Support and normalization:** for all n<=8, every r, representative/permuted O/validity patterns and tail blocks, enumerate every allowed mask set. Check support, exact r, mixture normalization, constant-energy uniformity, epsilon1 bypass, common-energy-shift invariance, temperature behavior and r0/rn singleton cases. At n16 enumerate r8 and random energies too. No padding/committed/prefix position can receive nonzero selection probability.
2. **Unary DP:** compare log Z, every set probability, marginals and expected mask count against independent subset enumeration. Use an n3/r2 counterexample showing successive softmax sampling differs. Trace exact uniform interval maps for small categorical cases, deterministic inclusion/exclusion, ascending ordering and all legal random inputs. Gradient of log Z w.r.t. u equals `-Pr(i in M)/T` in the mathematical reference.
3. **Pair factors:** compare host ordered energies against dense m_i/m_j references and selected gathers, including asymmetric W before symmetrization, unique edges, isolated vertices, negative pairs, zero pair matrix recovering unary law, r1 vanishing pair terms and a fixture whose two confident wrong positions need a joint mask. Verify scalar-rule-free behavior initially.
4. **Sampling:** check enumeration-CDF mapping using exact dyadic fixtures and all interval boundaries, then Monte Carlo frequencies against exact probabilities with predeclared family-wise confidence accounting. Verify the mixture branch itself occurs at its configured rate. Test extreme finite energies/T, impossible inputs, zero/nonfinite totals and no fake success on errors. Preserve exact-cardinality support even when learned weights underflow.
5. **Energy and rank gradients:** independent f64/central-difference checks of f->z->u, symmetric W, selected-set sums and softplus rank loss. Include zero eligible pairs with NaNs in inactive input energies, ties, margin0, duplicate candidate pairs and highly positive/negative softplus arguments. Frozen h/embeddings/probabilities receive no gradients; selected energy parameters do. Test zero-init S does not alter the unary policy.
6. **FP64 order:** port the predecessor cancellation fixture `[16777216,1,-16777216]` versus `[0.5]`, equal-energy index ties and cross-thread equality for final energy ranking. Compare training FP32 sums with the f64 reference under tolerance, not assumed bit equality. CDF/set order, candidate order and precision fingerprints must survive checkpoint restart.
7. **State transitions:** replay Algorithm 1 by hand/reference at L2/4/8, test first call all masked, every r, M+1 calls, patience+1 identical outputs, reset per level and final t0 behavior. Uniform sets can remask any provisional reveal. Observed/final commitments are bit-identical after every pass; noisy calls never mutate clean states or replay before completion. Length1 bypass is bit-identical.
8. **Soft/circuit composition:** remask visibility without erasing IDs/weights/mass; test hard versus retained-soft realizations, dirty NaN padding and noncontiguous inputs where supported. Compare per-block/batched, cached/reconstructed, CPU/Vulkan values and all energy/TCT gradients at the same supplied random draws. Codpula proposals and dependence scores must share the exact prepared distribution; score only supported proposals.
9. **Paired data:** both forks start from identical full states; continuation calls, r schedule, O and indexed random draws are identical except the first mask intervention. Verify no target leakage into features/refinement, no document-split overlap, proposal-weighted swap law and uniform zero-weight case, winner orientation and tie-zero gradients. Save actual call counts; zero eligibility creates no rollout. Include n0/n1, repeated IDs and short final blocks.
10. **TCT:** retain self-generated wrong revealed inputs and confirm they receive nonzero correction loss/gradient; permanent O does not. Test fully revealed endpoint before refill, K_train+1 input visits, pre-update reveal outputs, counter/refill/RNG restart equivalence and parameter-fresh clean caches. For joint TCT verify clean correction likelihood remains finite when a provisional revealed input is wrong, while making it permanent evidence would reject that target event. Verify intrinsic circuit flows and observed-potential gradient masks use permanent O only.
11. **Complete model:** preserve existing recurrence/HOLA/Step7 references and historical disabled-option serialization/checkpoint behavior. Enabling remasking changes decoding identity, energy parameters/features change model identity, and TCT changes training identity. Save policy/chain state for exact continuation. No changes to CoCommit are included.

Acceptance for this implementation is numerical correctness, exact-law enumeration/sampling checks and tiny context-32 smokes of each new command. A quality advantage is neither required nor measured in this turn. Record seeds, counts, checkpoints, selected options, work, errors and execution timing; do not report a numerical pass as evidence of improved text quality.

## 12. Optimization and implementation order

1. Implement the state separation and scalar f64 reference first, then uniform Algorithm 1 and exact set sampling. Reuse the current public noisy-block output/soft primitives; do not create another denoiser. Add energy fitting only after paired outcome replay and its random mapping are fixed. Add pairwise after unary, and TCT as a separate training path.
2. Prepare final h, model probabilities and sparse candidate expectations once/pass. Compute z once/position; prepare zS once and one scalar/pair. Batch the block axis and unique feature states, reuse local projections for both members of a rank pair, and use selected gathers rather than `[sets,positions,D]` or `[edges,257]` expansions. Set membership and enumeration remain host integer work.
3. Download one compact buffer of FP32 unary/pair terms per frontier or full batched collection call. Run ordered f64 accumulation/DP on the host with reusable scratch. No per-mask GPU dispatch or device readback. Keep detached teacher temporaries dead after candidates are extracted; do not retain every rollout graph.
4. Compare ordinary Burn primitives with at least one batched/fused feature-assembly design on identical inputs. A CubeCL kernel must show useful integrated savings after host synchronization, not just a faster isolated reduction. Do not import predecessor CSR/backend patches merely because they exist; its OOM came from huge relation axes absent here.
5. Measure B1 and B8, T32 pilot then T512, L8 and L16, selected feedback/output modes, with identical parameters/masks/uniforms. Separate cold compile/autotune, synchronization, transfer, host enumeration, prefix-cache preparation, history appends, forward, backward and optimizer. Record launch counts, scalar-buffer sizes, retained state and sampled memory with its limitations. Use synchronized warm interleaved pairs and a predeclared interval method; main integration owns verification, not parallel workers.
6. Set component budgets from measured integrated cost before longer contexts. A provisional target is energy feature/sampler overhead below 5% of matched denoising wall time, **not a correctness exemption or demonstrated performance result**. If the target is missed, preserve the law/precision while profiling head projection, entropy, round trips and L16 enumeration; adjust batching or scratch reuse before changing the model. Reject larger-L enumeration before allocation rather than silently switching to an approximate law.

**Completion:** all affected callers, numerical gates, dataset identities, checkpoint/chain continuation, execution selectors and implementation evidence are updated; mask-set selection remains stochastic with exact legal support; final commitments remain immutable; TCT is distinct from energy fitting. This turn authorizes the selected implementation, correctness checks, execution measurements and tiny smokes only.

## 13. Command interface

The optional `sbd.refinement` object selects `policy: uniform|unary|pairwise` and separate energy/data paths. Absence keeps the existing decoder, serialization and initialization. `sbd.training_mode: tct` selects streaming joint correction training; omitted or explicit `sbd` keeps the old training identity. `configs/lm512-remask.json` selects pairwise guided decoding and `configs/lm512-tct.json` selects actual TCT training.

`tct.rows_per_pass` optionally partitions a TCT optimizer batch by rows. Each pass uses its own parameter-fresh clean cache and immediately releases its graph after backward. Its neural loss is weighted by pass_rows/full_batch_rows, and unscaled positive/negative circuit flows and occurrence reaches accumulate in fixed row order. Exactly one neural optimizer and one circuit update occur after the full batch. Checkpoints include the chosen chunk size in training identity; omitted tct settings preserve prior serialization. The selected B8 config uses four rows/pass after the largest requested candidate fit; rows2/1 were not tried.

TCT FFBS uses indexed uniforms keyed by step seed, original batch-row position, block start and token position. One shared seed is consumed for the full training batch, not one per chunk, so full/chunked layouts retain identical proposal draws and loader/RNG continuation. The persistent TCT chain schema is2; no autograd/cache tensors enter it.

| Command | Work |
|---|---|
| `remask-reference CONFIG` | Independent host-f64 exact-law/energy/rank checks plus runtime state checks on CPU and requested device |
| `tct-reference CONFIG` | Joint correction loss/flows, transitions and persistence checks |
| `remask-generate CONFIG --resume CHECKPOINT` | Validated frozen-model R²D generation with selected policy; guided policies require a genuinely fitted separate energy |
| `remask-collect CONFIG --resume CHECKPOINT` | Training-only paired uniform-continuation refinement collection; saves frozen feature rows and outcome-ranked mask pairs |
| `energy-fit CONFIG` | Fresh offline fit of only the separate energy; pairwise fits unary first then enables zero-initialized pairs, fit_steps per part |
| `tct-train CONFIG [--resume CHECKPOINT]` | Streaming all-free-position joint correction training plus clean causal loss; exact TCT restart retains chains |
| `remask-benchmark CONFIG` | Synchronized execution-cost fixture; scratch model/energy are explicitly identified, not presented as trained generation |
| `tct-benchmark CONFIG` | B1/T512 per-block versus batched identical-objective execution pairs; B8/T512 one warmed diagnostic |

Training collection uses states512/pairs_per_state4/horizon2 by default. Explicit smaller counts in context32 smoke configurations test command plumbing, not quality. Fit defaults are100 optimizer steps per part and512 pairs/batch. Fit metadata records data/source hashes, feature version, exact sampling/precision identity and model/optimizer file hashes. A guided decoder rejects another frozen source model or feature definition even if shapes match. No continuation flag is silently ignored by energy-fit.

The R²D source loader first validates the existing exact training identity with refinement removed. When only output paths differ, a recorded source configuration from the source run's metrics is accepted only after its complete fingerprint reproduces state.json; all requested model/training settings must still match. This is not shape-only checkpoint loading. Exact TCT restart also restores candidate rows, reveals, target chains, counters and data/RNG positions, with fresh clean caches under current parameters.

TCT-trained refinement explicitly uses correction proposals under permanent-O-only evidence, so all free positions can be revised; ordinary-SBD refinement retains provisional carry-over. This behavior follows the explicit training_mode selector, not a heuristic checkpoint guess. Neither mode revises permanent O.


## 14. Implemented and observed

### 14.1 Implementation and numerical evidence

Implementation is in `crates/experiment/src/{remask,mask_energy,tct}.rs`, with small configuration/CLI/generation/training hooks. Configs are `configs/lm512-remask.json` and `configs/lm512-tct.json`. Energy source hashes and predecessor ranges are recorded in `reuse/PROVENANCE.json`. The intrinsic neural model gains no extra parameters when selectors are absent; energy weights remain a separate trained file.

All measurements below use the existing **`joint_output:codpula` profile**, Intel Arc Graphics(MTL), Mesa26.2.4, Vulkan f32, no Burn fusion or autotuning, D256/four layers and the selected Step7/HOLA components. They are not measurements of the separately implemented contextual profile. Source identity checks prevent an energy fitted on one frozen profile/checkpoint from silently serving another.

| Gate | Observed result |
|---|---|
| Standalone workspace suite | Historical62experiment+7kernel+16mixer passes, one ignored; `artifacts/remasking/summary.json`. Integrated11 later recorded73+7+16passes, one ignored, in `artifacts/integrated11/commands/conditioned-build-tests-1792953155667.stdout`; not post-fix acceptance |
| Mask-set laws | 990 exact-law cases, all n0..16/r, uniform/unary/pairwise; largest logZ error1.13687e-13, probability error3.33067e-16 |
| Fixed-seed sampling | 65,536 draws/case; largest frequency error0.00291679 versus simultaneous Hoeffding limit0.01355672; no numerical tolerance substituted for sampling uncertainty |
| Energy/features/rank CPU/Vulkan | Max absolute error3.305467803915718e-6, max elementwise tolerance ratio0.006872483698735233 |
| Joint TCT including accumulation | Max absolute error9.5367431640625e-7, max ratio0.003378682889902597 |
| Full batch versus row chunks | Tensor max2.384185791015625e-7; fused max9.5367431640625e-7; neural gradients, all dual-flow fields and active counts agree, token decisions/RNG agree |
| TCT persistence | CPU and fused-Vulkan resume through partial reveals, fully revealed endpoint and refill with bit-identical chain/RNG, neural/circuit rows and optimizer state |
| Existing model/checkpoint gates | Existing SBD CPU/Vulkan matrix passes; full Codpula gate max2.384185791015625e-6/ratio0.0474304075577138; exact historical causal checkpoint13 check passes |

The elementwise criterion remains `2e-5+3e-4*abs(reference)`. No tolerance was relaxed. References cover padding/dirty NaNs, transposed pair-projection operands, fixed O/prefix, L1, observed-potential derivatives, all circuit flows and selected energy/rank VJPs. Raw reports: `artifacts/remasking/{remask-reference.json,cpu/remask-reference.json,tct/tct-reference.json}`, `smoke-source/{sbd-reference.json,codpula-model-reference.json,update-loss-reference.json}` and `historical/checkpoint-check.json`.

A separate pre-existing native AdamW clipping repair is parent commit **64590d18**, documented in [Codpula §10.4](codpula.md#104-deterministic-native-optimizer-clipping-2026-10-25). The failure reproduced on the unmodified Codpula base. Raw gradients matched; an unordered global norm reduction changed clipped updates. Fixed-output norm reduction preserves native records/formula. The adopted fused optimizer kernel is unchanged and its f64 update/loss gate passes(max1.3050829517080587e-7). Tensor/fused dual-flow accumulators have fixed output ownership; unrelated finite-only/readback sums were not changed merely for containing `sum()`.

### 14.2 Actual command smokes

Every new user command completed on **context32**, with full D256/four-layer architecture:

1. Uniform R²D generation finalized/replayed8 tokens from a genuine tiny source checkpoint, using38 noisy calls.
2. Trajectory collection ran the actual default512states/fourpairs/H2 recipe: 2,048 pairs, 8,192 equal-work branch call-units, 4,096 endpoint call-units, 8,222 source call-units, 20,510 total noisy row-call units and133 batched dispatches. Its two non-tied pairs supply genuine outcome supervision; the remaining numerical ties have zero weight.
3. Offline fit used the collected frozen524-entry features, two unary updates then two pairwise updates, with2048pairs/update so the tiny number of eligible pairs was actually visited. This is a command smoke, not a fit-quality conclusion.
4. Guided generation loaded that genuinely fitted energy, validated file/source/feature identities and finalized/replayed8 tokens in38calls. There is no missing-energy or no-label fallback.
5. TCT warm-started from the ordinary source checkpoint and trained5updates/160 target visits, including the fully revealed endpoint and refill;64unique bytes were loaded. The final chain schema2 and explicit row-pass setting are exercised.

The initial8-state smoke collection produced only numerical ties and correctly could not supply rank training. Expanding to the **selected default collection recipe**, rather than relaxing tie tolerance or inventing labels, supplied two eligible pairs. No quality study, sweep, calibration or policy comparison followed.

Reports/configs/datasets: `artifacts/remasking/smoke-{base,uniform,collect-full,tct}.json`, `smoke-trajectories-full.json`, `smoke-energy.fit.json`, `smoke-base/{remask-collect,remask-generate}.json` and `smoke-tct-final/tct-kronlstm-hola-seed1/metrics.json`. After all gates,29owned model/optimizer/energy payloads were removed; JSON/data/hash evidence remains. Original historical payloads and unrelated scratch directories were untouched; see `checkpoint-cleanup.json`.

### 14.3 Generation execution cost

This is an actual batched active-block workload, not B8/T512 teacher-forced training: one8-token active block per row with immutable512-token prefix. The benchmark uses explicitly identified scratch model/energy parameters, not trained output-quality measurements. Initial prefix preparation is outside the repeated episode. **The timed episode includes state initialization, adaptive denoising, energy features/sampling/readbacks and one clean replay of the completed520-position prefix.**

| Work | B1(prefix512/L8), 2warmups/11samples | B8 same geometry, 1warmup/1diagnostic |
|---|---:|---:|
| Noisy batch evaluations | 38 | 38(304 row evaluations) |
| Refinement only | 2641.888ms median | 2912.708ms |
| Refinement per noisy batch pass | 69.523ms | 76.650ms |
| Clean replay component | 98.867ms median | 430.226ms |
| Complete resident-prefix episode | **2740.755ms median** | **3342.933ms** |
| Retained feedback/history buffers only | 8,928bytes estimated | 71,424bytes estimated |

Initial513-position prefix-cache preparation measured3.146883s/B1 and1.030727s/B8 in their respective fresh processes; these one-off samples can include JIT/setup and are not a paired prefill comparison. Logical prefix length is512; the extra position is the snapshot API dummy and never evidence. These samples must not be silently included as warmed per-pass latency or excluded when estimating first-output latency. Exact live peak memory was not measured; the small retained-feedback estimates exclude model, clean cache and circuit temporaries.

Raw samples and work boundaries: `artifacts/remasking/remask-timing-b1/remask-benchmark.json` and `artifacts/remasking/remask-benchmark.json`. These historical measurements include the then-required final replay. The current `remask-benchmark` method version 2 instead times the resident-prefix episode and completed token-history append, without a discarded final clean forward. Initial cache preparation remains outside the repeated timed episode and is reported separately as `prefill_seconds`; the actual fixture remains prefix512/L8. `history_append_seconds` replaces `clean_replay_seconds`, and current CLI generation reports `clean_history_appended_tokens` rather than claiming the final tokens were processed by recurrence. No new latency result or speedup is claimed here.

### 14.4 TCT execution and memory

B1/T512/L8 complete-step timing used2warmups and11same-session interleaved pairs, alternating order, with identical initial model/chain/data/RNG settings and independent optimizer trajectories. Both paths perform the same TCT objective, not different training arms. The measurement includes clean/cache construction, joint NLL/dual flows, FFBS/feedback, backward, finite checks, neural/circuit updates and refill; it excludes evaluation/checkpoint I/O.

| Complete TCT step | Time |
|---|---:|
| Per-block cache execution oracle | 17562.660ms median |
| Batched block-axis execution, selected | **1845.395ms median** |
| Paired saving | **15711.721ms**, 95% bootstrap interval[15625.973,16349.399],11/11 faster |
| B8/T512,4rows/pass, one warmed diagnostic | **7267.660ms** |

The B1 two-runner process sampled1,452,711,808bytes pool reservation, not a per-runner peak. Full B8 without row accumulation failed a2GiB allocation with either tensor or fused circuit execution. Circuit fusion alone therefore was not adopted as a memory fix. Exact row accumulation was selected; **four rows/pass fit immediately**, producing exactly two passes and one neural optimizer/signed dual-flow circuit update per full B8 batch, not EM (`codpula.md:211–222`). Smaller chunks were not tried. Sampled pool reservation was **4,399,520,256bytes**, not an exact physical/live peak.

The selected circuit execution remains tensor when omitted. `configs/lm512-tct.json` records rows_per_pass4. The B8 step is not a paired quality or speed comparison against the supplied~8s ordinary SBD+Codpula step: the objectives, target visits and detached-teacher work differ.

Evidence: `artifacts/remasking/timing-b1/tct-benchmark.json`, `tct/tct-benchmark.json`, `tct/tct-reference.json`. The GPU/build locks were released between commands. Only the shared target directory and copied workspace executable were used. The final helper build cleaned/recompiled the three local crates; published build times are not warm per-command inference overhead. No target/profile/dependency changes were made.


## 15. Prefix219 integration-QA investigation

The unnormalized diagnostic `sum(A.logits²)+sum(B.logits²)+sum(A.features²)+sum(B.features²)` is **not** the training objective. Its original plain-SBD disagreement ratio21.071893 and CoCommit ratio11.614331 remain failures, with original bytes retained under `artifacts/cocommit/` and hashed copies in `artifacts/remasking/integration-qa/originals/`. Neither check was rerun unchanged, relabelled as passing, or assigned a looser tolerance.

**Conclusion:** no incorrect retrieval operator/VJP was found. The evidence identifies cancellation and rounding amplification in this artificial f32 objective, rather than a changed cache, a changed MoBA selection, or a CoCommit-only error. Production code and accepted training gates are unchanged. This is not a claim that a complete scalar-f64 model was evaluated: independent f64 references below cover the relevant RMS, vocabulary-linear and fixed-source retrieval primitives; full-network CPUf32/Vulkanf32 comparisons are disagreement diagnostics only.

### 15.1 Separate the two loss terms

The same seed1/D256/four-layer initialization and prefix219/L8 geometry were used, with CPU tensor operators and the actual configured Vulkan fused operators. A/B cached reads were identical within each plain-SBD execution.

| Diagnostic | Largest CPUf32/Vulkanf32 tolerance ratio | Interpretation |
|---|---:|---|
| Logits squared only | 25.226726 | The failure is not solely final-feature RMS cancellation |
| Final normalized features squared only | 17.556043 | Nearly radial RMS cotangent |
| Replay the stable scalar-f64 feature-squared cotangent at the raw final-RMS input | 0.000142909 | Removing the radial subtraction removes this component's amplification |
| Replay scalar-f64 RMS+head logits-squared cotangent at the same input | 21.478384 | Earlier-network rounding/cancellation remains; final RMS alone is not an explanation for logits |
| Replay one common deterministic non-radial raw-input cotangent | 5.905627 | Full-network f32 Jacobians need not agree elementwise under this artificial loss |

For gamma1 and row mean square `s`, the exact feature-squared derivative is `2*epsilon*x/(s+epsilon)^2`. The independent scalar-f64 direct and projection terms have combined L2 magnitude181.043326, but their residual is only0.000090545887: worst coordinate cancellation is1,999,768×, about6.30decimal digits. CPU/Vulkan captured raw inputs differ by only3.070443e-6 relative L2. The local RMS VJPs, evaluated at each exact captured f32 input and its matched cotangent, nevertheless pass the original `2e-5+3e-4*abs(ref)` rule. Across all CPU tensor/Vulkan tensor/Vulkan fused probes, largest primitive error is1.097112e-5 and largest ratio0.088861, with zero failing elements. Relative error of a nearly zero radial residual is not substituted for the specified tolerance.

### 15.2 Use the exact detached production source lists

Both calls' actual original-position IDs and disabled slots were captured **after production selection**: tensor visibility/cache IDs and fused selector output IDs. No new ranking convention was written for the arithmetic reference. All32 query/head rows have identical enabled source sets across implementations, no duplicate IDs, and164–175 enabled sources including active K/V. Cache positions are identical. The earlier scalar-f64 selection-margin calculation gives a minimum boundary gap0.00240659, over25times a conservative f32 dot/summary roundoff estimate; it is supporting routing evidence, not a second VJP oracle.

On these fixed lists, a separate sequential scalar-f64 implementation computes the single softmax with its sink, weighted value read and analytic q/k/v/sink/active-K/active-V derivatives. The captured query-valid output gate is included; it is a discrete mask, not an invented trainable gate.

| Primitive checked at captured inputs | Largest absolute error | Largest tolerance ratio |
|---|---:|---:|
| CPU tensor retrieval, all outputs and six VJPs | 2.664782e-8 | 0.000924309 |
| Actual Vulkan fused retrieval, both captured input sets | 2.664782e-8 | 0.000920052 |
| CPU/Vulkan native vocabulary linear, output/input/weight VJPs | 2.205272e-6 | 0.015199263 |

Every element passes unchanged tolerance against f64. A repeated fused retrieval backward is also bit-identical for all six VJPs; this repeat is only determinism evidence, **not** the independent numerical reference. The inspected fused query/token/active/sink derivatives have fixed output ownership rather than float `fetch_add` accumulation.

### 15.3 Quantify the remaining logits cancellation

Eight separate final-RMS-input row cotangents were computed from the scalar-f64 RMS+linear logits-squared formula, rounded once for the existing f32 graph, and differentiated individually. Their parameter-coordinate contributions were then summed in host f64. This tests linearity and exposes conditioning; it does not turn f32 row gradients into an independent oracle.

| Parameter coordinate | Sum of absolute CPU row contributions | Absolute CPU f64 row sum | Cancellation factor | Largest individual-row CPU/Vulkan ratio | Difference between f64 sums |
|---|---:|---:|---:|---:|---:|
| `ohc_init.offsets[855]` | 178.828747 | 0.044420123 | 4025.85 | 0.649131 | 0.000933528 |
| `blocks.0.ohc_lngram.mix_bias[6]` | 116.149486 | 0.062150002 | 1868.86 | 0.070737 | 0.001029491 |

Each individual row passes the original comparison rule at these coordinates, yet their small residual sums fail it. The disagreements are only5.22e-6 and8.86e-6 of the respective sums of absolute contributions. Ordered host-f64 summation does not remove them: they arise before the final row sum in the differently rounded f32 graphs. This directly explains why fixing only a final host-scalar reduction or only the final RMS would not repair this stress comparison.

The actual normalized CoCommit training objective remains the accepted gate: existing `artifacts/cocommit/cocommit-reference.json`, prefix219/L8, all7,083,447 neural gradient elements, max absolute1.573563e-5 and ratio0.538998135, passes unchanged tolerance. That acceptance was read, not rerun or replaced with a rescaled squared diagnostic. No mechanisms were disabled.

Actual GPU Lngram execution was `fused_gpu` in all three active Lngram modules. The separate Tensor-Lngram duplicate-index fix `e50cc523` was not merged and does not explain this execution. The clipping fix `64590d18` and feature commit `b9e589b5` remain separate and unchanged.

Evidence is under `artifacts/remasking/integration-qa/`: `summary.json`, `retrieval-stress-decomposition.json`, `rms-captured-inputs.json`, `row-vjp-contributions.json`, `row-vjp-f64-sums.json`, `fixed-selected-list-equality.json`, `fixed-list-prefix{,-cpu}-f64.json`, `selection-f64.json`, captured arguments/IDs in `raw/`, and hashed original failures in `originals/`. Throwaway probes and capture hooks were removed. No source regression test was added because no faulty operator was identified; existing independent primitive references and actual-objective gates remain required. No quality study, selector sweep, attention redesign or execution-layout change was performed.

