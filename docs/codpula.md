# Step 9: Codpula joint block denoising

## Status and reading guide

This document records the paper equations, the model-specific decisions in §5, and the implementation and measurement evidence below. The earlier specification-only pass made no code changes; the implementation pass uses the explicitly recorded decisions rather than silently choosing alternatives.

**Implemented 2026-10-25.** The fixed model, full-vocabulary inference, dual-flow updates, seeded joint proposals and dependence commitments are integrated. CPU and Vulkan numerical/model gates pass at the unchanged tolerance. The tensor circuit remains the selected execution: fused kernels are correct and faster in isolation, but their complete-step confidence interval includes zero (§10).

Source: Shao et al., [Learning to Couple: Context-Adaptive Joint Denoising for Diffusion Language Models, arXiv:2610.25723v1](https://arxiv.org/html/2610.25723v1), including Appendices A–G. The [abstract/version page](https://arxiv.org/abs/2610.25723) identifies the version. References such as Eq. (10) refer to that paper, not numbered equations in this document.

Repository context: [PLAN.md §5.1 and Step 9](../PLAN.md), [SBD](sbd.md), and `configs/lm512-sbd.json`. Step 9 adds a joint output distribution and dependence-adaptive commitment; it does not change KronLSTM, HOLA, oHC, wavelet, Lngram, or MoBA prefix retrieval. The optional Step-10 [CoCommit](cocommit.md) now composes with this output: each marked tail regenerates token potentials and gates, prepares a fresh circuit, and recomputes joint proposals/dependence scores. Its marked training event fixes only selected targets, marginalizing other masks; the existing full conditional NLL is that generalized event's all-unresolved special case. Remasking remains separate work.

Fixed model requirements:

- Each actual SBD block is one independent HMM chain, with actual length `1 <= L <= 16` and **32 latent states**. The paper uses 1,024 states; the plan's possible 64-state comparison is not the implementation size specified here.
- Neural features are the **final RMSNorm output at the noisy block positions**, after the complete backbone and oHC readout when present.
- Emissions cover the **entire existing output vocabulary, 257 entries**, not only the four feedback candidates. IDs 0–255 are bytes; ID 256 is BOS and remains in the model support. There is no vocabulary MASK or PAD token: the SBD mask is a separate learned vector and padding is a validity mask.
- The existing clean causal SBD loss remains, plus conditional joint noisy-block NLLs summed and divided by the existing clean target count.
- Exact inference means exact sums within this HMM, up to floating-point error. It does not mean recovery of the true text distribution or exact joint-output MAP decoding.

## 1. Paper model and normalizers

### 1.1 Circuit and HMM structure

A probabilistic circuit has categorical input nodes, product nodes, and sum nodes. A node computes a univariate distribution, a product of its children, or `sum_c omega[n,c] p_c`, respectively (paper §3, Eq. (6)). Sum weights are nonnegative and sum to one within each row. Smoothness means all children of a sum cover the same variables; decomposability means children of a product cover disjoint variables (Appendix F.1, Definitions F.1–F.2). These properties make evidence queries exact with one bottom-up pass, linear in circuit edges after input evaluation.

Appendix A.1 instantiates a homogeneous HMM compiled into such a circuit: transitions are shared across positions, with categorical leaves over the full vocabulary. Its state count is 1,024 and chain lengths are 32, or 16 for SDAR. The paper does not provide a numbered HMM factorization; the following is the standard chain expansion of that stated architecture, with notation for our shorter blocks:

\[
q(x_{1:L},z_{1:L};\theta,\omega)
 = r_\theta(z_1)\,g_{1,z_1}(x_1)
   \prod_{i=2}^{L} A_{\theta,i}(z_{i-1},z_i)g_{i,z_i}(x_i),
\qquad
q(x)=\sum_{z_{1:L}}q(x,z).
\]

Here `r` is an initial-state categorical row, `A` is a row-stochastic transition matrix, and each `g[i,s]` is a categorical emission. The base transition and emission tables are shared across positions and lengths. One context gate grid is shared across all edges of each block, so its modulated transition is also identical across those edges; the learned initial row has no neural modulation. These are the model-specific decisions in §5, rather than details uniquely specified by the paper's grouping equation. The circuit can be written recursively as a sum over the next state followed by the product of the current emission and remaining suffix distribution, directly supplying the smooth/decomposable structure.

The model parameters are split into neural parameters `varphi` and intrinsic probabilistic parameters `omega`. Neural outputs `theta = f_varphi(x_t)` contain token potentials and coupling gates (paper §3, Eq. (4)). Within a denoising pass, all these outputs are held fixed while candidate completions are summed, queried, or sampled. Re-evaluating the backbone for different queried assignments would define different distributions and invalidate the conditional calculations.

### 1.2 Token potentials and local emission normalization

For every position `i`, the backbone supplies one nonnegative potential vector `psi[i,v]`, shared across all latent states at that position. Each categorical circuit leaf has a base distribution `pi[i,s,v]` belonging to `omega`. The modulated leaf is exactly (paper §4.1, Eq. (10)):

\[
Z^{\rm leaf}_{i,s}=\sum_{v\in\mathcal V}\pi_{i,s,v}\psi_{i,v},
\qquad
g_{i,s}(v)=\frac{\pi_{i,s,v}\psi_{i,v}}{Z^{\rm leaf}_{i,s}}.
\]

Distinct latent-state leaves can have distinct base token preferences. Do not replace this by a product of the neural marginal and a state mixture followed by only one vocabulary normalization: that is a different model. Retain the **state-specific local denominator in both likelihood and derivatives**.

Given neural logits `a[i,v]`, full-vocabulary `psi = softmax(a)` is consistent with §4.1. The common softmax denominator cancels inside each leaf, so the numerically equivalent expression is

\[
\log g_{i,s}(v)
 =\log\pi_{i,s,v}+a_{i,v}
 -\operatorname{LSE}_{u\in\mathcal V}(\log\pi_{i,s,u}+a_{i,u}).
\]

This equality does not remove the last log-sum-exp or its derivative. The logits need not themselves remain calibrated standalone denoising probabilities after joint training (Appendix B.4).

**Full vocabulary versus truncation.** Section 4.1 describes a computational top-k approximation: retain top-k neural logits, softmax on those candidates, set the remaining potentials to zero, and retain the gold token during training. Appendix A.1/Table 5 uses k=1,024 except SDAR uses all evidence candidates. Appendix B.2 states full-vocabulary evidence is the decoding default and top-k decoding is an alternative; the categorical base leaves always cover the full vocabulary. Thus truncation is **not required by Codpula's mathematics**. Here use all **257** entries for every local emission normalization and query. The current feedback width four is unrelated to circuit support. Gold insertion and discarded circuit-support mass are unnecessary under full support; circuit-support discarded mass is exactly zero. Retained feedback mass remains a separate quantity.

### 1.3 Grouped coupling gates and row normalization

Rows of a base matrix `W` index sum nodes, columns index their shared child distributions. Partition a square matrix into `n x n` groups of `b x b` entries, with matrix width `n*b`. A neural head emits log-gates `h`; exponentiation gives positive `G = exp(h)`. The modulated matrix is (paper §4.2, Eq. (11)):

\[
W_\theta=\operatorname{RowNorm}
 \left(W\odot(G\otimes\mathbf1_{b\times b})\right).
\]

For a 32-state transition this means

\[
A_{\theta,i}(s,t)=
\frac{A(s,t)\exp h_{i,\lfloor s/b\rfloor,\lfloor t/b\rfloor}}
 {\sum_u A(s,u)\exp h_{i,\lfloor s/b\rfloor,\lfloor u/b\rfloor}},
\quad b\mid32,\quad n=32/b.
\]

The `i` index is omitted here because one gate matrix is shared within the block. Zero log-gates recover the base transition. One scalar for the entire matrix (`b=32`) cancels in row normalization and cannot change coupling. A source-row-only scalar likewise cancels: effective gates must distinguish destination groups within a row. The paper's 1,024-state “Both” configuration uses a `32 x 32` gate grid, hence `b=32`; copying that group width to our 32-state matrix would remove modulation. This implementation instead uses `b=4`, an `8 x 8` grid.

### 1.4 Visible-evidence normalizer

Let `O` be observed, valid positions inside the active block, `U` its unresolved positions, and `e` their observed values. The clean prefix and current soft inputs have already determined `theta`. Define

\[
Z_e=\sum_{x_U}q(x_U,e)
 =\sum_x q(x)\prod_{i\in O}\mathbf1\{x_i=e_i\},
\qquad
p(x_U\mid e,\theta,\omega)=q(x_U,e)/Z_e.
\]

This is paper Eq. (5) specialized using the visible-token indicators in Appendix F.1, Eq. (22). With normalized leaves and rows the unconditional sum is one, but **the visible-evidence mass is generally not one**. For an all-masked block `Z_e=1`; for an all-observed block the conditional completion probability is one and its joint denoising loss is zero.

At an observed leaf use `g[i,s](e_i)`; at an unresolved leaf use `sum_v g[i,s](v)=1` (Appendix F.1). Appendix D.1 also describes an unobserved input as contributing a local `log Z[n,i]`. That wording must not be mixed with already normalized Eq. (10) leaves: adding the raw leaf normalizer after normalizing the leaf counts it twice. The normalized-leaf implementation uses log one, **zero**, for unresolved positions. An equivalent raw-product implementation must retain the reciprocal local normalizers on the corresponding state paths; simply using `sum_v pi*psi` as an unresolved emission with no compensation is a different distribution. Eq. (10) and Appendix F.1 give the normalized representation used by the formulas here.

## 2. Exact inference contract

The following recurrences are algebraic consequences of the paper's HMM instantiation and Appendix F.1, not extra paper equations. Use log-space reductions, or independently checked scaled recurrences. Symbols below use probability space to make their meaning explicit.

For evidence assignment `e`, set

\[
d_i^e(s)=\begin{cases}g_{i,s}(e_i)&i\text{ fixed by }e,\\1&i\text{ unresolved}.\end{cases}
\]

Forward and backward messages are

\[
\alpha_1(s)=r_\theta(s)d_1^e(s),\quad
\alpha_i(t)=d_i^e(t)\sum_s\alpha_{i-1}(s)A_{\theta,i}(s,t),
\]
\[
\beta_L(s)=1,\quad
\beta_i(s)=\sum_t A_{\theta,i+1}(s,t)d_{i+1}^e(t)\beta_{i+1}(t),
\quad Z_e=\sum_s\alpha_L(s).
\]

Every chain starts at its own initial row. No transition joins separate blocks, rows, or documents. Useful posterior statistics are

\[
\gamma_i^e(s)=\alpha_i(s)\beta_i(s)/Z_e,
\qquad
\xi_i^e(s,t)=\alpha_{i-1}(s)A_{\theta,i}(s,t)d_i^e(t)\beta_i(t)/Z_e.
\]

For an unresolved position,

\[
p(X_i=v\mid e)=\sum_s\gamma_i^e(s)g_{i,s}(v).
\]

An observed position has a point mass at its observed token when the evidence has positive mass; its posterior latent distribution is still needed for messages and training. Additional conditional queries change only which leaf values are fixed; they do not regenerate token potentials, gates, or local normalizers. An impossible evidence assignment has no conditional distribution. Do not replace it by a uniform distribution, an arbitrary token, or a successful zero loss. Full positive support should prevent this on ordinary byte data; numerical underflow must not be confused with mathematical zero mass.

### 2.1 Joint proposals and decoding

Paper §5 samples proposals **jointly** after conditioning on visible evidence. Temperature `gamma > 0` acts on neural logits, not on learned base leaves or transition weights:

\[
g^{(\gamma)}_{i,s}(v)=
 \frac{\pi_{i,s,v}\psi_{i,v}^{1/\gamma}}
 {\sum_u\pi_{i,s,u}\psi_{i,u}^{1/\gamma}}.
\]

This is the unnumbered equation in §5. In log form replace `a` by `a/gamma` before local normalization. Raising the *final joint* or *final marginal* to `1/gamma` is not equivalent.

An exact HMM sampler can draw a latent path from `p(z|e)` using forward-filter/backward-sampling, then draw unresolved `x_i` independently from `g[i,z_i]`, while copying observed values. Equivalently, draw the initial state proportional to `r*d_1*beta_1` and each subsequent state proportional to `A(previous,s)*d_i(s)*beta_i(s)`, then draw emissions. These are exact samples from the defined conditional joint. Independent draws from token marginals are not joint samples.

Deterministic alternatives must have accurate names: sequential conditional argmax is a joint-aware greedy decoder, not exact output-sequence MAP. Viterbi over latent paths, even with maximized emissions, maximizes a latent-and-output assignment; it does not generally maximize token probability after summing latent paths. Codpula proposals here use seeded exact joint sampling; factorized generation keeps its historical greedy policy.

### 2.2 Dependence-adaptive commitment

Let `C` be the candidate positions under consideration and `hat x_C` their proposals. Other unresolved positions are marginalized. The paper uses (paper §5, Eq. (12))

\[
\delta_i=
\left|\log p(\hat x_i\mid\hat x_{C\setminus\{i\}},e)
             -\log p(\hat x_i\mid e)\right|.
\]

Smaller scores mean the proposal probability is less sensitive to the other proposals. This is an absolute pointwise log-probability change, not entropy, confidence, a signed score, or an expectation of mutual information.

Define `ell(S) = log q(e, hat x_S)`, summing every other output and all latent states. Then (Appendix D.1, Eqs. (14)–(16))

\[
\log p(\hat x_S\mid e)=\ell(S)-\ell(\varnothing),
\quad
\log p(\hat x_i\mid\hat x_{C\setminus i},e)
 =\ell(C)-\ell(C\setminus i),
\]
\[
\delta_i=|\ell(C)-\ell(C\setminus i)-\ell(\{i\})+\ell(\varnothing)|.
\]

For `m=|C|`, the general circuit recipe uses one complete-candidate query, `m` singleton queries and `m` leave-one-out queries, plus the shared evidence normalizer: at most `2m+2` evaluations. Batch the evidence assignments and reuse neural outputs and input normalizers. A cached value is reusable only with identical circuit parameters, neural potentials, temperatures, support, and visible assignments. `m=1` gives exactly zero without extra queries.

An HMM-specific exact optimization is available: one forward-backward pass under `e` gives all singleton proposal probabilities; another under `e + hat x_C` gives the complete-candidate mass and all leave-one-out masses by replacing position `i`'s fixed emission with one in its **incoming-times-backward cavity**. Formally, with incoming `a_i(s)` excluding `d_i`, `q(e+hat x_{C\setminus i}) = sum_s a_i(s) beta_i(s)` for `i in C`. Do not divide by the selected emission when it can be zero. This computes the same scores, with two forward-backward traversals rather than `2m+2` independent forward queries; compare both against enumeration.

Commit exactly the candidates with `delta_i < tau`, **strictly less**, for `tau>0`; defer the others. If none qualify, commit the candidate with minimum `delta_i` (paper §5 and Appendix D Algorithm 3). If there is one candidate, commit it. Algorithm 3 also has a finite pass cap `M`: on its last pass it commits **all remaining unresolved positions**, irrespective of the threshold. Candidate ranking, width, pass cap, threshold, and tie behavior require the decisions below. The score adds circuit work but no additional backbone call inside that pass. A new backbone evaluation is required on the next denoising pass after the context/soft state changes.

## 3. Objective, flows, and updates

### 3.1 Conditional joint denoising objective

Paper §3 Eq. (7), and Appendix E Eq. (21), maximize

\[
\mathcal L(\omega,\varphi)=
\mathbb E_{t,x_0,x_t}[w(t)\log p_{\varphi,\omega}(x_0\mid x_t)].
\]

For one block let `e+` fix every valid position to its clean token and `e-` fix only observed valid positions. Then its log likelihood is

\[
\log p(x_{0,U}\mid e)=\log q(e^+)-\log q(e^-).
\]

Do not substitute a sum of log token marginals: that loses the conditional joint objective. Observed tokens appear in **both** evidence evaluations, not as separate prediction targets. Their effect on posterior states and on the visible-evidence denominator must remain.

Appendix E assumes independent absorbing masks, with masking probability `alpha_t`, sampling density `rho(t)`, and `w(t)=dot alpha_t/(alpha_t rho(t))`; uniform `t` and `alpha_t=t` give `w=1/t`. It derives a lower bound related to the marginal diffusion ELBO (Eqs. (17)–(21)), not equality of joint NLL and a sum of marginal NLLs. Table 5 records inverse-mask weighting with `10^-2` in most configurations, and none for SDAR; the text does not fully specify the corresponding floor formula. Do not infer a unique epsilon/clipping rule from that table entry.

### 3.2 Dual-flow multiplicative update: exact paper formula

For directly parameterized circuit weights, with neural outputs fixed, the paper separates nonnegative positive and negative derivatives. In the simplest unnormalized-circuit notation these are `F+ = d log tilde p(e+)/d log omega` and `F- = d log Z/d log omega` (paper §3, discussion before Eq. (9)). Normalizer derivatives are not optional. Average the two statistics separately with the same nonnegative example weights used by the objective (Algorithm 2).

For each active positive row, the exact Eq. (9) update is

\[
\eta_{n,c}=\alpha\,T_{n,c}/\omega_{n,c},\qquad
u_{n,c}=\omega_{n,c}
 \frac{1+\eta_{n,c}F_B^+(n,c)}{1+\eta_{n,c}F_B^-(n,c)},\qquad
\omega^{\rm new}_{n,c}=\frac{u_{n,c}}{\sum_d u_{n,d}}.
\]

Here `alpha>=0` is the base step size, `0<T<infinity`, and all quantities on the right use the pre-update parameters. Strictly positive weights remain positive for finite arithmetic-exact steps and each row sums to one; the derivative difference may be negative. The update agrees to first order with the row-normalized signed additive update, but not generally at finite step size (Proposition 3.1, Appendix G Eqs. (28)–(33)). It is not exponentiated-gradient Adam, signed EM counts clipped to zero, or an SGD step followed by repair.

Appendix F.2 gives the step multipliers: `T=1` yields the full-batch EM limit as `alpha -> infinity`; `T=1/C_B(n)` gives minibatch interpolation `lambda=alpha/(1+alpha)`; **Anemone uses `T=1/rho_n`**, where `rho_n` is the generative traversal probability of reaching the node (Eqs. (23)–(27)). Unreachable rows and rows with no update statistics remain unchanged.

Algorithm 1 uses different names for an equivalent Anemone form: it sets `T_alg(n,c)=TD_omega(n)*omega[n,c]`, then

\[
R_{n,c}=\frac{TD_\omega(n)\omega_{n,c}+\eta_{\rm alg} F^+_{n,c}}
                  {TD_\omega(n)\omega_{n,c}+\eta_{\rm alg} F^-_{n,c}},
\quad \omega^{\rm new}=\operatorname{RowNorm}(\omega\odot R).
\]

For `eta_alg=alpha` this equals Eq. (9) with `T=1/TD_omega(n)`. **Do not substitute `T_alg` directly as Eq. (9)'s `T`: they are different quantities.** Shared rows here sum occurrence reaches in the current context-modulated chain and average them over the same active blocks as the flows.

Appendix A.1/Table 5 reports Anemone, momentum 0.9, evidence temperature 1.0, and circuit step-size examples: SDAR 0.05; Dream 0.2 decreasing to 0.1; LLaDA 0.3 decreasing to 0.03. Schedules differ by backbone/task. Here alpha is the SDAR value 0.05, constant, with separate bias-corrected positive/negative flow averages at momentum 0.9. This explicit momentum recurrence is a repository decision because the paper does not print one.

### 3.3 Local-normalizer and gate derivatives must also enter the flows

Section 4.1 explicitly states that numerator and local emission denominator derivatives contribute to positive and negative flows when learning `pi`. Merely taking positive/negative parts of the final signed gradient is not the same algorithm: different nonnegative decompositions with the same difference give different finite Eq. (9) updates.

The following explicit nonnegative split is an **approved algebraic adaptation**, not an additional formula printed in the paper. It makes the local-normalizer derivatives implementable and independently checkable.

For either evidence query `a in {+,-}`, define emission expected counts

\[
N_i^a(s,v)=\begin{cases}
 \gamma_i^a(s)\mathbf1\{v=e_i^a\}&i\text{ observed in query }a,\\
 \gamma_i^a(s)g_{i,s}(v)&i\text{ unresolved in query }a.
\end{cases}
\]

A signed derivative of the conditional log likelihood with respect to a base emission log weight is

\[
\partial_{\log\pi_{i,s,v}}\log p
 =N_i^+(s,v)-N_i^-(s,v)
  -[\gamma_i^+(s)-\gamma_i^-(s)]g_{i,s}(v).
\]

One direct numerator/denominator split is

\[
F_{\pi,i,s,v}^+=N_i^+(s,v)+\gamma_i^-(s)g_{i,s}(v),\quad
F_{\pi,i,s,v}^-=N_i^-(s,v)+\gamma_i^+(s)g_{i,s}(v).
\]

For a base transition `A(s,t)` at one edge, let `R_i^a(s)=sum_t xi_i^a(s,t)`. The analogous split through Eq. (11)'s row denominator is

\[
F_{A,i,s,t}^+=\xi_i^+(s,t)+R_i^-(s)A_{\theta,i}(s,t),\quad
F_{A,i,s,t}^-=\xi_i^-(s,t)+R_i^+(s)A_{\theta,i}(s,t).
\]

Their difference is the exact derivative with respect to `log A(s,t)` holding gates fixed. If the initial row is explicitly renormalized/modulated, the same row formula applies with initial-state counts. If it is treated as raw independent circuit edges, use their raw edge derivatives instead; row projection can make signed gradients agree while the finite dual-flow steps differ. **The raw-edge convention in Appendix F.2 is not automatically the derivative through a softmax initial row.**

Sum the nonnegative statistics over tied occurrences before averaging over active blocks and updating each shared row. Do not perform an optimizer update per occurrence on one shared tensor. The flow decomposition, traversal scales and momentum are recorded in §5.

### 3.4 Neural optimization

Algorithm 2 obtains two evidence-query derivatives with respect to `theta`, subtracts them, then backpropagates through `f_varphi`. For maximizing likelihood, `grad_varphi = J_f^T(g+ - g-)`; a normal loss-minimizing optimizer receives the negative. The gradients must include leaf log-sum-exp denominators, gate row denominators, and the visible-evidence denominator. Ordinary gradient optimizers apply to backbone, token-potential head, and gate head parameters; **circuit probability rows use the separate dual-flow update**, not the neural optimizer.

The paper optimizes LoRA adapters with AdamW, no auxiliary cross-entropy, and masks gradients on observed evidence (Appendix A.1); Dream and SDAR optimize circuit and adapters jointly, whereas LLaDA fits the circuit and adapters in separate phases with the other part fixed (§6, Table 5). Our small model already uses full-parameter neural training and a required clean causal loss, so adapter choice, schedule, gradient masking, and optimizer hyperparameters are explicit adaptations, not automatic paper defaults.

Both neural and circuit updates for a training step must use statistics from the same pre-update model. Detached teacher feedback supplies no student gradient through its proposal-generation graph. Do not update the circuit between the teacher and student evaluations of one training example.

## 4. Mapping onto this SBD implementation

### 4.1 Features, axes, and call paths

Current relevant sources:

| Source | Existing behavior and required integration |
| --- | --- |
| `crates/experiment/src/sbd_training.rs`, `cached_block_profiled` and `cached_blocks_profiled` | Build soft/observed noisy embeddings, run all layers with clean-prefix snapshots, perform oHC readout when enabled, then final RMSNorm and `head_forward`. Expose/reuse that **same normalized tensor**, `[C,Lmax,256]` for the configured batched path; do not add another norm or use a sublayer input. |
| `crates/experiment/src/sbd.rs:22-48` | Prefix-reconstruction `block_forward` and detached teacher feedback. The new output must also work here, not only in cached training. |
| `crates/experiment/src/sbd.rs:59-156` | Existing sampler, independent blocks, all-observed skipping, per-block/batched noisy loss and teacher calls. Replace the noisy distribution/loss readout, not the sampler or clean snapshots. |
| `crates/experiment/src/sbd.rs`, `evaluate` and generation routines | Currently consume factorized logits and token entropy. Joint likelihood, marginals, joint proposals, and commitment queries need separate readouts; raw token potentials must not masquerade as logits of the trained joint distribution. |
| `crates/experiment/src/model.rs:217-221` | Existing clean loss and vocabulary projection; the same projection supplies joint token potentials. |
| `crates/experiment/src/config.rs:630-635`, `data.rs:250-258` | `vocab()=257`; input starts at BOS 256, targets are bytes 0–255, and current LM targets all have weight one. |
| `crates/experiment/src/soft.rs:281-447` | `SoftState` retains sparse candidate probabilities, visibility, explicit observations, and active positions. Its update expects logits; feeding `log joint_marginal` is only equivalent to direct marginal feedback when its extra temperature is one. |

`C` is the number of active block rows, not sequence batch size; chain position is the next axis. A block starting at target index `s` conditions on BOS and targets strictly before `s`. Noisy position `j` keeps input-position identity `s+1+j` (SBD documentation). Preserve temporary one-step recurrent states, noisy HOLA in-block reads, wavelet/Lngram histories, and optional retrieval exactly. The new HMM sees resulting features; it does not persist its hidden states in the clean-prefix cache.

### 4.2 Soft top-k feedback

The feedback quantity for a joint-trained denoiser is its **conditional token marginals**, not independent softmaxes of its raw token potentials. The existing feedback mechanism remains sparse even though circuit inference uses full support.

For an unresolved active position, let `p_i(v)=p(X_i=v|observed block evidence)` from the teacher circuit. After any approved feedback-temperature operation, retain the top-k IDs and their **original probability weights**, not weights renormalized to sum to one. With retained mass `m_i=sum_{v in topk}p_i(v)` and visibility `r_i`, the existing embedding rule is

\[
E_i=r_i\sum_{v\in\operatorname{topk}}p_i(v)\operatorname{embed}(v)
       +(1-r_i m_i)\operatorname{mask\_embed}.
\]

For training, the first pass and its marginal probabilities are detached. Its update uses interpolation coefficient one, and increases unresolved visibility by `1/soft_steps`, capped at one. Discrete IDs remain discrete; the selected embedding table values in the student pass remain differentiable. The student backbone produces **new** potentials and gates; the teacher circuit is not reused as the student's output. Train on the student's conditional joint NLL. Hard-mask training skips the detached teacher exactly as now.

At generation, each pass similarly uses full joint marginals for the soft belief update; committed tokens become immutable observations, while deferred positions retain soft beliefs and do not become hard evidence. The categorical proposal selected for a commitment is separate from its marginal feedback distribution. Append/replay the completed block into clean recurrence only after all positions are committed, as now. A partially visible soft embedding does **not** imply a fractional observed token in `Z_e`: only explicit observations supply evidence indicators. Conditioning the circuit a second time on its own soft beliefs would define another model and is not implied by this mapping.

### 4.3 Clean loss, masked joint loss, and weights

Currently SBD training is

`clean causal mean NLL + sum(masked per-token noisy NLL weights)/(B*T)`

for the uniform-length byte workload. It does not divide the auxiliary term by the number of masked tokens (`sbd.rs`, training-loss functions; `docs/sbd.md`, objective section). The joint path keeps the clean computation and denominator, replaces each multi-token block's masked loss by `-[log q(e+) - log q(e-)]/(B*T)`, and applies no inverse-mask or length multiplier. This is not per-token marginal cross-entropy.

Valid joint targets must have unit weights; unsupported nonunit weights produce a clear error. Padding is structurally absent, not a zero-weight token in the joint event.

### 4.4 Observations, padding, and singleton behavior

- Every unmasked/committed valid position inside a multi-token block is evidence in numerator and denominator, including positions to the right of an unresolved position. Prefix tokens outside the block influence neural features but are not extra HMM positions.
- Right padding ends the actual chain. It is not an observed byte 0, a BOS, or a masked variable. There are no extra transitions, emission counts, gate pooling contributions, loss terms, or parameter-flow contributions after actual `L`. A padded batch row must not change results or gradients of real positions. Completely inactive rows produce no query or update.
- All-observed blocks have zero denoising objective and zero signed gradient. Skip their circuit learning contribution rather than accumulating numerical differences of identical passes.
- The `L=1` path remains **bit-for-bit clean next-token logits from the prefix**, with no current-target embedding, even for an observed singleton. A generic one-position HMM instead gives `p(v)=sum_s r(s)g[1,s](v)` and need not match clean logits, so the existing causal bypass is deliberately retained without a circuit contribution.

## 5. Decisions

**Decided 2026-10-25 in the user's absence; closest to the paper / cheapest; reversible.** These decisions replace the earlier open alternatives. All use K=32, L<=16 and all 257 output symbols.

1. **Token head:** reuse the existing clean vocabulary projection, with no new token head. Reason: matches the paper's backbone-logit potentials without another projection.

2. **Emission sharing:** one shared `[32,257]` base emission table across positions and lengths. Reason: a homogeneous HMM is the cheapest shared model.

3. **Initial state:** one learned positive 32-entry row, shared across lengths, with no neural modulation; differentiate its row normalization consistently with decision 12. Reason: retains a learned start distribution without another neural output.

4. **Gate head:** masked mean of all valid final-normalized block states, including observed positions and excluding padding, followed by one biased linear head with no nonlinearity; share its grid across all edges. Reason: the cheapest context-dependent block router.

5. **Gate grouping:** `b=4`, an `8x8` grid of 64 log-gates. Reason: meaningful grouped coupling with a small head at K=32.

6. **Initialization:** near-uniform positive base rows with small seeded random perturbations; gate-head weights and bias start at zero. Reason: breaks latent symmetry while initially retaining the base transitions.

7. **Noisy objective:** sum conditional joint NLLs over blocks and divide by the existing clean target count `B*T`; no inverse-mask weight or length multiplier. Reason: keeps the existing SBD reduction and follows the paper's SDAR choice.

8. **Target weights:** require unit weights on valid joint targets and reject others clearly; padding is structurally absent. Reason: avoids inventing a fractional joint-event objective.

9. **Singletons:** keep the exact existing causal L=1 bypass unchanged. Reason: preserves the established next-token identity.

10. **Feedback temperature:** apply the existing soft-temperature value to neural potentials before the teacher circuit; feed joint marginals into `SoftState` with its temperature set to one. Reason: follows paper §5 without a second sharpening operation.

11. **Neural optimization:** jointly train backbone, shared head and gate head with the unchanged existing AdamW settings; update circuit rows every step. Mask token-potential derivatives at observed positions as in Appendix A.1, while backbone/gate gradients through observed features flow normally. Reason: follows the paper's joint schedule and observed-potential masking without adapters or another optimizer.

12. **Circuit statistics:** use the explicit normalized nonnegative split in §3.3, sum tied-occurrence flows, and compute Anemone traversal scales from summed occurrence reach probabilities, averaged over the same active block contexts as the flows. The initial row uses the normalized-row split. Reason: gives exact local-normalizer derivatives and a defined shared-row update.

13. **Circuit optimizer:** separately average positive and negative flows over active blocks; maintain separate exponential averages with momentum 0.9 and standard bias correction; use constant alpha 0.05 and update every step. Reason: uses the paper's SDAR/block-diffusion step size with an explicit momentum recurrence.

14. **Proposals:** exact conditional joint sampling by seeded forward-filter/backward-sampling. Reason: follows paper §5 rather than substituting marginal greedy decisions.

15. **Scored positions:** score every remaining unresolved position. Reason: L<=16 makes extra candidate ranking unnecessary.

16. **Commitment:** use the same temperature and full support for proposals and scores, threshold `tau=0.05`, Algorithm 3 with `M=L` passes and last-pass full completion, and existing ascending-position then token-ID ties. Reason: reuses the prepared circuit and the paper's reported `0.05` deferral traces (Appendix C, Figures 12–13). Section 5 and Algorithm 3 specify only `tau>0`; the paper does not give a universal numeric default or identify one threshold for all main-table operating points. The cited traces, not evidence-temperature sweeps, are the numeric source.

17. **Probability arithmetic:** store normalized base rows in log form and compute Eq. (9) and row normalization in log space; fail clearly on invalid evidence. Reason: preserves positive mathematical support without an arbitrary probability floor.

## 6. Compute and memory estimate

These are **arithmetic/storage estimates**, not measurements or a forecast of the integrated runtime. The measured existing baseline is approximately **8.2 s at B8/T512** (one post-warm diagnostic, 27,145 launches and about 8.82 GB sampled pool reservation), and **1.617 s at B1/T512** (median of 11 paired trials), from `docs/sbd.md`, block-axis execution section. The B8 number is not an 11-trial median, and sampled pool reservation is not exact live memory. The paper's Blackwell/7B-model overhead percentages do not predict this small Vulkan recurrent model.

### 6.1 Workload and one-pass counts

For B8/T512/L8, `N=B*T=4096` valid block positions, `C=N/L=512` chains, K=32, and the requested estimate vocabulary V=256:

| Quantity | Per valid noisy-block position | All 512 chains |
| --- | ---: | ---: |
| Leaf product/normalizer terms, K*V | 8,192 | 33,554,432 |
| Full token-marginal readout terms, K*V | 8,192 | 33,554,432 |
| One forward transition reduction, `(L-1)K²/L` | 896 | 3,670,016 |
| Forward + backward transition reductions | 1,792 | 7,340,032 |
| Transition posterior/flow entries, one evidence assignment | 896 | 3,670,016 |
| One message array, K entries | 32 | 131,072 floats |

A “term” here is one reduction contribution, not one FLOP: a probability-space multiply-add is approximately two FLOPs, whereas log-sum-exp entails exponentiation, reductions and a logarithm. A naïve tensor graph may reread or rematerialize each term several times. The exact implementation must use **V=257**, multiplying vocabulary-related work/storage by `257/256=1.00390625`; state-transition work does not change.

The sampler skips all-observed blocks. With uniform sequence-level mask probability and L=8, the mean all-observed probability is `integral_0^1 (1-t)^8 dt=1/9`: expected active chains are about **455.1**, with about **3,640.9** valid positions processed per noisy pass. Expected genuinely masked targets are **2,048**. Circuit messages still include observed positions of active chains. Thus per *masked target*, use about 16,384 emission terms and 3,584 forward-backward transition terms at the all-chain planning maximum, or about 14,564 and 3,186 using expected active-chain work. Counts per valid noisy-block position must not be mislabeled per masked target.

### 6.2 Training-step work

With detached soft feedback, the teacher needs emission preparation, evidence forward-backward and full marginals. The student needs a fresh emission preparation and two evidence flows, for `e+` and `e-`, plus emission/gate derivatives and shared-parameter flow reduction. At the full 512-chain planning size:

- Two local-emission preparations: **67.11 million** vocabulary-state terms.
- Teacher marginal readout: **33.55 million** terms.
- Teacher plus two student evidence forward-backward traversals: **22.02 million** transition terms.
- Two student transition posterior preparations: **7.34 million** edge-state terms, before signed/nonnegative split reduction.
- Emission gradients and positive/negative statistics require additional O(NKV) sweeps; roughly **2–6** vocabulary-state sweeps, or **67–201 million** terms, depending on fusion and whether full leaf tensors are retained.

Total circuit work is consequently on the order of **0.2–0.35 billion elementary reduction/product contributions per soft-feedback training step**, not counting backbone work, neural projections already present, transcendental cost, or memory traffic. This is a work estimate with explicitly variable fusion constants. Hard-mask training omits the teacher. Dependence commitment queries are **generation work**, not needed for the conditional-joint training objective, and must not be charged to every training step unless intentionally added.

The potential projection is already present if shared; a separate D=256, V=257 projection adds about 269.5 million multiply-accumulates per 4,096-position pass. A block-shared linear gate head adds `C*256*n²` multiply-accumulates: **134.2 million** for b=1, **8.39 million** for b=4, or **2.10 million** for b=8, per noisy pass. Per-edge gating multiplies that gate-head cost by up to L-1. These alternatives can cost more than the chain recurrences themselves.

### 6.3 Dependence queries and joint sampling

For m=8, the general circuit implementation uses up to **18 forward queries**, reusing leaf normalizers: `18*3,670,016 = 66,060,288` transition terms for all 512 chains, or **16,128 per valid position**. At m=16 and L=16 the maximum is 34 queries and work scales as O(m L K²). Store/batch evidence assignments, not 18 copies of the full emission tensor or gate matrices.

The HMM cavity method in §2.2 needs **two forward-backward traversals**, or **14,680,064** transition terms at L8/C512, plus O(mK) proposal gathers/reductions. If the original-evidence messages were already computed for feedback/proposals, only the second traversal is additional. At m=1 the score is zero. Exact joint sampling needs one evidence backward/forward preparation plus O(LK) categorical state draws and O(LV) emission draws; it does not enumerate V^L token assignments. Full marginal output still costs O(LKV) when feedback/ranking needs it.

### 6.4 Memory

f32 storage at the full C512/L8/K32/V256 planning size:

| Tensor | Size |
| --- | ---: |
| Full `[C,L,K,V]` modulated emissions, if materialized | **128 MiB**; **128.5 MiB** at V257 |
| One `[C,L,V]` potential or token-marginal tensor | 4 MiB; 4.016 MiB at V257 |
| One `[C,L,K]` message/leaf-normalizer tensor | 0.5 MiB |
| Block-shared `[C,K,K]` modulated transitions | 2 MiB |
| Per-edge `[C,L-1,K,K]` transitions or one edge-posterior tensor | 14 MiB |
| All 18 query forward-message arrays, if retained | 9 MiB; streaming queries can use less |
| Final-normalized noisy features `[C,L,256]` | 4 MiB; already a backbone output |
| Shared base emission/transition/initial probabilities | about 36 KiB at V257 |
| Position-specific 16 emission tables plus transition/initial rows | about 518 KiB |

Retaining teacher and student full emissions adds about 257 MiB before gradients, flow buffers, optimizer history or backend temporaries; the detached teacher need not remain live after producing sparse feedback. Avoid a second 128.5 MiB tensor solely for a gradient expression. With shared emission tables, local normalizers and token marginals can use matrix products or fused reductions without materializing the entire four-axis leaf tensor. Log-space cavity messages also avoid replicating leaves across queries. Momentum adds two row-shaped histories if applied separately to the two flows; neural optimizer memory depends on approved head sharing.

### 6.5 Runtime comparison and future measurement

No device throughput or per-launch timing for these new operations has been measured, so a defensible added wall time cannot be obtained from FLOP counts alone. As an explicitly hypothetical arithmetic scenario, 0.2–0.35 billion contributions at an effective 5–50 billion contributions/s would take roughly **4–70 ms**, before projection cost, logarithms, dispatches, synchronization and allocation. This is **not** a predicted integrated overhead. Thousands of small tensor launches could make the real addition much larger, as the current batching history demonstrates.

For scale only, a measured future addition of 10 ms / 100 ms / 1 s would be about **0.12% / 1.22% / 12.2%** of the 8.2 s B8 baseline. At B1 the arithmetic and tensor sizes above are one eighth (64 chains before skipping), but launch cost need not be one eighth; the same additions would be **0.62% / 6.2% / 61.8%** of its 1.617 s baseline. Do not advertise any of these hypothetical percentages as a result. Later measure the selected model at both B1 and B8, full optimizer step as well as isolated circuit work, on the same devices/backend features as the baseline.

## 7. Correctness and optimization checks

The implementation follows the checks below. Observed commands, errors, timings and adoption decisions are recorded in §10; none of the work is a quality study.

### 7.1 Independent f64 references and enumeration

Implement plain sequential f64 references independent of the Burn/CubeCL graph. Tiny cases should use V=2 or 3, K=2 or 3, and L=1–4 so both output sequences and latent paths can be exhaustively enumerated. Compare all quantities elementwise using the unchanged repository criterion

`abs(actual-ref) <= 2e-5 + 3e-4*abs(ref)`.

Compare f64 reference implementations to each other more strictly before using that threshold for f32 tensor/GPU paths. Compare log probabilities in log space where probability underflow could conceal a discrepancy. Enumeration must cover:

1. Positive base row sums and local emission sums; zero log-gates restore base transitions; within-group gate sharing and row normalization match Eq. (11). Test the cancelling one-group case to catch a wrong K32 grouping.
2. The unconditional partition sum over outputs and states equals one; complete-assignment likelihood equals explicit latent-path summation. With all positions unresolved, `log Z_e=0`.
3. Every visible-position subset and every supported visible assignment: evidence mass, numerator/denominator conditional likelihood, all singleton marginals, arbitrary multi-position conditional queries, and observed point masses. Sum conditional probabilities over completions to one. Test evidence to the right of a queried token.
4. Dependence scores by direct enumerated conditional probability, general `2m+2` queries, and the two-pass cavity optimization. Include m=1, exact independence (`delta=0`), strong dependence, thresholds exactly equal to scores, minimum-score fallback, ties, and last-pass completion.
5. Joint sampling: first compare every sampler conditional categorical row to enumerated values under the elementwise tolerance. Then use recorded-seed repeated samples and statistically specified multinomial/binomial confidence checks; the floating-point tolerance is **not** an empirical-frequency confidence interval. Verify cross-position correlations and observed-token preservation, not only singleton frequencies. Tiny exhaustive random-interval integration can additionally verify the exact sampler mapping without Monte Carlo error.
6. Both evidence-query posteriors (`gamma`, `xi`), local normalizer derivatives, gate row derivatives, neural VJPs and positive/negative circuit flows. Use f64 finite differences of the full conditional joint log likelihood; verify `F+ - F-` equals the log-weight derivative under the approved independent-coordinate convention. Do not finite-difference a softmax-logit parameter and compare it to a raw probability-coordinate flow.
7. The **chosen** Eq. (9) update, not just its signed derivative: alpha=0 identity; finite positive alpha; negative signed statistics; all-zero statistics; unreachable rows; tied occurrences; large/small probabilities; row sums and strict positivity where representable; Anemone equivalence to Algorithm 1; first-order agreement as alpha decreases. No monotonic likelihood improvement is promised by the paper for arbitrary finite conditional updates.
8. Padding and actual length: every L up to 16, short final blocks, multiple independent block rows, noncontiguous inputs where supported, dirty padded values, all-observed draws and completely inactive rows. Padded storage must not affect outputs, neural gradients, circuit flows or pooling. Test the approved L1 behavior explicitly.
9. Soft-feedback integration: full-support marginals to top-k original weights/retained mass, visibility and mask mixture, hard-mask arm, immutable observations, detached teacher gradients, student gradients through selected embeddings, and no circuit gradients from padded rows. Test that changing an unrevealed clean target changes neither teacher features nor evidence queries until it is supplied as observed/gold for the likelihood numerator.
10. Cached training versus prefix reconstruction, per-block versus batched paths, input-only versus Both, and optional retrieval enabled/disabled. The clean forward/loss must remain unchanged under the approved head-sharing semantics; shared-parameter gradients may change due to the added joint objective. Checkpoint/config identities must distinguish the selected circuit math and optimizer state from historical factorized models.

Under full positive support, impossible visible byte evidence should not occur. Include explicit synthetic zero-support tests to ensure undefined conditionals cause a clear error and do not silently become uniform, point masses, or ignored targets.

### 7.2 Fused CubeCL kernels and analytic VJP

Keep a transparent tensor implementation as a numerical comparison, but the performance implementation needs to avoid expressing every short chain edge and every query as many separate launches. Candidate kernel groups are:

- **Leaf normalization and selected-leaf evaluation:** stable reductions over all 257 symbols for each `(block,position,state)`, returning local log normalizers and gold/observed/proposal log emissions without a full `[C,L,K,V]` allocation. Analytic VJP includes `one_hot - g` and nonnegative emission-flow accumulation.
- **Grouped transition normalization:** apply expanded group log-gates logically, not with a materialized Kronecker product; row log-sum-exp with analytic base-row and group-gate derivatives. Keep a single transition per block if gates are block-shared. Root-row handling follows the approved decision.
- **Short-chain evidence forward/backward and flow accumulation:** L<=16/K32 lets each chain or chain/query use compact messages; fuse reductions, evidence masks, posterior counts and VJP, then reduce shared parameter statistics across independent chains. Do not use a generic exponentially sized circuit graph. Combine `e+` and `e-` work where it avoids memory traffic without subtracting away the separate flows.
- **Full marginal readout for feedback:** mix state posteriors with normalized leaves across the complete vocabulary. Reuse local normalizers, then call existing stable sparse selection and embedding machinery; no new top-k kernel is needed just because the output layer is joint.
- **Cavity dependence queries:** compute exact original-evidence and proposal-evidence messages and gathered proposal probabilities in a small number of launches. This readout is decision-only and needs no training graph unless explicitly used in a future loss.
- **Dual-flow row update:** fuse nonnegative flow reduction/approved momentum, log-ratio or probability update, and final row normalization, with distinct circuit state from AdamW. This is a parameter update, not an autodiff operation requiring a VJP.

Analytic VJPs are required for the differentiable fused leaf, gate and evidence/marginal paths. Sampling/commit decisions do not need VJPs. Existing Burn matmul/head projection, final RMSNorm, top-k, sparse expected embeddings and optimizer kernels should remain where they already do the required work; a custom projection kernel is not implied by Codpula.

After numerical acceptance, measure synchronized warm forward, backward and update sections, full B1/B8 optimizer steps, query overhead by m, peak/live allocation limitations, retained tensors and dispatch counts. Compare at least the transparent tensor path and the fused/cavity path with identical parameters and evidence. Separate compilation and autotuning from warm execution, and do not turn an isolated-kernel improvement into a full-step speedup estimate.

## 8. Audit of the reusable relation circuit

Source: `reuse/crates/dre-relations/src/joint_circuit.rs` (line references below refer to the inspected snapshot). This is migration input, not a buildable active crate; its missing relation-specific modules must not become new dependencies of the LM.

### 8.1 Actual existing model

The file is already a short **latent Markov chain**, but its five positions are relation label, subject start, subject end, object start, object end (`:18-19,103-119`). It uses a relation table `[R,K]`, endpoint projections `H->K`, initial logits `[K]`, **four different** transitions `[4,K,K]`, and five hidden-dependent destination-state gates (`:21-61,273-290`). The default state count is 32. Its query axis `[B,Q,H]` is not the chain-length axis. Relation and pointer vocabularies differ by field.

It locally normalizes tilted leaves and computes conditional chain messages. That makes several inference algorithms useful, but it is **not already Codpula Eq. (11)**: destination-state gates are not square groups of base transition probabilities; transition matrices are not shared; probability parameters are neural logits, not positive circuit rows trained with Eq. (9). Its leaf score parameterization and relation-specific loss also differ.

### 8.2 What can be ported

| Existing item | Porting assessment |
| --- | --- |
| `logsumexp`, `normalized_log_weights`, `log_softmax_with_zero_mass`, `conditional_log_mass` (`:220-249`) | Reuse stable reduction ideas, with explicit impossible-evidence handling. These handle all-negative-infinity rows, not arbitrary NaN or positive infinity. |
| Prepared forward/backward messages (`:345-381,397-415,558-580`) | Port recurrences and message reuse; replace five fixed fields/four transitions by actual L, common vocabulary, validity masks and the approved shared transition/gate model. Its forward message excludes the current emission; preserve that convention when forming cavities. |
| Local-normalizer and mixture-readout matrix products (`:301-307,371-380`) | Reuse the factorization that avoids a full query-by-vocabulary-by-state tensor when base emission tables are shared. The implementation is not entirely log space: these parts exponentiate shifted tables and use matmul. Recheck numerical assumptions rather than copying them unchanged. |
| Differentiable `prepared` versus detached `prepared_decisions` (`:216-218,397-415,592-665`) | Reuse the separation between training inference and decision-only readout. Teacher feedback needs the latter; the student likelihood needs the former or analytic VJP. |
| Sequential conditional greedy traversal (`:583-626`) | Reuse message/update structure for a deterministic comparison or as a starting point for categorical draws. **No random joint sampler exists in this file.** Its greedy output is not global token MAP. |
| Independent enumeration and finite differences (`:747-1009,1167-1239`) | Adapt f64 output/latent enumeration, all evidence masks, and gradient-test patterns to variable-length byte chains. Existing tests are source evidence only; none was run for this document. |

### 8.3 What must not be copied as Codpula

- `coupling_loss_from_call` (`:439-509`) computes log factorized alternative-set mass minus log conditional joint alternative-set mass with annotation groups. It is not the conditional clean-completion NLL and not the dual-flow training objective.
- Five-field gold groups, relation-given-spans, pointer preparation, mention readout and six relation dependency features are unrelated to byte block output (`:253-270,427-431,589-590,629-665`).
- Four learned transition matrices and per-field destination gates do not become a homogeneous grouped-gate HMM by increasing a field-count constant.
- No positive-probability circuit row storage, nonnegative dual-flow optimizer, gate-group head, byte-HMM sampler or CubeCL circuit VJP is supplied by this file.
- Existing leaf shifts rely on a limited score range and max subtraction (`:266-269,291-307`); lifting that score assumption while retaining the same probability-space products can underflow. Use independent stable references for the approved new parameterization.

### 8.4 Source-derived edge-case problems to avoid

These are inspection findings, not observed execution failures:

1. At `:299-307,327-342,384-393`, an all-negative-infinity leaf can have zero total emission mass, yet the unresolved-field path substitutes log one. That treats an empty leaf as normalized and can produce a nonzero evidence mass for an empty-support joint. Unknown normalized leaves legitimately contribute one; empty leaves do not.
2. At `:562-575`, observed-field readout can unconditionally restore a point mass even when the total supplied evidence has zero probability. A conditional on impossible evidence is undefined; returning a point mass does not repair it.
3. The exceptional tests (`:1106-1163`) check finite/negative-infinity outputs and valid indices but do not establish all required zero-total-mass semantics. They do not replace the new partition/evidence checks.

Port the inference ideas with explicit support semantics, not those exceptional behaviors. Under the chosen full positive byte support, ordinary training should never need an empty-support fallback.

### 8.5 Current predecessor sources: salvage and differences

The current predecessor was audited after the initial snapshot-based specification. Its `joint_circuit.rs` is 1,831 lines and implements A5 Codpula; it is newer than the `reuse/` snapshot described above. Source hashes, source ranges, destinations and adaptations are recorded in `reuse/PROVENANCE.json`. The old snapshot files were not replaced or imported as a second executable workspace.

| Current predecessor piece | Used here | Necessary changes or reason not copied |
| --- | --- | --- |
| `joint_circuit.rs:330-378`, stable reductions and categorical inverse CDF | Adapted zero-mass-safe log-sum-exp, strict uniform/CDF checks, positive-support selection and last-positive-class rounding behavior into `codpula.rs` | The sampler accepts log weights, rejects invalid or empty support clearly, and retains exact latent FFBS rather than the predecessor's fixed-order token chain-rule sampler. |
| `joint_circuit.rs:498-520,777-900`, prepared messages, indexed proposal uniforms and absolute candidate-subset dependence | Prepared-message/cavity semantics and explicit `[row,position,(state,token)]` proposal uniforms; existing seeded entry point uses the same sampler | Actual byte-block positions replace five relation/pointer fields. One shared transition and emission table replace field-specific tables and gates. Unused padding/observed-emission uniforms do not affect the draw. |
| `joint_circuit.rs:1378-1392,1474-1554,1580-1668`, support/sampling/dependence regressions | Adapted singleton support, impossible evidence, observed/padded rows, strict CDF boundaries, unused uniforms, negative signed dependence and zero-mass tests | The new f64 latent/output enumeration remains independent; no relation union-event or alias machinery is imported. |
| `commitment.rs:6,79-96,117-135` | Adapted supported-candidate selection and tests for strict threshold, candidate subsets, ties, fallback, empty candidates and final completion | Threshold is `0.05`, matching both the predecessor constant and the paper's Appendix C traces. Production rejects nonfinite proposal scores before selection; unsupported positions cannot be committed, even on the last pass. |
| `diffusion_scheduler.rs:125-216,265-301` and `diffusion_scheduler_tests.rs` | Adapted per-call indexed uniforms, pre-potential temperature, detach/absorb semantics and rollout regression patterns | SBD retains its immutable clean-prefix cache and sparse marginal feedback. The production `advance_generation_state` helper is exercised directly in cached/reconstructed, padded/unpadded multi-call tests. Relation windows, byte-offset heads and document banks are not imported. |
| `training_query_chain.rs:646-700` | Adapted dirty-padding, detached-transition and observed-feature gradient regression patterns | No all-round averaged training loss: this model keeps the approved detached teacher plus final-student joint NLL and the existing clean causal loss. |
| `training_query_chain.rs:19-132`, `training_query_batch_loss.rs:58-150,310-345` | Audited, training implementation not copied | Their per-document RNG streams select relation/mention alternatives and their losses average every call. This byte LM has unit byte targets, the existing pinned SBD mask RNG and one student event per block; importing that training frame would change the approved objective. |
| `learned_output.rs::probability_entropy`, lines 149-176 and regression cases | Audited as a numerical reference | It masks exact zero before logarithms so entropy adjoints remain finite without clipping positive mass. SBD's generation ranking already uses detached host log probabilities, with zero-mass terms omitted and no entropy adjoints; another tensor reduction is unnecessary. |
| `decode.rs:33-79` | Not copied | Relation argmax and temperature applied to a final posterior do not implement joint FFBS or temperature on neural potentials. The zero-mass defect below is not inherited. |

The predecessor's initial state count defaults to 64 and its four transitions and destination-state gates are relation-specific. Its FP32 leaf score limit of eight protects an exponentiated-table/matmul implementation; it is **not** imposed on these approved log-domain base rows and normalizers. The new model uses K32, all V257 symbols, a valid-state mean pool, a biased zero-initialized 64-log-gate head, and group width four.

No dual-flow/Anemone/Eq. (9) implementation or corresponding flow-statistics tests were found after literal and variant searches across predecessor crates, training sources, tests, README and session report. `src/train/training_loop.rs:304-310` constructs AdamW. Its momentum and circuit-learning-rate tests concern AdamW, not separate positive/negative circuit flows. Therefore the paper's explicit normalized split, shared occurrence reaches, log-domain multiplicative update and separate bias-corrected flow averages are new here, rather than mislabeled copies of predecessor training.

The predecessor session report records earlier CPU/CUDA checks (`.agents/SESSION_REPORT.md:244`) and a three-update A5 import smoke (`:568-570`). These are reported predecessor results, not checks rerun or quality evidence for this byte model. The adapted tests are checked separately in this workspace.

**Predecessor defect found by source inspection:** `decode.rs:56-61` receives `max_log=-infinity` for an all-zero posterior, then evaluates `-infinity-(-infinity)` and returns a NaN winning probability. Its `:119-132` test covers uniform and mixed-zero rows, not an all-zero row. That decoder was not ported. The new categorical regression explicitly starts from `[0,0,0]`, converts to log mass, and requires a clear zero-support error. This is a source-derived defect, not a predecessor production failure observed here.

The initial locally written commitment helper also needed a correction: its last-pass shortcut could include unsupported candidates. The adapted helper excludes them on every pass, and generation errors clearly if an unresolved proposal has no finite supported score.

## 9. Implementation readouts

The prepared block distribution exposes conditional joint log likelihood, token marginals, evidence-query log masses, seeded joint proposals and cavity dependence scores. Training returns the neural loss and separate nonnegative circuit statistics; the shared training-step helper applies AdamW and the circuit update from the same pre-update outputs. Generation uses detached prepared values.

Cached, per-block, batched, reconstruction, evaluation and generation callers use the selected output distribution. Omitted or explicit `joint_output: "factorized"` retains historical serialization and initialization. `joint_output: "codpula"` changes model identity; `joint_execution` selects mathematically equivalent tensor or fused circuit execution, and `commitment: "dependence"` selects the adaptive decoder. The independent factorized model is used only for cost accounting in this pass; no quality studies, ablations or sweeps are included.

## 10. Observed implementation evidence

### 10.1 Implementation and correctness

`crates/experiment/src/codpula.rs` contains normalized log rows, prepared evidence queries, forward/backward statistics, conditional NLL, full token marginals, indexed-uniform FFBS, cavity scores and the circuit optimizer. `sbd.rs`/`sbd_training.rs` supply the final-normalized features and integrate every output path. `train::optimizer_step` applies the existing neural optimizer and then the circuit update; checkpoint loading restores frozen circuit rows and validates their integer step counter as well as float tensors.

Initialization draws independent seed-local row logits in `[-0.01,0.01)` and applies row log-softmax. The gate weight and bias are zero. At D256 the added neural head has 16,448 parameters; base circuit rows contain 9,280 log probabilities. Base rows plus separate positive/negative momentum histories occupy 111,360 f32 bytes, plus the scalar step counter. They are checkpointed but excluded from AdamW and neural missing-gradient checks.

The independent references are `codpula_reference.rs` and `codpula_model_checks.rs`. All elementwise GPU/tensor comparisons use exactly `2e-5 + 3e-4*abs(ref)`; a tolerance ratio is error divided by that quantity.

| Accepted gate | Maximum absolute error | Maximum tolerance ratio |
| --- | ---: | ---: |
| CPU circuit/reference checks | 6.385e-6 | 0.1750 |
| Vulkan circuit/reference checks, including tensor path | 1.2103e-5 | 0.1512 |
| Fused circuit operations and full VJPs within that Vulkan report | 8.801e-7 | 0.02940 |
| CPU complete-model gates | 1.908e-6 | 0.09537 |
| Fused Vulkan complete-model gates versus CPU | 2.385e-6 | 0.04744 |

The model matrix covers GDN-2 and KronLSTM, each with ordinary features and with oHC/wavelet/Lngram/retrieval. Cached/reconstructed and per-block/batched outputs, losses, all neural gradients and circuit flows agree. The exact L1 bypass, immutable observations, future/hidden-target exclusion, dirty padding, final-norm provenance, teacher detachment, selected-embedding gradients and observed-feature gate gradients pass. Fixed-uniform multi-call rollouts preserve absorbing commitments, clean caches and the next device RNG value.

Checkpoint continuation passes two further complete neural/circuit updates, including exact circuit histories/step and canonical AdamW state. The pinned historical causal checkpoint remains loadable with identity `c7eca1dace65f65fea8db186a6c0cc5427a320f4bf89f76a4ea0482c86dc42ca`; the obsolete dense-attention Step-8 checkpoint remains rejected. Omitted/explicit factorized serialization, initialization and fingerprints agree. Historical payloads are not modified.

The f64 enumeration, finite-difference VJP, posterior-flow, update, support and indexed-uniform checks pass. Sampling checks use a declared statistical confidence criterion in addition to exact categorical/FFBS mapping checks, rather than misusing the floating-point tolerance for empirical frequencies. An explicit all-zero posterior requires a zero-support error.

Workspace tests pass **51 experiment + 7 kernel + 16 mixer tests**; one GPU-sharing test stays explicitly ignored. The unchanged `sbd-reference` and `sbd-batching-reference` commands pass on CPU and Vulkan. Evidence is under `artifacts/codpula/{cpu,vulkan-fused,factorized-cpu,factorized-vulkan}/`, with circuit reports also at `artifacts/codpula/codpula-reference.json` and a concise test summary in `cpu-tests.txt`.

One important new-kernel defect was found and fixed before acceptance: a compile-time `if !posterior { terminate!(); }` did not exclude the following CubeCL posterior code. The forward-only path could overwrite its evidence/transition aliases. The whole posterior suffix now sits inside positive `if posterior { ... }`; untracked repeated queries, single-chain mass, conditional NLL and complete evaluation/generation have regression coverage.

### 10.2 B1/T512 paired execution cost

Local Intel Arc Graphics (MTL), Mesa 26.2.4, Vulkan f32, no autotune or Burn fusion; B1/T512/L8, D256, four layers and the selected Step-7/HOLA components. There are two warmups and **11 same-session interleaved complete-step samples per mode**, alternating mode order, with common initial backbone parameters and identical examples/mask draws at each step. Every mode has its own optimizer trajectory. The process includes the clean pass, detached teacher, noisy loss/backward, finite checks, clipping, AdamW and circuit update; it excludes evaluation/checkpoint I/O and cold compilation.

| Complete optimizer step | Median ms | Launches |
| --- | ---: | ---: |
| Factorized cost reference | 1,522.502 | 24,242 |
| Codpula tensor, selected | 1,697.014 | 27,842 |
| Codpula fused GPU, explicit opt-in | 1,631.830 | 25,578 |

Tensor Codpula's median is 11.46% above the factorized median; its paired penalty is 154.355 ms, 95% paired bootstrap interval **[109.492,187.927] ms**. Fused Codpula's median is 7.18% above factorized; its paired penalty is 117.456 ms, interval **[57.658,149.567] ms**. These are cost accounts, not quality comparisons.

Fused versus tensor saves a paired median **60.951 ms**, but its interval is **[-23.545,84.668] ms**, with fused faster in 8/11 pairs. Therefore **do not adopt fused by default**: `joint_execution` stays tensor when omitted and `configs/lm512-codpula.json` retains that selection. The fused selector remains explicit for the verified kernel path and future measurements. No extra timing sessions were used to seek a favorable interval.

The isolated resident circuit check includes gate projection, local normalizers, both evidence passes, neural backward, flows and circuit update, but not the backbone/token projection/uploads/neural optimizer. Its medians are **80.350 ms tensor / 61.654 ms fused**, with **2,716 / 1,044 launches**. The paired saving is **20.025 ms [16.577,23.311]**, 11/11 faster. This real isolated gain does not establish a complete-step gain.

Sampled pool reservation in the three-runner process is approximately **1,697,281,920 bytes (1.697 GB)**. All three models/optimizers are resident, so this is neither a per-mode peak nor an incremental Codpula allocation. Logical tensor sizes remain in §6; exact live GPU peak memory is unavailable.

Raw samples, launch-family counts and method: `artifacts/codpula/b1/codpula-benchmark.json` and `codpula-isolated.json`. The decision is also recorded in `artifacts/codpula/adoption.json`.

### 10.3 B8 diagnostic and tiny smoke

The selected **tensor** path at B8/T512/L8 completed one warmup and one timed complete optimizer step in **8,000.071 ms**, with **30,745 launches** and **8,826,550,976 bytes** sampled pool reservation. Evidence: `artifacts/codpula/codpula-diagnostic.json`. It fits without subdividing the adopted batched block axis. The earlier factorized 8,167.400 ms / 8,818,099,392-byte result is historical context only: these are not paired timings or an incremental-memory experiment, so neither their time difference nor their pool difference establishes a Codpula saving.

The only new training smoke uses the full D256/four-layer model at **B1/context32, seed1, 96 bytes / three optimizer steps**. The context32 config uses the existing compatible `association_counts: [4]` setting; the default larger diagnostic counts cannot fit that context. Checkpoint evaluation/generation succeeds with tensor Codpula: clean NLL **5.447807**, joint block NLL per valid target **4.463265** on one 32-token held-out window, and **eight finalized/replayed tokens in one dependence pass** at threshold 0.05. These finite smoke values are not language-quality evidence.

Evidence: `artifacts/codpula/smoke.json`, `smoke/kronlstm-hola-seed1/metrics.json`, `smoke/sbd-evaluate.json` and `smoke/sbd-generate.json`. Smoke and scratch continuation **model/optimizer payloads were deleted after their gates**; JSON state, reports and hashes remain. No training study, ablation, sweep or quality comparison was run.

### 10.4 Deterministic native optimizer clipping (2026-10-25)

A requested continuation check exposed a pre-existing failure: both the Step-11 workspace and the unmodified Codpula workspace `bd39717eb8d4a328256ec7c39eaf0b8062365238` failed the same Vulkan `codpula-model-reference` invocation with `embedding.weight: values are not bit-identical`. The baseline was built with the shared helper before reproduction; this is not attributed to the new decoding selectors.

The neural gradients at the continuation boundary are bit-identical and clipping is active: the recorded embedding gradient has host-f64 norm **2.0594476658028906**. Burn0.22.1 native `GradientClipping::Norm` used `square().sum()`: without autotuning, CubeCL global sum dispatches `OneShot(4)` and each cube contributes by floating `fetch_add`. The rounded norm rescales every parameter gradient. Native AdamW now uses a flattened `sum_dim(0)` norm, whose output owner scans a fixed sequence, while retaining Burn's AdamW formula, moments, counter and record type. Native Lngram clipping uses the same helper. No tolerance or bit-identity condition was relaxed.

The strict narrow `codpula-continuation-reference` and full previously failing `codpula-model-reference` now pass. The regression repeats a clipped [257,16] gradient32 times, requires bit-identical output and compares to independent host f64: maximum absolute error **1.0094288405149143e-8**, maximum tolerance ratio **0.0002473880203463499** under `2e-5+3e-4*abs(ref)`. CPU tests also cover it. Continuation reports retain raw-gradient hashes and active-clipping norms; they still require exact parameter/optimizer continuation.

The user's exact historical causal `checkpoint-check` command against `artifacts/lm512-vulkan/kronlstm-hola-seed1/checkpoint-13` passes; report: `artifacts/remasking/historical/checkpoint-check.json`. Its existing stricter diagnostic tolerances were not changed.

**Adopted fused optimizer unchanged:** `crates/kernels/src/update.rs` has no changes versus985ce60f. Its squared-norm kernel produces at most64 fixed-order partials; the final global sum has only one nonzero cube, while other cubes add exact positive zero. Thus the final backend uses the same atomic primitive but has no competing nonzero inter-cube association. This differs from native clipping over thousands of input elements. No fused optimizer change or new default is introduced.

**Other reductions checked:** tensor dual flows use `sum_dim` with owned outputs; fused emission counts use an owned64-lane tree per state/token, and fused root backward scans blocks in order. Their only atomics are host integer launch/allocation counters, not flow arithmetic. Point-event scatter has one update per output row, so it has no collision. The model's host gradient-norm total is only a finite-value guard, not a clipping coefficient; loss/readback sums do not choose stopping or updates, and the strict continuation fixture's scalar loss has one active cube. Those paths were not changed merely because they contain `sum()`. The new energy fitter also used native norm clipping and now shares the deterministic native optimizer helper.

Source evidence: `burn-optim-0.22.1/src/grad_clipping/base.rs:70–93`; `burn-cubecl-0.22.1/src/kernel/reduce/base.rs:120–160,435–449,635–641`; `cubek-reduce-0.3.1/src/routines/shared_sum.rs:202–205` and `components/global/unit.rs:178–185`. This is an execution correctness repair, not a change to the Codpula probabilistic model.

