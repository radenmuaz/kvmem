# An MHE-Style Latent-Geometry Optimizer: Analysis and Experiment Plan

## 0. The reframing in one sentence

Treat optimization as a **state-estimation problem**: there is a hidden, low-dimensional *geometry state* `z_k` that evolves as the iterate moves through parameter space and *emits observations* (parameter displacements, gradients, losses, secant pairs). Instead of recomputing curvature from scratch at every point (what every standard method does), we **infer `z_k` by fixed-lag smoothing / moving-horizon estimation over a window of past observations**, exploiting the temporal coherence of the trajectory, and then take a trust-region step using the *predicted* geometry.

The single most important claim to test — and the one most deserving of skepticism — is whether this estimator can use information that is *structurally unavailable* to a memoryless second-order step. I argue below that there is exactly one such source of genuine extra information (curvature **dynamics**, i.e. third-order / `dH/dt` structure, plus optimal **denoising** of the curvature estimate), and that everything else reduces to a known method. The experiments are designed to isolate precisely that.

---

## 1. Latent parameterizations of `z_k`

Let `d` = number of parameters (10⁶–10⁹), `r` = subspace rank (≈ 5–50), `N` = horizon length (≈ 20–100).

| Option | State `z_k` | Pros | Cons |
|---|---|---|---|
| **(A) Low-rank-plus-scalar curvature + trust region** | `U_k ∈ St(d,r)` (orthonormal subspace), `λ_k ∈ ℝ^r` (log-curvatures in subspace), `c_k` (scalar background log-curvature), `δ_k` (log TR radius), small dynamics params | Interpretable; matches limited-memory SR1/BFGS trust-region structure; O(dr) memory; tractable inner solve | Subspace is a manifold (Stiefel) → nontrivial dynamics/inference |
| **(B) Pure spectral state** | eigenpairs `(U_k, Λ_k)` with Grassmann/Stiefel rotation dynamics | Cleanest "curvature drift" semantics | Needs HVPs for clean eigen-estimation → budget risk |
| **(C) Local quadratic-model coefficients** | `(g_k, H_k)` restricted to subspace + higher-order drift terms | Most direct Taylor-model interpretation; loss observations fit naturally | Redundant with (A); less structured |
| **(D) Learned black-box latent** | abstract `z_k`, with `f`, `h` as small nets trained offline | Maximally expressive; subsumes learned-optimizer (L2O) ideas | Loses interpretability; training/meta-overfitting risk; harder to reason about |

**Recommendation: build around (A).** It is the unique point that is (i) structured enough to admit cheap inference, (ii) expressive enough to carry the novel "curvature dynamics" signal, and (iii) directly compatible with a closed-form trust-region subproblem. (D) is worth keeping as a later hybrid: *learn* `f`/`h` offline, *run MHE* online — this is the genuinely new combination relative to pure L2O.

The implied Hessian model is the standard limited-memory form
```
H_k ≈ c_k·I + U_k (diag(e^{λ_k}) − c_k·I) U_kᵀ
```
which has cheap inverse, cheap matvec, and a closed-form ellipsoidal trust-region solution.

---

## 2. State dynamics — where the entire thesis lives

The value of the method is *entirely* determined by how informative `z_{k+1} = f(z_k) + w_k` is. Three regimes:

1. **Random-walk / persistence** (`f = identity`). The MAP estimate of `z_k` from windowed secant pairs is then essentially a **denoised, regularized limited-memory quasi-Newton model**. Useful (denoising is real value in the stochastic setting) but *not* fundamentally beyond L-BFGS. This is the null hypothesis.

2. **Curvature-magnitude drift** (`λ_{k+1} = A λ_k + b`). Captures "eigenvalues are systematically growing as we enter a sharper basin" or "flattening as we approach a minimum." Lets the step *anticipate* curvature it has not yet measured.

3. **Subspace rotation** (`U_{k+1} = exp(Ω_k) U_k`, `Ω_k` skew-symmetric, slowly varying). Captures a curving ravine/valley whose dominant directions rotate predictably. This is the strongest form of the predictive claim: you *extrapolate where the valley is going.*

**The crux.** A memoryless Newton/Gauss-Newton step uses `(g_k, H_k)` *at the current point*. It does not and cannot use: (a) the rate of change of `H` along the path (third-order tensor `∇³L` contracted along the trajectory), (b) the temporal correlation structure of stochastic gradient noise, (c) the fact that the iterate keeps moving, so the geometry at the *landing* point differs from the geometry at the *launch* point. Regimes 2–3 turn (a)–(c) into exploitable signal. **If the trajectory has temporal coherence — and deep-learning trajectories empirically do (top Hessian subspace is strikingly stable; gradients live in a slowly drifting low-dim subspace) — there is real predictive information here that no single-point second-order method has.**

---

## 3. Does it extract information unavailable to second-order methods? — honest verdict

**Yes, but only via two mechanisms, and only in a specific regime.**

What MHE can genuinely exploit that vanilla second-order cannot:

1. **Curvature velocity (third-order info).** Observing how secant pairs evolve across the window implicitly estimates `dH/dt` along the path. A "predictive Newton" step `p = −H(f(ẑ_k))⁻¹ g` using the *one-step-ahead* Hessian beats Newton precisely when `H` changes fast (early training, phase transitions, warmup). Single-point Hessians have zero access to this.
2. **Optimal denoising of curvature.** With a measurement-noise model `R`, the smoother separates signal curvature from gradient noise Kalman-optimally. Stochastic L-BFGS is famously fooled by noisy secant pairs and needs ad-hoc damping; MHE handles it in-model.

A third, softer mechanism is **long-horizon trend detection**: recognizing "we've been in the same slowly rotating valley for 50 steps" justifies a large, *confident* step along it — an explicit-model analog of conjugate-gradient accumulation / Anderson acceleration, but with a trust region for safety.

Where it **collapses to known methods or fails** (state these as kill conditions):

- Random-walk dynamics + secant-only observations ⇒ regularized limited-memory quasi-Newton. No free lunch.
- The "large informed step" requires the low-order model to hold over a *large* region; trust-region machinery caps the step when the landscape is strongly non-quadratic, removing the advantage exactly when you wanted it.
- **High-noise / small-batch SGD may destroy higher-order observability**: the `dH/dt` signal you are trying to estimate can be swamped by gradient noise. Longer windows average it down but add staleness (bias–variance in `N`). The method should shine in **large-batch / near-deterministic** regimes and degrade gracefully toward Adam/L-BFGS in high-noise regimes.

**Net:** plausibly a real win in large-batch training, problems with strong low-rank curvature and temporally coherent trajectories, and the early-to-mid phase where `H` is evolving. The honest fallback value proposition (if prediction fails) is "noise-optimal limited-memory trust-region method" — still useful, but not the headline.

---

## 4. Observability

Observability = can `z_k` be recovered uniquely from the window? This is where the design earns its keep.

- **Subspace `U_k` is observable only in directions the trajectory has excited.** If iterates only move in a 3-dim subspace over the window, curvature in unexcited directions is unobservable — identical to quasi-Newton capturing curvature only in the span of its secant pairs. This is the **persistent-excitation** condition from system identification. Practical consequence: you may need the natural stochastic noise (or mild injected probing, or one occasional HVP) to keep curvature observable. Each gradient difference reveals one Hessian-vector-like product `y_i ≈ H s_i` — directionally, exactly like Lanczos/Krylov.
- **Curvature magnitudes `λ`** are observable from Rayleigh quotients `y_iᵀs_i / s_iᵀs_i` restricted to the subspace; need ≳ `r` independent directions in the window ⇒ `N ≳ r` with margin.
- **Trust-region radius `δ`** is observable from the history of agreement ratios `ρ = actual/predicted decrease`.
- **Loss observations `L_i`** over-determine the local quadratic and tie `g`, `H` together through the Taylor expansion — an extra, *free* consistency constraint that pure quasi-Newton throws away. This is a concrete, cheap edge worth isolating in ablations.

**Observability Gramian intuition:** recoverability of in-subspace curvature is governed by the conditioning of `Σ_i s_i s_iᵀ` restricted to `U_k` — the same matrix that controls L-BFGS conditioning. Monitor its smallest eigenvalue online; when it collapses, shrink the trust region in the unobservable directions rather than guessing.

---

## 5. Computational complexity & memory scaling

| Quantity | Cost | Comment |
|---|---|---|
| Gradient | 1× SGD | Dominant FLOP when `d`·batch is large |
| Project `g_k, s_k` onto `U_k` | `O(dr)` | Same order as L-BFGS two-loop recursion |
| Subspace update (incremental SVD / rand. range finder over `[s|y]`) | `O(dr)`–`O(dr²)` | Allow ≤1 direction in/out per step |
| Inner smoother (EKF fixed-lag, linearized) over window | `O(N r³)` per inner Newton iter, few iters | Negligible iff `N r³ ≪ d·batch` |
| TR step (low-rank + scalar background) | `O(dr)` | Closed-form ellipsoidal solve |
| **Memory: `U_k`** | `O(dr)` | The binding constraint; equals L-BFGS with `m≈r` |
| **Memory: window of observations** | `O(N r)` | Store *subspace coefficients*, **not** full-`d` history — this is the memory win over "keep 100 gradients" (`O(Nd)`) |
| Low-dim state + dynamics | `O(r²)` | Negligible |

**The 2–3× SGD budget is realistic** for moderate `r` *provided we do not buy extra gradient/Hessian-vector evaluations*. A single HVP costs ≈1–2× a gradient and would blow the budget if done every step. **Design rule: rely on secant pairs (free, from consecutive gradients); budget at most ~1 HVP every few steps** if observability demands it.

**At LM scale** (`d` in the billions), `O(dr)` forces small `r`. The fix that also exploits structure: **per-layer / per-tensor block-diagonal geometry states** (each tensor gets its own small `z`), mirroring K-FAC's per-layer factorization. This keeps memory linear with a small constant and makes the inner solves embarrassingly parallel.

---

## 6. Low-rank / Krylov representation

Maintain `U_k` as a **temporally smoothed, denoised Krylov-like subspace**:

- **Secant-span basis** from recent `{s_i, y_i}` — *free* (no extra HVPs). Preferred under the budget.
- **Online gradient-subspace PCA** (incremental SVD of recent gradients, à la gradient-subspace / GaLore observations) — directly supports the "subspace drifts slowly" dynamics model.
- **Lanczos/randomized eigensolver** — cleanest spectrum but needs HVPs; reserve for occasional refresh.

**Crisp differentiator vs. Krylov-Newton (Newton-CG / Hessian-free):** those rebuild `K_r(H,g) = span{g, Hg, …, H^{r-1}g}` *from scratch at every iterate* (memoryless). Our `U_k` is the **amortized and extrapolated Krylov subspace** — maintained and *predicted forward* across iterations. That amortization is the mechanistic source of any speedup over Krylov-Newton.

---

## 7. Horizon objective & trust-region formulation

**MHE cost at step `k` over window `[k−N, k]`:**
```
J = Γ(z_{k−N})                                   # arrival cost (prior summarizing pre-window history)
  + Σ_i ‖ z_{i+1} ⊖ f(z_i) ‖²_{Q⁻¹}              # dynamics residual; ⊖ = Stiefel manifold difference for U
  + Σ_i ρ_g ‖ g_i − ĝ(z_i, w_i) ‖²               # gradient consistency
  + Σ_i ρ_s ‖ y_i − H(z_i) s_i ‖²                # secant consistency
  + Σ_i ρ_L ( L_i − L̂(z_i, w_i) )²              # loss consistency via local quadratic model
  + constraints:  e^{λ} ≥ 0,  c > 0,  U ∈ St(d,r)
```
The arrival cost `Γ` is the truncated-history prior; approximate it with the smoother's covariance at `k−N` (EKF-style). Use a **linearized fixed-lag smoother as the inner solver** rather than a full nonlinear MHE — it is far cheaper, warm-starts trivially from the previous step, and is stable; promote to full nonlinear MHE only if linearization is shown to bottleneck accuracy.

**Step computation (the payoff):** solve the trust-region subproblem with the predicted geometry
```
min_p  g_kᵀ p + ½ pᵀ H(f(ẑ_k)) p     s.t.  ‖p‖_{M_k} ≤ δ_k
```
With `H = cI + U diag(e^λ − c) Uᵀ` this has a closed-form solution (project to subspace, scalar background outside — the limited-memory SR1/OBS trust-region machinery). Two novel twists:

1. **Use `H(f(ẑ_k))`** — the one-step-ahead *predicted* Hessian — because the iterate moves before the step lands.
2. **Anisotropic, estimation-driven trust region.** Make `M_k`/`δ_k` direction-dependent: the smoother's *covariance on `z`* sets per-direction radius. Big confident steps along well-estimated, stable, low-curvature valley directions; tiny steps in uncertain directions. The TR radius `δ_k` is itself a latent state whose dynamics are driven by the agreement-ratio history — classic TR radius updates become *estimated* dynamics. This Bayesian justification for anisotropic trust regions is conceptually clean and not standard practice.

---

## 8. Positioning vs. prior art (so claims stay honest)

- **L-BFGS / limited-memory SR1 trust region:** the special case with random-walk dynamics, no loss observations, no denoising, no prediction.
- **K-FAC / natural gradient:** structured curvature but *memoryless* online EMA — no horizon optimization, no prediction.
- **Kalman-filter views of SGD:** filter on weights/gradients; the novelty here is a filter on a *geometry* state with nonlinear smoothing.
- **Anderson acceleration / regularized nonlinear acceleration:** the closest "use a window of iterates to extrapolate" relative — essentially the *linear fixed-point* analog. MHE generalizes it with explicit geometry, a noise model, constraints, and a trust region.
- **Learned optimizers (L2O):** overlaps option (D); MHE adds interpretability + an explicit model, and the *online inference in a (possibly learned) state-space model* is the new combination.

Honest framing: **not new physics, but a unifying estimation-theoretic generalization** that, in the right regime, exploits curvature dynamics and noise structure each prior method ignores individually.

---

## 9. Minimal first algorithm (implementable in a few hundred lines)

State: `U_k` (d×r, Stiefel), `λ_k` (r log-curvatures), `c_k` (scalar), `δ_k` (log TR radius), tiny linear drift on `λ`.
Per-step observations: `s_k = w_k − w_{k−1}`, `y_k = g_k − g_{k−1}`, `g_k`, `L_k` (all free).
Inner solver: **EKF fixed-lag smoother** over window `N` — `O(N r³)`.
Subspace update: project `y_k, s_k`; periodic re-orthonormalization + incremental SVD of recent `[s | y]`; ≤1 direction in/out per step.
Step: ellipsoidal TR solve with `H = cI + U diag(e^λ − e^c) Uᵀ`, radius from `δ` and per-direction covariance; optionally use predicted `f(ẑ_k)`.
No extra HVPs in v0; add ~1 HVP / few steps only if observability Gramian degrades.

---

## 10. Experiment ladder (with go/no-go gates)

Fair currency throughout: **loss vs. number of gradient evaluations**, plus **loss vs. wallclock**, plus "**effective second-order steps per MHE step**." Hold the gradient-eval budget fixed across methods. Baselines at every phase: SGD+momentum, Adam(W), L-BFGS (full + stochastic), K-FAC, Newton-CG / Hessian-free, Anderson acceleration.

### Phase 0 — Synthetic landscapes (the experiments that decide the thesis)
- **Static convex quadratic, low-rank-plus-shift Hessian.** Sanity: estimator recovers `U, λ`; matches Newton when geometry is static; beats L-BFGS when gradient noise is injected (isolates *denoising*).
- **Time-varying quadratic** — `H(t)` with scheduled eigenvalue drift and subspace rotation. **THE decisive test:** does the *predictive* variant beat memoryless Newton/L-BFGS at equal gradient budget? Isolates the *curvature-dynamics* claim.
- **Rosenbrock / curved valley** — tests subspace-rotation tracking.
- **Stochastic quadratic** — sweep batch-noise × window `N`; tests denoising and observability under noise.

Ablations: random-walk vs. learned dynamics; ±loss observations; ±prediction; sweep `r`, `N`; TR on/off.

> **GATE 0 (kill criterion):** If on the time-varying quadratic the predictive variant shows **no edge over Newton-CG at equal gradient budget**, the core hypothesis is likely false. Stop and pivot to the "noise-optimal limited-memory TR" value proposition, or abandon.

### Phase 1 — Small real nets
Logistic regression (convex sanity), MLP and small CNN on MNIST / Fashion-MNIST. Verify the predicted advantage survives real curvature; measure *actual* per-step overhead against the 2–3× SGD budget.

### Phase 2 — CIFAR-10
ResNet-18 and a small ViT, full training to target accuracy. Report **compute-normalized** curves, not per-step. Crucial comparison: **large-batch regime** (where the method should win) vs. standard-batch (where noise should erode the higher-order signal). Confirms or refutes the regime story from §3.

> **GATE 2:** Require a clear wallclock-to-target-accuracy win in at least the large-batch regime before scaling.

### Phase 3 — Language model
Small transformer (~100M params, character-level or a clean subset corpus). LMs are the real prize and have documented low-rank gradient structure. Test whether subspace tracking + big steps reduce **tokens-to-loss**. Binding constraint is memory: use **per-layer/per-tensor block-diagonal geometry**, keep `r` small. Watch for observability loss during sharp loss drops (phase transitions) — exactly where predictive curvature should help most, and where a stale subspace is most dangerous.

---

## 11. Risks & mitigations (summary)

1. **Collapses to expensive L-BFGS** if dynamics are uninformative → Gate 0 catches this early.
2. **High-noise observability loss** → longer `N`, anisotropic TR shrinking unobservable directions, occasional HVP refresh.
3. **Inner-solve instability** (Stiefel + PD constraints, non-convexity) → linearized fixed-lag smoother with warm starts instead of full nonlinear MHE.
4. **Subspace drops a suddenly-important direction** → background term `c·I` + trust region bound the damage; monitor observability Gramian.
5. **Too many knobs** (`Q, R, N, r`) → adaptive MHE: estimate noise levels `Q, R` from consistency residuals online; treat `N, r` as the only real hyperparameters.

---

## 12. Bottom line

The proposal is not a guaranteed win, but it is a *well-posed* and genuinely novel synthesis. Its one defensible source of advantage over second-order methods is the use of **curvature dynamics (third-order/`dH/dt`) plus optimal denoising**, amortized across iterations like a predicted Krylov subspace — not the mere fact of using a long history. The decision rests almost entirely on Phase-0's time-varying-quadratic test. If that fires, the method has a real, mechanistically-explained edge worth scaling to CIFAR and LMs in the large-batch regime; if it does not, the honest fallback is a noise-robust limited-memory trust-region optimizer, which is incremental rather than transformative.