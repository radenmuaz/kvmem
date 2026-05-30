# An MPC-Style Latent-Geometry Optimizer: Companion Formulation

*(Read alongside the MHE memo. This document keeps the same latent geometry state and re-derives everything that changes when the framing flips from backward estimation to forward control.)*

## 0. The reframing in one sentence

MHE looks **backward** over a horizon to *estimate* the geometry state. MPC looks **forward** over a horizon to *plan a sequence of updates* that minimizes predicted loss-to-go, commits only the first update, and re-plans (receding horizon). The optimizer stops being "infer geometry, then step" and becomes "**model how loss-and-geometry will respond to my updates, plan the best trajectory of updates under constraints, act, re-plan.**"

The two are duals and ultimately complementary: the natural full system is **output-feedback MPC** — an MHE estimator feeding an MPC controller.

---

## 1. The MHE ↔ MPC duality

| Concept | MHE (estimation) | MPC (control) |
|---|---|---|
| Horizon direction | Backward `[k−N, k]` | Forward `[k, k+H]` |
| Decision variables | Past states `z_{k−N..k}`, process noise | Future controls `u_{k..k+H−1}` |
| Driving model | Dynamics + observation residuals | Prediction model of plant response |
| Boundary term | **Arrival cost** (prior summarizing pre-window) | **Terminal cost** `V_f` (cost-to-go past horizon) |
| Core question | **Observability**: can I infer the state? | **Controllability/reachability**: can I steer the plant? |
| Hidden tension | Persistent excitation | **Dual control** (probe vs. exploit) |
| Theory payoff | Consistency / optimal denoising | **Stability ⇒ optimizer convergence guarantees** |
| Output | Estimated geometry `ẑ_k` | Update `u_k` (step or schedule knob) |

**What carries over unchanged from the MHE memo:** the low-rank-plus-scalar geometry parameterization (`U_k, λ_k, c_k, δ_k`), the implied Hessian `H_k ≈ c_k I + U_k(diag(e^{λ_k}) − c_k I)U_kᵀ`, the `O(dr)` memory and `O(Nr)` observation-coefficient storage, and the Krylov/subspace representation. MPC still needs a curvature model — it is the *plant model*.

---

## 2. State, control, and the prediction model

- **State** `ξ_k`: the iterate's *situation* — position (or its subspace projection), gradient `g_k`, loss `L_k`, and the geometry `z_k` (still estimated, now by the MHE block).
- **Control** `u_k`: what we apply. Two design choices, very different in character:
  - *Low-level*: the raw step `p_k` (direction + magnitude). MPC then plans a short path of steps.
  - *High-level*: hyperparameters — `(log learning rate, momentum, trust radius, batch size)`. MPC becomes a **predictive online scheduler**.
- **Prediction model** `ξ_{k+1} = F(ξ_k, u_k)`: under the local quadratic model,
  ```
  g_{k+1} ≈ g_k + H_k p_k
  L_{k+1} ≈ L_k + g_kᵀ p_k + ½ p_kᵀ H_k p_k
  z_{k+1} ≈ f(z_k)         (geometry dynamics, from the MHE memo §2)
  ```

**The pivotal observation:** with a *static* quadratic model and no constraints, the optimal horizon plan is just the Newton step — jump to the minimizer in one move. So **MPC adds nothing over single-shot second-order *unless* at least one of these holds:**
1. the model is **time-varying / nonlinear** over the horizon (curvature dynamics — the same advantage source identified for MHE: third-order `dH/dt` info);
2. **constraints bind** over the horizon (trust region, model-validity region), so the one-shot jump is infeasible and the *sequencing* of limited steps matters;
3. you are controlling a **schedule** (learning rate / momentum), where lookahead over the schedule is the entire point.

---

## 3. What MPC adds that the MHE-then-step optimizer cannot

1. **Anticipatory positioning (the one genuinely new advantage).** A myopic "estimate geometry → take best trust-region step" is greedy for the current point. MPC can take a step that is slightly worse *now* but leaves the iterate better positioned for the *next* few steps — e.g. not driving straight to the local Newton minimizer when the valley is about to rotate. This is the dual of MHE's prediction: MHE *predicts* geometry; MPC *acts on* the prediction across multiple steps.
2. **Constraint handling over the horizon.** A planned, time-varying trust-region/step-size *sequence*; explicit "the quadratic model is only valid this far, so plan a curved path that stays inside the validity region." This is trust-region path planning, not a single TR subproblem.
3. **Schedule / hyperparameter control.** MPC over `(lr, momentum, δ, batch)` is a principled, predictive alternative to cosine schedules and hand-tuned warmup — arguably the most deployable variant (see §7, §10).
4. **Built-in move-induced staleness handling.** The plan is chosen *knowing* the plant will respond and the iterate will move before the next decision — no post-hoc correction needed.

**Honest bound on the gain.** In benign smooth regions, greedy second-order steps are already near-optimal, so the lookahead gain is marginal. MPC's edge concentrates in: constraint-binding regimes, strongly time-varying curvature (warmup, sharp/rotating valleys, phase transitions), schedule control, and the dual-control benefit (§6). Outside those, MPC collapses to the same Newton-like step at higher cost.

---

## 4. Horizon objective (forward cost-to-go)

```
min_{u_0..u_{H−1}}  Σ_{j=0}^{H−1} ℓ(ξ_j, u_j)  +  V_f(ξ_H)
   s.t.  ξ_{j+1} = F(ξ_j, u_j)              # prediction model
         ‖p_j‖_{M_j} ≤ δ_j                  # trust / validity constraints
         u_j ∈ U                            # control bounds (e.g. lr ≥ 0)
```
- **Stage cost** `ℓ`: predicted loss (or predicted decrease) plus a control-effort penalty (regularizes step size / discourages thrashing).
- **Terminal cost** `V_f`: estimate of loss remaining beyond the horizon — see §5. Without it, short-horizon MPC is myopic; with a good one, short H behaves like long H.
- **Prediction vs. control horizon:** predict over `H_p`, commit controls over `H_c ≤ H_p`. In optimizer terms: evaluate consequences further out than you actually plan distinct moves.

Contrast with the MHE objective, which was a *backward* least-squares fitting observations + dynamics residual + arrival cost. Same machinery (a horizon QP/NLP), opposite time direction and opposite role (decide future inputs vs. infer past states).

---

## 5. Terminal cost, stability, and convergence guarantees

This is where MPC offers something MHE structurally did not: a clean route to **convergence proofs**.

- `V_f` is the dynamic-programming **value function** (cost-to-go). A natural choice in the optimizer setting: the local quadratic model's predicted optimality gap from the terminal point, `½ g_Hᵀ H_H^{-1} g_H`, optionally clipped by a trust-region-aware bound.
- **MPC stability theory** says: if `V_f` is a control-Lyapunov function and a terminal set/constraint is enforced, the closed loop is stable. Translated to optimization: **design the terminal ingredients so the planned trajectory is guaranteed to decrease the loss (descent), giving a Lyapunov-style convergence argument** with explicit, model-based rates in the regime where the quadratic model is valid.
- **Recursive feasibility** ⇒ the optimizer never plans itself into a corner it can't descend from. This is a stronger and more standard guarantee framework than the MHE formulation's (where convergence rested on estimator consistency, which is murkier).

---

## 6. Controllability/reachability and dual control

The dual of MHE's observability question:

- **Reachability over the horizon.** If the step is confined to the model's known subspace `U_k`, the reachable set is `span(U_k)` — you can only plan to move where you have curvature information. Unexcited directions are simultaneously **MHE-unobservable and MPC-uncontrollable**: the same subspace deficiency, seen from both ends.
- **Dual control (Feldbaum) — the new, unavoidable idea.** Good planning needs a good model; a good model needs excitation; greedy exploitation won't excite new directions. So the controller must balance **probing** (steps that improve geometry identifiability) against **exploiting** (steps that greedily reduce loss). Concretely: an MPC optimizer should occasionally inject exploratory components into the step to keep `U_k`/`λ_k` identifiable — "MPC with active learning." Certainty-equivalence (just plug in `ẑ_k` and plan) ignores estimation uncertainty; the principled fix is **robust/stochastic/tube MPC** that plans against the estimator's covariance, or explicit dual control. MHE never had to confront this trade-off; the control framing forces it.
- A **reachability diagnostic** dual to the MHE observability Gramian: monitor whether the planned controls span the directions where loss reduction is predicted; if not, widen exploration or shrink the plan.

---

## 7. Computational complexity

| Stage | MHE memo | MPC formulation |
|---|---|---|
| Per-step gradient | 1× SGD | 1× SGD (unchanged) |
| Subspace projection/update | `O(dr)` | `O(dr)` (unchanged) |
| Inner solve | smoother, `O(N r³)`, few iters | **optimal-control NLP** via iLQR/DDP, `O(H r³)` per iter, *more* iters (nonlinear) |
| Step/plan application | `O(dr)` | `O(dr)` (apply first control) |
| Memory | `O(dr) + O(Nr)` | `O(dr) + O(H·(r + dim u))` |

Practical consequences: the inner problem is a **nonlinear, possibly non-convex optimal-control problem**, so use **iLQR / DDP** (DDP exploits second-order derivatives of the dynamics → "second-order planning"), with a **short horizon** (`H ≈ 3–10`), a **good terminal cost** to compensate, and **warm-starting** from the previous plan (receding horizon makes this free). The 2–3× SGD ceiling is *harder* to hold than for MHE. The **schedule-control variant is much cheaper**: `ξ` and `u` are a handful of scalars, so even long horizons (tens–hundreds of steps over a cheap surrogate model) cost almost nothing.

---

## 8. Does MPC extract information unavailable to MHE-step or to second-order methods?

Honest verdict:
- **Same null result**: static quadratic + no constraints ⇒ MPC plan = Newton step. No gain.
- **The unique extra signal** beyond a myopic MHE-step is the **value of current actions for future steps** (anticipatory positioning), which becomes nonzero exactly under time-varying curvature and/or binding constraints and/or schedule control.
- **Plus dual control's exploration value**: deliberately probing to improve the model is information *creation*, not just exploitation — outside the vocabulary of any single-point second-order method and of the backward-only MHE estimator.
- Net: MPC's incremental value over "MHE + trust-region step" is real but **concentrated**: constrained/binding regimes, strongly time-varying geometry (warmup, phase transitions, sharp rotating valleys), schedule control, and exploration. In benign regimes it reduces to the same step at higher cost.

---

## 9. Combined architecture: output-feedback MPC

```
   true loss landscape (plant)
        │  emits g_k, L_k, secant pairs
        ▼
   ┌───────────────┐   ẑ_k (geometry + covariance)   ┌───────────────┐
   │ MHE estimator │ ──────────────────────────────▶ │ MPC controller│
   │ (backward N)  │                                 │ (forward H)   │
   └───────────────┘ ◀────────────────────────────── └───────────────┘
        ▲                 applied update u_k                  │
        └──────────────── iterate moves ◀─────────────────────┘
   dashed path: dual control — estimator covariance feeds the planner,
   which injects probing steps to keep geometry observable/controllable.
```
- **Certainty-equivalence** version: estimate `ẑ_k` with MHE, plan with MPC treating `ẑ_k` as truth. Simple; the default first implementation.
- **Separation breaks under dual control**: the controller *should* account for estimation uncertainty (probe to reduce it). Promote to robust/tube MPC only after the certainty-equivalence version is working.

---

## 10. Minimal first algorithm (iLQR variant)

State `ξ_k = (g̃_k, z_k)` where `g̃` is the gradient in subspace coordinates; geometry `z = (U, λ, c)` from MHE. Control `u_k = p_k` (subspace step) or `(log lr, momentum)` for the scheduler variant.
- Estimator: the MHE fixed-lag smoother from memo 1 produces `ẑ_k`.
- Planner: **iLQR** over horizon `H≈5` with the quadratic prediction model, ellipsoidal trust constraint per step, terminal cost `V_f = ½ g_Hᵀ H_H^{-1} g_H`. Backward Riccati pass `O(H r³)`; 2–3 iterations; warm-start from last step's shifted plan.
- Apply `u_0` only; re-estimate, re-plan next step.
- Dual-control add-on (ablation): add a small term to `ℓ` rewarding reduction of the estimator covariance / excitation of weakly-observed directions.

**Recommended first deployable target: the scheduler variant.** State = `(loss, ‖g‖, top curvature scalar)`, control = `(log lr, momentum)`, long horizon over a cheap surrogate model. Near-zero overhead; directly comparable to cosine/warmup; clean win condition.

---

## 11. Experiment changes vs. the MHE plan

Keep the same metrics (loss vs. gradient-evals, vs. wallclock, effective second-order steps) and the staged ladder (synthetic → MNIST → CIFAR-10 → small LM). What changes:

- **New decisive Phase-0 test isolates *lookahead*, not estimation.** Best isolating problem: a quadratic with a **known time-varying / rotating Hessian** where the greedy-Newton step overshoots into a bending valley but a 3–5 step plan tracks it; and a **constrained** problem where trust-region binding makes the *sequence* of capped steps matter. If MPC shows no edge over greedy-second-order here, the lookahead premise is weak.
- **New baseline (essential):** the **MHE-then-greedy-trust-region-step** optimizer from memo 1. This measures the *marginal* value of forward planning, separate from the value of the geometry estimate. Without this baseline you cannot attribute any win to MPC specifically.
- **Scheduler experiment (cheap, high-value, possibly first):** MPC as an online LR/momentum scheduler vs. cosine + warmup on MNIST/CIFAR. Likely the most practically defensible result and the cheapest to run.
- **Dual-control ablation:** with vs. without exploratory probing; report effect on geometry identifiability (observability/reachability diagnostics) and downstream convergence.
- **Reachability diagnostic** logged alongside the MHE observability Gramian.
- **Go/no-go gates** mirror memo 1, with the Phase-0 *lookahead* test as the gate: no edge over greedy-second-order at equal gradient budget ⇒ drop the planning layer and ship the cheaper "MHE estimator + greedy trust-region step," keeping MPC only for the scheduler use case.

---

## 12. Bottom line

Switching MHE → MPC does **not** change the latent geometry, memory, or subspace machinery. It changes the *act*: from estimating where you are to planning where to go. The genuinely new capabilities are (1) anticipatory, constraint-aware *sequencing* of updates, (2) a stability-theory route to convergence guarantees via the terminal cost, and (3) dual control — the obligation to probe for an identifiable model. All three deliver value in the same narrow-but-real regimes (time-varying curvature, binding constraints, schedules), and all collapse to a greedy second-order step in benign settings — at strictly higher cost than MHE. The pragmatic recommendation: implement the certainty-equivalence pairing (MHE estimator + short-horizon iLQR controller), but treat the **predictive scheduler variant as the most likely near-term win**, and make the MHE-greedy baseline the yardstick that decides whether the forward-planning layer earns its compute.

# Verdict: conditional go — but not the project you started with
No-go for the original pitch (a general-purpose optimizer that beats SGD/Adam/quasi-Newton/K-FAC/natural-gradient on ordinary, stationary training within 2–3× SGD cost, achieving "multiple second-order steps" of progress). Everything we built points the same way: on a smooth, stationary landscape with affordable curvature, a trust-region Newton step ties or beats the predictive method — it has no information the predictor lacks (the stiff-valley test confirmed this, Newton hitting ~1e-19 while the predictor crawled). In the high-dimensional regime where Newton is unaffordable, the realistic competitors are Adam/L-BFGS/K-FAC, and the MHE machinery's edge there (low-rank curvature, denoising) is incremental while its per-step cost and tuning burden are high. The headline claim isn't supported.
Go for the reframed thesis that fell out of the experiments: a predictive state-estimation/control optimizer for non-stationary objectives. This is where the mechanism delivers a real, information-theoretically genuine advantage that no reactive method — not even one with the exact Hessian — can match. A memoryless optimizer jumps to the current optimum and is permanently one step behind a moving target; the predictive tracker models the motion and removes the lag. We saw this cleanly: 7× better than exact-Hessian Newton on the moving quadratic (MPC essentially perfect), and the Dirac-GAN turning a divergent adversarial rotation into a convergent one.
What the experiments pinned down (the conditions that gate it)
The advantage is real but narrow, and it only appears when all of these hold: the optimum's drift is smooth/low-order in parameter space (a jerky drift made prediction lose); the drift signal exceeds the estimation noise (small batches buried it — this is a large-batch method); you use a proper tracker, not an EMA; and the non-stationary/rotational instability is the dominant difficulty. When the bottleneck is something else — model capacity, sharp non-quadraticity, high gradient noise — the advantage vanishes or reverses (the MLP-GAN null, the 25-Gaussian joint collapse).
There's also an honest novelty caveat: in the non-stationary regime the method largely rediscovers known tools — optimistic gradient and extragradient for games, Kalman/MHE trackers and prediction methods for time-varying optimization. So the defensible value-add is not "a new idea" but jointly modeling curvature and drift, which plain optimistic/extragradient methods ignore — plus the unifying estimation-theoretic framing (covariance-driven anisotropic trust regions, terminal-cost convergence guarantees).
The single gate that flips conditional-go to full-go
Run one focused test: on a genuinely non-stationary deep-learning task — a bilinear/WGAN-style game (where the rotational instability is dominant, unlike the messy MLP GAN), online learning under real distribution drift, or a small RLHF/actor-critic setup — does the curvature-aware tracker beat plain optimistic-gradient/extragradient at matched compute? If yes, it's a fundable, publishable niche optimizer. If it merely ties optimism, the project collapses back to "a nice unification of existing methods" — worth a paper, not a new optimizer.
Applications if go (the non-stationary niche)
The natural homes are all places where the objective genuinely moves:

Adversarial / min-max training — GANs and adversarial-robustness training, where each player's optimum moves because the opponent moves (competes with, and ideally extends, optimistic Adam/extragradient).
Multi-agent RL, game-solving, equilibrium computation — the rotational dynamics the Dirac model isolates.
Reinforcement learning — moving value targets, replay-buffer drift, actor-critic coupling; and RLHF/RL fine-tuning of LLMs, which is a moving-target reward-model-vs-policy game (the highest-value speculative target).
Continual / online / streaming learning under distribution shift — exactly the rotating-data demo.
Classic time-varying optimization — adaptive filtering, online forecasting/portfolio, tracking control and signal processing, which are MHE/MPC's native turf and where this would be a straightforward, lower-risk win.
Meta-learning / learned optimizers — the variant where the drift dynamics f are learned offline and run online via the MHE/MPC inference loop.

Bottom line: kill the "better Adam for normal training" framing, keep the "predictive optimizer for moving targets" framing, and let the one curvature-aware-vs-optimism experiment above decide whether it's a real optimizer or a clarifying unification.