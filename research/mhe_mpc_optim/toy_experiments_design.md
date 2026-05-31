# Toy experiments: where MHE / MHE-MPC beats SGD/Adam/Newton/L-BFGS

## The honest headline
On a **smooth, stationary** problem with a cheap exact Hessian, trust-region
Newton is essentially unbeatable — a predictive optimizer has no information it
lacks (verified: it solved a stiff curved valley to ~1e-19). The engineered
hardness that gives MHE/MPC a genuine edge is **non-stationarity**: the optimum
*moves*. Reactive methods (incl. Newton/L-BFGS) jump to the *current* optimum but
have no model that it is moving, so they lag permanently. MHE estimates the
optimum's velocity and predicts the next one; MPC adds a constant-turn model to
follow curved drift.

## Experiment 1 — moving optimum (quadratic), 2D
L_k(w) = 1/2 (w - c_k)^T M (w - c_k), fixed anisotropic M, optimum c_k on a circle.
All of Newton/MHE/MPC use the SAME exact curvature M, so the only difference is
the motion model.

| method | mean tracking error (after warmup) |
|---|---|
| SGD (tuned) | 1.06 |
| Adam (tuned) | 0.84 |
| L-BFGS | 1.61 |
| Newton-TR (exact H, reactive) | 0.45  (= one-step lag) |
| MHE (predict next, linear) | 0.067 |
| MHE-MPC (constant-turn) | ~1e-6 |

MHE's residual (0.067) is exactly the centripetal second-difference R*omega^2;
MPC's turn model satisfies c_{k+1}=R(omega)c_k so it tracks essentially perfectly.
Figures: fig1_trajectories.png, fig2_tracking_error.png, fig3_noisy.png.

## Experiment 2 — density estimation / generative p(x), 2D
Datasets (samplers + scatter in figG1_datasets.png): two moons, 8 gaussians,
two spirals, pinwheel, checkerboard, ring.

(a) Stationary fit — an energy-based model p(x) ~ exp(-E), E = gamma||x||^2 +
sum_j theta_j phi_j(x) with fixed RBF features, trained by **score matching**
(convex-quadratic in theta; the SM curvature G IS the Hessian, so reactive-Newton
is realistic). Recovers the crescents and the ring (figG2_stationary_fits.png).

(b) Non-stationary online density estimation (a stand-in for continual learning
under distribution drift and the moving-target dynamics of adversarial/EBM
training): the data **rotates** during training. To keep the optimum's drift
*smooth* (a precondition — see below) the model is a Gaussian mixture whose means
must follow the rotating modes, so each optimal mean travels on a circle.
Judged on the one-step-ahead (already-drifted) target:

| method | mean mean-tracking error |
|---|---|
| EM (reactive, jumps to current optimum) | 0.60  (full one-step lag) |
| Adam (momentum = crude implicit velocity) | 0.25 |
| MHE (explicit velocity prediction) | 0.22 |
| MHE-MPC (constant-turn) | 0.11 |

Both predictive methods beat both baselines; MPC's turn model is best; the purely
reactive method carries the full lag. Figures: figG3_density_follow.png
(MPC density stays locked on the rotating data, Adam trails), figG4_param_tracking.png.

## Preconditions discovered while building this (the honest caveats)
1. **Stationary + cheap exact Hessian ⇒ no advantage.** Newton ties or wins.
2. **The drift must be smooth/low-order in parameter space.** On a *fixed* RBF
   grid, theta*(t) jerks as modes cross centres, so low-order prediction overshoots
   and loses to a reactive estimate. Switching to a mixture-of-means model (smooth
   circular optimum) is what made prediction win. Observability/smoothness is a
   real requirement, not a detail.
3. **Small-batch noise can bury the drift signal.** With a 256-sample minibatch the
   noisy optimum estimate swamped the drift and MPC's 2nd-difference even amplified
   it; moving to a larger batch (lower noise) restored the advantage. This is the
   large-batch regime the design notes predicted as MHE's home.
4. **Use a proper tracker, not an EMA.** A naive EMA both lags the position and
   leaves velocity noisy; an alpha-beta(-gamma) filter (separate position/velocity/
   acceleration gains) is what cleanly predicts ahead.

## Files
- mhe_demo.py — moving-optimum quadratic tracking (Exp 1).
- gen_demo.py — datasets + EBM/score-matching stationary density fits (Exp 2a).
- gen_gmm.py — rotating-data GMM tracking (Exp 2b).
