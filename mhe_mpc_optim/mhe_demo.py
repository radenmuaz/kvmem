"""
Engineered NON-STATIONARY problem where SGD/Adam/Newton/L-BFGS are slow or lag,
but an MHE-style tracker (and its MPC extension) succeed.

THE HONEST MECHANISM
--------------------
On a smooth STATIONARY problem with a cheap exact Hessian, trust-region Newton is
unbeatable -- a predictive method has no information it lacks. So the engineered
hardness is NON-STATIONARITY: the optimum MOVES along a curved path.

  At iteration k the objective is   L_k(w) = 1/2 (w - c_k)^T M (w - c_k)
  with a fixed anisotropic curvature M and a moving optimum c_k that travels on a
  circle (constant-turn motion).

Why each method struggles:
  - SGD / Adam: anisotropy forces a small step, so they never catch the moving
    target -> large persistent lag; on the curve, Adam's momentum points along a
    stale tangent and overshoots.
  - Newton-TR / L-BFGS: MEMORYLESS & REACTIVE. With the exact Hessian, Newton
    jumps onto the *current* optimum c_k each step -- but the target has already
    moved to c_{k+1}. It is permanently one step behind. No amount of curvature
    information removes this lag, because the lag is about MOTION, not curvature.

The MHE / MPC fix (given the SAME exact curvature M, for a fair test):
  - Each step yields an estimate of the current optimum: c_hat_k = w_k - M^{-1} g_k.
  - MHE runs a fixed-lag smoother over c_hat history -> estimates target velocity,
    predicts c_{k+1} (constant-velocity), and steps THERE. Lag ~ 0.
  - MPC additionally estimates the TURN RATE and integrates the curved motion to
    intercept c_{k+1} on the arc -> removes the chord error MHE's linear
    extrapolation leaves on a curved path.
This isolates the core thesis: predictive info (the optimum's motion) that is
unavailable to a reactive second-order step, even one with the exact Hessian.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# ----------------------------- Problem -------------------------------
ALPHA = np.deg2rad(30.0)
Rrot = np.array([[np.cos(ALPHA), -np.sin(ALPHA)], [np.sin(ALPHA), np.cos(ALPHA)]])
M = Rrot @ np.diag([8.0, 1.0]) @ Rrot.T          # fixed anisotropic curvature
Minv = np.linalg.inv(M)
RAD, OMEGA, TH0 = 3.0, 0.15, 0.0                 # moving-optimum circle
CENTER = np.array([0.0, 0.0])
W_START = np.array([4.0, -2.5])
N = 150

def target(k):
    th = TH0 + OMEGA * k
    return CENTER + RAD * np.array([np.cos(th), np.sin(th)])

def grad(w, k, noise, rng):
    g = M @ (w - target(k))
    if noise: g = g + noise * rng.standard_normal(2)
    return g

def loss(w, k):
    d = w - target(k)
    return 0.5 * d @ M @ d

def track_err(traj):
    return np.array([np.linalg.norm(traj[k] - target(k)) for k in range(len(traj))])

# --------------------------- Trust region ----------------------------
def tr_solve(g, H, delta):
    if np.linalg.norm(g) < 1e-14: return np.zeros(2)
    d, Q = np.linalg.eigh(H); gt = Q.T @ g; dmin = d.min()
    if dmin > 1e-10:
        p = -Q @ (gt / d)
        if np.linalg.norm(p) <= delta: return p
    lo = max(0.0, -dmin) + 1e-9; hi = lo + 1.0
    pn = lambda l: np.linalg.norm(gt / (d + l))
    while pn(hi) > delta and hi < 1e12: hi *= 2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if pn(mid) > delta: lo = mid
        else: hi = mid
    return -Q @ (gt / (d + 0.5 * (lo + hi)))

# ----------------------------- Methods -------------------------------
def run_sgd(lr, n, noise=0.0, seed=0):
    rng = np.random.default_rng(seed); w = W_START.copy(); traj = [w.copy()]
    for k in range(n):
        w = w - lr * grad(w, k, noise, rng); traj.append(w.copy())
    return np.array(traj)

def run_adam(lr, n, noise=0.0, seed=0, b1=0.9, b2=0.999, eps=1e-8):
    rng = np.random.default_rng(seed); w = W_START.copy(); m = np.zeros(2); v = np.zeros(2)
    traj = [w.copy()]
    for k in range(n):
        g = grad(w, k, noise, rng); m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g*g
        mh = m/(1-b1**(k+1)); vh = v/(1-b2**(k+1))
        w = w - lr*mh/(np.sqrt(vh)+eps); traj.append(w.copy())
    return np.array(traj)

def run_newton(n, noise=0.0, seed=0, delta=8.0):
    rng = np.random.default_rng(seed); w = W_START.copy(); traj = [w.copy()]
    for k in range(n):
        g = grad(w, k, noise, rng); w = w + tr_solve(g, M, delta); traj.append(w.copy())
    return np.array(traj)

def run_lbfgs(n, m=8, noise=0.0, seed=0):
    rng = np.random.default_rng(seed); w = W_START.copy(); k = 0
    g = grad(w, k, noise, rng); S, Y = [], []; traj = [w.copy()]
    for k in range(n):
        q = g.copy(); al = []
        for s_i, y_i in zip(reversed(S), reversed(Y)):
            a = (s_i@q)/(y_i@s_i+1e-12); al.append(a); q = q - a*y_i
        gamma = (S[-1]@Y[-1])/(Y[-1]@Y[-1]+1e-12) if Y else 0.3
        z = gamma*q
        for (s_i, y_i), a in zip(zip(S, Y), reversed(al)):
            be = (y_i@z)/(y_i@s_i+1e-12); z = z + (a-be)*s_i
        d = -z; gd = g@d
        if gd >= 0: d = -g; gd = g@d
        t = 1.0; Lc = loss(w, k)
        for _ in range(30):
            if loss(w+t*d, k) <= Lc + 1e-4*t*gd: break
            t *= 0.5
        wn = w + t*d; kn = min(k+1, n-1); gn = grad(wn, kn, noise, rng)
        s_i = wn - w; y_i = gn - g
        if s_i@y_i > 1e-10:
            S.append(s_i); Y.append(y_i)
            if len(S) > m: S.pop(0); Y.pop(0)
        w, g = wn, gn; traj.append(w.copy())
    return np.array(traj)

def run_mhe(n, mpc=False, noise=0.0, seed=0, delta=8.0, ema=1.0):
    """Estimate current optimum c_hat = w - M^{-1} g, smooth it, predict next."""
    rng = np.random.default_rng(seed); w = W_START.copy(); traj = [w.copy()]
    c_s = None; v_s = np.zeros(2); omega_hat = 0.0; v_prev = None
    for k in range(n):
        g = grad(w, k, noise, rng)
        c_obs = w - Minv @ g                     # observed current optimum (noisy)
        if c_s is None:
            c_s = c_obs.copy(); pred = c_obs.copy()
        else:
            c_s = (1-ema)*c_s + ema*c_obs        # fixed-lag smoothing (denoise)
            v_new = c_s - c_prev_s
            v_s = (1-ema)*v_s + ema*v_new
            if v_prev is not None:               # constant-turn rate estimate
                cr = v_prev[0]*v_s[1] - v_prev[1]*v_s[0]
                dot = v_prev@v_s
                dth = np.arctan2(cr, dot)
                omega_hat = 0.7*omega_hat + 0.3*dth
            v_prev = v_s.copy()
            if mpc:                              # follow the arc (turn model)
                ca, sa = np.cos(omega_hat), np.sin(omega_hat)
                Rw = np.array([[ca, -sa], [sa, ca]])
                pred = c_s + Rw @ v_s
            else:                                # linear extrapolation
                pred = c_s + v_s
        c_prev_s = c_s.copy()
        p = pred - w; nrm = np.linalg.norm(p)
        if nrm > delta: p *= delta/nrm
        w = w + p; traj.append(w.copy())
    return np.array(traj)

# ------------------------------ Tuning -------------------------------
def best_lr(runner, grid):
    best = None
    for lr in grid:
        tr = runner(lr); e = track_err(tr)[n_warm:].mean()
        if np.isfinite(e) and (best is None or e < best):
            best, blr, btr = e, lr, tr
    return blr, btr

n_warm = 30
sgd_lr, sgd_tr = best_lr(lambda lr: run_sgd(lr, N), np.geomspace(1e-3, 0.25, 20))
adam_lr, adam_tr = best_lr(lambda lr: run_adam(lr, N), np.geomspace(1e-2, 1.0, 20))
new_tr = run_newton(N); lb_tr = run_lbfgs(N)
mhe_tr = run_mhe(N, mpc=False); mpc_tr = run_mhe(N, mpc=True)

print(f"tuned SGD lr={sgd_lr:.3e}  Adam lr={adam_lr:.3e}   (target speed ~ {RAD*OMEGA:.3f}/step)")
rows = [("SGD", sgd_tr), ("Adam", adam_tr), ("Newton-TR (exact H)", new_tr),
        ("L-BFGS", lb_tr), ("MHE", mhe_tr), ("MHE-MPC", mpc_tr)]
for name, tr in rows:
    e = track_err(tr)[n_warm:]
    print(f"{name:20s} mean tracking error (after warmup) = {e.mean():.4f}")

# ------------------------------ Figures ------------------------------
methods = [
    ("SGD",                 sgd_tr, "#888780"),
    ("Adam",                adam_tr, "#378ADD"),
    ("Newton-TR (exact H)", new_tr, "#BA7517"),
    ("L-BFGS",              lb_tr, "#993556"),
    ("MHE (predict next)",  mhe_tr, "#1D9E75"),
    ("MHE-MPC (arc intercept)", mpc_tr, "#534AB7"),
]
ths = np.linspace(0, OMEGA*N, 400)
circ = CENTER[:, None] + RAD*np.vstack([np.cos(ths), np.sin(ths)])

# snapshot contours at k=0 as faint backdrop
xs = np.linspace(-5, 5, 400); ys = np.linspace(-5, 4.5, 400)
Xg, Yg = np.meshgrid(xs, ys)
c0 = target(0)
D = np.stack([Xg - c0[0], Yg - c0[1]], -1)
Z = 0.5*np.einsum('...i,ij,...j->...', D, M, D)

fig, ax = plt.subplots(figsize=(11.5, 8))
ax.contour(Xg, Yg, Z, levels=np.geomspace(0.05, Z.max(), 12), colors="0.8", linewidths=0.7)
ax.plot(circ[0], circ[1], "k--", lw=1.4, alpha=0.8, label="path of moving optimum $c_k$")
for k in range(0, N, 25):
    ck = target(k); ax.plot(*ck, "k.", ms=6, alpha=0.5)
for name, tr, col in methods:
    ax.plot(tr[:, 0], tr[:, 1], color=col, lw=2.2, alpha=0.95, label=name)
ax.plot(*W_START, "k^", ms=12, label="start")
ax.set_aspect("equal"); ax.set_xlim(-5, 5); ax.set_ylim(-5, 4.5)
ax.set_xlabel("parameter $w_1$"); ax.set_ylabel("parameter $w_2$")
ax.set_title("Chasing a moving optimum: reactive methods lag, predictive methods stay on target")
ax.legend(loc="upper left", fontsize=9, framealpha=0.92)
fig.tight_layout(); fig.savefig("/home/claude/fig1_trajectories.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
for name, tr, col in methods:
    ax.semilogy(track_err(tr) + 1e-6, color=col, lw=2.0, label=name)
ax.set_xlabel("iteration"); ax.set_ylabel("tracking error  $\\|w_k - c_k\\|$")
ax.set_title("Tracking error vs iteration (lower = better)")
ax.legend(fontsize=9, ncol=2)
fig.tight_layout(); fig.savefig("/home/claude/fig2_tracking_error.png", dpi=140); plt.close(fig)

# noisy case: mean over seeds
NOISE = 3.0; SEEDS = range(12)
def mean_err(fn):
    es = [track_err(fn(s)) for s in SEEDS]
    L = min(map(len, es)); return np.mean([e[:L] for e in es], axis=0)
noisy = [
    ("SGD", mean_err(lambda s: run_sgd(sgd_lr, N, NOISE, s)), "#888780"),
    ("Adam", mean_err(lambda s: run_adam(adam_lr, N, NOISE, s)), "#378ADD"),
    ("Newton-TR", mean_err(lambda s: run_newton(N, NOISE, s)), "#BA7517"),
    ("L-BFGS", mean_err(lambda s: run_lbfgs(N, 8, NOISE, s)), "#993556"),
    ("MHE", mean_err(lambda s: run_mhe(N, False, NOISE, s, ema=0.3)), "#1D9E75"),
    ("MHE-MPC", mean_err(lambda s: run_mhe(N, True, NOISE, s, ema=0.3)), "#534AB7"),
]
fig, ax = plt.subplots(figsize=(9, 5))
for name, e, col in noisy:
    ax.semilogy(e + 1e-6, color=col, lw=2.0, label=name)
ax.set_xlabel("iteration"); ax.set_ylabel("tracking error (mean of 12 seeds)")
ax.set_title(f"Noisy gradients ($\\sigma$={NOISE}): smoothing also denoises the target estimate")
ax.legend(fontsize=9, ncol=2)
fig.tight_layout(); fig.savefig("/home/claude/fig3_noisy.png", dpi=140); plt.close(fig)
print("figures written")
