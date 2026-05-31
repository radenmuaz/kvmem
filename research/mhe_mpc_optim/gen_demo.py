"""
Density estimation / generative modelling of p(x) on 2D datasets, and where an
MHE/MPC-style optimizer helps.

MODEL: energy-based model  p(x) ~ exp(-E(x)),  E(x) = gamma||x||^2 + sum_j theta_j phi_j(x)
with fixed Gaussian RBF features phi_j (centres on a grid). Trained by SCORE
MATCHING, whose objective is convex-quadratic in theta:
        J(theta) = 1/2 theta^T G theta + (c - d)^T theta + const
so score matching itself HANDS US the curvature G (the Hessian). That makes the
"reactive Newton" baseline realistic (it really has the exact curvature), and any
predictive-optimizer win comes only from modelling how the optimum MOVES.

STATIONARY data  -> Newton solves it; predictive methods only tie it (honest).
NON-STATIONARY data (distribution drift / the moving-target nature of adversarial
& EBM training) -> the optimal parameters theta*(t) MOVE. Reactive optimizers lag;
MHE predicts theta*(t+1), MPC uses a curved (2nd-order) extrapolation. The learned
density visibly follows the drifting data for MHE/MPC and trails it for Adam.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

rng = np.random.default_rng(1)

# --------------------------- 2D toy datasets --------------------------
def ds_two_moons(n):
    t = np.pi * rng.random(n)
    s = rng.integers(0, 2, n)
    x = np.where(s == 0, np.cos(t), 1 - np.cos(t))
    y = np.where(s == 0, np.sin(t), 0.5 - np.sin(t))
    X = np.stack([x, y], 1) * 2.2
    X[:, 0] -= 1.1
    return X + 0.12 * rng.standard_normal((n, 2))

def ds_eight_gaussians(n, r=3.0, s=0.28):
    ang = (rng.integers(0, 8, n)) * (2 * np.pi / 8)
    c = r * np.stack([np.cos(ang), np.sin(ang)], 1)
    return c + s * rng.standard_normal((n, 2))

def ds_two_spirals(n):
    k = np.sqrt(rng.random(n)) * 540 * np.pi / 180
    sgn = rng.integers(0, 2, n) * 2 - 1
    x = sgn * (-k * np.cos(k)) * 0.35
    y = sgn * (k * np.sin(k)) * 0.35
    return np.stack([x, y], 1) + 0.18 * rng.standard_normal((n, 2))

def ds_pinwheel(n):
    rate, nblade = 0.25, 5
    rads = np.linspace(0, 2 * np.pi, nblade, endpoint=False)
    feat = rng.standard_normal((n, 2)) * np.array([0.18, 1.0]) + np.array([1.4, 0.0])
    lab = rng.integers(0, nblade, n)
    ang = rads[lab] + rate * np.exp(feat[:, 0])
    ca, sa = np.cos(ang), np.sin(ang)
    R = np.stack([np.stack([ca, -sa], 1), np.stack([sa, ca], 1)], 1)
    return np.einsum('nij,nj->ni', R, feat) * 1.3

def ds_checkerboard(n):
    x1 = rng.random(n) * 4 - 2
    x2 = rng.random(n) - rng.integers(0, 2, n) * 2 + np.floor(x1) % 2
    return np.stack([x1, x2 * 2], 1) * 1.1

def ds_ring(n, r=2.6, s=0.18):
    t = 2 * np.pi * rng.random(n)
    return r * np.stack([np.cos(t), np.sin(t)], 1) + s * rng.standard_normal((n, 2))

DATASETS = [("two moons", ds_two_moons), ("8 gaussians", ds_eight_gaussians),
            ("two spirals", ds_two_spirals), ("pinwheel", ds_pinwheel),
            ("checkerboard", ds_checkerboard), ("ring", ds_ring)]

# ---- Fig G1: the dataset zoo ----
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, (name, fn) in zip(axes.ravel(), DATASETS):
    X = fn(4000)
    ax.scatter(X[:, 0], X[:, 1], s=3, alpha=0.4, color="#534AB7")
    ax.set_title(name); ax.set_aspect("equal"); ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Engineered 2D datasets for density estimation / generative modelling p(x)", y=0.98)
fig.tight_layout(); fig.savefig("/home/claude/figG1_datasets.png", dpi=140); plt.close(fig)

# ----------------------- RBF energy-based model -----------------------
GRID = np.array([[a, b] for a in np.linspace(-3.6, 3.6, 8) for b in np.linspace(-3.6, 3.6, 8)])
MU = GRID                       # 64 RBF centres
H = 0.75
GAMMA = 0.04                    # fixed confinement so exp(-E) stays integrable
LAM = 2e-3
Mfeat = len(MU)

def feats(X):
    diff = X[:, None, :] - MU[None, :, :]      # n,M,2
    sq = (diff ** 2).sum(-1)                    # n,M
    phi = np.exp(-sq / (2 * H * H))
    gradphi = -(diff) / (H * H) * phi[..., None]  # n,M,2
    lap = phi * (sq / H ** 4 - 2.0 / H ** 2)      # n,M
    return phi, gradphi, lap

def sm_stats(X):
    _, gradphi, lap = feats(X)
    a = 2 * GAMMA * X                            # n,2
    G = np.einsum('imd,ind->mn', gradphi, gradphi) / len(X)
    c = np.einsum('imd,id->m', gradphi, a) / len(X)
    d = lap.mean(0)
    return G, c, d

def theta_star(X):
    G, c, d = sm_stats(X)
    return np.linalg.solve(G + LAM * np.eye(Mfeat), d - c)

# grid for density rendering
gx = np.linspace(-5, 5, 130); gy = np.linspace(-5, 5, 130)
GXg, GYg = np.meshgrid(gx, gy)
GP = np.stack([GXg.ravel(), GYg.ravel()], 1)
phi_grid = np.exp(-((GP[:, None, :] - MU[None, :, :]) ** 2).sum(-1) / (2 * H * H))
gnorm = (GP ** 2).sum(1)

def density(theta):
    E = GAMMA * gnorm + phi_grid @ theta
    E -= E.min()
    p = np.exp(-E)
    return (p / p.sum()).reshape(GXg.shape)

# ---- Fig G2: stationary fits (sanity + visualization quality) ----
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for col, (name, fn) in enumerate([("two moons", ds_two_moons), ("8 gaussians", ds_eight_gaussians)]):
    X = fn(6000); th = theta_star(X)
    axes[0, col].scatter(X[:2500, 0], X[:2500, 1], s=3, alpha=0.35, color="#534AB7")
    axes[0, col].set_title(f"{name}: data"); axes[0, col].set_aspect("equal")
    axes[1, col].imshow(density(th), origin="lower", extent=[-5, 5, -5, 5], cmap="magma", aspect="equal")
    axes[1, col].set_title(f"{name}: learned p(x)  (EBM + score matching)")
    for r in range(2):
        axes[r, col].set_xlim(-5, 5); axes[r, col].set_ylim(-5, 5)
        axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
fig.tight_layout(); fig.savefig("/home/claude/figG2_stationary_fits.png", dpi=140); plt.close(fig)
print("G1, G2 done")
