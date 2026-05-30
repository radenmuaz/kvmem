"""Generative tracking that succeeds: a Gaussian-mixture density whose component
means must follow a rotating 8-mode dataset. Each optimal mean travels on a CIRCLE
(smooth constant-turn motion), so the predictive optimizer's motion model applies
and the learned density stays locked on the rotating data while Adam trails and
the reactive (EM) baseline lags by one step."""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)
K, SIG, RAD = 8, 0.32, 3.0
N, OMG, BATCH = 90, 0.07, 1500
base_ang = np.arange(K) * (2*np.pi/K)
base_ring = RAD * np.stack([np.cos(base_ang), np.sin(base_ang)], 1)   # K,2

def rot(b): c, s = np.cos(b), np.sin(b); return np.array([[c, -s], [s, c]])
def sample(n, beta):
    lab = rng.integers(0, K, n)
    return (base_ring[lab] + SIG*rng.standard_normal((n, 2))) @ rot(beta).T
def true_means(beta): return base_ring @ rot(beta).T                  # moving optimum

def resp(X, M):                                                       # E-step (n,K)
    d2 = ((X[:, None, :] - M[None, :, :])**2).sum(-1)
    logr = -0.5*d2/SIG**2
    logr -= logr.max(1, keepdims=True)
    r = np.exp(logr); return r / r.sum(1, keepdims=True)
def em_obs(X, M0, iters=6):                                           # current-optimum estimate
    M = M0.copy()
    for _ in range(iters):
        r = resp(X, M); w = r.sum(0)
        M = (r.T @ X) / np.maximum(w[:, None], 1e-6)
    return M
def nll_grad(X, M):                                                  # grad wrt means
    r = resp(X, M)
    return -(np.einsum('nk,nkd->kd', r, X[:, None, :] - M[None, :, :]) / SIG**2) / len(X)

betas = OMG*np.arange(N)
Mstar = np.array([true_means(b) for b in betas])                     # current optimum
Mstar_next = np.array([true_means(b + OMG) for b in betas])           # the target a step-t update will face
def terr(traj): return np.linalg.norm((traj[1:] - Mstar_next).reshape(N, -1), axis=1)

def run_adam(lr, b1=0.9, b2=0.99, eps=1e-8):
    M = true_means(0.0) + 0.1*rng.standard_normal((K, 2)); out = [M.copy()]
    m = np.zeros((K, 2)); v = np.zeros((K, 2))
    for t in range(N):
        g = nll_grad(sample(BATCH, betas[t]), M)
        m = b1*m+(1-b1)*g; v = b2*v+(1-b2)*g*g
        M = M - lr*(m/(1-b1**(t+1)))/(np.sqrt(v/(1-b2**(t+1)))+eps); out.append(M.copy())
    return np.array(out)

def run_em_reactive():
    M = true_means(0.0).copy(); out = [M.copy()]
    for t in range(N):
        M = em_obs(sample(BATCH, betas[t]), M); out.append(M.copy())
    return np.array(out)

def run_track(mpc=False, alpha=0.6, beta=0.2, gamma=0.04):
    M = true_means(0.0).copy(); out = [M.copy()]
    x = M.copy(); vel = np.zeros((K, 2)); acc = np.zeros((K, 2)); init = True
    for t in range(N):
        xp = x + vel + (0.5*acc if mpc else 0.0)
        obs = em_obs(sample(BATCH, betas[t]), xp)                    # observe current optimum
        if init:
            x = obs.copy(); init = False; target = obs.copy()
        else:
            r = obs - xp
            if mpc:
                x = xp + alpha*r; vel = vel + acc + beta*r; acc = acc + gamma*r
                target = x + vel + 0.5*acc
            else:
                x = xp + alpha*r; vel = vel + beta*r
                target = x + vel
        M = target; out.append(M.copy())
    return np.array(out)

best = None
for lr in np.geomspace(0.02, 1.5, 14):
    e = terr(run_adam(lr))[20:].mean()
    if best is None or e < best: best, alr = e, lr
adam = run_adam(alr); em = run_em_reactive(); mhe = run_track(False); mpc = run_track(True)
print(f"Adam lr={alr:.3e}")
for nm, tr in [("Adam", adam), ("EM (reactive)", em), ("MHE", mhe), ("MHE-MPC", mpc)]:
    print(f"{nm:16s} mean mean-tracking error = {terr(tr)[20:].mean():.4f}")

# density grid
gx = np.linspace(-5, 5, 120); GXg, GYg = np.meshgrid(gx, gx)
GP = np.stack([GXg.ravel(), GYg.ravel()], 1)
def density(M):
    d2 = ((GP[:, None, :] - M[None, :, :])**2).sum(-1)
    p = np.exp(-0.5*d2/SIG**2).sum(1); return (p/p.sum()).reshape(GXg.shape)

# Fig G4: tracking error
fig, ax = plt.subplots(figsize=(9, 5))
for nm, tr, col in [("Adam", adam, "#378ADD"), ("EM-reactive (current optimum)", em, "#BA7517"),
                    ("MHE (predict next)", mhe, "#1D9E75"), ("MHE-MPC (constant-turn)", mpc, "#534AB7")]:
    ax.semilogy(np.arange(1, N+1), terr(tr)+1e-6, color=col, lw=2.0, label=nm)
ax.set_xlabel("training iteration (data rotating)"); ax.set_ylabel(r"mean tracking error $\|M_t-M^*_t\|$")
ax.set_title("Generative GMM under distribution drift: predictive optimizer follows, reactive lags")
ax.legend(fontsize=9); fig.tight_layout(); fig.savefig("/home/claude/figG4_param_tracking.png", dpi=140); plt.close(fig)

# Fig G3: density following the rotating data
snaps = [6, 30, 56, 82]
fig, axes = plt.subplots(2, len(snaps), figsize=(4*len(snaps), 8.2))
for j, t in enumerate(snaps):
    Xd = sample(1200, betas[t] + OMG)
    for row, (nm, tr) in enumerate([("Adam", adam), ("MHE-MPC", mpc)]):
        ax = axes[row, j]
        ax.imshow(density(tr[t+1]), origin="lower", extent=[-5, 5, -5, 5], cmap="magma", aspect="equal")
        ax.scatter(Xd[:, 0], Xd[:, 1], s=2, alpha=0.28, color="#7FE7C4")
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_xticks([]); ax.set_yticks([])
        if row == 0: ax.set_title(f"iter {t}")
        if j == 0: ax.set_ylabel(nm, fontsize=13)
fig.suptitle("Learned p(x) (heatmap) vs current rotating data (green): Adam trails, MHE-MPC stays locked on", y=0.99)
fig.tight_layout(); fig.savefig("/home/claude/figG3_density_follow.png", dpi=140); plt.close(fig)
print("G3, G4 done")
