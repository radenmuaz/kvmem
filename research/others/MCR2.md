# Closed-Loop Rate-Reduction Language Modeling — Implementation Handover

**Status:** research proposal / first implementation spec.
**Lineage:** MCR² (Yu et al. 2020) → ReduNet (Chan et al. 2022) → CTRL (Dai et al. 2022) → Parsimony & Self-Consistency (Ma, Tsao, Shum 2022) → CRATE (Yu et al. 2023). This doc ports the CTRL closed-loop idea to next-token (AR) and masked (NAR/MaskGIT-style) language modeling.

> **Read this first.** The goal here is **not** to beat a cross-entropy (CE) LM on perplexity — it almost certainly will not, since CE directly maximizes likelihood and our objective does not. The payoff is a **structured, inspectable latent** (between-subspace = token, within-subspace = context), a **unified generative + discriminative** model, anti-collapse diversity, and a training story that is better-conditioned than a GAN. Build Phase 1 first; it is a CE-cost baseline. Phase 2 is a higher-risk refinement.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $V$ | vocabulary, size $\lvert V\rvert$ |
| $x_{1:T}$ | token sequence, $x_t \in V$ |
| $d$ | feature dimension |
| $f_\theta$ | encoder (transformer); causal for AR, bidirectional for NAR |
| $g_\eta$ | decoder (transformer); training-only, **discarded at inference** |
| $z_t \in \mathbb{R}^d$ | unit-norm contextual feature; AR: $z_t=f_\theta(x_{\le t})$ predicts $x_{t+1}$ |
| $Z=[z_1,\dots,z_m]\in\mathbb{R}^{d\times m}$ | feature matrix, **columns are samples** |
| $\Pi$ | class membership; AR class label of $z_t$ is the **next token** $x_{t+1}$ |
| $U_c\in\mathbb{R}^{d\times p}$ | frozen orthonormal anchor basis for token $c$ (subspace $S_c$) |
| $\epsilon$ | coding precision (hyperparam, e.g. $\epsilon^2=0.5$) |

Features are L2-normalized per column before any rate computation.

---

## 2. Objective

### 2.1 Coding rate

$$
R(Z) \;=\; \tfrac12 \log\det\!\Big(I_d + \tfrac{d}{m\epsilon^2}\,Z Z^\top\Big).
$$

### 2.2 Per-class compression rate

With class sets $\{Z_c\}$, $m_c=\lvert Z_c\rvert$, $\gamma_c=m_c/m$:

$$
R^c(Z\mid\Pi)\;=\;\sum_c \gamma_c \cdot \tfrac12\log\det\!\Big(I_d + \tfrac{d}{m_c\epsilon^2}\,Z_c Z_c^\top\Big).
$$

### 2.3 MCR² (encoder maximizes — "expand whole, compress each class")

$$
\boxed{\;\Delta R(Z\mid\Pi)\;=\;R(Z)\;-\;R^c(Z\mid\Pi)\;}
$$

### 2.4 Self-consistency distance (closed loop)

For real $Z_c$ and decoded-then-re-encoded $\hat Z_c$:

$$
\Delta R(Z_c,\hat Z_c)\;=\;R(Z_c\cup\hat Z_c)\;-\;\tfrac12\big(R(Z_c)+R(\hat Z_c)\big).
$$

### 2.5 Closed-loop encoding and full game

Decoder outputs **soft token embeddings** (keeps everything differentiable, no CE, no hard sampling):

$$
h_{\theta,\eta}(X)\;=\;f_\theta\big(g_\eta(f_\theta(X))\big),\qquad \hat Z = h_{\theta,\eta}(X).
$$

$$
\boxed{\;\min_{\eta}\max_{\theta}\;\; \underbrace{\Delta R(Z)}_{\text{expand + discriminate}}
\;+\; \underbrace{\Delta R(\hat Z)}_{\text{compress decoded}}
\;+\; \underbrace{\sum_c \Delta R(Z_c,\hat Z_c)}_{\text{align real vs decoded}}\;}
$$

**Phase 1 (open loop)** drops $g_\eta$ entirely and optimizes only $\max_\theta \Delta R(Z\mid\Pi)$.

---

## 3. Class geometry (anchors)

Use **frozen** anchors by default. Inference logit for token $c$ is the subspace energy:

$$
\text{score}_c(z) = \lVert U_c^\top z\rVert^2 \quad\xrightarrow{p=1,\ \lVert z\rVert=1}\quad \langle z, u_c\rangle .
$$

Choice order:

1. **Frozen near-orthogonal frame (default).** Small/medium $\lvert V\rvert$: fixed simplex ETF. Large $\lvert V\rvert$: fixed Hadamard / random $\pm1$ (ECOC) code. Byte-level ($\lvert V\rvert=256$): a $256\times256$ Hadamard gives **exactly orthogonal** binary anchors. Connects to neural-collapse fixed-classifier results; removes the moving-target instability and saves the $\lvert V\rvert\times d$ output-head parameters.
2. **EMA prototypes.** Anchor $=$ EMA of class features. Use if the frozen frame underfits.
3. **Fully learned subspaces.** Most faithful, least stable; only with large batch.

Note: true orthogonality needs $d \ge \sum_c \dim S_c$. For $\lvert V\rvert=65\text{k}$ in modest $d$ this is impossible — rely on near-orthogonal packing, or **factorize** the vocab (e.g. $65\text{k}\approx 2\times256$) with one anchor frame per factor and a factorized prediction. Byte-level sidesteps the whole issue and is the recommended first experiment.

---

## 4. Algorithms

### 4.1 Rate functions

```python
def coding_rate(Z, eps):                      # Z: (d, m), unit-norm columns
    d, m = Z.shape
    cov = (d / (m * eps**2)) * (Z @ Z.T)      # (d, d)
    return 0.5 * logdet(I(d) + cov)

def compress_rate(Z, labels, eps):
    d, m = Z.shape; R = 0.0
    for c in unique(labels):                  # only classes present in the batch
        Zc = Z[:, labels == c]; mc = Zc.shape[1]
        R += (mc / m) * 0.5 * logdet(I(d) + (d / (mc * eps**2)) * (Zc @ Zc.T))
    return R

def delta_R(Z, labels, eps):                  # encoder MAXIMIZES this
    return coding_rate(Z, eps) - compress_rate(Z, labels, eps)

def delta_R_pair(Zc, Zc_hat, eps):
    Zu = cat([Zc, Zc_hat], dim=1)
    return coding_rate(Zu, eps) - 0.5 * (coding_rate(Zc, eps) + coding_rate(Zc_hat, eps))

def consistency(Z, Z_hat, labels, eps):
    return sum(delta_R_pair(Z[:, labels==c], Z_hat[:, labels==c], eps)
               for c in unique(labels))
```

### 4.2 Transformer blocks

```python
def block(h, mask):                           # pre-norm
    h = h + MHSA(LN(h), attn_mask=mask)
    h = h + MLP(LN(h))
    return h

class Encoder:        # f_theta ; accepts token ids OR soft embeddings (for re-encode)
    def forward(self, tokens=None, embeds=None, kind='ar'):
        h = (E_in[tokens] if embeds is None else embeds) + pos
        mask = causal_mask if kind == 'ar' else None
        for b in self.blocks: h = block(h, mask)
        return normalize(out_proj(LN(h)), dim=-1)     # (T, d), unit norm

class Decoder:        # g_eta ; latent -> soft token embedding ("data space"); training only
    def forward(self, z, kind='ar'):
        h = in_proj(z); mask = causal_mask if kind == 'ar' else None
        for b in self.blocks: h = block(h, mask)
        return out_proj(h)                            # (T, d_embed) soft reconstruction
```

### 4.3 Phase 1 — open loop (cost ≈ CE LM; fully parallel, teacher-forced)

```python
def train_step_phase1(tokens):                # tokens: (B, T)
    z = encoder(tokens, kind='ar')[:, :-1]    # z_t predicts x_{t+1}
    labels = flatten(tokens[:, 1:])           # class = next token
    Z = flatten(z).T                          # (d, M), M = B*(T-1)
    loss = -delta_R(Z, labels, eps)           # maximize delta_R
    loss.backward(); opt_enc.step()
```

### 4.4 Phase 2 — closed loop (parallel GDA; NO autoregressive rollout)

```python
def features(tokens):
    Z     = encoder(tokens, kind='ar')[:, :-1]            # f(X)
    e_hat = decoder(Z, kind='ar')                         # g(f(X)): soft-embed of CURRENT tok
    Z_hat = encoder(embeds=e_hat, kind='ar')[:, :-1]      # h(X) = f(g(f(X)))
    return flatten(Z).T, flatten(Z_hat).T

def utility(Z, Z_hat, labels):
    return delta_R(Z, labels, eps) + delta_R(Z_hat, labels, eps) \
         + consistency(Z, Z_hat, labels, eps)

def train_step_phase2(tokens):
    labels = flatten(tokens[:, 1:])
    # (1) encoder ASCENDS utility (decoder frozen): expand + discriminate
    freeze(decoder); Z, Z_hat = features(tokens)
    (-utility(Z, Z_hat, labels)).backward(); opt_enc.step()
    # (2) decoder DESCENDS utility (encoder frozen): compress + align decoded to real
    freeze(encoder); Z, Z_hat = features(tokens)
    (+utility(Z, Z_hat, labels)).backward(); opt_dec.step()
```

All three passes in `features` are **parallel over positions** (teacher-forced input). No per-token loop in training.

### 4.5 Inference — AR (decoder discarded; KV-cached; normal-LM cost)

```python
def subspace_scores(z, U):                    # U: (|V|, d) for p=1 anchors
    return U @ z                              # <z, u_c> ; or ||U_c^T z||^2 for p>1

def sample(logits, tau=1.0, top_p=0.9):
    p = softmax(logits / tau)
    return categorical_sample(nucleus_filter(p, top_p))

def generate_ar(prompt, n, U, tau=1.0, top_p=0.9):
    seq, cache = prompt, init_cache()
    for _ in range(n):
        z = encoder.step(seq[-1], cache)      # cached last-position latent
        tok = sample(subspace_scores(z, U), tau, top_p)
        seq = cat([seq, tok])
    return seq                                # greedy = argmax of scores
```

### 4.6 Inference — masked / MaskGIT (closest to image CTRL)

```python
def cos_schedule(s, steps):                   # fraction to commit, MaskGIT-style
    return cos(pi/2 * (s + 1) / steps)

def generate_masked(length, U, steps=10, tau=1.0):
    seq = [MASK] * length
    for s in range(steps):
        z = encoder(seq, kind='nar')          # bidirectional; all positions
        cand, conf = {}, {}
        for t in masked_positions(seq):
            p = softmax(subspace_scores(z[t], U) / tau)
            cand[t] = categorical_sample(p); conf[t] = p[cand[t]]
        k = ceil(length * (1 - cos_schedule(s, steps)))   # commit more over time
        for t in topk_by(conf, k): seq[t] = cand[t]       # rest stay MASK
    return seq                                # steps=1 -> one-shot ; steps>1 -> iterative
```

**Multi-token prediction (MTP):** give the encoder $K$ heads $z^{(+1)},\dots,z^{(+K)}$, each with its own frozen anchor frame; predict in parallel via `subspace_scores` per head (cross-offset coherence is weaker, as in any MTP head).

---

## 5. The current-vs-next tension (must handle)

The latent must (a) be classified by the **next** token and (b) let the decoder reconstruct the **current** token. MCR² assigns one partition, so resolve by making subspaces **multi-dimensional** ($p>1$): between-subspace identity = next token; within-subspace principal components carry current token + context. This is exactly CTRL's "attributes as within-class PCs," repurposed. Do **not** use $p=1$ if you keep Phase 2.

---

## 6. Recommended first experiment

**Byte-level masked CTRL.** Why: 256 classes → exactly orthogonalizable (Hadamard) → MCR² guarantees intact; fixed-size bidirectional block (e.g. 1024 bytes) → matches image CTRL's fixed-object assumption; parallel training and cheap iterative inference. It exercises every load-bearing assumption at once, with the subspace-assignment step (the most likely failure point) under the easiest conditions.

Config sketch: $d=256$ or $512$, $p=8$ subspace dim, $256$-byte Hadamard anchors, 6-layer encoder + 6-layer decoder, batch as large as fits (stability depends on it), $\epsilon^2=0.5$, GDA with equal LR.

---

## 7. What to measure (and honest expectations)

| Axis | Baseline | Expectation for this scheme |
|---|---|---|
| Perplexity / bits-per-byte | CE LM | comparable at best, likely **worse** |
| Training stability | CE LM | Phase 1 ≈ CE; Phase 2 **less** stable (minimax) |
| Training stability | GAN | Phase 2 **more** stable (closed-form utility, no separate $d$, no prior matching) |
| Diversity / anti-degeneration | CE sampling | plausibly **better** (expansion term fights collapse) |
| Inference speed (AR) | CE LM | **parity** (decoder discarded, KV cache) |
| Inference speed (masked) | AR LM | **faster** (parallel, ~8–12 steps) — inherited from MaskGIT |
| Latent structure / interpretability | CE LM | **clearly better** (explicit subspaces + PCs) |
| Calibration / OOD | CE LM | plausibly better (subspace residual) — unproven |

Diagnostics to log: per-class $\Delta R$ and the two rate terms separately; block-diagonality of $\lvert Z^\top \hat Z\rvert$ (self-consistency health, cf. CTRL Fig. 4); subspace incoherence $\max_{i\ne j}\lVert U_i^\top U_j\rVert$; nearest-subspace accuracy vs a CE head on held-out tokens; GDA gap (utility under encoder-step vs decoder-step) to watch for cycling.

---

## 8. Risk register

- **Minimax non-convergence / cycling** (Phase 2). Mitigate: large batch, equal LR, monitor GDA gap; fall back to Phase 1 if it diverges.
- **Streaming distribution shift.** Causal features' distribution changes as positions accrue; the masked/fixed-size variant avoids this — prefer it for the first build.
- **Combinatorial vocab vs orthogonality budget.** Use byte-level or factorized vocab; do not attempt full near-orthogonality for large $\lvert V\rvert$ in small $d$.
- **Soft-embedding re-encode instability.** The $f(g(f(X)))$ path can amplify; consider Gumbel-softmax / straight-through if soft embeddings drift, and stop-grad the non-active player each GDA sub-step.
- **Likelihood mismatch.** If a calibrated likelihood is needed downstream, add a light CE anchor $-\lambda\,\mathbb{E}[\log \hat p(x_{t+1}\mid z_t)]$ as a hybrid; treat rate reduction as the structural regularizer.

---

## 9. Build order

1. Rate functions + unit tests (verify $\Delta R \ge 0$, correct gradients, log-det stability).
2. Encoder + frozen Hadamard anchors; **Phase 1** byte-level training; compare nearest-subspace accuracy and bits-per-byte to a CE byte LM of equal size.
3. Add Decoder + **Phase 2** closed loop (masked variant); watch the diagnostics in §7.
4. MaskGIT inference loop; sweep `steps` (1 = one-shot vs 8–12 iterative).
5. Only then: AR variant, MTP heads, vocab factorization.