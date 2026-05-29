# Continuous Tokenizers, LM Interfaces, and Geometric-State Sequence Models — Implementation Handover

> Comprehensive design doc for three composable contributions. Audience: implementers. Format: equations, hparams, pseudocode, convergence reasoning. Honest about tradeoffs.

---

## TL;DR

You are building three components, each independently useful, that compose into a single architecture:

1. **Continuous tokenizer** — compress chunks of $K$ bytes into a continuous bottleneck, predict the next bottleneck distributionally, decode back to bytes. Competes with BPE+softmax on bandwidth-per-step, with BPE+MTP on bandwidth + joint modeling. The bottleneck *distribution family* is a design choice (Gaussian, FSQ, BSQ, vMF, continuous-Bernoulli, logistic-normal).
2. **LM ↔ tokenizer interface** — three concrete ways to wire the LM to the tokenizer: pure latent autoregression (CALM-style), latent autoregression with re-encoded grounding, or asymmetric byte-embedding-in / latent-out. Tradeoffs are elegance vs robustness.
3. **Geometric-state attention mixers** — the linear-attention covariance-state idea generalized to other exponential families: vMF sphere state (probabilistic linear attention, $O(D)$), Dirichlet simplex state (interpretable concept tracker, $O(D)$), Gaussian (original V1/V2, $O(D^2)$). Drop-in replacements/complements for softmax attention.

**Recommended default to build first:**
- Tokenizer: FSQ or BSQ bottleneck (easiest training, exact likelihood, no posterior collapse)
- Encoder: **causal-over-bytes SSM** with fixed-$K$ latent emission (eliminates chunk-boundary artifacts; see §1.3)
- Interface: Option A-grounded (latent autoregression with re-encoded feedback)
- Mixer: hybrid stack — bulk of layers = vMF sphere mixer, 1–2 softmax attention layers for recall, optional 1 Dirichlet head for global concept tracking
- Decoder: non-causal byte-level, MaskGIT-trained, 1-shot inference (or $T{=}2$ refinement passes if reconstruction degrades)
- Optional refinement: geometric residual stream matched to chosen family (see §3.8)

**BPB reporting:** if using FSQ/BSQ with reconstruction $\ge 99.9\%$, BPB ≈ LM code-cross-entropy alone (no decoder needed at eval time — see §1.7). For continuous bottlenecks, IWAE with 16 samples per token.

**Phase plan:** (1) standalone autoencoder validates the bottleneck; (2) LM in latent space validates the interface; (3) swap softmax for geometric mixers and ablate. Each phase has a clear go/no-go.

---

## 0. Notation

| symbol | meaning |
|---|---|
| $x_t$ | byte chunk at step $t$, $K$ bytes long |
| $K$ | chunk size (bytes per latent), default 8 |
| $z_t \in \mathbb{R}^{d_z}$ | continuous bottleneck latent |
| $d_z$ | latent dimension, default 64–128 |
| $D$ | LM model dim, default 768 |
| $H$ | number of attention/mixer heads |
| $d_{\text{head}} = D/H$ | per-head dimension |
| $L$ | number of LM layers |
| $V$ | vocab size when relevant (256 for bytes) |
| $T$ | sequence length in chunks |
| $\gamma_t \in (0,1)$ | data-dependent forgetting gate |
| $\pi_t$ | mixture weights vector |
| $\sigma(\cdot)$ | sigmoid; $\text{softmax}(\cdot)$ also used |

Conventions: $\odot$ elementwise, $\|\cdot\|$ Euclidean by default. Sufficient statistics $T(x)$ (avoid clash with sequence length $T$ — context disambiguates).

---

## 1. Continuous tokenizer

### 1.1 Architecture overview

Three components, trained jointly (recommended) or in two stages:

```
bytes x_t  --Encoder φ-->  bottleneck z_t  --LM ψ--> distribution p(z_{t+1} | z_{<t})
                                                              |
                                                              v sample
bytes x_{t+1} <-- Decoder θ (non-causal) <-- z_{t+1}
```

The LM is fully autoregressive over latents; the decoder is non-causal over the $K$ byte positions inside a chunk. The latency advantage ($K$-fold fewer sequential steps) survives only if the decoder is parallel — see §1.4.

### 1.2 Bottleneck choices

Each choice is a different way the encoder produces $z_t$ and the LM predicts $z_{t+1}$. All share the modular two-term per-step objective from §1.5:

$$
\mathcal{L}_t = \underbrace{-\mathbb{E}_{c_t \sim \tau_t}[\log p_\theta(x_t \mid c_t)]}_{\mathcal{L}^{\text{rec}}_t} + \beta_t\,\underbrace{\text{KL}(\tau_t \,\|\, \rho_t)}_{\mathcal{L}^{\text{pred}}_t}
$$

where $\tau_t$ is the encoder-induced target distribution over the code, $\rho_t$ is the LM's predictive distribution, and $\beta_t$ is a KL weight (used for warmup or fixed balance).

#### 1.2.1 Gaussian / GMM head

**Encoder output:** $\mu^{\text{enc}}_t \in \mathbb{R}^{d_z}$, $\log\sigma^{\text{enc},2}_t \in \mathbb{R}^{d_z}$, defining $\tau_t = \mathcal{N}(\mu^{\text{enc}}_t, \text{diag}\,\sigma^{\text{enc},2}_t)$. Sample $z_t = \mu^{\text{enc}}_t + \sigma^{\text{enc}}_t \odot \epsilon$, $\epsilon \sim \mathcal{N}(0,I)$ (reparameterization).

**LM head:** $K_{\text{mix}}$-component mixture of Gaussians,
$$
\rho_t = \sum_{k=1}^{K_{\text{mix}}} \pi_t^{(k)}\, \mathcal{N}\!\left(\mu_t^{(k)},\, \Sigma_t^{(k)}\right)
$$
with $\mu_t^{(k)} = h_t W_\mu^{(k)}$, $\log\text{diag}\,\Sigma_t^{(k)} = h_t W_\ell^{(k)}$, $\pi_t = \text{softmax}(h_t W_\pi)$.

**Prediction loss (single-sample KL estimate, reusing the reparameterized $z_t$):**
$$
\mathcal{L}^{\text{pred}}_t = \log \mathcal{N}(z_t; \mu^{\text{enc}}_t, \Sigma^{\text{enc}}_t) - \log \sum_k \pi_t^{(k)} \mathcal{N}(z_t; \mu_t^{(k)}, \Sigma_t^{(k)})
$$

**Pros:** explicit continuous likelihood, calibrated covariance, multimodal predictions.
**Cons:** the hardest to train of all bottlenecks — posterior collapse, variance cheating, mixture symmetries, $\Sigma^{-1}$ ill-conditioning, reparameterization variance. Requires $\beta$ warmup, free bits, entropy regularization on $\pi$, careful init. **Convergence ~2–4× slower than baseline.**

#### 1.2.2 FSQ — finite scalar quantization

**Encoder output:** project to $d_q < 10$ dims, bound, round:
$$
\tilde z_t = \tfrac{L-1}{2}\tanh(u_t W_{\text{fsq}}), \qquad \hat z_t = \tilde z_t + \text{sg}(\text{round}(\tilde z_t) - \tilde z_t)
$$
($\text{sg} = $ stopgrad; straight-through estimator.) $\tau_t = \delta_{\hat z_t}$ (point mass). Implicit codebook size $L^{d_q}$.

**LM head:** product of $d_q$ categoricals over $L$ levels,
$$
\rho_t = \prod_{j=1}^{d_q} \text{Cat}(\cdot_j \mid \text{softmax}(h_t W_j))
$$
**Prediction loss:** per-dimension cross-entropy,
$$
\mathcal{L}^{\text{pred}}_t = -\sum_{j=1}^{d_q} \log \text{softmax}(h_t W_j)\big[\hat z_{t,j}\big]
$$

**Pros:** trains like softmax (convex per-dim CE), no codebook collapse by design, exact discrete likelihood → exact BPB. Default choice. **Convergence ~1.2–1.4× baseline.**
**Cons:** STE bias on encoder gradients (mild for FSQ); factorized prediction misses cross-dim correlations.

**Defaults:** $d_q = 6$, $L = 8$ (codebook $8^6 = 262{,}144$).

**Codebook sizing.** Length-$K$ byte sequences live in a space of $256^K$ possibilities, but the codebook only needs to distinguish sequences that *occur* in your data plus a safety margin. Natural-text entropy is ~1–1.5 bits/byte (English), ~3 bits/byte for code/mixed, up to 8 for compressed/random data. Sizing rule of thumb:
$$
\text{codebook bits} = d_q \log_2 L \approx (\text{entropy per byte}) \cdot K + \text{8–12 bit safety margin}
$$
Recommended configurations:

| data type | $K$ | $(d_q, L)$ | codebook bits | implicit size |
|---|---|---|---|---|
| compact, text-only | 8 | $(6, 8)$ | 18 | $2.6 \times 10^5$ |
| balanced (default) | 8 | $(8, 8)$ | 24 | $1.7 \times 10^7$ |
| code-heavy / multilingual | 8 | $(12, 8)$ | 36 | $6.9 \times 10^{10}$ |
| aggressive bandwidth | 16 | $(10, 8)$ | 30 | $1.1 \times 10^9$ |
| CALM-style (K=4 BPE ≈ 16 bytes) | n/a | $(8, 8)$ | 24 | $1.7 \times 10^7$ |

Validate empirically in Phase 1: if reconstruction ≥ 99.5% holds on the full data distribution (including rare/code/multilingual subsets), the codebook is large enough. If text reconstructs well but code/UTF-8 lags, increase $d_q$ before increasing $L$ (each extra dim is $\log_2 L = 3$ bits at $L{=}8$; each $L$ step from 8→16 only adds 1 bit per dim but doubles the LM head cost).

#### 1.2.3 BSQ — binary spherical quantization

**Encoder output:** project, L2-normalize, sign:
$$
v_t = \frac{u_t W_{\text{bsq}}}{\|u_t W_{\text{bsq}}\|}, \qquad \hat z_t = \tfrac{1}{\sqrt{d_q}}\big(v_t + \text{sg}(\text{sign}(v_t) - v_t)\big)
$$
$\tau_t = \delta_{\hat z_t}$. Codebook $2^{d_q}$.

**LM head:** product of Bernoullis over sign-bits $b_{t,j} = \mathbb{1}[\hat z_{t,j}>0]$:
$$
\rho_t = \prod_{j=1}^{d_q} \text{Bernoulli}\big(b_{t,j} \mid \sigma(h_t w_j)\big)
$$
**Prediction loss:** per-bit BCE.

**Pros:** even simpler than FSQ (binary heads), bounded quantization error, exact BPB, parameter-free implicit codebook. **Convergence ~1.3–1.5× baseline.**
**Cons:** sign-STE coarser than rounding-STE; bit-granularity may need higher $d_q$ for same effective vocab.

**Defaults:** $d_q = 18$ (codebook $2^{18} \approx 262$k).

**Codebook sizing.** Same entropy logic as FSQ but with $L=2$: $\text{codebook bits} = d_q$, so the required dimension equals the required bit budget. Recommended:

| data type | $K$ | $d_q$ | codebook bits |
|---|---|---|---|
| compact, text-only | 8 | 18 | 18 |
| balanced (default) | 8 | 24 | 24 |
| code-heavy / multilingual | 8 | 36 | 36 |
| aggressive bandwidth | 16 | 30 | 30 |

BSQ needs more dimensions than FSQ for the same codebook (because each dim is 1 bit vs $\log_2 L$ bits), but in the **factor-attention variant (§3.6.6)** each dim is 1 bit of KV cache vs FSQ's $\log_2 L$ bits or continuous attention's $\sim 16$ bits — making BSQ the most KV-memory-efficient growing-memory variant by a wide margin. Pick BSQ over FSQ when growing-memory attention is in the stack and cache memory matters; pick FSQ when bottleneck dimensionality matters and cache memory doesn't.

#### 1.2.4 vMF — continuous sphere

**Encoder output:** mean direction $\mu^{\text{enc}}_t$ on $S^{d_z-1}$, concentration $\kappa^{\text{enc}}_t \in \mathbb{R}_{>0}$. $\tau_t = \text{vMF}(\mu^{\text{enc}}_t, \kappa^{\text{enc}}_t)$.

**LM head:** mixture-of-vMF (or single vMF),
$$
\rho_t = \sum_k \pi_t^{(k)}\, \text{vMF}(\mu_t^{(k)}, \kappa_t^{(k)}), \qquad \log\text{vMF}(z;\mu,\kappa) = \kappa\,\mu^\top z + \log C_d(\kappa)
$$
with normalizer $C_d(\kappa) = \kappa^{d/2-1} / \big((2\pi)^{d/2} I_{d/2-1}(\kappa)\big)$ (modified Bessel $I_\nu$).

**Prediction loss:** $-\log\sum_k \pi^{(k)}_t \text{vMF}(z_t; \mu^{(k)}_t, \kappa^{(k)}_t)$, sampled $z_t$.

**Pros:** matches sphere-native LM embeddings, no $\Sigma^{-1}$ blowup (compact support), calibrated concentration. **Convergence ~1.4–1.8× baseline.**
**Cons:** Bessel function stiffness for large $\kappa$ (use accurate approximations or clipping).

**Defaults:** $d_z = 64$, single component during early training, $K_{\text{mix}}{=}4$ later.

#### 1.2.5 Continuous-Bernoulli — continuous cube

**Encoder output:** $\lambda_t \in (0,1)^{d_z}$, per-dimension shape. $\tau_t = \prod_j \text{CB}(\lambda_{t,j})$.
$$
\log \text{CB}(z;\lambda) = z\log\lambda + (1-z)\log(1-\lambda) + \log C(\lambda)
$$
with $C(\lambda)$ a closed-form normalizer (handle removable singularity at $\lambda=0.5$ via Taylor expansion).

**LM head:** product of continuous-Bernoullis (or mixture).

**Pros:** principled continuous FSQ, no STE bias, per-dim factorized → well-conditioned. **Convergence ~1.2–1.4× baseline.**
**Cons:** less standard, fewer reference implementations.

#### 1.2.6 Logistic-normal / Dirichlet — simplex

**Encoder output:** logistic-normal — sample $g \sim \mathcal{N}(\mu^{\text{enc}}_t, \Sigma^{\text{enc}}_t)$ in $\mathbb{R}^{d_z-1}$, set $z_t = \text{softmax}([g, 0])$.

**LM head:** logistic-normal mixture, or Dirichlet (note: Dirichlet has digamma boundary stiffness for small $\alpha$).

**Pros:** "soft distribution over concepts" interpretation, ties naturally with MoE routing.
**Cons:** softmax non-linearity adds curvature; Dirichlet has boundary issues. **Convergence ~1.5–2× baseline.**

#### 1.2.7 Comparison table

| bottleneck | code | head | likelihood | convergence | failure modes |
|---|---|---|---|---|---|
| Gaussian/GMM | continuous, $\mathbb{R}^{d_z}$ | mixture density | IWAE bound | 2–4× | collapse, variance cheat |
| FSQ | discrete grid $L^{d_q}$ | prod. categoricals | exact | 1.2–1.4× | STE bias (mild) |
| BSQ | discrete hypercube $2^{d_q}$ | prod. Bernoullis | exact | 1.3–1.5× | STE bias (coarser) |
| vMF | continuous sphere | mixture vMF | exact | 1.4–1.8× | Bessel stiffness |
| Cont. Bernoulli | continuous $(0,1)^{d_z}$ | prod. CB | exact | 1.2–1.4× | normalizer at 0.5 |
| Logistic-normal | simplex | LN mixture | exact | 1.5–2× | softmax curvature |

**Default pick: FSQ.** Strongest convergence/likelihood/simplicity tradeoff. Move to BSQ if you want spherical structure; to vMF if you want calibrated continuous spherical; to GMM only if calibrated continuous covariance is essential.

### 1.3 Encoder

**Design rationale.** Fixed-$K$ chunking creates an information-boundary problem when the encoder is a non-causal block over $K$ bytes alone: the latent for chunk $t$ has no access to the prior bytes, so the encoder must learn position-shifted copies of the same content ("hello" at chunk offset 3 vs 7 must independently produce a useful latent), and UTF-8 multi-byte characters that straddle a chunk edge get split. The cleanest fix is to keep fixed $K$ as a *latent-emission rate* (predictable shape, easy batching) but make the encoder *causal over the full prior byte stream* so each latent absorbs all context up to its emission point.

**Architecture (causal-over-bytes, fixed-K latent emission):**

```
input:   full byte stream b_1, b_2, ...
embed:   B_i = ByteEmb[b_i] + PosEmb[i]
body:    2-layer causal SSM (Mamba-style)         <-- preferred: O(N) cost, unlimited context
         OR 2-layer causal Transformer            <-- alternative: O(N^2), needs context cap
emit:    every K bytes, project the hidden state at position tK to bottleneck params
         z_t = Proj(h_{tK})    for t = 1, 2, ...
heads:   project to bottleneck-specific parameters (same per-bottleneck heads as before)
```

The encoder sees all bytes up to position $tK$ when emitting $z_t$, so chunk boundaries become *sampling instants*, not information cuts. Pre-chunk context is never lost. Same per-bottleneck head dimensions:

- Gaussian: $2 d_z$ (mean + log-var)
- FSQ: $d_q$ scalars (pre-tanh)
- BSQ: $d_q$ scalars (pre-normalize, sign of normalized)
- vMF: $d_z + 1$ (direction, concentration)
- Cont. Bernoulli: $d_z$ logits → sigmoid → $\lambda$
- Logistic-normal: $d_z + d_z(d_z-1)/2$ (Gaussian mean + Cholesky, smaller in practice)

**Why SSM over causal Transformer:** the encoder benefits from arbitrary lookback (a paragraph's worth of context can inform a single chunk's latent), and a 2-layer Mamba-style SSM gives this at $O(N)$ cost where $N$ is total bytes — strictly better than the Transformer's $O(N^2)$ unless you cap the context window. Encoder stays small (~3M params) and parallel-trains.

**Inference:** the encoder maintains state (SSM scan state, or KV cache for causal Transformer) and emits one latent every $K$ bytes. Sequential at the byte level, but the byte-level cost is small and the LM is sequential at the chunk level anyway. No real latency penalty.

**Decoder stays non-causal one-shot** (or MaskGIT refinement, see §1.4) — it doesn't need access to prior bytes because $z_t$, predicted by the LM from $z_{<t}$, already carries inter-chunk context.

**Optional further hardening:** random chunk-offset augmentation at training time — shift the latent-emission positions by $\delta \in \{0,\ldots,K-1\}$ randomly per batch. Forces the encoder to be offset-robust. Cheap, complements the causal architecture.

**Variable-length latents are a research extension, not a default.** Letting the encoder learn *where* to emit (rather than fixing $K$) — through boundary prediction, learned segmentation, or rule-based (whitespace/sentence-aware) emission — aligns latents with natural units and could give better compression. But it complicates batching, makes boundary prediction its own learning problem, and forces the decoder to also predict chunk length. Reserve for Phase 4 after the fixed-$K$ baseline is validated.

**Size:** ~3M params total. Joint-trained with the LM (recommended); can be stage-wise on raw bytes for cold-start.

### 1.4 Decoder

Non-causal, parallel over $K$ positions, conditioned on $z_t$. Critical: an AR decoder over the bytes destroys the $K$-fold latency advantage. Use one-shot or MaskGIT/diffusion refinement only.

#### 1.4.1 Architecture

```
input:   z_t in R^{d_z}
expand:  z_t -> K slot embeddings (broadcast + K learned pos embeddings)
body:    2-layer non-autoregressive transformer, d=256, heads=4
         OR 3-layer dilated TCN (faster, fine for K<=16)
condition: FiLM(z_t) modulation each layer:  h <- gamma(z) * h + beta(z)
output:  K parallel heads, 256-way softmax each
```

Vocab is 257 (256 bytes + MASK) when training with masked diffusion / MaskGIT.

#### 1.4.2 Training — three options

**(a) One-shot factorized.** Train $p_\theta(x \mid z) = \prod_i p_\theta(b_i \mid z)$ directly:
$$
\mathcal{L}^{\text{rec}}_t = -\sum_{i=1}^K \log p_\theta(b_i \mid z_t)
$$
Simple, but assumes byte-independence given $z$. Works when $z$ is near-sufficient.

**(b) MaskGIT.** Sample mask rate from cosine schedule, mask independently, predict masked positions:
```
rate = cos_schedule(uniform(0,1))   # cosine bias toward higher rates
mask = bernoulli(rate, K)
x_t = where(mask, MASK, x)
loss = cross_entropy(decoder(x_t, z)[mask], x[mask])
```
Unweighted masked CE; heuristic but strong empirically.

**(c) Time-free masked diffusion (MDLM/MD4-style).** Absorbing-state discrete diffusion with no timestep input. ELBO-weighted masked CE:
$$
\mathcal{L} = \mathbb{E}_{t \sim U(0,1)} \mathbb{E}_{x_t}\left[ \frac{-\sigma'(t)}{\sigma(t)} \sum_{i \in M_t} -\log p_\theta(b_i \mid x_t, z) \right]
$$
For linear schedule $\sigma(t)=t$: weight is $1/t$. This gives a proper ELBO → exact BPB. Same architecture as MaskGIT, just different loss weighting.

**Recommendation:** train with (c) for proper BPB. Sample with the MaskGIT confidence scheduler regardless.

#### 1.4.3 Inference samplers

```
# One-shot (T=1)
logits = decoder(MASK^K, z)
bytes = argmax(logits) or sample(logits, temp=0)   # near-deterministic if z is sufficient

# MaskGIT confidence sampler (T = 2..8)
x = [MASK]*K
for s in 1..T:
    logits = decoder(x, z)
    pred   = sample_or_argmax(logits)
    conf   = prob(pred) + gumbel_noise(scale=0.01)
    n_keep = ceil(K * (1 - cos_schedule(s/T)))
    commit top-n_keep highest-conf positions; remask the rest
```

Use $T{=}1$ when $z$ is near-sufficient (>99% reconstruction); $T{=}2{-}4$ when scaling $K$ higher.

### 1.5 Training pipeline (joint, recommended)

Total per-step loss:
$$
\mathcal{L}_t = \mathcal{L}^{\text{rec}}_t + \beta_t\,\mathcal{L}^{\text{pred}}_t + \lambda_\pi\,\text{entropy-reg}(\pi_t) + \lambda_b\,\text{load-balance}(\pi_t)
$$

**Hyperparameters:**
- $\beta_t$: ramp $0 \to 1$ linearly over first 10% of training (Gaussian/vMF/LN only; quantized bottlenecks use $\beta = 1$ throughout).
- Free bits: floor per-dim KL at $\kappa = 0.5$ nats (Gaussian/vMF only).
- $\lambda_\pi = 0.01$ (entropy reg on mixture weights, prevents collapse).
- $\lambda_b = 0.01$ (load balance: penalize $\sum_k (\bar\pi^{(k)} - 1/K)^2$ across batch).
- Decoder noise augmentation: add $\sigma_{\text{aug}} = 0.3$ Gaussian noise to $z$ before decoding during training (smooths the latent space, prevents brittle codes).

**Collapse detection:**
- KL $\to$ 0 *and* reconstruction loss high $\Rightarrow$ encoder collapse, intervene (raise $\beta$ more slowly, raise free bits).
- KL $\to$ 0 *and* reconstruction loss low $\Rightarrow$ healthy convergence (LM is perfectly predicting the latent).

### 1.6 Position vs BPE+softmax and BPE+MTP

This architecture competes on a 2D axis:

| axis | choices |
|---|---|
| base unit | byte / BPE token |
| bandwidth mechanism | none / BPE-merge (cap ~3–4 B/step) / MTP (n parallel heads) / continuous compression (learned, can scale $K$ high) |

**Honest positioning:**
- **vs byte softmax:** continuous wins on bandwidth ($K$-fold) and BPB at matched compute. Easy win, but not the real comparison.
- **vs BPE softmax:** continuous wins on bandwidth (BPE caps ~3–4 B/step). On BPB the contest is closer.
- **vs BPE + MTP (the strong baseline):** this is the fair comparison. MTP gets bandwidth via $n$ parallel softmax heads at ~1× convergence (no VAE overhead). The continuous tokenizer must beat MTP on BPB-at-fixed-compute via the joint decoder modeling within-chunk correlations that MTP's independence misses, OR achieve higher $K$ than MTP can sustain. Otherwise MTP wins on engineering simplicity.

**Composability:** the base unit (byte/BPE) is orthogonal to the bandwidth mechanism. The continuous tokenizer works on top of either; you can also stack continuous + MTP for $K \cdot n$ bytes/step.

### 1.7 Reporting BPB — exact vs. ELBO

The bottleneck choice determines whether bits-per-byte can be reported *exactly* or only as a variational bound. This matters for fair comparison against discrete LM baselines.

**The setup.** With latent codes $c_t$, the byte log-likelihood is
$$
\log p(\text{bytes}_t \mid \text{bytes}_{<t}) = \log \sum_{c_t} p_\theta(\text{bytes}_t \mid c_t)\, p_{\text{LM}}(c_t \mid c_{<t})
$$
The sum over $c_t$ runs over the full codebook — astronomical for $L^{d_q}$ or $2^{d_q}$ — so direct evaluation is intractable.

**Standard ELBO (works for any bottleneck).** Use the encoder as a (variational) posterior $q_\phi(c \mid \text{bytes})$ and bound:
$$
\log p(\text{bytes}_t \mid \text{bytes}_{<t}) \;\ge\; \mathbb{E}_{c \sim q_\phi}\!\left[\log p_\theta(\text{bytes}_t \mid c) + \log p_{\text{LM}}(c \mid c_{<t}) - \log q_\phi(c \mid \text{bytes}_t)\right]
$$
This is your training loss (negated). Reported BPB is an *upper* bound (loose-to-tight depending on the bottleneck).

**FSQ / BSQ — the ELBO is tight when reconstruction is near-lossless.** Because the encoder is deterministic ($q_\phi = \delta_{\hat c}$), $-\log q_\phi$ vanishes and the bound collapses to
$$
\log p(\text{bytes}_t \mid \text{bytes}_{<t}) \;\ge\; \underbrace{\log p_\theta(\text{bytes}_t \mid \hat c_t)}_{\text{decoder reconstruction}} + \underbrace{\log p_{\text{LM}}(\hat c_t \mid \hat c_{<t})}_{\text{LM code-likelihood}}
$$
If reconstruction is near-lossless (e.g., $\ge 99.9\%$, which Phase 1 should hit), the reconstruction term $\approx 0$ and:
$$
\boxed{\;\text{BPB} \;\approx\; -\frac{1}{|\text{bytes}|\,\ln 2}\sum_t \log p_{\text{LM}}(\hat c_t \mid \hat c_{<t})\;}
$$
**The decoder is not needed at evaluation time.** This is a real speedup and is one of the strongest practical reasons to prefer FSQ/BSQ over continuous bottlenecks: your BPB number is just the LM's discrete cross-entropy on codes, directly comparable to BPE-softmax baselines, no IWAE machinery, no bound-looseness to defend.

**Continuous bottlenecks (Gaussian/vMF/CB) — always ELBO/IWAE.** The encoder posterior $q_\phi$ is a continuous distribution (not a delta), so the sum becomes an integral and the bound is genuinely an ELBO:
$$
\text{ELBO}_t = \mathbb{E}_{z \sim q_\phi(\cdot|\text{bytes}_t)}\!\left[\log p_\theta(\text{bytes}_t \mid z) + \log p_{\text{LM}}(z \mid z_{<t}) - \log q_\phi(z \mid \text{bytes}_t)\right]
$$
Single-sample is high-variance; tighten with **IWAE** using $S{=}16$ samples:
$$
\text{IWAE-BPB}_t = -\log\frac{1}{S}\sum_{s=1}^S \frac{p_\theta(\text{bytes}_t \mid z^{(s)})\,p_{\text{LM}}(z^{(s)} \mid z_{<t})}{q_\phi(z^{(s)} \mid \text{bytes}_t)}, \quad z^{(s)} \sim q_\phi
$$
Substantially more expensive (16× decoder + LM evaluations per token). Still an upper bound on true BPB. This is the other practical reason FSQ/BSQ wins on reporting: it sidesteps an entire class of evaluation complexity.

**Summary by bottleneck:**

| bottleneck | BPB | how to compute |
|---|---|---|
| FSQ | tight ELBO; LM-only when recon ≥ 99.9% | LM code-CE + decoder NLL (or LM-only proxy) |
| BSQ | tight ELBO; LM-only when recon ≥ 99.9% | LM bit-BCE + decoder NLL (or LM-only proxy) |
| Gaussian/vMF/CB | ELBO/IWAE always | 16-sample IWAE per token |
| Lossless exact-bijective encoder | exact, no decoder | LM code-CE alone |

**Practical convention.** During training, monitor LM code-CE as your primary signal (it's the fast proxy). For published BPB numbers, run a full ELBO/IWAE pass on the validation set; if FSQ/BSQ and reconstruction $\ge 99.9\%$, the LM-only number is essentially the published one and you can defend it as such. If recon is lower, include the decoder term.

---

## 2. LM ↔ tokenizer interface

How the LM's input and feedback connect to the tokenizer. Three concrete options.

### 2.1 Option A — pure latent autoregression (CALM-style)

**Training:**
$$
\text{input}_t = z_t = \text{Encoder}_\phi(x_t), \qquad \text{LM target}: z_{t+1}
$$
The LM sees encoder latents as input and predicts the next encoder latent. Input space = output space.

**Inference:**
```
z_history = [BOS]
loop:
  dist = LM(z_history)
  z_next ~ dist
  bytes = Decoder(z_next)         # emit to user
  z_history.append(z_next)         # feed sampled latent directly
```

**Pros:** symmetric, minimal, cheapest feedback (no decode+re-encode per step), allows *deferred decoding* (generate all latents first, decode in parallel at end → real latency win).
**Cons:** the LM's sampled $z$ can drift off the encoder-latent manifold over long generations — an *extra* exposure-bias source on top of standard AR drift. The latent is reconstruction-optimized, not prediction-optimized.

### 2.2 Option A-grounded — re-encode the decoded bytes

**Training:** same as A.

**Inference:**
```
z_history = [BOS]
loop:
  dist = LM(z_history)
  z_next ~ dist
  bytes = Decoder(z_next)
  z_grounded = Encoder(bytes)      # re-encode to snap back to manifold
  z_history.append(z_grounded)
```

**Pros:** stays on the encoder-latent manifold, reduces drift to *standard* AR exposure bias (well-understood, manageable). Same training as A.
**Cons:** one extra encoder pass per step (cheap, ~3M params). Still reconstruction-optimized latent. Loses deferred-decoding option (must decode every step).

### 2.3 Option B — asymmetric: byte-embedding input, latent output

**Training:**
$$
\text{input}_t = \text{Pool}(\text{ByteEmb}(x_t)), \qquad \text{LM target}: z_{t+1} = \text{Encoder}_\phi(x_{t+1})
$$
A *separate*, simpler input pathway (byte embed + small pool/MLP) feeds the LM; the output is still a VAE latent for the decoder.

**Inference:**
```
b_history = [BOS bytes]
loop:
  input = Pool(ByteEmb(b_history))
  dist = LM(input)
  z_next ~ dist
  bytes = Decoder(z_next)
  b_history.append(bytes)
```

**Pros:** input representation is *learned end-to-end for prediction* (decoupled from reconstruction). Always grounded in discrete bytes → robust to drift. Input side stays in the easy-to-train softmax-style regime.
**Cons:** asymmetric (input ≠ output space). Two compression pathways (input embed-pool + VAE encoder), some redundancy. Must decode + re-embed every step.

### 2.4 Comparison and recommendation

| | input=output | feedback cost | drift robustness | repr. optimization |
|---|---|---|---|---|
| A | yes | cheapest (latent direct) | weakest (latent drift) | reconstruction |
| A-grounded | yes | +1 encoder pass | standard AR exposure | reconstruction |
| B | no | +1 re-embed | standard AR exposure | prediction |

**Key insight:** with joint training of encoder + LM (§1.5), the latent in A/A-grounded becomes optimized for both reconstruction and prediction simultaneously, recovering B's main advantage while keeping symmetry.

**Recommendation:**
- **Default: A-grounded + joint training.** Best balance.
- **A native:** only if autoencoder is very robust (noise-augmented) and you want deferred decoding for max latency.
- **B:** when joint encoder training is unstable, or you want full decoupling for debug/ablation.

---

## 3. Geometric-state sequence mixers

The linear-attention covariance-state idea extended to other geometries. Replaces or complements softmax attention.

### 3.1 Unifying framework — conjugate exponential-family filtering

**Recognition.** Linear attention's $S_t = \sum_s \phi(k_s) v_s^\top$ and V1/V2's $\Lambda_t = \sum_s \gamma\text{-weighted}\,a_s a_s^\top$ are both *sufficient-statistic accumulators* of conjugate exponential families. The general template:
$$
\tau_t = \gamma_t\, \tau_{t-1} + w_t\, T(\text{obs}_t), \qquad n_t = \gamma_t\, n_{t-1} + w_t
$$
where $T(x)$ is the sufficient statistic of an exponential family, $\gamma_t$ is the V1-style forgetting gate, $w_t$ is the V2-style attention weight, and $n_t$ is the accumulated evidence (= linear attention's normalizer = posterior sample size).

Pick the family $\Rightarrow$ pick the state geometry $\Rightarrow$ pick the mixer.

### 3.2 vMF sphere-state mixer (recommended default)

**Projections (per head, from $x_t \in \mathbb{R}^D$):**
$$
v_t = \frac{x_t W_v}{\|x_t W_v\|} \in S^{d-1}, \quad w_t = \text{softplus}(x_t w_w), \quad \gamma_t = \sigma(x_t w_\gamma)
$$

**State update (parallel-scannable):**
$$
r_t = \gamma_t\, r_{t-1} + w_t\, v_t \in \mathbb{R}^d, \qquad n_t = \gamma_t\, n_{t-1} + w_t \in \mathbb{R}_{>0}
$$

**Readout:**
$$
\hat\mu_t = \frac{r_t}{\sqrt{\|r_t\|^2 + \epsilon}} \in S^{d-1}, \qquad \bar R_t = \frac{\|r_t\|}{n_t + \epsilon} \in [0,1]
$$

With $H$ heads, output is $[\bar R^{(1)}_t \hat\mu^{(1)}_t; \ldots; \bar R^{(H)}_t \hat\mu^{(H)}_t]\,W_o$.

**State cost:** $O(D)$ total (across all heads), $d_{\text{head}}$ × cheaper than linear attention's matrix state.

**Pros:** matches sphere-native embeddings, built-in confidence signal ($\bar R_t$), parallel scan training, **fastest convergence after softmax (~1.1–1.3× softmax steps)**.
**Cons:** no key-query content addressing (compressed summary, not retrieval) — gives up exact recall.

### 3.3 Dirichlet simplex-state mixer

**Projections:**
$$
c_t = \text{softmax}(x_t W_c) \in \Delta^{C-1}, \quad w_t = \text{softplus}(x_t w_w), \quad \gamma_t = \sigma(x_t w_\gamma)
$$

**State update:**
$$
\alpha_t = \gamma_t\, \alpha_{t-1} + w_t\, c_t \in \mathbb{R}^C_{>0}, \quad \alpha_0 = \alpha_{\text{prior}}\,\mathbf{1}
$$

**Readout:**
$$
\bar\theta_t = \frac{\alpha_t}{\mathbf{1}^\top\alpha_t} \in \Delta^{C-1}, \qquad n_t = \mathbf{1}^\top\alpha_t
$$
Output: $\bar\theta_t E_{\text{concept}} W_o$ where $E_{\text{concept}} \in \mathbb{R}^{C \times D}$ is a learned concept embedding table.

**Pros:** most stable mixer (positive accumulation, no normalizer singularity), interpretable as running concept histogram, **fastest convergence to its ceiling (~1.1–1.2×)**.
**Cons:** *low capacity ceiling* (a $C$-dim histogram is coarse). Use as complement, not primary.

**Defaults:** $C = 64$ concepts, single Dirichlet head alongside other mixers.

### 3.4 Gaussian state (V1/V2 — for completeness)

Diagonal-plus-low-rank precision: $\Lambda_t = \text{diag}(D_t) + U_t U_t^\top$, info vector $\eta_t$.
$$
D_t = \gamma_t D_{t-1} + a_t \odot a_t, \quad U_t = \text{trunc}_r([\sqrt{\gamma_t}\,U_{t-1}, a_t]), \quad \eta_t = \gamma_t\eta_{t-1} + u_t
$$
Recover mean via Woodbury: $\mu_t = D^{-1}\eta - D^{-1}U(I_r + U^\top D^{-1}U)^{-1}U^\top D^{-1}\eta$.

**State cost:** $O(Dr)$ (low-rank). Most expensive but models cross-dimension correlation.
**Convergence:** ~1.4× for diag-plus-low-rank with parallel scan; more if using full Cholesky.

### 3.5 Row activations (when keeping softmax-style attention)

If you keep full attention somewhere in the stack, the softmax can be swapped:

| activation | regularizer $\Omega$ | distribution | use case |
|---|---|---|---|
| softmax | Shannon entropy | Boltzmann categorical | default; dense, smooth |
| sparsemax | $\frac12\|p\|^2$ | Tsallis-2 | exact zeros, long context |
| $\alpha$-entmax | Tsallis-$\alpha$ | $q$-exponential | tunable sparsity ($\alpha{=}1.5$ middle) |
| argmax | 0 | point mass | discrete; needs STE |
| sigmoid | per-element | prod. Bernoullis | independent keys, leaves simplex |
| Gumbel-softmax | Shannon + Gumbel | Concrete | stochastic differentiable |

Each is the regularized argmax $p^\star = \arg\max_{p \in \Delta} \langle p, s\rangle - \Omega(p)$ for a different $\Omega$.

### 3.6 Append-memory growing-state variants (factor attention)

The fixed-state mixers in §3.2–3.4 compress the past into a bounded posterior; their content-addressed twins keep the entire history and use softmax attention as the selector. The Gaussian instance is the original V2 ("Gaussian-factor attention" — natural growing-memory analog of the fixed-state GMM block, same probabilistic head, no gating-pressure on history, attention-style exact recall). Every other geometry from §3.1–3.4 gets the same treatment by swapping the sufficient statistic that each past position writes into the KV cache.

#### 3.6.1 Universal template

Pick an exponential family with sufficient statistic $T(\cdot)$. Each past position writes $(T(\text{obs}_s),\, w_s,\, k_s)$ into the cache. At query time:
$$
\alpha_{ts} = \text{softmax}_s\!\left(\frac{q_t^\top k_s}{\sqrt{d_k}}\right) \quad \text{(causal: } s \le t\text{)}
$$
$$
\tau_t = \tau_0 + \sum_{s \le t} \alpha_{ts}\, w_s\, T(\text{obs}_s), \qquad n_t = n_0 + \sum_{s \le t} \alpha_{ts}\, w_s
$$
Attention weights $\alpha_{ts}$ replace the fixed-state recurrence's forgetting gate $\gamma_t$ — *content-addressed selectivity* instead of temporal decay. The aggregated $\tau_t$ is the natural-parameter posterior in the chosen family; read out its mean/density. Per-query compute is $O(T)$ as in standard attention; the cache stores the family's per-token write.

#### 3.6.2 Gaussian-factor attention (original V2)

**Write per position:**
$$
a_s = x_s W_a, \quad u_s = x_s W_u, \quad k_s = x_s W_k
$$
Cache $\{a_s, u_s, k_s\}_{s \le t}$, $O(Td)$ per head.

**Aggregate:**
$$
\Lambda_t = \epsilon I + \sum_s \alpha_{ts}\, w_s\, a_s a_s^\top, \qquad \eta_t = \sum_s \alpha_{ts}\, w_s\, u_s
$$
Treat $A_t = [a_1,\ldots,a_t] \in \mathbb{R}^{d \times t}$ as a rank-$t$ factor; never materialize $\Lambda_t$ as dense.

**Readout via Woodbury** (with $D_\alpha = \text{diag}(\sqrt{w_s\alpha_{ts}})$, $\tilde A = A_t D_\alpha$):
$$
\mu_t = \tfrac{1}{\epsilon}\eta_t - \tfrac{1}{\epsilon}\tilde A\left(I + \tfrac{1}{\epsilon}\tilde A^\top \tilde A\right)^{-1}\tilde A^\top \tfrac{1}{\epsilon}\eta_t
$$
In practice truncate to top-$r$ attention weights → rank-$r$ factor → $O(dr + r^3)$ readout. Multi-head concatenate.

**Cost:** $O(Td)$ attention + $O(dr+r^3)$ readout per query. Cache $O(Td)$.
**Use when:** correlated continuous calibration over retrieved memory matters.

#### 3.6.3 vMF-factor attention (sphere)

**Write per position:** unit direction + scalar vote
$$
v_s = \frac{x_s W_v}{\|x_s W_v\|} \in S^{d-1}, \quad w_s = \text{softplus}(x_s W_w), \quad k_s = x_s W_k
$$

**Aggregate:**
$$
r_t = \sum_s \alpha_{ts}\, w_s\, v_s \in \mathbb{R}^d, \qquad n_t = \sum_s \alpha_{ts}\, w_s
$$

**Readout:**
$$
\hat\mu_t = \frac{r_t}{\sqrt{\|r_t\|^2 + \epsilon}}\;\text{(direction)}, \qquad \bar R_t = \frac{\|r_t\|}{n_t + \epsilon} \in [0,1]\;\text{(confidence)}
$$
Output $\bar R_t \hat\mu_t$ per head, concatenate.

**Cost:** $O(Td)$ — same as standard softmax attention. The architectural diff is small: L2-normalize values to the sphere, expose the confidence signal $\bar R_t$, and aggregate one extra scalar accumulator $n_t$. Implements as a tiny modification to FlashAttention kernels.

**Why it's interesting:** $\bar R_t$ is a *free retrieval-confidence signal*. When the top-attended positions agree in direction, $\bar R_t \to 1$ (sharp retrieval); when they're contradictory or diffuse, $\bar R_t \to 0$. Standard softmax attention silently averages even when its retrieval is incoherent — this exposes that.

#### 3.6.4 Dirichlet-factor attention (simplex)

**Write per position:**
$$
c_s = \text{softmax}(x_s W_c) \in \Delta^{C-1}, \quad w_s = \text{softplus}(x_s W_w), \quad k_s = x_s W_k
$$

**Aggregate:**
$$
\alpha_t = \alpha_{\text{prior}}\mathbf{1} + \sum_s \alpha_{ts}\, w_s\, c_s \in \mathbb{R}^C_{>0}
$$

**Readout:**
$$
\bar\theta_t = \frac{\alpha_t}{\mathbf{1}^\top \alpha_t} \in \Delta^{C-1}, \qquad n_t = \mathbf{1}^\top \alpha_t
$$
Output $\bar\theta_t E_{\text{concept}}$ where $E_{\text{concept}} \in \mathbb{R}^{C \times d}$ is a learned concept embedding table.

**Cost:** $O(TC + Cd)$ per query, cache $O(TC)$.
**Use when:** you want interpretable *content-addressed concept retrieval* — "given this query, which concepts dominate the relevant past?" Pairs naturally with the fixed-state Dirichlet mixer.

#### 3.6.5 FSQ-factor attention (discrete grid)

**Write per position:** FSQ code as per-dim one-hots
$$
\hat z_s \in \{0,\ldots,L-1\}^{d_q}, \quad o_{s,j} = \text{onehot}(\hat z_{s,j}, L), \quad w_s, k_s \text{ as before}
$$
Cache the integer code (or packed bits) — $O(Td_q \log_2 L)$ memory, much smaller than continuous $O(Td)$.

**Aggregate per dim** (independent across $j$):
$$
\alpha_{t,j} = \alpha_{\text{prior}}\mathbf{1} + \sum_s \alpha_{ts}\, w_s\, o_{s,j} \in \mathbb{R}^L_{>0}
$$

**Readout per dim:**
$$
\bar\theta_{t,j} = \frac{\alpha_{t,j}}{\mathbf{1}^\top \alpha_{t,j}} \in \Delta^{L-1}
$$
Concatenate all $d_q$ per-dim distributions, project to model dim with $W_o \in \mathbb{R}^{d_q L \times d}$. Or take the argmax level per dim for a hard read.

**Cost:** $O(Td_q + d_q L)$ per query, cache stores integers only.
**Use when:** the values being aggregated are already discrete (e.g., from an FSQ-tokenized memory), or when KV-cache memory dominates and you can tolerate discrete retrieval granularity.

#### 3.6.6 BSQ-factor attention (binary sphere / hypercube)

**Write per position:** $d_q$ bits
$$
b_{s,j} \in \{0,1\} \text{ for } j=1,\ldots,d_q, \quad w_s, k_s \text{ as before}
$$
Cache $O(Td_q)$ bits — the smallest of any variant.

**Aggregate per bit** (independent):
$$
a_{t,j} = a_0 + \sum_s \alpha_{ts}\, w_s\, b_{s,j}, \qquad \tilde b_{t,j} = b_0 + \sum_s \alpha_{ts}\, w_s\,(1 - b_{s,j})
$$

**Readout per bit:** Beta posterior mean
$$
\bar p_{t,j} = \frac{a_{t,j}}{a_{t,j} + \tilde b_{t,j}} \in [0,1]
$$
Output $\{\bar p_{t,j}\}_{j=1}^{d_q}$, project to model dim. Threshold at 0.5 for hard bits.

**Cost:** $O(Td_q)$ per query.
**Use when:** memory cost is paramount. The most compact growing-memory attention: 1-bit-per-dim KV cache, $d_q$ Bernoulli probabilities per query. Naturally compatible with quantized KV storage.

#### 3.6.7 Continuous-Bernoulli-factor attention (continuous cube)

Same as BSQ-factor but writes are real-valued $z_s \in [0,1]^{d_q}$ (no STE):
$$
a_{t,j} = a_0 + \sum_s \alpha_{ts}\, w_s\, z_{s,j}, \quad \tilde b_{t,j} = b_0 + \sum_s \alpha_{ts}\, w_s\,(1 - z_{s,j})
$$
Beta readout per dim as in BSQ.

**Use when:** you want BSQ's structure but with smooth gradients (no STE bias).

#### 3.6.8 Comparison and selection

| variant | write/posn | cache/posn | readout cost | confidence signal | use case |
|---|---|---|---|---|---|
| Gaussian-factor | $a_s, u_s \in \mathbb{R}^d$ | $O(d)$ continuous | $O(dr+r^3)$ | $\Lambda^{-1}$ trace | calibrated continuous retrieval |
| vMF-factor | $v_s$ on $S^{d-1}$, $w_s$ | $O(d)$ continuous | $O(d)$ | $\bar R_t = \|r_t\|/n_t$ | minimal upgrade to softmax attn |
| Dirichlet-factor | $c_s \in \Delta^{C-1}$ | $O(C)$ continuous | $O(C+Cd)$ | $n_t = \mathbf{1}^\top\alpha_t$ | interpretable concept retrieval |
| FSQ-factor | code in $\{0..L{-}1\}^{d_q}$ | $O(d_q \log L)$ bits | $O(d_q L)$ | per-dim evidence | discrete values, small cache |
| BSQ-factor | $d_q$ bits | $O(d_q)$ bits | $O(d_q)$ | per-bit Beta confidence | smallest KV cache |
| Cont-Bern-factor | $z_s \in [0,1]^{d_q}$ | $O(d_q)$ continuous | $O(d_q)$ | per-bit Beta confidence | smooth BSQ analog |

All share the $O(T)$ attention cost per query (the softmax over the past). The variants differ in *what gets written into the cache* and *how the aggregated state is read out*. In every case the readout has the same probabilistic-posterior form as the matched fixed-state mixer from §3.2–3.4, just computed by content-weighted summation instead of gated recurrence.

#### 3.6.9 Quick guide

- **vMF-factor**: the minimal-effort upgrade to standard softmax attention — same compute, adds a free confidence signal. Drop-in candidate.
- **BSQ-factor**: when KV memory dominates. 1-bit cache; trade some recall fidelity for ~30× smaller cache than continuous attention.
- **Dirichlet-factor**: 1–2 heads per layer for interpretable concept tracking, alongside vMF/softmax heads carrying the precise content.
- **FSQ-factor**: when values are already discrete codes (e.g., aggregating over a tokenized memory bank).
- **Gaussian-factor**: full content-addressed continuous calibration. Use only if needed; the Woodbury readout is the most expensive.

### 3.7 Hybrid stack recommendation

For an $L$-layer model, mix fixed-state (§3.2–3.4) and growing-memory (§3.6) variants by layer:

```
Layer 1..2:           vMF-factor attention   (selective + confidence signal)
Layer 3..L-2:         vMF fixed-state mixer  (cheap O(T) compute, O(D) state)
Layer L-1:            softmax attention OR Gaussian-factor attention
                       (final integration with full recall)
Layer L:              vMF fixed-state mixer + LM head
Optional per attention layer: 1 head = Dirichlet-factor (concept tracking)
Optional very-long context: replace some attention layers with BSQ-factor
                             (1-bit KV cache for the bulk)
```

Three design knobs to set per layer:
1. **Compressed (fixed-state) or selective (growing-memory)?** Selectivity at attention layers where recall matters; compression for the bulk.
2. **Which geometry?** vMF for sphere-native, Dirichlet for interpretable concepts, Gaussian for correlated uncertainty, BSQ/FSQ for compact discrete cache.
3. **Row activation (when softmax-style)?** Softmax default; sparsemax/$\alpha$-entmax for explicit zeroing on long context; sigmoid if keys should fire independently.

This gives selectivity where it matters most, geometric-state cost savings everywhere else, and a single probabilistic-posterior readout style throughout the stack so the LM head sees a coherent representation.

### 3.8 Depth and geometric consistency — what NOT to stack

A tempting idea is to "stack latent distributions deep" by nesting bottlenecks or making activations distributional at every layer. Four interpretations, three of which are bad ideas:

**(A) Hierarchical bottlenecks (nested tokenizers): avoid.** Stack multiple tokenizers (Level 0 = bytes, Level 1 = vMF over 8 bytes, Level 2 = Dirichlet over 4 Level-1 latents, etc.). Theoretically appealing, empirically fragile — each level compounds reconstruction loss, posterior collapse at higher levels and information shortcut at lower levels are notorious, and at LM scale they have never decisively beaten flat architectures. Worth attempting *only* when the timescales are very separated (Level 1 over 8 bytes, Level 2 over 500 bytes for explicit long-context modeling). Within one semantic granularity, nesting compounds harm.

**(B) Distributional activations through depth (BNN-style): avoid at scale.** Pass $(h_t, S_t)$ — mean + covariance — between layers (moment propagation). Doubles or triples per-layer compute. Point-estimate networks have repeatedly beaten BNNs/moment-propagation at matched compute. Useful only locally (propagate uncertainty from one recall-critical block into the next), not stack-wide.

**(C) Geometric mixers stacked with vector residual stream: this is what §3.7 already specifies.** Each block is a distributional mixer (vMF / Dirichlet / Gaussian / softmax) maintaining a probabilistic state internally; the inter-layer signal is a standard vector with residual connections. This gives the "deep distributional" benefit (every layer reasons in a probabilistic geometry) without the cost of moment propagation between layers.

**(D) Consistent geometry across the stack: use this.** Match the geometric family across the tokenizer bottleneck, the bulk mixers, and the LM head. If the bottleneck is vMF, use vMF mixers, predict a vMF (or mixture-of-vMF) at the head. The LM converges faster because there's no representation mismatch at component interfaces — same family, same sufficient statistics, same posterior readout style throughout.

**Optional further refinement — geometric residual stream:** instead of additive residuals in vector space ($h_{l+1} = h_l + \text{block}(h_l)$), use the matched-geometry composition for that family at each residual connection:

| family | residual update |
|---|---|
| sphere (vMF) | $h_{l+1} = \text{normalize}(h_l + \text{block}(h_l))$ — project back to sphere |
| simplex (Dirichlet) | $h_{l+1} = \text{softmax}(\log h_l + \text{block}(h_l))$ — combine in log-space |
| cube (Beta/CB) | $h_{l+1} = \sigma(\sigma^{-1}(h_l) + \text{block}(h_l))$ — combine in logit-space |
| Euclidean (Gaussian) | $h_{l+1} = h_l + \text{block}(h_l)$ — standard residual (no projection needed) |

One line of code per residual connection, gives stability benefits (bounded magnitudes, smooth scale, no representation drift through depth), and aligns activations with the family the model is actually parameterizing. Sphere version is essentially RMSNorm-after-residual, which is already known to work; the simplex and cube versions are the same idea ported to those geometries. Worth ablating in Phase 3 alongside the mixer composition.

**Net guidance:** don't nest bottlenecks (A) or pay for full BNN-style moment propagation (B); do use the geometric-mixer stack from §3.7 (C) with consistent geometry across components (D), and consider the geometric residual stream as a small additional refinement when the stack is committed to one family.

---

## 4. Convergence analysis

### 4.1 The four factors

Convergence rate is governed by:

1. **Temporal gradient flow.** For recurrence $h_t = \gamma_t h_{t-1} + (\cdot)$, gradient signal decays as $\prod_{s'=s+1}^t \gamma_{s'} \approx \bar\gamma^{t-s}$. Initialize $\gamma$ near 1 to extend effective memory $1/(1-\bar\gamma)$. Softmax has *no* such product (direct gradient via attention weight) — its structural advantage.
2. **Normalization conditioning.** Jacobian of the readout normalization sets gradient magnitude. Sphere $1/\|r\|$ (needs $\epsilon$ floor). Simplex $1/n$ (no singularity, always bounded). Linear-attention sum-of-keys denominator *can* vanish (worst). Softmax $\sum e^{s_i} \ge $ #terms (best).
3. **Gradient variance.** Deterministic recurrences: low. Reparameterized continuous (GMM/vMF latent): some. Energy/Monte-Carlo (CALM-energy): high.
4. **Loss landscape.** Mixtures have $K!$ symmetry and dead-component risk. Single-head recurrences are benign. Boundary stiffness (Dirichlet near $\alpha{=}0$, hyperbolic near manifold boundary) creates conditioning issues.

### 4.2 Convergence rate table

Steps to a target loss, relative to softmax + cross-entropy baseline:

| component | rate | dominant cost |
|---|---|---|
| **Baseline: softmax + CE** | 1× | reference |
| **Bottlenecks** | | |
| FSQ | 1.2–1.4× | recon coupling + STE |
| Continuous-Bernoulli | 1.2–1.4× | recon coupling (no STE) |
| BSQ | 1.3–1.5× | sign-STE bias |
| vMF | 1.4–1.8× | Bessel + reparam variance |
| Logistic-normal | 1.5–2× | softmax curvature |
| Dirichlet | 2–2.5× | boundary stiffness |
| Gaussian/GMM | 2–4×, fragile | all 4 factors hit |
| **Mixers (fixed-state)** | | |
| Softmax attention | 1× (steps) | $O(T^2)$ compute |
| vMF sphere state | 1.1–1.3× | gated recurrence + $\epsilon$ floor |
| Dirichlet simplex | 1.1–1.2× (low ceiling) | bounded but coarse |
| Default linear attention | 1.3–1.6× | unstable denominator |
| Gaussian (diag+lowrank) | 1.4× | matrix recurrence |
| **Mixers (growing-memory / factor attention)** | | |
| vMF-factor attention | 1.0–1.1× | as softmax + one extra accumulator |
| BSQ-factor attention | 1.0–1.2× | sign-STE on writes; cheapest cache |
| FSQ-factor attention | 1.1–1.2× | round-STE on writes; per-dim Dirichlet readout |
| Cont-Bern-factor | 1.0–1.1× | smooth writes, Beta readout |
| Dirichlet-factor attention | 1.0–1.2× | concept softmax curvature |
| Gaussian-factor (V2) | 1.2–1.5× | Woodbury readout per query |
| **Decoders** | | |
| One-shot factorized | 1.0× | trivial |
| MaskGIT-trained | 1.1–1.2× | extra mask sampling |
| Time-free diffusion | 1.2–1.3× | $1/t$ weighting noise |

### 4.3 Caveats

- **Rate ≠ destination.** Softmax reaches the best quality on recall; compressed mixers reach a lower ceiling regardless of rate. MSE/zero-variance heads converge fast to the *wrong* (mean-seeking) solution.
- **Steps ≠ wall-clock.** Softmax's $O(T^2)$ compute means at fixed wall-clock, $O(T)$ mixers can complete more steps. Crossover ~ when $T \gtrsim d_{\text{head}}$.
- **All convergence numbers are reasoned from gradient structure, not measured.** Treat as orderings with uncertainty; ablate to confirm.

### 4.4 Composite convergence — full stack

For the recommended default (FSQ bottleneck + A-grounded + vMF mixer + MaskGIT decoder), composite convergence is roughly:
- Per-step rate: ~1.3–1.5× softmax baseline (FSQ ~1.3 + vMF ~1.2, taking the dominant cost)
- Per-step compute: substantially lower than full softmax attention at long context
- Net wall-clock to a target: comparable to or better than BPE+softmax for $T > 2$k tokens, worse for short context

---

## 5. Implementation phases

Build in this order. Each phase has a clear go/no-go.

### Phase 0 — infrastructure (1–2 weeks)

- Data pipeline: byte-level corpus, chunking by $K$, train/val splits.
- Eval harness: BPB on held-out, perplexity-equivalent metric, reconstruction accuracy.
- Reference baselines: byte softmax LM, BPE softmax LM, BPE+MTP. These are your numbers to beat.

**Go/no-go:** baselines reproduce literature numbers within ±5%.

### Phase 1 — standalone autoencoder (1–2 weeks)

Train only encoder + decoder, no LM yet. Tests that the bottleneck choice works.

- Pick FSQ first (simplest).
- Target: reconstruction accuracy > 99.5% on validation bytes at $K{=}8$.
- Eval: BPB of the autoencoder (lower bound for the full system).

**Go/no-go:** FSQ reconstruction > 99.5%. If not, try BSQ; if still not, reconsider $K$ (lower) or $d_q$ (higher).

### Phase 2 — LM in latent space (3–4 weeks)

Add the LM on top of frozen (or jointly-trained) tokenizer. Use Option A-grounded interface.

- Sub-phase 2a: LM with softmax attention everywhere (no geometric mixers yet). Validates the interface independently of the mixer choice.
- Sub-phase 2b: switch tokenizer interface to A-native, then to B. Compare on val BPB and generation quality.

**Go/no-go:** BPB matches or beats BPE+MTP baseline at matched compute. If not, the continuous-tokenizer advantage isn't paying for its complexity; reconsider design.

### Phase 3 — geometric mixers (2–3 weeks)

Replace softmax layers with vMF sphere mixer; ablate hybrid stacks.

- Start: pure vMF stack (all layers vMF). Expect worse recall, similar local quality.
- Then: hybrid (2 softmax + rest vMF). Should match Phase 2 quality at lower compute.
- Add: 1 Dirichlet head per layer. Measure concept tracking benchmarks.

**Go/no-go:** hybrid stack matches softmax-only quality at < 80% wall-clock cost at long context.

### Phase 4 — scale (optional)

If phases 1–3 succeed, scale to 1B parameters and longer context. Mostly engineering at this point.

---

## 6. Hyperparameter reference

### 6.1 Small scale (~100M params, 1M–1B token budget)

```yaml
# Tokenizer
chunk_size_K: 8
latent_dim_dz: 64
bottleneck: fsq
fsq_dims_dq: 6
fsq_levels_L: 8

encoder:
  type: causal_ssm   # Mamba-style; emits latent every K bytes
  layers: 2
  d_model: 256
  byte_context: unlimited   # SSM cost is O(N), context window is unlimited for free
  emit_every_K_bytes: 8     # latent emission rate
  random_offset_aug: true   # shift K-offset by 0..K-1 each batch
  params: ~3M

decoder:
  type: nat        # non-autoregressive transformer
  layers: 2
  d_model: 256
  heads: 4
  vocab: 257       # 256 + MASK
  conditioning: film
  training: maskgit_diffusion
  inference_T: 1   # one-shot at inference

# LM
lm:
  d_model: 768
  layers: 12
  heads: 12
  context: 2048    # latent positions (= 16k bytes of context)
  mixer: vmf       # see below

# Mixer (vMF)
mixer:
  vmf_head_dim: 64      # = d_model / heads
  epsilon_floor: 1e-3
  gamma_init: 0.99      # via w_gamma init for sigmoid(.) ~ 0.99
  parallel_scan: blelloch

# Interface
interface:
  mode: a_grounded
  joint_training: true

# Training
train:
  optimizer: adamw
  lr_peak: 6e-4
  schedule: cosine
  warmup_steps: 2000
  batch_tokens: 256000
  beta_warmup_steps: 10000  # frac of total
  free_bits: 0.5            # nats/dim (Gaussian-like bottlenecks only)
  decoder_noise_aug: 0.3
  entropy_reg_pi: 0.01
  load_balance_pi: 0.01
```

### 6.2 Medium (~1B params, 10B+ tokens)

Same as small with:
- LM $d_{\text{model}}=1536$, $L{=}24$, $H{=}16$
- Context 8192 latent positions (~64k bytes)
- Add 2 softmax attention layers (positions: $L/3$ and $2L/3$)
- Decoder $d{=}384$, $L{=}3$
- Encoder $d{=}384$, $L{=}3$
- Batch tokens 1M

### 6.3 Large (~10B params)

Add:
- Mixture FSQ head ($K_{\text{mix}}{=}4$ mixture-of-factorized) to recover cross-dim correlations
- Dirichlet head (1 per layer, $C{=}128$)
- Decoder $T{=}2$ MaskGIT refinement passes if reconstruction drops below 99.5%

---

## 7. Pseudocode

### 7.1 Training step (joint, FSQ + vMF + A-grounded)

```python
def training_step(byte_chunks, params):
    # byte_chunks: [B, T, K] bytes
    B, T, K = byte_chunks.shape

    # 1. Encode each chunk
    u = encoder(byte_chunks.reshape(B*T, K))           # [B*T, d_enc]
    z_pre = u @ W_fsq                                   # [B*T, d_q]
    z_bounded = (L-1)/2 * tanh(z_pre)
    z_hat = z_bounded + stop_grad(round(z_bounded) - z_bounded)  # STE
    z = z_hat.reshape(B, T, d_q)

    # 2. LM forward (parallel scan)
    h = lm_backbone(z)                                  # [B, T, D]
    # vMF mixer details inside lm_backbone use blelloch scan
    
    # 3. LM head -> next-latent prediction
    logits_per_dim = [h @ W_j for W_j in W_fsq_heads]   # list of [B, T, L]
    
    # 4. Prediction loss: per-dim CE, shifted by one position
    pred_loss = 0
    for j in range(d_q):
        targets = z_hat[:, 1:, j].long()                # next-chunk levels
        preds = logits_per_dim[j][:, :-1]
        pred_loss += cross_entropy(preds, targets)
    
    # 5. Reconstruction loss (current chunk)
    # Sample mask rate and apply MaskGIT-diffusion training
    t = uniform(0, 1)
    mask_rate = t                                       # linear schedule
    mask = bernoulli(mask_rate, shape=(B*T, K))
    x_masked = where(mask, MASK_TOKEN, byte_chunks.reshape(B*T, K))
    
    z_with_noise = z.reshape(B*T, d_q) + normal(0, 0.3) # noise aug
    decoder_logits = decoder(x_masked, z_with_noise)    # [B*T, K, 257]
    
    rec_loss_per_pos = cross_entropy(
        decoder_logits[mask], byte_chunks.reshape(B*T, K)[mask],
        reduction='none'
    )
    weight = 1.0 / (t + 1e-6)                           # diffusion ELBO weight
    rec_loss = weight * rec_loss_per_pos.sum() / (B*T*K)
    
    # 6. Auxiliary losses on LM mixture if applicable
    # (entropy reg, load balance — skip for pure FSQ)
    
    return rec_loss + pred_loss
```

### 7.2 Inference / generation (A-grounded)

```python
def generate(prompt_bytes, n_chunks, params):
    # Encode prompt
    K = 8
    prompt_chunks = chunk(prompt_bytes, K)              # [n_prompt, K]
    z_history = [encoder(c) for c in prompt_chunks]
    # Quantize via FSQ
    z_history = [fsq_quantize(z) for z in z_history]
    
    output_bytes = list(prompt_bytes)
    
    for _ in range(n_chunks):
        # 1. LM forward
        z_tensor = stack(z_history).unsqueeze(0)        # [1, T, d_q]
        h = lm_backbone(z_tensor)                       # [1, T, D]
        h_last = h[0, -1]
        
        # 2. Predict next latent (sample from factorized categoricals)
        z_next = zeros(d_q)
        for j in range(d_q):
            logits = h_last @ W_fsq_heads[j]
            level = sample_categorical(softmax(logits / temperature))
            z_next[j] = (level - (L-1)/2)               # de-shift to FSQ grid
        
        # 3. Decode bytes (one-shot, argmax — z is sufficient)
        masked_input = [MASK_TOKEN] * K
        decoder_logits = decoder(masked_input, z_next)  # [K, 257]
        bytes_next = argmax(decoder_logits, axis=-1)
        
        # Optional: MaskGIT refinement if T > 1
        if maskgit_T > 1:
            bytes_next = maskgit_refine(decoder_logits, z_next, T=maskgit_T)
        
        output_bytes.extend(bytes_next.tolist())
        
        # 4. A-grounded: re-encode the decoded bytes for next-step input
        z_grounded = encoder(bytes_next)
        z_grounded_q = fsq_quantize(z_grounded)
        z_history.append(z_grounded_q)
    
    return bytes(output_bytes)
```

### 7.3 vMF mixer with parallel scan (forward, one head)

```python
def vmf_mixer_forward(x, params):
    # x: [B, T, D]
    # Project per head
    v = normalize(x @ W_v, axis=-1)                     # [B, T, d_head]
    w = softplus(x @ w_w_param)                         # [B, T, 1]
    gamma = sigmoid(x @ w_gamma_param)                  # [B, T, 1]
    
    # Inputs to scan
    contrib_r = w * v                                   # [B, T, d_head]
    contrib_n = w                                       # [B, T, 1]
    
    # Associative scan: out[t] = gamma[t] * out[t-1] + contrib[t]
    # using the binary op (g, c) o (g', c') = (g*g', g'*c + c')
    r_scan = parallel_scan_first_order_linear(gamma, contrib_r)  # [B, T, d_head]
    n_scan = parallel_scan_first_order_linear(gamma, contrib_n)  # [B, T, 1]
    
    # Readout
    r_norm = sqrt(sum(r_scan**2, axis=-1, keepdims=True) + epsilon)
    mu_hat = r_scan / r_norm                            # direction
    R_bar = r_norm / (n_scan + epsilon)                 # confidence in [0,1]
    
    return R_bar * mu_hat                               # [B, T, d_head]
```

The `parallel_scan_first_order_linear` is a Blelloch scan; reuse the Mamba/GLA kernel.

### 7.4 Inference-time per-step (vMF, autoregressive)

```python
# Maintain state (r, n) per layer per head
state = {layer: {head: (zeros(d_head), 0.0) for head in heads} for layer in layers}

def vmf_step(x_t, layer, head):
    r_prev, n_prev = state[layer][head]
    v_t = normalize(x_t @ W_v[layer][head])
    w_t = softplus(x_t @ w_w[layer][head])
    gamma_t = sigmoid(x_t @ w_gamma[layer][head])
    
    r_new = gamma_t * r_prev + w_t * v_t
    n_new = gamma_t * n_prev + w_t
    state[layer][head] = (r_new, n_new)
    
    mu_hat = r_new / sqrt(norm(r_new)**2 + epsilon)
    R_bar = norm(r_new) / (n_new + epsilon)
    return R_bar * mu_hat
```

$O(D)$ state, $O(D)$ per-step compute. Memory grows zero with sequence length.

---

## 8. Pitfalls and gotchas

### 8.1 Posterior collapse (Gaussian/vMF/LN bottlenecks only)

**Symptom:** $\text{KL}(\tau \| \rho) \to 0$ at the same time reconstruction loss stays high.
**Fix:** $\beta$ warmup (start at 0, ramp linearly), free bits floor (KL bounded below at 0.5 nats/dim per latent), decoder noise augmentation 0.3.

### 8.2 Variance cheating (GMM/vMF only)

**Symptom:** predicted $\Sigma_t^{(k)}$ or $1/\kappa_t^{(k)}$ grows during training, NLL improves trivially.
**Fix:** clip $\log\sigma^2$ to $[-5, 2]$. For vMF, clip $\kappa \in [1, 100]$. Penalize average $\text{tr}(\Sigma_t)$ at small weight.

### 8.3 Mixture collapse / dead components

**Symptom:** $\pi_t^{(k)} \to 0$ for some $k$, that component stops updating.
**Fix:** entropy regularization $\lambda_\pi \mathbb{E}[H(\pi_t)]$ with $\lambda_\pi = 0.01$. Load-balance loss $\sum_k (\bar\pi^{(k)} - 1/K)^2$. Orthogonal init of $W_\mu^{(k)}$ across $k$.

### 8.4 STE bias (FSQ/BSQ)

**Symptom:** encoder gradients are biased; encoder learns slowly.
**Fix:** for BSQ, scale sign-STE pass-through by $1/\sqrt{d_q}$. For FSQ, the rounding-STE is locally identity, less of an issue. Use lower lr on encoder (0.5× LM lr) if needed.

### 8.5 Latent drift (Option A native only)

**Symptom:** generation quality degrades after ~hundreds of latent steps; sampled $z$ wanders off the encoder manifold.
**Fix:** switch to Option A-grounded (re-encode every step). Or use scheduled sampling during training (occasionally feed the LM its own sampled latents instead of encoder latents).

### 8.6 vMF normalization singularity

**Symptom:** gradient spikes early in training when $\|r_t\|$ is small.
**Fix:** $\epsilon$ floor: use $r/\sqrt{\|r\|^2+\epsilon}$ with $\epsilon = 10^{-3}$. Initialize $W_v$ small so $v_t$ starts well-conditioned.

### 8.7 Bessel function stiffness (vMF prediction head)

**Symptom:** training instability when $\kappa$ predictions get large.
**Fix:** parameterize $\kappa$ via $\kappa = \kappa_{\max} \cdot \sigma(h \cdot w_\kappa)$ with $\kappa_{\max} = 100$. Use stable Bessel implementation (e.g., asymptotic expansion for large $\kappa$).

### 8.8 Decoder collapse (when training jointly)

**Symptom:** decoder predicts byte distribution independent of $z$ (KL between $p(b|z)$ and $p(b)$ → 0).
**Fix:** decoder noise augmentation forces the decoder to use $z$ (small perturbations should yield different outputs, which only works if the decoder reads $z$ meaningfully). FiLM conditioning (rather than just prefix) makes $z$ inescapable.

### 8.9 Memory blowup at long context (softmax mixer)

**Symptom:** KV cache OOMs at long sequences.
**Fix:** the whole point of the geometric mixers is to avoid this — use vMF (state $O(D)$) for the bulk. If keeping any softmax, use sliding window or limit to 2 layers.

---

## 9. Brief references (paraphrased, no quotation)

- **CALM** (Shao et al. 2025): continuous next-vector prediction with autoencoder + energy-based head. The architecture this work directly contests; we replace the energy head with an explicit-likelihood head (FSQ/vMF/etc.) to recover exact BPB.
- **FSQ** (Mentzer et al. 2024): finite scalar quantization as VQ-VAE replacement. The drop-in discrete bottleneck used here.
- **BSQ** (Zhao et al. 2025): binary spherical quantization. Alternative discrete bottleneck.
- **MaskGIT** (Chang et al. 2022): confidence-based parallel decoding for masked-token models.
- **Time-free masked diffusion** (MDLM / MD4, Sahoo et al. / Shi et al. 2024): absorbing-state discrete diffusion without timestep conditioning. The principled training loss for our decoder.
- **MTP** (Gloeckle et al. 2024, DeepSeek-V3): multi-token prediction with parallel softmax heads. The strong baseline for bandwidth-per-step.
- **Mamba / GLA** (Gu & Dao 2024, Yang et al. 2024): selective state-space and gated linear attention. Source of the parallel-scan training kernel reused for vMF/Dirichlet mixers.
- **vMF embeddings** (Davidson et al. 2018; Hyperspherical VAE): closed-form von Mises-Fisher density for spherical latents.
- **VRNN / SRNN** (Chung et al. 2015 / Fraccaro et al. 2016): sequential VAE with autoregressive prior — the mathematical ancestor of our joint encoder-LM training.

---

## Appendix A — quick decision tree

```
Q: Easy training, exact likelihood, fast convergence?  -> FSQ + vMF + A-grounded
Q: Best calibration / continuous uncertainty?            -> vMF bottleneck + vMF mixer (no GMM unless essential)
Q: Match BPE+MTP at fixed compute?                       -> FSQ + softmax (Phase 2a) first; mixers later
Q: Best exact recall?                                    -> growing-memory attention (vMF-factor or Gaussian-factor) at recall-critical layers
Q: Smallest state at long context?                       -> vMF fixed-state mixer everywhere; or BSQ-factor for 1-bit KV cache
Q: Interpretable internals?                              -> add Dirichlet head per layer (fixed-state or factor-attention version)
Q: Push K high (16+)?                                    -> MaskGIT decoder with T=2-3 refinement, not one-shot
Q: Minimal upgrade to standard softmax attention?        -> vMF-factor attention (same compute, free confidence signal)
Q: Cache memory dominates inference cost?                -> BSQ-factor attention (1-bit per-position writes)
```

## Appendix B — what NOT to build first

- **GMM bottleneck.** Hardest to train. Use only if calibrated continuous covariance is essential and FSQ/vMF have been ablated.
- **Hyperbolic state.** Doesn't fit the additive-conjugate framework. Output-side only, if at all.
- **AR decoder over the K bytes.** Destroys the latency advantage. Use only if you don't care about generation speed.
- **Pure linear attention.** Default linear attention is dominated by vMF (better-conditioned normalizer, same compute). Use vMF instead.
- **Energy / score-matching head (CALM-style).** Gives up explicit likelihood; the whole point of this design is to *keep* the likelihood by using FSQ/vMF/Gaussian instead.

---

*End of handover. Questions for the implementer: which bottleneck do you commit to (FSQ recommended)? Which interface (A-grounded recommended)? Which mixer composition (hybrid vMF + softmax recommended)? Implement Phase 0 baselines first, then go in order through Phases 1–3 with the go/no-go criteria.*