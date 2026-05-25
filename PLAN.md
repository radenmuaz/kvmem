# KV-as-Fast-Weights — Implementation Handover

This is a handover document for implementing stages 0 and 1 of the KV-as-Fast-Weights project, plus outlines of subsequent stages. The architectural decisions in §2 are **locked** — don't relitigate them while implementing. The existing reference implementations (`kv_fast_weights_compression.py`, `kv_fast_weights_lm.py`, `kv_fast_weights_regression.py`) and `PROJECT_SUMMARY.md` are prerequisites; read those first.

---

## 1. Project Context

The project builds a transformer architecture in which **new information is absorbed by writing to the KV cache rather than by gradient updates to weights**. Slow MLP weights hold *procedural* skills (read, write, retrieve, compress); the KV cache holds *declarative* content. "Training on new data" at deployment becomes a single forward pass that produces a compressed KV memory — no backprop at inference.

Stages 0 and 1 validate the core primitive (compression + multi-pass refinement on a single chunk) before any of the streaming, SRS, or scale-up work begins. The synthetic task is in-context Markov-chain language modeling at the scale already used in the reference implementations.

---

## 2. Locked Architectural Decisions

These were settled before implementation begins. Do not change without checking back.

| Decision | Choice | Why |
|---|---|---|
| Architecture style | Decoder-only with custom attention mask | Simpler than encoder-decoder; existing reference works; switch to hybrid only at stage 2 |
| Position embeddings | **None** (NoPE — no learned positional terms) | Enables arbitrary memory length; better length generalization |
| Memory tokens | **Single fixed marker** from existing vocab (reserved low-frequency or held-out corpus token) | No vocab augmentation; arbitrary N; saves ~2Nd params |
| Strength gating | Per-position softplus gate on attention keys | Soft sparsification, capacity regularization |
| Training target | **NTP on independent continuation** (NOT reconstruction of `x_S`) | Theoretically motivated by IPTT (arXiv:2604.06169); eliminates shortcut issues that reconstruction had |
| Mask design | `Y` blocks are write-only sinks; `Y` can't see source; cross-`Y` blocked | Train-inference consistency: memory writes use only what's available at inference |

---

## 3. Notation

| Symbol | Meaning |
|---|---|
| `V` | Vocabulary size (no augmentation; one corpus token is reserved as marker `m`) |
| `L_S` | Source chunk length |
| `N` | Number of memory positions per pass |
| `L_y` | Continuation length per pass |
| `T` | Number of refinement passes (stage 1+) |
| `d` | Model dimension |
| `H` | Number of attention heads, `d_h = d/H` |
| `S, M, Y` | Source, memory, continuation region position sets |
| `m` | Reserved marker token (single fixed corpus token ID) |

---

## 4. Common Architecture (used by both stages)

### 4.1 Input embedding (NoPE)

For input token sequence `z` of length `L`:
```
h^(0)_i = E[z_i]      # token embedding only, no positional term
```
`E ∈ R^{V × d}`, learned from random init.

### 4.2 Strength head

Per-position scalar gate, computed once from `h^(0)` and reused at every attention layer:
```
s_tilde_i = w_2^T · GELU(W_1 · LayerNorm(h^(0)_i))
l_i       = -softplus(-s_tilde_i)         # = log sigmoid(s_tilde_i), in (-inf, 0]
s_i       = exp(l_i) = sigmoid(s_tilde_i) # in (0, 1]
```
`W_1 ∈ R^{d×d}`, `w_2 ∈ R^d`.

### 4.3 Transformer block (pre-norm)

For layer ℓ:
```
h_hat   = LayerNorm(h^(ℓ-1))
Q,K,V   = h_hat @ W_Q, h_hat @ W_K, h_hat @ W_V    # split into H heads, head dim d_h

# Attention logits with mask and strength gate on keys
A_ij^h  = (Q_i^h · K_j^h) / sqrt(d_h) + mask_ij + l_j
alpha^h = softmax_j(A^h)
o_i^h   = sum_j alpha^h_ij · V_j^h

# Merge heads, project, residual
o       = Concat_h(o^h) @ W_O
u       = h^(ℓ-1) + o
h^(ℓ)   = u + W_2 · GELU(W_1 · LayerNorm(u))       # FFN, no bias
```

### 4.4 Output

```
logits_i = W_out @ LayerNorm(h^(L_layers)_i)       # W_out ∈ R^{V × d}
```

### 4.5 Default hyperparameters

| Param | Stage 0/1 default |
|---|---|
| `V` | 32 (Markov chain vocab) |
| `L_S` | 96 |
| `L_y` | 32 |
| `N` | 8 (sweep over {1, 2, 4, 8, 16, 32, 64} in stage 0) |
| `T` | 1 (stage 0), sweep {1, 2, 4, 8} (stage 1) |
| `d` | 128 |
| `n_layers` | 4 (consider 8 for stage 1 if NoPE collapses; see §10) |
| `H` | 4 (`d_h = 32`) |
| `d_ff` | 4·d = 512 |
| `lambda_cont` | 2.0 (continuation NTP weight) |
| `lambda_c` | 1e-3 (capacity penalty) |
| Batch size | 64 |
| Optimizer | AdamW, lr = 1e-3, warmup 1000 steps, cosine to 1e-5 |
| Gradient clip | 1.0 |
| Training steps | 50k |

---

## 5. Stage 0 — Single-pass NTP through the bottleneck

### 5.1 Goal

Validate that `N` memory positions can encode a useful predictive prior over `L_S` source tokens, measured by continuation perplexity through the bottleneck.

### 5.2 Sequence layout

```
z = [ x_S ; m, m, ..., m ; y ]
    [ L_S ;     N         ; L_y ]
Total length L = L_S + N + L_y
```

- `x_S`: `L_S` tokens sampled from a Markov chain (fresh per sample, both transition matrix and stationary distribution).
- Memory section: `N` copies of the fixed marker token `m`.
- `y`: `L_y` tokens — **independent continuation** sampled from the same chain, starting from the terminal state `x_S[L_S - 1]`. Not a copy of `x_S`.

Position sets:
```
S = [0, L_S)
M = [L_S, L_S + N)
Y = [L_S + N, L_S + N + L_y)
```

### 5.3 Attention mask

```
m_ij = -inf  if j > i                                   (causal)
m_ij = -inf  if j in Y  and  i not in Y                 (Y is write-only)
m_ij = -inf  if i in Y  and  j in S                     (bottleneck)
m_ij = 0     otherwise
```

The "Y is write-only" rule is trivial at stage 0 (nothing comes after Y) but matters for consistency with stage 1.

```python
def make_mask_stage0(L_S, N, L_y):
    L = L_S + N + L_y
    rows = np.arange(L)[:, None]
    cols = np.arange(L)[None, :]
    causal = cols <= rows
    is_S = (cols < L_S)
    is_Y_row = (rows >= L_S + N)
    is_Y_col = (cols >= L_S + N)
    block_y_writeonly = is_Y_col & (~is_Y_row)
    block_y_sees_S    = is_Y_row & is_S
    blocked = block_y_writeonly | block_y_sees_S
    visible = causal & ~blocked
    return jnp.where(jnp.array(visible), 0.0, -1e9)
```

### 5.4 Loss

Position masks over the `L-1` shifted prediction positions:
```
mask_src[i]  = 1 if 0 <= i <= L_S - 2                     # source NTP
mask_mem[i]  = 1 if L_S - 1 <= i <= L_S + N - 2           # predicts deterministic markers → DROP
mask_cont[i] = 1 if L_S + N - 1 <= i <= L - 2             # continuation NTP
```

Per-token NLL: `n_i = -log_softmax(logits_i)[z_{i+1}]`.

```
L_src  = (1/B) sum_b  ( sum_i mask_src[i]  · n_{b,i}  / sum_i mask_src[i]  )
L_cont = (1/B) sum_b  ( sum_i mask_cont[i] · n_{b,i}  / sum_i mask_cont[i] )
L_cap  = (1/(B·L)) sum_b sum_i s_{b,i}

L = L_src + lambda_cont · L_cont + lambda_c · L_cap
```

### 5.5 Evaluation

Three conditions on held-out samples (replicate the existing `compression.py` setup):

- **matched**: `y` continues the same chain `x_S` came from.
- **cross**: `y` continues a *different* chain than `x_S`.
- **uniform**: `x_S` replaced by random tokens; `y` from a fresh chain.

Report:
```
bpt_matched = L_cont(matched) / log(2)
bpt_cross   = L_cont(cross)   / log(2)
bpt_uniform = L_cont(uniform) / log(2)
gain        = bpt_uniform - bpt_matched   # bits/token of useful info in memory
```

### 5.6 Success criteria

| Criterion | Required |
|---|---|
| `bpt_matched < bpt_uniform` | Yes (memory carries info) |
| `bpt_cross > bpt_uniform` | Yes (wrong cache actively hurts — confirms bottleneck is used) |
| `bpt_matched` approaches chain entropy as `N` grows | Trend check across `N` sweep |
| Memory strength `mean(s_i over M)` substantially higher than source strength | Yes (gating learned to expose memory) |

### 5.7 Sweep deliverable

Plot `bpt_matched, bpt_cross, bpt_uniform` as a function of `N ∈ {1, 2, 4, 8, 16, 32, 64}`. This is the headline output of stage 0.

### 5.8 Inference (deployment mode, not eval)

```
# Ingest
tokens = concatenate([x_S, [m]*N])
_, kv_cache = model.apply(params, tokens, return_cache=True)
M_kv = extract_kv_at_positions(kv_cache, positions=range(L_S, L_S+N))

# Query (reused N times)
def query(q):
    # Prepend M_kv as KV prefix; forward query tokens with causal mask
    logits, _ = model.apply_with_prefix_cache(params, q, prefix_cache=M_kv)
    return logits
```

---

## 6. Stage 1 — Multi-pass NTP refinement

### 6.1 Goal

Validate that `T` memory-writing passes produce monotonically better predictive memory, measured as decreasing continuation NLL across `t = 1, ..., T`.

### 6.2 Sequence layout

```
z = [ x_S ; m^N ; y^(1) ; m^N ; y^(2) ; ... ; m^N ; y^(T) ]
```

Total length `L = L_S + T·(N + L_y)`.

**Critical**: Each `y^(t)` is an **independent fresh sample** of the continuation from the same Markov chain (different random seed each time, but all starting from `x_S[L_S - 1]`). This is the key change from the autoencoding spec.

Position sets:
```
p_m(t) = L_S + (t-1)·(N + L_y)              # start of M^(t)
p_y(t) = p_m(t) + N                         # start of Y^(t)
M^(t)  = [p_m(t), p_m(t) + N)
Y^(t)  = [p_y(t), p_y(t) + L_y)
```

### 6.3 Attention mask

Four blocking rules combined with causality:

```
m_ij = -inf  if j > i                                       (causal)
m_ij = -inf  if j in any Y^(s)  and  i not in any Y^(s)     (Y is write-only)
m_ij = -inf  if i in Y^(t)  and  j in S                     (bottleneck)
m_ij = -inf  if i in Y^(t)  and  j in Y^(s),  s != t        (cross-Y blocked)
m_ij = 0     otherwise
```

Effect:
- `M^(t)` attends to: `S`, `M^(<t)`, and own causal positions in `M^(t)`. **Never** to any `Y`.
- `Y^(t)` attends to: `M^(≤t)` and own causal positions in `Y^(t)`. Not `S`, not other `Y^(s)`.

```python
def make_mask_stage1(L_S, N, L_y, T):
    L = L_S + T * (N + L_y)
    rows = np.arange(L)
    cols = np.arange(L)

    # Region tags
    is_S = rows < L_S
    in_block = ~is_S
    offset = np.where(in_block, rows - L_S, 0)
    pass_idx = offset // (N + L_y)        # 0..T-1
    within = offset % (N + L_y)
    is_M = in_block & (within < N)
    is_Y = in_block & (within >= N)

    is_S_col = is_S[None, :]
    is_Y_col = is_Y[None, :]
    is_Y_row = is_Y[:, None]
    pass_row = pass_idx[:, None]
    pass_col = pass_idx[None, :]

    causal = cols[None, :] <= rows[:, None]
    block_y_writeonly = is_Y_col & (~is_Y_row)
    block_y_sees_S    = is_Y_row & is_S_col
    block_y_cross     = is_Y_row & is_Y_col & (pass_row != pass_col)
    blocked = block_y_writeonly | block_y_sees_S | block_y_cross

    visible = causal & (~blocked)
    return jnp.where(jnp.array(visible), 0.0, -1e9)
```

### 6.4 Loss

Per-pass continuation NLL:
```
L_cont_t = (1/(B·L_y)) sum_b sum_{k=0..L_y-1}
             -log_softmax(logits_{b, p_y(t) - 1 + k})[ y^(t)_{b,k} ]
```

(Prediction at position `p_y(t) - 1`, the last memory token of pass `t`, predicts the first y-token `y^(t)_0`.)

Pass weighting (start with linear ramp normalized to sum to 1):
```
beta_t = (t/T)^gamma / sum_{s=1..T} (s/T)^gamma,    gamma = 1
```

Total:
```
L = L_src + lambda_cont · sum_{t=1..T} beta_t · L_cont_t + lambda_c · L_cap
```

### 6.5 Evaluation

**Primary metric — per-pass curve:**
```
bpt_cont(t) = L_cont_t / log(2),  for t = 1..T
```
The curve should be monotone decreasing in `t` (compute-via-unrolling refinement).

**Diagnostic — truncation ablation (critical):**

Train a model at `T = 4`. At eval, *truncate* the sequence after `M^(t)` for `t ∈ {1, 2, 3, 4}` and append a fresh `m^N; y` block, measuring continuation NLL with only the first `t` passes of memory available.

Compare to a *dedicated* `T = 1` model (trained from scratch with only one pass).

| Outcome | Interpretation |
|---|---|
| Truncated-at-1 ≈ dedicated `T=1` model, full `T=4` better | Multi-pass is real refinement |
| Truncated-at-1 worse than dedicated `T=1` model | Model leans on later passes; first pass underperforms — failure mode |
| All truncations flat | No real refinement; each pass redundant |

This diagnostic decides whether stage 1 is genuinely working before stage 2 starts.

### 6.6 Success criteria

| Criterion | Required |
|---|---|
| Monotone decrease `bpt_cont(t+1) ≤ bpt_cont(t)` | Yes |
| `bpt_cont(T=4) < bpt_cont(T=1)` by ≥ 0.1 bits | Yes |
| Truncation diagnostic: `T=1` truncated ≥ dedicated `T=1` | Yes |
| Memory slot similarity check (diversity diagnostic) | See §10 |

### 6.7 Inference

At deployment, drop `Y` blocks entirely (they're write-only sinks; nothing reads them):
```
tokens = concatenate([x_S, [m]*N, [m]*N, ..., [m]*N])  # T memory blocks back-to-back
_, kv_cache = model.apply(params, tokens, return_cache=True)
M_kv = extract_kv_at(kv_cache, positions=range_of(M^(T)))
```

This is shorter than the training-length sequence (no Y blocks), so inference is cheaper than training. Verify the mask between `M^(t)` positions matches training (it does because `M` never attends to `Y`).

### 6.8 Sweep deliverable

Plot `bpt_cont(t)` as a function of `t` for `T ∈ {1, 2, 4, 8}`, and the truncation diagnostic plot.

---

## 7. Future Stages (preview)

Implementation of stages 2+ is contingent on stages 0 and 1 passing their success criteria. Don't start any of these without confirmation.

### Stage 2 — Streaming, no rehearsal
- Layout: `[c_1; m^N; c_2; m^N; ...; c_C; m^N; y_target]`
- Each `m^N` block writes memory conditioned on current chunk + all prior memory
- Test interference: held-out `y_target` continues which chunk? Probe perplexity as a function of chunk-recency
- **At this boundary, consider switching to shared encoder-decoder hybrid** with explicit persistent `M` tensor and cross-attention. See §11.

### Stage 3 — Random-schedule rehearsal training
- Insert random "revisit" chunks into the stream
- Train the model to accept revisits (memory updates from a re-shown chunk)
- Tests whether the procedural runtime can use rehearsal slots

### Stage 4 — Self-evaluation head
- Add a head `r_phi(M, c)` predicting reconstruction NLL for chunk `c` given current memory
- Trained by regression against actual NLL (supervised by forward signal, no backprop at inference)
- Calibration check: rank correlation between `r_phi` and true NLL

### Stage 5 — SRS controller
- Self-evaluation head drives a priority queue of chunks to rehearse
- Decision at each step: ingest new chunk vs. rehearse highest-difficulty old chunk
- Compare SRS-driven streaming to no-rehearsal and random-rehearsal baselines

### Stage 6 — Backprop baseline
- Train an identical-architecture model with full backprop on the streamed corpus
- Headline number: gap between SRS streaming model and backprop baseline
- Sweep `(N, T, B)` to plot gap-vs-compute curve

### Stage 7 — Real text
- Move from synthetic Markov chains to small natural-language corpus (a few thousand tokens)
- Watch for: tokenizer choice, position-embedding extrapolation if you've reintroduced any, capacity scaling
- Single-epoch streaming over diverse corpus to avoid memorization

---

## 8. Implementation Notes

### 8.1 Repo structure (suggested)

```
kv_fast_weights/
├── PROJECT_SUMMARY.md             # existing
├── STAGE_IMPLEMENTATION_PLAN.md   # this file
├── kv_fast_weights_compression.py # existing reference (stage 0 baseline)
├── kv_fast_weights_lm.py          # existing reference
├── kv_fast_weights_regression.py  # existing reference
├── common/
│   ├── model.py                   # shared transformer block, strength head
│   ├── mask.py                    # mask constructors (stage 0, stage 1)
│   ├── data.py                    # Markov chain sampling
│   └── eval.py                    # bpt computation, conditions
├── stage0/
│   ├── train.py
│   └── eval.py
├── stage1/
│   ├── train.py
│   └── eval.py
└── reports/
    ├── stage0_sweep_N.png
    ├── stage1_curve_T.png
    └── stage1_truncation_diagnostic.png
```

Start by abstracting the existing `kv_fast_weights_compression.py` into the `common/` modules; the model and data code is reusable.

### 8.2 Dependencies

Match the existing reference implementations exactly:
- JAX (CPU-only OK for stages 0–1; small models)
- Flax (Linen API)
- Optax
- numpy
- matplotlib (for plots only)
- No other dependencies

### 8.3 Things to NOT do

- **Don't add position embeddings.** NoPE is locked.
- **Don't use dedicated memory token IDs.** Reserved corpus token only.
- **Don't make `y` a copy of `x_S` (autoencoding).** Always independent continuation.
- **Don't allow `M` to attend to `Y` blocks.** That's the train-inference consistency rule.
- **Don't allow `Y^(t)` to attend to `Y^(<t)`.** Different samples per pass, no cross-attention.
- **Don't introduce IPTT-style gradient-at-inference fast-weight updates.** This project's commitment is no-backprop at inference.

### 8.4 Things to watch for

- **Slot collapse**: with NoPE and a single marker token, all memory positions have identical input at layer 0. Differentiation comes from causal-mask asymmetry. At 4 layers / 128 dim this may be borderline; if memory slots end up encoding redundant information, increase depth to 8 layers before tweaking anything else. Diagnostic: cosine similarity between final-layer hidden states at `M` positions.
- **NoPE depth shortage**: NoPE relies on depth to derive positional info from causal-mask counts. Watch test perplexity and consider 8 layers if 4 isn't enough.
- **Strength-head collapse**: if `s_i` for memory positions collapses toward 0, capacity penalty is too aggressive — lower `lambda_c`.
- **Marker token leakage**: if the marker token `m` appears in the actual data (Markov chain output), it'll confuse memory-boundary detection. Reserve a token outside the chain's vocabulary by sampling chains over `[0, V-2)` and using `m = V-1`.

### 8.5 Validation order

1. Implement stage 0 with `N = 8`, single seed, verify `bpt_matched < bpt_uniform` and `bpt_cross > bpt_uniform` (sanity).
2. Sweep `N` for stage 0; produce the curve.
3. Implement stage 1 with `T = 4` and `N` from stage 0's best.
4. Verify monotone `bpt_cont(t)` curve.
5. Run the truncation diagnostic.
6. Stop and report back before stage 2.

### 8.6 What to report back

After stage 0:
- `bpt_matched, bpt_cross, bpt_uniform` for the `N` sweep
- The strength-head profile by region (mean `s_i` over S, M, Y)
- Whether the success criteria in §5.6 are met
- Training curves (loss vs. step) — sanity check

After stage 1:
- Per-pass `bpt_cont(t)` curve for `T ∈ {1, 2, 4, 8}`
- Truncation diagnostic results
- Slot-collapse diagnostic (cosine similarity matrix between memory positions at the final layer)
- Whether the success criteria in §6.6 are met

---

## 9. Theoretical background (reference only)

The training target is NTP on independent continuations rather than reconstruction of the source, based on:

- **IPTT (Feng et al., arXiv:2604.06169, 2026)**: proves that NTP-aligned targets are strictly better than reconstruction targets for in-context predictive tasks. Their Theorem 1: with reconstruction targets, the expected logit change for the correct next token is negligible; with NTP-aligned targets, it's bounded below by a positive quantity proportional to embedding norms and key-query alignment.

- **Fast-weight programmers (Schlag et al. 2021)**: linear attention is equivalent to fast-weight programming; the KV cache *is* the fast-weight matrix `W_fast = sum_i k_i v_i^T`.

The "multi-pass refinement" claim of stage 1 is about **compute unrolling**, not new information per pass — each pass sees the same `x_S` plus the prior memory state. The refinement signal is the same as in deep equilibrium models or Universal Transformers: more compute on the same data, not more data. Refinement should be expected to be bounded, not unlimited.

---

## 10. Key diagnostic: memory slot diversity

After training, run this check on the final model:

```python
def slot_diversity(params, x_S):
    """Returns NxN cosine similarity matrix between memory slot hidden states."""
    tokens = concatenate([x_S, [m]*N])
    h_final = model.apply(params, tokens, return_hidden=True)
    M_hidden = h_final[L_S:L_S+N]                           # (N, d)
    M_normed = M_hidden / jnp.linalg.norm(M_hidden, axis=-1, keepdims=True)
    return M_normed @ M_normed.T                             # (N, N) cosine sim
```

Expected: off-diagonal entries should average well below 0.9. If they're near 1.0, memory slots have collapsed to redundant copies and effective compression rank is much less than `N`.

If collapse is detected:
1. First: increase depth (`n_layers = 8`).
2. If still collapsing: add tiny input noise at memory positions *during training only* (Gaussian, std `1e-3`).
3. If still collapsing: switch to sequential markers `m_0, m_1, ..., m_{N-1}` (using the first `N` corpus tokens; zero param cost) instead of single shared marker.

---

## 11. Stage 2 architecture revisit

At the stage 1 → stage 2 boundary, evaluate whether to switch from decoder-only to shared-weight encoder-decoder hybrid:

| Pro decoder-only continuation | Pro hybrid switch |
|---|---|
| Existing code works | Persistent `M` becomes a first-class state |
| `L_src` keeps LM competence | Streaming + SRS naturally maps onto `M`-state operations |
| In-context learning capabilities preserved | Bottleneck structural (no mask gymnastics) |
| Simpler implementation | Multi-chunk attention growth is bounded |

If stage 1 results show that the decoder-only mask becomes unwieldy at `T > 8` or with multiple chunks queued, switch. The hybrid spec:

- Same backbone weights, one set of parameters
- Encoder pass: causal mask over `[x_S; m^N]`, extract `M ∈ R^{N×d}` from hidden states at marker positions
- Decoder pass: causal mask over query tokens, with cross-attention to `M` at each layer
- Cross-attention has a learned null memory `M_0 ∈ R^{1×d}` for the encoder-mode-or-pass-1 case (see §11 of the conversation log for derivation)

Don't implement this for stage 1 — only revisit after stage 1 completes.

---

## 12. Out-of-scope for this handover

- Real-text experiments (deferred to stage 7)
- Backprop baseline (deferred to stage 6)
- Self-evaluation head training (deferred to stage 4)
- SRS scheduling logic (deferred to stage 5)
- GPU scaling, larger models (deferred to stage 7)

If during implementation any of these seem necessary earlier, flag it and discuss before proceeding.

---

## 13. Acceptance summary

A successful stage 0 + stage 1 implementation produces:

1. Three reusable common modules (`model.py`, `mask.py`, `data.py`).
2. Stage 0: a sweep over `N` showing the `bpt_matched / bpt_cross / bpt_uniform` curves.
3. Stage 1: a multi-pass refinement curve (`bpt_cont(t)` vs `t` for several `T`).
4. Stage 1: the truncation diagnostic plot.
5. Diagnostic outputs: strength-head profile, slot-diversity matrix.
6. A short markdown report (`reports/stage01_summary.md`) summarizing what was found, what worked, what didn't, and a go/no-go recommendation for stage 2.

The go/no-go recommendation should be made *honestly* based on the data. If the truncation diagnostic fails (first-pass memory from a 4-pass-trained model is worse than dedicated single-pass training), say so — that's a failure mode that needs addressing before stage 2 is worth doing.