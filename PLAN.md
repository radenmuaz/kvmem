# KV-as-Fast-Weights — Implementation Plan

## 1. Project Context

The project builds a transformer architecture in which **new information is absorbed by writing to the KV cache rather than by gradient updates to weights**. Slow MLP weights hold *procedural* skills (read, write, retrieve, compress); the KV cache holds *declarative* content. "Training on new data" at deployment becomes a single forward pass that produces a compressed KV memory — no backprop at inference.

Stages 0 and 1 validate the core primitive (compression + multi-pass refinement on a single chunk) before any of the streaming, SRS, or scale-up work begins. The synthetic task is byte-level Markov-chain language modeling.

---

## 2. Architectural Decisions

| Decision | Choice | Why |
|---|---|---|
| Architecture style | Decoder-only with custom attention mask | Simpler than encoder-decoder |
| Position embeddings | **None** (NoPE) | Enables arbitrary memory length; better length generalization |
| Memory tokens | **Sequential bytes 0x00–0x{N-1}** bracketed by STX/ETX control bytes | Gives each slot unique layer-0 input (helps differentiation); bracket bytes signal region boundaries; all from byte vocab |
| Strength gating | **None** | Removed; mask structure alone enforces the bottleneck |
| Training target | **NTP on independent continuation** (NOT reconstruction of `x_S`) | Motivated by IPTT (arXiv:2604.06169); eliminates shortcut issues |
| Mask design | `Y` blocks are write-only sinks; `Y` can't see source; cross-`Y` blocked | Train-inference consistency |
| Vocab | **Byte-level: V = 256** (values 0–255) | Realistic; no vocab augmentation |
| **Variable N training** | **N sampled randomly each batch** from `N_set = {2, 4, 8, 16, 32}` | Model learns to compress into any N; masks precomputed per N at startup |

---

## 3. Notation

| Symbol | Meaning |
|---|---|
| `V` | Vocabulary size = 256 (byte-level, 0–255) |
| `L_S` | Source chunk length |
| `N` | Number of **inner** memory positions per pass (varies per batch during training) |
| `L_y` | Continuation length per pass |
| `T` | Number of refinement passes (stage 1+) |
| `d` | Model dimension |
| `H` | Number of attention heads, `d_h = d/H` |
| `S, M, Y` | Source, memory, continuation region position sets |
---

## 4. Segment Protocol

### 4.1 Design principle: delimiter-based, not content-based

Memory slots are identified by **position between delimiters**, not by their token ID. This generalizes to arbitrary N and arbitrary corpus without collision.

**Two layers of the byte vocab:**
- `0x00–0x1F` — ASCII control codes, reserved as **protocol bytes**. Never appear in data content. The model learns these as structural signals.
- `0x20–0xFF` — printable ASCII + high bytes, used for **data content**.

This separation is enforced in data generation: Markov chains and real text corpora are constrained to `0x20–0xFF`. If a corpus contains control bytes, strip or remap them before training.

### 4.2 Segment byte registry

A small set of control bytes act as open/close delimiters for named segment types. Allocate pairs as needed; unused pairs cost nothing.

| Open | Close | ASCII names | Segment type | Stage introduced |
|---|---|---|---|---|
| `0x02` | `0x03` | STX / ETX | **Memory write** (M region) | Stage 0 |
| `0x04` | `0x05` | EOT / ENQ | **Continuation / query** (Y region) | Stage 0 (implicit — Y is already delimited by mask, add explicit tokens if needed) |
| `0x06` | `0x07` | ACK / BEL | **SRS rehearsal chunk** | Stage 3 |
| `0x08` | `0x09` | BS / HT   | **Self-eval probe** | Stage 4 |
| `0x0A` | `0x0B` | LF / VT   | Reserved | — |
| `0x0C` | `0x0D` | FF / CR   | Reserved | — |

Adding a new segment type in a future stage = allocate the next pair, update the mask builder, update data generation. Zero architecture changes.

### 4.3 Memory block format

Inner memory slots are all **`0x00` (NUL)** — identical, N of them, no slot index encoded in token ID:

```
memory_block(N) = [ 0x02 | 0x00 × N | 0x03 ]
                  [ STX  | NUL × N   | ETX  ]
  total tokens  = N + 2
```

The **mask region M covers only the N NUL slots**. STX and ETX are treated as source-like (causal, visible to everything after them).

Slot differentiation comes from **depth + causal asymmetry** (NoPE): slot k sees k more tokens than slot 0, producing distinct attention patterns at every layer. This is the correct place to solve it — not in token IDs. If collapse occurs, increase n_layers.

**Why not sequential bytes `0x00, 0x01, ..., 0x{N-1}`:**
- Breaks if N exceeds training max (slots beyond max have embeddings trained only in data context)
- Collides with data if corpus contains control bytes
- Encodes N into the token sequence, making the model N-dependent
- False solution: slot diversity should come from model depth, not input encoding

### 4.4 Generalization properties

| Property | Sequential (Option B) | NUL + delimiters (adopted) |
|---|---|---|
| N generalization beyond training max | Breaks | Fully general |
| Corpus collision risk | Yes (`0x00–0x1F` in data) | No (data constrained to `0x20+`) |
| New segment types (SRS, self-eval, etc.) | No mechanism | Add open/close pair |
| Slot collapse fix | Fragile (baked into IDs) | Architecture (n_layers) |

### 4.5 Variable N during training

N is **randomized per batch** from `N_set = {2, 4, 8, 16, 32}`. This forces the model to learn compression at multiple granularities in a single training run.

Implementation:
- At startup, precompute `mask_cache[N]` for each N in `N_set` (masks are static numpy arrays, cheap)
- Each training step: sample `N ~ Uniform(N_set)`, fetch `mask_cache[N]`, build batch with that N
- Sequence length varies per step: `L = L_S + (N+2) + L_y` (the +2 is STX+ETX)
- Pass mask as a traced array into jit (preferred over `static_argnums` — avoids recompilation per N)

At **eval**, fix N to each value in `N_set` and measure the sweep.

### 4.6 N range justification

| N | KV floats (all layers) | vs model params |
|---|---|---|
| 2  | `2 × L_layers × 2 × d` | tiny |
| 4  | `2 × L_layers × 4 × d` | — |
| 8  | `2 × L_layers × 8 × d` | — |
| 16 | `2 × L_layers × 16 × d` | — |
| 32 | `2 × L_layers × 32 × d` | — |

With defaults `d=128, L_layers=4`:
- N=2:  **2,048 floats** (8 KB @ fp32)
- N=8:  **8,192 floats** (32 KB)
- N=32: **32,768 floats** (128 KB)

---

## 5. KV Cache Size Rule of Thumb

The KV cache at inference holds K and V tensors for each layer at each memory position:

```
KV_floats = 2 × n_layers × N × d
KV_bytes  = KV_floats × 4          (float32)
KV_bytes  = KV_floats × 2          (bfloat16)
```

Derivation: each layer stores `K ∈ R^{N×d}` and `V ∈ R^{N×d}`, so `2Nd` floats per layer, times `n_layers`.

**Comparison to model weight count:**

For our default model (`V=256, d=128, H=4, d_ff=512, n_layers=4`):

| Component | Params |
|---|---|
| Embedding `V×d` | 32,768 |
| Per-layer attn `4d²` (Q,K,V,O) | 65,536 |
| Per-layer FFN `2·d·d_ff` | 131,072 |
| Output head `V×d` | 32,768 (tied or separate) |
| **Total model** | **~820K** |

| N | KV floats | % of model params |
|---|---|---|
| 2  | 2,048  | 0.25% |
| 4  | 4,096  | 0.50% |
| 8  | 8,192  | 1.0% |
| 16 | 16,384 | 2.0% |
| 32 | 32,768 | 4.0% |
| 64 | 65,536 | 8.0% |

**Rule of thumb**: `KV_params ≈ 2·L·N·d`. For a 4-layer 128-dim model, each memory slot costs **1024 floats** (4 KB fp32). The KV cache is a very small fraction of model size at practical N values.

---

## 6. Common Architecture

### 6.1 Input embedding (NoPE)

```
h^(0)_i = E[z_i]      # token embedding only, no positional term
```
`E ∈ R^{V × d}`, learned. `V = 256`.

### 6.2 Transformer block (pre-norm)

```
h_hat   = LayerNorm(h^(ℓ-1))
Q,K,V   = h_hat @ W_Q, h_hat @ W_K, h_hat @ W_V

A_ij^h  = (Q_i^h · K_j^h) / sqrt(d_h) + mask_ij
alpha^h = softmax_j(A^h)
o_i^h   = sum_j alpha^h_ij · V_j^h

o       = Concat_h(o^h) @ W_O
u       = h^(ℓ-1) + o
h^(ℓ)   = u + W_2 · GELU(W_1 · LayerNorm(u))
```

### 6.3 Output

```
logits_i = LayerNorm(h^(n_layers)_i) @ W_out^T
```

### 6.4 Default hyperparameters

| Param | Value |
|---|---|
| `V` | 256 |
| `L_S` | 128 (fixed throughout training) |
| `L_y` | **curriculum-scheduled** — grows from 8 → 128 over training (see §6.5) |
| `N_set` | {2, 4, 8, 16, 32} (sampled per batch during training) |
| `T` | 1 (stage 0), sweep {1, 2, 4, 8} (stage 1) |
| `d` | 128 |
| `n_layers` | 4 (increase to 8 if slot collapse) |
| `H` | 4 (`d_h = 32`) |
| `d_ff` | 512 |
| `lambda_cont` | 2.0 |
| Batch size | 64 |
| Optimizer | AdamW (hand-rolled), lr=1e-3, warmup 1000 steps, cosine to 1e-5 |
| Gradient clip | 1.0 |
| Training steps | 50k |

### 6.5 Continuation length curriculum

The model is trained with a curriculum over `L_y` (continuation length). This forces progressive compression: easy tasks build the basic bottleneck, hard tasks force near-lossless encoding.

**Discrete schedule** — three phases, each with a fixed `L_y`:

| Phase | Steps | `L_y` | Compression task |
|---|---|---|---|
| Easy   | 0–15k  | 8   | Predict 8 tokens from N-slot KV (16:1 ratio at N=8) |
| Medium | 15k–35k| 32  | Predict 32 tokens — standard difficulty |
| Hard   | 35k–50k| 128 | Predict `L_S = 128` tokens — forces near-lossless KV compression |

`y` is always an **independent continuation** from the Markov chain (not reconstruction of `x_S`). At `L_y = L_S = 128`, `y` is a fresh 128-token walk from the terminal state — the model must store enough structure about the chain to predict an equally long continuation. This is strictly harder than reconstruction because there is no shortcut copy path.

Masks are precomputed at startup for the Cartesian product `N_set × L_y_set = {2,4,8,16,32} × {8,32,128}` — 15 masks total. Each step fetches `mask_cache[(N, L_y)]`.

**Why this curriculum:**
- At `L_y = 8`, even weak compression is sufficient — the model learns the basic write/read mechanism
- At `L_y = 32`, meaningful predictive structure must be stored — the default prior work benchmark
- At `L_y = 128 = L_S`, the continuation is as long as the source — the KV must encode the full statistical fingerprint of the chain to beat the uniform baseline by a meaningful margin

---

## 7. Stage 0 — Single-pass NTP through the bottleneck

### 7.1 Sequence layout

```
z = [ x_S (L_S) | STX | NUL×N | ETX | y (L_y) ]

S        = [0,       L_S)          source (L_S = 128, fixed)
STX_pos  =  L_S                    open delimiter
M        = [L_S+1,   L_S+1+N)     N inner memory slots — the KV region
ETX_pos  =  L_S+1+N               close delimiter
Y        = [L_S+2+N, L_S+2+N+L_y) continuation (L_y varies by curriculum phase)

Total length L = L_S + 2 + N + L_y
```

Token values:
- `x_S`: Markov chain bytes over `[0x20, 0xFF]`, length `L_S = 128`
- `STX = 0x02`, `NUL = 0x00`, `ETX = 0x03`
- `y`: independent continuation of same chain from terminal state, length `L_y`
- N sampled from `N_set = {2, 4, 8, 16, 32}` per batch
- L_y sampled from `L_y_set = {8, 32, 128}` per batch according to curriculum phase (see §6.5)

### 7.2 Attention mask

```python
def make_mask_stage0(L_S, N, L_y):
    L       = L_S + 2 + N + L_y
    M_start = L_S + 1
    Y_start = L_S + 2 + N

    rows = np.arange(L)[:, None]
    cols = np.arange(L)[None, :]

    causal          = cols <= rows
    is_S_or_STX     = cols < M_start          # S + STX col
    is_Y_col        = cols >= Y_start
    is_Y_row        = rows >= Y_start

    block_y_sink    = is_Y_col & ~is_Y_row    # Y is write-only
    block_y_sees_s  = is_Y_row & is_S_or_STX  # Y cannot see S or STX

    blocked = block_y_sink | block_y_sees_s
    visible = causal & ~blocked
    return np.where(visible, 0.0, -1e9).astype(np.float32)
```

Visibility: Y sees M, ETX, and causal-Y. Y does NOT see S or STX.

### 7.3 Loss

```
mask_src[i]  = 1  if  0 <= i <= L_S - 2
mask_cont[i] = 1  if  ETX_pos <= i <= L - 2

L_src  = mean NLL over src positions
L_cont = mean NLL over cont positions
L      = L_src + lambda_cont * L_cont
```

At `L_y = 128`, `mask_cont` spans 128 positions — as many as the source. This is the hardest curriculum level.

### 7.4 Curriculum mask cache

At startup, precompute `mask_cache[(N, L_y)]` for all `N ∈ {2,4,8,16,32}` × `L_y ∈ {8,32,128}` = 15 masks. Fetch by `(N, L_y)` each step.

### 7.5 Evaluation conditions

For each `(N, L_y)` in the eval grid, evaluate three conditions:

- **matched**: `y` continues same chain as `x_S`
- **cross**: `y` continues a different chain
- **uniform**: `x_S` is random bytes; `y` from a fresh chain

```
bpt(condition, N, L_y) = mean_cont_NLL / log(2)
gain(N, L_y) = bpt_uniform - bpt_matched
```

Primary eval uses `L_y = 128` (hardest) as the headline number.

### 7.5 Success criteria

| Criterion | Required |
|---|---|
| `bpt_matched < bpt_uniform` | Yes |
| `bpt_cross > bpt_uniform` | Yes |
| `bpt_matched` decreasing as N grows | Trend |

### 7.6 Sweep deliverable

Plot `bpt_matched, bpt_cross, bpt_uniform` vs `N ∈ {2, 4, 8, 16, 32}`.

---

## 8. Stage 1 — Multi-pass NTP refinement

### 8.1 Sequence layout

```
z = [ x_S | MB | y^(1) | MB | y^(2) | ... | MB | y^(T) ]

where MB = [ STX | 0x00..0x{N-1} | ETX ]  (N+2 tokens)

Total length L = L_S + T*(N+2+L_y)
```

Each `y^(t)` is an **independent fresh continuation** (same chain, different walk, same terminal state).

### 8.2 Attention mask

```python
def make_mask_stage1(L_S, N, L_y, T):
    block_size = N + 2 + L_y   # STX + inner + ETX + y
    L = L_S + T * block_size

    # For each position, compute which pass it belongs to and whether it's M or Y
    # M^(t) attends to: S, STX/ETX (own block), M^(<=t), own causal
    # Y^(t) attends to: M^(<=t), ETX of own block, own causal
    # Y^(t) does NOT see: S, other Y^(s), M^(>t)

    block_y_writeonly = ...    # Y cols blocked for non-Y rows
    block_y_sees_S    = ...    # Y rows can't see S
    block_y_cross     = ...    # Y rows can't see other Y blocks
    ...
```

### 8.3 Loss with pass weighting

```
beta_t = t / sum(1..T)    # linear ramp

L_cont_t = mean NLL over Y^(t) positions
L = L_src + lambda_cont * sum_t (beta_t * L_cont_t)
```

### 8.4 Success criteria

| Criterion | Required |
|---|---|
| Monotone `bpt_cont(t+1) ≤ bpt_cont(t)` | Yes |
| `bpt_cont(T=4) < bpt_cont(T=1)` by ≥ 0.1 bits | Yes |
| Truncation diagnostic: T=1 truncated ≥ dedicated T=1 | Yes |

---

## 9. File Layout

```
kvmem/
├── data.py      # Markov chain dataset (byte-level, V=256, variable N)
├── stage0.py    # Stage 0: single-pass, variable-N training, N sweep eval
└── stage1.py    # Stage 1: multi-pass, T sweep, truncation diagnostic
```

All model code (Equinox modules, mask builders, AdamW, training loop) lives inside each stage file.

### 9.1 Dependencies

- `jax`
- `equinox`
- `numpy`
- `matplotlib`

No optax. No flax.

### 9.2 AdamW (hand-rolled)

```python
def adam_step(params, grads, opt_state, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    m, v, step = opt_state
    m     = jax.tree.map(lambda m_, g: b1*m_ + (1-b1)*g,    m, grads)
    v     = jax.tree.map(lambda v_, g: b2*v_ + (1-b2)*g**2, v, grads)
    m_hat = jax.tree.map(lambda m_: m_ / (1 - b1**step), m)
    v_hat = jax.tree.map(lambda v_: v_ / (1 - b2**step), v)
    params = jax.tree.map(
        lambda p, mh, vh: p - lr * (mh / (jnp.sqrt(vh) + eps) + wd * p),
        params, m_hat, v_hat
    )
    return params, (m, v, step + 1)
```

LR: linear warmup 1000 steps → cosine decay to 1e-5 over 50k steps.

### 9.3 Things NOT to do

- No position embeddings (NoPE)
- No strength head / capacity penalty
- No single repeated marker token — use sequential bytes `0x00..0x{N-1}`
- No autoencoding target — always independent continuation `y`
- No M attending to Y
- No cross-Y attention in stage 1

### 9.4 Memory slot diversity diagnostic

```python
def slot_diversity(model, tokens, L_S, N):
    h = get_final_hidden(model, tokens)      # (L, d)
    M_h = h[L_S+1 : L_S+1+N]               # inner slots only, (N, d)
    M_n = M_h / jnp.linalg.norm(M_h, axis=-1, keepdims=True)
    return M_n @ M_n.T                       # (N, N) cosine sim
```

Off-diagonal average should be < 0.9. If collapse: increase n_layers to 8.

---

## 10. Future Stages (not yet)

Stage 2: Streaming (multiple chunks).
Stage 3: Random-schedule rehearsal.
Stage 4: Self-evaluation head.
Stage 5: SRS controller.
Stage 6: Backprop baseline.
Stage 7: Real text (byte-level, e.g. Quran text already in repo).

---

## 11. Acceptance Summary

1. `data.py` — byte-level Markov chains, variable N, bracket memory tokens
2. `stage0.py` — variable-N training, N sweep eval, bpt curves
3. `stage1.py` — T sweep, per-pass bpt, truncation diagnostic
4. `reports/stage01_summary.md` — go/no-go for stage 2
