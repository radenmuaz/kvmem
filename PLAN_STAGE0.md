# Stage 0 — Single-pass NTP through the Bottleneck

Detailed implementation spec. Read PLAN.md first for architectural decisions and notation.

---

## 1. Goal

Validate that N memory positions (KV slots) can encode a useful predictive prior over L_S source tokens, measured by continuation perplexity through the bottleneck. A single training run handles all N values simultaneously by randomizing N per batch.

---

## 2. Sequence Layout

```
z = [ x_S | STX | NUL×N | ETX | y ]
      L_S    1     N       1    L_y

Position sets:
  S        = [0,          L_S)          source
  STX_pos  =  L_S                       open delimiter (1 token)
  M        = [L_S+1,      L_S+1+N)     inner memory slots (KV region)
  ETX_pos  =  L_S+1+N                  close delimiter (1 token)
  Y        = [L_S+2+N,    L_S+2+N+L_y) continuation

Total length L = L_S + 2 + N + L_y
```

Token values:
- `x_S`: bytes sampled from Markov chain over `[0x20, 0xFF]`
- `STX`: `0x02` (fixed)
- `NUL×N`: `[0x00] * N` (all NUL, N varies per batch)
- `ETX`: `0x03` (fixed)
- `y`: bytes from independent continuation of same chain, starting from `x_S[-1]`

N is drawn uniformly from `N_set = {2, 4, 8, 16, 32}` each batch.

---

## 3. Attention Mask

Masking rules on top of causal (lower-triangular):

1. **Y is a write-only sink**: no position outside Y can attend to Y positions
2. **Y cannot see S**: bottleneck — Y reads only through M
3. STX and ETX are treated like S (causal, unrestricted)

```python
def make_mask_stage0(L_S: int, N: int, L_y: int) -> np.ndarray:
    """
    Returns (L, L) float mask with 0.0 (attend) or -1e9 (block).
    L = L_S + 2 + N + L_y
    """
    L       = L_S + 2 + N + L_y
    M_start = L_S + 1
    M_end   = L_S + 1 + N
    Y_start = L_S + 2 + N

    rows = np.arange(L)[:, None]
    cols = np.arange(L)[None, :]

    causal         = cols <= rows                            # lower-triangular
    is_S_or_delim  = cols < M_start                         # S + STX (col)
    is_Y_col       = cols >= Y_start                        # Y positions as key
    is_Y_row       = rows >= Y_start                        # Y positions as query

    # Rule 1: Y is write-only — block attending TO Y from outside Y
    block_y_sink    = is_Y_col & ~is_Y_row

    # Rule 2: Y cannot attend to S or delimiters before M
    block_y_sees_s  = is_Y_row & is_S_or_delim

    # M can see S, STX, earlier M, ETX (causal). No extra blocks needed for M.
    # Y can see M and ETX (causal, not blocked by rule 2 since M_start > L_S).

    blocked = block_y_sink | block_y_sees_s
    visible = causal & ~blocked
    return np.where(visible, 0.0, -1e9).astype(np.float32)
```

Visibility summary per region:

| Query \ Key | S | STX | M | ETX | Y |
|---|---|---|---|---|---|
| S   | causal | ✓ | ✗ (future) | ✗ | ✗ |
| STX | causal | ✓ | ✗ | ✗ | ✗ |
| M   | ✓ all  | ✓ | causal | ✗ | ✗ |
| ETX | ✓ all  | ✓ | ✓ all | ✓ | ✗ |
| Y   | ✗      | ✗ | ✓ all | ✓ | causal |

(✓ all = all positions up to current, ✗ = always blocked, causal = only positions ≤ own index)

---

## 4. Model Architecture

Single file `stage0.py` with all model code.

### 4.1 Equinox modules

```python
class MHAttention(eqx.Module):
    W_Q: Array   # (d, d)
    W_K: Array   # (d, d)
    W_V: Array   # (d, d)
    W_O: Array   # (d, d)
    n_heads: int = eqx.field(static=True)

    def __call__(self, x, mask):
        # x: (L, d), mask: (L, L)
        # Returns (L, d)
        ...

class FFN(eqx.Module):
    W1: Array    # (d, d_ff)
    W2: Array    # (d_ff, d)

    def __call__(self, x):
        return x @ self.W2.T @ jax.nn.gelu(x @ self.W1.T)
        # Note: no bias

class TransformerBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn:  MHAttention
    norm2: eqx.nn.LayerNorm
    ffn:   FFN

    def __call__(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class KVMemModel(eqx.Module):
    embed:    eqx.nn.Embedding  # (V, d)
    blocks:   list              # n_layers × TransformerBlock
    norm_out: eqx.nn.LayerNorm
    W_out:    Array             # (V, d) — untied from embed

    def __call__(self, tokens, mask):
        # tokens: (L,) int32
        # mask:   (L, L) float32
        x = self.embed(tokens)                       # (L, d)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm_out(x) @ self.W_out.T       # (L, V) logits
```

### 4.2 Hyperparameters

```python
V        = 256
L_S      = 96
L_y      = 32
N_set    = [2, 4, 8, 16, 32]
d        = 128
n_layers = 4
n_heads  = 4        # d_h = 32
d_ff     = 512
lambda_cont = 2.0
B        = 64
lr_max   = 1e-3
lr_min   = 1e-5
warmup   = 1000
n_steps  = 50_000
grad_clip = 1.0
wd       = 0.01

# Segment protocol bytes
STX = 0x02
ETX = 0x03
NUL = 0x00
```

---

## 5. Loss

```python
def loss_fn(model, tokens, mask, L_S, N, lambda_cont):
    """
    tokens: (B, L) int32
    mask:   (L, L) float32
    """
    B, L = tokens.shape

    # Forward pass (vmap over batch)
    logits = jax.vmap(lambda tok: model(tok, mask))(tokens)  # (B, L, V)

    # Shift: predict token i+1 from position i
    logits_s = logits[:, :-1, :]      # (B, L-1, V)
    targets   = tokens[:, 1:]          # (B, L-1)

    # Per-token NLL
    log_probs = jax.nn.log_softmax(logits_s, axis=-1)
    nll = -log_probs[jnp.arange(B)[:, None], jnp.arange(L-1)[None, :], targets]
    # nll: (B, L-1)

    # Position masks (over L-1 shifted positions)
    pos = jnp.arange(L - 1)

    # Source: positions 0 .. L_S-2  (predicts tokens 1 .. L_S-1)
    mask_src  = (pos >= 0) & (pos <= L_S - 2)

    # Continuation: positions ETX_pos .. L-2
    #   ETX_pos = L_S + 1 + N  (predicts first y token from ETX position)
    ETX_pos = L_S + 1 + N
    mask_cont = (pos >= ETX_pos) & (pos <= L - 2)

    # Losses (mean over positions, then mean over batch)
    def masked_mean(x, m):
        m = m.astype(jnp.float32)
        return jnp.sum(x * m[None, :], axis=-1) / (jnp.sum(m) + 1e-8)

    L_src  = jnp.mean(masked_mean(nll, mask_src))
    L_cont = jnp.mean(masked_mean(nll, mask_cont))

    total = L_src + lambda_cont * L_cont
    return total, (L_src, L_cont)
```

---

## 6. Optimizer (hand-rolled AdamW)

```python
def init_opt_state(params):
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    return (m, v, 1)   # step starts at 1

def lr_schedule(step, lr_max, lr_min, warmup, n_steps):
    # Linear warmup then cosine decay
    warmup_lr  = lr_max * step / warmup
    cos_frac   = (step - warmup) / (n_steps - warmup)
    cos_lr     = lr_min + 0.5 * (lr_max - lr_min) * (1 + jnp.cos(jnp.pi * cos_frac))
    return jnp.where(step < warmup, warmup_lr, cos_lr)

def adam_step(params, grads, opt_state, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    m, v, step = opt_state
    m     = jax.tree.map(lambda m_, g: b1*m_ + (1-b1)*g,    m, grads)
    v     = jax.tree.map(lambda v_, g: b2*v_ + (1-b2)*g**2, v, grads)
    m_hat = jax.tree.map(lambda m_: m_ / (1.0 - b1**step), m)
    v_hat = jax.tree.map(lambda v_: v_ / (1.0 - b2**step), v)
    params = jax.tree.map(
        lambda p, mh, vh: p - lr * (mh / (jnp.sqrt(vh) + eps) + wd * p),
        params, m_hat, v_hat
    )
    return params, (m, v, step + 1)

def clip_grads(grads, max_norm=1.0):
    leaves = jax.tree.leaves(grads)
    total_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (total_norm + 1e-6))
    return jax.tree.map(lambda g: g * scale, grads)
```

---

## 7. Training Loop

```python
@jax.jit
def train_step(model, opt_state, tokens, mask, step):
    lr = lr_schedule(step, lr_max, lr_min, warmup, n_steps)
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        model, tokens, mask, L_S, N, lambda_cont
    )
    grads  = clip_grads(grads)
    params, opt_state = adam_step(model, grads, opt_state, lr)
    return params, opt_state, loss, aux
```

Per-step:
1. Sample `N ~ Uniform(N_set)`
2. Build batch: `tokens = make_batch(key, B, V, L_S, L_y, N)` from `data.py`
3. Fetch precomputed `mask = mask_cache[N]`  (shape `(L_S+2+N+L_y, L_S+2+N+L_y)`)
4. Call `train_step(model, opt_state, tokens, mask, step)`
5. Log loss every 100 steps

Note: since mask shape changes with N, jit recompiles once per unique (L_S, N, L_y) combination. With 5 values of N, this gives 5 compilations at the start, then cached forever.

---

## 8. Evaluation

### 8.1 Computing bpt

```python
def eval_bpt(model, tokens, mask, L_S, N):
    """tokens: (B, L), returns scalar bpt (bits per token) over continuation."""
    logits = jax.vmap(lambda tok: model(tok, mask))(tokens)
    logits_s = logits[:, :-1, :]
    targets   = tokens[:, 1:]
    log_probs = jax.nn.log_softmax(logits_s, axis=-1)
    nll = -log_probs[jnp.arange(B)[:, None], jnp.arange(L-1)[None, :], targets]

    ETX_pos = L_S + 1 + N
    L = tokens.shape[1]
    pos = jnp.arange(L - 1)
    mask_cont = (pos >= ETX_pos) & (pos <= L - 2)
    cont_nll = jnp.mean(jnp.sum(nll * mask_cont[None, :], axis=-1) / jnp.sum(mask_cont))
    return cont_nll / jnp.log(2.0)   # nats → bits
```

### 8.2 Eval conditions (per N)

```python
eval_batches = make_eval_batches(key, B_eval=256, V, L_S, L_y, N)
bpt_matched = eval_bpt(model, eval_batches['matched'], mask_cache[N], L_S, N)
bpt_cross   = eval_bpt(model, eval_batches['cross'],   mask_cache[N], L_S, N)
bpt_uniform = eval_bpt(model, eval_batches['uniform'], mask_cache[N], L_S, N)
gain        = bpt_uniform - bpt_matched
```

Run for each `N ∈ N_set` after training.

### 8.3 Slot diversity diagnostic

```python
def slot_diversity(model, tokens_1d, L_S, N):
    """tokens_1d: (L,) single example. Returns (N, N) cosine sim matrix."""
    # Get final hidden states — requires model with return_hidden=True variant
    h = get_hidden(model, tokens_1d)   # (L, d)
    M_h = h[L_S+1 : L_S+1+N]          # inner NUL slots only, (N, d)
    M_n = M_h / jnp.linalg.norm(M_h, axis=-1, keepdims=True)
    return M_n @ M_n.T                  # (N, N)
```

Off-diagonal average < 0.9 required. If > 0.9, increase n_layers to 8.

---

## 9. Outputs and Plots

After training and eval:

1. **`reports/stage0_bpt_sweep.png`** — `bpt_matched, bpt_cross, bpt_uniform` vs N ∈ {2,4,8,16,32}
2. **`reports/stage0_training_loss.png`** — total loss, L_src, L_cont vs step
3. **`reports/stage0_slot_diversity_N8.png`** — heatmap of (N,N) cosine sim at N=8
4. **`reports/stage0_summary.md`** — numbers + go/no-go assessment

---

## 10. Success Criteria

| Criterion | Required |
|---|---|
| `bpt_matched < bpt_uniform` for all N | Yes |
| `bpt_cross > bpt_uniform` for all N | Yes |
| `bpt_matched` trend decreases as N grows | Yes |
| Off-diagonal slot cosine sim < 0.9 at N=8 | Yes |
| Training loss converges (no NaN, no plateau at step 0 level) | Yes |

If `bpt_cross ≤ bpt_uniform`: the bottleneck mask is not being used — debug mask construction first.
If `bpt_matched ≥ bpt_uniform`: memory carries no useful info — check Y cannot see S in mask.

---

## 11. File: `kvmem/stage0.py`

All of the following in one file:
- `MHAttention`, `FFN`, `TransformerBlock`, `KVMemModel` (Equinox)
- `make_mask_stage0(L_S, N, L_y)` → numpy array
- `loss_fn`, `train_step`
- `init_opt_state`, `adam_step`, `clip_grads`, `lr_schedule`
- `eval_bpt`, `slot_diversity`
- `main()` — training loop, eval sweep, plots

Imports from `kvmem/data.py`: `make_batch`, `make_eval_batches`.
