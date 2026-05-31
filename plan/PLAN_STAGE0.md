# Stage 0 — Single-pass NTP through the Bottleneck

Detailed implementation spec. Read PLAN.md first for architectural decisions and notation.

---

## 1. Goal

Validate that N memory positions (KV slots) can encode a useful predictive prior over L_S source tokens, measured by continuation perplexity through the bottleneck.

Two simultaneous training axes:
- **Variable N** per batch (randomized from `N_set`) — model learns compression at multiple granularities
- **Curriculum over L_y** (continuation length) — gradually harder prediction task forces increasingly rich compression

Qualitative test: after training, memorize Surah Al-Fatihah (single verse or whole file) and attempt verse completion from a short byte-level warmup prefix.

---

## 2. Sequence Layout

```
z = [ x_S | STX | NUL×N | ETX | y ]
      L_S    1     N       1    L_y

Position sets:
  S        = [0,          L_S)           source  (L_S = 128, fixed)
  STX_pos  =  L_S                        open delimiter (1 token, byte 0x02)
  M        = [L_S+1,      L_S+1+N)      inner memory slots — the KV region
  ETX_pos  =  L_S+1+N                   close delimiter (1 token, byte 0x03)
  Y        = [L_S+2+N,    L_S+2+N+L_y)  continuation

Total length L = L_S + 2 + N + L_y
```

**Per-batch sampling:**
- `N` drawn uniformly from `N_set = {2, 4, 8, 16, 32}`
- `L_y` drawn from `L_y_set = {8, 32, 128}` according to curriculum phase (§3)

Token values:
- `x_S`: bytes from Markov chain over `[0x20, 0xFF]`, length 128
- `STX = 0x02`, `NUL = 0x00` (×N), `ETX = 0x03`
- `y`: **independent continuation** from same chain, from terminal state of `x_S`, length `L_y`

Data bytes are in `[0x20, 0xFF]` — no collision with protocol bytes `0x00–0x1F`. Al-Fatihah UTF-8 bytes are confirmed to be all ≥ 0x20.

---

## 3. Continuation Length Curriculum

Training uses a three-phase schedule over `L_y`. Each phase has a fixed `L_y` for all batches in that range:

| Phase | Steps | `L_y` | What must the KV encode? |
|---|---|---|---|
| **Easy** | 0 – 15k | 8 | Just the chain's next 8 tokens — very coarse fingerprint |
| **Medium** | 15k – 35k | 32 | Meaningful predictive structure — default benchmark level |
| **Hard** | 35k – 50k | 128 | Full `L_S`-length continuation — KV must capture the whole chain's statistics |

```python
L_y_schedule = [(0, 8), (15_000, 32), (35_000, 128)]  # (start_step, L_y)

def get_L_y(step):
    L_y = 8
    for start, val in L_y_schedule:
        if step >= start:
            L_y = val
    return L_y
```

**Why L_y = 128 = L_S is the hardest level:**  
The continuation is as long as the source. The model cannot predict it from the uniform baseline — it must store the full Markov transition structure of the chain in N slots. At N=2 this is an extreme bottleneck (2 slots for 128-token prediction); at N=32 it should be tractable.

`y` is always an **independent continuation** (fresh walk from terminal state), never reconstruction of `x_S`. The hardest level is harder than autoencoding because there is no copy shortcut — the chain's future must be predicted, not its past recovered.

---

## 3. Attention Mask

Rules on top of causal (lower-triangular):

1. **Y is a write-only sink**: nothing outside Y can attend to Y positions
2. **Y cannot see S**: bottleneck — Y reads only through M and ETX

STX and ETX are treated like S (causal, not part of the bottleneck block).

```python
def make_mask_stage0(L_S: int, N: int, L_y: int) -> np.ndarray:
    """Returns (L, L) float32 mask: 0.0 = attend, -1e9 = block."""
    L       = L_S + 2 + N + L_y
    M_start = L_S + 1       # first NUL slot
    M_end   = L_S + 1 + N   # one past last NUL slot
    Y_start = L_S + 2 + N   # first y token

    rows = np.arange(L)[:, None]
    cols = np.arange(L)[None, :]

    causal        = cols <= rows
    is_S_or_delim = cols < M_start         # S + STX (col)
    is_Y_col      = cols >= Y_start
    is_Y_row      = rows >= Y_start

    block_y_sink  = is_Y_col & ~is_Y_row   # Y is write-only
    block_y_sees_s = is_Y_row & is_S_or_delim  # Y cannot see S or STX

    blocked = block_y_sink | block_y_sees_s
    visible = causal & ~blocked
    return np.where(visible, 0.0, -1e9).astype(np.float32)
```

Visibility table:

| Query \ Key | S | STX | M | ETX | Y |
|---|---|---|---|---|---|
| S   | causal | – | – | – | ✗ |
| STX | causal | ✓ | – | – | ✗ |
| M   | ✓ all  | ✓ | causal | – | ✗ |
| ETX | ✓ all  | ✓ | ✓ all | ✓ | ✗ |
| Y   | **✗**  | **✗** | ✓ all | ✓ | causal |

(– = blocked by causal; ✗ = explicitly blocked; ✓ = visible)

---

## 4. Model Architecture

### 4.1 Equinox modules

```python
import equinox as eqx
import jax, jax.numpy as jnp

class MHAttention(eqx.Module):
    W_Q: jax.Array   # (d, d)
    W_K: jax.Array   # (d, d)
    W_V: jax.Array   # (d, d)
    W_O: jax.Array   # (d, d)
    n_heads: int = eqx.field(static=True)
    d_head:  int = eqx.field(static=True)

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        # x: (L, d), mask: (L, L) → (L, d)
        L, d = x.shape
        Q = x @ self.W_Q.T   # (L, d)
        K = x @ self.W_K.T
        V = x @ self.W_V.T
        # split heads
        Q = Q.reshape(L, self.n_heads, self.d_head).transpose(1, 0, 2)  # (H, L, dh)
        K = K.reshape(L, self.n_heads, self.d_head).transpose(1, 0, 2)
        V = V.reshape(L, self.n_heads, self.d_head).transpose(1, 0, 2)
        scale = self.d_head ** -0.5
        attn = (Q @ K.transpose(0, 2, 1)) * scale + mask[None]  # (H, L, L)
        attn = jax.nn.softmax(attn, axis=-1)
        out = (attn @ V).transpose(1, 0, 2).reshape(L, d)       # (L, d)
        return out @ self.W_O.T

class FFN(eqx.Module):
    W1: jax.Array   # (d_ff, d)
    W2: jax.Array   # (d, d_ff)

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.gelu(x @ self.W1.T) @ self.W2.T   # no bias

class TransformerBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn:  MHAttention
    norm2: eqx.nn.LayerNorm
    ffn:   FFN

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class KVMemModel(eqx.Module):
    embed:    eqx.nn.Embedding   # (V, d)
    blocks:   list               # n_layers × TransformerBlock
    norm_out: eqx.nn.LayerNorm
    W_out:    jax.Array          # (V, d) — untied from embed

    def __call__(self, tokens: jax.Array, mask: jax.Array) -> jax.Array:
        # tokens: (L,) int32, mask: (L, L) → logits (L, V)
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm_out(x) @ self.W_out.T

    def hidden(self, tokens: jax.Array, mask: jax.Array) -> jax.Array:
        """Return final hidden states (L, d) — used for slot diversity diagnostic."""
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm_out(x)
```

### 4.2 Hyperparameters

```python
V           = 256
L_S         = 128
L_y_schedule = [(0, 8), (15_000, 32), (35_000, 128)]   # (start_step, L_y)
N_set       = [2, 4, 8, 16, 32]
d           = 128
n_layers    = 4
n_heads     = 4          # d_h = 32
d_ff        = 512
lambda_cont = 2.0
B           = 64
lr_max      = 1e-3
lr_min      = 1e-5
warmup      = 1000
n_steps     = 50_000
grad_clip   = 1.0
wd          = 0.01

# Segment protocol bytes
STX = 0x02
ETX = 0x03
NUL = 0x00
```

---

## 5. Loss

```python
def loss_fn(model, tokens, mask, L_S, N, lambda_cont):
    # tokens: (B, L), mask: (L, L)
    B, L = tokens.shape

    logits = jax.vmap(lambda tok: model(tok, mask))(tokens)  # (B, L, V)
    log_probs = jax.nn.log_softmax(logits[:, :-1], axis=-1)  # (B, L-1, V)
    targets   = tokens[:, 1:]                                  # (B, L-1)

    nll = -log_probs[jnp.arange(B)[:, None],
                     jnp.arange(L-1)[None, :],
                     targets]                                  # (B, L-1)

    pos      = jnp.arange(L - 1)
    ETX_pos  = L_S + 1 + N

    # Source NTP: positions 0 .. L_S-2
    mask_src  = (pos >= 0) & (pos <= L_S - 2)
    # Continuation NTP: ETX position predicts first y token, through end
    mask_cont = (pos >= ETX_pos) & (pos <= L - 2)

    def masked_mean(x, m):
        m = m.astype(jnp.float32)
        return jnp.sum(x * m[None, :], axis=-1) / (jnp.sum(m) + 1e-8)

    L_src  = jnp.mean(masked_mean(nll, mask_src))
    L_cont = jnp.mean(masked_mean(nll, mask_cont))
    total  = L_src + lambda_cont * L_cont
    return total, (L_src, L_cont)
```

---

## 6. Optimizer (hand-rolled AdamW)

```python
def init_opt_state(params):
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    return (m, v, 1)

def lr_schedule(step, lr_max, lr_min, warmup, n_steps):
    warmup_lr = lr_max * step / warmup
    frac      = jnp.clip((step - warmup) / (n_steps - warmup), 0.0, 1.0)
    cos_lr    = lr_min + 0.5 * (lr_max - lr_min) * (1 + jnp.cos(jnp.pi * frac))
    return jnp.where(step < warmup, warmup_lr, cos_lr)

def clip_grads(grads, max_norm=1.0):
    leaves = jax.tree.leaves(grads)
    norm   = jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))
    scale  = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree.map(lambda g: g * scale, grads)

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
```

---

## 7. Training Loop

Per step:
1. Sample `N ~ Uniform(N_set)`
2. Build batch via `make_batch(key, B, V, L_S, L_y, N)` from `data.py`
3. Fetch precomputed `mask_cache[N]`
4. `(loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(...)`
5. Clip grads, adam step, log every 100 steps

jit recompiles once per unique N (5 compilations total at startup), then cached.

---

## 8. Checkpoint: Save and Load

Save after training with `equinox.tree_serialise_leaves` + a sidecar JSON for hyperparams:

```python
import json, equinox as eqx

def save_checkpoint(path: str, model, step: int, hparams: dict):
    eqx.tree_serialise_leaves(path + ".eqx", model)
    with open(path + ".json", "w") as f:
        json.dump({**hparams, "step": step}, f, indent=2)

def load_checkpoint(path: str, model_template) -> tuple:
    with open(path + ".json") as f:
        meta = json.load(f)
    model = eqx.tree_deserialise_leaves(path + ".eqx", model_template)
    return model, meta
```

Default checkpoint path: `checkpoints/stage0_step{step}`.

---

## 9. Metrics

A complete set of information-theoretic and structural metrics, derived from three quantities: **input size** (`L_S` bytes), **KV memory size** (`N` slots × model dimension), and **output quality** (bpt on continuation).

### 9.1 Notation

```
L_S      = 128             source length in tokens/bytes
N        = memory slots    (varies: 2, 4, 8, 16, 32)
L_y      = continuation length  (varies by curriculum: 8, 32, 128)
n_layers = 4               transformer layers
d        = 128             model dimension

# KV size in floats (keys + values, all layers, all slots)
KV_floats = 2 * n_layers * N * d          # = 1024 * N

# Source size in bits (raw bytes)
S_bits    = L_S * 8                       # = 1024 bits (128 bytes)

# KV size in bits (float32)
KV_bits   = KV_floats * 32               # = 32768 * N  bits

# Chain entropy (oracle)
H_chain = chain_entropy_bits(T_mat, pi)   # bits/token, ~2-4 for random Dirichlet chain
```

---

### 9.2 Output quality metrics (bpt)

Computed on 256 eval samples per condition per `(N, L_y)` combination.

**Per-condition bpt** (bits per continuation token):
```
bpt_matched(N, L_y)  = mean cont NLL (matched  condition) / log(2)
bpt_cross(N, L_y)    = mean cont NLL (cross     condition) / log(2)
bpt_uniform(N, L_y)  = mean cont NLL (uniform   condition) / log(2)
```

**Source bpt** (sanity check — should be near chain entropy):
```
bpt_src(N, L_y)      = mean src NLL / log(2)
```

**Gain** (bits of useful info stored in KV, relative to no-memory baseline):
```
gain(N, L_y) = bpt_uniform(N, L_y) - bpt_matched(N, L_y)   [bits/token, higher = better]
```

**Penalty** (cost of wrong memory — confirms bottleneck is actually used):
```
penalty(N, L_y) = bpt_cross(N, L_y) - bpt_uniform(N, L_y)  [bits/token, must be > 0]
```

---

### 9.3 Information-theoretic compression metrics

**Compression ratio** (raw input bits vs KV bits):
```
CR(N) = S_bits / KV_bits
      = (L_S * 8) / (2 * n_layers * N * d * 32)
      = (128 * 8) / (1024 * N * 32)
      = 1024 / (32768 * N)
      = 1 / (32 * N)
```

At N=8:  CR = 1/256  (KV is 256× larger than source in bits — not compression yet; model is large)
At N=2:  CR = 1/64

Note: the KV is always larger than the source in raw bits for these model sizes. The meaningful compression is **semantic** — the KV stores the chain's transition structure (a V×V matrix) in N vectors, not the raw bytes. True bit-level compression requires smaller d or quantization (future work).

**Semantic compression ratio** (source tokens vs memory slots):
```
SCR(N) = L_S / N   [tokens per slot]
```

| N  | SCR   | Interpretation |
|----|-------|----------------|
| 2  | 64    | 64 source tokens summarized per slot |
| 4  | 32    | 32 per slot |
| 8  | 16    | 16 per slot |
| 16 | 8     | 8 per slot |
| 32 | 4     | 4 per slot |

This is the primary "compression difficulty" axis. `SCR = L_S / N` is the right lever.

**Predictive efficiency** (how much of the chain entropy is recovered from KV):
```
# Chain entropy rate (computed from known T_mat and stationary pi)
H = H_chain   # bits/token

# Baseline: model sees nothing (uniform over V=256)
bpt_max = log2(256) = 8.0

# Information captured by KV (bits/token recovered relative to uniform)
I_KV(N, L_y) = bpt_uniform(N, L_y) - bpt_matched(N, L_y)   # = gain

# Fraction of possible improvement captured (0 = useless KV, 1 = oracle)
eta(N, L_y) = I_KV(N, L_y) / (bpt_uniform(N, L_y) - H)
```

`eta` is the headline efficiency metric. It answers: "of the information that could possibly be captured (the gap between uniform and oracle), how much did the KV actually store?"

**Storage efficiency** (information gained per KV slot):
```
bits_per_slot(N, L_y) = I_KV(N, L_y) * L_y / N   [total bits captured per slot]
```

---

### 9.4 Curriculum metrics

Track per phase to confirm the curriculum is working:

```
# At end of easy phase (step ~15k, L_y=8):
gain_easy   = gain(N=8, L_y=8)

# At end of medium phase (step ~35k, L_y=32):
gain_medium = gain(N=8, L_y=32)

# At end of hard phase (step ~50k, L_y=128):
gain_hard   = gain(N=8, L_y=128)
```

Expected: `gain_easy ≥ gain_medium ≥ gain_hard` (harder task → lower gain, but `eta` should grow with training).

---

### 9.5 Slot diversity diagnostic

```python
def slot_diversity(model, tokens_1d, L_S, N):
    mask = make_mask_stage0(L_S, N, 0)    # no y needed; just run to ETX
    h    = model.hidden(tokens_1d, mask)   # (L, d)
    M_h  = h[L_S+1 : L_S+1+N]            # (N, d)
    M_n  = M_h / jnp.linalg.norm(M_h, axis=-1, keepdims=True)
    return M_n @ M_n.T                     # (N, N) cosine sim
```

Off-diagonal average < 0.9 required.

---

### 9.6 Model size metrics

Report alongside every eval run so results can be reproduced at different scales.

```python
def count_params(model) -> dict:
    """Count parameters by component."""
    leaves = jax.tree.leaves(eqx.filter(model, eqx.is_array))
    total  = sum(x.size for x in leaves)

    embed_params  = model.embed.weight.size                   # V * d
    block_params  = sum(
        sum(x.size for x in jax.tree.leaves(eqx.filter(b, eqx.is_array)))
        for b in model.blocks
    )
    head_params   = model.W_out.size + sum(            # output head + final norm
        x.size for x in jax.tree.leaves(eqx.filter(model.norm_out, eqx.is_array))
    )
    return {
        "total":       total,
        "embedding":   embed_params,
        "blocks":      block_params,
        "output_head": head_params,
        # KV memory at each N (floats, not params — for comparison)
        "kv_floats_per_N": {N: 2 * n_layers * N * d for N in N_set},
    }
```

Standard report line printed each eval:
```
Model: 820K params  |  KV@N=8: 8192 floats (1.0% of params)  |  SCR@N=8: 16 tok/slot
```

**Per-layer breakdown** (for debugging slot collapse):
```
attn per layer:  4 * d² = 65,536   (Q, K, V, O projections)
ffn  per layer:  2 * d * d_ff = 131,072
norm per layer:  2 * d = 256       (two LayerNorms)
total per layer: ~196,864
```

---

### 9.7 Backprop baseline

A second model trained **without** the KV bottleneck: standard causal LM that sees the full source `x_S` and continues `y` with full causal attention. This is the oracle upper bound — the best possible `bpt_matched` given the model architecture.

```
Baseline sequence: [ x_S | y ]
Baseline mask:     standard lower-triangular causal
```

The baseline model is **identical in architecture and size** — same d, n_layers, H, d_ff. Only the mask and the training objective differ: no STX/ETX/NUL, no bottleneck, Y sees S directly.

```python
def make_mask_baseline(L_S, L_y):
    """Standard causal mask — Y sees everything."""
    L    = L_S + L_y
    rows = np.arange(L)[:, None]
    cols = np.arange(L)[None, :]
    return np.where(cols <= rows, 0.0, -1e9).astype(np.float32)

def loss_fn_baseline(model, tokens, mask, L_S, L_y, lambda_cont):
    """Same loss structure but cont mask starts at L_S, no memory tokens."""
    B, L  = tokens.shape
    logits = jax.vmap(lambda tok: model(tok, mask))(tokens)
    lp     = jax.nn.log_softmax(logits[:, :-1], axis=-1)
    targets = tokens[:, 1:]
    nll    = -lp[jnp.arange(B)[:, None], jnp.arange(L-1)[None, :], targets]
    pos    = jnp.arange(L - 1)
    mask_src  = pos <= L_S - 2
    mask_cont = (pos >= L_S - 1) & (pos <= L - 2)  # L_S-1 predicts first y token

    def masked_mean(x, m):
        m = m.astype(jnp.float32)
        return jnp.sum(x * m[None, :], axis=-1) / (jnp.sum(m) + 1e-8)

    L_src  = jnp.mean(masked_mean(nll, mask_src))
    L_cont = jnp.mean(masked_mean(nll, mask_cont))
    return L_src + lambda_cont * L_cont, (L_src, L_cont)
```

**Training the baseline:** same hyperparameters, same steps, same curriculum `(L_S, L_y)` schedule. Identical data (`x_S`, `y` pairs from same Markov chains). Different mask only.

**Comparison table** (headline results):

| Model | N | bpt_matched | bpt_uniform | gain | eta |
|---|---|---|---|---|---|
| Backprop baseline | ∞ (sees all) | — | — | — | 1.0 (oracle) |
| KV bottleneck | 32 | — | — | — | — |
| KV bottleneck | 8  | — | — | — | — |
| KV bottleneck | 2  | — | — | — | — |

`eta = gain / (bpt_uniform - bpt_baseline)` with backprop baseline replacing chain entropy H as the oracle.

Both models live in `stage0.py`. Training flags:
```bash
python -m kvmem.stage0 train               # KV bottleneck model
python -m kvmem.stage0 train --baseline    # backprop baseline (no bottleneck)
```

---

## 10. Qualitative Test: Surah Al-Fatihah

### 10.1 Dataset

File: `datasets/quran_uthmani.txt`
Lines 0–6 are Surah Al-Fatihah (7 ayat). One verse per line. Each line is UTF-8, bytes confirmed all in `[0x20, 0xFF]` — no collision with protocol bytes.

```python
FATIHAH_PATH = "datasets/quran_uthmani.txt"

def load_fatihah() -> list[bytes]:
    with open(FATIHAH_PATH, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]
    return [line.encode("utf-8") for line in lines[:7]]
```

### 10.2 Inference task: memorize a verse → complete it from a warm-up prefix

**Task definition** (one verse per inference call):

```
1. Pick one ayah as x_S  (source to memorize)
2. Take its first W bytes as the warm-up prompt  (W ≥ 2, default 4)
3. Memorize x_S through the KV bottleneck
4. Feed the warm-up bytes after ETX
5. Greedily decode until newline or max_len bytes
6. Compare generated bytes to the remainder of the same ayah
```

This tests whether the KV memory of the *full* ayah helps the model reconstruct the verse given only a short byte-level hint — a form of associative recall.

```
Sequence during generation step k:
  [x_S | STX | NUL×N | ETX | warmup_bytes | generated_so_far]
   L_S   1     N       1     W               k

  mask = make_mask_stage0(L_S, N, L_y = W + k)
  next_byte = argmax(logits[-1])
```

Since M never attends to Y (train-inference consistency), this is equivalent to full separate-KV-cache inference.

### 10.3 Inference function

```python
def run_fatihah_inference(
    model,
    line: int = -1,          # -1 = random; 0–6 = specific ayah
    N: int = 8,
    warmup_bytes: int = 4,   # W: how many leading bytes of the verse to give as hint
    max_len: int = 300,       # max generated bytes
    temperature: float = 0.0, # 0.0 = greedy
    seed: int = 0,
):
    """
    Memorize one ayah of Al-Fatihah, give first W bytes as warm-up,
    attempt to complete the rest of the verse from KV memory alone.

    Default mode (line=-1): random ayah selected each call.
    """
    ayat = load_fatihah()    # list[bytes], 7 items
    if line == -1:
        line = int(jax.random.randint(jax.random.PRNGKey(seed), (), 0, 7))

    verse  = ayat[line]      # bytes for full ayah
    x_S    = list(verse)     # full ayah as source
    L_S    = len(x_S)

    warmup = list(verse[:warmup_bytes])   # first W bytes as prompt
    target = verse[warmup_bytes:]          # remaining bytes to compare against

    # Incremental decode
    STX_tok = STX   # 0x02
    ETX_tok = ETX   # 0x03
    mem_block = [STX_tok] + [NUL] * N + [ETX_tok]

    generated = list(warmup)
    key = jax.random.PRNGKey(seed)
    for step in range(max_len):
        L_y  = len(generated)
        mask = make_mask_stage0(L_S, N, L_y)
        cur  = jnp.array(x_S + mem_block + generated, dtype=jnp.int32)
        logits = model(cur, mask)          # (L, V)
        next_logit = logits[-1]            # (V,)
        if temperature == 0.0:
            next_byte = int(jnp.argmax(next_logit))
        else:
            key, subkey = jax.random.split(key)
            probs     = jax.nn.softmax(next_logit / temperature)
            next_byte = int(jax.random.choice(subkey, 256, p=probs))
        generated.append(next_byte)
        if next_byte == 0x0A or next_byte == 0x00:
            break   # newline or NUL = stop

    # Strip the warmup prefix back out for display
    full_gen  = bytes(generated)
    gen_after = bytes(generated[warmup_bytes:])

    print(f"\n{'='*60}")
    print(f"Ayah {line}  (N={N}, warmup={warmup_bytes} bytes)")
    print(f"  Full verse : {verse.decode('utf-8', errors='replace')}")
    print(f"  Warmup     : {bytes(warmup).decode('utf-8', errors='replace')!r}")
    print(f"  Generated  : {full_gen.decode('utf-8', errors='replace')}")
    print(f"  Target tail: {target.decode('utf-8', errors='replace')}")
    # Byte-level overlap as rough match metric
    min_len  = min(len(gen_after), len(target))
    matches  = sum(a == b for a, b in zip(gen_after, target))
    print(f"  Byte match : {matches}/{min_len} ({100*matches/max(min_len,1):.1f}%)")
```

### 10.4 Whole-file memorization mode

A second inference mode ingests **the entire file** (or all 7 ayat) as the source `x_S`, compresses it into the KV, then acts as a conditional LM for completion given a short prompt.

**Key constraint**: `L_S` must fit in `L_S` tokens. For Al-Fatihah (7 ayat, ~556 UTF-8 bytes), this fits comfortably in `L_S=600`. For larger files, chunk into segments (stage 2+ concern).

```
Whole-file memorization:
  x_S = bytes of entire file (all 7 ayat, with newlines between)
  L_S = len(x_S)

  Build: [x_S | STX | NUL×N | ETX | warmup_bytes]
  mask = make_mask_stage0(L_S, N, L_y=len(warmup_bytes))
  Decode greedily from position ETX + len(warmup_bytes) onward
```

```python
def run_file_memorize_infer(
    model,
    filepath: str,
    N: int = 8,
    warmup: bytes = b"",        # prompt prefix — can be empty or any byte prefix
    max_len: int = 300,
    temperature: float = 0.0,
    seed: int = 0,
):
    """
    Read entire file as x_S, compress into KV of N slots,
    then complete from `warmup` bytes as a conditional LM.

    If warmup is empty, the model generates freely conditioned on the KV memory.
    If warmup is a known prefix (e.g. first bytes of an ayah), it attempts completion.
    """
    with open(filepath, "rb") as f:
        file_bytes = f.read().rstrip(b"\n")  # strip trailing newline only
    # Ensure no protocol bytes in data (assert all bytes >= 0x20)
    assert all(b >= 0x20 for b in file_bytes), "File contains protocol bytes < 0x20"

    x_S      = list(file_bytes)
    L_S      = len(x_S)
    mem_block = [STX] + [NUL] * N + [ETX]
    prompt    = list(warmup)

    generated = list(prompt)
    key = jax.random.PRNGKey(seed)
    for _ in range(max_len):
        L_y   = len(generated)
        mask  = make_mask_stage0(L_S, N, L_y)
        cur   = jnp.array(x_S + mem_block + generated, dtype=jnp.int32)
        logits = model(cur, mask)
        nxt = logits[-1]
        if temperature == 0.0:
            next_byte = int(jnp.argmax(nxt))
        else:
            key, sk = jax.random.split(key)
            next_byte = int(jax.random.choice(sk, 256, p=jax.nn.softmax(nxt / temperature)))
        generated.append(next_byte)
        if next_byte == 0x0A and len(generated) > len(prompt) + 2:
            break  # generated a full line

    gen_text    = bytes(generated).decode("utf-8", errors="replace")
    warmup_text = warmup.decode("utf-8", errors="replace")
    print(f"\n{'='*60}")
    print(f"File memorized : {filepath}  ({L_S} bytes, N={N} slots)")
    print(f"Warmup prompt  : {warmup_text!r}")
    print(f"Generated      : {gen_text}")
```

**Default warmup selection**: pick the first 2–4 bytes of any line in the file. Since the KV holds the whole file, the model can (in principle) use memory to continue that specific line.

### 10.5 CLI usage

```bash
# Single-ayah mode (default): memorize one verse, warmup=first 4 bytes, complete rest
python -m kvmem.stage0 infer --ckpt checkpoints/stage0_step50000 \
    --line 0 --mem-size 8 --warmup 4

# All 7 ayat, N=8:
python -m kvmem.stage0 infer --ckpt checkpoints/stage0_step50000 \
    --all-lines --mem-size 8 --warmup 4

# Whole-file mode: memorize all of Al-Fatihah, prompt with start of ayah 3
python -m kvmem.stage0 infer --ckpt checkpoints/stage0_step50000 \
    --file datasets/quran_uthmani.txt --mem-size 32 \
    --warmup-text "مَـٰلِكِ" --temp 0.8

# Whole-file, empty warmup (free generation from KV memory):
python -m kvmem.stage0 infer --ckpt checkpoints/stage0_step50000 \
    --file datasets/quran_uthmani.txt --mem-size 32 --warmup-bytes 0
```

### 10.6 What to look for (qualitative)

Stage 0 is trained on Markov chains, not Arabic. The purpose is:

| Observation | Interpretation |
|---|---|
| Output varies across different ayat with same warmup bytes | Memory is being used (bottleneck functional) |
| Output identical regardless of which ayah is memorized | Model ignores M region — failure |
| Whole-file mode: output changes when file changes | KV encodes file content |
| Byte match improves as N increases | More KV slots = better compression |
| High-byte patterns in output (Arabic-like UTF-8) | Model learned data distribution structure |

Exact Arabic reconstruction is a stage 7 goal. Any variation across source files or ayat is positive signal.

---

## 11. Outputs

After training + eval:

1. `checkpoints/stage0_step50000.eqx` + `.json` — final checkpoint
2. `reports/stage0_bpt_sweep.png` — bpt matched/cross/uniform vs N
3. `reports/stage0_training_loss.png` — loss curves
4. `reports/stage0_slot_diversity_N8.png` — cosine sim heatmap at N=8
5. Terminal output from single-ayah and whole-file inference for N ∈ {2, 8, 32}

---

## 12. Success Criteria

| Criterion | Required |
|---|---|
| `bpt_matched < bpt_uniform` for all N | Yes |
| `bpt_cross > bpt_uniform` for all N | Yes |
| `bpt_matched` trend decreasing as N grows | Yes |
| Off-diagonal slot cosine sim < 0.9 at N=8 | Yes |
| Fatihah outputs vary across ayat (memory is used) | Qualitative check |

Failure diagnosis:
- `bpt_cross ≤ bpt_uniform` → mask bug: Y is seeing S directly
- `bpt_matched ≥ bpt_uniform` → memory carries no info: check ETX_pos in loss mask
- Fatihah output identical across ayat → model ignores M region

---

## 13. File: `kvmem/stage0.py`

Single file containing (in order):
1. Imports (`jax`, `equinox`, `numpy`, `matplotlib`, `json`, `argparse`)
2. Hyperparameter constants + segment protocol bytes
3. `make_mask_stage0(L_S, N, L_y)` → numpy array
4. `MHAttention`, `FFN`, `TransformerBlock`, `KVMemModel`
5. `build_model(key, hparams)` → `KVMemModel`
6. `loss_fn`, `clip_grads`, `lr_schedule`, `adam_step`, `init_opt_state`
7. `train_step` (jitted)
8. `save_checkpoint`, `load_checkpoint`
9. `eval_bpt`, `slot_diversity`
10. `load_fatihah()`, `run_fatihah_inference(...)` — single-verse mode
11. `run_file_memorize_infer(...)` — whole-file mode
12. `main()` with `argparse` subcommands:
    - `train` — full 50k-step training run, saves checkpoint
    - `eval --ckpt PATH` — bpt sweep over N, plots
    - `infer --ckpt PATH [--line N|-1] [--all-lines] [--mem-size N] [--warmup W] [--temp T]`
    - `infer --ckpt PATH --file PATH [--mem-size N] [--warmup-text STR] [--warmup-bytes W] [--temp T]`

Imports from `kvmem/data.py`: `make_batch`, `make_eval_batches`.
