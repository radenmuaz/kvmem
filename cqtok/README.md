# cqtok — Continuous Quantized Tokenizer

Experimental workbench for the continuous tokenizer design from `research/LM.md`.

This folder focuses on **Phase 0 and baseline experiments**: a clean byte-level LM with a causal Transformer backbone (RoPE), compared against BPE and raw-byte softmax baselines, before layering in the continuous bottleneck.

---

## Goals

1. Validate the eval harness (nats / BPB) on two clean baselines.
2. Find good hyperparameters at 1MB scale before scaling up.
3. Build reusable dataset and training infrastructure in JAX + Equinox.

---

## Baselines

### Baseline A — BPE + softmax (standard LM)

- Tokenize with byte-level BPE (tiktoken `cl100k_base` or train a small vocab on the corpus).
- Standard causal Transformer, predict next token with cross-entropy.
- Nats: `mean(-log p(token))` in natural log.
- BPB: multiply nats per token by `1 / (ln 2 * avg_bytes_per_token)`.
  `avg_bytes_per_token = total_bytes / total_tokens` on the validation set.

### Baseline B — byte-level softmax (256-way)

- No tokenization; the model predicts the next byte, vocab size 256.
- Nats: `mean(-log p(byte))`.
- BPB: `nats / ln(2)`.  Exact, no correction needed.

### Target — continuous tokenizer (cqtok, later phases)

- FSQ bottleneck, chunk K=8 bytes → 1 latent.
- BPB ≈ LM code cross-entropy when reconstruction ≥ 99.9% (see `research/LM.md §1.7`).

---

## Backbone: Causal Transformer with RoPE

Both baselines and the cqtok LM share the same backbone to isolate the tokenizer effect.

```
input tokens  →  embedding  →  [TransformerBlock × L]  →  LM head
```

Each `TransformerBlock`:
- Pre-norm (RMSNorm)
- Causal self-attention with RoPE
- Pre-norm
- FFN (gated SwiGLU, expansion 4×)

RoPE replaces learned positional embeddings. For byte-level sequences (long), RoPE's interpolation properties are important; set `theta=10000` initially, raise to `theta=500000` if sequence length > 4096.

---

## Hyperparameters — 1MB scale

Dataset: `datasets/quran_uthmani.txt` ≈ 1.36 MB ≈ 1.36M bytes.
Split: 90% train (≈ 1.22M bytes), 10% val (≈ 137K bytes).

### Byte-level baseline (Baseline B)

```yaml
# Model
d_model: 128
n_layers: 4
n_heads: 4          # head_dim = 32
ffn_mult: 4         # hidden = 4 * d_model = 512
vocab: 256
rope_theta: 10000

# Sequence
seq_len: 512        # bytes per example

# Training
batch_size: 32      # 32 * 512 = 16384 bytes/batch
lr: 3e-4
optimizer: adamw
betas: [0.9, 0.95]
weight_decay: 0.1
warmup_steps: 200
total_steps: 5000
grad_clip: 1.0

# Params estimate: ~1.5M
```

### BPE baseline (Baseline A)

Same architecture. Swap vocab to BPE vocab size (256–4096 for small corpus).
On a 1.4MB corpus, a vocab of 512–1024 is reasonable (many BPE merges will be Arabic digraphs).

```yaml
vocab: 1024        # train BPE on the corpus; 1024 merges over 256 bytes
seq_len: 256       # shorter in tokens, same byte coverage
# otherwise identical to byte baseline
```

### cqtok LM (later)

```yaml
# Tokenizer
chunk_size_K: 8
bottleneck: fsq
fsq_dims_dq: 6     # codebook 8^6 = 262144
fsq_levels_L: 8
encoder_layers: 2
encoder_d: 128

decoder_type: nat  # pure NAT for Phase 1 simplicity
decoder_layers: 2
decoder_d: 128

# LM (same backbone)
d_model: 128
n_layers: 4
n_heads: 4
seq_len: 192       # latent positions (= 192*8 = 1536 bytes context)

# Training
batch_size: 32
lr: 3e-4
total_steps: 8000  # more steps: joint enc+dec+lm training
beta_warmup: 800   # KL weight ramp (not needed for FSQ, but keep scaffold)
```

---

## Metrics

### Nats

Mean negative log-likelihood in natural log units:

```
nats = -mean(log p(target))   # log = natural log
```

### Bits per byte (BPB)

```
bpb = nats / ln(2)            # for byte-level models: exact
```

For BPE models, convert from nats-per-token to bits-per-byte:

```python
# on val set
total_bytes = count_bytes(val_text)
total_tokens = count_tokens(val_text)
avg_bytes_per_token = total_bytes / total_tokens

bpb = nats_per_token / (math.log(2) * avg_bytes_per_token)
```

This is an approximation that assumes nats are uniform across token lengths. It is standard and comparable across tokenizers.

For cqtok with FSQ + reconstruction ≥ 99.9%:

```
bpb ≈ lm_code_cross_entropy_nats / (ln(2) * K)
```

where `K=8` is bytes per latent.

---

## File layout

```
cqtok/
  README.md          ← this file
  data.py            ← dataset: load any text/binary file, tokenize, save to disk
  model.py           ← backbone: causal Transformer with RoPE, shared by all baselines
  train.py           ← training loop (JAX + Equinox + Optax)
  eval.py            ← compute nats and BPB on val set
  tokenizer.py       ← BPE tokenizer wrapper + byte-level passthrough
  fsq.py             ← FSQ encoder, decoder, quantizer (Phase 1+)
  run_baseline_byte.py
  run_baseline_bpe.py
  run_cqtok.py
```

---

## JAX / Equinox notes

MPS backend is installed but **JAX PRNG (random) does not work on MPS**. Use CPU for all random operations.

```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"   # put at top of every script
import jax
import jax.numpy as jnp
import equinox as eqx
```

Alternatively, keep MPS for forward/backward but generate random keys on CPU:

```python
key = jax.random.PRNGKey(0)           # works on CPU (default)
# jit-compiled forward will run on MPS if no random inside
```

Use `equinox.nn.MakeJaxArray` for parameter init — Equinox's `eqx.nn.Linear` etc. accept a `key` argument and are CPU-safe.

Optax for optimizers (AdamW). Gradient clipping via `optax.clip_by_global_norm`.

---

## Dataset code plan (`data.py`)

```python
# data.py
# Converts any text or binary file to a flat uint8 array saved as .npy
# Supports train/val split by byte offset (not random shuffle, to preserve order)

def prepare(
    src: str,           # path to text or binary file
    out_dir: str,       # where to write train.npy and val.npy
    val_frac: float = 0.1,
    encoding: str = "utf-8",   # ignored for binary
    mode: str = "text",        # "text" | "binary"
):
    ...

# Usage:
#   python data.py --src datasets/quran_uthmani.txt --out data/quran
# Writes:
#   data/quran/train.npy   shape (N_train,) dtype uint8
#   data/quran/val.npy     shape (N_val,)   dtype uint8
#   data/quran/meta.json   {"total_bytes": ..., "val_frac": ..., "mode": ...}
```

The `.npy` format is memory-mappable via `np.load(..., mmap_mode='r')`, so batches can be sliced without loading the full file into RAM.

Batching: sample contiguous windows of `seq_len` bytes at random offsets from the mmap array. Since JAX PRNG is CPU-only, generate offsets on CPU, slice, then pass the batch as a JAX array.

---

## Experiment sequence

1. `python data.py --src datasets/quran_uthmani.txt --out data/quran`
2. `python run_baseline_byte.py --data data/quran` → logs nats + BPB every 100 steps
3. `python run_baseline_bpe.py --data data/quran --vocab 1024` → same
4. Compare BPB. Byte baseline should win slightly at this scale (small corpus, BPE savings are minimal).
5. `python run_cqtok.py --data data/quran` → Phase 1 autoencoder only, check reconstruction accuracy.
6. Add LM and compare BPB against baselines.

---

## Expected numbers (1.4MB Quran, Arabic UTF-8)

Arabic text is UTF-8 multibyte (each Arabic character = 2 bytes).
Entropy is lower than English (~1–2 bits/char), but UTF-8 overhead matters.
Rough expected BPB:

| model | expected BPB |
|---|---|
| random byte model | 8.0 |
| byte bigram | ~3.5–4.0 |
| byte Transformer (d=128, L=4) | ~1.8–2.5 |
| BPE Transformer (same compute) | ~1.7–2.3 |
| cqtok FSQ K=8 (goal) | ≤ byte Transformer |

BPE advantage shrinks for non-Latin scripts because the BPE merges must first deduplicate the UTF-8 byte pairs (each Arabic codepoint is 2 bytes, so BPE immediately merges them, effectively creating a 2-byte "byte-pair" baseline).
