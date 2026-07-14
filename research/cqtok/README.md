# cqtok — Continuous Quantized Tokenizer

Experimental workbench for the continuous tokenizer design from `research/LM.md`.

Phase 0 and baseline experiments: byte-level LM and BPE LM with a shared causal Transformer + RoPE backbone, before layering in the continuous bottleneck (BSQ / FSQ).

---

## Quick start

```bash
# 1. Prepare data (train only)
python data.py --src ../datasets/quran_uthmani.txt --out data/quran

# 2a. Latent AR baseline (BSQ)
python lm_train.py --data data/quran --bottleneck bsq

# 2b. BPE baseline (SentencePiece trained on corpus)
python bpe_train.py --src ../datasets/quran_uthmani.txt --out data/quran

# Override log folder
python lm_train.py --data data/quran --log_dir logs --run_name my_run --no_date
```

---

## Files

```
cqtok/
  data.py        dataset: load any text/binary file, split, save .npy
  model.py       causal Transformer + RoPE backbone (shared by all scripts)
  bsq.py         Binary Spherical Quantization encoder + LM head
  fsq.py         Finite Scalar Quantization (L=2 and L=8) encoder + LM head
  codec.py       MLP byte encoder / decoder + ByteAutoencoder
  lm_train.py    latent autoregression with re-encoded grounding (A-grounded)
  bpe_train.py   BPE baseline: SentencePiece + causal Transformer
```

---

## data.py

Converts any text or binary file to memory-mappable `uint8` `.npy` splits.

### Split modes

| Command | Outputs |
|---|---|
| `--src f --out d` | `train.npy` only |
| `--src f --out d --val N` | `train.npy` + `val.npy` (split at byte N) |
| `--src f --out d --val N --test M` | `train.npy` + `val.npy` + `test.npy` |

`N` and `M` are byte indices (negative values count from end).  
`--test` requires `--val`; error raised if `M ≤ N`.

```bash
# train only
python data.py --src ../datasets/quran_uthmani.txt --out data/quran

# train + val at byte 1,100,000
python data.py --src ../datasets/quran_uthmani.txt --out data/quran --val 1100000

# train + val + test
python data.py --src ../datasets/quran_uthmani.txt --out data/quran \
    --val 1100000 --test 1300000

# binary file
python data.py --src myfile.bin --out data/myfile --mode binary --val 900000
```

`meta.json` always contains `train_bytes`, `val_bytes`, `test_bytes` (0 if not split), plus the raw `val_start`/`test_start` indices.

The `.npy` files are memory-mappable — `ByteDataset` slices them without loading into RAM.

---

## Bottlenecks

Both BSQ and FSQ target the same codebook size (2^18 ≈ 262K) for K=8 byte chunks.

| variant | d_q | L | codebook | head |
|---|---|---|---|---|
| BSQ | 18 | — | 2^18 | per-bit BCE |
| FSQ L=2 | 18 | 2 | 2^18 | per-bit BCE |
| FSQ L=8 | 6 | 8 | 2^18 | per-dim 8-way CE |
| BSQ balanced | 24 | — | 2^24 | per-bit BCE |
| FSQ L=8 bal. | 8 | 8 | 2^24 | per-dim 8-way CE |

**Codebook sizing for K=8 (256-way each):**
Full space = 256^8 = 2^64.
Arabic/English text has ~1–3 bits/byte effective entropy → K=8 → ~8–24 bits needed.
`d_q=18` covers the Quran corpus (~170K chunks) with headroom (2^18 = 262K).

---

## lm_train.py — latent autoregression

**Training (Option A-grounded, `research/LM.md §2.2`):**

```
byte_chunks [B, T, K]
  → ByteEncoder (MLP)           teacher-forced: always sees ground-truth bytes
  → BSQ / FSQ quantizer         → z_hat [B, T, d_q],  codes [B, T, d_q]
  → LatentLM (causal Transformer, RoPE)
  → LM head                     pred_loss: h[:,:-1] predicts codes[:,1:]
  → ByteDecoder (MLP)           rec_loss: z_hat → byte logits vs byte_chunks
total_loss = rec_loss + pred_loss
```

**Inference (A-grounded):**  
Sample z_next from LM → decode to bytes → re-encode → append z_grounded (not raw z_next).

```bash
python lm_train.py --data data/quran --bottleneck bsq
python lm_train.py --data data/quran --bottleneck fsq8 --d_q 6
python lm_train.py --data data/quran --bottleneck fsq2 --d_q 18
```

Key flags: `--K 8`, `--T 64`, `--d_model 128`, `--n_layers 4`, `--n_heads 4`, `--steps 5000`.

---

## bpe_train.py — BPE baseline

SentencePiece BPE trained on the corpus (`byte_fallback=True` → no UNK).

```bash
python bpe_train.py --src ../datasets/quran_uthmani.txt --out data/quran --vocab_size 1024
```

BPB conversion: `bpb = nats_per_token / (ln(2) * avg_bytes_per_token)`  
`avg_bytes_per_token` measured on the full corpus (stored in `meta_bpe.json`).

---

## Metrics

**Nats:** `mean(-log p(target))` (natural log).

**BPB (bits per byte):**

- Byte-level: `bpb = nats / ln(2)` — exact, no approximation.
- cqtok FSQ (reconstruction ≥ 99.9%): `bpb ≈ lm_code_ce / (ln(2) * K)`.
- BPE: **exact per-batch** via token→byte map:

  ```
  bpb = sum(nll_i) / (sum(bytes(token_i)) * ln(2))
  ```

  `bytes(token_i)` is the UTF-8 byte length of token `i` in decoded text.
  This is computed exactly from `build_token_bytes(sp)` — a `(vocab_size,)` int32 array
  built once from the SentencePiece model.

  The common approximation `nats_per_token / (ln(2) * avg_bytes_per_token)` is only
  exact at corpus level. Per-batch it is biased whenever short and long tokens happen
  to be sampled unevenly (std of token byte-length ≈ 2 for this corpus).

  Note: `sum(token_bytes)` matches the SentencePiece-normalized byte count, which
  may differ by a few bytes from the raw file due to NMT-NFKC normalization
  (e.g. extra whitespace stripped). This is expected and correct.

---

## Logging

Every run writes to `logs/<tag>_<timestamp>/`:

```
args.json      full CLI arguments
train.log      human-readable (also printed via tqdm.write)
train.jsonl    one JSON record per step; load with pd.read_json(..., lines=True)
```

Log folder options:

| flag | effect |
|---|---|
| `--log_dir DIR` | base directory (default: `logs`) |
| `--run_name foo` | folder becomes `logs/foo_<timestamp>/` |
| `--run_name foo --no_date` | folder becomes `logs/foo/` (no timestamp) |

Val is evaluated against `--val_file` (default: `../datasets/suratalfatihah.txt`).

---

## JAX / MPS

MPS backend is available (`jax-mps`) but **JAX PRNG does not work on MPS**.
Pattern used throughout: random on CPU, compute on MPS.

```python
cpu = jax.devices("cpu")[0]
mps = jax.devices("mps")[0]   # falls back to cpu if unavailable

# All random sampling on CPU
with jax.default_device(cpu):
    key   = jax.random.PRNGKey(seed)
    model = MyModel(..., key=key)

# Transfer to MPS, then compute there
model = jax.device_put(model, mps)
with jax.default_device(mps):
    out = jit_fn(model, batch)
```

Do **not** set `JAX_PLATFORMS=cpu` in training scripts — that disables MPS.
The utility modules (`bsq.py`, `fsq.py`, etc.) use `os.environ.setdefault("JAX_PLATFORMS", "cpu")` only as a fallback for standalone testing.

---

## Expected BPB (1.4MB Quran, Arabic UTF-8)

Arabic is UTF-8 multibyte (~2 bytes per character). BPE advantage shrinks because merges first deduplicate the byte pairs of each codepoint.

| model | expected BPB |
|---|---|
| random byte model | 8.0 |
| byte bigram | ~3.5–4.0 |
| byte Transformer (d=128, L=4, 5k steps) | ~1.8–2.5 |
| BPE Transformer (same compute) | ~1.7–2.3 |
| cqtok BSQ/FSQ K=8 (goal) | ≤ byte Transformer |
