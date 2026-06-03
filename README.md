# kvmem — Fast-Weight Language Model

## Vision

**A model that reads any document once and answers any question about it — without storing the document, without backprop, and without retraining.**

The `<h>` hidden state is a compressed fast-weight representation of whatever was ingested. At inference, base weights are frozen. Reading = forward passes that update `<h>`. Querying = next-token prediction from a warmup prefix.

**Instruction following without fine-tuning:**

Feed the instruction dataset as a corpus — the same way you feed any text. The fast weights compress the instruction-answer patterns. At query time, present the instruction as an NTP warmup:

```
# Ingest once, no weight update:
<x>Q: Capital of France?\nA: Paris\n</x><h>h_1</h>
<x>Q: Author of Hamlet?\nA: Shakespeare\n</x><h>h_2</h>

# Query — pure NTP:
<q>Q: Capital of Germany?\nA: </q>  →  <y>Berlin</y>
```

This is **compressed few-shot learning**: N examples compressed into `slot_len` tokens instead of O(N × example_len) in the context window. Generalises when the fast-weight compression is lossless enough.

**The theoretical ceiling:**
> Ingest any corpus forward-pass-only, then answer arbitrary queries — at quality comparable to a full-context LLM — but in O(slot_len) memory instead of O(corpus_length) KV cache.

---

## Architecture

**Tag vocabulary — RNN style:**

| Tag | Name | Role | DB op |
|-----|------|------|-------|
| `<x>` | input | source data | INSERT |
| `<z>` | intermediate | extract/pre-process | INDEX |
| `<h>` | hidden | fast-weight memory | STORE |
| `<q>` | query | warmup anchor | SELECT |
| `<y>` | output | predicted continuation | RESULT |

**Sequence layout (fully causal):**
```
<x>src</x> [<z>z_0..z_{P-1}</z>] <h>h_0..h_{N-1}</h>  <q>warmup</q>  <y>output</y>
```

Block order: `x → z → h` (all causal). `<q>/<y>` blocked from `x` and `z` — retrieval forced through `<h>` bottleneck.

**Causal access:**

| Row | Sees | Cannot see |
|-----|------|-----------|
| `<z>` intermediate | `x` (causal) | `<h>` (after it) |
| `<h>` hidden | `x` + `z` (causal) | — |
| `<q>` query | `<h>` (causal) | `x`, `z` (blocked) |
| `<y>` output | `<h>` + `<q>` (causal) | `x`, `z` (blocked) |

**Bottleneck:** `slot_len` is the only bottleneck. `slot_len=1` = single hidden state — everything must compress through 1 token.

---

## Fast Weights

`<h>` is a **running state**, not a stack. Each new block updates the state:
```
h_0 = compress(x_0)
h_t = update(h_{t-1}, x_t)   ← learns to mimic GD without GD
```

Cross-`<h>` attention is the update mechanism. The base model learns "how to update `<h>`" during meta-training. At inference, weights are frozen — only `<h>` changes.

This is informationally equivalent to a sliding-window KV cache but compressed: `corpus_tokens / slot_len` compression ratio.

---

## Vocab (v2 — learned embeddings)

- IDs 0–255: data bytes
- IDs 256–265: boundary tags (`<x>` `</x>` `<z>` `</z>` `<h>` `</h>` `<q>` `</q>` `<y>` `</y>`) — 1 token each
- IDs 266–265+slot_len: hidden slot tokens (unique per position, style D)
- IDs 266+slot_len–…: intermediate slot tokens
- V_in = 256 + 10 + slot_len + intermed_len
- V_out = 256 — output head predicts data bytes only

Two embedding matrices:
- `data_embed: Embedding(256, d)` — data bytes
- `special_embed: Embedding(V_in-256, d)` — tags and slot IDs

---

## Config DSL

```python
# configs/single_s16.py
hp = dict(
    seg_len=16, slot_len=1, intermed_len=7,
    warmup_len=4, out_len=8,
    B=16, lr_max=3e-4, n_steps=80000,
    dataset_size=10000, name='single_s16',
    curriculum=None,
)
```

DSL string equivalent: `<h:1><x:16><z:7><q:4><y:8>`

```bash
python -m kvmem.train --config configs/single_s16.py --device mps
```

---

## Results

| Config | val_bpb | match% | Notes |
|--------|---------|--------|-------|
| v1: seg=16, slot=8, active=1 (full-pass TF) | 0.249 | **93.8%** | 40k steps |
| v2: seg=16, slot=1, intermed=7 (causal) | 0.176 | **98.4%** | 70k steps, ds10k |

v2 architecture: RNN-style tags, learned embeddings, source-first causal layout, no active_slots.

---

## Training objective

**NTP on `<y>` positions only.** Source NTP would train reconstruction, not retrieval — wrong objective for random-data recall.

All queryable operations must be expressible as NTP warmup → continuation:
- LM completion: warmup = text prefix → continuation
- Line lookup: warmup = `"0042 "` → line content
- Dict lookup: warmup = `'"name": "'` → value
- IT queries: warmup = `"Q: instruction\nA: "` → response

```
train BPB ≈ val BPB   →  generalised: learned the algorithm
val BPB → entropy(corpus)  →  milestone: compression is effective
```

---

## Code layout

```
kvmem/
  train.py        — training (NTP on <y>, config DSL, --config flag)
  model.py        — transformer (dual embeddings, V_out=256, grad_checkpoint)
  data.py         — masks + batch builders (pure causal, make_mask_multi)
  seq_dsl.py      — DSL parser (<h:N><x:M><z:P><q:Q><y:R> → SeqSpec)
  kvcache.py      — blockwise KV-cache training (for large sequences)
  optim.py        — GrokAdamW
configs/
  single_s16.py          — slot=1, intermed=7, seg=16 (current baseline)
  ablate_t1.py           — 2-block recall from block 0
  ablate_t2.py           — 2-block recall from block 1
kvmem/old/v1/           — snapshot of prior source-first v1 code
reports/
  tag_naming.md          — tag vocabulary design rationale
```
