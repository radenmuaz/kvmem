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
- IDs 266–265+slot_len: hidden slot tokens — **one dedicated ID per position**
- IDs 266+slot_len–…: intermediate slot tokens — one dedicated ID per position
- V_in = 256 + 10 + slot_len + intermed_len (auto-computed)
- V_out = 256 — output head predicts data bytes only

**Slot token scheme — three options with different scaling properties:**

| Scheme | IDs | Vocab | Extrapolation | Collision | V per slot |
|--------|-----|-------|---------------|-----------|-----------|
| **Dedicated indexed** (current, K=slot_len) | 266+i | grows | ✗ unseen IDs | none | unique |
| **Dedicated cyclic** (K < slot_len) | 266+(i%K) | fixed at K | ✓ cycle repeats | none | unique within cycle |
| **Looped byte** (style A) | i % 256 | fixed at 256 | ✓ cycle repeats | with data | unique within cycle |

**Dedicated cyclic** is the best design for scale: choose cycle length K (e.g. K=8), allocate K dedicated token IDs above 255, and assign `slot_i → 266 + (i % K)`. Combines the fixed vocab and extrapolation of style A with zero data collision and gradient purity of dedicated indexed.

```
K=8 dedicated slot IDs: 266, 267, ..., 273
slot_0 → 266,  slot_1 → 267, ...,  slot_7 → 273
slot_8 → 266,  slot_9 → 267, ...   ← cycle repeats
```

Vocab size = 256 + 10 + K (independent of slot_len). Model trained on slot_len=8 extrapolates to slot_len=1024 — same 8 cyclic IDs, RoPE carries absolute position. K is the "slot vocabulary budget": small K → more RoPE reliance, large K → more unique per-slot identity.

**Current code** uses dedicated indexed (K=slot_len). Dedicated cyclic with small K is the right choice before scaling to large slot_len. `make_hidden_slot_ids` accepts an optional `cycle_len` parameter.

Two embedding matrices:
- `data_embed: Embedding(256, d)` — data bytes, std=0.02
- `special_embed: Embedding(V_in-256, d)` — tags + slot IDs, std=0.05

---

## DSL

Two DSLs covering all sequence and training configuration.

### Sequence DSL (`kvmem/seq_dsl.py`)

```
<x:16><z:7><h:1><q:4><y:8>
```

| Token | Param | Meaning |
|-------|-------|---------|
| `<h:N>` | slot_len=N | N hidden/memory slots |
| `<x:N>` | seg_len=N | N-byte source input |
| `<z:N>` | intermed_len=N | N intermediate tokens |
| `<q:N>` | warmup_len=N | N-byte query anchor |
| `<y:N>` | out_len=N | N-byte output |

### Curriculum DSL (`kvmem/curriculum_dsl.py`)

```
seq_spec | stage, stage, stage @eval:eval_spec
```

**Stage token:** `nN/rK/Xk[/wM]`

| Token | Meaning | Example |
|-------|---------|---------|
| `nN` | n_blocks | `n1`, `n2` |
| `rK` | recall_from single | `r0`, `r1` |
| `r[K,...]` | recall_froms mixed (per-example random draw) | `r[0,1]` |
| `Xk` | steps | `40k`, `160k` |
| `wM` | mem_window (-1=full, 1=isolated, N=window) | `w-1`, `w1` |
| `mMODE` | op sequence mode (see below) | `mint`, `macc`, `mmix` |

**Op sequence modes (`mMODE`):**

| Mode | Pattern per example | Use |
|------|---------------------|-----|
| (default) `mend` | `xh xh ... q` | all ingest then one recall |
| `mint` | `xhq xhq ... xhq` | recall after every block (interleaved) |
| `macc` | `xh xh ... xh` | ingest only, no recall, no loss |
| `mmix` | random per example | mix of all patterns — interactive training |

For `mmix`, each batch example randomly picks a pattern: some examples just accumulate, some have end-recall, some have interleaved queries. Trains the model to handle both "new data in" and "user query" interactively.

**`+` overlap:** prefix a stage with `+` to merge into the previous stage's batch distribution:
```
n2/r0/80k +n2/r1   →  n2, recall_froms=[0,1] for 80k steps  (same as n2/r[0,1]/80k)
```

**`@eval:` annotation:** eval configs tested at every `eval_every` step, independent of training stages. If omitted, auto-derived from all stages + `n1/r0` baseline.
```
@eval:n1/r0,n2/r0,n2/r1
```

**Full example:**
```
"<x:16><z:7><h:1><q:4><y:8> | n1/r0/40k, n2/r1/40k, n2/r[0,1]/80k/w1 @eval:n1/r0,n2/r0,n2/r1"
```

```python
from kvmem.curriculum_dsl import parse_curriculum
spec, curriculum, eval_configs = parse_curriculum(
    "<x:16><z:7><h:1><q:4><y:8> | n2/r[0,1]/160k @eval:n1/r0,n2/r0,n2/r1",
    B=16, dataset_size=20000
)
hp['curriculum']   = curriculum
hp['eval_configs'] = eval_configs
```

**Hparams absorbed by DSL** (no longer set manually):  
`seg_len`, `slot_len`, `intermed_len`, `warmup_len`, `out_len`, `n_blocks`, `recall_from`, `mem_window`  
**Removed entirely:** `active_slots`, `slot_style`, `V` — slot_len IS the bottleneck; dedicated indexed tokens always.

```bash
# Train:
python -m kvmem.train --config configs/single_s16.py --device mps

# Eval only (load checkpoint, run eval_configs, exit):
python -m kvmem.train --config configs/expB_chain_nullkv.py \
  --eval-only logs/role_<name>/checkpoints/stage0_end.pt --device mps

# Resume (full state: weights + optimizer + rng):
python -m kvmem.train --config configs/expB_chain_nullkv.py \
  --resume logs/role_<name>/checkpoints/stage0_end.pt --device mps

# Pretrained weights only (fresh training, warm init):
python -m kvmem.train --config configs/expB_chain_nullkv.py \
  --pretrained logs/role_<name>/checkpoints/stage0_end.pt --device mps
```

---

## Results

| Config | Best val_bpb | match% | Notes |
|--------|------------|--------|-------|
| v1: seg=16, slot=8, active=1 (full-pass TF) | 0.249 | **93.8%** | 40k steps |
| v2: seg=16, slot=1, intermed=7 | 0.176 | **98.4%** | 70k steps, ds10k |
| v2 + null_kv=True | **0.157** | **92%** | 26k steps — 1.5-2× faster |
| v2 mixed routing n=2 (cold start) | 0.252 | **91%** both dirs | 65k/160k — routing works |

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
  seq_dsl.py        — sequence DSL (<x:M><z:P><h:N><q:Q><y:R> → SeqSpec)
  curriculum_dsl.py — curriculum DSL (<seq> | nN/rK/Xk → stage list)
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
