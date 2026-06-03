# kvmem — Project Notes for Claude

## Vision

**A model that reads any document once and answers any question about it — without storing the document, without backprop, and without retraining.**

The `<h>` hidden state is a compressed fast-weight representation of whatever was ingested. At inference, base weights are frozen. Reading = forward passes that update `<h>`. Querying = NTP from a warmup prefix.

**Instruction following without fine-tuning:**
Feed the instruction dataset as a corpus — the same way you feed any text. The fast weights compress the instruction-answer patterns. At query time, present `"Q: [new instruction]\nA: "` as the NTP warmup. The model predicts the continuation.

```
# Ingest once (no weight update):
<x>Q: Capital of France?\nA: Paris</x><h>h_1</h>
<x>Q: Author of Hamlet?\nA: Shakespeare</x><h>h_2</h>

# Query (pure NTP):
<q>Q: Capital of Germany?\nA: </q>  →  <y>Berlin</y>
```

This is **compressed few-shot learning**: N instruction examples compressed into `slot_len` tokens instead of O(N × example_len) tokens in the context window. Generalises if compression is lossless enough.

**The theoretical ceiling:**
> A model that ingests any corpus forward-pass-only, then answers arbitrary natural language queries about it — at quality comparable to a full-context LLM — but in O(slot_len) memory instead of O(corpus_length) KV cache.

The compression ratio = corpus_tokens / slot_len. Whether achievable in practice is what this project tests.

---

## Primary Goal (updated 2026-06-03)

**Learn in-context language modelling without backprop.**

The model's frozen base weights learn ONE thing during meta-training:
> **How to update `<h>` fast weights so that `<y>` predictions improve.**

At inference, weights are frozen. The model ingests a new corpus chunk by chunk, updating `<h>` through forward passes only. The milestone is: val BPB ≈ train BPB on a held-out corpus — the fast-weight update generalises, not memorises.

**What good looks like:**
```
train BPB ≈ val BPB    →  generalised: learned the update algorithm
train BPB << val BPB   →  memorised:   overfit to training sequences
val BPB → oracle LM    →  milestone:   fast-weight compression is effective
```

**Windowed recall is a diagnostic task** — random bytes = maximally incompressible (entropy=8 bits/byte). Achieving low BPB on random bytes verifies the retrieval mechanism itself without statistical priors. Progression:
1. Random bytes recall → verify mechanism (current)
2. Structured text with line numbers → test content-addressable retrieval
3. Natural language corpus → test LM prior learning
4. Cross-corpus generalisation → the real milestone

**Self-correction and ground truth are a MEANS** — OCD/correction trains the model to improve its `<h>` update rule. At inference there is no ground truth; the algorithm runs forward-only.

---

## Current Status (2026-06-03)

Windowed recall with v1 role-tag scheme working: **val_bpb=0.249, 93.8% match** @40k (seg=16, active_slots=1, full-pass TF).

**v2 architecture (current):** 98.4% match @70k (ds10k), val_bpb=0.176 — RNN-style tags, learned embeddings, source-first causal layout.
- Cosine LR with small dataset → overfitting after peak

**Next steps:**
1. Scale from seg=16 to seg=32, seg=64 with same config (slot=8, active=1)
2. OCD fine-tuning from best checkpoint (separate run, low prob ≤0.05)
3. Multi-window eval (sweep eval_offset) to verify generalisation across positions
4. Best-val checkpoint saving (currently saves only end-of-stage)

---

## Architecture

Sequence format: `<s> x_S </s> <m> slots </m> <f> warmup </f> <c> output </c>`

- `<s>` wraps source (encoded into slot KV)
- `<m>` wraps KV slots (slot_len tokens, style='seq': IDs 0,1,...,N-1)
- `<f>` wraps warmup anchor (warmup_len bytes before output window)
- `<c>` wraps output (model generates here)
- `active_slots`: only last N slots visible to `<f>`/`<c>` (rest encode but masked)

Mask rules:
- slots attend to x_S (encode source)
- `<f>` attends to active slots only (not x_S directly — forces KV lookup)
- `<c>` attends to active slots + `<f>` region (not x_S)
- `<m>` and `</m>` always visible regardless of active_slots

KV dims: `KV_bytes = 4 × 2 × n_layers × active_slots × d`
- active_slots=1, d=64, n_layers=4 → 512 floats / 2KB per sequence

---

## OCD (Optimal Completion Distillation)

**OCD should only be used as final fine-tuning, not during main training.**

Findings from ablation (adamw_s16_ocd_sched, 2026-06-03):
- High-prob OCD (p=0.5, p=0.99) introduced late destabilises a converged model
- Train loss spikes 0.13 → 1.18 when OCD prob jumps at step 30k
- TF-only to step 40k: 93.8% match. Adding late OCD dropped to 75%.

**Revised finding (adamw_s16_ocd_sched, full run):**
- Model disrupted at step 35k (OCD 0.5 starts): match 60.9% → 43.75%
- Continued recovery through 15k more OCD steps: match reached **90.6% @50k**
- TF-only baseline: 93.8% @40k with 10k fewer steps
- OCD eventually works but needs a longer recovery window and softer transition
- Transition 0.01→0.5→0.99 is too abrupt; try 0.01→0.1→0.5→0.99 or start OCD earlier

**Revised guidance:** Late high-prob OCD causes temporary disruption but the model recovers. For best efficiency: either (a) TF-only to convergence then separate OCD fine-tune run from checkpoint, or (b) if scheduling OCD in-run, use a gradual ramp and budget ~15k extra steps for recovery after each prob increase.

OCD schedule in hp: `ocd_prob = [[step_from_start, prob], ...]` (ascending by step).

---

## Key Hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| active_slots | 0 (all) | Set to 1 for best results so far |
| eval_offset | 0.25 | `<f>` starts at 25% of seg_len |
| dataset_size | 5000 | Batches. 0=infinite stream |
| cycle_steps | 0 | 0=flat LR after warmup |
| drop_close_prob | 0.5 | Prob of dropping `</c>` per example |
| ocd_prob | 0.01 | Scalar or `[[step,prob],...]` schedule |

---

## Code Layout

```
kvmem/
  train.py           — main training script (fused from train_role + train_role_kvcache)
                       --kv-cache flag selects prefix-KV OCD rollout vs full-pass
  model.py           — transformer (encode_prefix / forward_with_prefix_kv added)
  data.py            — make_mask_role (active_slots param added)
  optim.py           — GrokAdamW
  eval_surah.py      — windowed recall eval on suratalfatihah.txt
reports/
  kv_dims.md         — KV memory dimension tables
logs/
  role_grok_s8_a1/          — GrokAdamW, seg=8, best val_bpb=0.130
  role_adamw_s8_flat/        — AdamW flat, seg=8, best val_bpb=0.130, 58.3% match
  role_adamw_s16_w8/         — AdamW flat, seg=16, TF-only, 93.8% match @40k
  role_kvcache_s16_w8/       — kvcache code, seg=16, TF-only (comparison run)
  role_adamw_s16_ocd_sched/  — OCD late-schedule ablation (model recovers to 90.6%)
```

## Multi-Step Corpus Reading Plan (next major milestone)

Extends the single-block format to multiple `<m>` blocks and multiple `<f><c>` recall pairs.
Each `<m>` block encodes one source segment into KV slots. Recall windows can query any prior block.

### Sequence formats for training batches

**Type 1 — two blocks, single recall from m1:**
```
<s>src1</s><m>slots1</m> <s>src2</s><m>slots2</m> <f>anchor_in_src1</f><c>output_from_src1</c>
```

**Type 2 — two blocks, single recall from m2:**
```
<s>src1</s><m>slots1</m> <s>src2</s><m>slots2</m> <f>anchor_in_src2</f><c>output_from_src2</c>
```

**Type 3 — interleaved ingestion and recall:**
```
<s>src1</s><m>slots1</m> <f>anchor1</f><c>out1</c>
<s>src2</s><m>slots2</m> <f>anchor2</f><c>out2</c>
<f>anchor1_again</f><c>out1_again</c>
```
Model must retain src1 recall after ingesting src2.

**Type 4 — interleaved with cross-block recall:**
```
<s>src1</s><m>slots1</m> <f>anchor1</f><c>out1</c>
<s>src2</s><m>slots2</m> <f>anchor1_from_src1</f><c>out_from_src1</c>
<f>anchor2_from_src2</f><c>out_from_src2</c>
```
After reading src2, recall both src1 and src2.

### Mask rules extension
- Each `<m>` block attends only to its paired `<s>` source (or accumulated prior KV)
- Each `<f><c>` pair can attend to ALL `<m>` blocks seen so far
- `<c>` blocks remain write-only (not attended to by anything outside)
- Blockwise KV caching enables efficient multi-block inference: cache each `<m>` block's KV, reuse for all subsequent `<f><c>` queries

### Prerequisites
- Single-block recall working at seg=16 (done: 93.8% match)
- Blockwise KV forward pass in model (done: `past_kv` / `return_kv` API)
- Scale to seg=32+ before adding multi-block (validate single-block generalises first)

### Ablation plan — validate each type independently before curriculum

**Rule: only advance to the next type when the current one achieves ≥80% match.**

**Step 1 — Type 1 ablation (two blocks, recall from m1):**
```bash
python -m kvmem.train --curriculum none \
  --n-blocks 2 --recall-from 0 \
  --seg-len 16 --slot-len 8 --active-slots 1 \
  --warmup-len 4 --out-len 8 \
  --steps 80000 --eval-every 5000 \
  --d 64 --n-layers 4 --B 16 --lr 3e-4 \
  --dataset-size 10000 --cycle-steps 0 \
  --no-grok --device mps --name ablate_t1
```
Pass: ≥80% match on held-out src1 recall after seeing both m1 and m2.

**Step 2 — Type 2 ablation (two blocks, recall from m2):**
Same config, `--recall-from 1`.
Pass: ≥80% match on src2 recall.

**Step 3 — Combined Type 1+2 (model must select correct block from anchor):**
Mix both recall targets in the same training pool.
Pass: ≥80% match on BOTH src1 and src2 recall (no confusion).

**Step 4 — Type 3 ablation (interleaved ingestion + recall, in-order):**
```
m1 | recall_m1 | m2 | recall_m2 | recall_m1_again
```
Pass: recall_m1_again ≥70% match (memory retention after m2 ingestion).

**Step 5 — Type 4 ablation (interleaved, cross-block):**
```
m1 | recall_m1 | m2 | recall_m1_via_m2 | recall_m2
```
Pass: ≥70% match on all three recall positions.

**Step 6 — Full curriculum (all 4 types mixed):**
Only after Steps 3+5 pass. Mix Types 1-4 in training pool with equal weight.
Final eval: sweep all recall positions across both blocks.

### Training curriculum
1. Single-block (current) → establish baseline at each seg_len
2. Type 1 ablation → Type 2 ablation → Type 1+2 combined
3. Type 3 ablation → Type 4 ablation
4. Full curriculum (Types 1-4 mixed)
5. N-block variable length → generalise to arbitrary corpus length

---

## Blockwise KV-cache vs Full-pass TF (seg=16, 40k steps)

Both use identical training data and loss formula. Results differ due to float32 gradient path differences compounding over 35k+ steps.

| | Full pass | Block pass (kvcache) |
|--|--|--|
| val_bpb @40k | 0.285 | **0.190** |
| match% @40k | **93.8%** | 54.7% |
| best val_bpb | 0.249 @20k | 0.190 @40k |
| best match% | **93.8%** | 60.9% @35k |

- Forward values are mathematically exact (max diff = 0.0 verified)
- Backward differs: full SDPA vs cat+SDPA have different float32 rounding → different gradients → different local optima
- Kvcache converges to better calibration (lower bpb) but worse exact match
- Full pass preferred when match% is the target; kvcache needed for large sequences (seg=576+) where L×L attention is the memory bottleneck

Split point for block pass: `mc1 = pos['mc1']` = end of `</m>` tag.
Prefix = `<s>x_S</s><m>slots</m>`, Suffix = `<f>warmup</f><c>output</c>`.

---

## Architecture (v2, current — 2026-06-03)

Major overhaul. Snapshot of prior code at `kvmem/old/v1/`.

**Tag vocabulary — RNN style (x/z/h/q/y):**

| Tag | Name | RNN var | DB op | Meaning |
|-----|------|---------|-------|---------|
| `<x>` | input | x_t | INSERT | source data to ingest |
| `<z>` | intermediate | z_t | INDEX | extract/pre-process before compression |
| `<h>` | hidden | h_t | STORE | compressed memory (KV bank) |
| `<q>` | query | — | SELECT | anchor/warmup lookup predicate |
| `<y>` | output | y_t | RESULT | retrieved value |

Extended CRUD ops (for self-correction / multi-block):
`<u>` update (UPDATE memory), `<d>` diff (DELTA error signal), `<c>` commit (COMMIT checkpoint), `<s>` seek (SEEK position), `<a>` attend (JOIN over multiple banks), `<n>` next (NEXT block advance)

**Sequence layout (fully causal, source-first within block):**
```
<x>src</x> [<z>z_0..z_{P-1}</z>] <h>h_0..h_{N-1}</h>  <q>warmup</q>  <y>output</y>
```

Block order: x → z → h. All attention is pure causal — no non-causal overrides.

**Causal access within each block:**
| Row | Sees | Cannot see |
|-----|------|-----------|
| `<z>` intermediate | x input (causal) | `<h>` (after it) |
| `<h>` hidden slots | x + z (both before it) | — |
| `<q>` query | `<h>` slots (causal) | x, z (explicit block) |
| `<y>` output | `<h>` + `<q>` (causal) | x, z (explicit block) |

`<q>/<y>` blocked from x and z → forced through `<h>` slot bottleneck.
`<y>` is write-only (nothing outside attends to it).

**Bottleneck:** `slot_len` (= h_len) IS the bottleneck directly. `active_slots` removed.
- `slot_len=1, intermed_len=7` ≈ v1's `slot_len=8, active_slots=1`
- `<z>` intermediate replaces "inactive slots" with productive compute (can see src)
- Inactive slots in v1 wasted capacity; `<z>` uses it for encoder pre-processing

**Config DSL:** `<h:1><x:16><z:7><q:4><y:8>` — 1-slot hidden, 16-byte input, 7-token intermediate, 4-byte query, 8-byte output.

**Vocab (v2, learned embeddings):**
- IDs 0–255: data bytes (unchanged)
- IDs 256–265: boundary tags (`<m>` `</m>` `<s>` `</s>` `<f>` `</f>` `<p>` `</p>` `<c>` `</c>`) — 1 token each
- IDs 266–265+slot_len: memory slot tokens (unique per position, style D)
- IDs 266+slot_len–…: ponder slot tokens (unique per position, style D)
- V_in = 256 + 10 + slot_len + ponder_len (auto-computed by `compute_vocab_size`)
- V_out = 256 (output head predicts data bytes only)

**Two separate embedding matrices:**
- `data_embed: Embedding(256, d)` — std=0.02
- `special_embed: Embedding(V_in-256, d)` — std=0.05 (tags + slot/ponder IDs)
- `W_out: Linear(d, 256)` — predicts only data bytes; loss gathered on `<c>` positions only

**Why fully causal:** supports streaming inference token-by-token. KV cache grows left-to-right. No mask complexity beyond explicit blocking rules.

**Why V_out=256:** special tokens never appear in `<c>` output. Loss gathered only at `<c>` positions where targets are always data bytes (0-255).

**Loss:** NTP only on `<c>` positions. Source NTP would train reconstruction (not retrieval) and is actively unhelpful for random-data recall.

**Config DSL:** `kvmem/seq_dsl.py` — parse spec strings like `<h:1><x:16><z:7><q:4><y:8>` into mask + batch + hp. See `configs/` for examples.

---

## Fast Weights and Multi-Step Corpus Reading

**`<h>` is a fast weight state, not a stack.** Cross-`<h>` attention is the update mechanism — `h_{t+1}` reads `h_t` and learns to update itself, like gradient descent compressed into a forward pass. Option A (blocking cross-h attention) would break this and should not be used.

```
h_0 = compress(x_0)                   ← cold start
h_1 = update(h_0, x_1)                ← fast-weight update
h_2 = update(h_1, x_2)                ← running summary
h_N = update(h_{N-1}, x_N)            ← "current memory" of entire corpus
```

**Recency is correct behaviour** for the LM goal — h_N IS the up-to-date running summary. Unlike recall, LM doesn't need to retrieve "what was on line 0" explicitly; it needs a good predictive model of the distribution.

**Practical limit:** The transformer can only read the last W `<h>` states. Early information must propagate through the chain (`h_0 → h_1 → ... → h_N`) — exactly the vanishing information problem from RNN literature. The model must learn to carry forward what matters.

**Line numbers as explicit keys:**
Embedding line numbers in `<x>` content (e.g. `"0042 The princess lived..."`) gives the fast-weight update an explicit content-addressable key. The model learns: "when seeing `NNNN content`, write an indexed entry under key NNNN into `<h>`." At query time, `<q>0042 </q>` retrieves that entry. Scales to:
- Byte offsets `@00512 `
- Section headers `# Introduction `
- Timestamps `14:32:05 `
- JSON paths `user.address `

**Multi-query over same memory (LLM analogy):**
```
<x>corpus</x><z>z</z><h>h_N</h>
<q>line 0 prefix</q><y>line 0 content</y>
<q>line 42 prefix</q><y>line 42 content</y>
```
Multiple `<q><y>` pairs against one `<h>`. Equivalent to an LLM answering multiple questions about the same document — but here the doc is compressed into `<h>` slots, not held in the KV cache verbatim.

**Map ops as NTP:**
Any structured operation fits the frame — dict lookup, array indexing, nested access, aggregation. `<q>` carries the key/path/op; `<y>` predicts the result bytes. The slot capacity (slot_len) bounds how many distinct key→value mappings can be reliably retrieved. Train/val BPB gap reveals whether the model learned the algorithm or memorised the corpus.

---

## Direction B: Weakly Imperfect Self-Correction

**Motivation:** Standard training (loss only on `<c>`) gives a one-shot recall signal. The model has no mechanism to iteratively refine at inference time. The `<p>` ponder region and recurrent memory provide the groundwork for teaching a correction *algorithm*.

**Core insight:** Training data is random distributions — source NTP would not help. The model must learn a retrieval algorithm, not memorise statistics. Self-correction trains this algorithm directly.

**Why ponder alone isn't enough:** Loss on `<c>` flows through `<p>` attention, but only if the model already uses `<p>`. There's no pressure to use the scratchpad. Need an objective that makes ponder *necessary*.

**Diffusion analogy:** Diffusion trains denoising at every noise level → learns a general denoising function. Goal here: train correction from *any imperfect starting state* → model learns a general error-correction function.

**Proposed sequence format (multi-pass recall):**
```
<x>src</x> <z>feat</z> <h>mem</h>
<q>anchor</q> <y>attempt_0</y>         ← first recall (may be wrong)
<d>error_signal</d>                     ← diff: sees src + attempt_0
<u>correction</u> <h>mem_v2</h>        ← update: refine memory
<q>anchor</q> <y>attempt_1</y>         ← corrected output
```

**Training procedure:**

| Mode | `attempt_0` | `attempt_1` target | Loss |
|------|-------------|-------------------|------|
| Pure TF | ground truth | ground truth | both equal |
| OCD-style | model's own rollout (no grad) | ground truth | only `attempt_1` |
| Curriculum | noise-corrupted truth | ground truth | only `attempt_1` |

The OCD-style pass is the key: `attempt_0` = model's actual imperfect output. `attempt_1` trained to correct from that starting point. Model sees many error patterns → learns a general correction function.

`<p>_1` now has a natural purpose: attends to `attempt_0` causally and to memory → carries error-analysis computation. Gradient from `attempt_1` loss flows back through `<p>_1` with a real signal.

**Recurrent memory update (deeper extension):**
```
<x>src</x><z>feat</z><h>mem_0</h>           ← initial encoding
<q>anchor</q><y>y_0</y>                      ← first attempt
<h>mem_1</h>                                  ← refined memory (trained via y_1 loss)
<q>anchor</q><y>y_1</y>                      ← second attempt with updated memory
```
`slots_1` learns to encode "what the model needs given that c_0 was wrong" — implicit correction signal, no explicit reconstruction target.

**Minimum viable first step:** Two `<f><p><c>` blocks per sequence, `attempt_0` filled via one of the methods below, loss only on `attempt_1`. All mask and position infrastructure already in place.

---

### Generating c1 (imperfect first attempt) — method comparison

AR sampling is too expensive (out_len forward passes). Two practical options:

**Cost/quality spectrum:**

| Method | Extra compute | Error type | Training-inference gap |
|--------|-------------|------------|----------------------|
| 2-step teacher force | 1× | None (perfect c1) | Huge — p2/m2 never see errors, learns no-op |
| Uniform substitution (denoise) | 1× | Random, i.i.d. | Medium — random ≠ model's systematic failures |
| **Parallel sample + reinput** | **2×** | **Systematic** (real model uncertainty) | **Small** |
| AR sample | out_len × | Most realistic | Smallest |

**2-step teacher force is wrong** — p2 sees perfect c1, learns to be a no-op. Exposure bias at its worst.

**Option A — Uniform substitution (denoising):**
Replace each c1 token independently with a random byte (0-255) at rate p ~ Uniform(0.05, 0.4).
Discrete-appropriate — no Gaussian, no mask tokens. p2 must compare c1 against s1 to detect errors.
Masking is wrong (p2 detects errors trivially from token ID, doesn't need to use memory or src comparison).
```python
p          = rng.uniform(0.05, 0.4)
flip       = rng.random((B, out_len)) < p
c1_corrupt = c1_gt.copy()
c1_corrupt[flip] = rng.integers(0, 256, flip.sum())
```
Cheap (1×), errors are random not systematic. Good for warmup / early training.

**Option B — Parallel sample + reinput (recommended):**
One forward pass: fill c1 positions with argmax of model logits at those positions simultaneously (non-AR).
Feed c1_parallel back as context, train c2. Errors are systematic — exactly where model is uncertain.
```python
# Pass 1: parallel decode c1 (no gradient)
with torch.no_grad():
    logits_c1 = model(tokens_with_blank_c1, mask)[:, c0-1:c1-1, :256]
    c1_parallel = logits_c1.argmax(-1)          # (B, out_len) — all positions in one shot
tokens_with_c1[:, c0:c1] = c1_parallel

# Pass 2: train c2 with c1_parallel as context
logits_c2 = model(tokens_with_c1, mask)
loss = NTP(logits_c2[:, c2_start-1:c2_end-1], tokens[:, c2_start:c2_end])
```
2× compute but qualitatively better errors (real uncertainty, not random).

**Mix parallel + teacher (70/30):** when c1 = ground truth, p2 learns "no correction needed." When c1 = parallel sample, p2 learns "here are the errors, fix them."

**Curriculum:** start 5k steps with denoising (model near-random, parallel sample has no signal). Switch to parallel sample + reinput once base recall is partially learned. Keep 30% teacher-forced c1 throughout.

---

## Experiment Roadmap

Full plan: `reports/PLAN_MULTITURN.md`

### Multi-Turn Corpus Recall (Stage 1-3 only, then refine)

| Stage | Config | DSL | Pass |
|-------|--------|-----|------|
| 1 — recent recall | `ablate_2b_recent` | `2x<h:1><x:16><z:7><q:4><y:8,from=1>` | ≥90% — warm-up |
| 2 — old recall | `ablate_2b_old` | `2x<h:1><x:16><z:7><q:4><y:8,from=0>` | ≥80% — key test |
| 3 — mixed routing | `ablate_2b_mixed` | curriculum: from=0 + from=1 | ≥80% both |

Gate: stop at Stage 3, validate, then proceed to Refine before scaling to N>2.

### Refine Experiment (after Stage 3)

Two `<q><y>` pairs per sequence. First `<y>` corrupted, second is the correction. Loss only on second `<y>`.

| Stage | Method | y_1 source | Compute |
|-------|--------|-----------|---------|
| A — denoise | `refine_denoise` | uniform substitution p~U(0.05,0.4) | 1× |
| B — parallel | `refine_parallel` | argmax forward pass | 2× |

Pass: `y_2` match% > `y_1` match% (positive Delta). Confirms correction mechanism works before adding `<d><u>` CRUD tags.

---

## Pending Refactor (do when starting multi-block work)

`train.py` is the old pre-role-tag file (509 lines, stale). Replace it:
1. Copy `train_role.py` → `train.py` (has all latest changes)
2. Fold in `train_role_kvcache.py` blockwise two-pass TF + kvcache OCD rollout under `--kv-cache` flag
3. Delete `train_role.py` and `train_role_kvcache.py`

Key additions needed in merged `train.py`:
- `--kv-cache` flag: uses blockwise prefix/suffix forward (two passes per step) + KV-cached OCD rollout
- OCD rollout: `ocd_rollout_full` (default) vs `ocd_rollout_kvcache` (prefix KV, `--kv-cache`)
- `model.forward(past_kv=, return_kv=, offset=)` API already in `model.py`
