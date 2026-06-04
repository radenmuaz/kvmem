# kvmem — Project Notes for Claude

## Vision

**A model that reads any document once and answers any question about it — without storing the document, without backprop, and without retraining.**

The `<h>` hidden state is a compressed fast-weight representation of whatever was ingested. At inference, base weights are frozen. Reading = forward passes that update `<h>`. Querying = NTP from a warmup prefix.

**Instruction following without fine-tuning:** Feed the IT dataset as a corpus. Fast weights compress the instruction-answer patterns. Query with `"Q: [instruction]\nA: "` as the NTP warmup. This is **compressed few-shot learning** — N examples in O(slot_len) tokens instead of O(N × example_len).

**Theoretical ceiling:**
> Ingest any corpus forward-pass-only, answer arbitrary queries at quality comparable to a full-context LLM — but in O(slot_len) memory instead of O(corpus_length) KV cache.

---

## Primary Goal

**Learn in-context LM without backprop.**

Base weights learn one thing: *how to update `<h>` fast weights so `<y>` predictions improve.*

```
train BPB ≈ val BPB    →  generalised: learned the algorithm
train BPB << val BPB   →  memorised
val BPB → entropy(corpus)  →  milestone: compression is effective
```

**Windowed recall on random bytes** is a diagnostic task (entropy=8 bits/byte). Progression:
1. Random bytes → verify mechanism ← *current*
2. Structured text with line numbers → content-addressable retrieval
3. Natural language corpus → LM prior learning
4. Cross-corpus generalisation → real milestone

Self-correction and ground truth are a **means**, not an end. OCD/correction trains the `<h>` update rule.

---

## Current Status (2026-06-04)

**v2 architecture, Exp 1 results (dataset ablation, seg=16, slot=1, intermed=7):**

| Run | Best match% | Steps |
|-----|------------|-------|
| ds10k | 98% | 70k |
| ds20k | **100%** | 65k |
| ds40k | 97% | 50k |
| ds_random | **100%** | 40k |

Infinite stream (ds_random) converges fastest — supports generalisation hypothesis.

**Exp 2 running (multi-turn, 9 stages):** See `plan/PLAN_EXP2.md` for plan.

Early results from generalisation evals at stage end:

| After stage | 1-block | 2b from=0 (old) | 2b from=1 (recent) |
|------------|---------|-----------------|-------------------|
| s0 (1-block only) | 92% | 9% | 0% |
| s1 (2-block recent) | 27% ← **forgot** | 0% | 98% |
| s2 (2-block old) | 5% ← **forgot** | 94% | 0% |

**Key finding:** catastrophic forgetting between stages — sequential training overwrites. Mixed stages (s3-4) will test if joint training retains both. Results pending.

---

## Architecture (v2)

Snapshot of prior code: `kvmem/old/v1/`

**Tag vocabulary — RNN/DB style:**

| Tag | Name | Meaning | Causal access |
|-----|------|---------|---------------|
| `<x>` | input | source data | sees prior only |
| `<z>` | intermediate | pre-process before compression | sees x |
| `<h>` | hidden | fast-weight memory (KV bank) | sees x + z |
| `<q>` | query | warmup anchor | sees h only (x,z blocked) |
| `<y>` | output | retrieved value | sees h + q (x,z blocked) |

Extended CRUD ops (planned): `<u>` update, `<d>` diff, `<c>` commit, `<s>` seek, `<n>` next.

**Sequence:** `<x>src</x>[<z>z_0..z_P</z>]<h>h_0..h_N</h><q>warmup</q><y>output</y>`

All attention is **pure causal** — no non-causal overrides. `<q>/<y>` are explicitly blocked from x and z, forced through `<h>` bottleneck. `<y>` is write-only.

**Bottleneck:** `slot_len` directly (no `active_slots` masking). `slot_len=1, intermed_len=7` ≈ v1's `slot_len=8, active_slots=1`.

**Vocab:** V_in = 256 + 10 + slot_len + intermed_len (auto-computed). V_out = 256 (data bytes only).
- `data_embed: Embedding(256, d)` — data bytes (std=0.02)
- `special_embed: Embedding(V_in-256, d)` — boundary tags + slot IDs (std=0.05)

**Slot token scheme — three options:**
- **Dedicated indexed** (current, K=slot_len): slot i → `266+i`. Unique V per slot, zero collision. Vocab grows with slot_len. No extrapolation beyond training slot_len.
- **Dedicated cyclic** (K < slot_len): slot i → `266 + (i % K)`. K dedicated IDs above 255 cycle over all slots. Fixed vocab (K tokens), zero collision, extrapolates to arbitrary slot_len. Best design for scaling.
- **Looped byte** (style A): slot i → `i % 256`. Fixed vocab=256, extrapolates, but collides with data bytes.

**Dedicated cyclic is the right choice for scaling.** K is the "slot vocab budget" — train with K=8, infer with slot_len=1024 using the same 8 IDs cycling, RoPE carries absolute position. Current code uses dedicated indexed (K=slot_len=1 for now, so no practical difference). `make_hidden_slot_ids(slot_len, cycle_len=slot_len)` — set `cycle_len=8` before scaling.

**mem_window:** controls how many prior `<h>` states each new `<h>` can attend to.
- 0 (default): full history — fast-weight accumulation
- 1: isolated — each `<h>` compresses only its own block
- N: N-step sliding window

**Sequence DSL:** `<x:16><z:7><h:1><q:4><y:8>` → parsed by `kvmem/seq_dsl.py` → `SeqSpec`.

**Curriculum DSL:** `kvmem/curriculum_dsl.py` — batch scheduler + eval config.
```
seq_spec | stage, stage @eval:eval_spec
```

Stage token: `nN/rK/Xk[/wM]` — n_blocks / recall / steps / mem_window

```
"<x:16><z:7><h:1><q:4><y:8> | n1/r0/40k, n2/r1/40k +n2/r0, n2/r[0,1]/80k/w1 @eval:n1/r0,n2/r0,n2/r1"
```

| Syntax | Meaning |
|--------|---------|
| `nN/rK/Xk` | stage: n_blocks=N, recall=K, steps=X |
| `r[0,1]` | mixed batch: each example randomly draws recall from list |
| `+nN/rK` | overlap: merge into previous stage's batch distribution |
| `wM` | mem_window (-1=full, 1=isolated) |
| `@eval:nN/rK,...` | eval configs tested every `eval_every` steps (independent of training) |

Eval is independent of curriculum — `@eval:` specifies exactly which (n_blocks, recall_from) pairs are tested at each eval step. If omitted: auto-derived from all stages + `n1/r0` baseline.

Returns `(SeqSpec, curriculum_list, eval_configs)` — pass `eval_configs` to `hp['eval_configs']`.

**Hparams absorbed by DSL (no longer set manually):**
`seg_len`, `slot_len`, `intermed_len`, `warmup_len`, `out_len` → from seq spec  
`n_blocks`, `recall_from`/`recall_froms`, `mem_window` → from curriculum stage tokens  
`active_slots`, `slot_style`, `V` → removed entirely (slot_len is the bottleneck, dedicated indexed always)

---

## Fast Weights

`<h>` is a **running state**, not a stack. Cross-`<h>` attention IS the update mechanism:
```
h_0 = compress(x_0)
h_t = update(h_{t-1}, x_t)   ← learns to mimic GD without GD
```

The model must learn to propagate what matters through the chain (vanishing information problem). **Line numbers as explicit keys** help: `"0042 content"` gives the update an addressable key; `<q>0042 </q>` retrieves it.

---

## Experiment Roadmap

| Exp | Plan | Status |
|-----|------|--------|
| Exp 1: Dataset ablation | — | ✓ Done — 100% match on ds20k and ds_random |
| Exp 2: Multi-turn recall + mem_window | `plan/PLAN_EXP2.md` | 🔄 Running (stage 2/9) |
| Exp 3: Refine (self-correction) | `plan/PLAN_EXP2.md` §Refine | After Exp 2 stage 3 |

---

## Key Findings Log

| Date | Finding |
|------|---------|
| 2026-06-03 | v1: 93.8% match (seg=16, active=1, full-pass TF) |
| 2026-06-03 | v2: 98-100% match with slot_len=1, intermed_len=7 |
| 2026-06-03 | active_slots masking was wrong — slot_len IS the bottleneck |
| 2026-06-03 | Non-causal slot→src was a mistake — pure causal works and is simpler |
| 2026-06-03 | kv_cache default hurts match% — full-pass TF is correct default |
| 2026-06-04 | Catastrophic forgetting between sequential stages confirmed |
| 2026-06-04 | 2-block recall (from=0 and from=1) each achieves ~98% in isolation |
| 2026-06-04 | ar_decode_role was broken for multi-block — now uses correct n_blocks eval |
| 2026-06-04 | Dedicated cyclic IDs (266+(i%K)) is the right scaling design — fixed vocab K, zero data collision, extrapolates to arbitrary slot_len via cycle; looped byte (i%256) also works but collides with data |

---

## Reference

| What | Where |
|------|-------|
| Exp 1 results | `reports/EXP1_DATASET_ABLATION.md` |
| Exp 2 live tracking | `reports/EXP2_MULTITURN_TRACKING.md` |
| Exp 2 plan + refine plan | `plan/PLAN_EXP2.md` |
| Tag naming rationale | `reports/tag_naming.md` |
| KV dimension tables | `reports/kv_dims.md` |
| v1 results and findings | `reports/v1/` |
| Old plans (archived) | `plan/archive/` |
| Self-correction theory | `plan/PLAN_REFINED.md` §Direction B |
| Hidden state viz ideas | `plan/VIZ_HIDDEN_STATES.md` |
| v1 code snapshot | `kvmem/old/v1/` |
| Active configs | `configs/` |
