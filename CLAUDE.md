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

## CLI Reference

```bash
# Standard training:
python -m kvmem.train --config configs/expB_chain_nullkv.py --device mps

# Eval only — load checkpoint, run eval_configs, print results, exit:
python -m kvmem.train --config configs/expB_chain_nullkv.py \
  --eval-only logs/role_<name>/checkpoints/stage0_end.pt --device mps

# Resume — full state (weights + optimizer + rng, exact continuation):
python -m kvmem.train --config configs/expB_chain_nullkv.py \
  --resume logs/role_<name>/checkpoints/stage0_end.pt --device mps

# Pretrained weights only — fresh training with warm init, no optimizer/rng restore:
python -m kvmem.train --config configs/expB_chain_nullkv.py \
  --pretrained logs/role_<name>/checkpoints/stage0_end.pt --device mps
```

**Checkpoint contents (full resume state):**
`model`, `opt`, `hp`, `stage`, `step`, `global_step`, `rng_np`, `rng_torch`

---

## Monitoring Runs

**Always print the tail command immediately after starting any training run.**
Task output path: `/private/tmp/claude-501/-Users-muaz-code-kvmem/5692f5ae-ec99-4edc-ac10-e65f033b3e3d/tasks/<task_id>.output`

```bash
# Live tqdm output (give this cmd every time a run starts):
tail -f /private/tmp/claude-501/-Users-muaz-code-kvmem/5692f5ae-ec99-4edc-ac10-e65f033b3e3d/tasks/<task_id>.output

# Structured eval metrics only:
tail -f logs/role_<name>/train.jsonl | python3 -c "
import sys,json
for l in sys.stdin:
    d=json.loads(l)
    keys=sorted(k for k in d if k.startswith('n') and '_r' in k)
    if keys: print(f's={d[\"stage\"]} @{d[\"global_step\"]}: ' + '  '.join(f'{k}={d[k]:.0f}%' for k in keys))
"
```

Current run: **none** (Exp 3b done, Exp 3c not started)
```bash
tail -f /Users/muaz/code/kvmem/logs/role_refine_multiturn/train.log

# Refine metrics:
tail -f logs/role_refine_multiturn/train.jsonl | python3 -c "
import sys,json
for l in sys.stdin:
    d=json.loads(l)
    if 'val_ref_bpb' in d:
        gap=round(d['val_bpb']-d['val_ref_bpb'],3)
        print(f'@{d[\"global_step\"]}: val_bpb={d[\"val_bpb\"]:.3f} val_ref_bpb={d[\"val_ref_bpb\"]:.3f} gap={gap:+.3f} n1_r0={d.get(\"n1_r0\",\"?\")}% {d[\"elapsed\"]}')
"
```

---

## Current Status (2026-06-04)

**Exp 3: Refine Stage A running** — `configs/refine_denoise.py`, 80k steps, ~28k complete.

| Metric | Value | Meaning |
|--------|-------|---------|
| val_ref_bpb | ~0.37 | NLL on final `<y>` given noisy `<r>` — near-perfect |
| val_bpb | ~12 | Single-turn NLL (out-of-distribution, expected) |
| draft match% | ~3% | Model's own first pass — nearly random |
| final match% | ~56% | After seeing draft, corrects using `<h>` |
| Δ | +54% | Correction gain per step |

`val_bpb >> 8` (entropy) is expected: model trained to expect `<r>` context, single-turn eval is OOD. The correction working from a 3% draft proves `<h>` is being used, not just `<r>` copying.

---

## Architecture (v2)

Snapshot of prior code: `kvmem/old/v1/`

**Tag vocabulary — RNN/DB style:**

| Tag | Name | Meaning | Causal access |
|-----|------|---------|---------------|
| `<x>` | input | source data | sees prior only |
| `<z>` | latent | intermediate learned representation before compression | sees x + prior h |
| `<h>` | hidden | fast-weight memory (KV bank) | sees x + z |
| `<q>` | query | warmup anchor (user-facing) | sees h only (x,z blocked) |
| `<y>` | output | retrieved value (user-facing final) | sees h + q (x,z blocked) |
| `<r>` | refine anchor | warmup for refine mode (like `<q>`, appears once) | sees h only (x,z blocked) |

**Attention masks — successful experiments:**

*Exp 1–B: Standard recall* `<x><z><h><q><y>` (single block)

| | `<x>` | `<z>` | `<h>` | `<q>` | `<y>` |
|---|---|---|---|---|---|
| `<x>` | ✓ | | | | |
| `<z>` | ✓ | ✓ | | | |
| `<h>` | ✓ | ✓ | ✓ | | |
| `<q>` | | | ✓ | ✓ | |
| `<y>` | | | ✓ | ✓ | ✓ |

*Exp 2/B: Multi-block recall* `<x1><z1><h1><x2><z2><h2><q><y>` (`<h2>` blocked from `<x1><z1>`)

| | `<x1>` | `<z1>` | `<h1>` | `<x2>` | `<z2>` | `<h2>` | `<q>` | `<y>` |
|---|---|---|---|---|---|---|---|---|
| `<x1>` | ✓ | | | | | | | |
| `<z1>` | ✓ | ✓ | | | | | | |
| `<h1>` | ✓ | ✓ | ✓ | | | | | |
| `<x2>` | ✓ | | | ✓ | | | | |
| `<z2>` | ✓ | ✓ | ✓ | ✓ | ✓ | | | |
| `<h2>` | | | ✓ | ✓ | ✓ | ✓ | | |
| `<q>` | | | ✓ | | | ✓ | ✓ | |
| `<y>` | | | ✓ | | | ✓ | ✓ | ✓ |

*Exp 3a: Refine Stage A* `<x><z><h><q><r><q><y>` (draft `<r>` visible to final `<y>`)

| | `<x>` | `<z>` | `<h>` | `<q1>` | `<r>` | `<q2>` | `<y>` |
|---|---|---|---|---|---|---|---|
| `<x>` | ✓ | | | | | | |
| `<z>` | ✓ | ✓ | | | | | |
| `<h>` | ✓ | ✓ | ✓ | | | | |
| `<q1>` | | | ✓ | ✓ | | | |
| `<r>` | | | ✓ | ✓ | ✓ | | |
| `<q2>` | | | ✓ | ✓ | ✓ | ✓ | |
| `<y>` | | | ✓ | ✓ | ✓ | ✓ | ✓ |

**Note on `<z>` and diff:** `<z>_t` in block t comes after `<h>_{t-1}` in sequence order, so it can attend to the prior memory state causally. This gives `<z>` the architectural capacity to compute diffs — encoding only what's new in `<x>_t` relative to `<h>_{t-1}` — without any extra tokens or mask changes. Whether it learns to use this depends on training pressure (multi-block recall with retention requirements).

Extended CRUD ops (planned): `<u>` update, `<d>` diff, `<c>` commit, `<s>` seek, `<n>` next.

**Sequence:** `<x>src</x>[<z>z_0..z_P</z>]<h>h_0..h_N</h><q>warmup</q><y>output</y>`

All attention is **pure causal** — no non-causal overrides. `<q>/<y>` are explicitly blocked from x and z, forced through `<h>` bottleneck. `<y>` is write-only.

**Bottleneck:** `slot_len` directly (no `active_slots` masking). `slot_len=1, latent_len=7` ≈ v1's `slot_len=8, active_slots=1`.

**Vocab:** V_in = 256 + 12 + slot_len + latent_len (auto-computed). V_out = 256 (data bytes only).
Note: vocab expanded from 10→12 boundary tags when `<r>/<r>` (IDs 266/267) were added. HIDDEN_SLOT_BASE shifted from 266→268. Old checkpoints (V=274) are incompatible with new code (V=276 for slot=1,latent=7).
- `data_embed: Embedding(256, d)` — data bytes (std=0.02)
- `special_embed: Embedding(V_in-256, d)` — boundary tags + slot IDs (std=0.05)

**Slot token scheme — three options:**
- **Dedicated indexed** (current, K=slot_len): slot i → `266+i`. Unique V per slot, zero collision. Vocab grows with slot_len. No extrapolation beyond training slot_len.
- **Dedicated cyclic** (K < slot_len): slot i → `266 + (i % K)`. K dedicated IDs above 255 cycle over all slots. Fixed vocab (K tokens), zero collision, extrapolates to arbitrary slot_len. Best design for scaling.
- **Looped byte** (style A): slot i → `i % 256`. Fixed vocab=256, extrapolates, but collides with data bytes.

**Dedicated cyclic is the right choice for scaling.** K is the "slot vocab budget" — train with K=8, infer with slot_len=1024 using the same 8 IDs cycling, RoPE carries absolute position. Current code uses dedicated indexed (K=slot_len=1 for now, so no practical difference). `make_hidden_slot_ids(slot_len, cycle_len=slot_len)` — set `cycle_len=8` before scaling.

**null_kv=True (recommended):** appends fixed (K=0, V=0) to every attention head before softmax. Zero-score "abstain" option. Result: 1.5-2× faster convergence, better peak bpb (0.157 vs 0.217 base). Set `null_kv=True` in hp or `--null-kv` CLI flag.

**mem_window:** controls how many prior `<h>` states each new `<h>` can attend to.
- 0 (default): full history — fast-weight accumulation
- 1: isolated — each `<h>` compresses only its own block
- N: N-step sliding window

---

## Terminology Reference

### Windows (all independent, all configurable)

| Term | What it controls | Config key | Typical value |
|---|---|---|---|
| **seg window** | bytes in one source block `<x>` | `seg_len` | 16 |
| **slot window** | `<h>` token count — compression bottleneck | `slot_len` | 1 |
| **latent length** | `<z>` latent token count per block | `latent_len` | 7 |
| **mem window** | prior `<h>` states visible to new `<h>` (-1=full, 1=isolated) | `mem_window` | -1 |
| **warmup window** | bytes used as query anchor prefix (`<q>` or `<r>` content) | `warmup_len` | 4 |
| **segment recall window** | bytes model must output per query (`<y>` length) | `out_len` | 16 (=seg_len for full recall) |
| **attempt window** | max refine attempts sampled per optimizer step | `n_attempts` | 5 |
| **noise window** | noise range per attempt `(noise_lo .. noise_hi)` | `noise_hi/lo` | 0.05–0.8 |

### Turns (sequence-level, per training example)

| Term | Structure | Loss |
|---|---|---|
| **ingest turn** | `<x><z><h>` — reads one source segment into memory | none |
| **attempt turn** | `<y><z><h>` — noisy output + memory correction block | none (context only) |
| **final turn** | `<y>` — clean ground truth after all attempts; trains copy mechanism | optional |
| **query turn** | `<q><y>` — post-refine query on updated `<h>`; must match 100% | **primary loss** |

Full refine sequence (n attempts + query):
```
[<x><z><h>] × n_blocks           ingest turns
<r>warmup</r>                     refine anchor (once)
(<y>noisy</y><z><h>) × k          attempt turns  k ~ Uniform(0, n_attempts)
<y>clean</y>                       final turn
<z><h>                             ghost correction (updates h from final)
<q>warmup</q><y>clean</y>          query turn  ← loss here
```
k=0: no attempt turns → `<r><y_final><z><h><q><y_query>` — same structure as standard recall.

### Steps (training-level)

| Term | Meaning | Config key |
|---|---|---|
| **optimizer step** | one forward+backward+weight update | `n_steps` |
| **warmup step** | LR ramp steps before cosine decay | `warmup_steps` |
| **eval step** | frequency of eval runs | `eval_every` |
| **log step** | frequency of JSONL train logging | `log_every` |
| **curriculum stage** | one training phase with fixed mode/n_blocks/steps | `curriculum` list |

---

**Sequence DSL:** `<x:16><z:7><h:1><q:4><y:8>` → parsed by `kvmem/seq_dsl.py` → `SeqSpec`.

**Curriculum DSL:** `kvmem/curriculum_dsl.py` — batch scheduler + eval config.
```
seq_spec | stage, stage @eval:eval_spec
```

Stage token: `nN/rK/Xk[/wM][/mMODE]` — n_blocks / recall / steps / mem_window / op-mode

```
"<x:16><z:7><h:1><q:4><y:8> | n1/r0/40k, n2/r1/40k +n2/r0, n2/r[0,1]/80k/w1 @eval:n1/r0,n2/r0,n2/r1"
```

| Syntax | Meaning |
|--------|---------|
| `nN/rK/Xk` | stage: n_blocks=N, recall=K, steps=X |
| `r[0,1]` | mixed batch: each example randomly draws recall from list |
| `+nN/rK` | overlap: merge into previous stage's batch distribution |
| `wM` | mem_window (-1=full, 1=isolated) |
| `mMODE` | op pattern: `mend` (default), `mint` (interleaved), `macc` (ingest-only), `mmix` (random mix) |
| `@eval:nN/rK,...` | eval configs tested every `eval_every` steps (independent of training) |

Eval is independent of curriculum — `@eval:` specifies exactly which (n_blocks, recall_from) pairs are tested at each eval step. If omitted: auto-derived from all stages + `n1/r0` baseline.

Returns `(SeqSpec, curriculum_list, eval_configs)` — pass `eval_configs` to `hp['eval_configs']`.

**Hparams absorbed by DSL (no longer set manually):**
`seg_len`, `slot_len`, `latent_len`, `warmup_len`, `out_len` → from seq spec  
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

| Exp | Status | Key result |
|-----|--------|-----------|
| Exp 1: Dataset ablation | ✓ Done | 100% match, ds_random fastest |
| Exp 2: Sequential routing | ✓ Done | catastrophic forgetting confirmed |
| Exp 2b: Cold mixed routing | ✓ Done | both dirs learn simultaneously, 91% |
| Exp A: null_kv ablation | ✓ Done | 1.5-2× faster, bpb 0.157 vs 0.217 |
| **Exp B: Chain extrapolation** | ✓ Done | **n=4 95%, n=5 91-94% trained only on n=1,2,3. Best overall: all ≥91%** |
| **Exp 3a: Refine Stage A** | ✓ Done | draft 1.6% → final 92.2% Δ+90.6%. val_ref_bpb=0.063. Sawtooth fails (75%). |
| **Exp 3b: Refine multi-turn** | ✓ Done | FAILED: 17.2% final (vs 92.2% Exp 3a). Train-eval dist mismatch. See findings. |
| **Exp 3c: Joint trajectory mix** | ⏳ Next | flat noise, joint I-Q/I-R/I-I-Q training, no regression + extrapolation goals |
| Exp 4: Natural language corpus | ⏳ Future | replace random bytes with text+line numbers |

Full results: `reports/EXP_RESULTS_SUMMARY.md`

**Key findings:**
- Cold mixed routing (`r[0,1]` from step 0) — no forgetting
- null_kv=True — always use, 1.5-2× faster  
- **Chain extrapolation** — training on n=1,2,3 gives 88-95% on n=4,5 (phase transition at 2k steps into n=3 training)
- mmix mode (k~Uniform(1,n) queries) — trains interactive ingestion+query patterns
- **Refine working** — model corrects from terrible first draft (3%) to 56% using `<h>`. val_bpb>>8 (expected: model requires `<r>` context). val_ref_bpb≈0.37 (near-perfect in-distribution).

---

## Training Trajectory Taxonomy

Three atomic ops (multi-turn refine counts as 1):
- `I` = ingest one block
- `Q` = query — single pass, `<q><y>`, loss on `<y>`
- `R` = refine — variable k draft turns `<q><r>` + final `<q><y>`, loss on final `<y>` only

Generalisation requirements:
- **No regression**: simpler trajectories (fewer ops) must stay at full performance
- **Extrapolation**: more ops/turns than trained on should work (like Exp B n-chain extrapolation)
- **Joint training**: training on complex trajectories alone breaks simple ones — must mix all types

**Enumeration — 1 block:**

| Trajectory | Tests | Status |
|---|---|---|
| `I Q` | baseline recall | ✓ Exp 1 |
| `I R` | single-block refine (1 turn) | ✓ Exp 3a |
| `I R` (k=1..5 rand) | multi-turn refine | ✓ Exp 3b (failed — see below) |
| `I Q Q` | consistency: same block twice | ✗ |
| `I Q R` | query then self-correct | ✗ |
| `I R Q` | refine then verify correction | ✗ |
| `I R R` | two rounds of refinement | ✗ |
| `I Q R Q` | query, refine, verify | ✗ |

**Enumeration — 2 blocks:**

| Trajectory | Tests | Status |
|---|---|---|
| `I I Q₀` | retention: query old after new ingest | ✓ Exp 2b r0 |
| `I I Q₁` | encoding: query new | ✓ Exp 2b r1 |
| `I I R₀` | retention + correction: refine OLD after update | ✗ high priority |
| `I I R₁` | refine new block | ✓ Exp 3b (partial) |
| `I Q₀ I Q₀` | interleaved same target — retention under update | ✗ |
| `I Q₀ I Q₁` | standard interleaved | ✓ int mode |
| `I Q₀ I R₀` | query, ingest, refine old | ✗ |
| `I R₁ I Q₀` | refine new, ingest again, test x0 survived | ✗ |
| `I I Q₁ Q₀` | sequential dual query | ✗ |
| `I I R₁ Q₀` | refine new, prove old retained | ✗ **Exp 3c SRS** |
| `I I R₀ Q₁` | refine old, prove new encoded | ✗ |
| `I I R₀ R₁` | refine both blocks | ✗ |

**Enumeration — 3+ blocks:**

| Trajectory | Tests | Status |
|---|---|---|
| `I I I Q₀` | deep retention | ✓ Exp B chain |
| `I I I R₀` | deep chain + correct oldest | ✗ |
| `I I I Q₀ Q₂` | deep retention + current | ✗ |
| `I Q₀ I Q₀ I Q₀` | SRS: review after every ingest | ✗ |
| `I I Q₀ I Q₀` | retention at spacing 1, then spacing 2 | ✗ |

**Sampled high-value set (novel, not yet trained):**

| Priority | Trajectory | Why |
|---|---|---|
| ✓✓ | `I I R₀` | h must retain x0 for self-correction — direct diff signal |
| ✓✓ | `I I R₁ Q₀` | **Exp 3c**: refine new, prove old survived — SRS under pressure |
| ✓✓ | `I Q₀ I Q₀` | retention under update: after ingesting x1, x0 must still work |
| ✓✓ | `I Q₀ I Q₀ I Q₀` | pure SRS: review x0 at spacing 1, 2, 3 |
| ✓ | `I Q R` | online correction: query own output, then self-correct |
| ✓ | `I I Q₁ Q₀` | sequential dual-query from same h |
| ✓ | `I I I R₀` | maximum spacing + correction |
| ✓ | `I R₁ I Q₀` | refine, continue ingesting, test x0 retained |

**Joint training distribution (Exp 3c design):**
```
30% → I Q         (no regression: standard recall)
20% → I R         (single-block refine, k~1..5 flat noise)
20% → I I Q₀      (retention baseline)
15% → I I R₁ Q₀   (SRS: refine new + test old)
15% → I Q₀ I Q₀   (interleaved same target)
```

---

## SRS Heuristic for Trajectory Design

**Spaced Repetition System insight:** the forgetting curve for `<h>` mirrors biological memory — each new `I_k` risks overwriting prior states. SRS says train at the spacing where the model is *about to forget*.

| SRS concept | Trajectory | Spacing |
|---|---|---|
| Immediate review | `I1 Y1` | 0 |
| Short interval | `I1 I2 → Y1` | 1 |
| Long interval | `I1...I5 → Y1` | 4 |
| Increasing intervals | `I1 Y1 I2 Y1 I3 I4 Y1 ...` | geometric |

**Key implication:** sample `n_blocks` per example from a distribution (geometric or uniform over 1..N_max) rather than fixing it per stage. This trains the model on all spacings simultaneously — easy retention (short spacing) and hard retention (long spacing) in the same batch.

**Refine + SRS:** `I^n → R Y` is "cued recall at spacing n" — the SRS hint mechanism. `<r>` is the retrieval cue; `<y>` is the test; the correction signal is the feedback. Long-spacing refine (`I^5 → R Y_0`) is the hardest SRS card.

**Non-monotonic improvement penalty:** when training multi-turn refine (R R Y), each turn should improve over the prior. A penalty on non-monotonic turns (turn k worse than turn k-1) provides direct training signal. `mono_penalty` hp key (default 0.0).

---

## SRS Experiment Plan

Progression of refine experiments following the SRS (Spaced Repetition System) heuristic:

**Exp 3a** (done) — Single block, 1 draft turn, fixed noise. 92.2% final from 1.6% draft. Sawtooth fails (75%).

**Exp 3b** (done, failed) — k~Uniform(1,5), descending noise. 17.2% final. Root cause: train-eval distribution mismatch — later drafts get near-zero synthetic noise during training (U(0, noise_hi*(K-j)/K)), but model's own AR-decoded drafts at eval time are still very noisy (≈15%). Model learns "5th draft is clean, just copy" but sees noisy 5th draft at eval. Fix: flat noise schedule — same noise range for all draft turns.

**Exp 3c** (next) — Joint trajectory mix with flat noise:
- Mix I Q, I R (flat noise k~1..5), I I Q₀, I I R₁ Q₀, I Q₀ I Q₀ per step
- Goals: no regression on I Q (92%+), extrapolation to k>5, retention
- Two-block SRS trajectory: ingest x1, ingest x2, refine x2 (k turns), query x0
- Requires: extending sequence layout to support trailing Q₀ after refine turn
- Config: `configs/refine_joint.py`

**Key constraint for all future refine experiments:**
- Flat noise: all draft turns use same U(lo, hi) range — keeps training drafts at same quality distribution as model's own eval outputs
- Joint training: always include I Q batches alongside I R batches to prevent I Q regression
- Extrapolation eval: always eval at k_max + 2 turns to track generalisation

---

## Key Findings Log

| Date | Finding |
|------|---------|
| 2026-06-03 | v1: 93.8% match (seg=16, active=1, full-pass TF) |
| 2026-06-03 | v2: 98-100% match with slot_len=1, latent_len=7 |
| 2026-06-03 | active_slots masking was wrong — slot_len IS the bottleneck |
| 2026-06-03 | Non-causal slot→src was a mistake — pure causal works and is simpler |
| 2026-06-03 | kv_cache default hurts match% — full-pass TF is correct default |
| 2026-06-04 | Catastrophic forgetting between sequential stages confirmed |
| 2026-06-04 | 2-block recall (from=0 and from=1) each achieves ~98% in isolation |
| 2026-06-04 | ar_decode_role was broken for multi-block — now uses correct n_blocks eval |
| 2026-06-04 | Dedicated cyclic IDs (266+(i%K)) is the right scaling design — fixed vocab K, zero data collision, extrapolates to arbitrary slot_len via cycle; looped byte (i%256) also works but collides with data |
| 2026-06-04 | null_kv=True: 1.5-2× faster convergence, better peak val_bpb (0.157 vs 0.217 base). Should be default for future runs. |
| 2026-06-04 | Cold mixed routing (r[0,1] from step 0) solves catastrophic forgetting — both directions learn simultaneously, no sequential overwriting |
| 2026-06-04 | Interleaved mode (mmix): random end/int per step, random query count k∈[1,n], each query targets random prior block — trains interactive "sometimes ingest, sometimes query" |
| 2026-06-05 | **Exp B complete: chain extrapolation confirmed.** Training on n=1,2,3 → n=4 95%, n=5 91-94% without ever training on 4 or 5 blocks. Phase transition at 2k steps into n=3 training. Algorithm generalises beyond training chain length. |
| 2026-06-04 | Added `<r>/<r>` refinement tags (IDs 266/267). HIDDEN_SLOT_BASE shifted 266→268, N_BOUNDARY_TAGS 10→12, V=276 (was 274). Old checkpoints incompatible. |
| 2026-06-04 | **Refine Stage A working**: model corrects from 3% draft to 56% final using `<h>`. val_ref_bpb≈0.37. `<z>` has architectural access to prior `<h>` causally — diff capacity available without new tokens. |
| 2026-06-04 | SRS heuristic: vary n_blocks per example (geometric/uniform over 1..N_max) to train all retention spacings simultaneously rather than fixed n per stage. |
| 2026-06-04 | **Exp 3b failed**: descending noise (turn j of K gets U(0, noise_hi*(K-j)/K)) causes train-eval distribution mismatch. Later training drafts are near-clean; AR eval drafts are still noisy. Model learns to copy clean drafts, fails on noisy eval. Fix: flat noise schedule, same range all turns. |
| 2026-06-04 | Joint training required: training only on I R breaks I Q (model expects `<r>` context). Must mix trajectory types per step. Simple trajectories in mix prevent regression. |
| 2026-06-04 | Trajectory generalisation goals: (1) no regression at k < trained max, (2) extrapolation at k > trained max (same mechanism as Exp B chain extrapolation). Both require rand_turns + flat noise + trajectory mixing. |

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
