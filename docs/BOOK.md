# kvmem — Research Book

> **A model that reads any document once and answers any question about it — without storing the document, without backprop, and without retraining.**

---

## Table of Contents

1. [Vision & Goal](#1-vision--goal)
2. [Architecture](#2-architecture)
3. [Sequence Design & DSL](#3-sequence-design--dsl)
4. [Training Reference](#4-training-reference)
5. [Experiment Record](#5-experiment-record)
6. [Trajectory Taxonomy](#6-trajectory-taxonomy)
7. [Raw Data & Reports](#7-raw-data--reports)

---

## 1. Vision & Goal

The `<h>` hidden state is a compressed fast-weight representation of whatever was ingested. At inference, base weights are frozen. Reading = forward passes that update `<h>`. Querying = next-token prediction from a warmup prefix.

**Theoretical ceiling:**
> Ingest any corpus forward-pass-only, answer arbitrary queries at quality comparable to a full-context LLM — but in O(slot_len) memory instead of O(corpus_length) KV cache.

**Primary learning objective:**

```
train BPB ≈ val BPB    →  generalised: learned the update algorithm
val BPB → entropy(corpus)  →  milestone: compression is effective
```

Base weights learn one thing: *how to update `<h>` fast weights so `<y>` predictions improve.*

**Diagnostic task progression:**
1. Random bytes — verify mechanism (entropy=8 bits/byte) ← *current*
2. Structured text with line numbers — content-addressable retrieval
3. Natural language corpus — LM prior learning
4. Cross-corpus generalisation — real milestone

---

## 2. Architecture

### 2.1 Tag Vocabulary

| Tag | Name | Meaning | Causal access |
|-----|------|---------|---------------|
| `<x>` | input | source data | sees prior only |
| `<z>` | latent | intermediate before compression | sees x + prior h |
| `<h>` | hidden | fast-weight memory (KV bank) | sees x + z |
| `<q>` | query | warmup anchor (user-facing) | sees h only |
| `<y>` | output | retrieved value | sees h + q |
| `<r>` | refine anchor | draft warmup, appears once | sees h only |

Boundary tag IDs 256–267. Data bytes 0–255. Slot tokens 268+. V = 256 + 12 + slot_len + latent_len.

Code snapshot of v1: `kvmem/old/v1/`

### 2.2 Attention Masks

All attention is **pure causal** — no non-causal overrides. `<q>/<y>` are explicitly blocked from `<x>` and `<z>`, forced through the `<h>` bottleneck. See §3.3 for the refine mask.

*Standard recall — single block:*

| | `<x>` | `<z>` | `<h>` | `<q>` | `<y>` |
|---|:---:|:---:|:---:|:---:|:---:|
| `<x>` | ✓ | | | | |
| `<z>` | ✓ | ✓ | | | |
| `<h>` | ✓ | ✓ | ✓ | | |
| `<q>` | | | ✓ | ✓ | |
| `<y>` | | | ✓ | ✓ | ✓ |

*Multi-block recall — `<h2>` blocked from `<x1><z1>`:*

| | `<x1>` | `<z1>` | `<h1>` | `<x2>` | `<z2>` | `<h2>` | `<q>` | `<y>` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `<x1>` | ✓ | | | | | | | |
| `<z1>` | ✓ | ✓ | | | | | | |
| `<h1>` | ✓ | ✓ | ✓ | | | | | |
| `<x2>` | ✓ | | | ✓ | | | | |
| `<z2>` | ✓ | ✓ | ✓ | ✓ | ✓ | | | |
| `<h2>` | | | ✓ | ✓ | ✓ | ✓ | | |
| `<q>` | | | ✓ | | | ✓ | ✓ | |
| `<y>` | | | ✓ | | | ✓ | ✓ | ✓ |

### 2.3 Fast Weights

`<h>` is a **running state**, not a stack. Cross-`<h>` attention IS the update mechanism:

```
h_0 = compress(x_0)
h_t = update(h_{t-1}, x_t)   ← learns to mimic GD without GD
```

`<z>_t` comes after `<h>_{t-1}` in sequence order, giving it architectural capacity to compute diffs — encoding only what's new in `<x>_t` relative to `<h>_{t-1}`.

### 2.4 Vocab & Slot Schemes

**Current:** dedicated indexed — slot i → `268+i`. Unique ID per position, vocab grows with slot_len.

**Recommended for scaling:** dedicated cyclic — slot i → `268 + (i % K)`. Fixed K-token vocab, extrapolates to arbitrary slot_len via RoPE position. Use `make_hidden_slot_ids(slot_len, cycle_len=8)`.

**Flags:**
- `null_kv=True` — appends fixed (K=0, V=0) before softmax. 1.5–2× faster convergence, better peak bpb. Always use. [→ Exp A]
- `mem_window` — how many prior `<h>` states each new `<h>` can attend to. `-1`=full, `1`=isolated, `N`=sliding window.

---

## 3. Sequence Design & DSL

### 3.1 Terminology

**Windows (config keys):**

| Term | Controls | Key | Default |
|---|---|---|---|
| seg window | bytes in one `<x>` block | `seg_len` | 16 |
| slot window | `<h>` token count — bottleneck | `slot_len` | 1 |
| latent length | `<z>` tokens per block | `latent_len` | 7 |
| mem window | prior `<h>` states visible | `mem_window` | -1 |
| warmup window | query anchor bytes | `warmup_len` | 4 |
| output length | bytes recalled per query | `out_len` | 16 (`-1` = seg_len) |
| attempt window | max refine turns per step | `n_attempts` | 5 |

**Training ops:**

| Op | Structure | Loss |
|---|---|---|
| `I` ingest | `<x><z><h>` | none |
| `Q` query | `<q><y>` | on `<y>` |
| `R` refine | `<r>` + k×(`<y><z><h>`) + `<y>` + `<z><h>` + `<q><y>` | on final `<q><y>` only |

### 3.2 Refine Sequence Layout

```
[<x><z><h>] × n_blocks           ingest turns
<r>warmup</r>                     refine anchor (once)
(<y>noisy</y><z><h>) × k          attempt turns  k ~ Uniform(0, n_attempts)
<y>clean</y>                       copy turn (trains copy mechanism)
<z><h>                             ghost correction (updates h from clean)
<q>warmup</q><y>clean</y>          post-refine query  ← PRIMARY LOSS
```

Post-refine `<q><y>` sees **only** the ghost correction `<h>` and own tokens — strict bottleneck forcing recall through the final updated memory.

### 3.3 Key Config Keys (refine mode)

| Key | Effect |
|---|---|
| `aux_attempt_loss` | NLL weight on each attempt `<y>` vs clean GT — **essential** to fix sawtooth [→ §5.3] |
| `mono_penalty` | Penalty for attempt k+1 worse than k |
| `noise_skew` | Draft noise ramps 0→2p left→right (matches AR error distribution) |
| `ls_max` | Positional label smoothing: ε=0 at pos 0, ε=ls_max at pos N-1 |
| `ls_anneal_steps` | ε decays linearly to 0 over N steps |

### 3.4 Sequence DSL

`<x:16><z:7><h:1><q:4><y:8>` → parsed by `kvmem/seq_dsl.py` → `SeqSpec`.

**Curriculum DSL** (`kvmem/curriculum_dsl.py`):

| Syntax | Meaning |
|---|---|
| `nN/rK/Xk` | stage: n_blocks=N, recall=K, steps=X |
| `r[0,1]` | mixed batch: each example draws recall randomly |
| `+nN/rK` | merge into previous stage's batch distribution |
| `wM` | mem_window |
| `mMODE` | `mend` (default) / `mint` (interleaved) / `macc` / `mmix` |
| `@eval:nN/rK,...` | eval configs at every eval step |

### 3.5 `mode='joint'` (added Exp 3c)

Per-step trajectory sampling from a weighted mixture. Config key `joint_mix`:

```python
joint_mix=[
    dict(traj='end', n_blocks=1, recall_from=0, weight=0.30),   # I Q
    dict(traj='ref', n_blocks=1, recall_from=0, weight=0.20,    # I R Q
         n_attempts=5, noise_lo=0.3, noise_hi=0.8),
    dict(traj='end', n_blocks=2, recall_from=0, weight=0.20),   # I I Q₀
    dict(traj='int', n_blocks=2, recall_from=0, weight=0.30),   # interleaved
]
```

`val_ref_bpb` (post-refine query NLL) is computed in joint mode via a fixed held-out refine batch.

---

## 4. Training Reference

### 4.1 CLI

```bash
# Train:
python -m kvmem.train --config configs/refine_joint.py --device mps

# Eval only:
python -m kvmem.train --config configs/refine_joint.py \
  --eval-only logs/role_<name>/checkpoints/stage0_end.pt --device mps

# Resume (full state: weights + optimizer + rng):
python -m kvmem.train --config configs/refine_joint.py \
  --resume logs/role_<name>/checkpoints/stage0_end.pt --device mps

# Warm init only (fresh optimizer):
python -m kvmem.train --config configs/refine_joint.py \
  --pretrained logs/role_<name>/checkpoints/stage0_end.pt --device mps
```

Checkpoint keys: `model`, `opt`, `hp`, `stage`, `step`, `global_step`, `rng_np`, `rng_torch`

### 4.2 Monitoring

```bash
# Live metrics (refine mode):
tail -f logs/role_<name>/train.jsonl | python3 -c "
import sys,json
for l in sys.stdin:
    d=json.loads(l)
    if 'val_ref_bpb' in d:
        t_keys=sorted(k for k in d if k.startswith('n1_r0_t'))
        tstr=' '.join(f'{k}={d[k]}%' for k in t_keys)
        print(f'@{d[\"global_step\"]}: val_bpb={d[\"val_bpb\"]:.3f} val_ref_bpb={d[\"val_ref_bpb\"]:.3f} n1_r0={d.get(\"n1_r0\",\"?\")}% {tstr} {d[\"elapsed\"]}')
"

# Match% only (standard mode):
tail -f logs/role_<name>/train.jsonl | python3 -c "
import sys,json
for l in sys.stdin:
    d=json.loads(l)
    keys=sorted(k for k in d if k.startswith('n') and '_r' in k)
    if keys: print(f'@{d[\"global_step\"]}: ' + '  '.join(f'{k}={d[k]:.0f}%' for k in keys))
"
```

**Verbose eval** (`verbose_eval=True` in hp): prints hex bytes for each attempt turn at every eval step. Useful for diagnosing sawtooth. Example output:

```
[up_counter] wm=20 20 20 20  gt=20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f
  t1: 21 22 20 24 23 25 26 28 29 2a 2a 2d 2e 2d 2c 00  (44%)
  t2: 20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d 2e 2f  (100%) ✓
```

### 4.3 Performance

On MPS (M-series Mac):
- Training: ~41 it/s
- AR decode eval (7 attempts × 16 tokens × 3 configs): **~10 min per checkpoint**
- Recommendation: `eval_every=10000` — `val_ref_bpb` is the sufficient live signal; save AR eval for final checkpoint inspection

---

## 5. Experiment Record

### 5.1 Roadmap

| Exp | Status | Result |
|-----|--------|--------|
| Exp 1: Dataset ablation | ✓ | 100% match, ds_random fastest |
| Exp 2: Sequential routing | ✓ | Catastrophic forgetting confirmed |
| Exp 2b: Cold mixed routing | ✓ | Both dirs simultaneously, 91% |
| Exp A: null_kv ablation | ✓ | 1.5–2× faster, bpb 0.157 vs 0.217 |
| Exp B: Chain extrapolation | ✓ | n=4 95%, n=5 91–94% from n=1,2,3 only |
| Exp 3a: Refine Stage A | ✓ | draft→final Δ+90.6%, val_ref_bpb=0.063. Sawtooth (75%) |
| Exp 3b: Refine multi-turn | ✓ | **FAILED** 17.2%. Descending noise → train-eval mismatch |
| Exp 3c: Joint trajectory mix | ✓ | **val_ref_bpb=0.025, n1_r0=82%, monotonic t1≤t2≤t3** |
| Exp 4: Natural language corpus | ⏳ | Replace random bytes with text+line numbers |

### 5.2 Established Principles

- **null_kv=True** — always use. 1.5–2× convergence, better peak bpb. [Exp A]
- **Cold mixed routing** — `r[0,1]` from step 0 prevents catastrophic forgetting. [Exp 2b]
- **Chain extrapolation** — training on n=1,2,3 generalises to n=4,5 (phase transition at 2k steps into n=3). [Exp B]
- **Joint training required** — training on I R alone breaks I Q. Always mix trajectory types. [Exp 3b→3c]
- **Flat noise** — all draft turns use same U(lo, hi) range. Descending noise causes train-eval mismatch. [Exp 3b]
- **aux_attempt_loss** — essential for multi-turn refine. Without it, model ignores drafts (sawtooth). [Exp 3c]

### 5.3 Exp 3c Detail (latest)

Config: `configs/refine_joint.py` — 80k steps, 6h43m, MPS.

**Training mix:** 30% I Q + 20% I R Q (k~0..5 flat noise) + 20% I I Q₀ + 30% interleaved n=2

**Key progression:**

| Step | val_ref_bpb | n1_r0 | Note |
|------|-------------|-------|------|
| 1 | 8.012 | 0% | init |
| 8k | 1.624 | 17% | dropping fast |
| 14k | 0.977 | 34% | **<1.0** |
| 20k | 0.520 | 53% | **>50%** |
| 32k | 0.179 | 62% | t2>t1 first time — sawtooth broken |
| 80k | 0.025 | 77% | monotonic t1≤t2≤t3 |

**The sawtooth problem and fix:** Without `aux_attempt_loss`, model has no gradient on attempt turns (loss only on post-refine query). Correction blocks learn to ignore drafts and regenerate fresh (t1=50%→t2=5%). Fix: direct NLL supervision on each attempt `<y>` vs clean GT gives gradient through the correction path. Sawtooth resolved by step 32k.

**New features added:**
- `mode='joint'` — per-step trajectory sampling
- `out_len=-1` — full segment recall (resolves to seg_len)
- `verbose_eval` — hex byte printout per attempt
- `val_ref_bpb` in joint mode
- `aux_attempt_loss`, `mono_penalty`
- `noise_skew` — positional noise ramp
- `ls_max` / `ls_anneal_steps` — positional label smoothing

---

## 6. Trajectory Taxonomy

Three atomic ops:
- `I` = ingest one block
- `Q` = standard query `<q><y>`
- `R Q` = refine (k attempts) + post-refine verify query

### 6.1 SRS Heuristic

The forgetting curve for `<h>` mirrors biological memory — each new `I_k` risks overwriting prior states. Train at the spacing where the model is *about to forget*.

| SRS concept | Trajectory | Spacing |
|---|---|---|
| Immediate review | `I Q` | 0 |
| Short interval | `I I Q₀` | 1 |
| Long interval | `I I I I I Q₀` | 4 |
| Cued recall | `I^n R Q` | n |

Sample `n_blocks` from a distribution (geometric or uniform) — trains all spacings simultaneously.

### 6.2 Status Table — 1 Block

| Trajectory | Tests | Status |
|---|---|---|
| `I Q` | baseline recall | ✓ Exp 1 |
| `I R Q` (k=1) | single-turn refine | ✓ Exp 3a |
| `I R Q` (k=1..5) | multi-turn refine, joint | ✓ Exp 3c |
| `I Q Q` | consistency | ✗ |
| `I R R Q` | two refine rounds | ✗ |

### 6.3 Status Table — 2 Blocks

| Trajectory | Tests | Status |
|---|---|---|
| `I I Q₀` | retention | ✓ Exp 2b, 3c |
| `I I Q₁` | encoding | ✓ Exp 2b |
| `I Q₀ I Q₁` | interleaved | ✓ int mode |
| `I I R₁ Q₀` | refine new, prove old retained (SRS) | ✗ priority |
| `I I R₀` | refine old after update | ✗ priority |
| `I Q₀ I Q₀` | retention under update | ✗ |

### 6.4 Status Table — 3+ Blocks

| Trajectory | Tests | Status |
|---|---|---|
| `I I I Q₀` | deep retention | ✓ Exp B |
| `I I I R₀` | deep chain + correct oldest | ✗ |
| `I Q₀ I Q₀ I Q₀` | SRS review at spacings 1,2,3 | ✗ |

### 6.5 Next High-Value Trajectories

| Priority | Trajectory | Why |
|---|---|---|
| ✓✓ | `I I R₁ Q₀` | refine new, prove old survived — direct SRS pressure |
| ✓✓ | `I I R₀` | `<h>` must retain x0 for self-correction — diff signal |
| ✓✓ | `I Q₀ I Q₀` | retention under update: x1 ingest must not overwrite x0 |
| ✓ | `I I I R₀` | maximum spacing + correction |
| ✓ | `I R₁ I Q₀` | refine, continue ingesting, test x0 retained |

---

## 7. Raw Data & Reports

### Active

| What | Path |
|------|------|
| Exp 3c config | `configs/refine_joint.py` |
| Exp 3c checkpoint | `logs/role_refine_joint/checkpoints/stage0_end.pt` |
| Exp 3c training log | `logs/role_refine_joint/train.jsonl` |
| All active configs | `configs/` |

### Historical Reports

| What | Path |
|------|------|
| **All experiment results** | `docs/EXP_RESULTS_SUMMARY.md` |
| Exp 1 dataset ablation | `docs/EXP1_DATASET_ABLATION.md` |
| Exp 2 multi-turn tracking | `docs/EXP2_MULTITURN_TRACKING.md` |
| Tag naming rationale | `docs/tag_naming.md` |
| KV dimension tables | `docs/kv_dims.md` |
| v1 results | `docs/v1/` |

### Plans & Theory

| What | Path |
|------|------|
| Exp 2 plan + refine plan | `docs/plan/PLAN_EXP2.md` |
| Self-correction theory | `docs/plan/PLAN_REFINED.md` §Direction B |
| Research direction reflection | `docs/plan/REFLECT.md` |
| Hidden state viz ideas | `docs/plan/VIZ_HIDDEN_STATES.md` |
| Archived plans | `docs/plan/archive/` |

### Code

| What | Path |
|------|------|
| Training loop | `kvmem/train.py` |
| Data / masks | `kvmem/data.py` |
| Model | `kvmem/model.py` |
| Sequence DSL | `kvmem/seq_dsl.py` |
| Curriculum DSL | `kvmem/curriculum_dsl.py` |
| v1 snapshot | `kvmem/old/v1/` |
