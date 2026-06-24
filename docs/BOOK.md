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

## 8. HashMemNet (v3) — Implicit Memory Architecture

All v2 chat tags removed. Memory is implicit: only `MEM_START/MEM_END/SLOT_*` tokens (IDs 256–267). Vocab size fixed at 268.

### Token Vocabulary

| Token | ID | Role |
|-------|----|------|
| `MEM_START` | 256 | Begin memory block |
| `MEM_END`   | 257 | End memory block |
| `SLOT_0–3`  | 258–261 | Slot positions (cycling for any `slot_len`) |
| `DEL_START` | 262 | Begin forget block (Stage 4+) |
| `DEL_END`   | 263 | End forget block |
| `DEL_SLOT_0–3` | 264–267 | Delete slot positions (cycling) |

`slot_len=0` → block is `[MEM_START, MEM_END]` (ponder/gate mode, 2 tokens only).

### Sequence Layout

BLEN = slot_len + 2 (one MEM block: MEM_START + slots + MEM_END).

```
k=0 (I Q):   [MEM_0][src][MEM_1][warmup][out]
k=1 (I R Q): [MEM_0][src][MEM_1][src][MEM_2][warmup][out]
k=n:         [MEM_0][src][MEM_1]...[src][MEM_{n+1}][warmup][out]
```

- **MEM_0** = prior state (zeros/cold start for first call; save KV → reload as MEM_0 for streaming)
- **MEM_1** = I output after first read of src (no h-loss; trained via NTP backprop only)
- **MEM_2..n+1** = R outputs, one per refinement turn (h-loss each vs teacher trajectory)
- **Recall** (warmup+out) sees **only MEM_{n+1}** — blocked from all src and MEM_0..n

Total L = (n+2)·BLEN + (n+1)·src_len + warmup_len + out_len

### Operation Semantics

| Op | Role | h-loss? | Teacher steps |
|----|------|---------|---------------|
| **I** | `[MEM_0][src][MEM_1]` — first compress | ✗ | 0 |
| **R** | `[src][MEM_t]` — refinement turn | ✓ MSE on slots | +1 per R |
| **Q** | `[warmup][out]` — recall (R with k=0) | ✗ NTP only | 0 |
| **D** | `[DEL_START][DEL_SLOT×n][DEL_END]` — forget | ✓ max-entropy | 1 |

Sequence programs:
```
I Q          k=0: [MEM_0][src][MEM_1][warmup][out]   — NTP only
I R Q        k=1: [MEM_0][src][MEM_1][src][MEM_2][warmup][out]   — 1 h-loss
I R R Q      k=2: +[src][MEM_3]  — 2 h-losses
I R R R Q    k=3:               — 3 h-losses
```

### Correct Refine Objective (and current code gap)

**Key insight**: supervision is always available during training (we have ground truth bytes). So every intermediate position can be teacher-forced — no AR decode needed during training. AR is only needed at inference for the **final Q**.

**The correct per-turn objective:**

At every R turn t, run a TF forward pass. Its output should argmax to the correct target bytes. If it fails, MEM_{t+1} must be corrected so the next turn's TF argmax succeeds. Only the final Q is AR at inference.

This requires per-turn outputs in the sequence:

```
[MEM_0][src][MEM_1][warmup][out_1_tf]        ← I: TF output, NTP supervised
[MEM_1][src][MEM_2][warmup][out_2_tf]        ← R1: TF output, NTP supervised
...
[MEM_k][warmup][out_final]                    ← Q: TF during training, AR at inference
```

h-loss target for MEM_{t+1}: run teacher on turn t's sub-sequence `[MEM_t][src][MEM_{t+1}][warmup][out_t]` until argmax is correct → h*. Push MEM_{t+1} → h*.

The cascade: if out_1 argmax fails → h-loss on MEM_2 corrects it → out_2 argmax should then pass → and so on until final Q.

**What the current code actually does:**

1. Builds an **IQ batch** `[MEM_0][src][MEM_1][warmup][out]`
2. Runs teacher once on that IQ batch — clones model, gradient-descends on NTP at `out` positions, stops when `argmax == tgt` for all tokens
3. Records MEM_1 slot activations at k evenly-spaced checkpoints → h_teachers
4. Builds IR batch `[MEM_0][src][MEM_1][src][MEM_2]...[warmup][out_final]` with same src
5. Computes h-loss: MSE(MEM_2 activations, h_teachers[0]), MSE(MEM_3, h_teachers[1]), ...
6. NTP loss only on the **single final `out`** position

**The gaps:**

| Issue | Description |
|-------|-------------|
| No per-turn outputs | IR sequence has no `out_t` at intermediate turns — no per-turn argmax check |
| Teacher context mismatch | Teacher runs on IQ `[MEM_0][src][MEM_1][…]`, but MEM_2 in IR sees `[MEM_0][src][MEM_1][src][MEM_2]` — different context |
| Single teacher run | Teacher is run once and targets reused across all turns. Correct version: run teacher separately for each turn in the correct turn context (sequential, expensive) |
| h-loss target semantics | h_teachers[t] = MEM_1 activations of a cloned model after t gradient-descent steps on model weights, not a direct "what activations would make argmax correct" |

**The teacher's argmax stopping criterion** (line 466–468 in train.py) is correct: it stops gradient descent when `(pred == tgt).all()`. So the oracle h* is "what a model-with-better-weights produces when it can correctly recall" — a reasonable proxy for the per-turn target, just computed in the wrong context (IQ instead of per-turn IR sub-sequence).

**Gap severity:** The teacher-context mismatch is the main issue. h_teachers gives targets for what MEM_1 should look like in an IQ sequence, but MEM_2 in IR is conditioned on MEM_1 before it. If the model learns to use MEM_1 as prior context in MEM_2's update, the IQ-derived target is off-distribution. May cause stagnation at higher k (k≥2).

**Fix:** Add per-turn outputs to the IR sequence and run the teacher on each turn's sub-sequence using the actual MEM_t from the current forward pass as prior state. Expensive (k teacher runs per training step) but matches the correct objective exactly.

**Simpler training: no h-targets needed**

If monotonic NLL decrease per turn is enforced, the teacher trajectory is unnecessary. The only signals needed are:

1. Per-turn NTP loss on each turn's output (TF during training)
2. Monotonicity penalty: heavy loss if NLL does not strictly decrease turn-over-turn

```python
nll = [NLL_0, NLL_1, ..., NLL_k, NLL_final]
mono_penalty = sum(F.relu(nll[t+1] - nll[t] + margin) for t in range(len(nll)-1))
loss = nll_final + λ_mono * mono_penalty
```

No model cloning, no h-loss MSE targets, no teacher context mismatch. The model discovers what MEM state to produce — the only constraint is that NLL must decrease. `λ_mono` must be large (heavily penalized) to force strict monotonicity rather than allowing plateaus.

This removes `compute_teacher_trajectory` entirely and replaces it with a single forward pass through the full per-turn sequence. Simpler code, cleaner gradient, no target mismatch.

**Requirement:** sequence must have per-turn outputs `[out_t]` at each R turn to compute per-turn NLL (not just the final Q). The current HMN sequence layout lacks these.

**Test-time implication — supervision is available at inference too:**

At inference, the src bytes are known (we just read them). So the argmax-correctness check works at test time with zero extra cost:

```
1. Store: [MEM_0][src][MEM_1]
2. Verify: forward([MEM_1][src_tf]) → check argmax == src bytes
3. If < 100% match: do another R turn → [MEM_1][src][MEM_2]
4. Verify again. Repeat until 100% or max_turns.
5. Recall: AR decode [MEM_k][warmup][out]
```

This gives **adaptive compute at inference** — easy inputs converge in 1–2 turns, hard inputs get more. The training monotonicity requirement (each R turn must improve argmax match, never regress) is what makes this safe to run until convergence: you know stopping when argmax is 100% is always valid, and you know more turns never hurt.

Training must enforce: match(turn t+1) ≥ match(turn t) strictly. If this monotonicity holds, the test-time verify-then-refine loop is a greedy decoder with a guaranteed stopping condition.

### Experiment Ladder

| Stage | Config | Sequence | Signal |
|-------|--------|----------|--------|
| 1 | `hmn_32` s1 | I Q full (warmup=0, out=32) | overfit→100%, val_hmn_bpb↓ |
| 2 | `hmn_32` s2 | I Q windowed (wm=8, out=24) | hmn_k0 match% |
| 3 | `hmn_32` s3 | I R Q k∈{0..4} windowed | Δ=tN−t1>0 per turn |
| 4 | `hmn_32` s4 | joint mix (IQ 40% + IR 60%) | no regression on k0 |

### Residual Corrector Idea (if Stage 3 stalls)

If the model struggles to express correction through slot activations alone, a two-pass residual corrector could help:

**Architecture variant:**
1. **Pass 1 (residual pass)**: forward with the same `[src][MEM][…]` sequence but a *residual corrector* block `[MEM_START][CORR_SLOT×n][MEM_END]` appended. This pass generates correction KV activations targeting the error signal.
2. **Pass 2 (memory pass)**: regular forward, but slot activations are `h_prev + h_corr` (residual add). Corrector output is added to the current memory state before the second decode.

This separates *"what is wrong"* from *"what to remember"* into two distinct representations. Each pass is a full single forward (no parallel shortcut).

**Format**: same cycling `SLOT_0–3` IDs, just a new block type. Model learns correction-block semantics from the h-loss target (teacher trajectory delta: `h_target - h_current`).

**Trade-off vs direct (current):**
- Current: single pass, slot activations must encode both memory and error in one representation
- Residual: two passes per step, stronger separation; correction block targets `Δh`, memory block targets `h`
- Try direct first; switch to residual if correction diverges or Δ(tN−t1) < 0 after step 40k

### Soft Residual Corrector (expensive, stronger error signal)

New token type: `[RES_START][RES_SLOT×n][RES_END]` — the residual correction block, same cycling IDs as MEM but separate token type so the model distinguishes "store" from "correct".

**Training loop per step** (converge inner loop before outer gradient step):

```
iter 0:
  Pass A — teacher-forced NLL:
    forward([MEM_1, tf_src, MEM_2])  →  logits, NLL_loss
    softmax(logits) → p  →  embed_mix = Σ_v p[v] · E[v]   (soft, no Gumbel, fully differentiable)

  Pass B — residual correction:
    forward([MEM_1, embed_mix, RES_1])  →  h_res1
    MSE(h_res1, h_teacher_target)  →  res_loss
    KV(MEM_2) += KV(RES_1)   (residual add in KV space; MEM_2 is now corrected)

iter 1:
  Pass A — teacher-forced NLL with updated MEM_2:
    forward([MEM_2_corrected, tf_src, MEM_3])  →  NLL_loss
    softmax(logits) → embed_mix

  Pass B — residual correction:
    forward([MEM_2_corrected, embed_mix, RES_2])  →  h_res2
    MSE(h_res2, h_teacher_target_2)
    KV(MEM_3) += KV(RES_2)

... repeat until NLL_loss converges or max_iter reached
```

**Why soft embedding (not Gumbel/discrete):**
- Softmax weighted sum `Σ p[v]·E[v]` is differentiable through the logits — gradients flow from the residual MSE loss back through Pass B embeddings, through the softmax, back through the NLL logits, into MEM_1 activations
- Gumbel-softmax would discretize and require straight-through estimator, weaker gradient signal
- This gives the residual block a direct gradient path to the memory slot that produced the wrong token distribution

**Why KV residual add (not activation add):**
- KV cache is the persistent memory structure — adding RES KV to MEM KV means downstream positions (MEM_3 forward) can attend to both the raw memory and the correction, weighted by attention
- Activation add would collapse both into a single vector before attention, losing the ability to query correction vs memory separately

**Cost vs benefit:**
- Cost: 2 forward passes per inner iteration × max_iter iterations (vs 1 forward in direct h-loss)
- Benefit: gradients flow through the full correction chain; error and memory are separate representations; each iteration refines the correction rather than forcing one-shot convergence
- Fall back to this if direct h-loss stalls (Δ match per turn < 1% after step 60k)

**Gradient path analysis:**

The connected graph from res_loss back to MEM_1 is: `res_loss → Pass B → embed_mix → softmax → logits_A → Pass A → MEM_1`. Standard `backward()` works — no retain_graph tricks because embed_mix is a leaf output of Pass A that flows into Pass B.

**Problems:**

1. **logits_A has two conflicting objectives.** NLL_loss wants logits_A[correct_token] high. res_loss (via embed_mix) wants logits_A[v] high for whichever token v has embedding E[v] that pushes h_res1 toward h_teacher. These agree only if the correct next token's embedding also happens to be the best correction signal — not guaranteed. The gradient at logits_A is a sum of two unrelated tasks.

2. **embed_mix is off-distribution for Pass B.** Pass B's transformer was trained on discrete token embeddings, not arbitrary weighted sums from a softmax. The attention patterns in Pass B for a continuous mixture input are unpredictable unless Pass B is trained from scratch with soft inputs.

3. **KV add requires non-standard forward.** `KV(MEM_2) += KV(RES_1)` has no standard transformer equivalent — MEM_2 token IDs are unchanged between iterations, so the transformer recomputes identical KV values. Injecting the RES KV requires hooking the K/V projection at MEM_2 positions per-layer, which breaks standard autograd.

**Cleaner alternative — context concatenation instead of KV add:**

Drop KV manipulation. RES_1 is just additional context tokens, carried forward:

```
iter 0: forward([MEM_1, tf_src, MEM_2]) → NLL, embed_mix
        forward([MEM_1, embed_mix, RES_1]) → MSE

iter 1: forward([MEM_1, RES_1, tf_src, MEM_3]) → NLL_2, embed_mix_2
        forward([MEM_1, RES_1, embed_mix_2, RES_2]) → MSE_2
```

MEM_3 attends to both MEM_1 and RES_1 via standard causal attention — no hooks, no KV surgery, clean gradient everywhere. The unresolved issue (conflicting objectives on logits_A) remains regardless of KV vs context. The only fix for that is to detach logits_A from res_loss entirely, making embed_mix a stop-gradient from the NLL path — then res_loss only trains Pass B, and Pass A is trained only by NLL. Weaker but cleaner signal separation.

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
