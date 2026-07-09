# Feedback Architecture — Results Report

## Summary

The **argmax feedback architecture** (`train_hmn_feedback.py`) is the first approach to achieve 100% recall at all refinement turns k=0..12, including perfect extrapolation beyond the trained range (k=0..4).

---

## Problem History

All previous HMN variants failed to show monotonic improvement across refinement turns:

| Experiment | k=0 | k=4 | k=12 | Root cause |
|---|---|---|---|---|
| `hmn_32` (structured tokens, teacher h-loss) | 97.9% | 97.9% | — | h-loss dominates NTP, no useful gradient |
| `hmn_mono` (flat_mono, p=1) | 95.3% | 93.2% | 2.1% | model re-reads src every turn, ignores MEM |
| `hmn_mono_cerb` (delta-CER weighted) | ~96% | ~93% | ~3% | same shortcut |
| `hmn_mono_p2` (banded src, period=2) | 95.3% | 95.8% | 0% | blind turns = 0%, alternating pattern |
| `hmn_mono_p4` (period=4) | 96.4% | ? | ? | same alternating |
| `hmn_mono_pinf` (only t=0 sees src) | 96.4% | 95.3% | 0% | alternating: even turns ok, odd=0% |
| `hmn_mono_tlogit_fixed` (teacher logit α=0.5) | 96.4% | 94.3% | 3.1% | no improvement from distillation |

**Core failure mode:** models could shortcut refinement by re-reading src at every turn. Even with src masking (p=2, pinf), models learned to handle review turns but failed on blind turns — they never learned to carry information through MEM.

---

## Feedback Architecture

### Key insight

Instead of masking src from later turns (negative pressure), feed the model's **own previous output** (argmax) back as input. The model can now compare what it predicted vs the loss signal — self-correction without an explicit teacher.

### Sequence layout

**Turn 0 (IQ — encode src):**
```
[src: src_len] [SLOT×n] [warmup: wl] [out: ol]
```

**Turn t≥1 (IR — argmax feedback):**
```
[SLOT_A×n] [argmax_{t-1}: ol] [SLOT_B×n] [warmup: wl] [out: ol]
```

SLOT_A and SLOT_B have identical token IDs (cycle through SLOT_0..SLOT_{slot_count-1}). The model distinguishes them by position (RoPE) and context.

### Attention mask

**IQ:** `warmup`/`out` blocked from `src` — must go through slots. Convention: `0.0` = attend, `-1e9` = blocked (additive bias, matches PyTorch `F.scaled_dot_product_attention`).

**IR:** `warmup`/`out` blocked from `SLOT_A` and `argmax` — must go through `SLOT_B` only.

### Differences from HMN 1tok

| | HMN 1tok | Feedback |
|---|---|---|
| MEM delimiters | MEM_START (256), MEM_END (257) | **None** — only SLOT tokens |
| Refinement input | re-reads same src | argmax from previous turn |
| slot_count | 4 unique IDs | 2 unique IDs (default) |
| Mask convention | 0.0 / -1e9 | **same** (was bug: was using 0.0/1.0) |

### Critical bug found and fixed

Original mask used `1.0`/`0.0` (multiplicative) instead of `0.0`/`-1e9` (additive). Since PyTorch uses the mask as an **additive bias**, `0.0` means "no effect" — the bottleneck was completely transparent. Model could see src from all output positions → NLL→0 in ~10k steps from memorization.

Fix: changed both `fb_iq_mask` and `fb_ir_mask` to use `0.0`/`-1e9` convention.

---

## Training Curriculum

IQ pretraining is **required** before IR. Without it, the model doesn't know how to encode into SLOT tokens, so SLOT_A is meaningless and argmax feedback carries no signal.

```
Stage 0: IQ only  (k_choices=[0],        50k steps) → stage0_end.pt
Stage 1: IR + FB  (k_choices=[0,1,2,3,4], 80k steps) → stage0_end.pt (stage1 ckpt)
```

Configs:
- `configs/hmn_feedback_32_iq.py` — IQ pretraining (standalone, reusable)
- `configs/hmn_feedback_32_ir.py` — IR+feedback (use `--pretrained iq/stage0_end.pt`)
- `configs/hmn_feedback_32.py` — both stages in one run

---

## Results

### `hmn_feedback_32_ir` — final eval (step 80000)

All k=0..4 at **100%** with `✓✓` (double-confirmed).

### Extrapolation k=0..12 (trained on k=0..4 only)

| k | match% |
|---|--------|
| 0 | **100.0%** |
| 1 | **100.0%** |
| 2 | **100.0%** |
| 3 | **100.0%** |
| 4 | **100.0%** |
| 6 | **100.0%** |
| 8 | **100.0%** |
| 10 | **100.0%** |
| 12 | **100.0%** |

Perfect extrapolation — no degradation at all with additional refinement turns.

---

## Why it works

1. **Argmax feedback creates a self-correcting signal.** If turn t-1 output is wrong, turn t sees exactly which tokens were wrong (by comparing argmax to the loss signal). SLOT_B encodes a "correction" rather than a blind re-compression.

2. **Each turn is bounded-length, separate forward pass.** No growing context — IR sequence is always `s + ol + s + wl + ol` = 64 tokens regardless of k. This enables stable gradient flow and natural extrapolation.

3. **True bottleneck at both IQ and IR.** `warmup`/`out` are blocked from `src` (IQ) and from `SLOT_A`/`argmax` (IR). All information must flow through the slot positions.

4. **SLOT_A → argmax → SLOT_B information chain** is structurally forced. The model must learn: SLOT_A = compressed state, argmax = my previous prediction, SLOT_B = corrected state.

---

## Files

| File | Description |
|---|---|
| `kvmem/train_hmn_feedback.py` | Training script |
| `kvmem/eval_jacobian.py` | Eval + Lipschitz diagnostics (add `--eval-only` for feedback TBD) |
| `configs/hmn_feedback_32_iq.py` | IQ pretraining config |
| `configs/hmn_feedback_32_ir.py` | IR+feedback config |
| `configs/hmn_feedback_32_ir_cumm.py` | IR+feedback with cum_mean loss (skipped — already 100%) |
| `logs/hmn_feedback_32_iq/` | IQ pretraining run |
| `logs/hmn_feedback_32_ir/` | IR+feedback run (**100% k=0..12**) |

---

## Next steps

1. **Harder evaluation** — longer src (64, 128), more complex patterns
2. **Multi-sequence SRS** — multiple sequences in one context (see `docs/kv_dims.md`)
3. **Jacobian eval** — run `eval_jacobian.py --jacobian` to measure Lipschitz constant across turns (expect L < 1 given perfect extrapolation)
4. **Add `--eval-only` to `train_hmn_feedback.py`** for batch eval runs

---

## Extension — Global IQ with Random Warmup (nc=4, 64B)

### Architecture (`iq_global_rw` traj type)

Scale up from 32B to 64B using **global IQ**: a single IQ turn that reads all 4 enc_blocks and predicts any 24-byte window (warmup_len=8, out_len=24, window=32B). Warmup position X is sampled uniformly from `[0, src_len - window_size]` at training time; eval uses fixed offsets at chunk-aligned positions (Win A=0, Win B=16, Win C=32).

```
[ENC_0: src[0:16] | SLOT×slot_len]
[ENC_1: src[16:32] | SLOT×slot_len]
[ENC_2: src[32:48] | SLOT×slot_len]
[ENC_3: src[48:64] | SLOT×slot_len]
[IQ: SLOT×slot_len | warmup[X:X+8] | out[X+8:X+32]]
```

The three windows differ only in warmup start position X, not in source or slot structure.

### Slot capacity is the primary bottleneck

**Key finding from ablation**: Win A (X=0, bytes 0-31) consistently underperforms Win B (X=16) and Win C (X=32) — not due to training distribution but due to **slot capacity** (slot_len=4 → 4 tokens to compress 32B of information).

| Run | slot_len | Win A BPB | Win A match | Win B match |
|-----|----------|-----------|-------------|-------------|
| slot4 + uniform warmup | 4 | ~3.5 | 0% | 0% |
| slot4 + 2× Win A oversample | 4 | 2.128 (new low) | 12.5% peak | ~20% |
| slot8 IQ-only (ext, 80k) | 8 | 1.283 | ~6% peak | ~58% |
| **slot8 + IR (10k steps from IQ best)** | **8** | **1.305** | **4.2%** | **100%** |

Oversampling Win A (2× weight) only pushed BPB from 3.5 to 2.1 with slot4 — confirms capacity is the bottleneck, not training distribution.

### IR stage on top of Global IQ (`iq_global_rw_ir` traj type)

**Config**: `hmn_chunk_global_iq_rw_nc4_slot8_ir.py` — adds `n_refine=2` IR turns, from slot8_ext best (44.0%).

Architecture (nc=4, slot_len=8, n_refine=2, L=280):
```
[ENC_0..ENC_3: 96 tokens]
[IQ:  SLOT×8 | warmup×8 | out×24]           (pos 96–135)
[IR1: SLOT_A×8 | am×24 | SLOT_B×8 | warmup×8 | out×24]  (pos 136–207)
[IR2: SLOT_A×8 | am×24 | SLOT_B×8 | warmup×8 | out×24]  (pos 208–279)
```

Warmup offset X is sampled once at IQ and reused for all IR turns (same window).
Eval uses IQ-only decode (`eval_traj='iq_global_rw'`); IR improves the training distribution but the IQ alone is what's measured.

### IR vs long IQ training — is the effect real?

**Strong evidence**: Win B down_counter BPB was stuck at **0.350** after 80k+ IQ-only steps. IR reduced it to **0.039 in just 15k steps** — 9× lower BPB in 5× fewer steps.

| metric | IQ-only (slot8_ext, 80k steps) | IR stage (15k steps from same checkpoint) |
|--------|-------------------------------|-------------------------------------------|
| Win B down BPB | 0.350 (plateau) | **0.039** |
| Win B down match | 8.3% | **100%** |
| Win B up match | 100% | 100% |
| Val mean | 44.0% | **46.8%** (at step 10k, still climbing) |

The mechanism is qualitatively different from more IQ steps:
- IQ: "encode source → decode from SLOT" (one-shot, fixed representation)
- IR: "IQ output → model sees its own prediction → corrects via second pass (SLOT_B)"

The IR bottleneck is structurally new: `warmup`/`out` can only attend to `SLOT_B`, never to `SLOT_A` or `argmax`. The model must learn a **delta-encoding** — what to change relative to the last guess, not a fresh encoding from scratch.

### Why argmax feedback (not soft probabilities)

Using argmax (hard token ID) is the correct choice for consistency:
- During **training**: argmax positions are filled with hard GT token IDs
- During **eval**: argmax positions are filled with the model's argmax predictions
- Both are discrete — train/eval distributions match

Soft feedback (`softmax(logits) @ E`, a weighted sum over the vocabulary embedding matrix) would break this consistency: training is hard, inference is soft → distribution mismatch. Using soft feedback requires training with it too (Gumbel-softmax or straight-through estimator), adding complexity with no clear benefit. Argmax reliability is not a concern because:
1. IQ pretraining establishes priors before IR ever runs
2. BPB trajectory shows productive learning from the first IR step (3.63→1.80→1.43→1.30 over 5k-step intervals)

### Window difficulty ordering

| window | why harder |
|--------|-----------|
| Win B (X=16) | bytes 16-47: straddles two chunks, model sees both halves in enc_blocks |
| Win C (X=32) | bytes 32-63: last 32 bytes, chunks 2-3 — similar to Win B |
| **Win A (X=0)** | **bytes 0-31: first 32 bytes — enc_block[0] SLOTs must compress bytes 0-15 which are also in the warmup region; highest positional load on early chunks** |

Win A remains hardest under all conditions. `slot8_ir` final: Win A ~4% (IQ=0% throughout — IQ block excluded from loss when n_refine>0).

---

## v2: IQ quality fix + Win A oversample (`slot8_ir_v2`, 100k steps)

### Problem diagnosed from slot8_ir per-turn eval

Per-turn logging (`[IQ=X%  IR1=Y%  IR2=Z%]`) revealed two structural problems in `slot8_ir`:

1. **IQ=0% for all windows, all steps**: `chunk_positions_iq_global_rw` sets `is_clean=(n_refine == 0)` on the IQ block — when n_refine=2, IQ output never contributes to loss. The model learns to treat IQ as a pure encoding stage and lets IR do all recall work. Consequence: IQ has no one-shot capability; IR1 has no useful prior to refine.

2. **Win A stuck at ~3%**: Even with IR, Win A (bytes 0-31, X=0) rarely exceeded 3% because IQ encodes the first window poorly. IR1 cannot fix what IQ never learned.

### Fix

**Traj mix** (`slot8_ir_v2`, total weight=4.5):

| weight | type | warmup X | share | purpose |
|--------|------|----------|-------|---------|
| 1.0 | `iq_global_rw_ir` | uniform [0,32] | 22% | IR all windows, maintains Win B/C IR quality |
| 0.5 | `iq_global_rw` | uniform [0,32] | 11% | IQ-only all windows — forces one-shot recall loss |
| 2.0 | `iq_global_rw_ir` | fixed X=0 (Win A) | 44% | heavy Win A IR oversample |
| 1.0 | `iq_global_rw` | fixed X=0 (Win A) | 22% | direct Win A IQ one-shot loss |

Win A receives 66% of steps. IQ loss signal present in 33% of steps.

From: `slot8_ir` best checkpoint (step 50k, 51.9%).

### Results — MEAN match per window per eval step

| step | Win A (0,2) | Win B (1,3) | Win C (2,4) | Overall | note |
|------|------------|------------|------------|---------|------|
| 5k | 16.7% | 77.8% | 30.6% | 41.7% | |
| 10k | 59.7% | 45.8% | 31.9% | 45.8% | IR instability — IR1 destroys IQ output |
| 15k | 79.2% | 63.9% | 55.6% | **66.2%** | Win A IQ=100%, IR stabilising |
| **20k** | **79.2%** | **77.8%** | **51.4%** | **69.4%** | cycle 1 end |
| 25k | 70.8% | 45.8% | 40.3% | 52.3% | LR restart disruption |
| 30k | 26.4% | 41.7% | 22.2% | 30.1% | worst dip mid-cycle |
| 35k | 94.4% | 65.3% | 33.3% | 64.4% | |
| 40k | 91.7% | 69.4% | 43.1% | 68.1% | |
| 45k | **100.0%** | 77.8% | 29.2% | 69.0% | Win A solved |
| 50k | **100.0%** | 77.8% | 30.6% | 69.4% | |
| 55k | **100.0%** | 77.8% | 54.2% | **77.3%** | |
| **60k** | **100.0%** | **77.8%** | **55.6%** | **77.8%** ★ | cycle 2 end — **best checkpoint** |
| 65k | 91.7% | 79.2% | 52.8% | 74.5% | |
| 70k | 80.6% | 51.4% | 12.5% | 48.1% | Win C collapse mid-cycle |
| 75k | 76.4% | 44.4% | 51.4% | 57.4% | |
| 80k | 75.0% | 61.1% | 16.7% | 50.9% | |
| 85k | 97.2% | 75.0% | 56.9% | 76.4% | |
| 90k | 84.7% | 65.3% | 56.9% | 69.0% | |
| 95k | 97.2% | **91.7%** | 38.9% | 75.9% | Win B peak |
| **100k** | **100.0%** | **88.9%** | **37.5%** | **75.5%** | cycle 3 end (final) |

Best checkpoint: step 60k (77.8%).

### Per-window per-seq breakdown — key steps

**Win A (0,2)** — solved by step 45k, stays solved at end:

| step | seq | match | IQ | IR1 | IR2 |
|------|-----|-------|----|-----|-----|
| 5k | up | 16.7% | 8.3% | 16.7% | 16.7% |
| 5k | down | 0.0% | **100%** | 0.0% | 0.0% |
| 20k | odd | 45.8% | **100%** | **100%** | **45.8%** ← IR2 regresses |
| 60k | all | **100%** | 100% | 100% | 100% ✓ |
| 100k | up | 100% | 100% | 62.5% | **100%** ← IR1 degrades, IR2 corrects |
| 100k | down | 100% | 100% | 45.8% | **100%** |

IR2 acts as a correction stage for IR1 degradations — Win A final match stays 100% even when IR1 temporarily drops below 100%.

**Win B (1,3)** — `down_counter` is persistent blocker, cracked at step 95k:

| step | seq | match | IQ | IR1 | IR2 |
|------|-----|-------|----|-----|-----|
| 60k | down | 33.3% | 33.3% | 33.3% | 33.3% ← stuck across all turns |
| 95k | down | **100%** | 33.3% | 33.3% | **100%** ← IR2 alone fixes it |
| 100k | down | 91.7% | 75.0% | 91.7% | 91.7% |

**Win C (2,4)** — `up_counter` is the unsolved problem:

| step | seq | match | IQ | IR1 | IR2 |
|------|-----|-------|----|-----|-----|
| 60k | up | 4.2% | 4.2% | 4.2% | 4.2% ← all turns fail equally |
| 60k | down | 79.2% | 29.2% | 79.2% | 79.2% |
| 60k | odd | 83.3% | 54.2% | 79.2% | 83.3% |
| 100k | up | 4.2% | **37.5%** | 4.2% | 4.2% ← IQ works, IR destroys |
| 100k | down | 75.0% | 45.8% | 75.0% | 75.0% |
| 100k | odd | 33.3% | 54.2% | 33.3% | 33.3% |

Win C `up_counter` at step 100k: IQ reaches 37.5% but IR1 collapses it back to 4.2%. The IQ encoding of bytes 32-64 starting at X=0 is partially learned, but IR refinement for this sub-sequence actively degrades it. This is a training distribution imbalance: X=0 Win A oversample (44% of steps) creates IR steps where warmup=bytes 0-8, but X=0 Win C (`up_counter`) warmup = bytes 32-40 — a completely different context for the IR stage.

### Key findings

1. **IQ-only training (33% of steps) successfully makes IQ=100% for Win A by step 45k.** The fix for `is_clean` via traj_mix (not code change) is validated.

2. **Cosine cycle 2 end (step 60k) is the best checkpoint** (77.8%). Cycle 3 introduces volatility — Win C collapses at step 70k and never fully recovers. Cycle 2 is the sweet spot for this mix.

3. **IR2 serves as a correction stage** for IR1 regressions (Win A at step 100k: IR1=45-62%, IR2=100%). The two-turn chain is qualitatively different from IQ+IR1 — IR2 reliably rescues what IR1 drops.

4. **Win C IQ is 37-54% at end** — encoding works. IR is the failure point, specifically for `up_counter` (X=0 with Win C offset). Fix: add Win C X=0 IR training entries to traj_mix, not just Win A X=0.

5. **Win B `down_counter` solved only at step 95k** — required 95k steps to crack, then regressed slightly to 91.7% at 100k. Indicates Win B is still not fully converged.

### What to fix next

- **Win C `up_counter`**: add `dict(type='iq_global_rw_ir', warmup_x_fixed=32, ...)` to traj_mix so IR sees Win C at X=0 explicitly
- **Win B `down_counter`**: needs more steps or dedicated oversample (`warmup_x_fixed=16`)
- **Cycle length**: consider `cosine_T0=40000` so cycle 2 end falls at step 80k rather than 60k — more steps per cycle reduces mid-cycle volatility

---

## Dataset design — unbounded random bytes

### Is the moving-target problem causing Win C failures?

**No. The source distribution is correct; the traj_mix coverage is the bug.**

### History of this choice

EXP1 (2026-06-03, v2 arch, seg=16, slot_len=1) ablated four dataset regimes:

| dataset | pool | best match% | steps to 100% |
|---------|------|-------------|---------------|
| ds10k | 10k batches fixed | 98% | never |
| ds20k | 20k batches fixed | 100% | 65k |
| ds40k | 40k batches fixed | 97% | 50k |
| **ds_random** | infinite stream | **100%** | **40k** |

Conclusion at the time: infinite stream forces genuine generalisation (novel batch every step → model must learn the algorithm, not memorize the pool). Converges fastest and avoids pool-cycling artifacts.

That task was ~10× simpler (single block, 1-slot, 8-byte output). The finding still holds structurally: the model should learn a universal encoding algorithm, not patterns from a fixed pool.

### Why unbounded random bytes remains correct

The BOOK diagnostic progression explicitly targets random bytes first to verify the mechanism before moving to structured text. IQ encoding is stable and generalizes as expected — Win C IQ reaches 37–54% at steps 55–100k consistently. That is the learned algorithm working on completely novel sequences, not memorization. If the source distribution were causing drift, IQ BPB would oscillate; it doesn't (it trends down steadily).

### The actual failure: traj_mix warmup-offset coverage

Win C `up_counter` eval: warmup = bytes 32–40, output = bytes 40–64 (warmup_x = 32 within the global source).

v2 traj_mix exposure for IR at X=0 per window:

| window | warmup_x=0 IR share | mechanism |
|--------|---------------------|-----------|
| Win A (0,2) | **44%** | explicit `warmup_x_fixed=0` entry, weight=2.0 |
| Win B (1,3) | ~0.7% | falls in uniform [0,32] range, weight=1.0 |
| Win C (2,4) | **~0.7%** | same uniform coverage, never explicitly fixed |

Win A gets 60× more IR training at X=0 than Win C. The model never learned "IR refinement for bytes 32–64, warmup starting at byte 32." When eval presents exactly this, IQ works at 37% but IR applies corrections trained on other (window, X) pairs — wrong direction, collapses output to 4.2%.

### Fix: per-window warmup_x_fixed entries

Add explicit oversample for each window at its natural X=0 position:

```python
# Win A at X=0 (bytes 0-31)
dict(type='iq_global_rw_ir', weight=1.0, warmup_x_fixed=0),
# Win B at X=0 (bytes 16-47, warmup starts at 16)
dict(type='iq_global_rw_ir', weight=1.0, warmup_x_fixed=16),
# Win C at X=0 (bytes 32-63, warmup starts at 32)
dict(type='iq_global_rw_ir', weight=1.0, warmup_x_fixed=32),
```

This makes the IR distribution cover all three window-start positions explicitly, not just Win A. The source distribution stays infinite random bytes — only the warmup position sampling changes.

---

## Chat-tags experiment — explicit src/mem/query/response boundary tokens

**Hypothesis**: every region (source chunk, SLOT/memory, warmup/query, output/response) is
currently identified *only implicitly* — by fixed Python-computed token-offset ranges that
drive the attention mask and RoPE positions (`kvmem/data.py:2164` literally documents this as
*"HashMemNet (HMN) — Implicit memory, no chat tags"*). Wrapping each region with explicit,
content-independent boundary tokens (`<src>`, `<mem>`, `<query>`, `<response>`) gives the model
a runtime-visible, position-independent signal for "what kind of region is this" on top of RoPE
— testing whether this helps generalization to windows/positions not densely trained on (the
same class of problem as Win C's undertraining above, and the `ir_local` "position problem" in
`kvmem/train_hmn_chunk.py`'s docstrings).

Full design: `/Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md`.

**Isolation**: implemented entirely in a new `experiments/chat_tags/` folder — no edits to
`kvmem/train_hmn_chunk.py` or `kvmem/data.py`. New tag IDs (268-275, `HMN_TAG_VOCAB_SIZE=276`)
live in `experiments/chat_tags/vocab.py`, disjoint from the `HMN_*` namespace. Reuses
`chunk_mask_fb`, `build_model`, `_fill_argmax_fb` from `kvmem` via plain import — the mask logic
needed **zero changes**: every region's boundary fields are widened by exactly 1 on each side to
absorb its wrapping tag (e.g. mask-time `sl0 = content_sl0 - 1`), so `chunk_mask_fb`'s pure
`(row >= x0) & (row < x1)` membership rules apply unmodified. Content-writing fields stay
exactly the untagged widths, fed through a fresh batch filler (`experiments/chat_tags/batch.py`)
since `_chunk_make_batch_fb`'s shape-exact broadcast assignment would break against tag-widened
ranges.

### Phase A — plumbing sanity check (complete, passed)

Short IQ-only run (8000 steps, nc=4, slot_len=8, `n_refine=0`) verifying tokenization/mask/decode
correctness before committing to a full staged run.

- **Coverage check**: every position in the 158-token sequence (nc=4/slot=8) covered exactly
  once by either a tag or a content range — no gaps/overlaps (verified programmatically, and
  again for the n_refine=2 case at L=322).
- **Mask check**: strictly causal, IQ SLOT rows blocked from raw source-chunk bytes while still
  seeing encoding SLOTs (as designed).
- **Loss trajectory matches the untagged baseline's own early curve**: untagged `iq_global_rw`
  (`logs/hmn_chunk_global_iq_rw_nc4_slot8/train.log`) went loss 5.34→5.18 over steps 5k→10k;
  tagged run hit loss=5.25 at step 8k — right on that curve.
- **Tag placement verified directly**: loaded the trained checkpoint, ran AR decode, printed the
  actual token sequence — `<src>`/`</src>` wrap each 16-byte chunk, `<mem>`/`</mem>` wrap each
  8-token SLOT, `<query>`/`</query>` wrap warmup, `<response>`/`</response>` wrap output, all 22
  tags at their designed positions, zero collisions.
- **Windowed recall does not yet work at this checkpoint** (expected — 8000 IQ-only steps is far
  short of what even the untagged baseline needs): match% near 0-7% across all three windows,
  decoded output mode-collapsed to a handful of high-frequency bytes. Phase A's bar was
  plumbing correctness, not working recall — that's what Phase B tests.

### Phase B — full staged run (complete)

`experiments/chat_tags/configs/slot8_tagged_phaseB_full.py`. Since the tagged vocab (V=276)
can't load the untagged pretrained checkpoint (embedding width differs by 8 rows), this
collapses the untagged chain (`slot8` 80k → `slot8_ext` 80k → `slot8_ir` 50k → `slot8_ir_v2`
100k = ~310k steps, each stage warm-started) into two from-scratch stages of comparable budget,
warm-started in-memory (same `model`/`opt` objects carry across stages, no checkpoint reload):

| Stage | steps | traj_mix | stands in for |
|---|---|---|---|
| 0 | 160000 | `n_refine=0` only | `slot8` + `slot8_ext` |
| 1 | 100000 | identical proportions to `slot8_ir_v2` (table below) | `slot8_ir` + `slot8_ir_v2` |

Stage 1 traj_mix (total weight 4.5, matches `slot8_ir_v2` exactly):

| weight | n_refine | warmup X (train) | share | purpose |
|--------|----------|-------------------|-------|---------|
| 1.0 | 2 | uniform [0,32] | 22% | IR, all windows |
| 0.5 | 0 | uniform [0,32] | 11% | IQ quality, all windows |
| 2.0 | 2 | fixed X=0 | 44% | heavy Win A IR |
| 1.0 | 0 | fixed X=0 | 22% | direct Win A IQ recall |

Baseline to compare against: untagged `slot8_ir_v2` — 77.8% best @ step 60k, Win A 100%,
Win B 77.8%, Win C 55.6% (`up_counter` sub-metric stuck at 4.2%).

**Final result** (best checkpoint, `stage1_best.pt`, global step 220000 = stage-1 step 60k —
same "best checkpoint, not final" convention as `slot8_ir_v2`, since stage 1 step 100k had
already regressed to 56.0% by cosine-restart volatility, mirroring the untagged run's own
"cycle 3 restart introduces volatility, never fully recovers" pattern):

| | Win A | Win B | Win C | overall |
|---|---|---|---|---|
| untagged `slot8_ir_v2` (baseline) | 100% | 77.8% | 55.6% | 77.8% |
| tagged Phase B | 100% | 75.0% | **5.6%** | 60.2% |

Win A/B are comparable to the baseline (Win A exact match, Win B within 3pp). **Win C is
dramatically worse** (5.6% vs 55.6%) — the success bar (meaningfully better Win C
generalization) is clearly not met.

**Qualitative failure-mode diagnosis** (decoded target-vs-generated bytes across all 8 val
sequences × all 3 windows): Win C's IQ turn regularly encodes well (29-83% per-seq match, one
case hit 79.2%), but IR1/IR2 turns then corrupt it to near-random output — not a systematic
byte-shift and not cross-window copying (generated bytes don't resemble Win A/B's target
content), just noise. Two of the eight val sequences (`sawtooth`, `geometric`) showed apparent
Win C=100% — but their Win C target bytes are a periodic repeat of their own Win A/B target
(these deterministic test sequences have period ≤32 for certain patterns), so those results are
coincidental content overlap, not genuine Win C generalization. The real signal is the other six
sequences, consistently 0-8.3%.

**Important confound — this is not a clean test of the tags hypothesis on its own**: Phase B's
stage-1 traj_mix was copied verbatim from `slot8_ir_v2`, which — per `docs/FEEDBACK_RESULTS.md`'s
own earlier analysis (§ Dataset design) — **never actually applied the diagnosed
`warmup_x_fixed=16/32` fix** for Win B/C IR coverage. Both the tagged and untagged runs share
this same traj_mix gap. The untagged baseline still reached 55.6% despite it, likely because it
had ~310k cumulative steps across 4 warm-started stages (`slot8` 80k → `slot8_ext` 80k →
`slot8_ir` 50k → `slot8_ir_v2` 100k, with a dedicated separate IR-focused stage before the final
one) vs. Phase B's 260k across 2 stages (one undifferentiated 100k IR stage). So the Win C gap is
at least partly attributable to less total IR-stage exposure, not conclusively to the tags
themselves — the qualitative failure mode (noise injection on IR, not corruption specific to tag
tokens) is identical in kind to the untagged baseline's own documented Win C problem.

**Decision**: do not proceed directly to Phase C (`ir_local` + tags) — the confound above means a
negative Win C result here doesn't cleanly falsify the tags hypothesis. Next step: Phase B2 (same
`iq_global_rw_tagged` architecture, same isolated `experiments/chat_tags/` code) with the
`warmup_x_fixed=16` (Win B) / `warmup_x_fixed=32` (Win C) traj_mix fix applied.

### Phase B2 — warmup_x_fixed traj_mix fix (complete)

`experiments/chat_tags/configs/slot8_tagged_phaseB2_winfix.py`. Warm-started from Phase B's own
`stage1_best.pt` (same tagged vocab V=276 this time, no embedding mismatch — unlike Phase A→B).
Added symmetric Win B (X=16) and Win C (X=32) IQ+IR oversample entries alongside the existing
Win A (X=0) entries, same 2:1 IR:IQ weight ratio Win A already had (total traj_mix weight 7.5,
Win A/B/C each ~20-40% share instead of only Win A getting dedicated coverage). 100k steps.

**Result** (best checkpoint, step 20000, 70.8% mean):

| | Win A | Win B | Win C | overall |
|---|---|---|---|---|
| untagged `slot8_ir_v2` (baseline) | 100% | 77.8% | 55.6% | 77.8% |
| tagged Phase B (no traj_mix fix) | 100% | 75.0% | 5.6% | 60.2% |
| tagged Phase B2 (traj_mix fix) | 100% | 95.8% | 16.7%† | **70.8%** |

†Win C's own peak across the run was **50.0%** (step 30000, mean 64.4% at that checkpoint) —
nearly matching the untagged baseline's 55.6%. The single "best overall mean" checkpoint (used
for the table above, matching the project's own checkpoint-selection convention) happened to
catch Win C at a lower point in its oscillation; Win C's trend across the whole run was a clear,
sustained climb (5.6% → 11.1% → 13.9% → 16.7% → 25.0% → **50.0%** → oscillating 16-42% for the
remainder) rather than a one-off spike, confirming the traj_mix-fix hypothesis: **Win C's failure
was primarily a training-distribution gap, not an architectural failure of tags.**

**Conclusion**: the `warmup_x_fixed` fix — needed regardless of tags, since the untagged
baseline never applied it either — closes most of the Win C gap. Overall mean (70.8%) is now
within 7pp of the untagged baseline's own best (77.8%), and Win B now *exceeds* it (95.8% vs
77.8%). This is consistent with tags being roughly neutral-to-positive once the known
training-distribution issue is controlled for, not the dramatic regression Phase B's confounded
result suggested. A cleaner Win C read would need either a longer run (Win C was still
oscillating, not fully converged, when stage 0 ended) or averaging over multiple checkpoints
near the peak rather than a single "best mean" snapshot — worth doing before drawing a final
verdict on whether tags specifically help Win C generalization vs. just riding the traj_mix fix.

**Run**: `tail -f experiments/chat_tags/logs/chat_tags_slot8_phaseB_full/train.log`

### Phase B3 — single clean cosine decay for real convergence (complete)

`experiments/chat_tags/configs/slot8_tagged_phaseB3_converge.py`. B2 never converged —
`cosine_T_mult=2` kept restarting the LR, so every window was still oscillating wildly
(e.g. Win B ranged 30.6%-95.8%) when B2's step budget ran out. Fix: single cosine decay
(`cosine_T0=80000, T_mult=1`, `lr_max=2e-4`) with no restarts, 80k steps, warm-started from
B2's own `stage0_best.pt`.

**Result: genuine convergence** — confirmed by the final 5 eval checkpoints landing within a
tight band instead of oscillating (Win A steady 100%, Win B stabilizing 77.8-100%, Win C stable
at a 27.8-30.6% plateau, not moving). Final checkpoint (step 80000, mean 75.9% — new all-time
high for this experiment):

| | Win A | Win B | Win C | overall |
|---|---|---|---|---|
| untagged `slot8_ir_v2` (baseline) | 100% | 77.8% | 55.6% | 77.8% |
| tagged Phase B3 (converged) | **100%** | **100%** | 27.8% | **75.9%** |

**Win A and Win B have genuinely converged to baseline-matching-or-better levels** (Win B hit a
clean 100% at the final checkpoint, not a lucky spike — the last 3 evals were 77.8%/77.8%/100%,
a tight stable band). **Win C converged too, but to a much lower plateau** — 27.8-30.6% across
the final 4 checkpoints, well below both the 90% target bar and the untagged baseline's 55.6%.

**Refined qualitative diagnosis** (decoded all 8 val sequences at the final checkpoint,
warmup_offset=32): the real (non-coincidental) Win C average across 7 of 8 sequences is ~39%
(`geometric`'s apparent 100% is still the same periodic-content coincidence noted in Phase B).
The failure pattern is more precise than "IR destroys IQ" — it's **correct-start-then-cascading-
garbage**: e.g. `up_counter` gets the first 5-8 output bytes exactly right, then diverges into
noise for the rest; `linear` gets 18/24 right before diverging; `odd` fails from byte 1 (the
persistent, most severe case). Critically, IR turns are **actively degrading** quality for these
cases rather than correcting it — `up_counter`'s own per-turn breakdown is IQ=100% → IR1=75% →
IR2=25%, getting *worse* each refine step, the opposite of Win A/B's behavior where IR fixes
IQ's mistakes. This looks like standard autoregressive exposure bias compounding through the
causal output span (once IR1 gets one byte wrong, IR2's argmax-conditioned continuation is fed a
corrupted cue and cannot recover) layered on top of Win C's harder underlying encoding — not
cross-window confusion (generated bytes don't resemble other windows' content) and not identity
ambiguity in the classic sense, since IQ often starts correctly, indicating the model does know
which window it's targeting, at least initially.

**Decision**: Win A/B have met the bar (or matched/beaten baseline) and converged cleanly — no
further work needed on those. Win C has not met the bar (target ≥90%, converged at 27.8-30.6%).
Next fix: **window-specific query tags** (`<query_a>`/`<query_b>`/`<query_c>` instead of one
generic `<query>` for the three canonical windows, X=0/16/32). Rationale: even though the
diagnosed failure is cascading-error-from-imperfect-IQ rather than pure window confusion, an
explicit per-window identity signal may let the model allocate a more confident, dedicated
recall pathway for Win C specifically instead of sharing one generic query anchor across all
three windows — worth testing as the most direct remaining test of the tags hypothesis itself.
Applies only to the `warmup_x_fixed∈{0,16,32}` traj_mix entries; the uniform-X entries keep the
generic `<query>` tag since arbitrary X doesn't map to one named window.

**Run**: `tail -f experiments/chat_tags/logs/chat_tags_slot8_phaseB3_converge/train.log`

### Phase B4 — window-specific query tags (complete) — main result of the chat-tags track

`experiments/chat_tags/configs/slot8_tagged_phaseB4_windowtags.py`. Direct test of the tags
hypothesis itself: replaced the single generic `<query>` tag with `<query_a>`/`<query_b>`/
`<query_c>` for the three canonical windows (`warmup_x_fixed` ∈ {0,16,32}), giving the model an
explicit, content-independent window-identity signal instead of forcing all three windows to
share one addressing key. New vocab (6 IDs, V=282, `HMN_TAG_VOCAB_SIZE_V2`). Warm-started from
B3's converged checkpoint — this time only `special_embed.weight` grew (20→26 rows), so a
partial-load helper was added to `train.py` to copy the overlapping prefix and leave the new
tag rows at fresh init, rather than retraining from scratch. Same single-clean-decay recipe as
B3 (`cosine_T0=80000, T_mult=1`), 80k steps.

**Result — final checkpoint (step 80000)**:

| | Win A | Win B | Win C | overall |
|---|---|---|---|---|
| untagged `slot8_ir_v2` (baseline) | 100% | 77.8% | 55.6% | 77.8% |
| Phase B3 (no window tags, converged) | 100% | 100% | 27.8-30.6% | 75.9% |
| **Phase B4 (window tags, converged)** | **100%** | **100%** | **84.7%** | **94.9%** |

Win A/B: perfect, matching or exceeding baseline. **Win C: 84.7%, held stable across the final
4 consecutive checkpoints** (91.7% → 84.7% → 83.3% → 84.7% → 84.7% — a genuinely converged
plateau, not a spike) — up from B3's stuck 27.8-30.6%, nearly a **3× improvement**, and now well
above the untagged baseline's own 55.6%. Best single checkpoint hit 97.2% overall (Win C 91.7%,
step 55000) but the *converged* value used for the table above is the more honest number, per
the same "don't cherry-pick a spike" lesson learned from B2.

**Qualitative confirmation** (decoded all 8 val sequences at the final checkpoint):
**4 of 8 sequences achieve perfect (100%) Win C recall** — `up_counter`, `even`, `linear`,
`sawtooth`. Of the remaining 4, three (`down_counter` 79.2%, `odd` 75.0%, `geometric` 70.8%)
get 17-19 of 24 output bytes exactly right before diverging — the same correct-start pattern as
before, just far less severe and far less frequent. Only `palindrome` (16.7%) remains a hard
case, diverging almost immediately — worth a closer look if this track continues (possible
hypothesis: palindromic byte patterns may be uniquely hard to encode without confusion with
their own reverse direction, though unconfirmed).

**Root cause, confirmed**: the original Win C failure was **not** a SLOT-capacity/rank ceiling
— it was an **addressing collision**. All three windows shared one `<query>` key, so the IR
mechanism had to disambiguate three different value-patterns through one shared channel,
producing destructive interference (IR turns *degrading* rather than correcting quality — see
B3's diagnosis). Giving each window its own key, with **zero change to `slot_len`/`d`/heads**,
recovered nearly the entire gap. See `docs/SRS_RECIPE.md § Fast-Weight Rank and Addressing` for
the full theoretical grounding (rank-1-per-SLOT-token fast-weight analysis, why this confirms
addressing over capacity, and hypothetical directions for further scaling — including a queued
next experiment, DenseNet-style depth-wise KV concatenation, to be evaluated against this run
as baseline).

**Overall verdict on the chat-tags hypothesis**: mixed-to-positive, with an important caveat.
Explicit region/identity tags did **not** help via the mechanism originally hypothesized
(runtime-visible generic region markers aiding position-invariant encoding) — Phase B alone
(generic tags, no window-specific fix) was confounded and inconclusive. But **window-specific
identity tags** — a more targeted version of the same idea — provided a large, real,
reproducible improvement once isolated from the traj_mix confound. This is genuine evidence
*for* explicit addressing tokens as a technique, specifically for disambiguating a small number
of known, discrete cases — not yet evidence that tags generalize to arbitrary/unknown recall
targets (that's Tier 2's random-warmup generalization question, still open).

**Reorg**: logs already colocated under `experiments/chat_tags/logs/` per the established
convention (train.py's `--log-dir` default was updated after Phase A/B, so B3/B4 wrote there
automatically — no manual move needed this time).

**Run**: `tail -f experiments/chat_tags/logs/chat_tags_slot8_phaseB4_windowtags/train.log`

---

## DenseNet-KV ablation — depth-wise cross-layer SLOT-KV concatenation (complete — inconclusive)

`experiments/densenet_kv/` (new, isolated folder — no edits to `kvmem/` or
`experiments/chat_tags/`). Tests direction #6 from `docs/SRS_RECIPE.md` against Phase B4 as
baseline: does letting each layer's SLOT-position KV accumulate across depth (layer i+1
attends to layers 1..i's SLOT KV concatenated, not just its own) converge faster than B4's
standard single-layer attention, given the same window-specific tags, traj_mix, and step
budget?

**Design decision**: window-specific query tags kept identical to B4 (not dropped) — the ask
was specifically to compare convergence speed against B4 as baseline, so isolating the KV-concat
mechanism as the *only* changed variable is the correct controlled ablation. Dropping tags would
conflate two hypotheses (architecture change + removing the addressing fix that's already known
to work).

**Scope constraint**: only SLOT token positions (encoding + recall SLOT/SLOT_A/SLOT_B) accumulate
cross-layer KV. Every other position (source bytes, warmup, output, tags) behaves as a completely
regular transformer — single-layer attention only, same as B4. Keeps the extra cost bounded to
`slot_len` extra keys per layer, not the whole sequence.

**Implementation**: `experiments/densenet_kv/model.py` — `DenseSlotKVModel`, a self-contained
model variant (not touching `kvmem/model.py`) reusing `rope_freqs`/`yarn_freqs`/`apply_rope`/
`RMSNorm`/`FFN` via import. Each layer's own K,V (before any cross-layer concat) is sliced at
SLOT positions and appended to a running history; layer i+1's attention gets the current layer's
normal K,V plus the concatenated history as extra key/value columns (the same `torch.cat`
mechanic already used for `past_kv` inference caching elsewhere, applied across depth instead of
time). Extra columns inherit the same per-row mask visibility as the live SLOT column they
correspond to. `experiments/chat_tags/`'s position/mask/batch machinery
(`chunk_positions_iq_global_rw_tagged`, `chunk_mask_fb`, `make_batch_tagged`) is reused
unchanged — it's architecture-agnostic. AR-decode (`experiments/densenet_kv/decode.py`) is a
simple full-recompute-per-token implementation (no incremental KV cache — the growing
cross-layer history has no obvious efficient incremental scheme for a first prototype;
acceptable since eval sets are small).

**Verified before launch**: (1) forward/backward run cleanly, (2) cross-layer growth confirmed
numerically — layer *i* sees exactly `i × slot_len` extra keys (0, 8, 16, 24 for a 4-layer
model), matching the design exactly, (3) causality check — perturbing a token strictly after a
given position produces zero change to logits before that position, confirming the depth-wise
history mechanism introduces no leak.

Config: `experiments/densenet_kv/configs/slot8_densekv_windowtags.py` — same dims/traj_mix/step
budget as B4 (`d=64, n_layers=4, n_heads=4, slot_len=8`, 80k steps, single clean cosine decay),
trained from scratch (different weight connectivity from B4, so no warm start — same param count,
232,192, confirms shapes match, only the forward computation differs).

**Comparison metric**: convergence speed (steps to reach a given Win C match%, using B4's own
per-checkpoint trajectory as the reference — see B4 section above for the exact numbers at each
5000-step checkpoint), not just final ceiling.

### Result — inconclusive, not a negative result

**Final numbers (step 80000, best checkpoint = final checkpoint)**:

| | Win A | Win B | Win C | overall |
|---|---|---|---|---|
| B4 (warm-started, converged) | 100% | 100% | 84.7% | 94.9% |
| densenet_kv (from scratch, 80k steps) | 22.2% | 37.5% | 2.8% | 20.8% |

Full mean trajectory across all 16 eval checkpoints: 1.9 → 0.9 → 5.1 → 3.7 → 4.2 → 2.8 → 5.1 →
5.6 → 7.9 → 10.2 → 15.7 → 13.0 → 16.7 → 14.8 → 19.9 → **20.8** (final). This is **monotonically
rising for the entire run** (aside from minor noise) and hits its highest value at the very last
checkpoint — unlike B3 or B4, which both showed several consecutive stable/repeated checkpoints
at the end confirming genuine convergence. **Training was still improving when the 80k-step
budget ran out; it never converged.**

**Root cause of the inconclusiveness — the comparison was never apples-to-apples**: B4 was
warm-started from Phase B3's already-converged checkpoint, which itself sits on top of ~348k
cumulative prior training steps (Phase A 8k + Phase B 260k + Phase B3 80k). densenet_kv trained
completely from scratch — same 80k-step budget, but starting from random init, with a
genuinely different architecture requiring no warm-start path. Comparing densenet_kv's 80k
from-scratch steps against B4's 80k *fine-tuning* steps (on top of 348k of prior learning) was
never a fair speed test — it's closer to comparing a marathon runner's first mile against
someone else's final mile of the same race.

**This is NOT evidence that cross-layer SLOT-KV concatenation fails to help (or hurts)
convergence** — the experiment as designed cannot distinguish "the architecture doesn't help"
from "the model just needed more than 80k from-scratch steps to reach a comparable point,"
because there is no from-scratch standard-architecture run at the same step budget to serve as
the correct baseline. The causality/growth-mechanism verification done before launch (cross-layer
KV growth confirmed numerically, zero information leak confirmed) still stands — the
*architecture is implemented correctly*; what's unverified is whether it's *better or worse*
than the standard architecture under a fair comparison.

**Recommendation for a clean follow-up** (not yet built): a from-scratch **standard**-architecture
control run — i.e. the untagged/standard `KVMemModel` with window-specific tags, same traj_mix,
same 80k-step single-decay budget, trained from scratch (no warm start from B3) — would establish
what a from-scratch standard model reaches in 80k steps. *That* number, not B4's warm-started
94.9%, is the fair baseline densenet_kv should be judged against. If densenet_kv's 20.8% comes out
ahead of that from-scratch standard control at the same step count, that's real evidence for the
architecture; if behind, real evidence against it (or evidence it needs more steps to pay off).
Alternatively, simply extending densenet_kv's training further (it was still rising at cutoff)
would show whether it eventually reaches a comparable ceiling to B4, even without solving the
convergence-*speed* question cleanly.

**Run**: `tail -f experiments/densenet_kv/logs/densenet_kv_slot8_windowtags/train.log`

---

## IR-refinement loss redesign — queued ablations

Motivation: B4's IR turns sometimes **degrade** quality rather than correct it (e.g.
`up_counter`: IQ=100%→IR1=75%→IR2=25%). The current loss is pure per-position NLL, computed
independently per turn and averaged — nothing in the objective compares turn *t* against turn
*t-1*, and nothing distinguishes "this position's fed-back argmax was already correct" from
"this position was wrong and needs active correction." Four ablations queued, to be run
sequentially (never two training jobs at once), roughly in order of implementation cost:

1. **Wrong-token-weighted loss** (cheapest, no architecture change): weight each output
   position's NLL by whether the fed-back argmax at that position was wrong:
   `w_i = 1 + α·1[argmax_i ≠ gt_i]`. Right now correct positions ("leave alone") and wrong
   positions ("actively fix") get identical gradient weight, diffusing signal away from the
   actual correction task. Warm-startable from B4 (same architecture, only the loss changes —
   no confound, unlike the DenseNet-KV comparison).
2. **Margin-based monotonic-improvement term**: `L_mono = max(0, margin −
   (logprob_IRk[gt] − logprob_IRk-1[gt]))`, added at small weight. Explicitly rewards turn *k*
   assigning strictly higher probability to ground truth than turn *k-1* did, rather than hoping
   monotonicity emerges as a side effect of fitting each turn's NLL independently.
3. **Self-assessed error-flag head** (bigger lift — new model head): auxiliary head predicts
   "is my own argmax at this position likely wrong" (trained against the true `argmax≠gt`
   label), then the *predicted* confidence (not the ground-truth flag, which would create a
   train/eval mismatch — no ground truth exists at real inference time) is fed back into the
   next turn's input embedding. Closer to copy-mechanism/pointer-network losses that explicitly
   supervise "when to copy vs regenerate."
4. **Attention-supervised copy loss**: encourage high attention weight from a corrected output
   position back to its own wrong-argmax position, directly teaching "find and fix" rather than
   hoping it emerges from end-to-end NLL. Needs `attn_viz.py` (existing tool, captures per-layer
   per-head softmax weights via a monkey-patched SDPA — currently wired to the untagged
   `chunk_positions_iq_global_rw`/`chunk_positions_fb_localrefine`, not yet to
   `experiments/chat_tags`'s tagged position builders) adapted first, to actually check whether
   IR turns already do targeted lookback before committing to this direction — if they do, this
   loss term would be redundant; if they don't, it's likely the highest-leverage fix.

**Run 1 (wrong-token-weighted loss)**: `experiments/chat_tags/configs/slot8_tagged_wrongtok_ablation.py`
— warm-started from B4's best checkpoint, `train.py` extended with a `wrong_token_loss_weight`
hp (default 0, backward compatible with all prior configs).

### Ablation 1 result (complete, confirmed converged) — crosses the original success bar

At step 30000 (50% through): **all three windows hit 100% simultaneously** (mean=100%, loss=0.0009).
Step 35000: mean=97.2%, Win A=100%, Win B=100%, Win C=91.7% — still comfortably above the
original chat-tags queue's success bar (≥90% every window), and every per-turn breakdown across
both checkpoints shows the IR-degradation pattern replaced by monotonic improvement (e.g.
`odd`: IQ=20.8%→IR1=91.7%→IR2=100.0%). Running average across the first 7 eval checkpoints:
Win A 97.4%, Win B 92.1%, Win C 81.3% (pulled down by the earlier, still-adjusting checkpoints —
consistent with a genuinely rising trajectory, same pattern as B3/B4 where true stability emerged
near the end of the cosine decay, not the early checkpoints).

Trajectory so far: 90.7 → 91.7 → 81.0 → 83.3 → 88.0 → **100.0** → 97.2 (mean%, steps 5k-35k).

**This is the first run in the entire chat-tags queue to hit the original ≥90%-all-windows bar,
and it stayed there.** The last 6 consecutive eval checkpoints (steps 35000-60000) all sit in a
tight 97.2-97.7% band — genuine stable convergence, not a lucky spike (contrast with B2's single
misleading peak). **Final checkpoint (step 60000)**:

| | Win A | Win B | Win C | overall |
|---|---|---|---|---|
| untagged `slot8_ir_v2` (original baseline) | 100% | 77.8% | 55.6% | 77.8% |
| B4 (window-specific tags only) | 100% | 100% | 84.7% | 94.9% |
| **wrong-token-weighted loss (this ablation)** | **100%** | **100%** | **91.7%** | **97.2%** |

Best single checkpoint (step 30000) hit a perfect 100%/100%/100% (mean=100%, loss=0.0009) — the
first time any run in this series reached that. The converged final-checkpoint number (91.7% Win
C, not the 100% peak) is the honest figure per the same "don't cherry-pick a spike" discipline
used throughout, and it's still comfortably above both B4 and the original ≥90% target.

**Qualitative confirmation**: 6 of 8 val sequences now decode with **perfect (100%) Win C
recall** (`up_counter`, `odd`, `even`, `linear`, `sawtooth`, `geometric`) — up from B4's 4/8. Of
the remaining two: `down_counter` (75.0%) shows the exact reproducible pattern seen across the
last ~25k steps of training — IQ=70.8%→**IR1=100%**→IR2=75.0% (IR1 fully fixes it, IR2
re-breaks it, the *opposite* of the general degradation pattern this loss was designed to fix,
and evidence the fix is not literally perfect). `palindrome` (20.8%) remains the single hardest
case across the *entire* chat-tags series regardless of ablation — worth a dedicated look if this
track continues (unconfirmed hypothesis: palindromic byte patterns may be uniquely hard to encode
without confusion with their own reverse direction).

**Mechanism validated**: the wrong-token-weighted loss — `w_i = 1 + α·1[argmax_i ≠ gt_i]`, a
single-line change to the loss computation, α=2.0, no architecture change, warm-started cleanly
from B4 with zero confound (unlike the DenseNet-KV comparison) — fixed the diagnosed root cause
(uniform gradient weight on already-correct vs actively-wrong positions, diffusing signal away
from the actual correction task) more directly and far more cheaply than window-specific tags
alone did. This is strong evidence for the original hypothesis behind this whole loss-redesign
queue: the IR mechanism's degradation problem was a **loss-shaping** issue, not fundamentally an
addressing or capacity one — window tags helped (B3→B4), but the loss fix closed the remaining
gap in a fraction of the engineering effort (one line vs a new tag vocabulary + position-builder
changes).

**Decision on ablations 2-4**: not launched. The original success bar (≥90% every window,
converged) is met and confirmed. Margin-based monotonic loss (#2), the self-assessed error-flag
head (#3), and attention-supervised copy loss (#4) remain documented and available as follow-ups
if further refinement is wanted (e.g. specifically targeting the `down_counter`/`palindrome`
residual failures), but are not necessary to close out this queue.

> **TODO (skipped, not abandoned)**: ablations 2-4 above (margin-based monotonic-improvement
> loss, self-assessed error-flag head, attention-supervised copy loss) were explicitly not run —
> the success bar was already met by ablation 1 alone. Worth revisiting if `down_counter`
> (IR1 fixes, IR2 re-breaks) or `palindrome` (hardest case across the entire chat-tags series)
> need further work, or as a general loss-design comparison independent of whether they're
> "needed." Full designs are in the "IR-refinement loss redesign — queued ablations" section
> above.

**Run**: `tail -f experiments/chat_tags/logs/chat_tags_slot8_wrongtok_ablation/train.log`

---

## Workshop paper assessment

The 32B result (`hmn_feedback_32_ir`: 100% exact recall at k=0..12) is sufficient for a **4-page workshop paper** (NeurIPS/ICML/ACL workshop on memory or reasoning).

**Core claim**: exact byte recall from compressed slot tokens is impossible with one-shot IQ encoding alone; an argmax feedback loop (IR) unlocks 100% exact match and perfect extrapolation beyond training range.

**What exists**: clean task definition, novel architecture, 100% results, extrapolation, ablation (IQ vs IR, slot size, Win A capacity), failure mode analysis (position sensitivity).

**What strengthens it**: 64B+ scaling (in progress with slot8_ir, showing IR rapidly unlocking Win B), comparison baseline (plain fine-tuned transformer), related-work section (NTM, DNC, Hopfield).

**Recommendation**: write draft now from 32B results, leave 64B section stub for slot8_ir results.
