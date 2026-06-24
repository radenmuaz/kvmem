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
