# Exp 2 — Multi-Turn Recall + mem_window Ablation
**Date started:** 2026-06-04  
**Plan:** `plan/PLAN_EXP2.md`  
**Config:** `configs/exp2_multiturn.py`  
**Architecture:** v2, seg=16, slot=1, intermed=7, cosine LR, B=16, dataset_size=20000

## Curriculum (9 stages)

| Stage | Task | n_blocks | recall_from | mem_window | Steps |
|-------|------|----------|-------------|------------|-------|
| 0 | 1-block baseline | 1 | 0 | 0 | 40k |
| 1 | 2-block recent | 2 | 1 | 0 | 80k |
| 2 | 2-block old | 2 | 0 | 0 | 80k |
| 3 | mixed from=0 | 2 | 0 | 0 | 80k |
| 4 | mixed from=1 | 2 | 1 | 0 | 80k |
| 5 | isolated from=0 (mem_window=1) | 2 | 0 | 1 | 80k |
| 6 | isolated from=1 (mem_window=1) | 2 | 1 | 1 | 80k |
| 7 | Markov from=0 (mem_window=2) | 2 | 0 | 2 | 80k |
| 8 | Markov from=1 (mem_window=2) | 2 | 1 | 2 | 80k |

## Generalisation Eval Design

At the end of each stage, the model is evaluated on **all 3 configurations** regardless of training config:
- `1b` = 1-block recall (does it still work after multi-block training?)
- `2b_old` = 2-block, recall from=0 (earlier chunk)
- `2b_recent` = 2-block, recall from=1 (most recent chunk)

For multi-block eval: `ar_decode_role` builds the correct n_blocks sequence with random distractor blocks, so match% reflects actual routing from anchor content.

## Results

### Per-stage training match%

| Stage | Best match% | At step | Val bpb (best) |
|-------|------------|---------|----------------|
| s0 (1-block) | 92% | 40k | 0.254 |
| s1 (2b recent) | 98% | 55k | 0.205 |
| s2 (2b old) | 98% | 125k | 0.482 |
| s3 (mixed from=0) | **100%** | 245k | 0.028 |
| s4 (mixed from=1) | 94% | 250k | 0.474 (running) |
| s5-8 | pending | | |

### Generalisation evals (end-of-stage cross-config)

| After stage | 1-block | 2b old (from=0) | 2b recent (from=1) | Interpretation |
|------------|---------|-----------------|-------------------|----------------|
| s0 | 92% | 9% | 0% | 1-block only, random multi-block |
| s1 | **27%** | 0% | 98% | forgot 1-block; mastered recent |
| s2 | **5%** | 94% | **0%** | forgot s1; mastered old |
| s3 | 6% | **100%** | **0%** | perfect on old; forgot recent |
| s4 | pending | | | |

## Key Findings So Far

**Catastrophic forgetting is structural.** Each stage overwrites the previous format. Even the "mixed" stage (s3: from=0 only) achieves 100% on 2b_old but 0% on 2b_recent — it didn't learn to route, it just specialised.

**Each task is learnable in isolation** — both 2b_old and 2b_recent reach ~98% when trained alone. The model CAN do both; it just can't retain both sequentially.

**1-block skill is quickly overwritten** — drops from 92% → 27% → 5% after just two multi-block stages. The single-block algorithm and the multi-block algorithm are different enough that they interfere.

**Pending:** Stage 4 (mixed from=1) result will show if training from=1 immediately after from=0 restores 2b_recent. Then stages 5-8 (mem_window ablation) will test whether isolated blocks or Markov-window updates change the forgetting pattern.

## Hypotheses to Test

1. **mem_window=1 (isolated):** Each `<h>` compresses only its own block, no fast-weight accumulation. Should eliminate the forgetting problem (blocks are independent) but may reduce overall recall quality.

2. **mem_window=2 (Markov):** Each `<h>` sees 1 previous `<h>`. Intermediate between isolated and full history. May preserve more cross-block information while limiting interference.

3. **Joint mixed curriculum (future):** Train from=0 and from=1 simultaneously in same batches rather than sequential stages. Expected to fix forgetting.
