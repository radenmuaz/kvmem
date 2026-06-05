# Exp 1 — Dataset Size Ablation
**Date:** 2026-06-03  
**Architecture:** v2 (RNN tags, learned embeddings, source-first causal, slot_len=1, intermed_len=7)  
**Config:** `configs/single_s16.py` — seg=16, slot=1, intermed=7, warmup=4, out=8, B=16, lr=3e-4, wd=0.001, warmup_steps=1000, cosine LR, grad_clip=10.0

## Setup

Four runs varying `dataset_size` (fixed pool of training batches):
- ds10k: 10,000 batches (160k examples)
- ds20k: 20,000 batches
- ds40k: 40,000 batches
- ds_random: 0 = infinite stream (new batch every step)

All runs: 80k steps, eval every 5k. Eval = AR decode on fixed test sequences, measuring match% (exact byte match over out_len=8 output).

## Results

| Run | Best match% | At step | Final val_bpb |
|-----|------------|---------|--------------|
| ds10k | 98% | 70k | 0.187 |
| ds20k | **100%** | 65k | 0.134 |
| ds40k | 97% | 50k | — |
| ds_random | **100%** | 40k | — |

## Key Observations

**Infinite stream (ds_random) converges fastest** — reached 100% at 40k steps vs 65k for ds20k. No fixed pool means every batch is novel, forcing genuine generalisation rather than cycling through the same examples.

**ds20k also hits 100%** — a fixed pool of 20k batches (320k examples, ~4 epochs in 80k steps) is sufficient. More pool (ds40k) does not help convergence speed.

**ds10k plateaued below 100%** (98%) — the fixed pool may be too small to avoid some memorisation, or simply needs more steps.

**Val_bpb trending down throughout all runs** — no overfitting observed within 80k steps. The model keeps improving calibration even as match% oscillates.

## Diagnosis

The match% oscillation (not monotonically increasing) is characteristic of the model approaching but not yet locking in the algorithm. Once locked in, match% should stay at 100% without backsliding. ds20k and ds_random both showed stable 100% once reached.

## Comparison vs v1 (for reference)

v1 baseline (same task, old arch): 93.8% match @40k (seg=16, slot=8, active=1, full-pass TF).  
v2 improvement: 98-100% with cleaner architecture (slot_len=1, intermed=7, no active_slots masking).

## Conclusion

**Recommended config:** ds_random (infinite stream) for primary training. It converges fastest and forces the strongest generalisation. Use ds20k as a reproducible fixed baseline.
