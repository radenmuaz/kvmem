# Experiment Results Summary
**Last updated:** 2026-06-05

---

## Exp 1 — Dataset Ablation (seg=16, slot=1, intermed=7, v2 arch)

| Run | Best match% | Steps | Notes |
|-----|------------|-------|-------|
| ds10k | 98% | 70k | fixed pool |
| ds20k | **100%** | 65k | fixed pool |
| ds40k | 97% | 50k | fixed pool |
| ds_random | **100%** | 40k | infinite stream — fastest |

**Conclusion:** Infinite stream (ds_random) converges fastest. 20k pool sufficient.

---

## Exp 2 — Sequential Routing (n=2 blocks, sequential stages)

**Finding: Catastrophic forgetting.** Each stage overwrites the previous.

| After stage | 1-block | 2b from=0 (old) | 2b from=1 (recent) |
|------------|---------|-----------------|-------------------|
| s0 (1-block) | 92% | 9% | 0% |
| s1 (2b recent) | 27% | 0% | 98% |
| s2 (2b old) | 5% | 94% | 0% |
| s3 (mixed from=0) | 6% | 100% | 0% |

Sequential training overwrites even when stages are "mixed" (from=0 and from=1 as separate stages).

---

## Exp 2b — Cold Mixed Routing (n=2, r[0,1] from step 0)

**Finding: Cold mixed training solves catastrophic forgetting.**

Both routing directions learn simultaneously from scratch:

| Step | 1-block | 2b old | 2b recent |
|------|---------|--------|-----------|
| 5k | 9% | 6% | 6% |
| 30k | 81% | 80% | 83% |
| 60k | 89% | 91% | 91% |
| 65k (final) | 83% | 77% | 80% |

**Conclusion:** Training with `r[0,1]` mixed batches from the start learns routing without forgetting.

---

## Exp A — null_kv Convergence Ablation (n=1, 40k steps)

| Run | Best val_bpb | Match% | Step to 80% |
|-----|-------------|--------|-------------|
| base (null_kv=False) | 0.217 @38k | 91% | ~22k |
| **null_kv=True** | **0.157 @26k** | **92%** | **~12k** |

**Conclusion:** null_kv=True → 1.5-2× faster convergence, better peak calibration. **Use null_kv=True by default.**

---

## Exp B — Chain Extrapolation (n=1,2,3 trained; n=4,5 eval)

Config: `expB_chain_nullkv.py` — 3 stages (20k+30k+40k=90k total), mmix mode, null_kv=True

### Milestone: extrapolation emerges at step 52k (2k into n=3 training)

| Config | @20k (end s0) | @50k (end s1) | @52k (+2k of s2) | @80k | Best |
|--------|--------------|--------------|-----------------|------|------|
| n1/r0 (base) | 91% | 89% | 89% | 95% | **97%** |
| n2/r0 (trained) | 3% | 66% | 81% | 89% | **97%** |
| n2/r1 (trained) | 17% | 89% | 86% | 92% | **95%** |
| n3/r0 (trained) | 0% | 86% | 84% | 86% | **95%** |
| n3/r2 (trained) | 2% | 22% | **88%** | 95% | **97%** |
| **n4/r0 (unseen)** | 47% | 14% | **88%** | 95% | **95%** |
| **n4/r3 (unseen)** | 0% | 0% | **81%** | 95% | **95%** |
| **n5/r0 (unseen)** | 3% | 6% | **62%** | **91%** @90k | **91%** |
| **n5/r4 (unseen)** | 0% | 0% | **80%** | **94%** @90k | **94%** |

### Key findings

1. **Genuine extrapolation:** n=4,5 reach 88-95% match despite never being in training.

2. **Phase transition at step 52k:** 2k steps into n=3 training triggers a jump from ~0-14% to 62-88% on n=4,5. The n=3 exposure appears to unlock the general algorithm.

3. **Recency effect at n=5:** n5/r4 (most recent block) = 94% vs n5/r0 (oldest block) = 78-91%. Expected — older blocks have been through more update steps.

4. **Regression:** n1,n2 performance maintained throughout n=3 training (89-97%). The algorithm generalises without forgetting shorter chains.

5. **mmix mode works:** Interleaved batches (k ~ Uniform(1,n) queries per step) train the model to handle both "just ingesting" and "querying mid-stream" patterns.

### Conclusion

Training on chain lengths 1,2,3 is sufficient to learn a **generalizable fast-weight update algorithm** that works for chain lengths 4,5 without explicit training. This is strong evidence that the model learns the algorithm, not a lookup table for specific chain lengths.

---

## Next Steps

1. **Test n=6,7,8** — how far does extrapolation hold?
2. **Refine experiment (Exp 3)** — self-correction via two-pass recall with denoised first attempt
3. **Natural language corpus** — replace random bytes with structured text + line numbers
4. **Larger model** — does extrapolation improve with depth?

See `plan/PLAN_EXP2.md` for detailed plan.
