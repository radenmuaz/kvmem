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

---

## Exp 3a — Refine Stage A (single block, 1 draft turn)

Config: `configs/refine_denoise.py` — 80k steps, out_len=8, noise U(0.05, 0.8), single attempt.

| Metric | Value |
|--------|-------|
| val_ref_bpb | 0.063 |
| draft match% (t1) | 1.6% |
| final match% | 92.2% |
| Δ correction | +90.6% |

**Sawtooth fails at 75%** — when evaluated with 2+ draft turns, the model degrades: t1=75%, t2<50%. Sawtooth means the model learned to "fix one noisy turn" but not to iteratively improve.

---

## Exp 3b — Refine Multi-Turn (FAILED)

Config: `configs/refine_multiturn.py` — k~Uniform(1,5), descending noise schedule.

| Metric | Value |
|--------|-------|
| final match% | 17.2% |

**Root cause: train-eval distribution mismatch.** Descending noise (turn j gets U(0, noise_hi×(K-j)/K)) makes later training drafts near-clean. At eval time, model's own AR drafts are still noisy (≈15% error). Model learns "last draft is clean, just copy" — fails when last draft is noisy.

**Fix:** flat noise schedule — all turns use same U(lo, hi) range.

---

## Exp 3c.1 — Joint Mix, out_len=16 (BROKEN DESIGN)

Config: `configs/refine_joint.py`, out_len=-1 (=16=seg_len), 80k steps.

Reported 82% match but had two bugs: (1) `n_win=max(1,0)=1` forced y_start∈{0,1}, making 50% of training examples zero-pad position 15 — identical warmup, contradictory targets; (2) eval f_start went negative, giving y_start=0 with empty warmup padded to [seg[0]]×4 — no real NTP anchor. The design was broken: `out_len=seg_len` leaves no preceding context for warmup.

**Key lesson:** `out_len` must be < `seg_len` to preserve NTP-style warmup (preceding bytes as anchor).

---

## Exp 3c.2 — Joint Mix, out_len=12, NTP Warmup (CURRENT)

Config: `configs/refine_joint.py` — 80k steps, 74 min on MPS. **out_len=12, warmup_len=4** → warmup=seg[0:4] (real NTP anchor), gt=seg[4:16] (12-byte continuation = full segment coverage).

**Training mix:** 30% I Q + 20% I R Q (k~0..5 flat noise) + 20% I I Q₀ + 30% interleaved n=2

**Final metrics (step 80k):**

| Metric | Value |
|--------|-------|
| val_ref_bpb | 0.134 |
| val_bpb (IQ) | 0.126 |
| n1_r0 (last turn AR) | 65.6% |
| IQ baseline match | **95.8%** (7/8 seqs 100%) |
| Refine 100% hit rate | **0/8 seqs** in 20 turns |

**IQ outperforms refine.** The plain `<q><y>` recall reaches 100% on 7/8 sequences; multi-turn refine never reaches 100%.

**Failure analysis — two patterns:**

*Pattern A — converges to wrong fixed point (4/8 seqs):*
```
up_counter t1: ...2c 2d [2c 2d]   gt: ...2c 2d [2e 2f]   (last 2 bytes wrong)
t2 = t1 (identical) — correction does nothing, stuck forever
```
Errors are always at positions 10–11 (last 2 bytes of 12). First 10 bytes correct. No improvement across any number of turns.

*Pattern B — catastrophic divergence (4/8 seqs):*
```
odd t1: 91.7% (1 wrong)
odd t7: 58.3% → t12: 16.7% → t20: 8.3%  ← cascading collapse
```
Starts near-correct, then each correction turn corrupts already-correct earlier bytes. By turn 20 output is nearly random.

**Root causes:**
- End-of-sequence blind spot: model almost always fails at the last 1–2 bytes (positions 10–11), correct on 0–9.
- Correction divergence: correction `<h>` updates toward wrong direction — with each additional attempt context, more of the sequence gets corrupted, not less.
- `aux_attempt_loss` helps teacher-forcing but doesn't prevent AR drift — the distribution of training drafts (synthetic noise) vs eval drafts (model's own correlated errors) still diverges.

**Checkpoint:** `logs/role_refine_joint/checkpoints/stage0_end.pt`

---

## Next Steps

**Refine mechanism (priority):**
- End-of-sequence blind spot: model fails at last 1–2 bytes — investigate positional bias in loss (positional LS did not fix it)
- Correction divergence: each refine turn corrupts output rather than fixing it — the correction `<h>` update direction is wrong
- Possible fix: train with model's OWN AR drafts (online rollout) rather than synthetic noise — closes train-eval distribution gap
- Alternative: restrict correction to learn delta only (what's wrong), not regenerate the whole sequence

**Other experiments:**
- Exp 4: Natural language corpus — replace random bytes with structured text + line numbers
- I I R₁ Q₀ — refine new block, prove old retained (SRS under pressure)
- Test n=6,7,8 — chain extrapolation limit
