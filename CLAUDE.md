# kvmem

Fast-weight language model. `<h>` = compressed memory updated by forward passes, no backprop.
Full reference: [`docs/BOOK.md`](docs/BOOK.md)

---

## Status

**Exp 4a done (step 8k)** — online_refine, seg=16, out=12. `configs/online_refine.py`

| Metric | Value |
|--------|-------|
| val_bpb | 0.082 |
| n1_r0 t1 | **100%** exact match (all 8 seqs) |
| n1_r0 final | **100%** |

**Key finding:** Teacher h training (gradient-guided h targets, zero-noise teacher force) fixes correction divergence. Converges to 100% in single refine turn by step 8k.

**Current:** Exp 4b — harder test `src=64, out=56, slot=1, latent=3`. `configs/online_refine_64.py`
- k sampled from {0, 4, 8} turns; teacher runs k gradient steps, pairs turn t with h_t*
- Expect t1 < 100%, multi-turn correction visible
- If 64 too easy (t1=100%): try src=128, or src=64 out=60 (warmup=4), or slot_len=0 (latent only)
- If 64 too hard (no convergence): try src=32 out=24, or src=64 out=48 (more warmup context)
- If error signal unclear: tune lengths until multi-turn improvement is visible, then ablate
- eval_n_attempts=20 to test extrapolation beyond trained max (8 turns)

Checkpoints:
- `logs/role_refine_joint/checkpoints/stage0_end.pt` (Exp 3c.2 baseline)
- `logs/role_online_refine/checkpoints/stage0_end.pt` (Exp 4a, 100% at t1)

---

## Ablations TODO

### A1 — `<r>` vs `<q>` tag fusion
**Question:** should refine mode use a separate `<r>` anchor tag, or fuse with `<q>`?

**Current:** `<r>` (ID 266) is used as the refine warmup anchor, distinct from `<q>` (ID 260).
The model can learn that seeing `<r>` means "correction turns will follow" and adjust t1 behavior.

**Fused:** replace all `REFINE_OPEN/CLOSE` with `QUERY_OPEN/CLOSE` in `make_refine_batch` (data.py).
- Pro: t1 is always the model's best single-shot attempt (same behavior as IQ)
- Pro: simpler — one fewer pair of tag embeddings to learn
- Con: model can't distinguish "final answer" from "first attempt" mode
- Implementation: one-line change per tag in data.py, no vocab change

**Hypothesis:** fused is better — teacher h training already drives correction quality through
MSE loss, not through the tag. A "lazy first attempt" behavior under `<r>` would hurt t1 quality.

### A2 — diff residual target
**Question:** supervise correction h to output delta (-lr·grad) vs direct updated h (h - lr·grad)?

**Diff:** MSE(h_enc + h_corr_t, h_teacher) — model learns correction delta, needs residual add at inference.
**Direct (current):** MSE(h_corr_t, h_teacher) — model learns absolute target h directly.

**Hypothesis:** direct simpler and no residual add needed at inference. Diff may help if correction
blocks naturally learn incremental updates.

### V1 — verify both refine and query paths reach 100%
**Check:** at eval, two independent outputs must both match 100%:
1. Last refine attempt (`all_attempts[-1]` from `ar_decode_refine`) — currently logged as `n1_r0_tN`
2. Post-refine query (`<q>wm</q><y>` decode after copy+final-h) — currently only NTP loss `val_ref_bpb`, NOT a match%

**Implementation needed:** extend `ar_decode_refine` to continue past the last attempt:
- Write last attempt as copy turn input
- Let model generate final `<h>` correction block
- AR decode `<q>wm</q><y>` under the refine mask
- Report this as a separate `n1_r0_query` match% in eval

**Why it matters:** training loss is on the query path, not the attempt path. If query=100% but attempt<100% (or vice versa), the two paths disagree — suggests the h correction works on one path but not the other.

### A3 — progressive teacher steps vs fixed target
**Current:** teacher runs k gradient steps (same as sampled k), pairs turn t with h_t*.
**Alternative:** always 1 teacher step → same h_teacher for all turns (monotonicity from mono_penalty only).

**Hypothesis:** progressive targets enforce monotonicity in h-space directly, should be stronger than
token-NLL mono_penalty alone.

---

## Run

```bash
python -m kvmem.train --config configs/refine_joint.py --device mps
python -m kvmem.train --config <cfg> --eval-only <ckpt> --device mps
python -m kvmem.train --config <cfg> --resume <ckpt> --device mps
python -m kvmem.train --config <cfg> --pretrained <ckpt> --device mps
```

---

## Monitor

```bash
# Refine mode — val_ref_bpb is the primary signal:
tail -f logs/role_<name>/train.jsonl | python3 -c "
import sys,json
for l in sys.stdin:
    d=json.loads(l)
    if 'val_ref_bpb' in d:
        t_keys=sorted(k for k in d if k.startswith('n1_r0_t'))
        tstr=' '.join(f'{k}={d[k]}%' for k in t_keys)
        print(f'@{d[\"global_step\"]}: vbpb={d[\"val_bpb\"]:.3f} vrbpb={d[\"val_ref_bpb\"]:.3f} n1={d.get(\"n1_r0\",\"?\")}% {tstr} {d[\"elapsed\"]}')
"

# Standard mode — match%:
tail -f logs/role_<name>/train.jsonl | python3 -c "
import sys,json
for l in sys.stdin:
    d=json.loads(l)
    keys=sorted(k for k in d if k.startswith('n') and '_r' in k)
    if keys: print(f'@{d[\"global_step\"]}: ' + '  '.join(f'{k}={d[k]:.0f}%' for k in keys))
"
```

**Performance:** 41 it/s training. AR decode eval ~10 min/checkpoint. Use `eval_every=10000`.

---

## Key Principles

- `null_kv=True` always — 1.5–2× faster, better bpb
- `dataset_size=0` (infinite stream) — fastest convergence
- `mode='joint'` for mixed trajectories — prevents regression
- `aux_attempt_loss=0.3` for refine — necessary but not sufficient; fixes teacher-forced sawtooth but AR eval still diverges
- flat noise (`noise_lo/hi`) — same range all draft turns; synthetic noise still mismatches model's own correlated AR errors
- `out_len < seg_len` required — full recall (`out_len=seg_len`) leaves no prior context for NTP warmup
- `eval_every=10000` — `val_ref_bpb` is sufficient live signal

---

## Docs

| What | Where |
|------|-------|
| Full reference book | [`docs/BOOK.md`](docs/BOOK.md) |
| All experiment results | [`docs/EXP_RESULTS_SUMMARY.md`](docs/EXP_RESULTS_SUMMARY.md) |
| Exp 2 tracking | [`docs/EXP2_MULTITURN_TRACKING.md`](docs/EXP2_MULTITURN_TRACKING.md) |
| Plans & theory | [`docs/plan/`](docs/plan/) |
| Active configs | [`configs/`](configs/) |
