# KV Memory — Session Summary (June 2, 2026)

## Current Status

All ablations complete. Ready to launch full curriculum run with OCD on stages 2–3.

---

## Ablation Results

### Run 1: GrokAdamW + flat LR — `logs/role_20260602_083819`
Curriculum V1 (40k/40k/40k/20k/20k, `cycle_steps=999999` → flat LR).

| Stage | seg | best match | final match |
|---|---|---|---|
| 2 | 128 | **28.1%** (step 35k) | 21.9% |
| 3 | 256 | 21.9% | 8.8% |
| 4 | 576 | 19.4% (step 5k) | **13.2%** |

Root cause: flat LR + GrokAdamW → model oscillates between grokking and forgetting.

### Run 2: AdamW + cosine — `logs/role_20260602_114432`
Curriculum V2, stopped at stage 2 step 873 (killed for ablation).

### OCD ablation (seg=32, 10k steps, AdamW, seed=42)

| Config | Wallclock | Synthetic match | Surah windowed recall |
|---|---|---|---|
| TF only (`role_20260602_122443`) | 153s | 9.4% peak | 43.8% |
| OCD every=10 / prob=0.1 (`role_20260602_122736`) | 229s (+50%) | 9.4% peak | 47.5% |
| OCD prob=0.1 stochastic (`role_20260602_132251`) | 214s (+40%) | 7.8% peak | — |

**Findings:**
- OCD does not improve convergence on the synthetic training distribution
- OCD gives +3.7pp on surah windowed recall (closing the TF/AR inference gap)
- Every-K and prob-p are equivalent in expectation; prob-p preferred (no periodic pattern)
- NLL-gated OCD dropped — introduces `ocd_nll_frac` hyperparameter; better to manually inspect CER

---

## OCD Cost Analysis

OCD cost per step = `out_len × tf_step_time` (no KV cache → full forward pass per generated token).

| Stage | seg | out_len | TF cost | OCD cost | p_rec | every_K | OCD steps | Extra time |
|---|---|---|---|---|---|---|---|---|
| 0 | 32 | 8 | 16ms | 0.13s | **0** | — | 0 | 0 |
| 1 | 64 | 16 | 31ms | 0.50s | **0** | — | 0 | 0 |
| 2 | 128 | 32 | 50ms | 1.60s | **0.05** | 20 | 2000 | +52 min |
| 3 | 256 | 64 | 52ms | 3.33s | **0.01** | 100 | 1000 | +55 min |
| 4 | 576 | 128 | 211ms | 27s | **0** | — | 0 | 0 |

- Stages 0–1: TF only — out_len too short, exposure bias negligible
- Stage 4: 27s/OCD step — needs KV cache before OCD is viable at useful rate (1000 OCD steps = 7.5h)
- **Total with OCD: ~10.2h (+105 min over TF-only 8.4h)**

`--ocd-every` applies globally. Per-stage `ocd_prob` in curriculum dict is a TODO.

---

## Architecture: Role-Tag Scheme

**Sequence format:**
```
<s> x_S </s> <m> slots </m> <f> warmup </f> <c> output </c>
```

Tags (printable ASCII, from 256-token vocab):
- `<s>/<s>` src open/close, `<m></m>` mem open/close, `<f></f>` anchor, `<c></c>` output

**Mask rules:**
- slots attend to x_S (encode source into KV)
- `<f>` region attends to slots only (locate via KV, not source directly)
- `<c>` region attends to slots + `<f>…</f>`, CANNOT see x_S
- Nothing outside `<c>` attends to `<c>` (write-only)

**`</c>` dropout**: 50% probability — teaches open-ended generation past window boundary.

---

## Stack (PyTorch)

- `kvmem/model.py` — Transformer (B,L)→(B,L,V), RoPE/YaRN, SDPA
- `kvmem/train_role.py` — Role-tag curriculum training
  - `--curriculum v1/v2/none` (none = single stage from CLI args, for ablations)
  - `--ocd --ocd-mode every/prob --ocd-every K --tf-warmup N`
  - `--no-grok` (plain AdamW)
- `kvmem/optim.py` — GrokAdamW (SNR-gated AdamW)
- `kvmem/eval_surah.py` — Windowed recall eval on suratalfatihah.txt

**OCD compilation note:** The model forward pass is compiled (`torch.compile`), but the rollout loop (`for k in range(out_len)`) is not. Sequential Python loop over compiled model calls — no graph fusion across steps. Fundamental to greedy AR.

---

## Curriculum Configs

```python
# V2 — recommended (100k steps stages 3-4, per-stage cosine LR decay)
CURRICULUM_SURAH_V2 = [
    dict(seg_len= 32, N= 32, warmup_len= 8, out_len= 8,  B=16, n_steps= 40000, cycle_steps= 40000),
    dict(seg_len= 64, N= 64, warmup_len=16, out_len=16,  B=16, n_steps= 40000, cycle_steps= 40000),
    dict(seg_len=128, N=128, warmup_len=32, out_len=32,  B= 8, n_steps= 40000, cycle_steps= 40000),
    dict(seg_len=256, N=256, warmup_len=32, out_len=64,  B= 4, n_steps=100000, cycle_steps=100000),
    dict(seg_len=576, N=576, warmup_len=32, out_len=128, B= 4, n_steps=100000, cycle_steps=100000),
]
```

TODO: add per-stage `ocd_prob` to curriculum dict (same pattern as `cycle_steps`).

---

## Next Run Command

```bash
# V2 + OCD on stages 2-3 (global ocd-every=20 approximates p≈0.05 for stage2, p≈0.01 for stage3 once per-stage is wired)
python -m kvmem.train_role \
  --d 64 --n-layers 4 --lr 3e-4 \
  --eval-every 5000 --log-every 1000 \
  --drop-close 0.5 --curriculum v2 \
  --ocd --ocd-mode prob --ocd-prob 0.05 --tf-warmup 2000 \
  --no-grok --compile --device mps --log-dir logs
```

---

## Surah Eval

```bash
python -m kvmem.eval_surah --ckpt logs/<run>/checkpoints/stage4_end.pt --device mps --n-windows 20
```

File: `datasets/suratalfatihah.txt` — 562 bytes raw UTF-8 Arabic, no preprocessing.
Target: ≥50% windowed recall. Best so far: **47.5%** (TF seg=32, 10k steps, OCD p=0.1).

---

## Key Findings

| Result | Details |
|---|---|
| Full-seq recall | 100% at seg=576, JAX model (`mini_recall_20260531_185919`) |
| Suratalfatihah full recall | 100% (full AR decode, no chunking) |
| Windowed recall best | **47.5%** (seg=32, OCD p=0.1, 10k steps) |
| GrokAdamW vs AdamW | No convergence advantage; GrokAdamW amplifies oscillation with flat LR |
| OCD vs TF | +3.7pp windowed recall, +50% wallclock, zero improvement on synthetic |
| OCD scaling | Cost ∝ out_len — impractical for stage 4 (128 tokens) without KV cache |

---

## Important Files

```
kvmem/
  model.py          — PyTorch transformer
  train_role.py     — Main training script (curriculum + OCD)
  optim.py          — GrokAdamW
  eval_surah.py     — Windowed recall eval

logs/
  mini_recall_20260531_185919/  — JAX seg=576 100% ckpt
  role_20260602_083819/         — GrokAdamW + flat LR (best: 28.1% stage2)
  role_20260602_114432/         — AdamW + cosine V2 (stopped stage2 step 873)
  role_20260602_122443/         — TF ablation (seg=32, 10k)
  role_20260602_122736/         — OCD every=10 ablation (seg=32, 10k)
  role_20260602_132251/         — OCD prob=0.1 ablation (seg=32, 10k)

datasets/
  suratalfatihah.txt — 562 bytes raw UTF-8 Arabic
old/
  ocd.py            — JAX OCD implementation (reference)
  stage0_ocd.py     — JAX OCD training loop (reference)
```
