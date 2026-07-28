# TRC / TPU setup estimate

Not yet implemented — this is a planning estimate for if/when TPU Research
Cloud (TRC) access is used for this project, written before any actual
porting work started. Nothing in `kvmem/` currently runs on TPU; the model
is pure eager PyTorch (`kvmem/hmn.py`/`kvmem/hmn_notags.py`), and every
number below assumes that porting work happens first (see "Porting
prerequisite" at the end — this is the actual blocker, not batch size or
model size).

Grounded in `hmn_locate_nope_curriculum_dense.py` specifically (current
architecture: `d=64, n_layers=8, n_heads=4`, single_attn, 165,568 params;
sequence length 36-150 tokens across its 4 curriculum stages, measured
directly via `chunk_positions_traj`):

```
stage0: chunk_len= 8  n_entries= 6  L min=36 max=38  mean=37.3  B=16
stage1: chunk_len=16  n_entries=25  L min=36 max=54  mean=47.7  B=12
stage2: chunk_len=32  n_entries=39  L min=36 max=86  mean=64.5  B=6
stage3: chunk_len=64  n_entries=51  L min=36 max=150 mean=90.5  B=4
```

## The core situation

165K params and L=36-150 is a vastly smaller workload than TPUs are built
for — a single TPU core has orders of magnitude more compute/memory than
this model could ever use at its current size. The real bottleneck isn't
FLOPs, it's host-side overhead: per-step Python trajectory sampling, NumPy
mask/batch construction (`make_batch_tagged`, `chunk_mask_fb_traj`), all of
which currently runs eagerly on CPU before each step. Batch size and model
size decisions only matter once that's addressed.

## TRC's actual tiers — working assumption: v5e / v5p / v6e

**Still unconfirmed which exact generation TRC grants** — their public
pages don't publish a fixed tier list (quota is "as listed in the
[applicant's] activation email," per-applicant, not a published default).
Per explicit instruction, this doc now assumes the grant is one of
v5e/v5p/v6e (Trillium) — Google Cloud's current GA generations — rather
than the old (2022-2023-era, now likely stale) v2/v3/v4 default. **The
authoritative source is still your own TRC activation email or
`trc-support@google.com`** — treat everything below as the estimate to
redo once that's in hand, not a confirmed grant.

Specs confirmed live (WebSearch/WebFetch, 2026-07-28, cross-checked against
`docs.cloud.google.com/tpu/docs/v6e`):

| tier | HBM/chip | HBM bandwidth/chip | peak bf16 TFLOPS/chip | MXU array | typical small slice |
|---|---|---|---|---|---|
| v5e | 16 GB | 819 GB/s | ~197 | 128×128 | v5e-8 = 8 chips, 128GB total |
| v5p | 95 GB | — | ~459 | 128×128 | v5p-8 = 8 chips, 760GB total |
| v6e (Trillium) | 32 GB | 1638 GB/s | ~918 | **256×256** | v6e-8 = 8 chips, 256GB total (confirmed via docs), v6e-4 = 4 chips/128GB, v6e-1 = 1 chip/32GB (testing only) |

Trillium's MXU is 256×256 (quadruple the ops/cycle of v5e/v5p's 128×128) —
one more reason the `d ∈ {256, 512, 1024}` architecture choices below are
already well-suited: all three are clean multiples of 256, not just 128,
so they fill v6e's wider array too, not only v5e/v5p's.

(Old v2/v3/v4-8 table, kept for reference only — NOT the current
assumption: v2-8 8GB/core/64GB total, v3-8 16GB/core/128GB total, v4-8
32GB/chip/256GB total.)

## Model size: current (165K) is too small to use a TPU meaningfully

At `d=64`, matmuls are far smaller than a TPU's MXU systolic array (128×128
on v5e/v5p, 256×256 on v6e) — most of the array sits idle regardless of
batch size. Scaling the model up to a `d` that's a clean multiple of 256
uses the hardware correctly on ALL THREE assumed generations (256 is also
a multiple of 128) and gives a more interesting model to actually study.
Three target sizes, computed via `params = 4*n_layers*d*d + 256*d +
n_special*d + d*V_out` (single_attn: 4 d×d linears per layer, no bias, no
FFN):

| target | d | n_layers | n_heads | d_head | actual params |
|---|---|---|---|---|---|
| ~1M | 256 | 4 | 8 | 32 | 1,184,256 |
| ~10M | 512 | 10 | 8 | 64 | 10,757,120 |
| ~50M | 1024 | 12 | 16 | 64 | 50,874,368 |

## Batch size per (tier × model size)

Rough ballpark only — no empirical profiling behind these, treat as a
starting point to binary-search from once actually ported, not a
guarantee. Bigger model → smaller max batch (the standard activation-memory
tradeoff — activations scale roughly with `B·L·d·n_layers`, and both `d`
and `n_layers` are going up a lot relative to the current 165K baseline).
Scaled from each generation's real HBM/chip above (v5e 16GB ≈ old v3-8's
16GB/core; v6e 32GB ≈ old v4-8's 32GB/chip; v5p's 95GB is ~3x v6e's, so
its batch ceiling scales up proportionally) — HBM sets the max batch size,
while v6e's much higher peak TFLOPS (918 vs v5e's 197) mainly buys higher
steps/sec at a GIVEN batch size, not a larger one.

| model | v5e (per chip, 16GB) | v5p (per chip, 95GB) | v6e (per chip, 32GB) |
|---|---|---|---|
| 1.18M | ~1024-2048 | ~4096-8192 | ~2048-4096 |
| 10.76M | ~512-1024 | ~2048-4096 | ~1024-2048 |
| 50.87M | ~256-512 | ~1024-2048 | ~512-1024 |

(×8 for the total across a full -8 slice.)

## Sequence packing

Every stage's dense mix has different `L` per entry (stage3: 36 to 150),
each currently trained one-at-a-time via weighted random trajectory
sampling. XLA compiles per unique shape, so naively this triggers a
recompile storm.

**How much this matters scales WITH model size**:
- At the current 165K/`d=64` scale, padding every entry up to one fixed
  max-`L` per stage (simplest option — pad + mask, one XLA compile per
  stage, 4 total for this config) wastes some FLOPs on padding, but the
  model is so cheap that the waste is irrelevant.
- At the 10M/50M tier, each token costs real FLOPs, so naive padding starts
  leaving real throughput on the table. True block-diagonal packing
  (concatenate several entries into one padded row, block-diagonal
  attention mask preventing cross-entry attention — mechanically
  straightforward since `chunk_mask_fb_traj` already produces a per-entry
  additive mask that composes onto a block-diagonal super-mask) becomes
  worth the added engineering effort at that scale.

Start with fixed-max-`L` padding at small scale; only build real packing
once/if the model size actually moves to the 10M+ tier.

## Precision (bf16)

TPU's native fast path, and doubles both memory headroom and MXU
throughput — more valuable as the model grows (at 165K params, fp32 vs
bf16 was a rounding error either way; at 50M params it's a real win).
**Caveat that doesn't go away with scale**: this task's success metric is
byte-EXACT match, not just low loss — verify bf16 doesn't quietly degrade
match% (rounding near decision boundaries) before trusting it for real
runs, rather than assuming a speed win is free.

## Other levers

- **Gradient checkpointing**: worth it at the 50M/`n_layers=12` tier if
  pushing batch size toward the top of its range; not worth it at 1M/10M,
  where activation memory isn't the binding constraint.
- **Parallelism**: data-parallel only, at every size considered here (even
  50M params is far too small to need model/tensor parallelism across 8
  cores) — replicate the model, shard the batch.

## Porting prerequisite (the actual blocker)

None of the above is reachable as-is. Current code is eager PyTorch
(`F.scaled_dot_product_attention`, Python-level trajectory sampling and
NumPy mask/batch construction every step) — a real TPU run needs either a
PyTorch/XLA (`torch_xla`) port or a JAX/Flax rewrite, with the batch/mask
construction moved OFF the hot path (precomputed/cached on host, fed via a
pipeline) since XLA is notoriously sensitive to host callbacks breaking up
the compiled step. This porting work is the prerequisite for exercising
ANY of the batch-size/model-size numbers above — it doesn't get easier or
harder based on which model size is chosen, so it should be scoped and
done once, independent of the size decision.
