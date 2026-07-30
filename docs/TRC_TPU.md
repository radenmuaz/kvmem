# TRC / TPU setup estimate

**Update (2026-07-30): the port described as a prerequisite below has now
been done**, driven by a real scale-up experiment (`kvmem/configs/
hmn_tpu_recall1024_flat.py` — 1024-byte perfect recall from any source
index, 1.12M params, `d=128/n_layers=16/n_heads=8`; see CLAUDE.md's
scale-up entry and `kvmem/gate_check.py` for the verification gates run
against it). `kvmem/hmn.py`'s `train()` now accepts `--device tpu`
end-to-end: length bucketing (`hp['bucket_lengths']`, `_bucket_ceilings`/
`_pad_mask_to`/`_pad_tok_to`), per-bucket batch sizing (`token_budget`/
`attn_sq_budget`), `torch_xla.sync()` + bf16 autocast + host-sync-throttled
loss logging, a CPU eval replica (autoregressive decode never ported to
XLA — see "Sequence packing" below, now corrected, and "Other levers"),
and a vectorized (no more per-row Python loop) `make_batch_tagged`. All
opt-in via `hp['bucket_lengths']`/`device_str='tpu'` — every existing
CPU/MPS config is untouched. The rest of this doc is kept as the original
pre-port estimate PLUS corrections/lessons learned marked inline — read
the "Update" callouts, not the surrounding prose, for what's now known
rather than estimated.

Original framing (pre-port, kept for context): a planning estimate for
if/when TPU Research Cloud (TRC) access is used for this project, written
before any actual porting work started.

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

## TRC's actual tiers — CONFIRMED: `tpu1` is v5e, single chip

**Update (2026-07-30): confirmed, not assumed.** `gcloud compute tpus
tpu-vm describe tpu1 --zone=europe-west4-b` reports `acceleratorType:
v5litepod-1`, `runtimeVersion: v2-alpha-tpuv5-lite`. This is **one v5e
chip** (`torch_xla.runtime.global_runtime_device_count()` returns `1`,
`xm.get_xla_supported_devices()` returns `['xla:0']`), NOT a `-8` slice —
the ×8 multi-chip table further down (batch size, "replicate the model,
shard the batch") does not apply to this VM at all; there is nothing to
shard across. Host: 24 vCPU, 47GB RAM, 77GB free disk. If a `-8` grant is
obtained later, revisit that table; until then, every number in this doc
should be read as PER-CHIP with no multiplication.

The tier-uncertainty framing below (v5e vs v5p vs v6e) is now moot for
`tpu1` specifically — kept for reference in case a different/larger grant
is obtained later:

**Still unconfirmed for any OTHER grant** — TRC's public pages don't
publish a fixed tier list (quota is "as listed in the [applicant's]
activation email," per-applicant, not a published default). Per explicit
instruction, this doc assumes v5e/v5p/v6e (Trillium) — Google Cloud's
current GA generations — rather than the old (2022-2023-era, now likely
stale) v2/v3/v4 default. **The authoritative source is still your own TRC
activation email or `trc-support@google.com`.**

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

**Update (2026-07-30): the block-diagonal packing recommendation below was
WRONG, and has been reversed** — see "Correction" immediately below. The
original text is kept struck-through-in-spirit (not deleted, so the
reasoning error is on record) followed by the actual implemented approach.

Every stage's dense mix has different `L` per entry (stage3: 36 to 150),
each currently trained one-at-a-time via weighted random trajectory
sampling. XLA compiles per unique shape, so naively this triggers a
recompile storm.

~~**How much this matters scales WITH model size**:~~
~~- At the current 165K/`d=64` scale, padding every entry up to one fixed
  max-`L` per stage (simplest option — pad + mask, one XLA compile per
  stage, 4 total for this config) wastes some FLOPs on padding, but the
  model is so cheap that the waste is irrelevant.~~
~~- At the 10M/50M tier, each token costs real FLOPs, so naive padding starts
  leaving real throughput on the table. True block-diagonal packing
  (concatenate several entries into one padded row, block-diagonal
  attention mask preventing cross-entry attention — mechanically
  straightforward since `chunk_mask_fb_traj` already produces a per-entry
  additive mask that composes onto a block-diagonal super-mask) becomes
  worth the added engineering effort at that scale.~~

~~Start with fixed-max-`L` padding at small scale; only build real packing
once/if the model size actually moves to the 10M+ tier.~~

**Correction: block-diagonal packing is wrong for this codebase at ANY
model size, not just small ones.** Attention here is dense `O(L^2)` with an
arbitrary `[L,L]` additive mask (`chunk_mask_fb_traj`'s output — not a
causal-only pattern a flash-attention kernel could exploit for free).
Concatenating `K` mix entries into one packed row of length `K*L` costs
`K^2` attention work where `K` separate batch rows cost `K*(attention work
for one entry)` — packing is a net FLOP *loss*, not a saving, regardless of
model size. It only pays off with a block-sparse/varlen attention kernel
that skips the cross-entry blocks entirely, and `torch_xla`'s available
flash-attention-style kernels do not accept an arbitrary dense additive
bias, so that route is closed for this project's masking scheme without a
custom kernel (out of scope).

**What's actually implemented (verified working, `kvmem/hmn.py`)**: bucket
+ pad + widen the batch axis, exactly as the original small-scale option
above described, but as the ONLY approach (not a small-scale stopgap) —
`hp['bucket_lengths']=True` groups a weave_mix's distinct `L` values into
`<=hp['max_shape_buckets']` ceiling shapes (`_bucket_ceilings`, a weighted
k-segment DP minimizing `sum(ceiling^2 * weight)` — squared because
attention cost scales with `L^2`, so this is the FLOP-correct objective),
pads every trajectory's mask/tokens up to its assigned ceiling ONCE at
stage setup (`_pad_mask_to`/`_pad_tok_to`, not per-step), and derives a
per-bucket batch size from TWO memory ceilings (`token_budget`, `B ~ 1/Lb`,
and `attn_sq_budget`, `B ~ 1/Lb^2` — see "Other levers" below for why the
second one turned out to be load-bearing, not optional). Verified: for
`hmn_tpu_recall1024_flat.py`'s 16-entry, 8-distinct-length mix, this
produces exactly 8 buckets with 0% padding waste (every distinct L already
gets its own bucket at `max_shape_buckets=8`) — bucketing only pads/wastes
FLOPs once a mix has MORE distinct lengths than the bucket budget allows,
and even then only rounds up to the nearest observed length among the
survivors, never to an arbitrary power of 2.

## `repeat_batch` under sequence packing — global, not per-sample

`repeat_batch` (`kvmem/hmn.py`'s `hp['repeat_batch']`, current CPU/MPS
mechanism — see CLAUDE.md's `repeat_batch` ablation entry for why it
exists: it fixes a training-loss plateau by taking N gradient steps on the
same sampled batch before resampling) is already, necessarily, a **single
counter shared across the whole `[B, L]` batch**, not something that could
vary per row: every step processes one `tok_t` tensor from one sampled
trajectory/DSL entry, so all `B` rows share one shape already, and there's
only one place a "resample or reuse" decision can live (`_cached_repeat_
left`/`_cached_batch` in the `weave_mix` path, `_cached_base_np` in the
`chain_steps`/`traj_mix` paths — all three gate on `(local_step - 1) %
repeat_batch == 0`). This generalizes cleanly to the block-diagonal packing
described above: once several DSL entries are concatenated into one packed
row, that row is still one tensor, one forward/backward per step — so
`repeat_batch` would still have to be one global counter for the whole
packed super-batch, not independent per sub-entry packed within it. Any
future packing implementation should keep this single-counter structure
rather than trying to give each packed sub-entry its own repeat count.

**Refine ops (`n_refine>0`) do NOT let `repeat_batch` cache the argmax
feedback — and must not.** `repeat_batch` only caches the raw
ground-truth bytes (`_cached_base_np`, pre-argmax). Every step, regardless
of where it falls in the `repeat_batch` window, still runs a **fresh**
`with torch.no_grad(): logits_1 = model(tok_t, mask_t)` pass to get the
model's CURRENT argmax before `_fill_argmax_fb` bakes it into the batch —
because the argmax is a function of the model's current weights, which
change every step, caching it across `repeat_batch` steps would train
rounds 1+ on increasingly stale feedback from a model that no longer
exists by the time step 2..N of the window runs. Net effect: any
trajectory with `n_refine>0` costs **2 forward passes per step, every
step**, independent of `repeat_batch` — `repeat_batch` only amortizes the
CPU-side `make_batch_tagged`/mask-construction cost (the actual host-
overhead bottleneck flagged in "The core situation" above), never the
argmax pass itself.

**Real cost under XLA specifically**: that extra no-grad argmax pass is a
second host↔device round trip per step unless it's fused into the SAME
compiled step function as the real forward/backward. If the port leaves it
as a separate Python-level `model(...)` call (today's eager-PyTorch
structure), any refine-containing entry pays 2x the per-step XLA dispatch
overhead — directly undermining the "reduce host-side/XLA-call overhead"
goal this whole doc is organized around. Any TPU port needs to fuse the
argmax pass and the real pass into one compiled graph (e.g. a single
`jax.jit`/`torch_xla` step function computing both), not port the current
two-separate-Python-calls structure as-is.

## Precision (bf16)

TPU's native fast path, and doubles both memory headroom and MXU
throughput — more valuable as the model grows (at 165K params, fp32 vs
bf16 was a rounding error either way; at 50M params it's a real win).
**Caveat that doesn't go away with scale**: this task's success metric is
byte-EXACT match, not just low loss — verify bf16 doesn't quietly degrade
match% (rounding near decision boundaries) before trusting it for real
runs, rather than assuming a speed win is free.

## Other levers

- **Gradient checkpointing**: ~~worth it at the 50M/`n_layers=12` tier if
  pushing batch size toward the top of its range; not worth it at 1M/10M,
  where activation memory isn't the binding constraint.~~ **Correction
  (2026-07-30): WRONG — this significantly underweighted `L`.** The
  original claim implicitly assumed activation memory is dominated by the
  `B*L*d` term (linear in L), true for the short sequences (`L<=150`) this
  doc was originally scoped around. It is NOT true once `L` grows into the
  thousands (exactly the `hmn_tpu_recall1024_flat.py` regime, `L` up to
  2128): the `O(B*H*L^2)` attention-score-matrix term, retained per layer
  for backward, dominates instead and grows QUADRATICALLY with `L`.
  Measured directly on `tpu1`: at `d=128/n_layers=16` (1.12M params — the
  "small" end of this doc's own size ladder) with `B=64, L=1232`,
  training WITHOUT `grad_checkpoint` hit a hard HBM OOM — `RuntimeError:
  ... RESOURCE_EXHAUSTED: ... Used 52.85G of 15.75G hbm. Exceeded hbm
  capacity by 37.10G` — and the requested amount matches
  `B*H*L^2*n_layers*4bytes` almost exactly (`64*8*1232^2*16*4 ≈ 52.8G`),
  confirming the mechanism: without checkpointing, ALL 16 layers'
  attention matrices were being retained simultaneously for backward.
  Setting `grad_checkpoint='block'` (`HMNModel`'s existing model-depth
  checkpointing — checkpoints each `SingleAttnBlock`, so backward
  recomputes one layer's activations at a time instead of retaining all of
  them) is the fix, and made this exact case trainable. **Revised
  guidance: gradient checkpointing is worth it whenever `L` is large
  relative to `d` — a function of sequence length, not model/param size —
  and should be treated as load-bearing (verify via a real forward+backward
  step before trusting a batch size), not merely a batch-size-maximizing
  nice-to-have, for any config with `L` in the thousands regardless of how
  small the model itself is.**
  - **A second, real gotcha surfaced by this**: `torch.utils.checkpoint.
    checkpoint`'s default non-reentrant path (`use_reentrant=False`, this
    project's default) calls `_get_device_module(device_type)` ->
    `getattr(torch, device_type)` to save/restore per-device RNG state —
    this works for `'cuda'`/`'mps'`/`'cpu'` (real submodules of `torch`)
    but raises `AttributeError: module 'torch' has no attribute 'xla'` for
    XLA tensors, since `torch_xla` does not register itself under `torch.
    xla`. Fix: `torch_xla.utils.checkpoint.checkpoint` is torch_xla's own
    checkpoint implementation (reentrant-based, never calls that lookup) —
    `kvmem/hmn.py`'s `_ckpt` wrapper now dispatches to it when any argument
    is an XLA tensor, falling back to the original `torch.utils.checkpoint`
    call (`use_reentrant=False`, byte-for-byte unchanged) everywhere else.
    Any future torch_xla port that uses `torch.utils.checkpoint` directly
    (rather than through a dispatch wrapper) will hit this.
- **Parallelism**: data-parallel only, at every size considered here (even
  50M params is far too small to need model/tensor parallelism across 8
  cores) — replicate the model, shard the batch. Moot for `tpu1` itself
  (single chip, see the tier section above) but still the right guidance
  for a `-8` grant.
- **Mixing CPU and XLA `.backward()` calls in one process**: a real bug hit
  while building the verification gates (`kvmem/gate_check.py`) — running a
  CPU `train()` call followed by a TPU `train()` call in the SAME Python
  process crashes the SECOND (TPU) call with `RuntimeError: 0 <=
  device.index() && device.index() < ... device_ready_queues_.size()
  INTERNAL ASSERT FAILED`. Root cause: PyTorch's autograd Engine singleton
  sizes its `device_ready_queues_` when first used (at the first
  `.backward()` call); if that first call is a plain CPU backward, the
  engine never learns about the XLA device registered afterward. Not
  specific to this codebase — any script comparing CPU and TPU training in
  one process will hit this. Fix: one device per process (`gate_check.py`'s
  `gate3_cpu`/`gate3_tpu`/`gate3_compare` are three separate `python3 -m`
  invocations for exactly this reason, compared only via their logged
  output on disk, never in-process).

## TPU VM access confirmed (2026-07-30)

`gcloud compute tpus tpu-vm ssh tpu1 --zone=europe-west4-b` logs in successfully (host `t1v-n-d023ff26-w-0`, user `muaz`, Python 3.10.12). `torch_xla` is already installed on the VM and detects the TPU (`libtpu.so and TPU device found, setting PJRT_DEVICE=TPU`) — so the environment itself is ready. This does not change the porting-prerequisite conclusion below: `kvmem/hmn.py` is still eager PyTorch with Python-level trajectory sampling/NumPy mask construction on the hot path, so none of this doc's batch-size/model-size estimates are exercisable yet. Slice topology (chip count) not yet checked.

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
