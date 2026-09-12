"""
`hmn_tpu_sanity_w25_rope_jax_nocurr.py` — JAX port of
`hmn_tpu_sanity_w25_rope.py` (torch_xla), restructured per explicit
direction after `hmn_tpu_recall1024_jax_hopdrop.py`'s from-scratch
multi-stage curriculum proved hard to converge (stage 0 plateaued at
8-10% val MEAN through steps 50000-75000 before `tpu-v6e-8-1` was
preempted): **no curriculum** — one single stage, one weave_mix covering
MANY source lengths (`chunk_len` 8/16/32/64), warmup_lens, and anchor
positions all mixed together, so the model has to get "one step encode
decode" (single-chunk recall) working across the whole length range at
once rather than being walked up a ladder. Same architecture as the torch
reference (`d=128, n_layers=16, n_heads=8`, ~1.12M params — the actual
recall1024-target size, not the smaller `d=64/n_layers=8` architecture
`hmn_notags_w25_rope_jax_sanity_c8.py` already proved converges) — the
real open question this answers is whether the LARGER model converges on
the EASY single-chunk task at all, before ever touching multi-chunk
suffix recall.

JAX-specific notes: `kvmem/hmn_jax.py`'s `train_jax` never implements bf16
autocast at all (fp32 throughout — see its own module docstring), so the
torch reference's `no_autocast=True`/RoPE-bf16-NaN concern (CLAUDE.md bugs
6/7) simply doesn't apply here; not carried over. The reference's `B8`/
`B16` DSL suffix tokens (`repeat_batch`) are parsed by `parse_traj_dsl`
but never wired into `train_jax`'s training loop (dead value, confirmed by
reading `_build_trajectory`'s `_repeat_batch` — underscore-prefixed,
discarded) — dropped from this port's `_grid` rather than including a
token that would silently do nothing.

Explicit deltas from the torch reference, per direction:
  - `adapt_ema_alpha=0.8` (up from the reference's 0.5) — slower-reacting
    difficulty signal, less noisy re-weighting swings entry-to-entry.
  - `grad_checkpoint` not set at all (`kvmem.hmn_jax.build_model` has no
    such param in its signature — this file's architecture doesn't
    support/need it at this L; the torch reference's `grad_checkpoint=
    'block'` has no JAX equivalent wired into this port).
  - `B=16` — single-chunk (`n_chunks=1`) trajectories keep `L` small even
    at `chunk_len=64` (~130-190 tokens, nothing like the suffix-recall
    family's `L` in the thousands), so this is a deliberately conservative
    starting point ("smaller batch that fits") rather than a measured
    ceiling — no `bucket_lengths` here (can't: different `chunk_len`
    entries have different `w0`, `_make_train_step_bucket`'s single-
    shared-`w0` requirement is violated, same constraint documented in
    `hmn_tpu_recall1024_jax_adaptive_mix.py`'s own docstring) so `B` is
    the ONE value used for every entry in the stage — must fit the
    largest, not the average.

Run (never two jobs at once):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_tpu_sanity_w25_rope_jax_nocurr.py
"""


def _grid(chunk_len, warmup_lens, n_anchors, min_recall_len, weight=1.0):
    """Verbatim (minus the inert `rb_token` param) port of `hmn_tpu_sanity_
    w25_rope.py`'s own `_grid`."""
    entries = []
    for wl in warmup_lens:
        max_start = chunk_len - min_recall_len - wl
        if max_start < 0:
            continue
        if n_anchors == 1 or max_start == 0:
            starts = [0]
        else:
            starts = sorted(set(round(i * max_start / (n_anchors - 1)) for i in range(n_anchors)))
        for s in starts:
            entries.append(dict(weight=weight, dsl=f'E({chunk_len}) Q(0,1,{s},{wl})'))
    return entries


_ALL_MIX = (
    _grid(8, [2, 3, 4], n_anchors=4, min_recall_len=4)
    + _grid(16, [4, 6, 8], n_anchors=4, min_recall_len=4)
    + _grid(32, [8, 12, 16], n_anchors=4, min_recall_len=4)
    + _grid(64, [16, 24], n_anchors=4, min_recall_len=4)
)

hp = dict(
    d=128, n_layers=16, n_heads=8, V=271,
    lr_max=1e-4, wd=1e-5,
    warmup_steps=1000, log_every=500,
    rope=True, yarn=True,
    null_kv=True,
    rmsnorm=True,
    name='hmn_tpu_sanity_w25_rope_jax_nocurr', seed=48,

    adaptive=True,
    adapt_signal='val_match',
    adapt_temp=1.0,
    adapt_ema_alpha=0.8,  # slower EMA — per explicit direction
    adapt_floor=0.05,

    state_len=4, state_vocab_size=1,
    warmup_len=8,  # stage-level fallback, unused — every entry's DSL sets its own wl
    val_n_seqs=3,

    bucket_lengths=False,

    curriculum=[
        dict(n_chunks=1, chunk_len=8, B=16, n_steps=200000, eval_every=5000,
             early_stop_mean=80.0, weave_mix=_ALL_MIX),
    ],
)
