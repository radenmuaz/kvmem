"""
`hmn_notags_w25_rope_jax.py` — exact JAX port of `hmn_notags_w25_rope.py`
(torch/MPS), rewritten 2026-07-31 per explicit direction to revert all the
way back to the known-converging recipe (torch history: this exact config
reached val_mean=91.7% on its own chunk_len=8 stage, `logs/hmn_notags_w25_
rope/checkpoints/stage0_best.pt`) before any further architecture-scaling
or curriculum-redesign experiments. The version of this file that existed
before this rewrite was a cut-down LOCAL CPU SANITY DRAFT only (n_steps
2000-3000, no `adaptive`/`adapt_*` keys at all — i.e. a STATIC sampler,
not matching the reference) — this replaces it with the real thing.

Same model (`d=64, n_layers=8, n_heads=4, V=271` — the SMALL, already-
proven architecture, not the ~1.1M-param one used by every `hmn_tpu_
sanity_w25_rope_jax*`/`hmn_tpu_recall1024_jax*` config this session), same
hparams (`lr_max=1e-4, wd=1e-5, warmup_steps=1000`), same sampler
(`adaptive=True, adapt_signal='val_match', adapt_temp=1.0, adapt_ema_
alpha=0.5` — the ORIGINAL default, not this session's later "slower EMA"
0.8 experiment), same curriculum shape (`chunk_len` 8->16->32->64,
`early_stop_mean=80.0`, identical `n_steps`/`eval_every`/`B` per stage,
identical `_grid` anchor sweep). Only difference from the torch reference:
runs on TPU via `kvmem.hmn_jax` instead of MPS via `kvmem.hmn` —
`state_vocab_size=2`/`yarn=True` match torch's own `train()` defaults for
this exact config (`hp.get('yarn', True)` — not set explicitly in the
original file, but that's what it actually built with). The reference's
`use_actual_argmax`/`wrong_token_weight` keys are refine-round-only
settings (`hmn_jax.py` never supports refine rounds — `n_refine` is
always 0) and the `rb_token` DSL suffix (`repeat_batch`) is parsed but
never wired into `train_jax`'s loop (confirmed by reading `_build_
trajectory`'s own `_repeat_batch`, underscore-prefixed/discarded) — both
omitted here as genuinely inert for this file's scope, not silently
dropped hparams.

Deploy to exactly ONE TPU (no parallel ablation variant) — this run's job
is to verify the known-good recipe reproduces on TPU, not to explore
further.

Run (never two jobs at once):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_notags_w25_rope_jax.py
"""
import math


def _grid(chunk_len, warmup_lens, n_anchors, min_recall_len, weight=1.0, min_warmup_frac=0.0):
    if min_warmup_frac > 0:
        min_wl = math.ceil(chunk_len * min_warmup_frac)
        bad = [wl for wl in warmup_lens if wl < min_wl]
        assert not bad, (
            f'_grid(chunk_len={chunk_len}, ...): warmup_lens {bad} are below the '
            f'min_warmup_frac={min_warmup_frac} floor ({min_wl}) — pass an already-'
            f'filtered warmup_lens list instead of relying on a runtime filter')
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


hp = dict(
    d=64, n_layers=8, n_heads=4, V=271,
    lr_max=1e-4, wd=1e-5,
    warmup_steps=1000, log_every=2000,
    rope=True, yarn=True,
    null_kv=True,
    rmsnorm=True,
    name='hmn_notags_w25_rope_jax', seed=48,

    adaptive=True,
    adapt_signal='val_match',
    adapt_temp=1.0,
    adapt_ema_alpha=0.5,
    adapt_floor=0.05,

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    val_n_seqs=3,

    bucket_lengths=False,

    curriculum=[
        dict(n_chunks=1, chunk_len=8, B=16, n_steps=480000, eval_every=24000, early_stop_mean=80.0,
             weave_mix=_grid(8, [2, 3, 4], n_anchors=4, min_recall_len=4, min_warmup_frac=0.25)),

        dict(n_chunks=1, chunk_len=16, B=12, n_steps=720000, eval_every=36000, early_stop_mean=80.0,
             weave_mix=(
                 _grid(16, [4, 6, 8], n_anchors=4, min_recall_len=4, min_warmup_frac=0.25)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, weight=0.5, min_warmup_frac=0.25)
             )),

        dict(n_chunks=1, chunk_len=32, B=6, n_steps=480000, eval_every=24000, early_stop_mean=80.0,
             weave_mix=(
                 _grid(32, [8, 12, 16], n_anchors=4, min_recall_len=4, min_warmup_frac=0.25)
                 + _grid(16, [4, 6, 8], n_anchors=2, min_recall_len=4, weight=0.5, min_warmup_frac=0.25)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, weight=0.5, min_warmup_frac=0.25)
             )),

        dict(n_chunks=1, chunk_len=64, B=4, n_steps=720000, eval_every=36000, early_stop_mean=80.0,
             weave_mix=(
                 _grid(64, [16, 24], n_anchors=4, min_recall_len=4, min_warmup_frac=0.25)
                 + _grid(32, [8, 12, 16], n_anchors=2, min_recall_len=4, weight=0.5, min_warmup_frac=0.25)
                 + _grid(16, [4, 6, 8], n_anchors=2, min_recall_len=4, weight=0.5, min_warmup_frac=0.25)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, weight=0.5, min_warmup_frac=0.25)
             )),
    ],
)
