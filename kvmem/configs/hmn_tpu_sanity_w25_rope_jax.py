"""
`hmn_tpu_sanity_w25_rope_jax.py` — direct JAX port of `hmn_tpu_sanity_
w25_rope.py` (torch_xla), a deliberate step BACK from `hmn_tpu_sanity_
w25_rope_jax_nocurr.py`'s single-stage all-lengths-mixed design (which
completed its 200000-step budget at only best=17.0% val MEAN, qualitative
eyeball showing partial-then-collapse-to-repetition generation — not a
clean pass). This restores the ORIGINAL, already-proven staged curriculum
exactly (`chunk_len` 8 -> 16 -> 32 -> 64, each stage gated by `early_stop_
mean=80.0`, each later stage rehearsing earlier lengths at `weight=0.5`)
at the CURRENT ~1.1M-param architecture (`d=128, n_layers=16, n_heads=8`
— this was already the torch reference's own size, unchanged from
`_nocurr`; the actual difference from `_nocurr` is curriculum STAGING
returning, not a model-size change).

Motivation, stated directly: the mixed/no-curriculum design was too big a
jump — go back to the smallest proven-working shape (this exact config's
own torch history: `hmn_notags_w25`/`hmn_notags_w25_rope` converged
cleanly at `d=64/n_layers=8`) and re-verify it holds at the larger target
architecture ONE stage at a time, rather than asking the model to solve
every length simultaneously with no gating checkpoint in between.

Same `_grid` helper as the torch reference (verbatim, minus the inert
`rb_token`/JAX-unused `repeat_batch` param — see `_nocurr`'s own docstring
for why that token does nothing in `train_jax`).

Run (never two jobs at once):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_tpu_sanity_w25_rope_jax.py
"""


def _grid(chunk_len, warmup_lens, n_anchors, min_recall_len, weight=1.0):
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
    d=128, n_layers=16, n_heads=8, V=271,
    lr_max=1e-4, wd=1e-5,
    warmup_steps=1000, log_every=500,
    rope=True, yarn=True,
    null_kv=True,
    rmsnorm=True,
    name='hmn_tpu_sanity_w25_rope_jax', seed=48,

    adaptive=True,
    adapt_signal='val_match',
    adapt_temp=1.0,
    adapt_ema_alpha=0.8,  # slower/less reactive EMA — the default 0.5 looked too aggressive,
                          # oscillating 14-40% eval-to-eval rather than trending cleanly
    adapt_floor=0.05,

    state_len=4, state_vocab_size=1,
    warmup_len=8,
    val_n_seqs=3,

    bucket_lengths=False,

    curriculum=[
        dict(n_chunks=1, chunk_len=8, B=16, n_steps=100000, eval_every=5000,
             early_stop_mean=80.0,
             weave_mix=_grid(8, [2, 3, 4], n_anchors=4, min_recall_len=4)),

        dict(n_chunks=1, chunk_len=16, B=12, n_steps=100000, eval_every=5000,
             early_stop_mean=80.0,
             weave_mix=(
                 _grid(16, [4, 6, 8], n_anchors=4, min_recall_len=4)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, weight=0.5)
             )),

        dict(n_chunks=1, chunk_len=32, B=6, n_steps=100000, eval_every=5000,
             early_stop_mean=80.0,
             weave_mix=(
                 _grid(32, [8, 12, 16], n_anchors=4, min_recall_len=4)
                 + _grid(16, [4, 6, 8], n_anchors=2, min_recall_len=4, weight=0.5)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, weight=0.5)
             )),

        dict(n_chunks=1, chunk_len=64, B=4, n_steps=100000, eval_every=5000,
             early_stop_mean=80.0,
             weave_mix=(
                 _grid(64, [16, 24], n_anchors=4, min_recall_len=4)
                 + _grid(32, [8, 12, 16], n_anchors=2, min_recall_len=4, weight=0.5)
                 + _grid(16, [4, 6, 8], n_anchors=2, min_recall_len=4, weight=0.5)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, weight=0.5)
             )),
    ],
)
