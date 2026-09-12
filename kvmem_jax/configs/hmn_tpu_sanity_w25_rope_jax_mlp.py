"""
`hmn_tpu_sanity_w25_rope_jax_mlp.py` — **kvmem_mlp fork, Experiment 1 (MLP
variant)**, forked from `kvmem/configs/hmn_tpu_sanity_w25_rope_jax.py`
(2026-09-06). Identical curriculum, identical `d=128, n_layers=16, n_heads=8`
attention depth — the ONE change is `block_type='attn_mlp'` + `d_ff=256`
(2x-d FFN expansion — trimmed down from an initial 4x-d=512 pass, judged
too large a jump), adding a feed-forward sublayer to every block that the
original `single_attn` architecture doesn't have at all.

**Why this exists**: Experiment 2 (the original `single_attn`,
`kvmem/configs/hmn_tpu_recall1024_jax_incremental.py`) collapsed hard once
`chunk_len` grew past 16 — stage 5 (chunk_len=32) landed at 4.6% MEAN
against a 60% gate, stage 6 (chunk_len=64) at 3.1% against a 55% gate, a
step-change failure after stages 0-4 (n_chunks scaling) had worked cleanly.
Hypothesis: `single_attn`'s `x = x + attn(norm(x))` — one attention op per
layer, NO feed-forward/MLP sublayer anywhere in the model — is genuinely
undercapacity for this task once `chunk_len` grows, not a
data/curriculum/masking problem. This config re-runs Experiment 1 (the
easy, single-chunk, chunk_len-only-varies stage) with an FFN added, as the
first half of the fork's own two-experiment redo — Experiment 2's MLP
counterpart (`hmn_tpu_recall1024_jax_incremental_mlp.py`) warm-starts from
this config's own winning checkpoint, exactly mirroring how the original
Experiment 2 warm-started from the original Experiment 1.

Same `_grid` helper as the original (verbatim).

Run (never two jobs at once; run from the repo root so relative log/config
paths resolve):
    python3 -m kvmem_mlp.hmn_jax --config kvmem_mlp/configs/hmn_tpu_sanity_w25_rope_jax_mlp.py
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
    block_type='attn_mlp', d_ff=256,  # THE ablation: FFN added, everything else unchanged
    lr_max=1e-4, wd=1e-5,
    warmup_steps=1000, log_every=500,
    rope=True, yarn=True,
    null_kv=True,
    rmsnorm=True,
    name='hmn_tpu_sanity_w25_rope_jax_mlp', seed=48,

    adaptive=True,
    adapt_signal='val_match',
    adapt_temp=1.0,
    adapt_ema_alpha=0.8,

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
