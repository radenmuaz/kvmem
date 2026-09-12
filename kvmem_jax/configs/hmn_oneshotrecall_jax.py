"""
`hmn_oneshotrecall_jax.py` — **new ablation lineage, renamed from the
`hmn_tpu_sanity_w25_rope_jax*` naming** ("oneshotrecall" describes what this
config actually is: single-chunk, one-shot recall — `n_chunks=1` the whole
time, `chunk_len` staged 8->16->32->64). Same curriculum shape as
`hmn_tpu_sanity_w25_rope_jax_mlp.py`, but two changes, per explicit
direction:

1. **STATE-capacity ablation, not FFN-capacity**: `state_len=8` (up from 4)
   and `state_vocab_size=2` (up from 1) — a bigger/richer memory register,
   testing capacity via the STATE channel itself rather than the model's
   FFN. Motivated directly by the MLP experiment's negative result
   (`hmn_tpu_recall1024_jax_incremental_mlp.py` collapsed at chunk_len
   32/64 just as badly as the original no-MLP run, 2.4%/3.2% vs 4.6%/3.1%)
   — that ruled out FFN capacity as the fix, so this tries the OTHER
   capacity lever, deliberately kept separate from the FFN change:

2. **`block_type='single_attn'` — MLP explicitly DISABLED** (no `d_ff`
   key at all), so this run isolates STATE capacity as the only variable,
   not conflated with the (already-tested, already-negative) FFN change.

3. **Batch size x4** across every stage (16->64, 12->48, 6->24, 4->16),
   per explicit direction — the TPU was severely underutilized at the
   original batch sizes (v4-8 HBM headroom was nowhere near the limit).
   Everything else (lr_max, curriculum shape, adaptive reweighting)
   unchanged — this is a pure throughput lever, watch whether convergence
   speed/quality holds, degrades, or improves at the larger batch.

Trains from scratch (no warm-start) — `state_len=8` doesn't shape-match any
existing checkpoint (`state_len=4`), so there's nothing to warm-start from.

Run (never two jobs at once; run from the repo root so relative log/config
paths resolve):
    python3 -m kvmem_jax.hmn_jax --config kvmem_jax/configs/hmn_oneshotrecall_jax.py
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
    # block_type left at its default ('single_attn') — MLP disabled, no d_ff key.
    lr_max=1e-4, wd=1e-5,
    warmup_steps=1000, log_every=500,
    rope=True, yarn=True,
    null_kv=True,
    rmsnorm=True,
    name='hmn_oneshotrecall_jax', seed=48,

    adaptive=True,
    adapt_signal='val_match',
    adapt_temp=1.0,
    adapt_ema_alpha=0.8,

    state_len=8, state_vocab_size=2,  # THE ablation: bigger STATE register, not FFN
    warmup_len=8,
    val_n_seqs=3,

    bucket_lengths=False,

    curriculum=[
        dict(n_chunks=1, chunk_len=8, B=64, n_steps=100000, eval_every=5000,
             early_stop_mean=80.0,
             weave_mix=_grid(8, [2, 3, 4], n_anchors=4, min_recall_len=4)),

        dict(n_chunks=1, chunk_len=16, B=48, n_steps=100000, eval_every=5000,
             early_stop_mean=80.0,
             weave_mix=(
                 _grid(16, [4, 6, 8], n_anchors=4, min_recall_len=4)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, weight=0.5)
             )),

        dict(n_chunks=1, chunk_len=32, B=24, n_steps=100000, eval_every=5000,
             early_stop_mean=80.0,
             weave_mix=(
                 _grid(32, [8, 12, 16], n_anchors=4, min_recall_len=4)
                 + _grid(16, [4, 6, 8], n_anchors=2, min_recall_len=4, weight=0.5)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, weight=0.5)
             )),

        dict(n_chunks=1, chunk_len=64, B=16, n_steps=100000, eval_every=5000,
             early_stop_mean=80.0,
             weave_mix=(
                 _grid(64, [16, 24], n_anchors=4, min_recall_len=4)
                 + _grid(32, [8, 12, 16], n_anchors=2, min_recall_len=4, weight=0.5)
                 + _grid(16, [4, 6, 8], n_anchors=2, min_recall_len=4, weight=0.5)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, weight=0.5)
             )),
    ],
)
