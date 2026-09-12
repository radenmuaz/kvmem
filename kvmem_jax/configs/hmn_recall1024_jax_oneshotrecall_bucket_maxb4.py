"""
`hmn_recall1024_jax_oneshotrecall_bucket_maxb4.py` — gets past the stage-5
compile wall that killed `hmn_recall1024_jax_oneshotrecall_b128.py` (died
3 times entering stage 5's 32-distinct-shape compile at chunk_len=32,
across B=64/256/512 — a real compile-time host-memory ceiling, not a
training-time issue). Enables `bucket_lengths=True`: instead of one
compiled step_fn PER distinct shape (32 of them), trajectories sharing a
stage get grouped into <= `max_shape_buckets` padded-length buckets, each
sharing ONE compiled step_fn — drastically fewer simultaneous compiles.

**This node's lever: reduce `max_shape_buckets` to 4** (down from the
default 8) — fewer buckets, more padding waste per bucket, but the
smallest compile burden of the two variants tried in parallel (the sibling
config, `_bucket_b32.py`, keeps the default bucket count but reduces
batch size instead).

Continues from `hmn_recall1024_jax_oneshotrecall_b128.py`'s own
`stage4_best.pt` (val_mean=71.1% at chunk_len=16 — the best Phase-B-so-far
result of any capacity ablation tried) — only stages 5-7 included
(chunk_len=32, chunk_len=64, and the hop_drop_prob=0.0 stage); stage 8
(hop_drop_prob=0.2) is deliberately omitted here since `bucket_lengths` and
`hop_drop_prob>0` are asserted incompatible in `kvmem_jax/hmn_jax.py` — not
needed to answer the immediate chunk_len-collapse question anyway.

Run (run from the repo root so relative log/config paths resolve):
    python3 -m kvmem_jax.hmn_jax --config kvmem_jax/configs/hmn_recall1024_jax_oneshotrecall_bucket_maxb4.py
"""


def _grid_stitch(n_chunks, chunk_len, warmup_lens, min_recall_len=None):
    total = n_chunks * chunk_len
    min_recall_len = min_recall_len if min_recall_len is not None else max(4, chunk_len // 2)
    mix = []
    for wl in warmup_lens:
        for a in range(n_chunks):
            anchor = a * chunk_len
            if total - anchor - wl < min_recall_len:
                continue
            if n_chunks > 1:
                dsl = f'E({chunk_len}) E{n_chunks - 1} Q(0,{n_chunks},{anchor},{wl})'
            else:
                dsl = f'E({chunk_len}) Q(0,{n_chunks},{anchor},{wl})'
            mix.append(dict(weight=1.0, dsl=dsl))
    return mix


hp = dict(
    d=128, n_layers=16, n_heads=8, V=271,
    lr_max=1e-4, wd=1e-5,
    warmup_steps=1000, log_every=500,
    rope=True, yarn=True,
    null_kv=True,
    rmsnorm=True,
    name='hmn_recall1024_jax_oneshotrecall_bucket_maxb4', seed=48,

    adaptive=True,
    adapt_signal='val_match',
    adapt_temp=1.0,
    adapt_ema_alpha=0.5,
    adapt_floor=0.05,

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    val_n_seqs=3,

    grad_checkpoint='block',  # required — without it, all 16 layers' attention matrices are
                              # retained for backward simultaneously; a real RESOURCE_EXHAUSTED
                              # HBM OOM (Used 33.19G of 30.75G) was hit at Lb~784-1168 without this
    bucket_lengths=True,
    max_shape_buckets=4,  # THE lever this node tests: fewer buckets, smallest compile burden
    eval_combinatorial_hops=True,
    warm_start_from_best=True,
    pretrained_ckpt='kvmem_jax/logs/hmn_recall1024_jax_oneshotrecall_b128/checkpoints/stage4_best.pt',

    curriculum=[
        dict(n_chunks=16, chunk_len=32, B=64, n_steps=80000, eval_every=5000,
             hops=-1, early_stop_mean=60.0,
             weave_mix=_grid_stitch(16, 32, [8, 16])),
        dict(n_chunks=16, chunk_len=64, B=64, n_steps=100000, eval_every=5000,
             hops=-1, early_stop_mean=55.0,
             weave_mix=_grid_stitch(16, 64, [16, 32])),
        dict(n_chunks=16, chunk_len=64, B=64, n_steps=80000, eval_every=5000,
             hops=-1, enc_hops=4, hop_drop_prob=0.0, early_stop_mean=50.0,
             weave_mix=_grid_stitch(16, 64, [16, 32])),
    ],
)
