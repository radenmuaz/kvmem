"""
`hmn_recall1024_jax_oneshotrecall.py` — the multi-chunk stitch/recall1024
redo built on `hmn_oneshotrecall_jax.py`'s STATE-capacity ablation
(`state_len=8, state_vocab_size=2`, `block_type='single_attn'` — MLP
disabled). Same 9-stage curriculum shape as `hmn_tpu_recall1024_jax_
incremental_mlp.py` (Phase A: n_chunks 2->4->8->16 at chunk_len=8; Phase B:
chunk_len 16->32->64 at n_chunks=16; Phase C: enc_hops/hop_drop_prob
introduced), warm-started from `hmn_oneshotrecall_jax.py`'s own
`stage3_best.pt` instead of either prior baseline (the plain no-MLP
recall1024 run, which collapsed at chunk_len 32/64 at 4.6%/3.2% MEAN, or
the MLP variant, which collapsed just as badly at 2.4%/3.2% — this run
tests whether a bigger STATE register succeeds where more FFN capacity
did not).

**Batch size x4** across every stage (matching `hmn_oneshotrecall_jax.py`'s
own throughput change — TPU was severely underutilized at the original
sizes): Phase A 16->64, Phase B/C 8->32. Everything else (lr_max,
`_grid_stitch` anchor sweep, early_stop_mean gates, enc_hops/hop_drop_prob
schedule) unchanged from the MLP incremental config.

Run (never two jobs at once; do not run before
`hmn_oneshotrecall_jax.py`'s stage3_best.pt actually exists — run from the
repo root so relative log/config paths resolve):
    python3 -m kvmem_jax.hmn_jax --config kvmem_jax/configs/hmn_recall1024_jax_oneshotrecall.py
"""


def _grid_stitch(n_chunks, chunk_len, warmup_lens, min_recall_len=None):
    """One anchor per chunk index (the start of each chunk 0..n_chunks-1),
    per warmup_len — guarantees every possible "how many chunks left to
    recall" case is represented, from n_chunks (anchor in chunk 0) down to
    a single partial chunk (anchor in the last chunk)."""
    total = n_chunks * chunk_len
    min_recall_len = min_recall_len if min_recall_len is not None else max(4, chunk_len // 2)
    mix = []
    for wl in warmup_lens:
        for a in range(n_chunks):
            anchor = a * chunk_len
            if total - anchor - wl < min_recall_len:
                continue  # anchor too close to the true end for even min_recall_len bytes
            if n_chunks > 1:
                dsl = f'E({chunk_len}) E{n_chunks - 1} Q(0,{n_chunks},{anchor},{wl})'
            else:
                dsl = f'E({chunk_len}) Q(0,{n_chunks},{anchor},{wl})'
            mix.append(dict(weight=1.0, dsl=dsl))
    return mix


hp = dict(
    d=128, n_layers=16, n_heads=8, V=271,
    # block_type left at its default ('single_attn') — MLP disabled, no d_ff key.
    lr_max=1e-4, wd=1e-5,
    warmup_steps=1000, log_every=500,
    rope=True, yarn=True,
    null_kv=True,
    rmsnorm=True,
    name='hmn_recall1024_jax_oneshotrecall', seed=48,

    adaptive=True,
    adapt_signal='val_match',
    adapt_temp=1.0,
    adapt_ema_alpha=0.5,
    adapt_floor=0.05,

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    val_n_seqs=3,

    bucket_lengths=False,
    eval_combinatorial_hops=True,  # inert (no-op) until stage H/I set enc_hops!=-1
    warm_start_from_best=True,
    pretrained_ckpt='kvmem_jax/logs/hmn_oneshotrecall_jax/checkpoints/stage3_best.pt',

    curriculum=[
        # --- A: n_chunks alone, chunk_len fixed at 8, hops=-1 ---
        dict(n_chunks=2, chunk_len=8, B=64, n_steps=60000, eval_every=5000,
             hops=-1, early_stop_mean=80.0,
             weave_mix=_grid_stitch(2, 8, [2, 4])),
        dict(n_chunks=4, chunk_len=8, B=64, n_steps=60000, eval_every=5000,
             hops=-1, early_stop_mean=80.0,
             weave_mix=_grid_stitch(4, 8, [2, 4])),
        dict(n_chunks=8, chunk_len=8, B=64, n_steps=60000, eval_every=5000,
             hops=-1, early_stop_mean=75.0,
             weave_mix=_grid_stitch(8, 8, [2, 4])),
        dict(n_chunks=16, chunk_len=8, B=64, n_steps=60000, eval_every=5000,
             hops=-1, early_stop_mean=70.0,
             weave_mix=_grid_stitch(16, 8, [2, 4])),

        # --- B: chunk_len alone, n_chunks fixed at 16, hops=-1 ---
        dict(n_chunks=16, chunk_len=16, B=32, n_steps=80000, eval_every=5000,
             hops=-1, early_stop_mean=65.0,
             weave_mix=_grid_stitch(16, 16, [4, 8])),
        dict(n_chunks=16, chunk_len=32, B=32, n_steps=80000, eval_every=5000,
             hops=-1, early_stop_mean=60.0,
             weave_mix=_grid_stitch(16, 32, [8, 16])),
        dict(n_chunks=16, chunk_len=64, B=32, n_steps=100000, eval_every=5000,
             hops=-1, early_stop_mean=55.0,
             weave_mix=_grid_stitch(16, 64, [16, 32])),

        # --- C: enc_hops/hop_drop_prob, ONLY at the full target shape ---
        dict(n_chunks=16, chunk_len=64, B=32, n_steps=80000, eval_every=5000,
             hops=-1, enc_hops=4, hop_drop_prob=0.0, early_stop_mean=50.0,
             weave_mix=_grid_stitch(16, 64, [16, 32])),
        dict(n_chunks=16, chunk_len=64, B=32, n_steps=100000, eval_every=5000,
             hops=-1, enc_hops=4, hop_drop_prob=0.2, early_stop_mean=45.0,
             weave_mix=_grid_stitch(16, 64, [16, 32])),
    ],
)
