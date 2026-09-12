"""
`hmn_tpu_recall1024_jax_incremental_mlp.py` — **kvmem_mlp fork, Experiment 2
(MLP variant)**, forked from `kvmem/configs/hmn_tpu_recall1024_jax_incremental.py`
(2026-09-06). Identical 9-stage curriculum (Phase A: n_chunks 2->4->8->16 at
chunk_len=8; Phase B: chunk_len 16->32->64 at n_chunks=16; Phase C: enc_hops/
hop_drop_prob introduced) — the ONE change is `block_type='attn_mlp'` +
`d_ff=256`, matching `hmn_tpu_sanity_w25_rope_jax_mlp.py`'s own architecture
exactly. Warm-starts from THAT config's checkpoint (not the original
single_attn Experiment 1's), so this run's foundation already has the FFN
capacity from step 0 of stage 0.

**Why this exists**: the original (single_attn, no FFN) Experiment 2 run
was stopped early (2026-09-06, after ~10h40m wall-clock) once stages 5/6
(chunk_len 32/64) collapsed to 4.6%/3.1% MEAN against 60%/55% gates — a
step-change failure right where Phase A (n_chunks scaling, stages 0-3) had
just worked cleanly (100.0% -> 83.5%). This config tests the hypothesis
directly: does adding an FFN sublayer (same attention depth, `n_layers=16`
unchanged) fix the chunk_len-scaling collapse, or does it persist regardless
of the missing-MLP capacity — which would point back to a data/curriculum/
masking explanation instead.

Same `_grid_stitch` helper as the original (verbatim).

Run (never two jobs at once; do not run before
`hmn_tpu_sanity_w25_rope_jax_mlp.py`'s stage3_best.pt actually exists — run
from the repo root so relative log/config paths resolve):
    python3 -m kvmem_mlp.hmn_jax --config kvmem_mlp/configs/hmn_tpu_recall1024_jax_incremental_mlp.py
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
    block_type='attn_mlp', d_ff=256,  # THE ablation: FFN added, everything else unchanged
    lr_max=1e-4, wd=1e-5,
    warmup_steps=1000, log_every=500,
    rope=True, yarn=True,
    null_kv=True,
    rmsnorm=True,
    name='hmn_tpu_recall1024_jax_incremental_mlp', seed=48,

    adaptive=True,
    adapt_signal='val_match',
    adapt_temp=1.0,
    adapt_ema_alpha=0.5,
    adapt_floor=0.05,

    state_len=4, state_vocab_size=1,
    warmup_len=8,
    val_n_seqs=3,

    bucket_lengths=False,
    eval_combinatorial_hops=True,  # inert (no-op) until stage H/I set enc_hops!=-1
    warm_start_from_best=True,
    pretrained_ckpt='kvmem_mlp/logs/hmn_tpu_sanity_w25_rope_jax_mlp/checkpoints/stage3_best.pt',

    curriculum=[
        # --- A: n_chunks alone, chunk_len fixed at 8, hops=-1 ---
        dict(n_chunks=2, chunk_len=8, B=16, n_steps=60000, eval_every=5000,
             hops=-1, early_stop_mean=80.0,
             weave_mix=_grid_stitch(2, 8, [2, 4])),
        dict(n_chunks=4, chunk_len=8, B=16, n_steps=60000, eval_every=5000,
             hops=-1, early_stop_mean=80.0,
             weave_mix=_grid_stitch(4, 8, [2, 4])),
        dict(n_chunks=8, chunk_len=8, B=16, n_steps=60000, eval_every=5000,
             hops=-1, early_stop_mean=75.0,
             weave_mix=_grid_stitch(8, 8, [2, 4])),
        dict(n_chunks=16, chunk_len=8, B=16, n_steps=60000, eval_every=5000,
             hops=-1, early_stop_mean=70.0,
             weave_mix=_grid_stitch(16, 8, [2, 4])),

        # --- B: chunk_len alone, n_chunks fixed at 16, hops=-1 ---
        dict(n_chunks=16, chunk_len=16, B=8, n_steps=80000, eval_every=5000,
             hops=-1, early_stop_mean=65.0,
             weave_mix=_grid_stitch(16, 16, [4, 8])),
        dict(n_chunks=16, chunk_len=32, B=8, n_steps=80000, eval_every=5000,
             hops=-1, early_stop_mean=60.0,
             weave_mix=_grid_stitch(16, 32, [8, 16])),
        dict(n_chunks=16, chunk_len=64, B=8, n_steps=100000, eval_every=5000,
             hops=-1, early_stop_mean=55.0,
             weave_mix=_grid_stitch(16, 64, [16, 32])),

        # --- C: enc_hops/hop_drop_prob, ONLY at the full target shape ---
        dict(n_chunks=16, chunk_len=64, B=8, n_steps=80000, eval_every=5000,
             hops=-1, enc_hops=4, hop_drop_prob=0.0, early_stop_mean=50.0,
             weave_mix=_grid_stitch(16, 64, [16, 32])),
        dict(n_chunks=16, chunk_len=64, B=8, n_steps=100000, eval_every=5000,
             hops=-1, enc_hops=4, hop_drop_prob=0.2, early_stop_mean=45.0,
             weave_mix=_grid_stitch(16, 64, [16, 32])),
    ],
)
