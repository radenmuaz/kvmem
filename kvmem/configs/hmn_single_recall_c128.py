"""
`hmn_single_recall_c128.py` — same single-chunk/single-STATE recall task as
`hmn_single_recall_c64.py`, bumped to `chunk_len=128` (2x). **Warm-started
from `hmn_single_recall_c64`'s best checkpoint**
(`logs/hmn_single_recall_c64/checkpoints/stage0_best.pt`, early-stopped at
step ~100000, eval_mean=100.0%) — architecture (d/n_layers/n_heads/state_len/
V) is unchanged, only `chunk_len` grows, so all weights transfer directly
(rope handles the longer packed sequence; no shape mismatch).

Part of an incremental chunk_len ladder: c64 (done) -> c128 (this config) ->
c256 (warm-start from this config's best) -> c512 (warm-start from c256's
best) — each stage builds on the previous stage's checkpoint, not on c64
directly.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_single_recall_c128.py --device mps
"""

hp = dict(
    d=64, n_layers=8, n_heads=4, V=274,
    block_type='single_attn',
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=100000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True,
    rmsnorm=True,
    name='hmn_single_recall_c128', seed=48,
    _pretrained_ckpt='logs/hmn_single_recall_c64/checkpoints/stage0_best.pt',

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=1, chunk_len=128, n_refine=0, B=6, n_steps=100000, eval_every=10000,
             chain_steps=[(0, 1)]),
    ],
)
