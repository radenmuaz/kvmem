"""
Stage 4 v2: 128B src, 7 overlapping 32B windows, B=4.

Same as hmn_chunk_local_128.py but B=4 to prevent MPS swapping on the
all-7-windows stitch trajectory (L=1308). At B=8 the stitch batch is
8×1308=10464 tokens/step — too large for MPS, causing ~0.5 it/s.
At B=4: 4×1308=5232 tokens (≈ v2's 8×572=4576 which ran at 5 it/s).

Stitch weight=2.0, 7 singles weight=1.0 each (22% stitch, 78% singles).

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_local_128_b4.py \\
        --pretrained logs/hmn_chunk_local_64_v2/checkpoints/stage0_end.pt \\
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_local_128_b4', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    curriculum=[
        dict(
            n_chunks=8, chunk_len=16, B=4, n_steps=80000, eval_every=10000,
            traj_mix=[
                # stitch weight=2.0: 22% of steps, L=1308, B=4 → 5232 tok/batch
                dict(type='ir_local', weight=2.0,
                     windows=[(0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8)], n_refine=2),
                # 7 singles weight=1.0: each window independently, L=324, B=4
                dict(type='ir_local', weight=1.0, windows=[(0,2)], n_refine=2),
                dict(type='ir_local', weight=1.0, windows=[(1,3)], n_refine=2),
                dict(type='ir_local', weight=1.0, windows=[(2,4)], n_refine=2),
                dict(type='ir_local', weight=1.0, windows=[(3,5)], n_refine=2),
                dict(type='ir_local', weight=1.0, windows=[(4,6)], n_refine=2),
                dict(type='ir_local', weight=1.0, windows=[(5,7)], n_refine=2),
                dict(type='ir_local', weight=1.0, windows=[(6,8)], n_refine=2),
            ],
            eval_traj='ir_local',
        ),
    ],
)
