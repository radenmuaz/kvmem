"""
Continue vlen_ext for 80k more steps (from vlen_ext 40k end checkpoint).

vlen_ext ran 40k steps (total 120k from start), oscillating 37-51% stitch.
Win C nc=4 stuck at 33-35%. Give it 80k more steps to see if it breaks
through — prior runs showed late-stage jumps (stitch 36→56% in last 10k of vlen).

From: logs/hmn_chunk_local_64_vlen_ext/checkpoints/stage0_end.pt

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_local_64_vlen_ext2.py \\
        --pretrained logs/hmn_chunk_local_64_vlen_ext/checkpoints/stage0_end.pt \\
        --device mps
    tail -f logs/hmn_chunk_local_64_vlen_ext2/train_status.log

Traj mix (same as vlen_ext):
| weight | windows/nc | SLOT pos | trains |
|--------|-----------|----------|--------|
|   2.0  | stitch nc=4 | 80/244/408 | 3-window 64B stitch |
|   1.0  | win A nc=2 | 40 | independent |
|   0.5  | win A nc=4 | 80 | |
|   0.5  | win A nc=8 | 160 | bridge |
|   1.0  | win B nc=4 | 80 | independent |
|   0.5  | win B nc=8 | 160 | bridge |
|   1.0  | win C nc=4 | 80 | independent (target) |
|   0.5  | win C nc=8 | 160 | bridge |
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=200, log_every=1000,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='hmn_chunk_local_64_vlen_ext2', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,
    mask_nochain=True,

    curriculum=[
        dict(
            n_chunks=8, chunk_len=16, B=8, n_steps=80000, eval_every=10000,
            traj_mix=[
                dict(type='ir_local', weight=2.0,
                     windows=[(0,2),(1,3),(2,4)], n_refine=2, n_chunks=4),

                dict(type='ir_local', weight=1.0, windows=[(0,2)], n_refine=2, n_chunks=2),
                dict(type='ir_local', weight=0.5, windows=[(0,2)], n_refine=2, n_chunks=4),
                dict(type='ir_local', weight=0.5, windows=[(0,2)], n_refine=2, n_chunks=8),

                dict(type='ir_local', weight=1.0, windows=[(1,3)], n_refine=2, n_chunks=4),
                dict(type='ir_local', weight=0.5, windows=[(1,3)], n_refine=2, n_chunks=8),

                dict(type='ir_local', weight=1.0, windows=[(2,4)], n_refine=2, n_chunks=4),
                dict(type='ir_local', weight=0.5, windows=[(2,4)], n_refine=2, n_chunks=8),
            ],
            eval_traj='ir_local',
        ),
    ],
)
