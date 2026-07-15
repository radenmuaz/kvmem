"""
Continue global_iq_rw_nc4 for 100k more steps (from 50k end checkpoint).

Loss plateaued at ~3.0 at step 50k. The nc=2 IQ stage needed 50k steps to
reach 81.9%; this task (nc=4, random offsets, global SLOT) is harder so 50k
is too short. Give it 100k more steps to see if loss breaks through.

From: logs/hmn_chunk_global_iq_rw_nc4/checkpoints/stage0_end.pt

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_global_iq_rw_nc4_ext.py \\
        --pretrained logs/hmn_chunk_global_iq_rw_nc4/checkpoints/stage0_end.pt \\
        --device mps
    tail -f logs/hmn_chunk_global_iq_rw_nc4_ext/train_status.log

Traj mix:
| weight | nc | warmup_len | out_len | warmup offsets (random) | SLOT pos |
|--------|----|-----------|---------|------------------------|----------|
|   1.0  |  4 |         8 |      24 | uniform [0,32] (any)   |       80 |
Eval: fixed {0, 16, 32} → win(0,2), win(1,3), win(2,4)
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=200, log_every=1000,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='hmn_chunk_global_iq_rw_nc4_ext', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=False,
    val_n_seqs=3,
    mask_nochain=False,

    curriculum=[
        dict(
            n_chunks=4, chunk_len=16, B=8, n_steps=100000, eval_every=10000,
            traj_mix=[
                dict(type='iq_global_rw', weight=1.0, n_chunks=4, window_chunks=2),
            ],
            eval_traj='iq_global_rw',
        ),
    ],
)
