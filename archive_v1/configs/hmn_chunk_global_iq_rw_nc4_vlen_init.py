"""
Global IQ with random window training, initialized from vlen_ext2 best checkpoint.

Same iq_global_rw task as hmn_chunk_global_iq_rw_nc4, but starting from the
vlen_ext2 step-20k best checkpoint (43.3% val_mean) instead of from the nc=2
IQ-only checkpoint.

Hypothesis: vlen_ext2 already learned multi-chunk SLOT compression and position-
invariant enc_block attention across nc=2/4/8. That representation may generalize
better to the random-offset IQ task than the nc=2 IQ-only starting point.

From: logs/hmn_chunk_local_64_vlen_ext2/checkpoints/stage0_best.pt
  (step 20k, val_mean=43.3%, stitch 44.6%, win A/B/C ~35%)

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_global_iq_rw_nc4_vlen_init.py \\
        --pretrained logs/hmn_chunk_local_64_vlen_ext2/checkpoints/stage0_best.pt \\
        --device mps
    tail -f logs/hmn_chunk_global_iq_rw_nc4_vlen_init/train.log

Traj mix:
| weight | nc | warmup offsets (train) | warmup offsets (eval) | SLOT pos |
|--------|----|------------------------|----------------------|----------|
|   1.0  |  4 | uniform [0, 32]        | {0, 16, 32}          |       80 |
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=200, log_every=1000,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='hmn_chunk_global_iq_rw_nc4_vlen_init', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=False,
    val_n_seqs=3,
    mask_nochain=False,

    curriculum=[
        dict(
            n_chunks=4, chunk_len=16, B=8, n_steps=50000, eval_every=5000,
            traj_mix=[
                dict(type='iq_global_rw', weight=1.0, n_chunks=4, window_chunks=2),
            ],
            eval_traj='iq_global_rw',
        ),
    ],
)
