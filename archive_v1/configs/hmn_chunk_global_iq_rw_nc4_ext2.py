"""
Global IQ random-window continuation — 300k more steps from ext end checkpoint.
Total ext steps: 400k (100k ext + 300k ext2).
Adds cosine restarts + per-restart warmup to the schedule.

Training schedule rationale:
  - lr_max=3e-4: same as original iq_rw_nc4 run
  - lr_min=1e-6: floor
  - cosine_T0=50000: first cycle 50k steps (longer — model has more to unlearn from ext stagnation)
  - cosine_T_mult=2: 50k→100k→200k cycles over 350k remaining
  - cosine_cycle_warmup=2000: 2k warmup at each restart to avoid lr spikes
  - warmup_steps=0: no initial warmup (continuing existing checkpoint, already warm)

From: logs/hmn_chunk_global_iq_rw_nc4_ext/checkpoints/stage0_end.pt (100k ext steps)

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_global_iq_rw_nc4_ext2.py \\
        --pretrained logs/hmn_chunk_global_iq_rw_nc4_ext/checkpoints/stage0_end.pt \\
        --device mps
    tail -f logs/hmn_chunk_global_iq_rw_nc4_ext2/train.log

Traj mix:
| weight | nc | warmup offsets (train) | warmup offsets (eval) | SLOT pos |
|--------|----|------------------------|----------------------|----------|
|   1.0  |  4 | uniform [0, 32]        | {0, 16, 32}          |       80 |
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=0, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=50000,
    cosine_T_mult=2,
    cosine_cycle_warmup=2000,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='hmn_chunk_global_iq_rw_nc4_ext2', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=False,
    val_n_seqs=3,
    mask_nochain=False,

    curriculum=[
        dict(
            n_chunks=4, chunk_len=16, B=8, n_steps=300000, eval_every=5000,
            traj_mix=[
                dict(type='iq_global_rw', weight=1.0, n_chunks=4, window_chunks=2),
            ],
            eval_traj='iq_global_rw',
        ),
    ],
)
