"""
Global IQ random-window, restarted from stage-1 (nc=2 IQ+IR) checkpoint.
Cosine annealing with warm restarts + long warmup + lower LR.

Training schedule rationale:
  - lr_max=1e-4: 3x lower — large distribution shift (nc=2 IQ+IR → nc=4 random-offset IQ)
  - warmup_steps=10000: long ramp before cosine starts
  - lr_min=1e-6: floor so model never goes fully cold
  - cosine_T0=20000: first cycle 20k steps, then doubles each restart
  - cosine_T_mult=2: T0→40k→80k (fewer but longer restarts over 200k)
  - cosine_cycle_warmup=1000: 1k-step warmup ramp (lr_min→lr_max) at start of EACH restart
  - adam_b2=0.95: faster gradient moving avg — adapts to new task quicker

From: logs/hmn_chunk_local_32_stage1/checkpoints/stage0_end.pt (87.5%, nc=2 IQ+IR)

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_global_iq_rw_nc4_s1_restart.py \\
        --pretrained logs/hmn_chunk_local_32_stage1/checkpoints/stage0_end.pt \\
        --device mps
    tail -f logs/hmn_chunk_global_iq_rw_nc4_s1_restart/train.log

Traj mix:
| weight | nc | warmup offsets (train) | warmup offsets (eval) | SLOT pos |
|--------|----|------------------------|----------------------|----------|
|   1.0  |  4 | uniform [0, 32]        | {0, 16, 32}          |       80 |
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=1e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=10000, log_every=1000,
    adam_b2=0.95,
    lr_schedule='cosine_restarts',
    cosine_T0=20000,
    cosine_T_mult=2,
    cosine_cycle_warmup=1000,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='hmn_chunk_global_iq_rw_nc4_s1_restart', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=False,
    val_n_seqs=3,
    mask_nochain=False,

    curriculum=[
        dict(
            n_chunks=4, chunk_len=16, B=8, n_steps=200000, eval_every=5000,
            traj_mix=[
                dict(type='iq_global_rw', weight=1.0, n_chunks=4, window_chunks=2),
            ],
            eval_traj='iq_global_rw',
        ),
    ],
)
