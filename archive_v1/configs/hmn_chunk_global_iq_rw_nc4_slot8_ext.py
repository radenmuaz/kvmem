"""
slot8 continuation — single clean cosine decay from best checkpoint.

Motivation: slot8 (100k) ended mid-cycle (step 100k = 38k into an 80k cycle).
Best checkpoint was at step 80k (36.6% val_mean). The high-LR cycle 2 caused
AR match% oscillation even as BPB continued dropping (win B up_counter BPB=0.108).

This run uses a single cosine decay over 80k steps (no restarts) so the model
ends at a proper cycle minimum. Starting from the step 80k best checkpoint.

Key improvements expected:
  - Win B/C: stabilize the high BPB → AR match gap (0.1 BPB with only 48% match)
  - Win A: BPB was 2.2 at step 100k and still declining — needs more steps
  - No oscillation: single cycle ends cleanly at lr_min

From: logs/hmn_chunk_global_iq_rw_nc4_slot8/checkpoints/stage0_best.pt (step 80k, 36.6%)

Traj mix:
| weight | nc | warmup offsets (train) | warmup offsets (eval) | SLOT pos |
|--------|----|------------------------|----------------------|----------|
|   1.0  |  4 | uniform [0, 32]        | {0, 16, 32}          |       96 |

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_global_iq_rw_nc4_slot8_ext.py \\
        --pretrained logs/hmn_chunk_global_iq_rw_nc4_slot8/checkpoints/stage0_best.pt \\
        --device mps
    tail -f logs/hmn_chunk_global_iq_rw_nc4_slot8_ext/train.log
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=80000,
    cosine_T_mult=1,
    cosine_cycle_warmup=0,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='hmn_chunk_global_iq_rw_nc4_slot8_ext', seed=42,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=False,
    val_n_seqs=3,
    mask_nochain=False,

    curriculum=[
        dict(
            n_chunks=4, chunk_len=16, B=8, n_steps=80000, eval_every=5000,
            traj_mix=[
                dict(type='iq_global_rw', weight=1.0, n_chunks=4, window_chunks=2),
            ],
            eval_traj='iq_global_rw',
        ),
    ],
)
