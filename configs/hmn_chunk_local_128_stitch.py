"""
Stage 4 stitch-only: 128B src, 7 overlapping 32B windows.
Pure stitch training to establish a strong 7-window baseline before adding independence.

Rationale:
  b4 (mixed from v2 end): started from damaged baseline (6.2% stitch) → can't recover.
  This config instead builds from v3's end (stitch≥60%, win1/2 independent≥40%).

  Phase A (this config): pure stitch — teach the model to stitch 7 windows together.
  Phase B (hmn_chunk_local_128_v3.py): add win1..win6 independence fine-tuning.

  Same lesson as stages 1→2: master the base task before adding complexity.

Sequence lengths:
  all-7-windows: enc(8×20=160) + 7×164 = 160+1148 = 1308 tokens
  B=4 → 4×1308=5232 tokens/batch (proven safe for MPS, same as b4)

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_local_128_stitch.py \\
        --pretrained logs/hmn_chunk_local_64_v3/checkpoints/stage0_end.pt \\
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_local_128_stitch', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    curriculum=[
        dict(
            n_chunks=8, chunk_len=16, B=4, n_steps=80000, eval_every=10000,
            traj_mix=[
                # pure stitch only: 100% of steps on all-7-windows
                # goal: establish strong 7-window stitch quality (target ≥60%)
                dict(type='ir_local', weight=1.0,
                     windows=[(0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8)], n_refine=2),
            ],
            eval_traj='ir_local',
        ),
    ],
)
