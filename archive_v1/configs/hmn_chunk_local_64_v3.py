"""
Stage 3 v3: 64B src, 3 overlapping 32B windows.
Targeted independence fix: resume from v1 (pure stitch, 70.8%) and add win1/win2 singles.

Rationale:
  v1 (pure stitch): stitch=76.8%, win0=85.9%, win1=0%, win2=13%
  v2 (4-way equal mix): stitch=6.2%, win0=13.5%, win1=15.1%, win2=24.5%

  v2 failed because: (1) started from v2's damaged stitch checkpoint, AND
  (2) equal mix gave stitch only 25% of steps → convergence too slow.

  v3 fixes both:
  (1) Resumes from v1 (70.8% stitch, win0 already independent).
  (2) Targeted mix: stitch×3 + win1×1 + win2×1 (60% stitch, 20% each problematic window).
  (3) Skips win0 — it's already independent and doesn't benefit from solo training.

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_local_64_v3.py \\
        --pretrained logs/hmn_chunk_local_64/checkpoints/stage0_end.pt \\
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_local_64_v3', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    curriculum=[
        dict(
            n_chunks=4, chunk_len=16, B=8, n_steps=80000, eval_every=10000,
            traj_mix=[
                # stitch all 3 windows (weight=3.0): 60% of steps
                # maintains the v1 stitch quality as the primary objective
                dict(type='ir_local', weight=3.0,
                     windows=[(0,2),(1,3),(2,4)], n_refine=2),
                # win1 independence (weight=1.0): 20% of steps
                # window (1,3) covers bytes 16-47 — chains from win0 in v1
                dict(type='ir_local', weight=1.0, windows=[(1,3)], n_refine=2),
                # win2 independence (weight=1.0): 20% of steps
                # window (2,4) covers bytes 32-63 — also chains in v1
                dict(type='ir_local', weight=1.0, windows=[(2,4)], n_refine=2),
            ],
            eval_traj='ir_local',
        ),
    ],
)
