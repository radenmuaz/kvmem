"""
Stage 3 v2: 64B src, 3 overlapping 32B windows, mixed training.

Root cause of v1 failure (window 1/2 = 0% in independent eval):
  In all-windows training, window i's IQ SLOT can attend to window j<i's
  SLOT tokens (not blocked -- only raw output c0:c1 regions are blocked).
  The model learned to CHAIN: window 1's SLOT reads from window 0's SLOT
  rather than encoding chunks 1-2 independently. Result: window 1 only
  works when window 0 is present in context.

Fix -- mixed training:
  - all-3-windows (weight 1.0): stitch training, preserves chaining benefit
  - single-window per window (weight 1.0 each): forces independent encoding
    for each window (no prior window SLOT in context)

Eval reports 4-way average: stitch + win0 + win1 + win2 match%.
val_mean must improve all 4 to get a high score.

Resumes from stage 2's checkpoint (same start as v1):
    logs/hmn_chunk_local_32_stage1/checkpoints/stage0_end.pt

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_local_64_v2.py \\
        --pretrained logs/hmn_chunk_local_32_stage1/checkpoints/stage0_end.pt \\
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_local_64_v2', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    curriculum=[
        dict(
            n_chunks=4, chunk_len=16, B=8, n_steps=80000, eval_every=10000,
            traj_mix=[
                # --- stitch: all 3 windows together, chaining + full-seq coverage ---
                dict(type='ir_local', weight=1.0,
                     windows=[(0, 2), (1, 3), (2, 4)], n_refine=2),
                # --- single-window: each window must encode independently ---
                dict(type='ir_local', weight=1.0,
                     windows=[(0, 2)], n_refine=2),
                dict(type='ir_local', weight=1.0,
                     windows=[(1, 3)], n_refine=2),
                dict(type='ir_local', weight=1.0,
                     windows=[(2, 4)], n_refine=2),
            ],
            eval_traj='ir_local',
        ),
    ],
)
