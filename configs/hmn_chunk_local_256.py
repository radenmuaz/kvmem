"""
Stage 5: 256B src, 15 overlapping 32B windows (stride 16B, 50% overlap).
Mixed training: all-15-windows (stitch) + 15 single-window (independent encoding).
Resumes from stage 4 checkpoint.

Growth rule: n_windows = (src_len - 32) / 16 + 1
  128B (n_chunks=8)  → 7 windows  [stage 4]
  256B (n_chunks=16) → 15 windows [this stage]
  512B (n_chunks=32) → 31 windows [stage 6, future]

Sequence lengths:
  all-15-windows: enc(16×20=320) + 15×164 = 320+2460 = 2780 tokens
  single-window:  enc(320) + 164 = 484 tokens

With 16 trajectories (1 stitch + 15 singles), each batch samples one at random
(uniform weight). Stitch gets 6.25% of steps; each single gets 6.25% = ~5000 steps
per window over 80k total. This may be too few stitch steps — consider weight=3.0
for stitch if 15-window stitching is unstable.

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_local_256.py \\
        --pretrained logs/hmn_chunk_local_128/checkpoints/stage0_end.pt \\
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_local_256', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    curriculum=[
        dict(
            n_chunks=16, chunk_len=16, B=4, n_steps=80000, eval_every=10000,
            traj_mix=[
                # stitch: all 15 windows (weighted higher: only 6.25% of steps otherwise)
                dict(type='ir_local', weight=3.0,
                     windows=[(i, i+2) for i in range(15)], n_refine=2),
                # single-window: each window encodes independently
            ] + [
                dict(type='ir_local', weight=1.0, windows=[(i, i+2)], n_refine=2)
                for i in range(15)
            ],
            eval_traj='ir_local',
        ),
    ],
)
