"""
Stage 4: 128B src, 7 overlapping 32B windows (stride 16B, 50% overlap).
Mixed training: all-7-windows (stitch) + 7 single-window (independent encoding).
Resumes from stage 3 v2 checkpoint.

Growth rule (fixed window=32B, stride=16B): n_windows = (src_len - 32) / 16 + 1
  64B  (n_chunks=4) → 3 windows  [stage 3]
  128B (n_chunks=8) → 7 windows  [this stage]
  256B (n_chunks=16)→ 15 windows [stage 5]

Sequence lengths:
  all-7-windows: enc(8×20=160) + 7×(IQ36+IR64+IR64=164) = 160+1148 = 1308 tokens
  single-window: enc(160) + 164 = 324 tokens

chunk_attn=256 is safe: it is ROW-CHUNKED SDPA for memory efficiency only,
not a receptive field limit. Full mask is applied in each chunk.

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_local_128.py \\
        --pretrained logs/hmn_chunk_local_64_v2/checkpoints/stage0_end.pt \\
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_local_128', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    curriculum=[
        dict(
            n_chunks=8, chunk_len=16, B=8, n_steps=80000, eval_every=10000,
            traj_mix=[
                # stitch: all 7 windows together (full-seq coverage + chaining)
                # weight=2.0 so stitch gets ~22% of steps (vs 12.5% at equal weights)
                # v2 lesson: equal weights made stitch too slow to converge
                dict(type='ir_local', weight=2.0,
                     windows=[(0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8)], n_refine=2),
                # single-window: each window must encode independently
                dict(type='ir_local', weight=1.0, windows=[(0,2)], n_refine=2),
                dict(type='ir_local', weight=1.0, windows=[(1,3)], n_refine=2),
                dict(type='ir_local', weight=1.0, windows=[(2,4)], n_refine=2),
                dict(type='ir_local', weight=1.0, windows=[(3,5)], n_refine=2),
                dict(type='ir_local', weight=1.0, windows=[(4,6)], n_refine=2),
                dict(type='ir_local', weight=1.0, windows=[(5,7)], n_refine=2),
                dict(type='ir_local', weight=1.0, windows=[(6,8)], n_refine=2),
            ],
            eval_traj='ir_local',
        ),
    ],
)
