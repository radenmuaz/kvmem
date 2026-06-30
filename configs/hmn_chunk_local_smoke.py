"""
Smoke test for the new local-refine curriculum (ir_local trajectory type).

Stage 1: n_chunks=2, chunk_len=16 (32B src), windows=[(0,2)], n_refine=0
          -> IQ-only, single window, no feedback.
Stage 2: same scale, n_refine=2 -> IQ + 2-step argmax-refine IR, single
          window = the proven hmn_feedback_32_ir mechanism exactly.
Stage 3: n_chunks=4, chunk_len=16 (64B src), windows=[(0,2),(1,3),(2,4)],
          n_refine=2 -> 3 overlapping windows (50% overlap, stride 16B),
          exercises the new ar_decode_chunk_fb_stitch_kv full-sequence
          stitched ("prolonged AR") eval path.

4 steps/stage, eval_every=2, tiny batch — just checking for shape/mask
errors before any real launch.

Run:
    python -m kvmem.train_hmn_chunk \
        --config configs/hmn_chunk_local_smoke.py \
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=2,
    log_every=1,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_local_smoke', seed=42,

    slot_len=2, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=2,

    curriculum=[
        dict(n_chunks=2, chunk_len=16, B=2, n_steps=4, eval_every=2,
             traj_mix=[dict(type='ir_local', weight=1.0, windows=[(0, 2)], n_refine=0)],
             eval_traj='ir_local'),
        dict(n_chunks=2, chunk_len=16, B=2, n_steps=4, eval_every=2,
             traj_mix=[dict(type='ir_local', weight=1.0, windows=[(0, 2)], n_refine=2)],
             eval_traj='ir_local'),
        dict(n_chunks=4, chunk_len=16, B=2, n_steps=4, eval_every=2,
             traj_mix=[dict(type='ir_local', weight=1.0,
                            windows=[(0, 2), (1, 3), (2, 4)], n_refine=2)],
             eval_traj='ir_local'),
    ],
)
