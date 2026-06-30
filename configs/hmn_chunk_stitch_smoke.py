"""
Smoke test for overlapping-window stitching (ir_stitch trajectory type).

src=128B (n_chunks=4, chunk_len=32), window=64B (2 chunks), stride=32B
(1 chunk) -> windows [(0,2),(1,3),(2,4)], 50% overlap, covers full source.

Tests both final_iq=False (windows only) and final_iq=True (+ stitched
full-span IQ) paths in one run, 4 steps/stage, eval_every=2.

Run:
    python -m kvmem.train_hmn_chunk \
        --config configs/hmn_chunk_stitch_smoke.py \
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=2,
    log_every=1,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_stitch_smoke', seed=42,

    slot_len=2, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=2,

    curriculum=[
        dict(n_chunks=4, chunk_len=32, depth=2, B=2, n_steps=4, eval_every=2,
             traj_mix=[dict(type='ir_stitch', weight=1.0,
                            windows=[(0, 2), (1, 3), (2, 4)], final_iq=False)],
             eval_traj='ir_stitch'),
        dict(n_chunks=4, chunk_len=32, depth=2, B=2, n_steps=4, eval_every=2,
             traj_mix=[dict(type='ir_stitch', weight=1.0,
                            windows=[(0, 2), (1, 3), (2, 4)], final_iq=True)],
             eval_traj='ir_stitch'),
    ],
)
