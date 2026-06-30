"""
Stitch ablation, stage A: overlapping-window IR only, NO final stitched-IQ
turn. Tests whether the model can even learn to recall overlapping 64B
windows of a 128B source via argmax-IR, before testing whether it can
stitch them into a full start-to-finish reconstruction (stage B).

src=128B (n_chunks=4, chunk_len=32), window=64B (2 chunks), stride=32B
(1 chunk) -> windows [(0,2),(1,3),(2,4)], 50% overlap.

Run (random init):
    python -m kvmem.train_hmn_chunk \
        --config configs/hmn_chunk_stitch_a.py \
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_stitch_a', seed=42,

    slot_len=2, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=4, chunk_len=32, depth=2, B=4, n_steps=8000, eval_every=2000,
             traj_mix=[dict(type='ir_stitch', weight=1.0,
                            windows=[(0, 2), (1, 3), (2, 4)], final_iq=False)],
             eval_traj='ir_stitch'),
    ],
)
