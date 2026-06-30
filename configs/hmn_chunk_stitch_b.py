"""
Stitch ablation, stage B: resume from stage A's checkpoint and add the final
stitched-IQ turn (full 128B span, reads all 4 chunks after the 3 overlapping
windows have been refined). This is the actual start-to-finish stitching
test val/test eval now scores against the full-span IQ output, not a window.

Run (loads stage A's checkpoint):
    python -m kvmem.train_hmn_chunk \
        --config configs/hmn_chunk_stitch_b.py \
        --pretrained logs/hmn_chunk_stitch_a/checkpoints/stage0_end.pt \
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_stitch_b', seed=42,

    slot_len=2, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=4, chunk_len=32, depth=2, B=4, n_steps=8000, eval_every=2000,
             traj_mix=[dict(type='ir_stitch', weight=1.0,
                            windows=[(0, 2), (1, 3), (2, 4)], final_iq=True)],
             eval_traj='ir_stitch'),
    ],
)
