"""
Local-refine curriculum, stage 3: 64B src, 3 overlapping 32B windows
(stride 16B, 50% overlap), resumes from stage 2's checkpoint.

n_chunks=4, chunk_len=16 -> 64B src, windows=[(0,2),(1,3),(2,4)] -> byte
ranges 0-32, 16-48, 32-64. Each window gets its own IQ + 2-step
argmax-refine IR (same unit as stage 2, just applied 3x to overlapping
spans). Val/test report full-sequence (0..64) stitched match% via the
"prolonged AR" decode (ar_decode_chunk_fb_stitch_kv) — only the very
first window's warmup is seeded from ground truth, everything else comes
from the model's own previously decoded output.

Run (loads stage 2's checkpoint):
    python -m kvmem.train_hmn_chunk \
        --config configs/hmn_chunk_local_64.py \
        --pretrained logs/hmn_chunk_local_32/checkpoints/stage1_end.pt \
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_local_64', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=4, chunk_len=16, B=8, n_steps=80000, eval_every=10000,
             traj_mix=[dict(type='ir_local', weight=1.0,
                            windows=[(0, 2), (1, 3), (2, 4)], n_refine=2)],
             eval_traj='ir_local'),
    ],
)
