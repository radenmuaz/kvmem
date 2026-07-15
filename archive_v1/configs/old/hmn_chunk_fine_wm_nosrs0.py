"""
Ablation q2: warmup_len=8, IR ckpt, **no stage 0** — SRS from step 1.

Tests: is the 50k IQ-only warmup stage wasted compute?
Same total steps as hmn_chunk_fine_wm (130k for stages 0+1 combined, 80k for stage 2).

Run:
    python -m kvmem.train_hmn_chunk \
        --config configs/hmn_chunk_fine_wm_nosrs0.py \
        --pretrained logs/hmn_feedback_32_ir/checkpoints/stage0_end.pt \
        --device mps
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    eval_every=5000, log_every=500,
    rope=True, yarn=True, null_kv=True, compile=False,
    name='hmn_chunk_fine_wm_nosrs0', seed=42,

    slot_len=1, slot_count=2,
    warmup_len=8,
    ir_turns=2, noise_p=0.5,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        # Stage 0+1 merged: SRS from step 1, 16-byte chunks
        dict(n_chunks=8, chunk_len=16, use_srs=True, B=8, n_steps=130000),
        # Stage 2: 32-byte chunks
        dict(n_chunks=8, chunk_len=32, use_srs=True, B=8, n_steps=80000),
    ],
)
