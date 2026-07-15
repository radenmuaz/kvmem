"""
Ablation q3: warmup_len=8, **random init**, no stage 0 — SRS from step 1.

Tests: does the pretrained IR checkpoint actually help, or does the new
multi-block architecture diverge enough that random init is just as good?

Run WITHOUT --pretrained:
    python -m kvmem.train_hmn_chunk \
        --config configs/hmn_chunk_fine_wm_randinit.py \
        --device mps

(Do NOT pass --pretrained — that is the whole point of this ablation.)
Only run this if q2 (nosrs0) shows meaningful improvement over q1 (wm).
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    eval_every=5000, log_every=500,
    rope=True, yarn=True, null_kv=True, compile=False,
    name='hmn_chunk_fine_wm_randinit', seed=42,

    slot_len=1, slot_count=2,
    warmup_len=8,
    ir_turns=2, noise_p=0.5,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        # SRS from step 1, 16-byte chunks
        dict(n_chunks=8, chunk_len=16, use_srs=True, B=8, n_steps=130000),
        # Stage 2: 32-byte chunks
        dict(n_chunks=8, chunk_len=32, use_srs=True, B=8, n_steps=80000),
    ],
)
