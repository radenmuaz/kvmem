"""
Full 1024-byte config: 4 chunks × 256 bytes, depth-2 SRS, feedback argmax IR.

SRS schedule: [(0→2),(2→4),(0→4)] = 3 spans × 2 turns = 6 recall blocks.

Turn 0 (IQ):  [SLOT×2][warmup:8][out:248]           per span
Turn 1 (IR):  [SLOT_A×2][argmax:248][SLOT_B×2][warmup:8][out:248]  per span

Budget (n=4, chunk=256, slot=2, wl=8): L=7170 tokens ✓ (under 8192)

Run (after sanity check passes):
    python -m kvmem.train_hmn_chunk \
        --config configs/hmn_chunk_1024.py \
        --pretrained logs/hmn_feedback_32_ir/checkpoints/stage0_end.pt \
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    eval_every=5000, log_every=500,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=512,            # chunk attention to bound peak memory
    name='hmn_chunk_1024', seed=42,

    slot_len=2, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,

    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        # Stage 0: warm up on smaller chunks (no depth-2 SRS yet, use standard)
        dict(n_chunks=4, chunk_len=64, depth=2, B=8, n_steps=50000),
        # Stage 1: full 1024-byte depth-2 SRS
        dict(n_chunks=4, chunk_len=256, depth=2, B=2, n_steps=80000),
    ],
)

