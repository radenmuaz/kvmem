"""
Sanity check: 256 bytes = 2 chunks × 128, depth-2 SRS, feedback argmax IR.

slot_len=2 differs from pretrained checkpoint's slot_len=4 (hmn_feedback_32_ir
was trained at src_len=32/slot_len=4 = 8 bytes/slot). Different slot geometry
means the pretrained slot-compression weights don't transfer — train from
random init instead.

SRS schedule: [(0→1), (1→2), (0→2)] = 3 spans × 2 turns = 6 recall blocks.

Run:
    python -m kvmem.train_hmn_chunk \
        --config configs/hmn_chunk_sanity.py \
        --device mps
"""

hp = dict(
    train_fn='fb',             # route to train_chunk_fb
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    eval_every=2000, log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,            # chunk attention every 256 query rows
    name='hmn_chunk_sanity', seed=42,

    slot_len=2, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,    # feedback: use model's own argmax

    eval_file='datasets/suratalkauthar.txt',   # 3 ayahs, 179 bytes — fits in 2×128

    curriculum=[
        dict(n_chunks=2, chunk_len=128, depth=2, B=2, n_steps=40000),
    ],
)

