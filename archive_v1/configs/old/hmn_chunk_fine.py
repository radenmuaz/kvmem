"""
Config B: 8 chunks × 16→32 bytes, slot_len=1 (32× expansion).
Matches surah's 7-ayah granularity (8 ≈ 7 chunks, 32 bytes ≈ short ayah prefix).

Budget (chunk_len=32, slot_len=1, N=8, 13 spans × 2 IR):
  Enc:8×33=264  Singles:8×2×33=528  Pairs:4×2×65=520  Full:2×257=514
  Total: 1826 tokens (very compact)

Run:
    python -m kvmem.train_hmn_chunk --config configs/hmn_chunk_fine.py --device mps
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    eval_every=5000, log_every=500,
    rope=True, yarn=True, null_kv=True, compile=False,
    name='hmn_chunk_fine', seed=42,

    slot_len=1, slot_count=2,
    ir_turns=2, noise_p=0.5,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        # Stage 0: IQ only
        dict(n_chunks=8, chunk_len=16, use_srs=False, B=8, n_steps=50000),
        # Stage 1: Full SRS schedule
        dict(n_chunks=8, chunk_len=16, use_srs=True,  B=8, n_steps=80000),
        # Stage 2: Longer chunks (256 bytes total)
        dict(n_chunks=8, chunk_len=32, use_srs=True,  B=8, n_steps=80000),
    ],
)
