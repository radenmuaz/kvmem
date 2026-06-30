"""
Config A: 4 chunks × 64→128 bytes, slot_len=2 (64× expansion).
Surah (~280 bytes) packs into ~4 chunks cleanly.

Budget (chunk_len=128, slot_len=2, N=4, 7 spans × 2 IR):
  Enc:4×130=520  Singles:4×2×130=1040  Pairs:2×2×258=1032  Full:2×514=1028
  Total: 3620 tokens

Run:
    python -m kvmem.train_hmn_chunk --config configs/hmn_chunk_wide.py --device mps
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    eval_every=5000, log_every=500,
    rope=True, yarn=True, null_kv=True, compile=False,
    name='hmn_chunk_wide', seed=42,

    slot_len=2, slot_count=2,
    ir_turns=2, noise_p=0.5,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        # Stage 0: IQ only (no SRS prefix, just single full-sequence recall)
        dict(n_chunks=4, chunk_len=64,  use_srs=False, B=8, n_steps=50000),
        # Stage 1: Full SRS schedule, shorter chunks
        dict(n_chunks=4, chunk_len=64,  use_srs=True,  B=8, n_steps=80000),
        # Stage 2: Longer chunks (512 bytes total)
        dict(n_chunks=4, chunk_len=128, use_srs=True,  B=4, n_steps=80000),
    ],
)
