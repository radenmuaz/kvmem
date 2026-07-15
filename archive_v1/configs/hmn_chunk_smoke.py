"""
Smoke test: verify ar_decode_chunk_fb runs end-to-end without errors.
Same geometry as hmn_chunk_sanity.py, but tiny — 10 train steps, eval every 5.
Not meant to show learning, just confirm the eval path executes cleanly.

Run:
    python -m kvmem.train_hmn_chunk \
        --config configs/hmn_chunk_smoke.py \
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=2,
    eval_every=5, log_every=1,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_smoke', seed=42,

    slot_len=2, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,

    eval_file='datasets/suratalkauthar.txt',

    curriculum=[
        dict(n_chunks=2, chunk_len=128, depth=2, B=2, n_steps=10),
    ],
)
