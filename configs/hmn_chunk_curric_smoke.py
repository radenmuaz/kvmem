"""
Smoke test: exercise all three trajectory types (iq_windowed, ir_winrefine,
ir_srs) through a few train+eval steps to catch shape/mask bugs cheaply
before launching the real curriculum/srs_ir runs.

Run:
    python -m kvmem.train_hmn_chunk \
        --config configs/hmn_chunk_curric_smoke.py \
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=2,
    log_every=1,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_curric_smoke', seed=42,

    slot_len=2, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,

    eval_file='datasets/suratalkauthar.txt',

    curriculum=[
        dict(n_chunks=2, chunk_len=128, depth=2, B=2, n_steps=4, eval_every=2,
             traj_mix=[dict(type='iq_windowed', weight=1.0)],
             eval_traj='iq_windowed'),
        dict(n_chunks=2, chunk_len=128, depth=2, B=2, n_steps=4, eval_every=2,
             traj_mix=[dict(type='ir_winrefine', weight=0.7),
                       dict(type='iq_windowed', weight=0.3)],
             eval_traj='ir_winrefine'),
        dict(n_chunks=2, chunk_len=128, depth=2, B=2, n_steps=4, eval_every=2,
             traj_mix=[dict(type='ir_srs', weight=0.6),
                       dict(type='ir_winrefine', weight=0.25),
                       dict(type='iq_windowed', weight=0.15)],
             eval_traj='ir_srs'),
    ],
)
