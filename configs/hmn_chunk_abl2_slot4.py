"""
Architecture ablation: slot_count=4 (4 unique slot IDs cycling instead of 2).
Fixed 2-stage testbed (20k IQ + 30k IR).
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_abl2_slot4', seed=42,

    slot_len=4, slot_count=4,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=2, chunk_len=16, B=8, n_steps=20000, eval_every=5000,
             traj_mix=[dict(type='ir_local', weight=1.0, windows=[(0, 2)], n_refine=0)],
             eval_traj='ir_local'),
        dict(n_chunks=2, chunk_len=16, B=8, n_steps=30000, eval_every=10000,
             traj_mix=[dict(type='ir_local', weight=1.0, windows=[(0, 2)], n_refine=2)],
             eval_traj='ir_local'),
    ],
)
