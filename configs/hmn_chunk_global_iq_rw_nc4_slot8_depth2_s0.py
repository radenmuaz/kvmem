"""
IQ-only from scratch — slot8, depth=2, fixed dims (d=64 n_heads=4 d_ff=256).

Base for depth ablation: depth=2 (132k params) vs depth=4 (231k, slot8_wina_s0)
vs depth=6 (329k, slot8_depth6_s0). All same slot_len=8, same training recipe.

Hypothesis: if depth=2 significantly underperforms depth=4, depth adds compression
quality per SLOT token. If depth=2 ≈ depth=4, the bottleneck is slot token count
not layer count.

Traj mix:
| weight | nc | warmup_x | SLOT pos |
|--------|----|----------|----------|
|   1.0  |  4 | uniform  |       96 |
"""
hp = dict(
    train_fn='fb',
    d=64, n_layers=2, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=2000, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=20000, cosine_T_mult=2, cosine_cycle_warmup=500,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='hmn_chunk_global_iq_rw_nc4_slot8_depth2_s0', seed=42,
    slot_len=8, slot_count=2, warmup_len=8,
    use_actual_argmax=False, val_n_seqs=3,
    curriculum=[dict(
        n_chunks=4, chunk_len=16, B=8, n_steps=100000, eval_every=20000,
        traj_mix=[
            dict(type='iq_global_rw', weight=1.0, n_chunks=4, window_chunks=2),
        ],
        eval_traj='iq_global_rw',
    )],
)
