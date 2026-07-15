"""
Head count ablation — depth=4, d=64, n_heads=8 (dh=8).
Distribution: arcsine (Beta(0.5,0.5)) warmup_x — equalizes byte coverage,
upweights X=0 (Win A) and X=32 (Win C) endpoints.

dh=8, narrow heads

Baseline for this ablation: d4_h4_arcsine_s0 (n_heads=4).
Compare to: slot8_wina_s0 (n_heads=4, uniform) → 30.6% best.

Traj mix:
| weight | nc | warmup_x dist | SLOT pos |
|--------|----|---------------|----------|
|   1.0  |  4 | arcsine [0,32]|       96 |
"""
hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=8, d_ff=256, V=268,
    lr_max=3e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=2000, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=20000, cosine_T_mult=2, cosine_cycle_warmup=500,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='hmn_chunk_global_iq_rw_nc4_slot8_d4_h8_earlybias_s0', seed=42,
    slot_len=8, slot_count=2, warmup_len=8,
    use_actual_argmax=False, val_n_seqs=3,
    curriculum=[dict(
        n_chunks=4, chunk_len=16, B=8, n_steps=100000, eval_every=20000,
        traj_mix=[
            dict(type='iq_global_rw', weight=1.0, n_chunks=4, window_chunks=2,
                 warmup_x_dist='early_bias'),
        ],
        eval_traj='iq_global_rw',
    )],
)
