"""
IQ-only from scratch — slot12, uniform warmup. More capacity: 12 tok/32B.

slot12: enc_block = 16+12=28 tok, enc_end=112, IQ=12+8+24=44 tok, total L=156.
0.375 tok/byte vs slot8's 0.25 tok/byte. Hypothesis: Win A BPB reaches <1.0.

Traj mix:
| weight | nc | warmup_x_fixed | SLOT pos |
|--------|----|----------------|----------|
|   1.0  |  4 | None (uniform) |      112 |
"""
hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=2000, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=20000, cosine_T_mult=2, cosine_cycle_warmup=500,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='hmn_chunk_global_iq_rw_nc4_slot12_wina_s0', seed=42,
    slot_len=12, slot_count=2, warmup_len=8,
    use_actual_argmax=False, val_n_seqs=3, mask_nochain=False,
    curriculum=[dict(
        n_chunks=4, chunk_len=16, B=8, n_steps=100000, eval_every=20000,
        traj_mix=[
            dict(type='iq_global_rw', weight=1.0, n_chunks=4, window_chunks=2),
        ],
        eval_traj='iq_global_rw',
    )],
)
