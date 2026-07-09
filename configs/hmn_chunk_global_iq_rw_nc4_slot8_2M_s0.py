"""
IQ-only from scratch — ~2.07M params, slot8, uniform warmup.

Scale ablation: 9× more parameters than baseline (231k → 2.07M).
Fixed slot_len=8, same task, same training recipe. Tests whether raw compute
and parameter budget lifts the IQ-only ceiling, specifically Win A which is
stuck at 0–4% across all slot4/8/12 and depth2/4/6 runs.

Architecture: d=144, n_layers=8, n_heads=8, d_ff=576
  d_head = 144/8 = 18
  d_ff/d = 4 (same ratio as baseline)
  depth = 8

Compare against:
  slot8_wina_s0    231k  depth=4  → MEAN 30.6% best
  slot8_depth6_s0  329k  depth=6  → queued
  THIS             1.25M depth=6  → rough upper bound for IQ-only

Traj mix:
| weight | nc | warmup_x | SLOT pos | L   |
|--------|----|----------|----------|-----|
|   1.0  |  4 | uniform  |       96 | 136 |
"""
hp = dict(
    train_fn='fb',
    d=144, n_layers=8, n_heads=8, d_ff=576, V=268,
    lr_max=2e-4, lr_min=1e-6, wd=0.01,
    warmup_steps=3000, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=20000, cosine_T_mult=2, cosine_cycle_warmup=500,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='hmn_chunk_global_iq_rw_nc4_slot8_2M_s0', seed=42,
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
