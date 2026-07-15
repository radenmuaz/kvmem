"""
IQ-only from scratch — slot8, depth=6, fixed dims (d=64 n_heads=4 d_ff=256).

Hypothesis: deeper model can compress more information per SLOT token, compensating
for fixed slot_len=8. Tests whether depth (not width) raises the IQ-only ceiling,
specifically for Win A which stayed flat at 4% across all 100k steps at depth=4.

If depth helps: Win A should break above ~10% at step 80k.
If depth does not help: Win A stays flat — confirms information bottleneck is purely
slot capacity (tokens), not computation depth per token.

Baseline: slot8_wina_s0 (depth=4, 231k params) → Win A=4%, Win B=67%, Win C=29% best.
This run: depth=6, 329k params (+43%), same slot_len=8, same training recipe.

Traj mix:
| weight | nc | warmup_x | SLOT pos |
|--------|----|----------|----------|
|   1.0  |  4 | uniform  |       96 |
"""
hp = dict(
    train_fn='fb',
    d=64, n_layers=6, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=2000, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=20000, cosine_T_mult=2, cosine_cycle_warmup=500,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='hmn_chunk_global_iq_rw_nc4_slot8_depth6_s0', seed=42,
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
