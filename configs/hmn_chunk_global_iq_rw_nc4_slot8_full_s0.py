"""
Global IQ with full-continuation output — from scratch, uniform warmup_x.

Instead of a fixed 24-byte output window, the model must continue from any
warmup position X to the END of the 64-byte source. Output length = 56 - X
(variable). Trained with padded sequences and masked CE loss on padding.

warmup_len=8 (same as all other runs). X ~ Uniform[0, 55].

Traj mix:
| weight | nc | warmup_x | out_len | L |
|--------|----|----------|---------|---|
|  1.0   |  4 | uniform  | 56-X    | 168 |

Eval at X={0,16,32,48}: output lengths {56, 40, 24, 8}.
"""
hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=2000, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=20000, cosine_T_mult=2, cosine_cycle_warmup=500,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='hmn_chunk_global_iq_rw_nc4_slot8_full_s0', seed=42,
    slot_len=8, slot_count=2, warmup_len=8,
    use_actual_argmax=False, val_n_seqs=3,
    curriculum=[dict(
        n_chunks=4, chunk_len=16, B=8, n_steps=100000, eval_every=20000,
        traj_mix=[
            dict(type='iq_global_rw_full', weight=1.0, n_chunks=4),
        ],
        eval_traj='iq_global_rw_full',
    )],
)
