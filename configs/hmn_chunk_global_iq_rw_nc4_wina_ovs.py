"""
Win A oversampling ablation — slot_len=4, double the Win A training weight.

Question: Is Win A (window 0,2, warmup=bytes 0-7) harder due to SLOT capacity
or training distribution? Uniform random-warmup gives X=0 equal probability to
X=16 (win B) and X=32 (win C). If the ceiling is distribution, adding an explicit
fixed-X=0 entry (2× win A share) should unlock Win A. If it's capacity (4 SLOT
tokens can't compress 32B starting at position 0), oversampling won't help.

slot_len=4: enc_end=80, SLOT at pos 80-83 (4 tokens for 2 chunks = 32B total).
slot_len=8 (slot8_ext) achieved Win B/C but Win A stagnated at BPB~1.3 — possibly
capacity is the bottleneck, but worth testing distribution hypothesis on slot4.

Traj mix:
| weight | nc | warmup_x_fixed | warmup offsets (eval) | SLOT pos |
|--------|----|----------------|----------------------|----------|
|   1.0  |  4 | None (uniform) | {0, 16, 32}          |       80 |
|   1.0  |  4 | 0 (Win A only) | {0, 16, 32}          |       80 |

Win A (X=0) gets 2× the training share vs uniform alone.
Net warmup distribution: X=0 at ~2/3 + uniform 1/3 (effective double-weight at X=0).

From: logs/hmn_chunk_global_iq_rw_nc4_ext2/checkpoints/stage0_best.pt (step 150k, 17.1%)

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_global_iq_rw_nc4_wina_ovs.py \\
        --pretrained logs/hmn_chunk_global_iq_rw_nc4_ext2/checkpoints/stage0_best.pt \\
        --device mps
    tail -f logs/hmn_chunk_global_iq_rw_nc4_wina_ovs/train.log
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=50000,
    cosine_T_mult=2,
    cosine_cycle_warmup=500,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='hmn_chunk_global_iq_rw_nc4_wina_ovs', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=False,
    val_n_seqs=3,
    mask_nochain=False,

    curriculum=[
        dict(
            n_chunks=4, chunk_len=16, B=8, n_steps=100000, eval_every=5000,
            traj_mix=[
                dict(type='iq_global_rw', weight=1.0, n_chunks=4, window_chunks=2),
                dict(type='iq_global_rw', weight=1.0, n_chunks=4, window_chunks=2, warmup_x_fixed=0),
            ],
            eval_traj='iq_global_rw',
        ),
    ],
)
