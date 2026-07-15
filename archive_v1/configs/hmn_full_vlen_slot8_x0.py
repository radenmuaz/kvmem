"""
Ablation: slot_count=8 + x0_fraction=0.3 fix for long-output positions.

Same as hmn_full_vlen_slot4_x0 but slot_count=8 (IDs 258-265).
Tests whether 8 distinct slot IDs provide enough capacity headroom for large nc.
Start from hmn_full_vlen_s0 best.pt — V=268 already has embeddings for IDs 260-265.
"""
hp = dict(
    # Model
    d=128, n_layers=4, n_heads=2, d_ff=256, V=268,
    chunk_attn=256,

    # Training
    lr_max=5e-4, lr_min=5e-6, wd=1e-4,
    warmup_steps=5000,
    log_every=1000, eval_every=10000,
    label_smooth=0.0, seed=42, B=8,

    # Source length
    chunk_len=4, slot_len=8, slot_count=8, warmup_len=8,
    min_nc=3,

    # x=0 oversample fix
    x0_fraction=0.3,

    # Curriculum: (step_end, max_nc, eval_ncs)
    nc_curriculum=[
        (20000,  8,  [4, 6, 8]),
        (60000,  16, [8, 12, 16]),
        (120000, 32, [16, 24, 32]),
    ],

    val_n_seqs=3,

    name='hmn_full_vlen_slot8_x0',
)
