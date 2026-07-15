"""
Full-continuation IQ memorization with variable source length — small setup, zero init.

Same as hmn_full_vlen_s0 but chunk_len=4 (src 4-16B) and use_zero_init=True.
Previous s0 (chunk_len=16, ZerO init) was stuck at loss=5.54 (uniform baseline) for 10k steps.
s1_randinit (chunk_len=16, random init) also failed; now retrying ZerO init at smaller scale.
"""
hp = dict(
    # Model
    d=64, n_layers=8, n_heads=2, d_ff=256, V=268,
    chunk_attn=256,
    use_zero_init=True,

    # Training
    lr_max=3e-4, lr_min=3e-6, wd=1e-4,  # lr_min = 0.01 * lr_max
    warmup_steps=2000, n_steps=100000,
    log_every=1000, eval_every=20000,
    label_smooth=0.0, seed=42, B=8,

    # Source length
    chunk_len=4, slot_len=8, slot_count=2, warmup_len=8,
    min_nc=1, max_nc=4,

    # Eval across nc values to test scale invariance
    eval_ncs=[1, 2, 4],
    val_n_seqs=3,

    name='hmn_full_vlen_s1_randinit',
)
