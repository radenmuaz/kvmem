"""
Full-continuation IQ memorization with variable source length — 3-stage curriculum.

chunk_len=4, min_nc=3 (src ≥ 12B > warmup_len=8B avoids x_max<=0).

Curriculum (nc_curriculum drives n_steps and eval_ncs):
  Stage 1 — steps    1-20k:  nc~U[3,8],  max src=32B  (8×4)
  Stage 2 — steps 20k-60k:  nc~U[3,16], max src=64B  (16×4)
  Stage 3 — steps 60k-120k: nc~U[3,32], max src=128B (32×4)

Eval at each checkpoint includes all stages seen so far (regression test).
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
    chunk_len=4, slot_len=8, slot_count=2, warmup_len=8,
    min_nc=3,

    # Curriculum: (step_end, max_nc, eval_ncs)
    # eval_ncs accumulate across stages for regression testing
    nc_curriculum=[
        (20000,  8,  [4, 6, 8]),        # stage 1: 32B max
        (60000,  16, [8, 12, 16]),       # stage 2: 64B max
        (120000, 32, [16, 24, 32]),      # stage 3: 128B max
    ],

    val_n_seqs=3,

    name='hmn_full_vlen_s0',
)
