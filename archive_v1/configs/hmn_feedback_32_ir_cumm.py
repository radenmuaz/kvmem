hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    eval_every=10000, log_every=500,
    rope=True, yarn=True, null_kv=True, compile=False,
    name='hmn_feedback_32_ir_cumm', seed=42,

    # IR + feedback with cum_mean loss aggregation.
    # Run with: --pretrained logs/hmn_feedback_32_iq/checkpoints/stage0_end.pt
    curriculum=[dict(
        src_len=32, slot_len=4, slot_count=2,
        warmup_len=8, out_len=24,
        k_choices=[0, 1, 2, 3, 4],
        loss_agg='cum_mean',
        B=8, n_steps=80000,
        hmn_eval_turns=[0, 1, 2, 3, 4],
    )],
)
