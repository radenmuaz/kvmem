hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    eval_every=10000, log_every=500,
    rope=True, yarn=True, null_kv=True, compile=False,
    name='hmn_feedback_32', seed=42,

    # Stage 0: IQ only — establish slot compression before introducing feedback.
    # The feedback IR turns require SLOT_A to be a meaningful representation;
    # without this, SLOT_A is noise and the argmax feedback signal is garbage.
    # Stage 1: introduce feedback refinement (k=0..4).
    curriculum=[
        dict(
            src_len=32, slot_len=4, slot_count=2,
            warmup_len=8, out_len=24,
            k_choices=[0],          # IQ only
            B=8, n_steps=50000,
            hmn_eval_turns=[0],
        ),
        dict(
            src_len=32, slot_len=4, slot_count=2,
            warmup_len=8, out_len=24,
            k_choices=[0, 1, 2, 3, 4],
            B=8, n_steps=80000,
            hmn_eval_turns=[0, 1, 2, 3, 4],
        ),
    ],
)
