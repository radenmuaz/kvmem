hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    eval_every=10000, log_every=500,
    rope=True, yarn=True, null_kv=True, compile=False,
    silent_eval=False, verbose_eval_n=4,
    name='hmn_mono_tlogit', seed=42,

    curriculum=[
        # Ablation 2 (teacher logit): per-turn alpha schedule.
        # alpha_k = 0.1 + 0.9*(k-1)/11, reaching 1.0 at k=12.
        # Train k=1..4 only; eval k=0..12 tests if the schedule extrapolates.
        # Turn 0 always alpha=0 (pure GT).
        dict(
            src_len=32, slot_len=4, warmup_len=8, out_len=24,
            mode='joint',
            joint_mix=[
                dict(
                    traj='hmn_ir', weight=1.0,
                    n_turns=4, n_turns_choices=[1, 2, 3, 4],
                    h_loss_w=0.0,
                    mono_w=0.0, mono_margin=0.0, cum_w=0.0, cer_b_w=0.0,
                    src_period=1,
                    teacher_distill_w=1.0,
                    teacher_iters=10,
                    teacher_lr=1e-3,
                    teacher_alpha_start=0.1,   # alpha at k=1
                    teacher_alpha_k_max=12,    # alpha=1.0 at k=12
                ),
            ],
            B=8, n_steps=80000, dataset_size=-1,
            hmn_eval_turns=[0, 1, 2, 3, 4],
        ),
    ],

    eval_configs=[(1, 0)],
)
