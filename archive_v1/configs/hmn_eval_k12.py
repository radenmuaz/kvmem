hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256,
    rope=True, yarn=True, null_kv=True, compile=False,
    silent_eval=False, verbose_eval_n=4,
    name='hmn_eval_k12', seed=42,
    eval_every=1,

    curriculum=[
        dict(
            src_len=32, slot_len=4, warmup_len=8, out_len=24,
            mode='joint',
            joint_mix=[
                dict(
                    traj='hmn_ir', weight=1.0,
                    n_turns=4, n_turns_choices=[1, 2, 3, 4],
                    h_loss_w=0.0,
                    mono_w=1.0, mono_margin=0.0, cum_w=0.0, cer_b_w=0.0,
                ),
            ],
            B=8, n_steps=1, dataset_size=-1,
            hmn_eval_turns=[0, 1, 2, 3, 4, 6, 8, 10, 12],
        ),
    ],

    eval_configs=[(1, 0)],
)
