hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    eval_every=10000, log_every=500,
    rope=True, yarn=True, null_kv=True, compile=False,
    silent_eval=False, verbose_eval_n=4,
    name='hmn_32', seed=42,

    curriculum=[
        # Stage 1: I Q full window — can 4 slots losslessly compress 32 bytes?
        dict(
            src_len=32, slot_len=4, warmup_len=0, out_len=32,
            mode='joint',
            joint_mix=[dict(traj='hmn_iq', weight=1.0)],
            B=8, n_steps=50000, dataset_size=-1,
            hmn_eval_turns=[0],
        ),
        # Stage 2: I Q windowed recall — bottleneck with partial context
        dict(
            src_len=32, slot_len=4, warmup_len=8, out_len=24,
            mode='joint',
            joint_mix=[dict(traj='hmn_iq', weight=1.0)],
            B=8, n_steps=50000, dataset_size=-1,
            hmn_eval_turns=[0],
        ),
        # Stage 3: I R Q, k sampled from {0..4} — test monotonic improvement per turn
        dict(
            src_len=32, slot_len=4, warmup_len=8, out_len=24,
            mode='joint',
            joint_mix=[
                dict(
                    traj='hmn_ir', weight=1.0,
                    n_turns=4, n_turns_choices=[0, 1, 2, 3, 4],
                    h_loss_w=1.0,
                    teacher_lr=1e-3, teacher_max_iter=10, teacher_max_max_iter=80,
                ),
            ],
            B=8, n_steps=80000, dataset_size=-1,
            hmn_eval_turns=[0, 1, 2, 3, 4],
        ),
        # Stage 4: joint mix — regression check, both paths healthy
        dict(
            src_len=32, slot_len=4, warmup_len=8, out_len=24,
            mode='joint',
            joint_mix=[
                dict(traj='hmn_iq', weight=0.4),
                dict(
                    traj='hmn_ir', weight=0.6,
                    n_turns=4, n_turns_choices=[0, 1, 2, 3, 4],
                    h_loss_w=1.0,
                    teacher_lr=1e-3, teacher_max_iter=10, teacher_max_max_iter=80,
                ),
            ],
            B=8, n_steps=40000, dataset_size=-1,
            hmn_eval_turns=[0, 1, 2, 3, 4],
        ),
    ],

    eval_configs=[(1, 0)],
)
