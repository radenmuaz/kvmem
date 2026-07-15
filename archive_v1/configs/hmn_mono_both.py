hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    eval_every=10000, log_every=500,
    rope=True, yarn=True, null_kv=True, compile=False,
    silent_eval=False, verbose_eval_n=4,
    name='hmn_mono_both', seed=42,

    curriculum=[
        # Ablation 2: mono + cum combined.
        # loss = ntp + adaptive_mono * l_mono + adaptive_cum * l_cum
        # l_mono = mean ReLU(nll_t - nll_{t-1})  [local/2-step]
        # l_cum  = ReLU(mean_t(cum_mean_t) - mean(nll_t))  [global]
        # Each term independently EMA-balanced to ntp magnitude.
        dict(
            src_len=32, slot_len=4, warmup_len=8, out_len=24,
            mode='joint',
            joint_mix=[
                dict(
                    traj='hmn_ir', weight=1.0,
                    n_turns=4, n_turns_choices=[1, 2, 3, 4],
                    h_loss_w=0.0,
                    mono_w=1.0, mono_margin=0.0, cum_w=1.0,
                ),
            ],
            B=8, n_steps=80000, dataset_size=-1,
            hmn_eval_turns=[0, 1, 2, 3, 4],
        ),
    ],

    eval_configs=[(1, 0)],
)
