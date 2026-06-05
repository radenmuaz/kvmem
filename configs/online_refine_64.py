"""
Exp 4b — Harder error correction test: src_len=32, out=24, slot=1.

Goal: stress multi-turn correction. With 24 bytes to recall through 1 h slot,
t1 should be imperfect, so turns t2-t8 can demonstrate progressive improvement.

out_len=-1 → src_len - warmup_len = 32 - 8 = 24 bytes output.
latent_len=3: lighter ponder than Exp 4a (was 7).
Train from scratch (different sequence layout from seg=16 experiments).
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    eval_every=10000, log_every=500,
    rope=True, yarn=True, grok=False,
    null_kv=True, compile=False,
    silent_eval=False, verbose_eval_n=8,
    noise_skew=False, ls_max=0.0, ls_anneal_steps=0,
    aux_attempt_loss=1.0, mono_penalty=1.0,
    dataset_size=10000,     # fixed pool; -1 = unbounded stream
    name='online_refine_64',
    seed=42,
    curriculum=[dict(
        src_len=32, slot_len=1, warmup_len=8,
        out_len=24,           # 32 - 8 = 24 bytes output
        latent_len=3, mem_window=-1,
        mode='joint',
        joint_mix=[
            dict(traj='end',        n_blocks=1, recall_from=0, weight=0.30),
            dict(traj='online_ref', n_blocks=1, recall_from=0, weight=0.30,
                 n_attempts=8,                    # max turns (used for cache precompute)
                 n_attempts_choices=[0, 4, 8],    # sampled uniformly each step
                 h_loss_w=1.0,
                 teacher_max_iter=100,            # starting iter count (doubles if no overfit)
                 teacher_max_max_iter=1600,       # hard ceiling; warns if hit
                 teacher_lr=3e-4,
                 teacher_loss_threshold=0.01),    # stop when loss tiny (overfit)
            dict(traj='end',        n_blocks=2, recall_from=0, weight=0.20),
            dict(traj='int',        n_blocks=2, recall_from=0, weight=0.20),
        ],
        B=8,
        n_steps=160000,
        # dataset_size handled at top level
    )],
    eval_configs=[(1, 0), (2, 0), (2, 1)],
    eval_n_attempts=16,    # test extrapolation beyond trained max (8 turns); stop_early=False
)
