"""
Exp 3c — Refine multi-turn, full segment recall window.

Segment recall window: out_len = seg_len = 16.
  n_win = max(1, seg_len - out_len) = 1 → y_start always 0.
  warmup = [seg[0]] * warmup_len (first byte padded, no prior context).
  y = seg[0:16] — full segment, model must recall everything.

Training: k ~ Uniform(0, 5) attempt turns per step.
  k=0: standard <r><y_final><z><h><q><y> — no attempts, direct query.
  k=1..5: attempt turns with linear noise descent + correction blocks.
  Always appends: <y_copy><z_final><h_final><q><y_query>.
  Loss on post-refine <q><y_query> (must reach 100% match).

Noise: linear descent, attempt 0 = noise_hi=0.8, attempt k-1 = noise_lo=0.05.
Eval: t1..t(n+2) tracks extrapolation beyond training max.
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    eval_every=2000, log_every=500,
    ocd=False, ocd_prob=0.0,
    rope=True, yarn=True, grok=False,
    stablemax=False, eval_offset=0.0, grad_clip=10.0,
    null_kv=True,
    compile=False,
    mono_penalty=0.1,
    name='refine_multiturn',
    seed=42,
    curriculum=[dict(
        seg_len=16, slot_len=1, warmup_len=4,
        out_len=16,             # full segment recall window: out_len = seg_len
        latent_len=7, mem_window=-1,
        n_blocks=1,
        recall_from=0,
        recall_froms=0,
        mode='ref',
        n_attempts=5,           # max attempt turns; k sampled from 0..5 each step
        rand_turns=True,        # sample k ~ Uniform(0, n_attempts) per step
        noise_hi=0.8,
        noise_lo=0.05,
        noise_schedule=None,
        B=16,
        n_steps=80000,
        dataset_size=0,
    )],
    eval_configs=[(1, 0)],
)
