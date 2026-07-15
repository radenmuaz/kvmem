"""
Exp 3c — Joint trajectory mix with flat noise.

Fixes Exp 3b failure (descending noise → train-eval mismatch):
  - Flat noise: all draft turns use same U(0.3, 0.8) range
  - Joint training: per step, sample one trajectory type

Training mix:
  30% → I Q       n=1  (no regression baseline)
  20% → I R Q     n=1  k~0..5 flat noise (refine + verify)
  20% → I I Q₀   n=2  recall block 0 (retention)
  30% → interleaved n=2  (retention + current, mixed targets)

Goals:
  - n1_r0 ≥ 92%  (no regression vs Exp 3a)
  - refine t1→final Δ ≥ 50%  (correction still works)
  - n2_r0 ≥ 80%  (retention under update)
  - extrapolation: k > 5 draft turns should still improve

out_len=-1: full segment recall (all 16 bytes).
warmup_len=4: first 4 bytes as query anchor.
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    eval_every=2000, log_every=500,
    ocd=False, ocd_prob=0.0,
    rope=True, yarn=True, grok=False,
    stablemax=False, eval_offset=0.0, grad_clip=10.0,  # eval_offset=0 → y_start=warmup_len=4
    null_kv=True,
    compile=False,
    silent_eval=False,   # set True to suppress per-attempt hex output
    verbose_eval_n=2,
    # Diffusion-style denoising:
    noise_skew=True,          # draft noise: linear ramp 0→2p (low at start, high at end)
    ls_max=0.2,               # positional label smoothing: ε=0 at pos 0, ε=0.2 at pos N-1
    ls_anneal_steps=40000,    # ε decays linearly to 0 over first half of training
    # Correction supervision: direct NLL loss on each attempt turn vs clean GT
    # Fixes "ignore draft and regenerate" sawtooth by giving gradient through correction path
    aux_attempt_loss=0.3,
    mono_penalty=0.05,
    name='refine_joint',
    seed=42,
    curriculum=[dict(
        seg_len=16, slot_len=1, warmup_len=4,
        out_len=12,             # full segment: warmup(4) + out(12) = seg_len(16)
        latent_len=7, mem_window=-1,
        mode='joint',
        joint_mix=[
            dict(traj='end', n_blocks=1, recall_from=0, weight=0.30),
            dict(traj='ref', n_blocks=1, recall_from=0, weight=0.20,
                 n_attempts=5, noise_lo=0.3, noise_hi=0.8),
            dict(traj='end', n_blocks=2, recall_from=0, weight=0.20),
            dict(traj='int', n_blocks=2, recall_from=0, weight=0.30),
        ],
        B=16,
        n_steps=80000,
        dataset_size=-1,
    )],
    eval_configs=[(1, 0), (2, 0), (2, 1)],
)
