"""
Exp 4a — Online rollout refine (teacher h).

Fixes Exp 3c.2 correction divergence by replacing synthetic flat noise with
gradient-guided teacher h targets.

Each online_ref training step:
  1. I Q forward pass → compute h_teacher = h_acts - lr_h * ∂L/∂h_acts  (stop_gradient)
  2. I R Q forward pass, k~0..10 turns, zero noise (teacher force on token drafts)
  3. Loss = NTP(final y) + h_loss_w * MSE(h_turn_t, h_teacher) per turn

n_turns=0 regresses to standard I Q (prevents forgetting).
Teacher h is computed from the current model at every step — no frozen teacher.

h_lr=1.0: step size for the gradient update on h activations.
h_loss_w=0.5: weight of the h MSE loss relative to NTP final.

Start from Exp 3c.2 checkpoint (IQ=95.8%, refine broken).
Goal: monotonically improving n1_r0_tN, at least 1 seq reaching 100% in 20 turns.

Ablation notes (future):
  - diff target: supervise -lr*grad instead of h_acts - lr*grad (needs residual arch op)
  - h_lr sweep: 0.1, 0.5, 1.0, 2.0
  - h_loss_w sweep: 0.1, 0.5, 1.0
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
    silent_eval=False,
    verbose_eval_n=8,      # show all test sequences per eval
    noise_skew=False,       # no noise in online_ref, skew irrelevant
    ls_max=0.0,             # no positional label smoothing (was annealed to 0 by 3c.2 end)
    ls_anneal_steps=0,
    aux_attempt_loss=1.0,   # NTP on each attempt turn vs clean GT (same scale as primary)
    mono_penalty=0.05,      # later turns must have lower NLL than earlier turns
    dataset_size=-1,         # infinite stream
    name='online_refine',
    seed=42,
    curriculum=[dict(
        seg_len=16, slot_len=1, warmup_len=4,
        out_len=12,
        latent_len=7, mem_window=-1,
        mode='joint',
        joint_mix=[
            dict(traj='end',        n_blocks=1, recall_from=0, weight=0.30),
            dict(traj='online_ref', n_blocks=1, recall_from=0, weight=0.30,
                 n_attempts=10, h_lr=1.0, h_loss_w=0.1),
            dict(traj='end',        n_blocks=2, recall_from=0, weight=0.20),
            dict(traj='int',        n_blocks=2, recall_from=0, weight=0.20),
        ],
        B=16,
        n_steps=80000,
        dataset_size=-1,
    )],
    eval_configs=[(1, 0), (2, 0), (2, 1)],
    eval_n_attempts=20,     # test up to 20 turns at eval to check convergence
)
