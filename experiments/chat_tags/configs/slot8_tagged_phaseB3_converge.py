"""
Phase B3 — same traj_mix fix as B2, but a single clean cosine decay (no restart)
so the run actually converges instead of oscillating through cosine-restart cycles.

B2 (warmup_x_fixed fix) showed real promise (peak mean 70.8%, Win B peaked 95.8%,
Win C peaked 50.0%) but never converged — cosine_T_mult=2 kept restarting the LR,
and every window was still swinging wildly (e.g. Win B ranged 30.6%-95.8% across
the run) when stage 0 ended at step 100k. This mirrors the same "cycle 3 restart
introduces volatility, never fully recovers" pattern documented for the untagged
baseline's own runs (docs/FEEDBACK_RESULTS.md).

Fix: single cosine_T0 == n_steps, T_mult=1 — one smooth decay to lr_min, matching
the pattern that worked for the untagged slot8_ext continuation ("single clean
cosine decay ... ends at a proper cycle minimum").

Warm-started from B2's own stage0_best.pt (70.8%, step 20000) — same tagged
vocab, no embedding mismatch.

Goal (per user: iterate autonomously until all windows excellent/perfect):
all three windows >=90% at the final (converged) checkpoint. If not reached,
next fix in queue: window-specific query tags, per-window loss reweighting,
n_refine=3, or rehearsal — decided automatically after this run completes.

Run:
    caffeinate -i python3 -m experiments.chat_tags.train \\
        --config experiments/chat_tags/configs/slot8_tagged_phaseB3_converge.py \\
        --pretrained experiments/chat_tags/logs/chat_tags_slot8_phaseB2_winfix/checkpoints/stage0_best.pt \\
        --device mps
    tail -f experiments/chat_tags/logs/chat_tags_slot8_phaseB3_converge/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256, V=276,
    lr_max=2e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=80000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='chat_tags_slot8_phaseB3_converge', seed=44,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=4, chunk_len=16, window_chunks=2, B=8, n_steps=80000, eval_every=5000,
            traj_mix=[
                dict(weight=1.0, n_refine=2),
                dict(weight=0.5, n_refine=0),
                dict(weight=2.0, n_refine=2, warmup_x_fixed=0),
                dict(weight=1.0, n_refine=0, warmup_x_fixed=0),
                dict(weight=1.0, n_refine=2, warmup_x_fixed=16),
                dict(weight=0.5, n_refine=0, warmup_x_fixed=16),
                dict(weight=1.0, n_refine=2, warmup_x_fixed=32),
                dict(weight=0.5, n_refine=0, warmup_x_fixed=32),
            ]),
    ],
)
