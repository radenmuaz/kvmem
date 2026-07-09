"""
Phase B2 — apply the diagnosed warmup_x_fixed fix, continued from Phase B's stage1_best.

Phase B (60.2% best, Win A 100%/Win B 75.0%/Win C 5.6%) copied slot8_ir_v2's traj_mix
verbatim, which never actually added dedicated warmup_x_fixed=16 (Win B) / =32 (Win C)
IR entries — the fix docs/FEEDBACK_RESULTS.md already diagnosed but never applied even
in the untagged baseline. Win C's qualitative failure mode (IQ often encodes 30-80%
correctly, IR1/IR2 crush it to near-random noise, not cross-window confusion) is
exactly what you'd expect from an IR mechanism that's never seen Win C's true X=32
argmax distribution at meaningful density.

Fix: add symmetric Win B (X=16) and Win C (X=32) IQ+IR oversample entries, scaled the
same way Win A's entries already are (IR weight 2x IQ weight), so all three windows get
comparable X-fixed coverage instead of only Win A.

Traj mix (stage new, total weight 7.5):
| weight | n_refine | warmup X (train) | share | purpose |
|--------|----------|-------------------|-------|---------|
| 1.0 | 2 | uniform [0,32] | 13% | IR, all windows (unchanged from B) |
| 0.5 | 0 | uniform [0,32] |  7% | IQ quality, all windows (unchanged) |
| 2.0 | 2 | fixed X=0  (Win A) | 27% | Win A IR (unchanged from B) |
| 1.0 | 0 | fixed X=0  (Win A) | 13% | Win A IQ (unchanged) |
| 1.0 | 2 | fixed X=16 (Win B) | 13% | Win B IR — NEW |
| 0.5 | 0 | fixed X=16 (Win B) |  7% | Win B IQ — NEW |
| 1.0 | 2 | fixed X=32 (Win C) | 13% | Win C IR — NEW |
| 0.5 | 0 | fixed X=32 (Win C) |  7% | Win C IQ — NEW |

Warm-started from Phase B's own stage1_best.pt (same tagged vocab V=276, no embedding
mismatch this time — unlike Phase A->B which needed a from-scratch restart).

See /Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md and
docs/FEEDBACK_RESULTS.md § Chat-tags experiment for full context.

Run:
    caffeinate -i python3 -m experiments.chat_tags.train \\
        --config experiments/chat_tags/configs/slot8_tagged_phaseB2_winfix.py \\
        --pretrained experiments/chat_tags/logs/chat_tags_slot8_phaseB_full/checkpoints/stage1_best.pt \\
        --device mps
    tail -f experiments/chat_tags/logs/chat_tags_slot8_phaseB2_winfix/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256, V=276,
    lr_max=3e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=20000, cosine_T_mult=2,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='chat_tags_slot8_phaseB2_winfix', seed=43,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=4, chunk_len=16, window_chunks=2, B=8, n_steps=100000, eval_every=5000,
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
