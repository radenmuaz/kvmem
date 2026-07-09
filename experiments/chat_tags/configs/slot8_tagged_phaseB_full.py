"""
Phase B — full staged run, chat-tags iq_global_rw_tagged.

Mirrors the untagged best-known recipe (configs/hmn_chunk_global_iq_rw_nc4_slot8_ir_v2.py,
77.8% best @ step 60k, Win A 100%, Win C 55.6%) as closely as possible from scratch:
untagged got there via slot8 (IQ, 80k) -> slot8_ext (IQ continue, 80k) -> slot8_ir
(IQ+IR, 50k) -> slot8_ir_v2 (IQ+IR w/ traj_mix fix, 100k) = ~310k steps total,
each stage warm-started from the previous checkpoint.

Since the tagged vocab (V=276) can't load the untagged pretrained checkpoint
(embedding width differs by 8 rows), this collapses that whole chain into two
from-scratch stages of comparable total step budget:
  Stage 0 (IQ-only, single trajectory, 160k steps): stands in for slot8+slot8_ext.
  Stage 1 (IQ+IR traj_mix, 100k steps): identical traj_mix/step count to slot8_ir_v2,
    continuing from stage 0's checkpoint (loaded automatically at stage boundary
    within the same run — no separate --pretrained needed).

traj_mix (stage 1, total weight 4.5, identical proportions to slot8_ir_v2):
| weight | n_refine | warmup X (train) | share | purpose                 |
|--------|----------|-------------------|-------|--------------------------|
|   1.0  |    2     | uniform [0,32]    |  22%  | IR, all windows          |
|   0.5  |    0     | uniform [0,32]    |  11%  | IQ quality, all windows  |
|   2.0  |    2     | fixed X=0         |  44%  | heavy Win A IR           |
|   1.0  |    0     | fixed X=0         |  22%  | direct Win A IQ recall   |

Eval reports the n_refine=2 trajectory (IQ/IR1/IR2 per-turn match%), same
convention as the untagged run.

See /Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md.

Run:
    caffeinate -i python3 -m experiments.chat_tags.train \\
        --config experiments/chat_tags/configs/slot8_tagged_phaseB_full.py \\
        --device mps
    tail -f experiments/chat_tags/logs/chat_tags_slot8_phaseB_full/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256,
    lr_max=3e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=20000, cosine_T_mult=2,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='chat_tags_slot8_phaseB_full', seed=42,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    curriculum=[
        # Stage 0: IQ-only pretraining (stands in for slot8 + slot8_ext, 160k combined)
        dict(n_chunks=4, chunk_len=16, window_chunks=2, B=8, n_steps=160000, eval_every=10000,
            traj_mix=[
                dict(weight=1.0, n_refine=0),
            ]),
        # Stage 1: IQ+IR traj_mix, identical proportions/steps to slot8_ir_v2
        dict(n_chunks=4, chunk_len=16, window_chunks=2, B=8, n_steps=100000, eval_every=5000,
            traj_mix=[
                dict(weight=1.0, n_refine=2),
                dict(weight=0.5, n_refine=0),
                dict(weight=2.0, n_refine=2, warmup_x_fixed=0),
                dict(weight=1.0, n_refine=0, warmup_x_fixed=0),
            ]),
    ],
)
