"""
Dual-attn (no MLP) staged pretraining — Stage 1: IQ+IR, warm-started from
Stage 0 (dualattn_nc4_slot8_iq.py).

Same task/windows as the original from-scratch ablation (dualattn_nc4_slot8.py,
which plateaued at 50.6%/55.4% val/test after 60k steps with ZERO staging) —
warm-started from the IQ-only checkpoint instead of random init, mirroring
the IQ-then-IR staging every other success in this project has used. Tests
whether dual-attn's poor from-scratch result was a training-curriculum
problem (fixable by staging) rather than a fundamental architecture
limitation (not fixable by staging).

Step count (100000) matches chat_tags_slot8_phaseB_full's own Stage 1 (IR) —
see dualattn_nc4_slot8_iq.py's docstring for the "middle ground" matched-depth
reasoning (260k total: 160k IQ + 100k IR, isolating the architecture variable
without replaying ~280k steps of unrelated chat-tags-specific bug-fix history).

`rmsnorm=True`: Stage 0 (IQ) was switched from LayerNorm to RMSNorm mid-flight
and hit a PERFECT sweep (100%/100% val+test) by step 30000/160000 — stopped
early (step 40000, 2nd consecutive perfect checkpoint confirming stability)
since continuing to 160000 had no further value. Dramatic reversal from both
the LayerNorm version (25.0%/44.6% at the same step) and the prior small-scale
RMSNorm ablation's negative result (configs/hmn_chunk_abl_rmsnorm.py). This
stage carries rmsnorm=True forward — must match Stage 0's architecture exactly
for the warm-start to load correctly.

Run (only after dualattn_nc4_slot8_iq.py finishes):
    caffeinate -i python3 -m experiments.attn_dual.train \\
        --config experiments/attn_dual/configs/dualattn_nc4_slot8_ir.py \\
        --pretrained experiments/attn_dual/logs/dualattn_nc4_slot8_iq/checkpoints/stage0_best.pt \\
        --device mps
    tail -f experiments/attn_dual/logs/dualattn_nc4_slot8_ir/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, V=282,
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=100000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True,
    rmsnorm=True,
    name='dualattn_nc4_slot8_ir', seed=49,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=2.0,
    val_n_seqs=3,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        dict(n_chunks=4, chunk_len=16, n_refine=2, B=6, n_steps=100000, eval_every=10000,
             windows=[(0, 2), (1, 3), (2, 4)]),
    ],
)
