"""
Dual-attn (no MLP) staged pretraining — Stage 0: IQ-only.

Replicates this project's own proven curriculum principle ("IQ pretraining
required before IR — model must learn slot compression before feedback is
meaningful", every prior success in this codebase from hmn_feedback_32_iq
onward follows this staging) instead of a single from-scratch 60k-step shot
on the full IQ+IR task (which is what dualattn_nc4_slot8.py did, and which
plateaued at 50.6%/55.4% — see docs/SRS_RECIPE.md "Final verdict: dual-attn
ablation").

Same task/windows as the target (windows=[(0,2),(1,3),(2,4)]), but n_refine=0
(IQ-only — no argmax-feedback IR turns). Establishes basic slot-compression/
addressing capability first, matching e.g. hmn_chunk_local_32.py's role in
the ir_local curriculum.

Step count (160000) matches chat_tags_slot8_phaseB_full's own Stage 0 (IQ,
also trained from scratch) — the "middle ground" matched-depth budget decided
in docs/SRS_RECIPE.md "Final verdict: dual-attn ablation" after determining
the full ~700k-step historical lineage included ~280k steps of chat-tags-
specific bug fixes (window-tag addressing, IR-degradation) not relevant to
the MLP-vs-no-MLP question — matching just the principled IQ+IR depth (260k
total) isolates the architecture variable without an unnecessary ~20+ hour
replay of unrelated debugging history.

`rmsnorm=True`: uses true RMSNorm (scale-only, no mean-centering, no bias)
instead of nn.LayerNorm at every norm in the model — reason: elegance/
bias-free alignment with the project's MDL philosophy (every Linear layer is
already bias=False; LayerNorm was the one remaining place an implicit bias
term existed — see CLAUDE.md "Key Principles" discussion). CAUTION: a prior
small-scale ablation (configs/hmn_chunk_abl_rmsnorm.py) showed RMSNorm
converging clearly WORSE than LayerNorm at the same unretuned LR (IR-stage
final loss 2.17 vs 0.54) — this run uses the SAME lr_max=1.5e-4 as the
LayerNorm version (no retune) as an initial single-variable test. If this
IQ stage fails to converge (comparably to or worse than the LayerNorm
version's step-20000 checkpoint: val 25.0%, test 44.6%), revert to
rmsnorm=False rather than attempting an LR retune first.

Run:
    caffeinate -i python3 -m experiments.attn_dual.train \\
        --config experiments/attn_dual/configs/dualattn_nc4_slot8_iq.py \\
        --device mps
    tail -f experiments/attn_dual/logs/dualattn_nc4_slot8_iq/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, V=282,
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=160000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True,
    rmsnorm=True,
    name='dualattn_nc4_slot8_iq', seed=48,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        dict(n_chunks=4, chunk_len=16, n_refine=0, B=6, n_steps=160000, eval_every=10000,
             windows=[(0, 2), (1, 3), (2, 4)]),
    ],
)
