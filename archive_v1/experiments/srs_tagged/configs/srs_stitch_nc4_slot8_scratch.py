"""
From-scratch control for the dual-attn (no-MLP) ablation
(experiments/attn_dual/configs/dualattn_nc4_slot8.py).

Identical task/hyperparameters to srs_stitch_nc4_slot8.py (the proven
warm-started baseline: 100%/100% sustained val+test, steps 35000-60000) —
ONLY difference is no `--pretrained` flag, i.e. trained from scratch, same
as dualattn_nc4_slot8.py necessarily is (dual-attn has no FFN, so it can't
receive the original warm-start checkpoint's FFN weights — see
docs/SRS_RECIPE.md "Dual-attention-block ablation" for why warm-starting
dualattn to "match" isn't a valid fix instead).

This is the missing control: srs_stitch_nc4_slot8's headline 100%/100% result
was always warm-started, so nothing in this project has shown what the
STANDARD attn+ffn block does on this exact stitched task from scratch. Without
this run, comparing dualattn (scratch) against srs_stitch_nc4_slot8
(warm-started) conflates "no MLP" with "no warm start" as confounded
variables — this run isolates the "no MLP" variable alone.

Run (only after dualattn_nc4_slot8 finishes — never two jobs at once):
    caffeinate -i python3 -m experiments.srs_tagged.train \\
        --config experiments/srs_tagged/configs/srs_stitch_nc4_slot8_scratch.py \\
        --device mps
    tail -f experiments/srs_tagged/logs/srs_stitch_nc4_slot8_scratch/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256, V=282,
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=60000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='srs_stitch_nc4_slot8_scratch', seed=48,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=2.0,
    val_n_seqs=3,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        dict(n_chunks=4, chunk_len=16, n_refine=2, B=6, n_steps=60000, eval_every=5000,
             windows=[(0, 2), (1, 3), (2, 4)], eval_mode='stitch'),
    ],
)
