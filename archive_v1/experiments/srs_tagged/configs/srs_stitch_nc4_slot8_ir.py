"""
Standard architecture (attn+ffn), matched-depth staging control — Stage 1:
IQ+IR, warm-started from srs_stitch_nc4_slot8_iq.py (Stage 0). Mirrors
experiments/attn_dual/configs/dualattn_nc4_slot8_ir.py exactly (same steps,
same task/windows, same n_refine=2, same wrong_token_weight) — the only
difference between this run and the dual-attn staged run is the block
architecture (attn+ffn vs attn+attn), at matched 2-stage (30k+60k) training
depth. See srs_stitch_nc4_slot8_iq.py's docstring for why this control exists.

Run (only after srs_stitch_nc4_slot8_iq.py finishes):
    caffeinate -i python3 -m experiments.srs_tagged.train \\
        --config experiments/srs_tagged/configs/srs_stitch_nc4_slot8_ir.py \\
        --pretrained experiments/srs_tagged/logs/srs_stitch_nc4_slot8_iq/checkpoints/stage0_best.pt \\
        --device mps
    tail -f experiments/srs_tagged/logs/srs_stitch_nc4_slot8_ir/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256, V=282,
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=100000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    rmsnorm=True,
    name='srs_stitch_nc4_slot8_ir', seed=49,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=2.0,
    val_n_seqs=3,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        dict(n_chunks=4, chunk_len=16, n_refine=2, B=6, n_steps=100000, eval_every=10000,
             windows=[(0, 2), (1, 3), (2, 4)], eval_mode='stitch'),
    ],
)
