"""
Dual-attention-block ablation (no MLP anywhere — attn+attn per block instead
of attn+ffn) vs the proven 3-window 64B stitched SRS baseline
(experiments/srs_tagged/configs/srs_stitch_nc4_slot8.py: warm-started, sustained
100%/100% val+test from step 35000-60000).

Same task/schedule/hyperparameters as that baseline (windows=[(0,2),(1,3),(2,4)],
n_refine=2, slot_len=8, wrong_token_weight=2.0) — only the block architecture
differs. Trained FROM SCRATCH (no warm-start possible, different state_dict
shape) — a harder starting point than the baseline's warm-started run, so a
weaker or slower result here doesn't necessarily mean dual-attn is worse; a
result that MATCHES or APPROACHES the baseline's ceiling despite the harder
start would be strong evidence for the MLP-not-needed hypothesis.

Run:
    caffeinate -i python3 -m experiments.attn_dual.train \\
        --config experiments/attn_dual/configs/dualattn_nc4_slot8.py \\
        --device mps
    tail -f experiments/attn_dual/logs/dualattn_nc4_slot8/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, V=282,
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=60000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True,
    name='dualattn_nc4_slot8', seed=48,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=2.0,
    val_n_seqs=3,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        dict(n_chunks=4, chunk_len=16, n_refine=2, B=6, n_steps=60000, eval_every=5000,
             windows=[(0, 2), (1, 3), (2, 4)]),
    ],
)
