"""
Standard architecture (attn+ffn), matched-depth staging control — Stage 0,
LayerNorm variant. IDENTICAL to srs_stitch_nc4_slot8_iq.py except
rmsnorm=False — the missing comparison cell flagged when auditing the
RMSNorm findings this session.

Why this exists: srs_stitch_nc4_slot8_iq/_ir (rmsnorm=True) showed a
persistent val/test generalization gap (val 100%, test stuck ~53.6%) that
dual-attn+rmsnorm never showed. But we've never seen standard+LayerNorm at
this SAME shallow (260k) matched-depth budget — only at the much deeper
700k-step warm-started lineage (which reached 100%/100%). Without this run,
we can't tell whether RMSNorm caused the generalization gap, or whether
standard-arch just needs deeper staging than 260k regardless of norm choice
(a training-depth confound, not a norm-choice finding). This run isolates
that variable. Comparison matrix this feeds into: {dual-attn, standard} x
{LayerNorm, RMSNorm} x {scratch, matched-depth-staged}.

Run:
    caffeinate -i python3 -m experiments.srs_tagged.train \\
        --config experiments/srs_tagged/configs/srs_stitch_nc4_slot8_iq_ln.py \\
        --device mps
    tail -f experiments/srs_tagged/logs/srs_stitch_nc4_slot8_iq_ln/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256, V=282,
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=160000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    rmsnorm=False,
    name='srs_stitch_nc4_slot8_iq_ln', seed=48,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        dict(n_chunks=4, chunk_len=16, n_refine=0, B=6, n_steps=160000, eval_every=10000,
             windows=[(0, 2), (1, 3), (2, 4)], eval_mode='stitch'),
    ],
)
