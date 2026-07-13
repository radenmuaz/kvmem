"""
Standard architecture (attn+ffn), matched-depth staging control — Stage 0:
IQ-only, mirrors experiments/attn_dual/configs/dualattn_nc4_slot8_iq.py
exactly (same steps, same task/windows, same n_refine=0) so the two
architectures can be compared at the SAME staging depth.

Why this exists: the "100%/100%" srs_stitch_nc4_slot8 result was warm-started
from a much deeper lineage (chat-tags phaseB_full IQ(160k, from scratch)->
IR(100k)->phaseB2(100k)->phaseB3(80k)->phaseB4(80k)->wrongtok(60k)->
srs_depth2(60k)->srs_stitch(60k) = 700k cumulative steps across 8 stages).
Precise investigation (docs/SRS_RECIPE.md "Final verdict: dual-attn ablation")
found phaseB2/B3/B4/wrongtok (280k steps) were fixing bugs SPECIFIC to the
chat-tags addressing scheme (shared query tag collisions, IR-degradation) —
not generic pretraining depth the architecture needs, and already baked into
the position-builder code this experiment reuses. The principled matched
budget is therefore phaseB_full's own IQ(160k)+IR(100k) = 260k total, applied
to the SAME stitched task throughout (not replicating chat-tags' exact
intermediate task structure) — isolating the architecture variable at a
depth that's actually relevant, without an unnecessary ~20+ hour replay of
unrelated debugging history. See docs/SRS_RECIPE.md for the full comparison
matrix this feeds into: {dual-attn, standard} x {scratch, matched-depth-staged}.

Run:
    caffeinate -i python3 -m experiments.srs_tagged.train \\
        --config experiments/srs_tagged/configs/srs_stitch_nc4_slot8_iq.py \\
        --device mps
    tail -f experiments/srs_tagged/logs/srs_stitch_nc4_slot8_iq/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256, V=282,
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=160000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    rmsnorm=True,
    name='srs_stitch_nc4_slot8_iq', seed=48,

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
