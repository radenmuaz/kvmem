"""
RoPE ablation — IQ-only, from scratch, identical to dualattn_nc4_slot8_iq.py
(the trained rope baseline: val 100.0% / test 100.0% at step 160000) except
`rope=False` (and `yarn=False`, since YaRN only matters when RoPE is on).

Motivation: the model currently has NO position information source other than
RoPE — no learned absolute positional embedding anywhere in DualAttnModel.
Without RoPE, the only way a query can know "where" a key sits (e.g. which of
the 8 per-chunk encoding SLOTs is chunk 0 vs chunk 3, or which byte offset
within a chunk it's reading) is via the FIXED STRUCTURE the attention mask
enforces (which row-ranges can attend to which other row-ranges) plus content
embeddings (data byte value, or the special SLOT/warmup/tag token identity)
each token carries at a token-type level, NOT a position level. This ablation
tests whether that's enough for the same task/windows/step budget, and if not,
how much SLOWER convergence is (not just whether it fails outright) — see
docs/SRS_RECIPE.md for the "does positional invariance actually help" question
this was launched to answer empirically rather than argue about.

Same step budget/eval_every as the baseline for a direct step-aligned
convergence-curve comparison (not just final-accuracy comparison).

Run:
    caffeinate -i python3 -m experiments.attn_dual.train \\
        --config experiments/attn_dual/configs/dualattn_nc4_slot8_iq_norope.py \\
        --device mps
    tail -f experiments/attn_dual/logs/dualattn_nc4_slot8_iq_norope/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, V=282,
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=160000, cosine_T_mult=1,
    rope=False, yarn=False, null_kv=True,
    rmsnorm=True,
    name='dualattn_nc4_slot8_iq_norope', seed=48,

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
