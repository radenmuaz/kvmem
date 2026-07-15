"""
Stage `squeeze` — dedicated compression-capacity experiment, RANDOM-BYTE
CONTROL arm (per docs/HISTORY.md §10). Identical to hmn_squeeze_markov_n4.py
in every respect except data_kind='random' (no data_target_bits — plain
uniform random bytes, the source distribution every prior architecture in
this project trained on).

This is the run to launch FIRST (simpler, and its match% is the baseline
that hmn_squeeze_markov_n4.py needs to clearly beat at the same chunk_len to
demonstrate genuine compression rather than just "the model got bigger/
better at storage regardless of content").

chunk_len=96 (CORRECTED, was 32 — measured, not assumed): the first attempt
at chunk_len=32 was expected to show degraded match% (2x solo's proven
~128-bit near-ceiling), but this run actually converged to ~100% val,
loss~0 by step 70000/160000 on pure random (genuinely incompressible)
256-bit content — no capacity pressure at all, so the paired comparison
against hmn_squeeze_markov_n4.py would have been uninformative (nothing for markov
to beat). Killed that run at step ~115000 once the saturation was clear.
chunk_len=96 (768 bits, 3x the point that just saturated) is a first
escalation estimate — re-verify this run ALSO shows genuine degradation
below ~100% before trusting hmn_squeeze_markov_n4.py's comparison; escalate
chunk_len further (or back off if this overshoots to near-0%) if not.
n_steps reduced to 60000 (from 160000) for this probe — the original run's
own eval curve showed convergence well before 160000 steps at the smaller
scale, so a shorter budget should be enough to see the trend before
committing to the full budget.

Run (only after Stage `relay` finishes and the chain-memory recovery probe
runs — never two jobs at once; run this BEFORE hmn_squeeze_markov_n4.py):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_squeeze_random_n4.py --device mps
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, V=274,
    block_type='single_attn',
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=160000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True,
    rmsnorm=True,
    name='hmn_squeeze_random_n4', seed=48,

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    data_kind='random',

    curriculum=[
        dict(n_chunks=1, chunk_len=96, n_refine=0, B=6, n_steps=60000, eval_every=5000,
             chain_steps=[(0, 1)]),
    ],
)
