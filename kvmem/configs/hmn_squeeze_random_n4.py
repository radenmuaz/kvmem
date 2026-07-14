"""
Stage `squeeze` — dedicated compression-capacity experiment, RANDOM-BYTE
CONTROL arm (per docs/HMN_RECIPE.md §10). Identical to hmn_squeeze_ca_n4.py
in every respect except data_kind='random' (no data_target_bits — plain
uniform random bytes, the source distribution every prior architecture in
this project trained on).

This is the run to launch FIRST (simpler, and its match% is the baseline
that hmn_squeeze_ca_n4.py needs to clearly beat at the same chunk_len=32 to
demonstrate genuine compression rather than just "the model got bigger/
better at storage regardless of content"). At n_layers=4/state_len=8/
chunk_len=32, this run is EXPECTED to show degraded match% relative to
Stage `solo`'s chunk_len=16 ceiling — that expected failure is the point,
it establishes the raw-capacity floor hmn_squeeze_ca_n4.py needs to beat.

Run (only after Stage `relay` finishes and the chain-memory recovery probe
runs — never two jobs at once; run this BEFORE hmn_squeeze_ca_n4.py):
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
        dict(n_chunks=1, chunk_len=32, n_refine=0, B=6, n_steps=160000, eval_every=10000,
             chain_steps=[(0, 1)]),
    ],
)
