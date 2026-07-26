"""
Tiny/fast feasibility probe for the `hops=1` accum-RNN relay mechanism —
same weave_mix (batch/stream/interleave_delayed) + hops=1 forced single-hop
recurrence as `hmn_weave_mix_accum_rnn.py`, shrunk (d=16, n_layers=4,
n_heads=2, chunk_len=8, 20000 steps, trained from scratch — no checkpoint to
warm-start from at this size) so it trains in minutes instead of hours. Not
a substitute for the full run's results — just a cheap way to see whether
the relay learns anything at all before committing to the expensive config.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_accum_rnn_sanity.py --device mps
"""

hp = dict(
    d=16, n_layers=4, n_heads=2, V=274,
    block_type='single_attn',
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=20000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True,
    rmsnorm=True,
    name='hmn_accum_rnn_sanity', seed=50,

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=4, chunk_len=8, window_chunks=2, B=6, n_steps=20000, eval_every=2000,
             hops=1,
             weave_mix=[
                 dict(weight=1.0, pattern='batch'),
                 dict(weight=1.0, pattern='stream'),
                 dict(weight=1.0, pattern='interleave_delayed'),
             ]),
    ],
)
