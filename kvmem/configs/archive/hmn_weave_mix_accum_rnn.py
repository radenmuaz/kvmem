"""
`weave_mix` + `hops=1` — same training-mix design as `hmn_weave_mix.py`
(`batch`/`stream`/`interleave_delayed`, uniform weight), but with `hops=1`
set explicitly instead of `hmn_weave_mix.py`'s implicit default (`hops=-1`,
unbounded/routing-style — that config never sets `hops` at all, so it keeps
permanent, unrestricted attention access to every encoding-pass STATE
regardless of op_idx, and its own relay exception is layered on top rather
than being the only channel).

`chunk_mask_fb_traj`'s `hops` parameter now controls both the relay
window AND recurrent mode together (no separate flag — see that function's
docstring, 2026-07-15): `hops=1` additionally blocks every op after the
first (`op_idx>0`) from all encoding-pass STATEs, leaving the single-hop
relay as its ONLY channel — verified directly for the `stream` pattern
(op_idx=0 keeps access to its own already-encoded chunks; op_idx=1/2 show
0 visible against every encoding STATE, once hops>=1).

Why this variant matters beyond `hmn_recall_queue.py` (which itself already
sets `hops=1`, so it automatically gets this same corrected recurrent
masking with no config change needed): `hmn_recall_queue.py` tests whether
forced accumulation is learnable AT ALL on `hop`'s own fixed 3-query
schedule. THIS config additionally tests whether that forced accumulation
generalizes across trajectory shapes (`stream`'s interleaved encode/query
order, `interleave_delayed`'s reversed query order) the same way
`hmn_weave_mix.py` tested generalization for the (unbounded) routing-style
relay. If `repeat_query`/`long_hop_recovery` still fail after THIS run,
that argues genuinely against the relay mechanism itself (not a
training-exposure gap and not an unintended encoding-pass bypass) — the
cleanest version of that question this project has built so far.

Warm-started from the same `hmn_single_recall_c64.py` checkpoint as
`hmn_weave_mix.py`, for the same reason (`hmn_routing_4to1_state`/`solo`
is treated as an archived experiment with no checkpoint on disk;
`hmn_single_recall_c64` is a same-architecture, already-converged
(100% val match) base to build op_idx=0's own skill from, and `hop`'s own
checkpoint is the weaker, non-reproduced run — see `hmn_weave_mix.py`'s
docstring for the full caveat, unchanged here).

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_weave_mix_accum_rnn.py --device mps
"""

hp = dict(
    d=64, n_layers=8, n_heads=4, V=274,
    block_type='single_attn',
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=160000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True,
    rmsnorm=True,
    name='hmn_weave_mix_accum_rnn', seed=50,
    _pretrained_ckpt='logs/hmn_single_recall_c64/checkpoints/stage0_best.pt',

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=4, chunk_len=16, window_chunks=2, B=6, n_steps=160000, eval_every=10000,
             hops=1,
             weave_mix=[
                 dict(weight=1.0, pattern='batch'),
                 dict(weight=1.0, pattern='stream'),
                 dict(weight=1.0, pattern='interleave_delayed'),
             ]),
    ],
)
