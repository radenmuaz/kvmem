"""
`hmn_single_recall_c64.py` — a more literal "single recall" than the
config that originally held this name (renamed to
`hmn_routing_4to1_state.py` earlier this session, since it actually had
n_chunks=4 with routing across 4 encoding STATEs). THIS config is truly
single: n_chunks=1, chain_steps=[(0,1)] — one chunk, one STATE, one
query/response, nothing chained, nothing to route across.

Same shape as `hmn_squeeze_sanity_bigmodel_n4.py` (single-register layout)
but using `hmn_routing_4to1_state.py`'s (`solo`'s) proven architecture
(d=64, n_layers=8, n_heads=4, state_len=8) instead of a shrunk/tiny one,
and `chunk_len=64` — a modest bump from `solo`'s own `chunk_len=16`, not
the long-sequence (1024+) regime `squeeze`'s sweet-spot configs use. Plain
`data_kind='random'` (unset, matching `solo`'s own default) — this is not
a compression test, just the simplest possible encode/recall task at a
slightly longer chunk than `solo` uses, with `solo`'s own already-proven
capacity (128x+ nominal headroom at this chunk_len, see
`nominal_capacity_accounting` in `kvmem/eval_compression.py` — capacity is
not remotely the constraint here).

Not warm-started from anything — `solo`'s own checkpoint was trained at
chunk_len=16, n_chunks=4, chain_steps=[(0,2)], a different packed-sequence
shape (position offsets don't line up with this single-chunk, chunk_len=64
layout), so there's nothing valid to transfer weights from positionally;
trains from scratch like `solo` itself did.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_single_recall_c64.py --device mps

Early-stopped 2026-07-15: eval_mean hit 100.0% at step 70000, dipped to 95.83%
(noise) at 80000, then held 100.0% at both 90000 and 100000 (checkpoint
`stage0_best.pt` is from this run's best eval). Killed at step ~100000 rather
than running the full 160000 — the last 60000 steps were pure plateau, no
further gain. `cosine_T0`/`n_steps` trimmed to 100000 below to match.

Ported to the `weave_mix`+`dsl=` path (`dsl='E1 Q(0,1)'`) instead of
`chain_steps=[(0,1)]` — verified byte-identical mask/positions/tags against
the old `chunk_positions_hop` path before switching (only cosmetic diff:
`chunk_positions_traj`'s rec_blocks carry an extra `op_idx` field). Purely a
config-definition change — the existing `stage0_best.pt` checkpoint (used as
the warm-start for `hmn_weave_c64`/`hmn_stitch_src1024`/`hmn_recall_queue`/
`hmn_weave_mix` and others) is untouched, since positions/masks are
byte-identical regardless of which code path built them.
"""

hp = dict(
    d=64, n_layers=8, n_heads=4, V=274,
    block_type='single_attn',
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=100000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True,
    rmsnorm=True,
    name='hmn_single_recall_c64', seed=48,

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=1, chunk_len=64, B=6, n_steps=100000, eval_every=10000,
             weave_mix=[dict(weight=1.0, dsl='E1 Q(0,1)')]),
    ],
)
