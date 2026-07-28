"""
`hmn_single_recall_c64_dualrope.py` — redo of `hmn_single_recall_c64.py`
FROM SCRATCH under the new dual-clock RoPE mechanism (`dual_rope=True`,
`apply_rope_dual`/`_dual_positions`, `kvmem/hmn.py`). Necessary rebuild, not
optional: `dual_rope` changes what position values get fed into every
attention layer's rotation, so the existing `hmn_single_recall_c64`
checkpoint's weights are trained under a fundamentally different positional
scheme and cannot be warm-started from directly.

Why this mechanism exists: `kvmem/probe_positional_shortcut.py` measured
that `traj1`(`batch`)/`traj3`(`interleave_delayed`) — two DSL trajectories
sharing a byte-identical `E2` encode prefix but differing in query order —
were being resolved via PURE POSITION, not content: swapping a query slot's
warmup bytes for the wrong chunk's real bytes still produced that slot's
USUAL chunk's continuation (91.1% match to the wrong-but-usual chunk, 0.4%
to the right one). `dual_rope` tracks two separate position clocks instead
of one absolute index: `pos_state` only advances at STATE-emission events
and freezes everywhere else (so every query following the same encoding
pass sees an IDENTICAL relative distance to each STATE regardless of query
order — verified directly, offline, before trusting: batch's and
interleave_delayed's two queries now see literally the same macro distance
profile to both STATE0/STATE1), and `pos_local` resets to 0 at the start of
every encode/query block (preserving genuine local byte-order for coherent
generation). See `docs/HISTORY.md` §12 for the full derivation and the bug
this design caught in its own first draft.

This config: identical architecture/curriculum to `hmn_single_recall_c64`
(single chunk, `E1 Q(0,1)`, no query-order ambiguity exists here at all —
this is just the simplest possible task, establishing a clean base
checkpoint under the new positional scheme before testing the actual
multi-query case in `hmn_weave_c64_dualrope.py`). Not warm-started from
anything, same as the original.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_single_recall_c64_dualrope.py --device mps
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
    name='hmn_single_recall_c64_dualrope', seed=48,
    dual_rope=True,

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
