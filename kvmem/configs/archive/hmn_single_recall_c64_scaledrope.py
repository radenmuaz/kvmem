"""
`hmn_single_recall_c64_scaledrope.py` — redo of `hmn_single_recall_c64.py`
FROM SCRATCH under `rope_state_scale` (`kvmem/hmn.py`), the mechanism that
SUPERSEDES the earlier `dual_rope`/`_dual_positions` design (see
`docs/HISTORY.md` §12). Necessary rebuild, not optional: this changes what
position values feed every attention layer's rotation, so the existing
`hmn_single_recall_c64` checkpoint (trained under plain single-clock RoPE)
cannot be warm-started from directly — same reasoning as `_dualrope`, just
a different replacement mechanism.

Why this superseded dual_rope: `dual_rope` needed a two-clock scheme with
per-block-type reset bookkeeping (encode STATE freezes a macro clock,
non-STATE content resets a local clock at every block boundary) — real
surface area that already produced one bug (an early draft advanced the
macro clock on a query's own recall-STATE row too, silently recreating the
very order-dependence it was supposed to remove). Discussion also
surfaced a real concern about the STATE register's own cyclic token IDs
(`_cyclic_state_ids` — only `state_vocab_size` distinct IDs repeating
through `state_len` slots) needing careful positional disambiguation that
a reset-heavy scheme risks getting subtly wrong.

`rope_state_scale` (`_scaled_state_positions`, `kvmem/hmn.py`) is simpler
and more robust: ONE single absolute clock (identical to plain RoPE, zero
special-casing) for every non-STATE token, and STATE-region tokens' real
index divided by `rope_state_scale` (1e6 here). Within one STATE region the
compressed values still preserve the same relative ordering as the real
indices (slot k > slot k-1, just compressed) — cyclic-ID slot
disambiguation via position is untouched — while cross-query distance to
any STATE becomes numerically negligible rather than exactly frozen by
construction: verified offline before trusting (see kvmem/hmn.py's
`_scaled_state_positions` docstring) that `batch`'s and
`interleave_delayed`'s two queries see distance-to-STATE0/STATE1 profiles
identical to 5+ decimal places regardless of query order, at 1e6 scale.

This config: identical architecture/curriculum to `hmn_single_recall_c64`
(single chunk, `E1 Q(0,1)`, no query-order ambiguity exists here at all —
establishing a clean base checkpoint under the new scheme before testing
the actual multi-query case in `hmn_weave_c64_scaledrope.py`). Not
warm-started from anything.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_single_recall_c64_scaledrope.py --device mps
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
    name='hmn_single_recall_c64_scaledrope', seed=48,
    rope_state_scale=1e6,

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
