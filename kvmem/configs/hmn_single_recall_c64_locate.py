"""
`hmn_single_recall_c64_locate.py` — tests whether DATA DIVERSITY ALONE (no
architectural position-compression) can force content-addressed retrieval,
as an alternative/complementary angle to the `dual_rope`/`rope_state_scale`/
`relpos` mechanisms (`docs/HISTORY.md` §12-13). `rope_state_scale` is
DELIBERATELY NOT SET here (plain RoPE, unmodified) — an isolated test of
whether curriculum diversity by itself reduces reliance on the query-slot
positional shortcut `kvmem/probe_positional_shortcut.py` measured, without
conflating the result with the architectural fix being tested elsewhere.

Mechanism: `traj_locate_and_continue(query_start)` (`kvmem/hmn.py`) builds a
single-chunk (`n_chunks=1`) trajectory whose warmup/query excerpt is a real
ground-truth SEGMENT starting at an arbitrary BYTE OFFSET within the source
— not always byte 0 the way every other single-chunk config in this
project has trained — response covers everything after that excerpt
through the TRUE END of the source (variable length, not a fixed
constant). The model has to LOCATE the excerpt (which could sit anywhere)
via content, not a fixed/predictable position, before it can continue.
Required three real DSL/position-builder extensions to make this
expressible at all (previously genuinely unsupported, confirmed before
implementing — see the DSL grammar comment and `_scaled_state_positions`-
adjacent history in `kvmem/hmn.py`):
  - `Q(s,e,w)` — a 3rd argument on the query token giving the BYTE offset
    `w` within the span where the excerpt starts (`Q(s,e)` unchanged,
    defaults to `w=0`, so every existing config/checkpoint is unaffected).
  - `E(len)` — chunk_len embedded directly in the DSL string itself
    (`E1`/`E4` unchanged — still "n same-length chunks using the stage
    default"; `E(64)` is new syntax, distinguished by the parens, meaning
    "one chunk of exactly 64 bytes") — lets each entry in a mix carry its
    own source length with no external per-entry key needed.
  - Per-trajectory `warmup_len` override in the `weave_mix` dispatch
    (previously stage-wide only, shared by every entry in the mix) —
    needed since this config mixes multiple query-excerpt lengths too.
All three verified with direct numerical checks before trusting them
(byte-content correctness of the warmup/response slicing re-derived
independently and compared against `make_batch_tagged`'s actual output;
the boundary assert in `chunk_positions_traj` confirmed to fire exactly at
the invalid `query_start` and not before) — not just "does it run."

**The grid** (kept deliberately small and discrete per explicit guidance —
too many simultaneously-varying shapes makes optimization harder, not
better): `src_len` (chunk_len) in {32, 64}, `query_len` (warmup_len) in
{8, 16}, and exactly 2 anchor points per valid `(src_len, query_len)` pair
— `query_start=0` (today's ordinary behavior, kept as a baseline in every
group) and `query_start=max_valid` (the hardest case, excerpt near the very
end) — where `max_valid = src_len - min_recall_len(8) - query_len`. 4
valid pairs x 2 anchors = 8 trajectory entries total:
  (32, 8):  start in {0, 16}
  (32, 16): start in {0, 8}
  (64, 8):  start in {0, 48}
  (64, 16): start in {0, 40}

Warm-started from the ORIGINAL `hmn_single_recall_c64` checkpoint (plain
RoPE, no `dual_rope`/`rope_state_scale`) — valid despite the varying
chunk_len/query_start shapes here, since RoPE computes position on the
fly (no learned absolute position table) and the (32,8,0)/(64,8,0) entries
in this mix are byte-shape-identical to what that checkpoint already
trained on for at least part of the mix.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_single_recall_c64_locate.py --device mps
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
    name='hmn_single_recall_c64_locate', seed=48,
    _pretrained_ckpt='logs/hmn_single_recall_c64/checkpoints/stage0_best.pt',

    state_len=8, state_vocab_size=2,
    warmup_len=8,  # stage-level default, overridden per-entry below
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=1, chunk_len=64, B=6, n_steps=100000, eval_every=10000,
             weave_mix=[
                 dict(weight=1.0, dsl='E(32) Q(0,1,0,8)'),
                 dict(weight=1.0, dsl='E(32) Q(0,1,16,8)'),
                 dict(weight=1.0, dsl='E(32) Q(0,1,0,16)'),
                 dict(weight=1.0, dsl='E(32) Q(0,1,8,16)'),
                 dict(weight=1.0, dsl='E(64) Q(0,1,0,8)'),
                 dict(weight=1.0, dsl='E(64) Q(0,1,48,8)'),
                 dict(weight=1.0, dsl='E(64) Q(0,1,0,16)'),
                 dict(weight=1.0, dsl='E(64) Q(0,1,40,16)'),
             ]),
    ],
)
