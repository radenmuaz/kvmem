"""
`hmn_notags_weave_anchor_rope.py` — clone of `hmn_notags_weave_anchor.py`
with `rope=True` instead of `rope=False`, and warm-started from `hmn_
notags_w25_rope`'s checkpoint instead of `hmn_notags_w25`'s.

Context: `hmn_notags_weave_anchor.py` tests whether NoPE + VARYING RECALL
ANCHORS fixes the `batch`/`interleave_delayed` positional shortcut that
every prior RoPE-based fix (`dual_rope`/`rope_state_scale`/`relpos`, all
archived/deprecated) failed to solve. But a live head-to-head comparison
(`hmn_notags_w25` vs `hmn_notags_w25_rope`, same architecture/curriculum,
only `rope` differing) found RoPE converges dramatically faster and higher
than NoPE at EVERY comparable stage/step (e.g. stage2: RoPE early-stopped
at 90.0% by step 72000 vs NoPE's 62.3% at the same step; stage3: RoPE hit
64.9% by step 36000, matching NoPE's best-ever result reached only after
288000 steps). That raises the obvious follow-up this config answers:
does RoPE PLUS the same anchor-variation manipulation do even better on
`batch`/`interleave_delayed` specifically than either RoPE alone or NoPE+
anchor-variation alone? This isolates whether anchor variation and RoPE
are complementary (fixing different parts of the shortcut) or redundant
(RoPE alone already fixes what anchor variation was meant to fix).

Everything else — architecture, two-stage curriculum (`hops=1`, nc=2/wc=1
then nc=4/wc=2), the `_grid_shapes` chunk_len(8/16/32/64) x anchor sweep
across `batch`/`stream`/`interleave_delayed` — is identical to `hmn_
notags_weave_anchor.py`; only `rope` and `_pretrained_ckpt` differ.

Comparison targets: (1) `hmn_notags_weave_anchor.py`'s own NoPE+anchor
results, (2) `hmn_weave_c64_scaledrope`/`_dualrope`/`_relpos` (archived)
which all failed to move `batch`/`interleave_delayed` off their ceiling
under RoPE-family fixes WITHOUT anchor variation.

Run (never two jobs at once — queued ahead of `hmn_notags_weave_anchor`,
behind `hmn_notags_w25_rope`):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_notags_weave_anchor_rope.py --device mps
"""
import math


def _e_block(n, chunk_len):
    """`n` encode ops at `chunk_len` — E(len) sets chunk_len GLOBALLY for the
    whole DSL string (parse_traj_dsl) and emits exactly one (E,S) pair, so
    a plain bare 'E{n}' token (no chunk_len embedded) would silently fall
    back to the stage-level chunk_len instead of the one being swept here.
    One E(len) covers the first chunk; n-1 bare E's cover the rest."""
    return f'E({chunk_len})' if n == 1 else f'E({chunk_len}) E{n - 1}'


def _dsl_batch(n_chunks, wc, chunk_len, s, wl, rb):
    spans = [(i, i + wc) for i in range(n_chunks - wc + 1)]
    qs = [f'Q({a},{b},{s},{wl})' for a, b in spans]
    return _e_block(n_chunks, chunk_len) + ' ' + ' S '.join(qs) + f' {rb}'


def _dsl_stream(n_chunks, wc, chunk_len, s, wl, rb):
    spans = [(i, i + wc) for i in range(n_chunks - wc + 1)]
    parts = [_e_block(wc, chunk_len)]
    for i, (a, b) in enumerate(spans):
        if i > 0:
            parts.append('E')
        parts.append(f'Q({a},{b},{s},{wl})')
        if i < len(spans) - 1:
            parts.append('S')
    parts.append(rb)
    return ' '.join(parts)


def _dsl_interleave_delayed(n_chunks, wc, chunk_len, s, wl, rb):
    spans = [(i, i + wc) for i in range(n_chunks - wc + 1)]
    qs = [f'Q({a},{b},{s},{wl})' for a, b in reversed(spans)]
    return _e_block(n_chunks, chunk_len) + ' ' + ' S '.join(qs) + f' {rb}'


def _grid_shapes(chunk_len, n_chunks, window_chunks, warmup_lens, n_anchors,
                  min_recall_len, rb_token, weight=1.0, min_warmup_frac=0.25):
    """`_grid`-style anchor/length sweep (see `hmn_dense_w25.py`), generalized
    to batch/stream/interleave_delayed's multi-query shape: for each
    warmup_len and each of `n_anchors` evenly-spaced anchor offsets INTO a
    query's own span (span_len = window_chunks*chunk_len), emits THREE
    weave_mix entries (one per named shape) all sharing that (warmup_len,
    anchor) pair — so no entry ever anchors its warmup at the span's own
    byte 0, the shortcut kvmem/probe_positional_shortcut.py measured.
    `min_warmup_frac`: defensive assertion only (see hmn_dense_w25.py),
    every call site below already passes pre-filtered warmup_lens."""
    span_len = window_chunks * chunk_len
    if min_warmup_frac > 0:
        min_wl = math.ceil(span_len * min_warmup_frac)
        bad = [wl for wl in warmup_lens if wl < min_wl]
        assert not bad, (
            f'_grid_shapes(chunk_len={chunk_len}, window_chunks={window_chunks}, ...): '
            f'warmup_lens {bad} are below the min_warmup_frac={min_warmup_frac} floor '
            f'({min_wl}) — pass an already-filtered warmup_lens list')
    entries = []
    for wl in warmup_lens:
        max_s = span_len - min_recall_len - wl
        if max_s < 0:
            continue
        if n_anchors == 1 or max_s == 0:
            anchors = [0]
        else:
            anchors = sorted(set(round(i * max_s / (n_anchors - 1)) for i in range(n_anchors)))
        for s in anchors:
            entries.append(dict(weight=weight, dsl=_dsl_batch(n_chunks, window_chunks, chunk_len, s, wl, rb_token)))
            entries.append(dict(weight=weight, dsl=_dsl_stream(n_chunks, window_chunks, chunk_len, s, wl, rb_token)))
            entries.append(dict(weight=weight, dsl=_dsl_interleave_delayed(n_chunks, window_chunks, chunk_len, s, wl, rb_token)))
    return entries


hp = dict(
    d=64, n_layers=8, n_heads=4, V=271,
    block_type='single_attn',
    lr_max=1e-4, lr_min=1e-6, wd=0.0001,
    warmup_steps=5000, log_every=2000,
    lr_schedule='cosine_restarts',
    cosine_T0=160000, cosine_T_mult=1,
    rope=True,
    null_kv=True,
    rmsnorm=True,
    name='hmn_notags_weave_anchor_rope', seed=51,
    _pretrained_ckpt='logs/hmn_notags_w25_rope/checkpoints/stage3_best.pt',

    state_len=8, state_vocab_size=2,
    warmup_len=8,  # stage-level fallback default — unused, every entry's DSL sets its own wl
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=2, chunk_len=64, B=4, n_steps=80000, eval_every=10000, hops=1,
             weave_mix=(
                 _grid_shapes(8,  n_chunks=2, window_chunks=1, warmup_lens=[2, 3, 4],
                              n_anchors=3, min_recall_len=2, rb_token='B8')
                 + _grid_shapes(16, n_chunks=2, window_chunks=1, warmup_lens=[4, 6],
                                n_anchors=3, min_recall_len=2, rb_token='B8')
                 + _grid_shapes(32, n_chunks=2, window_chunks=1, warmup_lens=[8, 12],
                                n_anchors=3, min_recall_len=4, rb_token='B8')
                 + _grid_shapes(64, n_chunks=2, window_chunks=1, warmup_lens=[16, 24],
                                n_anchors=3, min_recall_len=4, rb_token='B8')
             )),

        dict(n_chunks=4, chunk_len=64, B=2, n_steps=160000, eval_every=10000, hops=1,
             weave_mix=(
                 _grid_shapes(8,  n_chunks=4, window_chunks=2, warmup_lens=[4, 6],
                              n_anchors=3, min_recall_len=4, rb_token='B16')
                 + _grid_shapes(16, n_chunks=4, window_chunks=2, warmup_lens=[8, 12],
                                n_anchors=3, min_recall_len=4, rb_token='B16')
                 + _grid_shapes(32, n_chunks=4, window_chunks=2, warmup_lens=[16, 24],
                                n_anchors=3, min_recall_len=4, rb_token='B16')
                 + _grid_shapes(64, n_chunks=4, window_chunks=2, warmup_lens=[32, 48],
                                n_anchors=3, min_recall_len=4, rb_token='B16')
             )),
    ],
)
