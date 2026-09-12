"""
`hmn_notags_weave_anchor.py` — tests whether NoPE (no positional encoding
at all, `rope=False`, the notags/opcode design's default) combined with
VARYING RECALL ANCHORS actually fixes the `batch`/`interleave_delayed`
positional shortcut that every RoPE-based fix (`dual_rope`, `rope_state_
scale`, `relpos` — all now archived/deprecated, see CLAUDE.md's
"Positional shortcut" entry and docs/HISTORY.md §12-13) failed to solve.
`kvmem/probe_positional_shortcut.py` found the model resolves `batch`/
`interleave_delayed`'s shared queries via pure attention POSITION, not
warmup content (91.1% match to the wrong-but-positionally-usual chunk vs.
0.4% to the chunk whose real bytes were actually given) — every prior fix
attacked this by changing the position ENCODING itself. This config
attacks it a different way: never let there BE a single fixed position a
query's warmup lands at. `hmn_dense_w25`/`hmn_notags_w25`'s own `_grid`
helper already proved this idea works for single-chunk recall (varying
`warmup_start` within a chunk so the model can't shortcut on absolute
position); this config is that same idea generalized to `chunk_positions_
traj`'s multi-query `Q(s,e,warmup_start,warmup_len)` DSL token — every
`batch`/`stream`/`interleave_delayed` entry below anchors its queries'
warmup at a swept, non-zero offset INTO their own span (never always byte
0 of the span), across a `_grid`-style sweep of `chunk_len` and anchor
position.

**Base**: `hmn_weave_c64.py`'s architecture/curriculum shape (`hops=1`,
two-stage difficulty ramp — `n_chunks=2/window_chunks=1` then `n_chunks=4/
window_chunks=2` — the exact shape whose `batch`/`interleave_delayed`
numbers are what every positional-shortcut fix attempt was measured
against). Differs from `hmn_weave_c64.py` in:
  - `rope=False` (NoPE, not RoPE) and `V=271` (opcode vocab, not the old
    tagged `V=274`) — this is a `kvmem.hmn` (post-promotion) config, not a
    `hmn_v4_backup.py`-era one.
  - Each stage's `weave_mix` is a `_grid_shapes`-style sweep across
    `chunk_len in {8,16,32,64}` x a few anchor offsets x the three named
    shapes, instead of `hmn_weave_c64.py`'s fixed single-chunk_len/no-
    anchor-variation entries — this is the actual manipulation under test.
  - `_pretrained_ckpt` points at `hmn_notags_w25`'s checkpoint, not
    `hmn_single_recall_c64`'s — `hmn_notags_w25` is itself a `kvmem.hmn`
    (opcode-vocab, NoPE) run, so this is a same-vocab, same-mechanism
    warm-start (unlike warm-starting from any pre-promotion checkpoint,
    which CLAUDE.md's `_pretrained_ckpt` caveat flags as unsafe).

Comparison target: `hmn_weave_c64_scaledrope`/`hmn_weave_c64_dualrope`/
`hmn_weave_c64_relpos` (archived) all measured `batch`/`interleave_
delayed` match% at this same `nc=4/wc=2` shape and all failed to move it
meaningfully. If this config's `batch`/`interleave_delayed` numbers clear
that same ceiling, the anchor-variation-under-NoPE angle succeeded where
changing the position encoding itself did not.

Run (never two jobs at once — queued behind `hmn_notags_w25` and
`hmn_notags_w25_rope`, launches automatically once both finish):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_notags_weave_anchor.py --device mps
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
    rope=False,
    null_kv=True,
    rmsnorm=True,
    name='hmn_notags_weave_anchor', seed=51,
    _pretrained_ckpt='logs/hmn_notags_w25/checkpoints/stage3_best.pt',

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
