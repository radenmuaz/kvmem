"""
`hmn_stitch_src1024_anchor.py` — `_grid_shapes`-style anchor sweep applied
to `hmn_stitch_src1024.py`'s suffix-recall design (see that file's own
docstring for the base mechanism: encode n_chunks chunks, then ONE query
recalling the SUFFIX of the source from `traj_suffix`'s
`Q(n_chunks-window_chunks, n_chunks)` span). That base config only ever
anchors the warmup at byte 0 of the window (`Q(2,4)` always starts
recalling from chunk 2's own first byte) — this config instead sweeps a
non-zero-biased BYTE-level anchor within each window, the same
`_grid_shapes` idea `hmn_notags_weave_anchor(_rope).py` already validated
for `batch`/`stream`/`interleave_delayed`, generalized here to the single
"suffix" shape.

Ported to the CURRENT (post-promotion) design and warm-started from
`hmn_notags_weave_anchor_rope`'s finished checkpoint (`V=271` opcode
vocab, `rope=True`, same architecture) rather than `hmn_weave_c64`'s
pre-promotion one (`V=274`, unsafe per CLAUDE.md's `_pretrained_ckpt`
caveat). `yarn` deliberately left at its default (`False`), NOT enabled
despite this config's much longer sequences (L~1000+ at n_chunks=16) —
`hmn_notags_weave_anchor_rope` was itself trained with `yarn=False`, and
`MHAttention`'s `freqs` buffer is `persistent=True` (included in the
state_dict), so warm-starting into a fresh `yarn=True` build would just
have its freshly-computed YaRN-scaled freqs silently overwritten by the
checkpoint's own non-YaRN buffer on load — enabling yarn here would be a
no-op at best, a confusing mismatch at worst. If long-sequence
extrapolation becomes a real question later, that needs its own from-
scratch (or explicitly-freqs-excluded) run, not a same-checkpoint warm-start.

Same 3-stage `n_chunks` ramp as the base config (`{2,4}` -> `{2,4,8}` ->
`{2,4,8,16}`), same `chunk_len=64`, `hops` unused (single query, `op_idx=0`
is always relay-exempt, matching the base config's own reasoning) — each
stage's entries now generated via `_grid_stitch` (chunk_len x window x
anchor sweep) instead of hand-written single-anchor DSL strings.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_stitch_src1024_anchor.py --device mps
"""


def _e_block(n, chunk_len):
    """`n` encode ops at `chunk_len` — E(len) sets chunk_len GLOBALLY for the
    whole DSL string and emits exactly one (E,S) pair; n-1 bare E's cover
    the rest (see hmn_notags_weave_anchor.py's identical helper)."""
    return f'E({chunk_len})' if n == 1 else f'E({chunk_len}) E{n - 1}'


def _grid_stitch(chunk_len, n_chunks, window_chunks, warmup_lens, n_anchors,
                 min_recall_len, rb_token, weight=1.0):
    """`_grid_shapes`-style anchor sweep for the single "suffix" query shape:
    for each warmup_len and each of `n_anchors` evenly-spaced byte-offset
    anchors INTO the window's own span (span_len = window_chunks*chunk_len),
    emits one weave_mix entry `E{n_chunks} Q(start,n_chunks,anchor,wl)` —
    `start = n_chunks - window_chunks`, matching traj_suffix's own span
    convention. No entry ever anchors at the window's own byte 0 alone
    (unless that's the only value min_recall_len/span_len allow)."""
    start = n_chunks - window_chunks
    span_len = window_chunks * chunk_len
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
            dsl = f'{_e_block(n_chunks, chunk_len)} Q({start},{n_chunks},{s},{wl}) {rb_token}'
            entries.append(dict(weight=weight, dsl=dsl))
    return entries


hp = dict(
    d=64, n_layers=8, n_heads=4, V=271,
    block_type='single_attn',
    # Optim settings copied from hmn_notags_weave_anchor_rope.py (its own
    # architecture/warm-start lineage), not this file's earlier ad hoc values.
    lr_max=1e-4, lr_min=1e-6, wd=0.0001,
    warmup_steps=5000, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=150000, cosine_T_mult=1,  # matches the longest stage's own n_steps (see hmn_weave_c64.py's own rationale)
    rope=True, null_kv=True,
    rmsnorm=True,
    name='hmn_stitch_src1024_anchor', seed=51,
    _pretrained_ckpt='logs/hmn_notags_weave_anchor_rope/checkpoints/stage1_best.pt',

    state_len=8, state_vocab_size=2,
    warmup_len=8,  # stage-level fallback default — unused, every entry's DSL sets its own wl
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    adaptive=True,
    adapt_signal='val_match',
    adapt_temp=1.0,
    adapt_ema_alpha=0.5,
    adapt_floor=0.05,

    # n_steps 10x'd from the original draft; eval_every scaled proportionally.
    # rb_token: B4 for stage0 (shortest L), B8 for stage1/stage2 (harder,
    # longer L) — mirrors hmn_notags_weave_anchor_rope.py's own escalating-
    # with-difficulty repeat_batch pattern (its B8/B16 pair), using the two
    # values requested here instead.
    curriculum=[
        dict(n_chunks=4, chunk_len=64, B=6, n_steps=100000, eval_every=20000, early_stop_mean=80.0,
             weave_mix=(
                 _grid_stitch(64, n_chunks=2, window_chunks=2, warmup_lens=[32],
                              n_anchors=2, min_recall_len=8, rb_token='B4')
                 + _grid_stitch(64, n_chunks=4, window_chunks=2, warmup_lens=[32],
                                n_anchors=3, min_recall_len=8, rb_token='B4')
                 + _grid_stitch(64, n_chunks=4, window_chunks=4, warmup_lens=[64],
                                n_anchors=2, min_recall_len=8, rb_token='B4')
             )),
        dict(n_chunks=8, chunk_len=64, B=6, n_steps=150000, eval_every=30000, early_stop_mean=80.0,
             weave_mix=(
                 _grid_stitch(64, n_chunks=2, window_chunks=2, warmup_lens=[32],
                              n_anchors=2, min_recall_len=8, rb_token='B8')
                 + _grid_stitch(64, n_chunks=4, window_chunks=4, warmup_lens=[64],
                                n_anchors=2, min_recall_len=8, rb_token='B8')
                 + _grid_stitch(64, n_chunks=8, window_chunks=4, warmup_lens=[64],
                                n_anchors=3, min_recall_len=8, rb_token='B8')
                 + _grid_stitch(64, n_chunks=8, window_chunks=8, warmup_lens=[96, 128],
                                n_anchors=2, min_recall_len=8, rb_token='B8')
             )),
        dict(n_chunks=16, chunk_len=64, B=6, n_steps=150000, eval_every=30000, early_stop_mean=80.0,
             weave_mix=(
                 _grid_stitch(64, n_chunks=4, window_chunks=4, warmup_lens=[64],
                              n_anchors=2, min_recall_len=8, rb_token='B8')
                 + _grid_stitch(64, n_chunks=8, window_chunks=8, warmup_lens=[96, 128],
                                n_anchors=2, min_recall_len=8, rb_token='B8')
                 + _grid_stitch(64, n_chunks=16, window_chunks=8, warmup_lens=[128],
                                n_anchors=3, min_recall_len=8, rb_token='B8')
                 + _grid_stitch(64, n_chunks=16, window_chunks=16, warmup_lens=[128, 192],
                                n_anchors=2, min_recall_len=8, rb_token='B8')
             )),
    ],
)
