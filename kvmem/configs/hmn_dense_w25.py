"""
`hmn_dense_w25.py` — clone of `hmn_locate_nope_curriculum_dense.py` with one
change: every `_grid()` call now filters out `warmup_len` values below 25%
of that entry's own `chunk_len` (e.g. for `chunk_len=32`, the smallest
`warmup_len` kept is 8, not 2). Motivated directly by the qualitative
decode finding on `hmn_locate_nope_curriculum`'s stage1 checkpoint
(CLAUDE.md/docs/HISTORY.md): at `warmup_len=2` the model produced
near-random, non-printable garbage completely unrelated to the true
continuation, while the SAME shape at `warmup_len=8` produced close
near-misses (small byte transpositions) — an information-insufficiency
collapse, not a positional-shortcut or capacity problem. This config tests
whether dropping the shortest, most-ambiguous warmup_lens entirely (rather
than trying to have the model learn them) gives cleaner, faster
convergence on the lengths that remain.

Each `_grid()` call site below passes ONLY the already-filtered
`warmup_lens` list for its own `chunk_len` (e.g. `_grid(64, [16, 24], ...)`,
not `_grid(64, [2, 4, 8, 12, 16, 24], ...)` with a silent runtime filter
discarding the first four) — listing values that get thrown away is
misleading to read, even if the end result is correct. `min_warmup_frac`
is kept on `_grid` only as a defensive ASSERTION (raises if a call site's
list contains a value below the floor), catching a future mistake rather
than papering over one.

Per-length floors (min_wl = ceil(chunk_len*0.25)), and what each call site
passes:
  chunk_len=8  (min_wl=2):  [2, 3, 4]   -> unchanged (2 already >= floor)
  chunk_len=16 (min_wl=4):  [4, 6, 8]   -> was [2,3,4,6,8]
  chunk_len=32 (min_wl=8):  [8, 12, 16] -> was [2,4,6,8,12,16]
  chunk_len=64 (min_wl=16): [16, 24]    -> was [2,4,8,12,16,24]
Rehearsal entries use the floor for the length being rehearsed, not the
current stage's chunk_len (e.g. stage3's len=32 rehearsal passes
[8,12,16], its len=16 rehearsal passes [4,6,8], its len=8 rehearsal is
unchanged).

Same architecture/adaptive/early_stop/lr/wd as the base config — only the
weave_mix grids shrink (fewer, less-ambiguous entries per stage).

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_dense_w25.py --device mps
"""
import math


def _grid(chunk_len, warmup_lens, n_anchors, min_recall_len, rb_token, weight=1.0, min_warmup_frac=0.0):
    """Generates weave_mix entries for one length: `n_anchors` evenly-spaced
    query_start values per (chunk_len, warmup_len) pair (deduped/clamped
    when the valid range is too small to fit them all distinctly). Also
    used for REHEARSAL by passing a smaller `n_anchors` and `weight=0.5`.
    `min_warmup_frac`: DEFENSIVE ASSERTION only, not a filter — every call
    site below is expected to already pass a pre-filtered `warmup_lens`
    list (see module docstring for why: listing values that get silently
    discarded is misleading), so this just catches a future mistake."""
    if min_warmup_frac > 0:
        min_wl = math.ceil(chunk_len * min_warmup_frac)
        bad = [wl for wl in warmup_lens if wl < min_wl]
        assert not bad, (
            f'_grid(chunk_len={chunk_len}, ...): warmup_lens {bad} are below the '
            f'min_warmup_frac={min_warmup_frac} floor ({min_wl}) — pass an already-'
            f'filtered warmup_lens list instead of relying on a runtime filter')
    entries = []
    for wl in warmup_lens:
        max_start = chunk_len - min_recall_len - wl
        if max_start < 0:
            continue
        if n_anchors == 1 or max_start == 0:
            starts = [0]
        else:
            starts = sorted(set(round(i * max_start / (n_anchors - 1)) for i in range(n_anchors)))
        for s in starts:
            entries.append(dict(weight=weight, dsl=f'E({chunk_len}) Q(0,1,{s},{wl}) {rb_token}'))
    return entries


hp = dict(
    d=64, n_layers=8, n_heads=4, V=274,
    block_type='single_attn',
    lr_max=1e-4, lr_min=1e-6, wd=1e-5,
    warmup_steps=500, log_every=500,
    lr_schedule='cosine_restarts',
    cosine_T0=180000, cosine_T_mult=1,
    rope=False,
    null_kv=True,
    rmsnorm=True,
    name='hmn_dense_w25', seed=48,

    adaptive=True,
    adapt_signal='val_match',
    adapt_temp=1.0,
    adapt_ema_alpha=0.5,
    adapt_floor=0.05,

    state_len=8, state_vocab_size=2,
    warmup_len=8,  # stage-level fallback default — unused here, every entry's DSL sets its own wl
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=1, chunk_len=8, B=16, n_steps=480000, eval_every=24000, early_stop_mean=80.0,
             weave_mix=_grid(8, [2, 3, 4], n_anchors=4, min_recall_len=4, rb_token='B8',
                             min_warmup_frac=0.25)),

        dict(n_chunks=1, chunk_len=16, B=12, n_steps=720000, eval_every=36000, early_stop_mean=80.0,
             weave_mix=(
                 _grid(16, [4, 6, 8], n_anchors=4, min_recall_len=4, rb_token='B8',
                      min_warmup_frac=0.25)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5,
                        min_warmup_frac=0.25)
             )),

        dict(n_chunks=1, chunk_len=32, B=6, n_steps=480000, eval_every=24000, early_stop_mean=80.0,
             weave_mix=(
                 _grid(32, [8, 12, 16], n_anchors=4, min_recall_len=4, rb_token='B16',
                      min_warmup_frac=0.25)
                 + _grid(16, [4, 6, 8], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5,
                        min_warmup_frac=0.25)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5,
                        min_warmup_frac=0.25)
             )),

        dict(n_chunks=1, chunk_len=64, B=4, n_steps=720000, eval_every=36000, early_stop_mean=80.0,
             weave_mix=(
                 _grid(64, [16, 24], n_anchors=4, min_recall_len=4, rb_token='B16',
                      min_warmup_frac=0.25)
                 + _grid(32, [8, 12, 16], n_anchors=2, min_recall_len=4, rb_token='B16', weight=0.5,
                        min_warmup_frac=0.25)
                 + _grid(16, [4, 6, 8], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5,
                        min_warmup_frac=0.25)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5,
                        min_warmup_frac=0.25)
             )),
    ],
)
