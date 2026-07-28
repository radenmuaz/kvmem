"""
`hmn_notags_w25_rope.py` — clone of `hmn_notags_w25.py` with `rope=True`
instead of `rope=False`. Everything else (architecture, curriculum, the
25%-of-chunk_len minimum warmup_len floor, adaptive/early_stop machinery,
DSL strings) is identical — only this one hp flag differs.

Question this asks: `hmn_notags_w25`/`hmn_notags_locate` both test the
no-chat-tags/opcode design under NoPE specifically (companion to
`hmn_locate_nope_curriculum_dense`'s own NoPE-vs-tags question). This
config isolates whether RoPE (rather than no positional encoding at all)
changes the picture for the same tag-free/opcode/end-of-turn-STATE design
— i.e. is NoPE load-bearing for this design's own results, or incidental.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_notags_w25_rope.py --device mps
"""
import math


def _grid(chunk_len, warmup_lens, n_anchors, min_recall_len, rb_token, weight=1.0, min_warmup_frac=0.0):
    """Generates weave_mix entries for one length: `n_anchors` evenly-spaced
    query_start values per (chunk_len, warmup_len) pair (deduped/clamped
    when the valid range is too small to fit them all distinctly). Also
    used for REHEARSAL by passing a smaller `n_anchors` and `weight=0.5`.
    `min_warmup_frac`: DEFENSIVE ASSERTION only, not a filter — every call
    site below is expected to already pass a pre-filtered `warmup_lens`
    list (listing values that get silently discarded is misleading), so
    this just catches a future mistake."""
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
    d=64, n_layers=8, n_heads=4, V=271,
    block_type='single_attn',
    lr_max=1e-4, wd=1e-5,
    warmup_steps=1000, log_every=2000,
    rope=True,
    null_kv=True,
    rmsnorm=True,
    name='hmn_notags_w25_rope', seed=48,

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
