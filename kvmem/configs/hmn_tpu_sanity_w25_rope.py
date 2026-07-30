"""
`hmn_tpu_sanity_w25_rope.py` — clone of `hmn_tpu_sanity_w25.py` with
`rope=True` instead of `rope=False`. Everything else identical (`state_
vocab_size=1`, `lr_max=1e-4`, same curriculum/batch sizes/architecture).

Motivation: CLAUDE.md's "Positional shortcut" entry records a live
head-to-head (`hmn_notags_w25` vs `hmn_notags_w25_rope`, the same
architecture/curriculum this file's own base config already mirrors) where
RoPE converged dramatically faster/higher than NoPE at every comparable
stage/step — this config re-runs that same comparison at THIS (d=128,
n_layers=16) scale, on real TPU hardware, using the just-fixed lr_max=1e-4
(the NoPE run's own step-5000/10000 match% — 21.7%/17.4% — is the direct
comparison point).

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_tpu_sanity_w25_rope.py --device tpu
"""
import math


def _grid(chunk_len, warmup_lens, n_anchors, min_recall_len, rb_token, weight=1.0, min_warmup_frac=0.0):
    """Verbatim copy of hmn_notags_w25.py's own `_grid` — see that file for
    the full docstring. Generates weave_mix entries for one length: `n_anchors`
    evenly-spaced query_start values per (chunk_len, warmup_len) pair."""
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
    d=128, n_layers=16, n_heads=8, V=271,
    block_type='single_attn',
    lr_max=1e-4, wd=1e-5,
    warmup_steps=1000, log_every=200,
    rope=True,  # ABLATED from False — see module docstring
    null_kv=True,
    rmsnorm=True,
    grad_checkpoint='block',
    name='hmn_tpu_sanity_w25_rope', seed=48,

    adaptive=True,
    adapt_signal='val_match',
    adapt_temp=1.0,
    adapt_ema_alpha=0.5,
    adapt_floor=0.05,

    # DIAGNOSTIC: loss=NaN from step 1 with bf16 autocast (default) — same architecture/
    # settings that trained cleanly with rope=False. no_autocast=True forces fp32 to
    # test whether this is a precision issue specific to RoPE, mirroring bug 5's own
    # bf16-vs-fp32 diagnostic (CLAUDE.md's TPU port entry).
    no_autocast=True,
    state_len=4, state_vocab_size=1,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    bucket_lengths=True,
    max_shape_buckets=8,
    token_budget=4_000_000,
    attn_sq_budget=500_000_000,

    curriculum=[
        dict(n_chunks=1, chunk_len=8, B=16, n_steps=30000, eval_every=5000,
             weave_mix=_grid(8, [2, 3, 4], n_anchors=4, min_recall_len=4, rb_token='B8',
                             min_warmup_frac=0.25)),

        dict(n_chunks=1, chunk_len=16, B=12, n_steps=40000, eval_every=5000,
             weave_mix=(
                 _grid(16, [4, 6, 8], n_anchors=4, min_recall_len=4, rb_token='B8',
                      min_warmup_frac=0.25)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5,
                        min_warmup_frac=0.25)
             )),

        dict(n_chunks=1, chunk_len=32, B=6, n_steps=30000, eval_every=5000,
             weave_mix=(
                 _grid(32, [8, 12, 16], n_anchors=4, min_recall_len=4, rb_token='B16',
                      min_warmup_frac=0.25)
                 + _grid(16, [4, 6, 8], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5,
                        min_warmup_frac=0.25)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5,
                        min_warmup_frac=0.25)
             )),

        dict(n_chunks=1, chunk_len=64, B=4, n_steps=40000, eval_every=5000,
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
