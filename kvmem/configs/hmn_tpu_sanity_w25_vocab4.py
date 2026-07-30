"""
`hmn_tpu_sanity_w25_vocab4.py` — clone of `hmn_tpu_sanity_w25.py` with
`state_vocab_size=4` instead of `1`. This is the STATE-vocab ablation
CLAUDE.md's TPU port entry flags as the natural follow-up: `state_vocab_
size=1` means `_cyclic_state_ids` (`kvmem/hmn.py`) emits the SAME value
token at every STATE slot, so under NoPE (`rope=False`, unchanged here)
position within a STATE block is recoverable ONLY through causal depth —
the model must count. `state_vocab_size=4` gives each STATE slot a
richer, more distinguishable per-slot token identity (still cyclic —
`state_len=4` means every slot gets its OWN distinct value id at
vocab_size=4, one full period), testing whether that per-slot signal
measurably changes convergence speed/ceiling relative to the vocab=1
baseline, everything else (architecture, lr_max=1e-4, curriculum,
batch sizes) held identical. `V=271` is unchanged (no vocab constant
edits) — the 12 reserved STATE value ids stay free for this exact
config-only ladder.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_tpu_sanity_w25_vocab4.py --device tpu
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
    rope=False,
    null_kv=True,
    rmsnorm=True,
    grad_checkpoint='block',
    name='hmn_tpu_sanity_w25_vocab4', seed=48,

    adaptive=True,
    adapt_signal='val_match',
    adapt_temp=1.0,
    adapt_ema_alpha=0.5,
    adapt_floor=0.05,

    state_len=4, state_vocab_size=4,  # ABLATED from 1 — see module docstring
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
