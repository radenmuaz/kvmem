"""
`hmn_tpu_sanity_w25_ablate.py` — DIAGNOSTIC, not a real config. Isolates why
`hmn_tpu_sanity_w25.py`'s stage 1 (chunk_len=16, the first stage whose
buckets contain REAL padding — e.g. Lb=21 holds entries with real L=19/20/21
— stage 0's buckets never padded anything, every entry in a bucket had
identical real L) hit loss=NaN from step 1 on tpu1, even AFTER the
torch_xla-checkpoint/autocast fix (CLAUDE.md's bug 4) resolved stage 0's
earlier NaN. Extensive local CPU repro (bf16 autocast, with/without
checkpointing, with/without the exact reentrant path, all WITH real
padding matching stage 1's Lb=21 bucket) never reproduced the NaN — it
appears genuinely XLA-execution-specific, not something CPU-emulated bf16
autocast can surface. This config swaps `rope=False, state_vocab_size=1`
(the scale-up target's settings) for `rope=True, state_vocab_size=2` (every
historically-proven-working config's settings) on EXACTLY stage 1's shape,
to test whether the NaN is specific to the NoPE+vocab=1 combination or a
generic padding/XLA issue that would reappear regardless.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_tpu_sanity_w25_ablate.py --device tpu
"""
import math


def _grid(chunk_len, warmup_lens, n_anchors, min_recall_len, rb_token, weight=1.0, min_warmup_frac=0.0):
    """Verbatim copy of hmn_notags_w25.py's own `_grid`."""
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
    lr_max=6e-4, wd=1e-5,
    warmup_steps=200, log_every=20,
    rope=True,   # ABLATED from False
    null_kv=True,
    rmsnorm=True,
    grad_checkpoint='block',
    name='hmn_tpu_sanity_w25_ablate', seed=48,

    adaptive=True,
    adapt_signal='val_match',
    adapt_temp=1.0,
    adapt_ema_alpha=0.5,
    adapt_floor=0.05,

    state_len=4, state_vocab_size=2,  # ABLATED from 1
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    bucket_lengths=True,
    max_shape_buckets=8,
    token_budget=4_000_000,
    attn_sq_budget=500_000_000,

    # Stage 1 ONLY (hmn_tpu_sanity_w25.py's own stage index 1) — the shape that hit
    # NaN. Trained from scratch (no warm-start) since this is a pure isolation test,
    # not a real curriculum.
    curriculum=[
        dict(n_chunks=1, chunk_len=16, B=64, n_steps=200, eval_every=200,
             weave_mix=(
                 _grid(16, [4, 6, 8], n_anchors=4, min_recall_len=4, rb_token='B8',
                      min_warmup_frac=0.25)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5,
                        min_warmup_frac=0.25)
             )),
    ],
)
