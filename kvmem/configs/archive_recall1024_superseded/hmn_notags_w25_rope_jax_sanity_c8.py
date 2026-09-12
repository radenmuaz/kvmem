"""
`hmn_notags_w25_rope_jax_sanity_c8.py` — a genuine convergence sanity check,
replacing `hmn_notags_w25_rope_jax_databl_random.py`'s own attempt (which
used `chunk_len=32` at only 6000 steps — far too short AND not even the
easiest stage; torch's own `hmn_notags_w25_rope.py` needs up to 480000
steps at `chunk_len=8`, its FIRST/easiest stage, to hit `early_stop_
mean=80.0`). This config matches that exact stage 0 shape (`chunk_len=8,
warmup_lens=[2,3,4], n_anchors=4, min_recall_len=4, B=16`) — same
architecture (`d=64/n_layers=8/n_heads=4`, ~165K params), same DSL grid —
so a real convergence trend (or lack of one) at a MEANINGFUL fraction of
the original budget is actually informative, unlike the earlier 6000-step
attempt.

`n_steps=100000` (~21% of torch's own 480000-step budget for this exact
stage) — long enough to see genuine convergence trending clearly if the
pipeline is healthy, short enough to fit the current session's time
budget. Includes BOTH fixes landed today: gradient clipping
(`grad_clip_norm=1.0`, default) and warm-start-from-best (irrelevant here,
single stage, but left at its default True for consistency).

Run (never two jobs at once):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_notags_w25_rope_jax_sanity_c8.py
"""

from kvmem.hmn_jax import load_config

hp = load_config('kvmem/configs/hmn_tpu_recall1024_jax_adaptive_mix.py')  # hp-shape defaults only
hp['d'] = 64
hp['n_layers'] = 8
hp['n_heads'] = 4
hp['V'] = 271
hp['lr_max'] = 1e-4
hp['wd'] = 1e-5
hp['warmup_steps'] = 1000
hp['log_every'] = 500
hp['rope'] = True
hp['yarn'] = True
hp['null_kv'] = True
hp['rmsnorm'] = True
hp['grad_checkpoint'] = False
hp['no_autocast'] = True
hp['name'] = 'hmn_notags_w25_rope_jax_sanity_c8'
hp['adaptive'] = True
hp['adapt_signal'] = 'val_match'
hp['state_len'] = 8
hp['state_vocab_size'] = 2
hp['warmup_len'] = 2
hp['val_n_seqs'] = 3
hp['bucket_lengths'] = False
hp['data_kind'] = 'random'

_CHUNK_LEN = 8
_WARMUP_LENS = [2, 3, 4]
_N_ANCHORS = 4
_MIN_RECALL_LEN = 4

_mix = []
for wl in _WARMUP_LENS:
    max_start = _CHUNK_LEN - _MIN_RECALL_LEN - wl
    if max_start < 0:
        continue
    starts = sorted(set(round(i * max_start / (_N_ANCHORS - 1)) for i in range(_N_ANCHORS))) if max_start > 0 else [0]
    for s in starts:
        _mix.append(dict(weight=1.0, dsl=f'E({_CHUNK_LEN}) Q(0,1,{s},{wl})'))

hp['curriculum'] = [
    dict(n_chunks=1, chunk_len=_CHUNK_LEN, B=16, n_steps=100000, eval_every=5000,
         hops=-1, early_stop_mean=80.0, weave_mix=_mix),
]
