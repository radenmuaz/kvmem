"""
`hmn_notags_w25_rope_jax_databl_random.py` — ONE of a 2-config ablation
(`_databl_random.py` / `_databl_markov.py`) testing whether STRUCTURED
training data converges faster than uniform-random bytes, at TINY scale
(matches `hmn_notags_w25_rope.py`'s own architecture: `d=64/n_layers=8/
n_heads=4`, ~165K params) — single stage, single `chunk_len`, NO curriculum,
unlike the 1.12M-param Run-A-scale ablation this supersedes
(`hmn_tpu_recall1024_jax_databl_*.py`, never run). Small/fast enough to
budget ~30 min total for BOTH configs combined on tpu2.

`chunk_len=32`, `warmup_lens=[8,12,16]` (25%-of-chunk_len floor, matching
`hmn_notags_w25_rope.py`'s own `min_warmup_frac=0.25` convention),
`n_anchors=4` evenly-spaced query starts per warmup_len -> 12 entries,
single-Q non-refine DSL only (`E(32) Q(0,1,{s},{wl})`, no `B<n>` repeat-
batch token — `train_jax` doesn't wire `repeat_batch` at all, unlike
torch's `train()`, so this deliberately doesn't use it, keeping the DSL
within what's actually tested/supported here).

This: `data_kind='random'` (the baseline — Shannon's source coding theorem
means genuine compression can't emerge from this data, so any speed
DIFFERENCE against the markov sibling isolates whether the model can
exploit order-1 structure specifically, not some generic quirk of the
data pipeline).

Run (never two jobs at once):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_notags_w25_rope_jax_databl_random.py
"""

from kvmem.hmn_jax import load_config

hp = load_config('kvmem/configs/hmn_tpu_recall1024_jax_adaptive_mix.py')  # reused only for its hp-shape defaults
hp['d'] = 64
hp['n_layers'] = 8
hp['n_heads'] = 4
hp['V'] = 271
hp['lr_max'] = 1e-4
hp['wd'] = 1e-5
hp['warmup_steps'] = 500
hp['log_every'] = 200
hp['rope'] = True
hp['yarn'] = True
hp['null_kv'] = True
hp['rmsnorm'] = True
hp['grad_checkpoint'] = False
hp['no_autocast'] = True
hp['name'] = 'hmn_notags_w25_rope_jax_databl_random'
hp['adaptive'] = True
hp['adapt_signal'] = 'val_match'
hp['state_len'] = 8
hp['state_vocab_size'] = 2
hp['warmup_len'] = 8
hp['val_n_seqs'] = 3
hp['bucket_lengths'] = False
hp['data_kind'] = 'random'

_CHUNK_LEN = 32
_WARMUP_LENS = [8, 12, 16]
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
    dict(n_chunks=1, chunk_len=_CHUNK_LEN, B=16, n_steps=6000, eval_every=1000,
         hops=-1, weave_mix=_mix),
]
