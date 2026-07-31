"""
`hmn_notags_w25_rope_jax_sanity_c8_noadaptive.py` — A/B control for
`hmn_notags_w25_rope_jax_sanity_c8.py`. Identical shape/architecture/steps,
only `adaptive=False` (uniform sampling weights, no val_match-driven
reweighting feedback loop).

Motivation: the `adaptive=True` sanity run (chunk_len=8, n_chunks=1, the
easiest proven-convergent stage in torch's own `hmn_notags_w25.py` ladder)
showed a genuine, non-noisy collapse — val MEAN peaked 43.6% at step 15000,
then declined near-monotonically to 10.6-13.3% by step 35000-45000 (9 evals:
37.5/31.4/43.6/30.8/23.1/16.2/10.6/13.3/11.9%). This happened AFTER the
session's two curriculum-collapse fixes (gradient clipping via
`optax.clip_by_global_norm`, checkpoint warm-start-from-best between stages)
were already deployed — this run is single-stage, so warm-start doesn't
apply, and grad clipping alone evidently did not prevent the instability
(the crash is less severe than the pre-fix pattern documented in CLAUDE.md,
which went all the way to ~1%, but it's still a real ~30pp collapse over
20000 steps at a perfectly flat lr=1e-4, no cosine restarts involved).

`lr_schedule` here is warmup-then-CONSTANT (`kvmem.hmn_jax._make_schedule`),
ruling out an LR-spike artifact. The next live hypothesis is the adaptive
reweighting mechanism itself creating a destabilizing feedback loop
(sampling weight shifts toward under-performing entries based on val_match,
which could concentrate gradient updates on exactly the entries whose
current representations are least stable). This config isolates that single
variable.

Run (never two jobs at once):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_notags_w25_rope_jax_sanity_c8_noadaptive.py
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
hp['name'] = 'hmn_notags_w25_rope_jax_sanity_c8_noadaptive'
hp['adaptive'] = False
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
