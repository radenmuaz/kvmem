"""
`hmn_tpu_recall1024_jax_databl_random_tiny.py` — ONE of a 2-config ablation
(`_databl_random_tiny.py` / `_databl_markov_tiny.py`) testing whether
STRUCTURED training data converges faster than uniform-random bytes.
Uses the FULL `hmn_tpu_recall1024_jax.py` MODEL architecture (`d=128/
n_layers=16/n_heads=8`, ~1.12M params, `rope=True/yarn=True`) — same model
hyperparameters as the real Run-A-scale configs, so any speed difference
found here is actually informative about the model we care about — but a
TINY, single-stage, no-curriculum TASK (mirrors `hmn_notags_w25_rope.py`'s
own single-stage simplicity, not its small architecture): `n_chunks=2`
(the smallest/fastest recall1024 task shape, `chunk_len=64`), 12 entries
(4 anchors x 3 warmup_lens, same `_weave_mix_for` pattern the adaptive-mix
config uses), `n_steps=6000` — short enough that both configs together
should fit roughly a 30-minute budget on tpu2 including compile.

This: `data_kind='random'` (the baseline).

Run (never two jobs at once):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_tpu_recall1024_jax_databl_random_tiny.py
"""

from kvmem.hmn_jax import load_config

hp = load_config('kvmem/configs/hmn_tpu_recall1024_jax.py')
hp['name'] = 'hmn_tpu_recall1024_jax_databl_random_tiny'
hp['data_kind'] = 'random'


def _weave_mix_for(n_chunks, chunk_len=64, warmup_lens=(16, 32, 64)):
    total = n_chunks * chunk_len
    fracs = [0.0, 0.25, 0.5, 0.75]
    mix = []
    for wl in warmup_lens:
        if wl >= total:
            continue
        for f in fracs:
            a = int(f * (total - wl))
            mix.append(dict(weight=1.0, dsl=f'E(64) E{n_chunks-1} Q(0,{n_chunks},{a},{wl})'))
    return mix


hp['curriculum'] = [
    dict(n_chunks=2, chunk_len=64, B=8, n_steps=6000, eval_every=1000,
         hops=-1, weave_mix=_weave_mix_for(2)),
]
