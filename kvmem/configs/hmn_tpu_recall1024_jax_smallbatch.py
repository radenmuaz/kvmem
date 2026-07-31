"""
`hmn_tpu_recall1024_jax_smallbatch.py` (renamed from `..._flat_rope_jax_
smallbatch.py`) — same architecture as `hmn_tpu_recall1024_jax.py`
(`d=128/n_layers=16/n_heads=8`, ~1.12M params), but `B=8` instead of `64`,
and `lr_max` reverted to `1e-4` instead of `6e-4`.

Motivation (2026-07-31): the `B=64` bucketed run's loss was still at
essentially the random-guess baseline (`ln(256)≈5.545`) after 1000 steps,
projected ~46-60 hours to complete 200000 steps. Two suspected compounding
causes: (1) `lr_max=6e-4` was √-scaled in the original plan for a B≈256
target, too high for the actual `B=64` (this exact failure mode already
happened once, `hmn_tpu_sanity_w25.py`'s own bug); (2) large `B` means far
fewer optimizer updates per wall-clock minute. `1e-4` is this project's own
repeatedly-verified-converging baseline at small `B`.

Run (never two jobs at once):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_tpu_recall1024_jax_smallbatch.py
"""

from kvmem.hmn_jax import load_config

hp = load_config('kvmem/configs/hmn_tpu_recall1024_jax.py')
hp['name'] = 'hmn_tpu_recall1024_jax_smallbatch'
hp['lr_max'] = 1e-4
hp['curriculum'][0]['B'] = 8
hp['curriculum'][0]['eval_every'] = 2000
