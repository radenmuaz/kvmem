"""
`hmn_oneshotrecall_jax_b512.py` — batch-size ablation of `hmn_oneshotrecall_
jax.py` (same experiment: state_len=8/state_vocab_size=2 STATE-capacity
test, MLP disabled). Batch size x32 of the original (16->512, 12->384,
6->192, 4->128) — the largest batch tried yet in this sweep (x1 -> x4 ->
x8 -> x16 -> x32), continuing the trend from `_b128.py` (clean win over x4)
per explicit direction.

Run in PARALLEL on a separate node from `hmn_oneshotrecall_jax_b256.py`
(never two jobs on the SAME device) — same experiment/question, different
batch size, direct comparison.

Run (run from the repo root so relative log/config paths resolve):
    python3 -m kvmem_jax.hmn_jax --config kvmem_jax/configs/hmn_oneshotrecall_jax_b512.py
"""

from kvmem_jax.hmn_jax import load_config

hp = load_config('kvmem_jax/configs/hmn_oneshotrecall_jax.py')
hp['name'] = 'hmn_oneshotrecall_jax_b512'
hp['curriculum'][0]['B'] = 512
hp['curriculum'][1]['B'] = 384
hp['curriculum'][2]['B'] = 192
hp['curriculum'][3]['B'] = 128
