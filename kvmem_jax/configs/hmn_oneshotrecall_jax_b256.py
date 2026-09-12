"""
`hmn_oneshotrecall_jax_b256.py` — batch-size ablation of `hmn_oneshotrecall_
jax.py` (same experiment: state_len=8/state_vocab_size=2 STATE-capacity
test, MLP disabled). Batch size x16 of the original (16->256, 12->192,
6->96, 4->64) — continuing the batch-size sweep after `_b128.py` showed a
clean win over the x4 baseline (finished stage 2 at 88.4% val_mean by local
step 30000, vs. the x4 run's 66.8% at step 25000 and still not early-
stopped) — pushing further per explicit direction.

Run in PARALLEL on a separate node from `hmn_oneshotrecall_jax_b512.py`
(never two jobs on the SAME device) — same experiment/question, different
batch size, direct comparison.

Run (run from the repo root so relative log/config paths resolve):
    python3 -m kvmem_jax.hmn_jax --config kvmem_jax/configs/hmn_oneshotrecall_jax_b256.py
"""

from kvmem_jax.hmn_jax import load_config

hp = load_config('kvmem_jax/configs/hmn_oneshotrecall_jax.py')
hp['name'] = 'hmn_oneshotrecall_jax_b256'
hp['curriculum'][0]['B'] = 256
hp['curriculum'][1]['B'] = 192
hp['curriculum'][2]['B'] = 96
hp['curriculum'][3]['B'] = 64
