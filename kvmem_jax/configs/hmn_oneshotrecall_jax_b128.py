"""
`hmn_oneshotrecall_jax_b128.py` — batch-size ablation of `hmn_oneshotrecall_
jax.py` (same experiment: state_len=8/state_vocab_size=2 STATE-capacity
test, MLP disabled). Identical in every respect except batch size DOUBLED
again from that config's own x4 (16->64) to x8 of the original (16->128,
12->96, 6->48, 4->32) — per explicit direction to push batch size further
(tpu7 runs the x4 config, tpu8 runs this x8 one) since the TPU was still
underutilized even at x4.

Run in PARALLEL on a separate node from `hmn_oneshotrecall_jax.py` (never
two jobs on the SAME device) — this is deliberately the same experiment at
a different batch size, not a different question, so compare convergence
speed/quality directly against that config's own numbers.

Run (run from the repo root so relative log/config paths resolve):
    python3 -m kvmem_jax.hmn_jax --config kvmem_jax/configs/hmn_oneshotrecall_jax_b128.py
"""

from kvmem_jax.hmn_jax import load_config

hp = load_config('kvmem_jax/configs/hmn_oneshotrecall_jax.py')
hp['name'] = 'hmn_oneshotrecall_jax_b128'
hp['curriculum'][0]['B'] = 128
hp['curriculum'][1]['B'] = 96
hp['curriculum'][2]['B'] = 48
hp['curriculum'][3]['B'] = 32
