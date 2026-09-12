"""
`hmn_recall1024_jax_oneshotrecall_b512.py` — batch-size ablation of
`hmn_recall1024_jax_oneshotrecall_b256.py` (same experiment: recall1024
redo warm-started from the state_len=8/state_vocab_size=2/MLP-disabled
`oneshotrecall_b256` checkpoint). Batch size doubled again (Phase A
256->512, Phase B/C 128->256) — per explicit direction to run a genuinely
different batch size on the second node instead of a redundant copy of
`_b256.py`.

Run on a SEPARATE node from `hmn_recall1024_jax_oneshotrecall_b256.py`
(never two jobs on the same device) — same warm-start checkpoint, same
curriculum shape, different batch size, direct comparison.

Run (run from the repo root so relative log/config paths resolve):
    python3 -m kvmem_jax.hmn_jax --config kvmem_jax/configs/hmn_recall1024_jax_oneshotrecall_b512.py
"""

from kvmem_jax.hmn_jax import load_config

hp = load_config('kvmem_jax/configs/hmn_recall1024_jax_oneshotrecall_b256.py')
hp['name'] = 'hmn_recall1024_jax_oneshotrecall_b512'
for stage in hp['curriculum'][:4]:
    stage['B'] = 512
for stage in hp['curriculum'][4:]:
    stage['B'] = 256
