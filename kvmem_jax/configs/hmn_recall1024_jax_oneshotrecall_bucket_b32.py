"""
`hmn_recall1024_jax_oneshotrecall_bucket_b32.py` — sibling of `_bucket_
maxb4.py` (see that file's own docstring for the full rationale: gets past
the stage-5 32-shape compile wall via `bucket_lengths=True`). Run in
PARALLEL on a separate node — same fix (bucketing), different lever:

**This node's lever: keep the DEFAULT `max_shape_buckets` (8), reduce
batch size to 32 instead** (down from 64) — more buckets/less padding
waste than the sibling config, but a smaller per-step compile+activation
footprint from the smaller batch. Direct comparison: does fewer-buckets
or smaller-batch get past the compile wall more reliably / converge
better once past it.

Same warm-start (`hmn_recall1024_jax_oneshotrecall_b128.py`'s own
`stage4_best.pt`, val_mean=71.1%), same stages 5-7 (chunk_len=32/64 +
hop_drop_prob=0.0; stage 8 omitted, same bucket_lengths/hop_drop_prob>0
incompatibility as the sibling config).

Run (run from the repo root so relative log/config paths resolve):
    python3 -m kvmem_jax.hmn_jax --config kvmem_jax/configs/hmn_recall1024_jax_oneshotrecall_bucket_b32.py
"""

from kvmem_jax.hmn_jax import load_config

hp = load_config('kvmem_jax/configs/hmn_recall1024_jax_oneshotrecall_bucket_maxb4.py')
hp['name'] = 'hmn_recall1024_jax_oneshotrecall_bucket_b32'
hp['max_shape_buckets'] = 8  # default bucket count — the other lever, not reduced here
for stage in hp['curriculum']:
    stage['B'] = 32
