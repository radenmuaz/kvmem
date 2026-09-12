"""
`hmn_tpu_sanity_w25_rope_jax_nocurr_b32.py` — batch-size ablation of
`hmn_tpu_sanity_w25_rope_jax_nocurr.py` (see that file's own docstring for
the full design rationale). Identical in every respect except `B=32`
(double the baseline's `B=16`) — run in PARALLEL on a separate TPU VM
(`v6e-4-2`, not the same node as the baseline on `v6e-4-1`) per explicit
direction to use available chips for hparam/ablation exploration rather
than sequentially. Tests whether a larger batch (still comfortably small
given this design's short `L` — single-chunk, `n_chunks=1`, nothing like
the suffix-recall family's multi-thousand-token sequences) measurably
changes convergence speed/stability on the "one step encode decode across
many lengths" task.

Run (never two jobs at once on the SAME device — this is a separate VM
from the baseline, so running concurrently with it is fine):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_tpu_sanity_w25_rope_jax_nocurr_b32.py
"""

from kvmem.hmn_jax import load_config

hp = load_config('kvmem/configs/hmn_tpu_sanity_w25_rope_jax_nocurr.py')
hp['name'] = 'hmn_tpu_sanity_w25_rope_jax_nocurr_b32'
hp['curriculum'][0]['B'] = 32
