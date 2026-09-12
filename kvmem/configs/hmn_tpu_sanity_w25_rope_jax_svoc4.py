"""
`hmn_tpu_sanity_w25_rope_jax_svoc4.py` — `state_vocab_size` ablation of the
STAGED curriculum `hmn_tpu_sanity_w25_rope_jax.py` (see that file's own
docstring — the reverted, proven-shape design: `chunk_len` 8->16->32->64,
each stage gated by `early_stop_mean=80.0`). `state_vocab_size=4` (up from
the baseline's `state_vocab_size=1`) — same open question as the earlier
`_nocurr_svoc4` ablation (does more per-slot STATE signal help), now asked
against the CORRECT staged design instead of the abandoned no-curriculum
one, so this result is actually informative about the current best-shaped
config rather than a variant of something already known not to converge
cleanly.

Run (never two jobs at once on the SAME device):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_tpu_sanity_w25_rope_jax_svoc4.py
"""

from kvmem.hmn_jax import load_config

hp = load_config('kvmem/configs/hmn_tpu_sanity_w25_rope_jax.py')
hp['name'] = 'hmn_tpu_sanity_w25_rope_jax_svoc4'
hp['state_vocab_size'] = 4
