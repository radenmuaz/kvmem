"""
`hmn_tpu_sanity_w25_rope_jax_nocurr_svoc4.py` — `state_vocab_size` ablation
of `hmn_tpu_sanity_w25_rope_jax_nocurr.py` (see that file's own docstring
for the full design). Identical in every respect except `state_vocab_size
=4` (up from the baseline's `state_vocab_size=1`, matching the torch
`hmn_tpu_sanity_w25_rope.py` reference) — no vocab-constant edit needed:
`HMN_STATE_0=259` plus up to 12 reserved tail IDs already accommodates
`state_vocab_size` up to 12 without touching `V=271` (see CLAUDE.md's
vocab-layout docstring in `kvmem/hmn.py`).

Motivation: CLAUDE.md's own open question from the original recall1024
scale-up plan — "if [training] stalls with mechanistic evidence of smeared
within-block attention, re-run at state_vocab_size=1 (does NoPE's/RoPE's
positional signal alone suffice to address STATE slots?) or =4 (does more
per-slot signal help?)". Both `hmn_tpu_sanity_w25_rope_jax_nocurr.py`
variants (`B=16`/`B=32`, `state_vocab_size=1`) plateaued around 12-14% val
MEAN with low/declining teacher-forced loss (the exposure-bias signature
already on record this session) — this tests whether giving each STATE
slot within a block more distinguishable per-slot signal (cycling through
4 distinct placeholder IDs instead of 1) changes that plateau, run in
place of the now-redundant `B=32` ablation (which converged to essentially
the same plateau as `B=16`, answering its own question already).

Run (never two jobs at once on the SAME device):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_tpu_sanity_w25_rope_jax_nocurr_svoc4.py
"""

from kvmem.hmn_jax import load_config

hp = load_config('kvmem/configs/hmn_tpu_sanity_w25_rope_jax_nocurr.py')
hp['name'] = 'hmn_tpu_sanity_w25_rope_jax_nocurr_svoc4'
hp['state_vocab_size'] = 4
