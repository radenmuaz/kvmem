"""
SUPERSEDED (archived 2026-07-31): predates the `B=64->B=8`/`lr_max=6e-4->1e-4`
fix documented in CLAUDE.md (the original large-batch LR was miscalibrated
for the real small batch actually used, stalling training at the random
baseline for 1000+ steps) - these 4 trials' results are not trustworthy as
hyperparameter guidance. Superseded by the staged-curriculum approach
(`hmn_tpu_recall1024_jax_hopdrop.py`) using the already-corrected `lr_max=
1e-4`. Re-run a search under the corrected base hparams if one is needed.

`hmn_tpu_recall1024_jax_search2.py` — ONE of a 4-trial random hyperparameter
search (`_search0`..`_search3`), sampled `lr_max ~ loguniform(3e-5,3e-4)`,
`adapt_temp ~ uniform(0.5,2.0)`, `warmup_steps in {500,1000,2000}` (numpy
seed=42, see the search launch command for the exact draw). Uses a REDUCED
weave_mix (one entry per n_chunks region — 20 total instead of the full
60) to keep compile overhead from dominating these short (half-budget)
search trials; the winning config gets re-verified against the full
60-entry mix afterward. `n_steps=2500` per trial (4 trials x 2500 =
10000 steps total = half of the just-completed 20000-step
`hmn_tpu_recall1024_jax_adaptive_mix.py` run's budget).

This trial: lr_max=2.8400e-04  adapt_temp=1.6400  warmup_steps=2000

Run (never two jobs at once):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_tpu_recall1024_jax_search2.py
"""

from kvmem.hmn_jax import load_config

hp = load_config('kvmem/configs/hmn_tpu_recall1024_jax_adaptive_mix.py')
hp['name'] = 'hmn_tpu_recall1024_jax_search2'
hp['lr_max'] = 0.000284
hp['adapt_temp'] = 1.64
hp['warmup_steps'] = 2000

_full_mix = hp['curriculum'][0]['weave_mix']
# one entry per n_chunks region (E1/E3/E7/E11/E15 prefixes), 4 anchors each -> 20 total
_reduced_mix = [e for e in _full_mix if e['dsl'].split()[1] in ('E1', 'E3', 'E7', 'E11', 'E15')
                and e['dsl'].split()[2].endswith(',16)')]
assert len(_reduced_mix) == 20, f'expected 20, got {len(_reduced_mix)}'
hp['curriculum'][0]['weave_mix'] = _reduced_mix
hp['curriculum'][0]['n_steps'] = 2500
hp['curriculum'][0]['eval_every'] = 500
