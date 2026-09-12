"""
`hmn_tpu_recall1024_jax_databl_random.py` — ONE of a 2-config ablation
(`_databl_random.py` / `_databl_markov.py`) testing whether STRUCTURED
training data converges faster than uniform-random bytes, at a short
(~3000-step) budget, same architecture/mix/hparams otherwise. This is the
RANDOM baseline (`data_kind='random'`, this project's default everywhere
else — Shannon's source coding theorem means genuine compression cannot
emerge from this data, so a fast improvement under `markov` specifically
would indicate the model is exploiting the order-1 conditional structure
`gen_markov` provides, something zlib itself is structurally blind to per
CLAUDE.md's own structured-data track notes).

Uses the same reduced 20-entry weave_mix and hparams as the winning
config from the just-completed random hparam search (lr_max/adapt_temp/
warmup_steps get filled in from that search's winner once known — this
file inherits from `hmn_tpu_recall1024_jax_adaptive_mix.py` as a
placeholder base and should be checked against the search winner before
being trusted as apples-to-apples).

Run (never two jobs at once):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_tpu_recall1024_jax_databl_random.py
"""

from kvmem.hmn_jax import load_config

hp = load_config('kvmem/configs/hmn_tpu_recall1024_jax_adaptive_mix.py')
hp['name'] = 'hmn_tpu_recall1024_jax_databl_random'
hp['data_kind'] = 'random'

_full_mix = hp['curriculum'][0]['weave_mix']
_reduced_mix = [e for e in _full_mix if e['dsl'].split()[1] in ('E1', 'E3', 'E7', 'E11', 'E15')
                and e['dsl'].split()[2].endswith(',16)')]
assert len(_reduced_mix) == 20, f'expected 20, got {len(_reduced_mix)}'
hp['curriculum'][0]['weave_mix'] = _reduced_mix
hp['curriculum'][0]['n_steps'] = 3000
hp['curriculum'][0]['eval_every'] = 1000
