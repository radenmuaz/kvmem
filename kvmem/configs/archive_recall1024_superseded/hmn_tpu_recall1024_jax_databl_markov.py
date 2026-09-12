"""
`hmn_tpu_recall1024_jax_databl_markov.py` — see `_databl_random.py`'s own
docstring for the full rationale (structured-vs-random convergence-speed
ablation). This one: `data_kind='markov'` (order-1 Markov chain over the
full 256-byte alphabet, `target_bits=2.0` — well below random's 8 bits/
byte, so genuinely compressible), `data_target_bits=2.0`. Same reduced
20-entry weave_mix, same `n_steps=3000`, everything else identical to
the random baseline.

Recovery-probe contamination note (from CLAUDE.md's structured-data
section) does NOT apply here — that warning is specifically about
`gen_match_distance`'s exact-byte-repetition risk for chain-memory
recovery probes; this is a plain convergence-speed comparison on the
SAME single-query task both configs already use, not a recovery probe.

Run (never two jobs at once):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_tpu_recall1024_jax_databl_markov.py
"""

from kvmem.hmn_jax import load_config

hp = load_config('kvmem/configs/hmn_tpu_recall1024_jax_adaptive_mix.py')
hp['name'] = 'hmn_tpu_recall1024_jax_databl_markov'
hp['data_kind'] = 'markov'
hp['data_target_bits'] = 2.0

_full_mix = hp['curriculum'][0]['weave_mix']
_reduced_mix = [e for e in _full_mix if e['dsl'].split()[1] in ('E1', 'E3', 'E7', 'E11', 'E15')
                and e['dsl'].split()[2].endswith(',16)')]
assert len(_reduced_mix) == 20, f'expected 20, got {len(_reduced_mix)}'
hp['curriculum'][0]['weave_mix'] = _reduced_mix
hp['curriculum'][0]['n_steps'] = 3000
hp['curriculum'][0]['eval_every'] = 1000
