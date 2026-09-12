"""
`hmn_stage0_structured_ablation.py` — structured-data ablation, queued earlier
this session ("choose two from the list, probabilistic but information
theoretic and one hidden rule with obvious extrapolation if learnt") and
never launched (deferred behind `nocurr`'s sanity check, which only reached
best=17.0% — not a clean pass — before session priority shifted to the
critical KV-cache decode bug). Selected generators from
`kvmem/structured_data.py`'s 9 implemented families:

  - `'markov'` — order-1 Markov chain, EXACT closed-form entropy-rate
    calibration (probabilistic/information-theoretic: the "hidden rule" is a
    conditional distribution, not a deterministic function, so there's no
    single fixed pattern to extrapolate, only a statistical regularity to
    exploit).
  - `'ca'` — 1D cellular automaton, random rule table + initial condition
    (a hidden DETERMINISTIC rule: if learnt, extrapolates obviously/exactly
    to any position, unlike the Markov case).

Same stage0 shape as `hmn_tpu_recall1024_jax_hopdrop.py`'s own foundation
stage (chunk_len=8, n_chunks=1, `_intra_chunk_grid` dense anchor sweep) and
the SAME fixed optimizer hparams as that config (B=16, lr_max=1e-4) —
deliberately not swept here (that's what the separate random-search task,
`hmn_stage0_search.py`, is for). This isolates data_kind as the only
variable against the already-characterized random-byte stage0 baseline
(best=... see CLAUDE.md's `hopdrop` stage0 entries once available).

`data_target_bits=2.0` — moderate compressibility (not near-incompressible,
not trivially-collapsed) so structure is genuinely present without being so
strong the task becomes near-trivial pattern matching.

Run (never two jobs at once):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_stage0_structured_ablation_markov.py
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_stage0_structured_ablation_ca.py
"""
from kvmem.hmn_jax import load_config


def _intra_chunk_grid(chunk_len, warmup_lens, n_anchors, min_recall_len):
    mix = []
    for wl in warmup_lens:
        max_start = chunk_len - min_recall_len - wl
        if max_start < 0:
            continue
        starts = (sorted(set(round(i * max_start / (n_anchors - 1)) for i in range(n_anchors)))
                  if max_start > 0 else [0])
        for s in starts:
            mix.append(dict(weight=1.0, dsl=f'E({chunk_len}) Q(0,1,{s},{wl})'))
    return mix


def base_hp(data_kind: str):
    hp = load_config('kvmem/configs/hmn_tpu_recall1024_jax_hopdrop.py')
    hp['data_kind'] = data_kind
    hp['data_target_bits'] = 2.0
    hp['name'] = f'hmn_stage0_structured_{data_kind}'
    hp['curriculum'] = [
        dict(n_chunks=1, chunk_len=8, B=16, n_steps=100000, eval_every=5000,
             hops=-1, early_stop_mean=70.0, hop_drop_prob=0.0,
             weave_mix=_intra_chunk_grid(8, [2, 3, 4], n_anchors=4, min_recall_len=4)),
    ]
    return hp
