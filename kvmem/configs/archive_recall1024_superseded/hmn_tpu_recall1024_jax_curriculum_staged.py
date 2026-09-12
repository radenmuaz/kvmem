"""
`hmn_tpu_recall1024_jax_curriculum_staged.py` — a REAL sequential curriculum
(5 stages, `n_chunks` in {2,4,8,12,16} in order), unlike `..._adaptive_mix.py`
(one stage, static mixed-difficulty weave_mix + adaptive reweighting).

Motivation (2026-07-31): `hmn_tpu_recall1024_jax_adaptive_mix.py`'s 20000-step
run plateaued at MEAN=33.7-33.9% for the last 12000 steps, UNIFORMLY across
every entry regardless of `n_chunks`/`out_len` (33.3-35.4% band, no
difficulty gradient at all) — strong evidence of a degenerate solution
(1-of-3 `make_test_sequences` patterns solved, likely `const_mid` which is
trivially continuable with zero real content-addressed recall, since
1/3=33.3% matches almost exactly), not a genuine capacity ceiling. The
"mix everything, let adaptive reweighting sort it out" hypothesis is
effectively disproven by that uniformity — reweighting can shift SAMPLING
frequency, but can't force the model past a degenerate local optimum that's
equally exploitable at every difficulty level.

This config instead makes GENUINE progress the only way to advance: each
stage trains ONLY on its own `n_chunks` level (own weave_mix, own `w0`,
`early_stop_mean=90.0`) and must reach 90% val MEAN on ITS OWN entries
before the curriculum moves to the next (longer) stage — `train_jax`'s
existing `early_stop_mean` mechanism (already used for exactly this,
verified working via the `break` in the eval block) enforces this
directly, no new code needed. `adaptive=True` stays ON within each stage
(reweights sampling among that stage's own anchors — near-start anchors are
historically harder than near-end ones per this project's own recurring
pattern, so within-stage adaptation is still useful even though the
CROSS-STAGE mixing hypothesis didn't work).

`n_steps=30000` per stage — deliberately generous (a pure safety ceiling,
not the expected stopping point) so `early_stop_mean=90.0` is what actually
ends each stage in the typical case; `eval_every=1000` for prompt detection
once a stage crosses 90%. Model/optimizer weights carry over between stages
(`train_jax` builds a fresh `nnx.Optimizer` per stage but reuses the SAME
`model` object — matches the existing multi-stage warm-continuation
pattern). Worst case (no stage ever early-stops) is 5*30000=150000 steps;
expected case is far less if the hypothesis (genuine per-level mastery
generalizes better than uniform mixing) holds.

`bucket_lengths=False` kept consistent with the already-validated
`adaptive_mix` pipeline (not reintroduced here even though within-stage
entries now share one `w0`, to avoid combining two changes at once).

Run (never two jobs at once):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_tpu_recall1024_jax_curriculum_staged.py
"""

from kvmem.hmn_jax import load_config

hp = load_config('kvmem/configs/hmn_tpu_recall1024_jax_adaptive_mix.py')
hp['name'] = 'hmn_tpu_recall1024_jax_curriculum_staged'


def _weave_mix_for(n_chunks, chunk_len=64, warmup_lens=(16, 32, 64)):
    total = n_chunks * chunk_len
    fracs = [0.0, 0.25, 0.5, 0.75]
    mix = []
    for wl in warmup_lens:
        if wl >= total:
            continue
        for f in fracs:
            a = int(f * (total - wl))
            mix.append(dict(weight=1.0, dsl=f'E(64) E{n_chunks-1} Q(0,{n_chunks},{a},{wl})'))
    return mix


hp['curriculum'] = [
    dict(n_chunks=nc, chunk_len=64, B=8, n_steps=30000, eval_every=1000,
         hops=-1, early_stop_mean=90.0, weave_mix=_weave_mix_for(nc))
    for nc in [2, 4, 8, 12, 16]
]
