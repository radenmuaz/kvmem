"""
`hmn_tpu_recall1024_jax_hopdrop_randgrid.py` — same curriculum shape/staging
as `hmn_tpu_recall1024_jax_hopdrop.py` (chunk_len/n_chunks ramp, `enc_hops`/
`hop_drop_prob` mechanism, `early_stop_mean` gating), but replaces every
stage's deterministic evenly-spaced anchor/warmup-length grid with a
RANDOMLY SAMPLED (seeded, reproducible) set of (anchor, warmup_len) combos
of the same size.

**Rationale** (per direct user guidance 2026-08-01, see also
`feedback_curriculum_randomization` memory): what has converged fastest and
avoided degenerate/shortcut solutions in this project's past experiments is
gradual difficulty curriculum + RANDOMIZED src/warmup/anchor lengths, not a
fixed enumerated grid. A deterministic grid (even a dense, one-per-chunk-
index one, as `hmn_tpu_recall1024_jax_hopdrop.py`'s `_weave_mix_for`/
`_intra_chunk_grid` already use) is still a FINITE set of shapes the model
can partially key on — the exact "positional shortcut" mechanism
`probe_positional_shortcut.py` documented (anchor=0-vs-anchor=1 split).
Randomizing the SET ITSELF (not just widening it) removes that finite
surface without needing per-step re-sampling/recompilation: each stage
still gets its own fixed list of `weave_mix` DSL entries (required, since
each distinct (n_chunks, chunk_len, anchor, warmup_len) shape needs its own
XLA compile), but that list is now drawn from `np.random.default_rng(seed)`
over the FULL valid (anchor, warmup_len) range at 2-4x the entry COUNT of
the deterministic version, rather than evenly-spaced positions — a materially
different, less memorizable sample each stage while keeping compile cost in
the same ballpark.

Kept deliberately parallel to `hmn_tpu_recall1024_jax_hopdrop.py` (same
architecture/hparams/`enc_hops`/`hop_drop_prob` schedule) so results are a
clean A/B against that config's own (now-running) results — the only
variable changed is HOW the weave_mix entries are chosen, not how many
stages or what mechanism governs them.

Run (never two jobs at once):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_tpu_recall1024_jax_hopdrop_randgrid.py
"""

import numpy as np
from kvmem.hmn_jax import load_config

hp = load_config('kvmem/configs/hmn_tpu_recall1024_jax_hopdrop.py')
hp['name'] = 'hmn_tpu_recall1024_jax_hopdrop_randgrid'

_rng = np.random.default_rng(12345)


def _random_intra_chunk_mix(chunk_len, warmup_lens, n_samples, min_recall_len):
    """Random (not evenly-spaced) anchor positions within one chunk, per
    warmup_len — the randomized counterpart to `_intra_chunk_grid`."""
    mix = []
    for wl in warmup_lens:
        max_start = chunk_len - min_recall_len - wl
        if max_start < 0:
            continue
        if max_start == 0:
            starts = [0]
        else:
            k = min(n_samples, max_start + 1)
            starts = sorted(set(_rng.integers(0, max_start + 1, size=max(k, n_samples)).tolist()))[:n_samples]
        for s in starts:
            mix.append(dict(weight=1.0, dsl=f'E({chunk_len}) Q(0,1,{s},{wl})'))
    return mix


def _random_weave_mix_for(n_chunks, chunk_len, warmup_lens, n_samples_per_wl, min_recall_len=None):
    """Random (anchor chunk index, exact byte offset within it, warmup_len)
    combos spanning the full n_chunks source — the randomized counterpart to
    `_weave_mix_for`. `n_samples_per_wl` random anchors per warmup_len,
    typically set to 2-4x the deterministic per-chunk-index count so
    coverage stays comparable while the exact set sampled is not a fixed
    evenly-spaced grid."""
    total = n_chunks * chunk_len
    min_recall_len = min_recall_len if min_recall_len is not None else max(4, chunk_len // 2)
    mix = []
    for wl in warmup_lens:
        max_s = total - wl - min_recall_len
        if max_s < 0:
            continue
        k = min(n_samples_per_wl, max_s + 1)
        starts = sorted(set(_rng.integers(0, max_s + 1, size=max(k * 2, n_samples_per_wl)).tolist()))[:n_samples_per_wl]
        for s in starts:
            dsl = f'E({chunk_len}) E{n_chunks - 1} Q(0,{n_chunks},{s},{wl})'
            mix.append(dict(weight=1.0, dsl=dsl))
    return mix


hp['curriculum'] = [
    dict(n_chunks=1, chunk_len=8, B=16, n_steps=100000, eval_every=5000,
         hops=-1, early_stop_mean=70.0, hop_drop_prob=0.0,
         weave_mix=_random_intra_chunk_mix(8, [2, 3, 4], n_samples=6, min_recall_len=4)),
    dict(n_chunks=4, chunk_len=8, B=16, n_steps=40000, eval_every=5000,
         hops=-1, early_stop_mean=70.0, hop_drop_prob=0.3,
         weave_mix=_random_weave_mix_for(4, 8, [4, 8], n_samples_per_wl=8)),
    dict(n_chunks=8, chunk_len=8, B=16, n_steps=40000, eval_every=5000,
         hops=-1, early_stop_mean=70.0, hop_drop_prob=0.4,
         weave_mix=_random_weave_mix_for(8, 8, [8, 16], n_samples_per_wl=16)),
    dict(n_chunks=16, chunk_len=8, B=16, n_steps=40000, eval_every=5000,
         hops=-1, early_stop_mean=70.0, hop_drop_prob=0.5,
         weave_mix=_random_weave_mix_for(16, 8, [16, 32], n_samples_per_wl=32)),
    dict(n_chunks=16, chunk_len=32, B=8, n_steps=50000, eval_every=5000,
         hops=-1, early_stop_mean=60.0, hop_drop_prob=0.6,
         weave_mix=_random_weave_mix_for(16, 32, [32, 64], n_samples_per_wl=32)),
    dict(n_chunks=16, chunk_len=64, B=8, n_steps=60000, eval_every=5000,
         hops=-1, early_stop_mean=50.0, hop_drop_prob=0.7,
         weave_mix=_random_weave_mix_for(16, 64, [32, 64], n_samples_per_wl=32)),
]
