"""
`hmn_tpu_recall1024_jax_adaptive_mix.py` (renamed from `..._jax_curriculum.py`
— "curriculum" was misleading: there is no staged/sequential difficulty
progression here, just one stage whose entries span multiple difficulties
with sampling probability adaptively reweighted toward whichever entries
are currently hardest. "adaptive_mix" names what this actually is.) — same
architecture as `hmn_tpu_recall1024_jax_smallbatch.py` (`d=128/n_layers=16/
n_heads=8`, ~1.12M params, `B=8`, `lr_max=1e-4`), but the `weave_mix`
EXTENDED to include shorter sources (`n_chunks` in {2,4,8,12,16}, not just
the full 1024-byte/16-chunk target) mixed together in ONE stage.

Anchors are FRACTION-based per `n_chunks` (`[0, 0.25, 0.5, 0.75]` of
`total_len - warmup_len`, `warmup_len` in `{16, 32, 64}` whichever fit
under that `n_chunks`'s total length) — guarantees a valid (non-degenerate)
response region at every source length automatically, verified directly via
`_build_trajectory` against all 60 resulting entries before deploying
(12 entries per n_chunks x 5 n_chunks values = 60 total).

`bucket_lengths` is OFF — a real constraint, not a choice: `w0` scales with
`n_chunks` (`w0 = n_chunks * (chunk_len + 1 + state_len)`), so entries at
different `n_chunks` have DIFFERENT `w0` (verified: 5 distinct values —
{138, 276, 552, 828, 1104}), violating `_make_train_step_bucket`'s
single-shared-`w0`-per-stage requirement.

`adaptive=True` — without this, uniformly weighting 60 entries of wildly
different difficulty (128-byte recall vs 1024-byte suffix recall) means the
model never gets pushed toward whichever entries it's actually struggling
with; sampling stays static at the config weight forever. JAX port of
kvmem.hmn's own `_adapt_reweight` (identical formula, see `kvmem/hmn_jax.py`'s
`train_jax`) — `adapt_signal='val_match'` (default, matches kvmem.hmn's own
default): after the 2nd eval onward, sampling weight shifts toward entries
with the LOWEST match%, scaled by `adapt_floor + (1-adapt_floor)*softmax(
normalized_difficulty)`, so no entry (even one already at 100%) drops below
`adapt_floor`'s relative share — verified the rescaling formula directly
(uniform difficulty stays uniform, a harder entry gets upweighted, easier
ones scale down, sum preserved) before deploying. This is the mechanism
that makes the difficulty MIX actually useful — without it, mixing short
and long entries at static weight is not meaningfully different from
training on the full-length target alone plus noise.

Run (never two jobs at once):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_tpu_recall1024_jax_adaptive_mix.py
"""

from kvmem.hmn_jax import load_config

hp = load_config('kvmem/configs/hmn_tpu_recall1024_jax_smallbatch.py')
hp['name'] = 'hmn_tpu_recall1024_jax_adaptive_mix'
hp['bucket_lengths'] = False
hp['adaptive'] = True
hp['adapt_signal'] = 'val_match'


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


_ALL_MIX = []
for _nc in [2, 4, 8, 12, 16]:
    _ALL_MIX.extend(_weave_mix_for(_nc))

hp['curriculum'][0]['weave_mix'] = _ALL_MIX
hp['curriculum'][0]['n_steps'] = 20000
hp['curriculum'][0]['eval_every'] = 2000
