"""
`hmn_tpu_recall1024_jax_databl_markov_tiny.py` — see `_databl_random_tiny.py`'s
own docstring for the full rationale. This one: `data_kind='markov'`,
`data_target_bits=2.0` (well below random's 8 bits/byte, genuinely
compressible order-1 structure). Same model architecture (full recall1024
`d=128/n_layers=16/n_heads=8`), same tiny single-stage `n_chunks=2` task,
same `B=8`, `n_steps=6000` as the random baseline — only `data_kind`/
`data_target_bits` differ.

As with the earlier (Run-A-scale, never run) version of this ablation:
`make_test_sequences`' eval patterns are fixed/deterministic regardless of
`data_kind`, so TRAINING LOSS is the valid comparison signal here (directly
measures fit to each config's own data distribution) — match% would be
comparing against out-of-distribution eval patterns for the markov-trained
model, not a clean apples-to-apples read.

Run (never two jobs at once):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_tpu_recall1024_jax_databl_markov_tiny.py
"""

from kvmem.hmn_jax import load_config

hp = load_config('kvmem/configs/hmn_tpu_recall1024_jax.py')
hp['name'] = 'hmn_tpu_recall1024_jax_databl_markov_tiny'
hp['data_kind'] = 'markov'
hp['data_target_bits'] = 2.0


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
    dict(n_chunks=2, chunk_len=64, B=8, n_steps=6000, eval_every=1000,
         hops=-1, weave_mix=_weave_mix_for(2)),
]
