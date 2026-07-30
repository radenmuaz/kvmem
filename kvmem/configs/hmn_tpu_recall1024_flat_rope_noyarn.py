"""
`hmn_tpu_recall1024_flat_rope_noyarn.py` — DIAGNOSTIC clone of
`hmn_tpu_recall1024_flat_rope.py` with `yarn=False` instead of `yarn=True`.

`hmn_tpu_recall1024_flat_rope.py` (yarn=True, `no_autocast=True`/fp32,
`L_train=2200`/`L_max=8192`) hit `loss=nan` across every entry in its own
gate5-style 30-step smoke test — even with fp32, unlike the sanity-scale
(`L~19-170`) RoPE test where fp32 alone fixed an earlier bf16-only NaN
cleanly. This is length-dependent (`L` here is 1232-2128) and not yet
isolated. `yarn=False` removes YaRN's interpolation-ramp scaling entirely
(`kvmem.hmn.rope_freqs`, not `yarn_freqs` — same frequency at every
length, no `L_train`/`L_max`-based ramp at all) — testing whether the ramp
formula itself is the culprit, one variable at a time, before assuming
it's something else (RoPE's raw position magnitude at `pos~2000`,
`grad_checkpoint` interacting with RoPE at this length, etc.).

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_tpu_recall1024_flat_rope_noyarn.py --device tpu
"""

_ANCHORS = [0, 128, 256, 384, 512, 640, 768, 896]
_WARMUP_LENS = [32, 64]

_WEAVE_MIX = [
    dict(weight=1.0, dsl=f'E(64) E15 Q(0,16,{a},{wl})')
    for a in _ANCHORS
    for wl in _WARMUP_LENS
]

hp = dict(
    d=128, n_layers=16, n_heads=8, V=271,
    block_type='single_attn',
    lr_max=6e-4, lr_min=1e-6, wd=0.0001,
    warmup_steps=2000, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=200000, cosine_T_mult=1,
    # DIAGNOSTIC: hmn_tpu_recall1024_flat_rope.py (yarn=True, otherwise identical)
    # hit loss=NaN across every entry at gate5's smoke test, EVEN with no_autocast=True
    # (fp32) — different from the sanity-scale (L~19-170) result where fp32 alone fixed
    # RoPE's NaN cleanly. This is length-dependent (L here is 1232-2128) and not yet
    # isolated. yarn=False removes YaRN's interpolation-ramp scaling entirely (plain
    # RoPE, same frequency at every length) — testing whether the ramp formula itself
    # is the culprit, one variable at a time.
    rope=True, yarn=False,
    L_train=2200, L_max=8192,  # unused when yarn=False (rope_freqs ignores both), left
                                # set for an easy diff against the yarn=True sibling config
    no_autocast=True,
    null_kv=True,
    rmsnorm=True,
    grad_checkpoint='block',
    name='hmn_tpu_recall1024_flat_rope_noyarn', seed=51,

    state_len=4, state_vocab_size=1,
    warmup_len=32,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,
    repeat_batch=1,

    bucket_lengths=True,
    max_shape_buckets=4,
    token_budget=131072,
    attn_sq_budget=31_000_000,

    curriculum=[
        dict(n_chunks=16, chunk_len=64, B=64, n_steps=200000, eval_every=10000,
             hops=-1,
             weave_mix=_WEAVE_MIX),
    ],
)
