"""
`hmn_tpu_recall1024_jax.py` (renamed from `hmn_tpu_recall1024_flat_rope_jax.py`
— "_flat_rope" dropped since this is now the base of a family that includes
non-flat/curriculum variants; RoPE itself is still on, just no longer named
after it specifically). Self-contained (2026-07-31) — previously chain-
loaded its `hp` from `kvmem/configs/hmn_tpu_recall1024_flat_rope.py` (the
torch config), which has since been deleted from this working tree; inlined
directly here instead of depending on a file that may not exist. Architecture
`d=128/n_layers=16/n_heads=8/V=271`, ~1.12M params, `rope=True/yarn=True`,
`no_autocast=True` (required — bf16+RoPE NaN'd, see CLAUDE.md's TPU port
entry), single-query 1024-byte suffix-recall (`n_chunks=16, chunk_len=64`,
16 anchor x warmup_len entries), run via `kvmem.hmn_jax` (not torch_xla —
this exact config reliably NaN'd there regardless of yarn/grad_checkpoint/
precision, resolved by switching to JAX, see CLAUDE.md's TPU port entry).

`token_budget`/`attn_sq_budget` recalibrated for what's actually verified on
JAX/tpu2 (not reused from torch's own gate-5 OOM investigation): a real run
of this exact architecture at `B=64, Lb=2128` measured only 29.51/31.25 GiB
HBM (2026-07-31) — `token_budget=200_000` (> 64*2128=136,192),
`attn_sq_budget=320_000_000` (> 64*2128**2=289,816,576), both set so no
bucket shrinks below the already-verified-safe `B=64`. See `kvmem/hmn_jax.py`'s
own bucketing-section docstring (above `_bucket_ceilings`) for the general
tuning formula.

Run (never two jobs at once):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_tpu_recall1024_jax.py
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
    rope=True, yarn=True,
    L_train=2200, L_max=8192,
    no_autocast=True,
    null_kv=True,
    rmsnorm=True,
    grad_checkpoint='block',
    name='hmn_tpu_recall1024_jax', seed=51,

    state_len=4, state_vocab_size=1,
    warmup_len=32,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,
    repeat_batch=1,

    bucket_lengths=True,
    max_shape_buckets=4,
    token_budget=200_000,
    attn_sq_budget=320_000_000,

    curriculum=[
        dict(n_chunks=16, chunk_len=64, B=64, n_steps=200000, eval_every=10000,
             hops=-1,
             weave_mix=_WEAVE_MIX),
    ],
)
