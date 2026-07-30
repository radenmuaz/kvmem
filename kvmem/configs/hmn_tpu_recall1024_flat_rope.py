"""
`hmn_tpu_recall1024_flat_rope.py` — clone of `hmn_tpu_recall1024_flat.py`
with `rope=True` instead of `rope=False`. Everything else identical
(architecture, curriculum, anchor sweep, `max_shape_buckets=4`/`attn_sq_
budget=31_000_000` memory fix from the earlier gate-5 OOM).

Motivation: `hmn_tpu_recall1024_flat.py`'s own docstring already flagged
RoPE as converging faster/higher in the measured head-to-head elsewhere,
but shipped with `rope=False` as "final." A live re-test at THIS scale
(`hmn_tpu_sanity_w25_rope.py`, same architecture, short-L sanity curriculum)
just reproduced that result dramatically: match=50.1% at step 5000 vs.
NoPE's 21.7% at the same step — more than double. That config change
uncovered a real, SEPARATE bug first (`rope=True` + bf16 autocast ->
`loss=nan` from step 1, CLAUDE.md's TPU port entry) — fixed here the same
way it was fixed there: `no_autocast=True` (forces fp32).

`L_train=2200`/`L_max=8192` set explicitly (just past the longest real
entry's `L=2128`, with headroom to `8192` for the off-grid/length-
extrapolation evals `hmn_tpu_recall1024_flat.py`'s own eval plan already
calls for) — the base config's docstring called these "irrelevant under
rope=False and left unset"; under `rope=True` they calibrate YaRN's
scaling directly against this run's actual training lengths rather than
train()'s own `rope=True, yarn=True` defaults (which would otherwise
silently apply the wrong bare defaults instead of a config tuned to this
run's real L range).

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_tpu_recall1024_flat_rope.py --device tpu
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
    rope=True, yarn=True,  # ABLATED from rope=False — see module docstring
    L_train=2200, L_max=8192,
    no_autocast=True,  # REQUIRED with rope=True — see module docstring's bf16+RoPE NaN note
    null_kv=True,
    rmsnorm=True,
    grad_checkpoint='block',
    name='hmn_tpu_recall1024_flat_rope', seed=51,

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
