"""
`hmn_tpu_recall1024_flat_rope_noyarn_nockpt.py` — DIAGNOSTIC clone of
`hmn_tpu_recall1024_flat_rope_noyarn.py` (itself `yarn=False`, otherwise
`hmn_tpu_recall1024_flat_rope.py`) with `grad_checkpoint=False` instead of
`'block'`, and `attn_sq_budget` cut ~16x (`31_000_000` -> `2_000_000`) to
compensate — without checkpointing, all 16 layers' attention-score
matrices are alive simultaneously instead of one recomputed layer at a
time, so the per-bucket memory budget needs to shrink roughly by the same
factor checkpointing was saving.

Two separate questions this answers, not one: (1) is `grad_checkpoint`
still actually NEEDED at the batch sizes the post-OOM budget fix produces
(`attn_sq_budget=31_000_000` already cut `B` ~4x from the original
OOM'ing `B=64` — the original 52.85G-vs-15.75G OOM math was for `B=64`
specifically, never re-verified at the smaller `B` this budget now
produces); (2) does removing checkpointing change the RoPE NaN finding at
all (checkpointing recomputes each block's forward during backward, which
COULD interact with RoPE + fp32 in a way the sanity-scale test, which also
had checkpointing on, never isolated from RoPE itself).

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_tpu_recall1024_flat_rope_noyarn_nockpt.py --device tpu
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
    rope=True, yarn=False,
    L_train=2200, L_max=8192,  # unused when yarn=False (rope_freqs ignores both), left
                                # set for an easy diff against the yarn=True sibling config
    no_autocast=True,
    null_kv=True,
    rmsnorm=True,
    grad_checkpoint=False,  # ABLATED from 'block' — see module docstring
    name='hmn_tpu_recall1024_flat_rope_noyarn_nockpt', seed=51,

    state_len=4, state_vocab_size=1,
    warmup_len=32,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,
    repeat_batch=1,

    bucket_lengths=True,
    max_shape_buckets=4,
    token_budget=131072,
    attn_sq_budget=2_000_000,  # cut ~16x from 31_000_000 — see module docstring:
                                # compensates for no longer checkpointing (all 16
                                # layers' attention matrices alive at once, not 1)

    curriculum=[
        dict(n_chunks=16, chunk_len=64, B=64, n_steps=200000, eval_every=10000,
             hops=-1,
             weave_mix=_WEAVE_MIX),
    ],
)
