"""
`hmn_weave_c64.py` — same mechanism as `hmn_weave_mix_accum_rnn_repeat8.py`
(`weave_mix=[batch, stream, interleave_delayed]`, `hops=1`, `repeat_batch=8`)
but at `chunk_len=64` instead of `chunk_len=16` — built specifically so
`hmn_stitch_src1024.py` (which also uses `chunk_len=64`) can warm-start
from a checkpoint trained at the SAME chunk size, instead of `repeat8`'s
chunk_len=16 checkpoint (a real architectural mismatch: STATE compresses a
64-byte chunk very differently than a 16-byte one, even though d/n_layers/
state_len are identical and the weights technically load without error).

`n_chunks=4, chunk_len=64` (256-byte total src per training example,
window_chunks=2 -> 128-byte query spans) — matches this project's usual
nc4 primitive, just at 4x the chunk size. Warm-started from
`hmn_single_recall_c64`'s checkpoint (same as `repeat8`'s own lineage — see
CLAUDE.md's `hmn_routing_4to1_state` "archived" note for why that's the
base, not `hmn_routing_4to1_state` itself).

`n_steps=80000` (doubled from an original 40000) and `repeat_batch=16`
(doubled from 8) — the first attempt at 40000/repeat_batch=8 was NOT
converging: loss barely moved (5.51->5.13, near the ~5.545 random floor)
and val MEAN crawled 0.6%->2.4% over 28000/40000 steps, the same
underfitting-plateau signature `repeat_batch` fixed for
`hmn_weave_mix_accum_rnn` at chunk_len=16.

**Second attempt (80000/repeat_batch=16) was WORSE, not just stuck**: loss
actually trended UP (5.31->5.56, past the ~5.545 random floor) and val
MEAN oscillated 0.1-0.7% with no upward trend at all through step 36000.
Doubling `repeat_batch` alone didn't help and may have hurt (16 gradient
steps on the same batch is a lot, especially combined with `lr_max` still
in its (now very long, 80000-step) warmup ramp for a while).

**Third attempt: tune the optimizer itself.** `warmup_steps` 500->2000
(4x longer — gives the LR ramp more room before `repeat_batch=16`'s
multiple-steps-per-batch dynamic kicks in at full strength) and `lr_max`
1.5e-4->3e-4 (2x — every other config in this project uses 1.5e-4, but
this chunk_len=64 task may need more signal per step to escape the
region near the random floor rather than creep out of it slowly).
`repeat_batch=16`/`n_steps=80000` left unchanged from attempt 2.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_weave_c64.py --device mps
"""

hp = dict(
    d=64, n_layers=8, n_heads=4, V=274,
    block_type='single_attn',
    lr_max=1e-4, lr_min=1e-6, wd=0.0001,
    warmup_steps=5000, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=80000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True,
    rmsnorm=True,
    name='hmn_weave_c64', seed=50,
    _pretrained_ckpt='logs/hmn_single_recall_c64/checkpoints/stage0_best.pt',
    repeat_batch=4,

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=2, chunk_len=64, window_chunks=1, B=4, n_steps=40000, eval_every=10000,
                     hops=1,
                     weave_mix=[
                         dict(weight=1.0, pattern='batch'),
                         dict(weight=1.0, pattern='stream'),
                         dict(weight=1.0, pattern='interleave_delayed'),
                     ]),

        dict(n_chunks=4, chunk_len=64, window_chunks=2, B=2, n_steps=80000, eval_every=10000,
             hops=1,
             weave_mix=[
                 dict(weight=1.0, pattern='batch'),
                 dict(weight=1.0, pattern='stream'),
                 dict(weight=1.0, pattern='interleave_delayed'),
             ]),
    ],
)
