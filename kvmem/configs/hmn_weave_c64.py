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

**Fourth attempt: two-stage curriculum** (`n_chunks=2/window_chunks=1` then
`n_chunks=4/window_chunks=2`), gentler hp (`lr_max=1e-4, wd=0.0001,
warmup_steps=5000, repeat_batch=4`). Stage0 genuinely converged (val MEAN
14%->23.6% over 40000 steps) — real progress, unlike every single-stage
attempt before it. Stage1 (the harder task) still struggled though: loss
crept UP (4.74->5.05) across its first 30000/80000 steps even though val
MEAN kept slowly climbing (7.6%->8.9%, not flat/random like earlier
attempts) — better than before, but not yet converging cleanly.

**Fifth attempt (current): double both stages' step budgets again**
(`n_steps`: 40000->80000, 80000->160000) with **slower LR decay for
both**: `cosine_T0` is a single top-level value shared by every stage
(each stage's own LR schedule restarts from `local_step=1` using the same
`T0` — see `train()`'s per-branch `_lr(s)` closures, `kvmem/hmn.py`), so
setting it to `160000` (matching the LONGER stage exactly) gives stage1 a
full, properly-paced anneal across its whole length, and stage0 — which
only uses the first half of that same cosine cycle — a correspondingly
slower, more gradual decay too, without needing a separate per-stage T0
(which the framework doesn't currently support).

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_weave_c64.py --device mps
"""

hp = dict(
    d=64, n_layers=8, n_heads=4, V=274,
    block_type='single_attn',
    lr_max=1e-4, lr_min=1e-6, wd=0.0001,
    warmup_steps=5000, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=160000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True,
    rmsnorm=True,
    name='hmn_weave_c64', seed=50,
    _pretrained_ckpt='logs/hmn_single_recall_c64/checkpoints/stage0_best.pt',

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    # Each entry spelled out as an explicit DSL string (parse_traj_dsl's grammar
    # comment, kvmem/hmn.py) instead of pattern='batch'/'stream'/'interleave_delayed'
    # — verified byte-identical ops/n_refine against the named-pattern constructors
    # before switching (see traj_batch/traj_stream/traj_interleave_delayed).
    # `B4` on every entry preserves this run's tested repeat_batch=4 behavior now
    # that repeat_batch moved from a stage-wide hp flag to a per-trajectory DSL
    # token (default B1/no-repeat if omitted) — all three shapes here were judged
    # equally hard when this config was tuned, so all three keep the same B4;
    # nothing stops a future edit from giving a harder shape its own higher count.
    curriculum=[
        dict(n_chunks=2, chunk_len=64, B=4, n_steps=80000, eval_every=10000,
             hops=1,
             weave_mix=[
                 dict(weight=1.0, dsl='E2 Q(0,1) Q(1,2) B8'),        # batch(nc=2,wc=1)
                 dict(weight=1.0, dsl='E1 Q(0,1) E Q(1,2) B8'),      # stream(nc=2,wc=1)
                 dict(weight=1.0, dsl='E2 Q(1,2) Q(0,1) B8'),        # interleave_delayed(nc=2,wc=1)
             ]),

        dict(n_chunks=4, chunk_len=64, B=2, n_steps=160000, eval_every=10000,
             hops=1,
             weave_mix=[
                 dict(weight=1.0, dsl='E4 Q(0,2) Q(1,3) Q(2,4) B16'),       # batch(nc=4,wc=2)
                 dict(weight=1.0, dsl='E2 Q(0,2) E Q(1,3) E Q(2,4) B16'),   # stream(nc=4,wc=2)
                 dict(weight=1.0, dsl='E4 Q(2,4) Q(1,3) Q(0,2) B16'),       # interleave_delayed(nc=4,wc=2)
             ]),
    ],
)
