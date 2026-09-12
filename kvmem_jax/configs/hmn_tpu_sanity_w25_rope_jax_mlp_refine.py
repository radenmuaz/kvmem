"""
`hmn_tpu_sanity_w25_rope_jax_mlp_refine.py` — **kvmem_mlp fork, first-ever
refine-round training run in this project** (2026-09-06). "No `n_refine>0`
experiment has ever been trained" was true up through this session's work —
the refine mechanism (`OP_FEEDBACK` opcode, `feedback_state`, exposure-bias
`am`/`wa` training) has existed as verified-but-unexercised code in
`kvmem/hmn.py` since the §17 redesign, and in `kvmem_mlp/hmn_jax.py` since
this session ported the two-pass argmax-feedback training step + refine-
aware decode/eval that JAX was missing (the position/mask/batch-construction
layer already supported refine rounds verbatim — only the training LOOP and
decode/eval needed new code, see this session's own investigation notes).

**Basic form, per explicit direction**: exactly `hmn_tpu_sanity_w25_rope_
jax_mlp.py`'s own sanity-style curriculum (`chunk_len` 8->16->32->64, single
chunk, `n_chunks=1`, same `_grid` anchor/warmup_len sweep) — but every entry
gets a trailing `R1` OR `R2` DSL token (never the plain non-refine form),
so the model trains on a MIX of one-refine-round and two-refine-round
trajectories at every stage, per explicit direction ("let 1 and 2 random
sample, so that generalize") rather than fixing n_refine at a single value.

**Warm-start ("standby checkpoint")**: `hmn_tpu_sanity_w25_rope_jax_mlp.py`'s
own `stage3_best.pt` (the already-converged, non-refine MLP baseline,
val_mean=80.96%) — same pattern as Experiment 2's own warm-start from
Experiment 1. No architecture change is needed for this to load cleanly:
`block_type='attn_mlp'`/`d_ff=256`/`V=271` are all unchanged from that
checkpoint's own config — the refine mechanism is purely a TRAJECTORY-SHAPE
change (which DSL/mask/batch-construction path gets exercised), not a model
change; `HMN_OP_FEEDBACK`'s vocab slot already exists in the shared `V=271`
opcode/value alphabet whether or not any prior run ever actually emitted it.

**Cost note**: every refine trajectory now costs TWO forward passes per
training step (pass 1: no-grad, to get the argmax to feed back; pass 2: the
real forward+backward) instead of one — expect roughly double the per-step
wall-clock of the non-refine `_mlp.py` sibling at the same `n_steps` budget.
Kept the same step budget as that config for now; revisit if this proves
too slow in practice.

Run (never two jobs at once; do not run before
`hmn_tpu_sanity_w25_rope_jax_mlp.py`'s stage3_best.pt actually exists — run
from the repo root so relative log/config paths resolve):
    python3 -m kvmem_mlp.hmn_jax --config kvmem_mlp/configs/hmn_tpu_sanity_w25_rope_jax_mlp_refine.py
"""


def _grid_refine(chunk_len, warmup_lens, n_anchors, min_recall_len, weight=1.0):
    """Same anchor/warmup_len sweep as `hmn_tpu_sanity_w25_rope_jax_mlp.py`'s
    own `_grid`, but emits BOTH an `R1` and an `R2` variant of every shape
    instead of the plain non-refine form — see this file's own docstring."""
    entries = []
    for wl in warmup_lens:
        max_start = chunk_len - min_recall_len - wl
        if max_start < 0:
            continue
        if n_anchors == 1 or max_start == 0:
            starts = [0]
        else:
            starts = sorted(set(round(i * max_start / (n_anchors - 1)) for i in range(n_anchors)))
        for s in starts:
            entries.append(dict(weight=weight, dsl=f'E({chunk_len}) Q(0,1,{s},{wl}) R1'))
            entries.append(dict(weight=weight, dsl=f'E({chunk_len}) Q(0,1,{s},{wl}) R2'))
    return entries


hp = dict(
    d=128, n_layers=16, n_heads=8, V=271,
    block_type='attn_mlp', d_ff=256,
    lr_max=1e-4, wd=1e-5,
    warmup_steps=1000, log_every=500,
    rope=True, yarn=True,
    null_kv=True,
    rmsnorm=True,
    name='hmn_tpu_sanity_w25_rope_jax_mlp_refine', seed=48,

    adaptive=True,
    adapt_signal='val_match',
    adapt_temp=1.0,
    adapt_ema_alpha=0.8,

    state_len=4, state_vocab_size=1,
    warmup_len=8,
    val_n_seqs=3,

    bucket_lengths=False,
    warm_start_from_best=True,
    pretrained_ckpt='kvmem_mlp/logs/hmn_tpu_sanity_w25_rope_jax_mlp/checkpoints/stage3_best.pt',

    curriculum=[
        dict(n_chunks=1, chunk_len=8, B=16, n_steps=100000, eval_every=5000,
             early_stop_mean=80.0,
             weave_mix=_grid_refine(8, [2, 3, 4], n_anchors=4, min_recall_len=4)),

        dict(n_chunks=1, chunk_len=16, B=12, n_steps=100000, eval_every=5000,
             early_stop_mean=80.0,
             weave_mix=(
                 _grid_refine(16, [4, 6, 8], n_anchors=4, min_recall_len=4)
                 + _grid_refine(8, [2, 3, 4], n_anchors=2, min_recall_len=4, weight=0.5)
             )),

        dict(n_chunks=1, chunk_len=32, B=6, n_steps=100000, eval_every=5000,
             early_stop_mean=80.0,
             weave_mix=(
                 _grid_refine(32, [8, 12, 16], n_anchors=4, min_recall_len=4)
                 + _grid_refine(16, [4, 6, 8], n_anchors=2, min_recall_len=4, weight=0.5)
                 + _grid_refine(8, [2, 3, 4], n_anchors=2, min_recall_len=4, weight=0.5)
             )),

        dict(n_chunks=1, chunk_len=64, B=4, n_steps=100000, eval_every=5000,
             early_stop_mean=80.0,
             weave_mix=(
                 _grid_refine(64, [16, 24], n_anchors=4, min_recall_len=4)
                 + _grid_refine(32, [8, 12, 16], n_anchors=2, min_recall_len=4, weight=0.5)
                 + _grid_refine(16, [4, 6, 8], n_anchors=2, min_recall_len=4, weight=0.5)
                 + _grid_refine(8, [2, 3, 4], n_anchors=2, min_recall_len=4, weight=0.5)
             )),
    ],
)
