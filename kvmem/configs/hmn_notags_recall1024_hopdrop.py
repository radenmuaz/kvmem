"""
`hmn_notags_recall1024_hopdrop.py` — torch port of `kvmem/configs/
hmn_tpu_recall1024_jax_hopdrop.py`, built on the `enc_hops`/`hop_drop_prob`
mechanism now ported back into `kvmem/hmn.py` (2026-07-31, from
`kvmem/hmn_jax.py`): the single-query suffix-recall design gives its one
query PERMANENT, UNBOUNDED attention to every encoded chunk's STATE by
default (`hops` is structurally inert for this design — `op_idx` is always
0). `enc_hops=N` generalizes the project's existing chain-step relay
concept (bounded N-back window, the immediately-preceding element never
dropped) to the ENCODING-CHUNK sequence itself. `hop_drop_prob` (per-stage,
annealed upward here) independently drops each back-distance 2..enc_hops
at every training step (never back=1) — LayerDrop-style regularization
along the chunk/time axis, meant to discourage the model leaning on a
dense all-chunks-at-once shortcut instead of genuinely propagated,
robust cross-chunk state. See `chunk_mask_fb_traj`'s own docstring in
`kvmem/hmn.py` for the full mechanism and its mask-matrix-level
verification.

Foundation stage skipped deliberately: `hmn_notags_w25_rope.py`'s own
stage 0 (chunk_len=8, n_chunks=1, same `_grid` dense-anchor-sweep style
used here) already converged locally — `logs/hmn_notags_w25_rope/
checkpoints/stage0_best.pt`, step=48000, val_mean=91.7%. Warm-starting
FROM that checkpoint (`hp['_pretrained_ckpt']`) instead of re-running it
from scratch. Same architecture required for the (shape-mismatch-
tolerant, but here an exact match anyway) loader: d=64/n_layers=8/
n_heads=4/V=271/state_len=8/state_vocab_size=2/rope=True — every value
below is copied from that config, not re-chosen.

`_grid_multi` mirrors `hmn_notags_w25_rope.py`'s own `_grid` (dense,
evenly-spaced anchor sweep per warmup_len via `round(i * max_start /
(n_anchors-1))`) generalized to n_chunks>1 suffix recall — this is the
exact "hmn_notags_w25_rope style variable size" mechanism the user asked
to be reused here, since a sparser anchor grid at n_chunks>1 has no more
reason to avoid the positional-shortcut collapse than a sparse grid did
at n_chunks==1 (see this session's CLAUDE.md "JAX curriculum val-MEAN
collapse root-caused" entry for the concrete anchor=0-vs-anchor=1 trade-
off this was built to avoid).

`hops=-1` (op-relay, irrelevant here — single query per entry, no chain
steps) stays at its harmless default; `enc_hops`/`hop_drop_prob` are the
mechanism actually in play. `eval_combinatorial_hops=True` reports val
MEAN for every subset of {2..enc_hops} (always unioned with {1}) at each
eval.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_notags_recall1024_hopdrop.py --device mps
"""


def _grid_multi(n_chunks, chunk_len, warmup_lens, n_anchors=6, min_recall_len=None):
    total = n_chunks * chunk_len
    min_recall_len = min_recall_len if min_recall_len is not None else max(4, chunk_len // 2)
    entries = []
    for wl in warmup_lens:
        max_start = total - min_recall_len - wl
        if max_start < 0:
            continue
        if n_anchors == 1 or max_start == 0:
            starts = [0]
        else:
            starts = sorted(set(round(i * max_start / (n_anchors - 1)) for i in range(n_anchors)))
        for s in starts:
            if n_chunks > 1:
                dsl = f'E({chunk_len}) E{n_chunks - 1} Q(0,{n_chunks},{s},{wl})'
            else:
                dsl = f'E({chunk_len}) Q(0,{n_chunks},{s},{wl})'
            entries.append(dict(weight=1.0, dsl=dsl))
    return entries


hp = dict(
    d=64, n_layers=8, n_heads=4, V=271,
    block_type='single_attn',
    lr_max=1e-4, wd=1e-5,
    warmup_steps=1000, log_every=500,
    rope=True,
    null_kv=True,
    rmsnorm=True,
    name='hmn_notags_recall1024_hopdrop', seed=48,

    adaptive=True,
    adapt_signal='val_match',

    state_len=8, state_vocab_size=2,
    warmup_len=2,
    val_n_seqs=3,

    enc_hops=4,
    eval_combinatorial_hops=True,
    warm_start_from_best=True,  # between THIS config's own stages
    _pretrained_ckpt='logs/hmn_notags_w25_rope/checkpoints/stage0_best.pt',

    curriculum=[
        dict(n_chunks=4, chunk_len=8, B=16, n_steps=40000, eval_every=5000,
             hops=-1, early_stop_mean=70.0, hop_drop_prob=0.1,
             weave_mix=_grid_multi(4, 8, [4, 8])),
        dict(n_chunks=8, chunk_len=8, B=16, n_steps=40000, eval_every=5000,
             hops=-1, early_stop_mean=70.0, hop_drop_prob=0.2,
             weave_mix=_grid_multi(8, 8, [8, 16])),
        dict(n_chunks=16, chunk_len=8, B=16, n_steps=40000, eval_every=5000,
             hops=-1, early_stop_mean=70.0, hop_drop_prob=0.3,
             weave_mix=_grid_multi(16, 8, [16, 32])),
        dict(n_chunks=16, chunk_len=32, B=8, n_steps=50000, eval_every=5000,
             hops=-1, early_stop_mean=60.0, hop_drop_prob=0.4,
             weave_mix=_grid_multi(16, 32, [32, 64])),
        dict(n_chunks=16, chunk_len=64, B=8, n_steps=60000, eval_every=5000,
             hops=-1, early_stop_mean=50.0, hop_drop_prob=0.5,
             weave_mix=_grid_multi(16, 64, [32, 64])),
    ],
)
