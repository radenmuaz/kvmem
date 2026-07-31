"""
`hmn_tpu_recall1024_jax_hopdrop.py` — recall1024 curriculum built on the new
`enc_hops`/`hop_drop_prob` mechanism (`kvmem/hmn_jax.py`, 2026-07-31):
the single-query suffix-recall design gives its one query PERMANENT,
UNBOUNDED attention to every encoded chunk's STATE by default (`hops` was
structurally inert for this design — `op_idx` is always 0, and `op_idx==0`
was unconditionally exempt from any bound). `enc_hops=N` generalizes the
project's existing chain-step relay concept (bounded N-back window, the
immediately-preceding element never dropped) to the ENCODING-CHUNK sequence
itself: chunk k's own STATE computation may attend to at most the previous
`enc_hops` chunks' STATE (not just its own raw bytes — "encoding isolation"
for RAW bytes is untouched), and the query is windowed the same way against
the last `enc_hops` chunks. `hop_drop_prob` (per-stage, annealed upward
here) then independently drops each back-distance 2..enc_hops at every
TRAINING step (never back=1) — LayerDrop-style regularization along the
chunk/time axis rather than the depth axis, intended to discourage the
model from leaning on a dense all-chunks-at-once shortcut (the exact
"positional shortcut" failure mode CLAUDE.md already documents — see the
2026-07-31 "JAX curriculum val-MEAN collapse root-caused" entry) and instead
force genuinely propagated, robust cross-chunk state.

Foundation stage skipped deliberately: `hmn_notags_w25_rope_jax_sanity_c8_
noadaptive.py` (chunk_len=8, n_chunks=1, matching torch's own proven-
convergent easiest stage) already ran to completion this session — final
best=35.4% val MEAN, `stage0_best.pt` on tpu2. Warm-starting FROM that
checkpoint (`hp['pretrained_ckpt']`, new this session — loads before this
config's own stage 0 begins) instead of re-running it from scratch. Same
architecture required for the shape-exact `nnx.update` load: d=64/
n_layers=8/n_heads=4/V=271/state_len=8/state_vocab_size=2 — every value
below is copied from that config, not re-chosen.

`hops=-1` (op-relay, irrelevant here — single query per entry, no chain
steps) stays at its harmless default; `enc_hops`/`hop_drop_prob` are the
mechanism actually in play. `eval_combinatorial_hops=True` reports val MEAN
for every subset of {2..enc_hops} (always unioned with {1}) at each eval —
the direct read of which relay distances the model actually depends on
vs. tolerates losing, per the user's explicit ask ("in eval need to
combinatorial try each hop size").

`_weave_mix_for` uses the `hmn_notags_w25_rope`/`sanity_c8` DENSE anchor-
sweep style (many roughly-evenly-spaced anchor positions per warmup_len,
via the same `round(i * max_start / (n_anchors-1))` formula those configs
use) instead of a few coarse fixed fractions — this is the mechanism
CLAUDE.md's `probe_positional_shortcut.py` work found necessary to stop the
model resolving recall via attention POSITION rather than warmup CONTENT.
Applied across the full multi-chunk suffix-recall span here (not just
within one chunk), since the exact same anchor=0-vs-anchor=1 positional
trade-off this session's `sanity_c8_noadaptive` run surfaced at n_chunks=1
has no reason to disappear at n_chunks>1 — if anything a sparser anchor
grid make it MORE likely to collapse onto a position-based shortcut, since
there are fewer distinct positions to generalize across.

Run (never two jobs at once):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_tpu_recall1024_jax_hopdrop.py
"""

from kvmem.hmn_jax import load_config

hp = load_config('kvmem/configs/hmn_tpu_recall1024_jax_adaptive_mix.py')  # hp-shape defaults only
hp['d'] = 64
hp['n_layers'] = 8
hp['n_heads'] = 4
hp['V'] = 271
hp['lr_max'] = 1e-4
hp['wd'] = 1e-5
hp['warmup_steps'] = 1000
hp['log_every'] = 500
hp['rope'] = True
hp['yarn'] = True
hp['null_kv'] = True
hp['rmsnorm'] = True
hp['grad_checkpoint'] = False
hp['no_autocast'] = True
hp['name'] = 'hmn_tpu_recall1024_jax_hopdrop'
hp['adaptive'] = True
hp['adapt_signal'] = 'val_match'
hp['state_len'] = 8
hp['state_vocab_size'] = 2
hp['warmup_len'] = 2
hp['val_n_seqs'] = 3
hp['bucket_lengths'] = False
hp['data_kind'] = 'random'
hp['enc_hops'] = 4
hp['eval_combinatorial_hops'] = True
hp['warm_start_from_best'] = True  # between THIS config's own stages
hp['pretrained_ckpt'] = 'logs/hmn_notags_w25_rope_jax_sanity_c8_noadaptive/checkpoints/stage0_best.pt'


def _weave_mix_for(n_chunks, chunk_len, warmup_lens, n_anchors=6, min_recall_len=None):
    """Dense anchor sweep across the full [0, n_chunks*chunk_len) span, one
    sweep per warmup_len — see this file's own docstring for why (avoiding
    the positional-shortcut collapse `probe_positional_shortcut.py` and
    this session's own `sanity_c8_noadaptive` run both found). `min_
    recall_len` defaults to half the chunk_len (floor 4), matching `sanity_
    c8`'s own convention scaled to this file's larger chunk_len values."""
    total = n_chunks * chunk_len
    min_recall_len = min_recall_len if min_recall_len is not None else max(4, chunk_len // 2)
    mix = []
    for wl in warmup_lens:
        max_start = total - min_recall_len - wl
        if max_start < 0:
            continue
        starts = (sorted(set(round(i * max_start / (n_anchors - 1)) for i in range(n_anchors)))
                  if max_start > 0 else [0])
        for s in starts:
            if n_chunks > 1:
                dsl = f'E({chunk_len}) E{n_chunks - 1} Q(0,{n_chunks},{s},{wl})'
            else:
                dsl = f'E({chunk_len}) Q(0,{n_chunks},{s},{wl})'
            mix.append(dict(weight=1.0, dsl=dsl))
    return mix


hp['curriculum'] = [
    dict(n_chunks=4, chunk_len=8, B=16, n_steps=40000, eval_every=5000,
         hops=-1, early_stop_mean=70.0, hop_drop_prob=0.1,
         weave_mix=_weave_mix_for(4, 8, [4, 8])),
    dict(n_chunks=8, chunk_len=8, B=16, n_steps=40000, eval_every=5000,
         hops=-1, early_stop_mean=70.0, hop_drop_prob=0.2,
         weave_mix=_weave_mix_for(8, 8, [8, 16])),
    dict(n_chunks=16, chunk_len=8, B=16, n_steps=40000, eval_every=5000,
         hops=-1, early_stop_mean=70.0, hop_drop_prob=0.3,
         weave_mix=_weave_mix_for(16, 8, [16, 32])),
    dict(n_chunks=16, chunk_len=32, B=8, n_steps=50000, eval_every=5000,
         hops=-1, early_stop_mean=60.0, hop_drop_prob=0.4,
         weave_mix=_weave_mix_for(16, 32, [32, 64])),
    dict(n_chunks=16, chunk_len=64, B=8, n_steps=60000, eval_every=5000,
         hops=-1, early_stop_mean=50.0, hop_drop_prob=0.5,
         weave_mix=_weave_mix_for(16, 64, [32, 64])),
]
