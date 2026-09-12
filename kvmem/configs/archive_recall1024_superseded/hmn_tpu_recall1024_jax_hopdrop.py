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

**UPDATE 2026-07-31 (later same day)**: the original plan here was to warm-
start from `hmn_notags_w25_rope_jax_sanity_c8_noadaptive.py`'s own converged
checkpoint (best=35.4% val MEAN, `stage0_best.pt` on `tpu2`) via `hp
['pretrained_ckpt']`. `tpu2` was subsequently deleted/reclaimed (confirmed
via `gcloud ... describe` returning `NOT_FOUND`) before that checkpoint was
ever copied off it — it's gone, not recoverable. This config now trains the
chunk_len=8/n_chunks=1 foundation stage ITSELF, from scratch, as its own
stage 0 (matching `sanity_c8`'s exact shape/dense-anchor-sweep), rather than
depending on external state — `hop_drop_prob` is 0 for this stage (n_chunks
=1 means there is no chunk-to-chunk relationship for `enc_hops` to window in
the first place). `pretrained_ckpt` removed. Deployed to a fresh `v6e-8`
TPU (`tpu-v6e-8-1`, `us-east1-d`) — note this file has NO multi-device
sharding/pmap, so it uses only 1 of the pod's 8 chips as-is; flagged, not
yet fixed (correctness prioritized first for this 12-hour run given several
pieces of this file's own machinery — `enc_hops`/`hop_drop_prob` — have
never been run at real TPU scale before).

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


def _weave_mix_for(n_chunks, chunk_len, warmup_lens, min_recall_len=None):
    """ONE anchor per chunk index (the start of every chunk 0..n_chunks-1),
    per warmup_len — UPGRADED 2026-07-31 from a sparse `n_anchors`-fraction
    sweep (which left most individual chunk positions never sampled as an
    anchor at all, out of n_chunks possible ones) to guarantee every chunk
    index appears as an anchor. The real reason variable/dense anchor
    coverage matters at all: preventing POSITION bias — a model trained on
    a sparse, clustered anchor set can learn to resolve recall via
    attention POSITION (which positions tend to be anchors) rather than
    WARMUP CONTENT, exactly the failure `probe_positional_shortcut.py`
    documented and this session's own `sanity_c8_noadaptive` run reproduced
    (anchor=0 stuck at 0% while anchor=1 hit 90%+, a pure position
    artifact). Dense per-chunk coverage removes the shortcut by construction
    — no chunk index is ever systematically under-represented. `min_
    recall_len` defaults to half the chunk_len (floor 4)."""
    assert n_chunks > 1, "_weave_mix_for is for the multi-chunk (inter-chunk anchor) case " \
        "only — use _intra_chunk_grid for n_chunks==1 (this function's per-chunk-index " \
        "anchor placement degenerates to a single fixed anchor at 0 when n_chunks==1, " \
        "losing the dense WITHIN-chunk sweep the foundation stage actually needs)"
    total = n_chunks * chunk_len
    min_recall_len = min_recall_len if min_recall_len is not None else max(4, chunk_len // 2)
    mix = []
    for wl in warmup_lens:
        for a in range(n_chunks):
            s = a * chunk_len
            if total - s - wl < min_recall_len:
                continue
            dsl = f'E({chunk_len}) E{n_chunks - 1} Q(0,{n_chunks},{s},{wl})'
            mix.append(dict(weight=1.0, dsl=dsl))
    return mix


def _intra_chunk_grid(chunk_len, warmup_lens, n_anchors, min_recall_len):
    """n_anchors evenly-spaced anchor positions WITHIN one chunk, per
    warmup_len — the single-chunk (n_chunks==1) foundation stage's own
    sweep (matches `hmn_notags_w25_rope_jax.py`'s/`sanity_c8`'s `_grid`),
    kept distinct from `_weave_mix_for` above (inter-chunk anchor placement
    has no meaning for a single chunk)."""
    mix = []
    for wl in warmup_lens:
        max_start = chunk_len - min_recall_len - wl
        if max_start < 0:
            continue
        starts = (sorted(set(round(i * max_start / (n_anchors - 1)) for i in range(n_anchors)))
                  if max_start > 0 else [0])
        for s in starts:
            mix.append(dict(weight=1.0, dsl=f'E({chunk_len}) Q(0,1,{s},{wl})'))
    return mix


hp['curriculum'] = [
    # Foundation stage — chunk_len=8, n_chunks=1, matching sanity_c8's own proven-
    # convergent shape/dense-anchor-sweep, trained from scratch (see this file's
    # docstring — the external warm-start checkpoint this stage was meant to replace
    # was lost when tpu2 was deleted/reclaimed). hop_drop_prob=0 — n_chunks=1 has no
    # chunk-to-chunk relationship for enc_hops to window.
    dict(n_chunks=1, chunk_len=8, B=16, n_steps=100000, eval_every=5000,
         hops=-1, early_stop_mean=70.0, hop_drop_prob=0.0,
         weave_mix=_intra_chunk_grid(8, [2, 3, 4], n_anchors=4, min_recall_len=4)),
    dict(n_chunks=4, chunk_len=8, B=16, n_steps=40000, eval_every=5000,
         hops=-1, early_stop_mean=70.0, hop_drop_prob=0.3,
         weave_mix=_weave_mix_for(4, 8, [4, 8])),
    dict(n_chunks=8, chunk_len=8, B=16, n_steps=40000, eval_every=5000,
         hops=-1, early_stop_mean=70.0, hop_drop_prob=0.4,
         weave_mix=_weave_mix_for(8, 8, [8, 16])),
    dict(n_chunks=16, chunk_len=8, B=16, n_steps=40000, eval_every=5000,
         hops=-1, early_stop_mean=70.0, hop_drop_prob=0.5,
         weave_mix=_weave_mix_for(16, 8, [16, 32])),
    dict(n_chunks=16, chunk_len=32, B=8, n_steps=50000, eval_every=5000,
         hops=-1, early_stop_mean=60.0, hop_drop_prob=0.6,
         weave_mix=_weave_mix_for(16, 32, [32, 64])),
    # hop_drop_prob=0.7 at the final/target shape — since hops=1 is stated as the REAL
    # deployment use case (not just a robustness bonus), bias training so the pure-
    # hop=1 case (all of back 2/3/4 dropped) occurs MORE often than the full window
    # (P(pure hop=1)=34.3% vs P(full window)=2.7% at p=0.7, vs. a symmetric 12.5%/12.5%
    # at the old p=0.5) — every step still keeps back=1 (never dropped), so this
    # doesn't touch the "hops=-1 unbounded" op-relay default, only the enc_hops window.
    dict(n_chunks=16, chunk_len=64, B=8, n_steps=60000, eval_every=5000,
         hops=-1, early_stop_mean=50.0, hop_drop_prob=0.7,
         weave_mix=_weave_mix_for(16, 64, [32, 64])),
]
