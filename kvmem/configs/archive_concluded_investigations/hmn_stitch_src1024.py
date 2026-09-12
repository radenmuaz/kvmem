"""
`hmn_stitch_src1024.py` — REDESIGNED (superseded the original multi-hop
relay-chain version): single query per training example, no relay chain at
all. Encode n_chunks chunks, then ONE query recalling the SUFFIX of the
source — warmup anchors partway through (8 real ground-truth bytes), and
the response must generate everything after that anchor through the TRUE
END of the source, whatever length that happens to be. `traj_suffix`
(`kvmem/hmn.py`) builds this: `Q(n_chunks-window_chunks, n_chunks)` — here
`window_chunks` means "how many chunks back from the end the warmup
anchor sits," not a sliding-window size. Since there's only ever ONE query
(`op_idx=0`, always exempt from the `hops`-bounded relay restriction — see
`chunk_mask_fb_traj`), none of `hops`/`forward_granularity`/
`segment_checkpoint` are needed — this is a plain dense forward pass, much
cheaper than the abandoned relay-chain design (L=1452 at n_chunks=16 vs.
2800-4900 for the old 30-126-hop chains).

Warm-started from `hmn_weave_c64`'s checkpoint (chunk_len=64, matching this
config, see that file's own docstring for why chunk-size matching matters).

**Curriculum** (3 stages within one run, each continuing the same
model/optimizer): ramps `n_chunks` up gradually rather than training on the
full 1024-byte source from the start —
  - stage 0: n_chunks in {2,4} — short sources, easy
  - stage 1: n_chunks in {2,4,8} — adds the mid-length case
  - stage 2: n_chunks in {2,4,8,16} — the full 1024-byte source

Within EVERY stage, `window_chunks` is also mixed across a few values
(always `>=2`, enforced by `traj_suffix` itself — "if warmup lands too
close to the end, there'd be nothing meaningful left to generate; the
minimum case is no multi-chunk stitching at all, just a plain 2-chunk
recall") so warmup anchors at varying distances from the end, not just one
fixed position — this is the practical stand-in for "warmup from any
index": exact byte-arbitrary anchoring isn't expressible in one packed
sequence (each shape needs its own fixed layout), but mixing several
chunk-aligned anchor points across weighted trajectories approximates it.

**Extrapolation check (held out from training entirely)**: after this run,
evaluate at n_chunks=24 or 32 — sizes NEVER seen during training (which
caps at 16) — using the existing `ar_decode_traj_nokv` decode function
(already generic over any `chunk_positions_traj`-built layout, no new code
needed) to see whether recall-to-the-true-end generalizes past the
trained source length, the same kind of test `eval_weave.py`'s
`long_hop_recovery` already does for the old relay-chain track.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_stitch_src1024.py --device mps
"""

hp = dict(
    d=64, n_layers=8, n_heads=4, V=274,
    block_type='single_attn',
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=40000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True,
    rmsnorm=True,
    name='hmn_stitch_src1024', seed=51,
    _pretrained_ckpt='logs/hmn_weave_c64/checkpoints/stage0_best.pt',

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    # Each entry is spelled out as an explicit DSL string (see parse_traj_dsl's
    # grammar comment, kvmem/hmn.py) rather than pattern='suffix'+n_chunks+
    # window_chunks — e.g. 'E4 Q(2,4)' reads directly as "ingest+compress 4
    # chunks, then one query recalling chunks [2,4) — i.e. warmup anchors at
    # chunk 2, response covers chunk 2's tail + chunk 3 through the true end."
    # No n_chunks/window_chunks bookkeeping needed — the string IS the shape.
    curriculum=[
        dict(n_chunks=4, chunk_len=64, B=6, n_steps=10000, eval_every=2000,
             weave_mix=[
                 dict(weight=1.0, dsl='E2 Q(0,2)'),   # n_chunks=2, window_chunks=2 (whole thing)
                 dict(weight=1.0, dsl='E4 Q(2,4)'),   # n_chunks=4, window_chunks=2 (anchor at chunk 2)
                 dict(weight=1.0, dsl='E4 Q(0,4)'),   # n_chunks=4, window_chunks=4 (whole thing)
             ]),
        dict(n_chunks=8, chunk_len=64, B=6, n_steps=15000, eval_every=3000,
             weave_mix=[
                 dict(weight=1.0, dsl='E2 Q(0,2)'),
                 dict(weight=1.0, dsl='E4 Q(0,4)'),
                 dict(weight=1.0, dsl='E8 Q(4,8)'),   # n_chunks=8, window_chunks=4 (anchor at chunk 4)
                 dict(weight=1.0, dsl='E8 Q(0,8)'),   # n_chunks=8, window_chunks=8 (whole thing)
             ]),
        dict(n_chunks=16, chunk_len=64, B=6, n_steps=15000, eval_every=3000,
             weave_mix=[
                 dict(weight=1.0, dsl='E4 Q(0,4)'),
                 dict(weight=1.0, dsl='E8 Q(0,8)'),
                 dict(weight=1.0, dsl='E16 Q(8,16)'),  # n_chunks=16, window_chunks=8 (anchor at chunk 8)
                 dict(weight=1.0, dsl='E16 Q(0,16)'),  # n_chunks=16, window_chunks=16 (whole thing)
             ]),
    ],
)
