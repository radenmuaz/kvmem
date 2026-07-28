"""
`hmn_weave_c64_scaledrope.py` — redo of `hmn_weave_c64.py` under
`rope_state_scale` (see `hmn_single_recall_c64_scaledrope.py`'s docstring
and `docs/HISTORY.md` §12 for the full derivation — this mechanism
supersedes the earlier `dual_rope` design). Warm-started from
`hmn_single_recall_c64_scaledrope.py`'s checkpoint, not the original
RoPE's or `_dualrope`'s (each is a distinct, incompatible positional
mechanism).

THIS is the actual test the mechanism was built for: `traj1`(`batch`)/
`traj3`(`interleave_delayed`) share a byte-identical `E2` encode prefix and
were measured (`kvmem/probe_positional_shortcut.py`) to be resolved via
pure query-slot POSITION rather than content. `rope_state_scale`
compresses STATE-region positions toward numerically-negligible distance
differences specifically to remove that signal; if it worked, `batch`/
`interleave_delayed` should no longer be stuck near-random (8-20% match
across every previous run of this config, under both plain RoPE and the
`dual_rope` attempt). `stream` was never the problem (42-49% match
previously) and isn't expected to change much.

Same architecture/curriculum/hyperparameters as `hmn_weave_c64.py`
otherwise (2-stage, `hops=1`, per-shape `B8`/`B16`, `cosine_T0=160000`
shared slow decay) — only `rope_state_scale=1e6` and the warm-start
checkpoint differ, so any change in `batch`/`interleave_delayed` outcome
is attributable to the positional mechanism, not a confound from also
changing the optimizer/curriculum at the same time.

Run (never two jobs at once, and only after
hmn_single_recall_c64_scaledrope.py has produced a checkpoint):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_weave_c64_scaledrope.py --device mps
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
    name='hmn_weave_c64_scaledrope', seed=50,
    _pretrained_ckpt='logs/hmn_single_recall_c64_scaledrope/checkpoints/stage0_best.pt',
    rope_state_scale=1e6,

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=2, chunk_len=64, B=4, n_steps=80000, eval_every=10000,
             hops=1,
             weave_mix=[
                 dict(weight=1.0, dsl='E2 Q(0,1) Q(1,2) B8'),        # batch(nc=2,wc=1)
                 dict(weight=1.0, dsl='E Q(0,1) E Q(1,2) B8'),      # stream(nc=2,wc=1)
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
