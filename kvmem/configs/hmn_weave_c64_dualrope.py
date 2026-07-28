"""
`hmn_weave_c64_dualrope.py` — redo of `hmn_weave_c64.py` under the new
dual-clock RoPE mechanism (`dual_rope=True`, see
`hmn_single_recall_c64_dualrope.py`'s docstring and `docs/HISTORY.md` §12
for the full derivation). Warm-started from
`hmn_single_recall_c64_dualrope.py`'s checkpoint instead of the original
`hmn_single_recall_c64`'s, since dual_rope changes the position values fed
into every attention layer — not warm-startable from a checkpoint trained
under the old single-clock scheme.

THIS is the actual test the whole dual-clock mechanism was built for:
`traj1`(`batch`)/`traj3`(`interleave_delayed`) share a byte-identical `E2`
encode prefix and were measured (`kvmem/probe_positional_shortcut.py`) to
be resolved via pure query-slot POSITION rather than content — under the
old scheme, the two queries were literally distinguishable by raw
token-distance to each STATE. `dual_rope` freezes the macro clock during
the query phase specifically to remove that distinguishing signal; if it
worked, `batch`/`interleave_delayed` should no longer be stuck near-random
(8-20% match across every previous run of this config) — `stream` was
never the problem (42-49% match previously) and isn't expected to change
much.

Same architecture/curriculum/hyperparameters as `hmn_weave_c64.py`
otherwise (2-stage, `hops=1`, per-shape `B8`/`B16`, `cosine_T0=160000`
shared slow decay) — only `dual_rope=True` and the warm-start checkpoint
differ, so any change in `batch`/`interleave_delayed` outcome is
attributable to the positional mechanism, not a confound from also
changing the optimizer/curriculum at the same time.

Run (never two jobs at once, and only after
hmn_single_recall_c64_dualrope.py has produced a checkpoint):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_weave_c64_dualrope.py --device mps
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
    name='hmn_weave_c64_dualrope', seed=50,
    _pretrained_ckpt='logs/hmn_single_recall_c64_dualrope/checkpoints/stage0_best.pt',
    dual_rope=True,

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
