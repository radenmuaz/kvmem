"""
`hmn_weave_c64_relpos.py` — redo of `hmn_weave_c64.py` under
`kvmem/hmn_relpos.py`'s alternative positional mechanism (`rope=False`,
`relpos_enabled=True` — see `hmn_single_recall_c64_relpos.py`'s docstring).
Warm-started from `hmn_single_recall_c64_relpos.py`'s checkpoint, not the
original RoPE checkpoint (different, incompatible positional mechanism).

Same test target as `hmn_weave_c64_dualrope.py`: `traj1`(`batch`)/
`traj3`(`interleave_delayed`) share a byte-identical `E2` encode prefix and
were measured to be resolved via query-slot POSITION rather than content.
This config attacks the same bug from the opposite direction — instead of
scoping distance information to a macro/local dual clock, it removes
essentially all distance signal except immediate-adjacency. If the model
was relying on smooth RoPE distance decay specifically, this should also
fix `batch`/`interleave_delayed`; if it doesn't, that's evidence the
shortcut isn't really about "distance decay" as such and something else
(e.g. some other structural cue) is doing the work.

Same architecture/curriculum/hyperparameters as `hmn_weave_c64.py`
otherwise (2-stage, `hops=1`, per-shape `B8`/`B16`, `cosine_T0=160000`
shared slow decay) — only the positional mechanism and warm-start
checkpoint differ from the original, and both `_dualrope`/`_relpos`
siblings share the SAME baseline curriculum/hparams so the three results
(original RoPE, dual-clock RoPE, relpos) are directly comparable.

Run (never two jobs at once, and only after
hmn_single_recall_c64_relpos.py has produced a checkpoint):
    python3 -m kvmem.hmn_relpos --config kvmem/configs/hmn_weave_c64_relpos.py --device mps
"""

hp = dict(
    d=64, n_layers=8, n_heads=4, V=274,
    block_type='single_attn',
    lr_max=1e-4, lr_min=1e-6, wd=0.0001,
    warmup_steps=5000, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=160000, cosine_T_mult=1,
    rope=False, null_kv=True,
    relpos_enabled=True,
    rmsnorm=True,
    name='hmn_weave_c64_relpos', seed=50,
    _pretrained_ckpt='logs/hmn_single_recall_c64_relpos/checkpoints/stage0_best.pt',

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
