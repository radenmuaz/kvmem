"""
`hmn_weave_c64_adaptive.py` — clone of `hmn_weave_c64.py`, run through
`kvmem/hmn_adaptive_trainer.py` instead of `kvmem/hmn.py`. Same
architecture/curriculum/warm-start/DSL trajectories (`batch`/`stream`/
`interleave_delayed`, `hops=1`, per-shape `B8`/`B16`) — the only change is
turning on adaptive weave_mix reweighting (`adaptive=True`), which shifts
each stage's sampling weight toward whichever trajectory shape is
currently lagging instead of holding a fixed uniform 1/3-1/3-1/3 split.

`adapt_signal='train_loss'` — uses the per-trajectory train-loss EMA
(updates every step) rather than `val_match` (only updates at eval steps,
and the trainer's own `_eval_count>=2` gate means val_match wouldn't even
start adapting until the second eval of each stage). train_loss gives a
finer-grained, earlier-available signal at the cost of being loss-scale
rather than the actual target metric.

repeat_batch adaptation is disabled in the trainer itself (see
`hmn_adaptive_trainer.py`'s own docstring) — only sampling weight adapts;
each entry's `B8`/`B16` stays fixed exactly as authored below.

Cosine LR config removed entirely (not just ignored) — the adaptive
trainer's weave_mix branch always uses fixed-lr-after-linear-warmup and
never reads `cosine_T0`/`cosine_T_mult`/`lr_schedule`/`lr_min`, so keeping
those keys around would just be dead, misleading config.

Run (never two jobs at once):
    python3 -m kvmem.hmn_adaptive_trainer --config kvmem/configs/hmn_weave_c64_adaptive.py --device mps
"""

hp = dict(
    d=64, n_layers=8, n_heads=4, V=274,
    block_type='single_attn',
    lr_max=1e-4, wd=0.0001,
    warmup_steps=5000, log_every=1000,
    rope=True, yarn=True, null_kv=True,
    rmsnorm=True,
    name='hmn_weave_c64_adaptive', seed=50,
    _pretrained_ckpt='logs/hmn_single_recall_c64/checkpoints/stage0_best.pt',

    adaptive=True,
    adapt_signal='train_loss',
    adapt_temp=0.2,  # aggressive (softmax temperature, lower = more aggressive) — at the loss spreads
                     # actually observed last run (e.g. [3.68,2.92,4.59]), the default T=1.0 only moved
                     # weights to ~[0.32,0.27,0.41]; T=0.2 pushes the same spread to ~[0.22,0.09,0.69],
                     # a real reallocation instead of staying near uniform

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
