"""
Ablation of `hmn_weave_mix_accum_rnn.py` — identical in every way (same
architecture, seed, warm-start, curriculum, 160000 steps, `hops=1`) except
`repeat_batch=8`: each sampled batch gets 8 consecutive gradient steps
before a fresh batch is drawn, instead of a fresh batch every step. Same
total optimizer-step count and LR schedule as the baseline, so `loss`/
per-pattern val is directly comparable between
`logs/hmn_weave_mix_accum_rnn/train.log` (baseline) and
`logs/hmn_weave_mix_accum_rnn_repeat8/train.log` (this config).

Motivation: the baseline's train loss looked like it was plateauing/noisy
mid-run (bouncing 2.1-3.3 around step 60000-70000, val MEAN crawling
36.6%->41.4%->43.8% across steps 20000/60000/70000) rather than dropping
cleanly. `repeat_batch` lets several gradient steps actually fit one
sampled batch before moving on, in case one step per batch isn't enough to
make progress on this much harder (varied-trajectory-shape + forced
single-hop relay) task, the same hypothesis `hmn_single_recall_c64_repeat4`
tests on the simpler single-chunk task.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_weave_mix_accum_rnn_repeat8.py --device mps
"""

hp = dict(
    d=64, n_layers=8, n_heads=4, V=274,
    block_type='single_attn',
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=160000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True,
    rmsnorm=True,
    name='hmn_weave_mix_accum_rnn_repeat8', seed=50,
    _pretrained_ckpt='logs/hmn_single_recall_c64/checkpoints/stage0_best.pt',
    repeat_batch=8,

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=4, chunk_len=16, window_chunks=2, B=6, n_steps=160000, eval_every=10000,
             hops=1,
             weave_mix=[
                 dict(weight=1.0, pattern='batch'),
                 dict(weight=1.0, pattern='stream'),
                 dict(weight=1.0, pattern='interleave_delayed'),
             ]),
    ],
)
