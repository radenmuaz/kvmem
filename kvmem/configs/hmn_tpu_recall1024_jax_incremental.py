"""
`hmn_tpu_recall1024_jax_incremental.py` — the careful, one-variable-at-a-
time expansion path from `hmn_tpu_sanity_w25_rope_jax.py`'s single-chunk
foundation (chunk_len 8->16->32->64, staged, `early_stop_mean=80.0`
gated) toward the full 1024-byte recall1024 target, designed explicitly
to avoid this session's two prior over-ambitious jumps:
  - `hmn_tpu_recall1024_jax_curriculum_staged.py` (original) started at
    `n_chunks=2, chunk_len=64` directly — already a multi-chunk AND
    largest-chunk_len shape at once, with no single-chunk foundation
    proven first. Got stuck near-chance every stage.
  - `hmn_tpu_recall1024_jax_hopdrop.py` started its multi-chunk stages at
    `n_chunks=4` (not 2) with `hop_drop_prob>0` active from the very
    first multi-chunk stage — three new axes of difficulty (multi-chunk
    AT ALL, a bounded+dropped relay window, and n_chunks=4 not 2) all
    introduced simultaneously.

This file changes exactly ONE axis per stage, in the order judged
cheapest-to-hardest:
  A. n_chunks alone, 2 -> 4 -> 8 -> 16, chunk_len FIXED at the smallest
     proven value (8) and hops=-1 (unbounded/routing — the query keeps
     PERMANENT access to every chunk's STATE, so "can it do multi-chunk
     recall at all" is tested in isolation from "can it do it through a
     bounded window").
  B. chunk_len alone, 8 -> 16 -> 32 -> 64, n_chunks FIXED at 16 (the full
     target chunk count), still hops=-1 — reaches the true 1024-byte
     target shape (stage G) while still in the "easy"/unbounded relay
     mode.
  C. ONLY THEN, `enc_hops`/`hop_drop_prob` introduced at the full 1024-
     byte shape — first the deterministic bounded window alone (`hop_
     drop_prob=0.0`, stage H — does windowing alone, without stochastic
     dropout, still work), then dropout annealed in gently (stage I).

Every multi-chunk stage's `weave_mix` uses `_grid_stitch`: ONE anchor
position per chunk index (the very start of each chunk 0..n_chunks-1,
not just fractions clustered near the end) x 2 warmup_lens — this alone
satisfies the corrected roadmap stage-2 requirement ("anchor may sit on
ANY chunk, response covers 2+ chunks unless the anchor's chunk is the
last one") because the plain suffix-to-true-end response length is
already `(n_chunks - anchor_chunk_idx)` chunks by construction; the fix
needed was just making sure every chunk index actually appears as an
anchor, not clustering anchors near the end as `hmn_tpu_recall1024_jax_
hopdrop.py`'s `n_anchors`-fraction sweep implicitly did.

`hp['pretrained_ckpt']` points at `hmn_tpu_sanity_w25_rope_jax.py`'s own
`stage3_best.pt` (the chunk_len=64 single-chunk stage, the hardest/last
stage of that foundation run) — **this file must not be launched until
that checkpoint actually exists** (i.e. until the foundation run reaches
its stage 3 or is deliberately stopped after a good-enough intermediate
stage). Architecture matches exactly (`d=128/n_layers=16/n_heads=8/V=271/
state_len=4/state_vocab_size=1`) for the load to succeed.

Run (never two jobs at once; do not run before the foundation checkpoint exists):
    python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_tpu_recall1024_jax_incremental.py
"""


def _grid_stitch(n_chunks, chunk_len, warmup_lens, min_recall_len=None):
    """One anchor per chunk index (the start of each chunk 0..n_chunks-1),
    per warmup_len — guarantees every possible "how many chunks left to
    recall" case is represented, from n_chunks (anchor in chunk 0) down to
    a single partial chunk (anchor in the last chunk). See this file's own
    docstring for why this alone satisfies the generalized-anchor
    requirement without needing a separate "2+ chunks" filter."""
    total = n_chunks * chunk_len
    min_recall_len = min_recall_len if min_recall_len is not None else max(4, chunk_len // 2)
    mix = []
    for wl in warmup_lens:
        for a in range(n_chunks):
            anchor = a * chunk_len
            if total - anchor - wl < min_recall_len:
                continue  # anchor too close to the true end for even min_recall_len bytes
            if n_chunks > 1:
                dsl = f'E({chunk_len}) E{n_chunks - 1} Q(0,{n_chunks},{anchor},{wl})'
            else:
                dsl = f'E({chunk_len}) Q(0,{n_chunks},{anchor},{wl})'
            mix.append(dict(weight=1.0, dsl=dsl))
    return mix


hp = dict(
    d=128, n_layers=16, n_heads=8, V=271,
    lr_max=1e-4, wd=1e-5,
    warmup_steps=1000, log_every=500,
    rope=True, yarn=True,
    null_kv=True,
    rmsnorm=True,
    name='hmn_tpu_recall1024_jax_incremental', seed=48,

    adaptive=True,
    adapt_signal='val_match',
    adapt_temp=1.0,
    adapt_ema_alpha=0.5,
    adapt_floor=0.05,

    state_len=4, state_vocab_size=1,
    warmup_len=8,
    val_n_seqs=3,

    bucket_lengths=False,
    eval_combinatorial_hops=True,  # inert (no-op) until stage H/I set enc_hops!=-1
    warm_start_from_best=True,
    pretrained_ckpt='logs/hmn_tpu_sanity_w25_rope_jax/checkpoints/stage3_best.pt',

    curriculum=[
        # --- A: n_chunks alone, chunk_len fixed at 8, hops=-1 ---
        dict(n_chunks=2, chunk_len=8, B=16, n_steps=60000, eval_every=5000,
             hops=-1, early_stop_mean=80.0,
             weave_mix=_grid_stitch(2, 8, [2, 4])),
        dict(n_chunks=4, chunk_len=8, B=16, n_steps=60000, eval_every=5000,
             hops=-1, early_stop_mean=80.0,
             weave_mix=_grid_stitch(4, 8, [2, 4])),
        dict(n_chunks=8, chunk_len=8, B=16, n_steps=60000, eval_every=5000,
             hops=-1, early_stop_mean=75.0,
             weave_mix=_grid_stitch(8, 8, [2, 4])),
        dict(n_chunks=16, chunk_len=8, B=16, n_steps=60000, eval_every=5000,
             hops=-1, early_stop_mean=70.0,
             weave_mix=_grid_stitch(16, 8, [2, 4])),

        # --- B: chunk_len alone, n_chunks fixed at 16, hops=-1 ---
        dict(n_chunks=16, chunk_len=16, B=8, n_steps=80000, eval_every=5000,
             hops=-1, early_stop_mean=65.0,
             weave_mix=_grid_stitch(16, 16, [4, 8])),
        dict(n_chunks=16, chunk_len=32, B=8, n_steps=80000, eval_every=5000,
             hops=-1, early_stop_mean=60.0,
             weave_mix=_grid_stitch(16, 32, [8, 16])),
        dict(n_chunks=16, chunk_len=64, B=8, n_steps=100000, eval_every=5000,
             hops=-1, early_stop_mean=55.0,
             weave_mix=_grid_stitch(16, 64, [16, 32])),

        # --- C: enc_hops/hop_drop_prob, ONLY at the full target shape ---
        dict(n_chunks=16, chunk_len=64, B=8, n_steps=80000, eval_every=5000,
             hops=-1, enc_hops=4, hop_drop_prob=0.0, early_stop_mean=50.0,
             weave_mix=_grid_stitch(16, 64, [16, 32])),
        dict(n_chunks=16, chunk_len=64, B=8, n_steps=100000, eval_every=5000,
             hops=-1, enc_hops=4, hop_drop_prob=0.2, early_stop_mean=45.0,
             weave_mix=_grid_stitch(16, 64, [16, 32])),
    ],
)
