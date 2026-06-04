"""
Experiment 2 — Multi-turn corpus recall.

Tests all 4 recall situations in a single curriculum with cosine LR.
mem_window=0: full fast-weight accumulation (h_i sees all prior h_j).

Situations:
  Stage 0: 1 block, recall (baseline — should hit ~100%)
  Stage 1: 2 blocks, recall from block 1 (recent, warm-up)
  Stage 2: 2 blocks, recall from block 0 (old, key test)
  Stage 3: 2 blocks, mixed from=0 and from=1 (routing)
  Stage 4: 2 blocks, mixed — mem_window=1 (isolated blocks, no history)
  Stage 5: 2 blocks, mixed — mem_window=2 (1-step Markov update)

Pass criteria:
  Stage 0: >=98%  (sanity — regression from single-block baseline)
  Stage 1: >=90%  (recency should be easy)
  Stage 2: >=80%  (key test: fast-weight retention)
  Stage 3: >=80% both  (content routing)
  Stage 4 vs 5 vs 0: compare to understand window effect

Run:
    python -m kvmem.train --config configs/exp2_multiturn.py --device mps
"""

_BLOCK = dict(seg_len=16, slot_len=1, intermed_len=7, warmup_len=4, out_len=8)

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256,
    B=16, lr_max=3e-4, wd=0.001,
    warmup_steps=1000, cycle_steps=-1,   # cosine over each stage
    eval_every=5000, log_every=1000,
    drop_close_prob=0.5, dataset_size=20000, seed=42,
    ocd=False, ocd_prob=0.01, tf_warmup=0,
    rope=True, yarn=True, grok=False,
    stablemax=False, eval_offset=0.25, grad_clip=10.0,
    mem_window=0,
    compile=False,
    name='exp2_multiturn',

    curriculum=[
        # Stage 0: baseline single-block (regression check)
        dict(**_BLOCK, n_blocks=1, recall_from=0, mem_window=0, B=16, n_steps=40000),

        # Stage 1: 2 blocks, recall recent (warm-up)
        dict(**_BLOCK, n_blocks=2, recall_from=1, mem_window=0, B=16, n_steps=80000),

        # Stage 2: 2 blocks, recall old (key test)
        dict(**_BLOCK, n_blocks=2, recall_from=0, mem_window=0, B=16, n_steps=80000),

        # Stage 3: 2 blocks, mixed routing (from=0 and from=1)
        dict(**_BLOCK, n_blocks=2, recall_from=0, mem_window=0, B=16, n_steps=80000),
        dict(**_BLOCK, n_blocks=2, recall_from=1, mem_window=0, B=16, n_steps=80000),

        # Stage 4: isolated blocks (mem_window=1) — no fast-weight accumulation
        dict(**_BLOCK, n_blocks=2, recall_from=0, mem_window=1, B=16, n_steps=80000),
        dict(**_BLOCK, n_blocks=2, recall_from=1, mem_window=1, B=16, n_steps=80000),

        # Stage 5: 1-step Markov (mem_window=2)
        dict(**_BLOCK, n_blocks=2, recall_from=0, mem_window=2, B=16, n_steps=80000),
        dict(**_BLOCK, n_blocks=2, recall_from=1, mem_window=2, B=16, n_steps=80000),
    ],
)
