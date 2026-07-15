"""
Multi-block ablation — Type 1+2 combined: two blocks, model selects correct block.

Randomly mixes recall_from=0 and recall_from=1 within the same training run
by using two curriculum stages (one per recall target).

Pass criterion: >=80% match on BOTH src0 and src1 recall (no block confusion).
Prerequisites: ablate_t1 and ablate_t2 must each pass individually first.

Run:
    python -m kvmem.train --config configs/ablate_t1t2_combined.py --device mps
"""

hp = dict(
    seg_len=16, slot_len=8, active_slots=2,
    warmup_len=4, out_len=8,
    d=64, n_layers=4, n_heads=4, d_ff=256,
    lr_max=3e-4, wd=0.001, warmup_steps=1000, cycle_steps=0,
    eval_every=5000, log_every=1000,
    dataset_size=10000, seed=42,
    ocd=False, ocd_prob=0.01, tf_warmup=0,
    rope=True, yarn=True, grok=False,
    stablemax=False, eval_offset=0.25, grad_clip=10.0,
    kv_cache=True, compile=False,
    name='ablate_t1t2',

    # Two stages: one for each recall target (interleaved equally)
    curriculum=[
        dict(n_blocks=2, recall_from=0,
             seg_len=16, slot_len=8, warmup_len=4, out_len=8,
             B=16, n_steps=80000),
        dict(n_blocks=2, recall_from=1,
             seg_len=16, slot_len=8, warmup_len=4, out_len=8,
             B=16, n_steps=80000),
    ],
)
