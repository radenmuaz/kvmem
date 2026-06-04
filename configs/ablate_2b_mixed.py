"""
Stage 3 — Two blocks, content-addressed routing.
Mix from=0 and from=1. Model must route to the correct chunk from anchor alone.

Pass: >=80% on both from=0 and from=1 simultaneously.
Prerequisite: ablate_2b_recent AND ablate_2b_old must pass first.

Run:
    python -m kvmem.train --config configs/ablate_2b_mixed.py --device mps
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256,
    B=16, lr_max=3e-4, wd=0.001,
    warmup_steps=1000, cycle_steps=-1,  # -1 = cosine over full run,
    eval_every=5000, log_every=1000,
    drop_close_prob=0.5, dataset_size=20000, seed=42,
    ocd=False, ocd_prob=0.01, tf_warmup=0,
    rope=True, yarn=True, grok=False,
    stablemax=False, eval_offset=0.25, grad_clip=10.0,
    mem_window=0,  # 0=full; 1=isolated; N=window
    compile=False,
    name='ablate_2b_mixed',

    # Two stages: one for each recall target, equal steps
    curriculum=[
        dict(n_blocks=2, recall_from=0,
             seg_len=16, slot_len=1, intermed_len=7,
             warmup_len=4, out_len=8,
             B=16, n_steps=80000),
        dict(n_blocks=2, recall_from=1,
             seg_len=16, slot_len=1, intermed_len=7,
             warmup_len=4, out_len=8,
             B=16, n_steps=80000),
    ],
)
