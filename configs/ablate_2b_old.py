"""
Stage 2 — Two blocks, recall from earlier chunk.
DSL: 2x<h:1><x:16><z:7><q:4><y:8,from=0>

Key test. After seeing chunk_1, model must still recall chunk_0 from h_1.
Tests whether fast-weight update propagates and retains early information.
Pass: >=80% match.

Run:
    python -m kvmem.train --config configs/ablate_2b_old.py --device mps
"""

hp = dict(
    n_blocks=2, recall_from=0,
    seg_len=16, slot_len=1, intermed_len=7,
    warmup_len=4, out_len=8,

    d=64, n_layers=4, n_heads=4, d_ff=256,

    B=16, lr_max=3e-4, wd=0.001,
    warmup_steps=1000, cycle_steps=-1,  # -1 = cosine over full run,
    n_steps=80000, eval_every=5000, log_every=1000,

    drop_close_prob=0.5,
    dataset_size=20000, seed=42,

    ocd=False, ocd_prob=0.01, tf_warmup=0,

    rope=True, yarn=True, grok=False,
    stablemax=False, eval_offset=0.25, grad_clip=10.0,
    mem_window=0,  # 0=full; 1=isolated; N=window
    compile=False,
    name='ablate_2b_old',
    curriculum=None,
)
