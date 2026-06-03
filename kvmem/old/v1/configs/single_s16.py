"""
Single-block recall, seg=16, slot=8, active_slots=1.
Best config so far: val_bpb=0.249, 93.8% match @40k steps (full-pass TF).
KV-cache mode is now default — trains same task with blockwise forward
that matches inference computation.

Run:
    python -m kvmem.train --config configs/single_s16.py --device mps
"""

hp = dict(
    # Sequence layout
    n_blocks=1, recall_from=0,
    seg_len=16, slot_len=8, active_slots=2,
    warmup_len=4, out_len=8,

    # Model
    d=64, n_layers=4, n_heads=4, d_ff=256, V=256,

    # Training
    B=16, lr_max=3e-4, wd=0.001,
    warmup_steps=1000, cycle_steps=0,    # flat LR after warmup
    n_steps=40000, eval_every=5000, log_every=1000,

    # Data
    slot_style='seq', drop_close_prob=0.5,
    dataset_size=10000, seed=42,

    # OCD
    ocd=False, ocd_prob=0.01, tf_warmup=0,

    # Misc
    rope=True, yarn=True, grok=False,
    stablemax=False, eval_offset=0.25, grad_clip=10.0,
    kv_cache=True,   # blockwise KV (default, matches inference)
    compile=False,
    name='single_s16',
    curriculum=None,   # single stage from this config
)
