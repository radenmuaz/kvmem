"""
Single-block recall, seg=16.

Bottleneck: slot_len=1, ponder_len=7  (matches v1 active_slots=1 with slot_len=8)
active_slots=0 — no masking needed; slot_len IS the bottleneck directly.
Ponder tokens see src + slots, providing productive extra compute vs inactive slots.

Sequence: <s>src(16)</s><m>slot_0</m><f>warmup(4)</f><p>p0..p6</p><c>output(8)</c>
L = 1+16+1+1+1+1+1+4+1+1+7+1+1+8+1 = 46

Run:
    python -m kvmem.train --config configs/single_s16.py --device mps
"""

hp = dict(
    # Sequence layout — slot_len IS the bottleneck, no active_slots masking
    n_blocks=1, recall_from=0,
    seg_len=16, slot_len=1, ponder_len=7, active_slots=0,
    warmup_len=4, out_len=8,

    # Model
    d=64, n_layers=4, n_heads=4, d_ff=256,

    # Training
    B=16, lr_max=3e-4, wd=0.001,
    warmup_steps=1000, cycle_steps=0,
    n_steps=40000, eval_every=5000, log_every=1000,

    # Data
    drop_close_prob=0.5,
    dataset_size=10000, seed=42,

    # OCD
    ocd=False, ocd_prob=0.01, tf_warmup=0,

    # Misc
    rope=True, yarn=True, grok=False,
    stablemax=False, eval_offset=0.25, grad_clip=10.0,
    compile=False,
    name='single_s16',
    curriculum=None,
)
