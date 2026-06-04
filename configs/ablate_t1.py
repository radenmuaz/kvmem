"""
Multi-block ablation — Type 1: two ingestion blocks, recall from block 0.

Sequence:
  <s>src0</s><m>slots0</m>  <s>src1</s><m>slots1</m>
  <f>anchor_in_src0</f><c>output_from_src0</c>

The model must recall src0 despite having seen src1 between ingestion
and recall. slots1 cannot attend to src0 (strict block isolation).

Pass criterion: >=80% match on src0 recall (ignoring src1 content).

Run:
    python -m kvmem.train --config configs/ablate_t1.py --device mps
"""

hp = dict(
    # Multi-block layout
    n_blocks=2, recall_from=0,
    seg_len=16, slot_len=8, active_slots=2,
    warmup_len=4, out_len=8,

    # Model (same as single-block baseline for fair comparison)
    d=64, n_layers=4, n_heads=4, d_ff=256,
    # Training
    B=16, lr_max=3e-4, wd=0.001,
    warmup_steps=1000, cycle_steps=0,
    n_steps=80000, eval_every=5000, log_every=1000,

    # Data
    dataset_size=10000, seed=42,

    # OCD
    ocd=False, ocd_prob=0.01, tf_warmup=0,

    # Misc
    rope=True, yarn=True, grok=False,
    stablemax=False, eval_offset=0.25, grad_clip=10.0,
    kv_cache=True,
    compile=False,
    name='ablate_t1',
    curriculum=None,
)
