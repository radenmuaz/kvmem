"""
Multi-block ablation — Type 2: two ingestion blocks, recall from block 1.

Sequence:
  <s>src0</s><m>slots0</m>  <s>src1</s><m>slots1</m>
  <f>anchor_in_src1</f><c>output_from_src1</c>

After encoding src0, the model must also correctly encode src1 and
recall from it — testing that the second slot bank works correctly.

Pass criterion: >=80% match on src1 recall.

Run:
    python -m kvmem.train --config configs/ablate_t2.py --device mps
"""

hp = dict(
    n_blocks=2, recall_from=1,
    seg_len=16, slot_len=8, active_slots=2,
    warmup_len=4, out_len=8,

    d=64, n_layers=4, n_heads=4, d_ff=256,
    B=16, lr_max=3e-4, wd=0.001,
    warmup_steps=1000, cycle_steps=0,
    n_steps=80000, eval_every=5000, log_every=1000,
    dataset_size=10000, seed=42,

    ocd=False, ocd_prob=0.01, tf_warmup=0,

    rope=True, yarn=True, grok=False,
    stablemax=False, eval_offset=0.25, grad_clip=10.0,
    kv_cache=True,
    compile=False,
    name='ablate_t2',
    curriculum=None,
)
