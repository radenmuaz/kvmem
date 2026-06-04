"""
Exp A — null_kv convergence ablation (1-step only).
Two runs compare convergence rate with/without zero KV append.
Both: n1/r0, 40k steps.

Run without null_kv:
    python -m kvmem.train --config configs/expA_nullkv.py --device mps --name expA_base

Run with null_kv:
    python -m kvmem.train --config configs/expA_nullkv.py --null-kv --device mps --name expA_nullkv
"""

from kvmem.curriculum_dsl import parse_curriculum

_seq, _curriculum, _eval = parse_curriculum(
    "<x:16><z:7><h:1><q:4><y:8> | n1/r0/40k @eval:n1/r0",
    B=16, dataset_size=20000, cycle_steps=-1,
)

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    eval_every=2000, log_every=500,
    drop_close_prob=0.5, seed=42,
    ocd=False, ocd_prob=0.01, tf_warmup=0,
    rope=True, yarn=True, grok=False,
    stablemax=False, eval_offset=0.25, grad_clip=10.0,
    null_kv=False,   # override with --null-kv flag
    compile=False,
    name='expA_base',
    curriculum=_curriculum,
    eval_configs=_eval,
)
