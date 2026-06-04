"""
Exp 2b (null_kv variant) — Mixed routing from scratch with null KV.
Run this if exp2b_mixed_only fails to generalise to both routing directions.
null_kv=True gives each attention head a fixed (K=0,V=0) abstain option.
"""

from kvmem.curriculum_dsl import parse_curriculum

_DSL = "<x:16><z:7><h:1><q:4><y:8> | n2/r[0,1]/160k, n2/r[0,1]/160k/w1"

_seq, _curriculum, _eval_cfgs = parse_curriculum(
    _DSL,
    B=16, dataset_size=20000,
    cycle_steps=-1,
)

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256,
    lr_max=3e-4, wd=0.001, warmup_steps=1000,
    eval_every=5000, log_every=1000,
    ocd=False, ocd_prob=0.01, tf_warmup=0,
    rope=True, yarn=True, grok=False,
    stablemax=False, eval_offset=0.25, grad_clip=10.0,
    null_kv=True,   # zero KV append — abstain option in softmax
    compile=False,
    name='exp2b_mixed_nullkv',
    curriculum=_curriculum,
    eval_configs=_eval_cfgs,
)
