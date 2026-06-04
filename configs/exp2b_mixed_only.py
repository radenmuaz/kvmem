"""
Exp 2b — Mixed routing only, trained from scratch.
No pre-training on individual directions — model must learn routing cold.

DSL: <x:16><z:7><h:1><q:4><y:8> | n2/r[0,1]/80k, n2/r[0,1]/160k/w1
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
    drop_close_prob=0.5, seed=42,
    ocd=False, ocd_prob=0.01, tf_warmup=0,
    rope=True, yarn=True, grok=False,
    stablemax=False, eval_offset=0.25, grad_clip=10.0,
    compile=False,
    name='exp2b_mixed_only',
    curriculum=_curriculum,
    eval_configs=_eval_cfgs,
)
