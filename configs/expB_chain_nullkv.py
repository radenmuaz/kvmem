"""
Exp B + null_kv — 1-2-3 step chain with zero KV, eval extrapolation to 4-5 steps.
null_kv=True: 1.5-2x faster convergence, better calibration.
mmix mode: random end/int per step, random query count, any-prior-block recall.
"""

from kvmem.curriculum_dsl import parse_curriculum

_DSL = (
    "<x:16><z:7><h:1><q:4><y:8> | "
    "n1/r0/20k/mmix, n2/r[0,1]/30k/mmix, n3/r[0,1,2]/40k/mmix "
    "@eval:n1/r0,n2/r0,n2/r1,n3/r0,n3/r2,n4/r0,n4/r3,n5/r0,n5/r4"
)

_seq, _curriculum, _eval = parse_curriculum(
    _DSL,
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
    null_kv=True,
    compile=False,
    name='expB_chain_nullkv',
    curriculum=_curriculum,
    eval_configs=_eval,
)
