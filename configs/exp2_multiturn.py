"""
Experiment 2 — Multi-turn recall with growing routing curriculum.

Curriculum uses the DSL: seq spec once, then stage tokens nN/rK/Xk[/wM].

Growing routing tree:
  s0: 1-block baseline
  s1: 2-block recent (introduce)
  s2: 2-block old (introduce)
  s3: 2-block mixed [0,1] — true per-example routing, full history
  s4: 2-block mixed [0,1] — isolated blocks (mem_window=1)

Run:
    python -m kvmem.train --config configs/exp2_multiturn.py --device mps
"""

from kvmem.curriculum_dsl import parse_curriculum

_DSL = "<x:16><z:7><h:1><q:4><y:8> | n1/r0/40k, n2/r1/40k, n2/r0/40k, n2/r[0,1]/80k, n2/r[0,1]/80k/w1"

_seq, _curriculum, _eval_cfgs = parse_curriculum(
    _DSL,
    B=16, dataset_size=20000,
    cycle_steps=-1,   # cosine LR per stage
)

hp = dict(
    # Model
    d=64, n_layers=4, n_heads=4, d_ff=256,
    # Training (shared)
    lr_max=3e-4, wd=0.001, warmup_steps=1000,
    eval_every=5000, log_every=1000,
    ocd=False, ocd_prob=0.01, tf_warmup=0,
    rope=True, yarn=True, grok=False,
    stablemax=False, eval_offset=0.25, grad_clip=10.0,
    compile=False,
    name='exp2_multiturn',
    # Curriculum from DSL
    curriculum=_curriculum,
    eval_configs=_eval_cfgs,
)
