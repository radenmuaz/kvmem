"""
`hmn_recall1024_jax_oneshotrecall_b128.py` — smaller-batch retry after
`hmn_recall1024_jax_oneshotrecall_b256.py` died mid-compile on tpu7 (stage
4's 32-distinct-shape compile at B=256, only 6/32 shapes compiled before
the process silently vanished — no traceback, no reboot, plausibly a
compile-time host-memory issue given the large number of simultaneous
shapes). Batch size halved from that config (Phase A 256->128, Phase B/C
128->64) to reduce compile-time memory pressure while `hmn_recall1024_jax_
oneshotrecall_b512.py` continues at the higher end on tpu8.

Same warm-start checkpoint (`hmn_oneshotrecall_jax_b256`'s stage3_best.pt,
val_mean=83.8%), same curriculum shape, same state_len=8/state_vocab_size=2/
MLP-disabled ablation — only the batch size differs.

Run (run from the repo root so relative log/config paths resolve):
    python3 -m kvmem_jax.hmn_jax --config kvmem_jax/configs/hmn_recall1024_jax_oneshotrecall_b128.py
"""

from kvmem_jax.hmn_jax import load_config

hp = load_config('kvmem_jax/configs/hmn_recall1024_jax_oneshotrecall_b256.py')
hp['name'] = 'hmn_recall1024_jax_oneshotrecall_b128'
for stage in hp['curriculum'][:4]:
    stage['B'] = 128
for stage in hp['curriculum'][4:]:
    stage['B'] = 64
