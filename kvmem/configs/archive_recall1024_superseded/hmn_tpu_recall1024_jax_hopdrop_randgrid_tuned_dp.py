"""4-chip data-parallel variant of `hmn_tpu_recall1024_jax_hopdrop_randgrid_tuned.py`
-- identical hparams/curriculum (B=128/lr=4e-4 at stage0, same randomized-anchor
schedule), only `hp['data_parallel']=True` added. Launched for a direct
apples-to-apples throughput comparison against that config's own measured
single-chip baseline (~160 steps/sec at stage0 on a v6e-4 node's chip 0)."""
from kvmem.hmn_jax import load_config

hp = load_config('kvmem/configs/hmn_tpu_recall1024_jax_hopdrop_randgrid_tuned.py')
hp['name'] = 'hmn_tpu_recall1024_jax_hopdrop_randgrid_tuned_dp'
hp['data_parallel'] = True
