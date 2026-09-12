"""
`hmn_stage0_search.py` — random hparam search harness for stage 0 only
(chunk_len=8, n_chunks=1 — `hmn_tpu_recall1024_jax_hopdrop.py`'s own
foundation shape, tiny L~20-25, fastest to iterate on), searching for the
config that maximizes BOTH loss-reduction-per-step AND TPU utilization
(duty_cycle_pct, via the new `_log_tpu_util` instrumentation in
`train_jax`) — not run directly; imported by a driver script that overrides
`hp['curriculum'][0]['B']`/`hp['lr_max']` per trial and calls `train_jax`
in a loop, since B directly changes the batch dimension (and therefore the
compiled shape / TPU matmul utilization) while lr_max changes optimization
quality independent of hardware efficiency — both matter for "loss
reduction per step" and only B meaningfully affects "TPU usage."

Base hp mirrors `hmn_tpu_recall1024_jax_hopdrop.py`'s own stage 0 exactly
(same architecture, same `_intra_chunk_grid` anchor sweep) so search
results transfer directly to that config.
"""
from kvmem.hmn_jax import load_config


def _intra_chunk_grid(chunk_len, warmup_lens, n_anchors, min_recall_len):
    mix = []
    for wl in warmup_lens:
        max_start = chunk_len - min_recall_len - wl
        if max_start < 0:
            continue
        starts = (sorted(set(round(i * max_start / (n_anchors - 1)) for i in range(n_anchors)))
                  if max_start > 0 else [0])
        for s in starts:
            mix.append(dict(weight=1.0, dsl=f'E({chunk_len}) Q(0,1,{s},{wl})'))
    return mix


def base_hp():
    hp = load_config('kvmem/configs/hmn_tpu_recall1024_jax_hopdrop.py')
    hp['curriculum'] = [
        dict(n_chunks=1, chunk_len=8, B=16, n_steps=3000, eval_every=3000,
             hops=-1, hop_drop_prob=0.0,
             weave_mix=_intra_chunk_grid(8, [2, 3, 4], n_anchors=4, min_recall_len=4)),
    ]
    return hp
