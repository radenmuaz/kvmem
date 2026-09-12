"""
`hmn_tpu_recall1024_jax_hopdrop_randgrid_tuned.py` — applies the winning
hparams from the stage0 random search (2026-08-01, `kvmem/configs/
hmn_stage0_search.py` + the scratchpad `random_search_stage0.py` driver, 6
trials of B x lr_max on the same tiny chunk_len=8/n_chunks=1 shape):
B=128, lr_max=4e-4 won clearly on loss-reduction-per-step (final loss 0.11
vs the next-best 0.55, and vs B=16/lr=2e-4 which actually DIVERGED to
loss=6.09). Notably B=256 at the SAME lr=4e-4 did markedly worse than
B=128 (final loss 1.43 vs 0.11) -- bigger batch without a correspondingly
higher lr is not free, it just means fewer effective updates for the same
step budget, so B=128 is applied here, not a larger value.

Scope of what's actually verified vs extrapolated: the search only tested
stage0's own shape (L~20-25). Applying B=128 to stage 0 here is a direct,
verified transfer. lr_max=4e-4 is applied GLOBALLY (this codebase's
`_make_schedule` uses one hp['lr_max'] across every stage) since lr was
the more robust/general finding of the two -- but this IS an extrapolation
beyond what was tested, not re-verified at the later, much longer-L
stages. B for stages 1-5 is left UNCHANGED from `_randgrid.py` (16,16,16,
8,8) -- NOT blindly scaled up to 128, since those shapes were never
memory-tested at that batch size (dense O(L^2) attention at L~1300-2200
would plausibly OOM a single chip at B=128) and scaling batch size without
re-verifying memory headroom would risk crashing mid-run rather than a
clean, defensible extrapolation.

Otherwise identical to `hmn_tpu_recall1024_jax_hopdrop_randgrid.py` (same
randomized-anchor curriculum, same enc_hops/hop_drop_prob schedule).
"""
from kvmem.hmn_jax import load_config

hp = load_config('kvmem/configs/hmn_tpu_recall1024_jax_hopdrop_randgrid.py')
hp['name'] = 'hmn_tpu_recall1024_jax_hopdrop_randgrid_tuned'
hp['lr_max'] = 4e-4
hp['curriculum'][0]['B'] = 128
