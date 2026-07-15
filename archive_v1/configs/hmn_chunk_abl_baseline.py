"""
Architecture ablation, baseline: current model exactly as-is (no new flags).

Fixed exp constant across all hmn_chunk_abl_*.py configs (deliberately the
*unstable* setup): single-stage IR, n_chunks=2, chunk_len=16 (32B src),
windows=[(0,2)], n_refine=2, RANDOM INIT — no IQ pretraining stage first.
This is the exact setup that diverged in the first hmn_chunk_local_32
attempt (eval BPB climbing 10->19 while train loss kept falling), so it's
a sensitive testbed for comparing which architecture tweaks improve
convergence/stability, not just final accuracy.

Compare against: hmn_chunk_abl_zeroinit.py, _qknorm.py, _gatedffn.py,
_rmsnorm.py, _embedscale.py, _warmup.py — each flips exactly ONE flag
(or, for _warmup, one hp value) relative to this baseline.

Run:
    python -m kvmem.train_hmn_chunk \
        --config configs/hmn_chunk_abl_baseline.py --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_abl_baseline', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=2, chunk_len=16, B=8, n_steps=20000, eval_every=5000,
             traj_mix=[dict(type='ir_local', weight=1.0, windows=[(0, 2)], n_refine=0)],
             eval_traj='ir_local'),
        dict(n_chunks=2, chunk_len=16, B=8, n_steps=30000, eval_every=10000,
             traj_mix=[dict(type='ir_local', weight=1.0, windows=[(0, 2)], n_refine=2)],
             eval_traj='ir_local'),
    ],
)
