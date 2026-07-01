"""
Architecture/schedule ablation: longer warmup_steps vs baseline.
See hmn_chunk_abl_baseline.py for the fixed exp constant + rationale.

No model code change — pure training-schedule hp. Baseline uses a fixed
warmup_steps=500 regardless of stage length (only 6.25% of this 8000-step
run, but a much smaller fraction of the real 50k/80k stage lengths used in
hmn_chunk_local_32.py). Here: warmup_steps=800 (10% of n_steps), testing
whether a longer LR ramp alone reduces the early-training instability.

Run:
    python -m kvmem.train_hmn_chunk \
        --config configs/hmn_chunk_abl_warmup.py --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=800,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_abl_warmup', seed=42,

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
