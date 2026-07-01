"""
Architecture ablation: gated_ffn=True (SwiGLU) vs baseline.
See hmn_chunk_abl_baseline.py for the fixed exp constant + rationale.

gated_ffn: SwiGLU (silu(W1 x) * W3 x -> W2) instead of plain GELU-MLP.
Note: at the same d_ff this adds ~50% more FFN params (extra W3) — not
param-matched against baseline, purely testing convergence-per-step.

Run:
    python -m kvmem.train_hmn_chunk \
        --config configs/hmn_chunk_abl_gatedffn.py --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_abl_gatedffn', seed=42,

    gated_ffn=True,

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
