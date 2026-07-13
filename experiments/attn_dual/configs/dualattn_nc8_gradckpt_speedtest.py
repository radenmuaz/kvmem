"""
Short, early-stop speed comparison: grad_checkpoint off vs 'block' vs 'attn',
same base config as dualattn_nc8_slot8_ir.py (nc=8, L=1694, chunk_attn=256,
rmsnorm=True), to answer empirically whether checkpointing helps or hurts
throughput on this machine (MPS) — see docs/SRS_RECIPE.md's grad-checkpointing
discussion: predicted SLOWER in isolation (checkpointing always adds recompute
cost; only pays off indirectly via a bigger batch size, and MPS has much less
batch-size headroom than a big CUDA GPU) — this run measures it directly
instead of relying on that prediction.

NOT a real training run — no eval (eval_every set beyond n_steps), short
(2000 steps, enough for tqdm's it/s to stabilize past warmup) purely to read
off steady-state throughput. Run each of the 3 modes back to back (never two
jobs at once), compare it/s directly from each run's tqdm output.

Run (three separate invocations, one per grad_checkpoint value — edit the
`grad_checkpoint` line below between runs, or pass via a wrapper script):
    caffeinate -i python3 -m experiments.attn_dual.train \\
        --config experiments/attn_dual/configs/dualattn_nc8_gradckpt_speedtest.py \\
        --device mps
    tail -f experiments/attn_dual/logs/dualattn_nc8_gradckpt_speedtest/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, V=290,
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=200, log_every=200,
    lr_schedule='cosine_restarts',
    cosine_T0=2000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True,
    rmsnorm=True, chunk_attn=256,
    grad_checkpoint=None,  # edit to 'block' or 'attn' for the other two runs
    name='dualattn_nc8_gradckpt_speedtest_off', seed=48,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=2.0,
    val_n_seqs=1,

    curriculum=[
        dict(n_chunks=8, chunk_len=16, n_refine=2, B=3, n_steps=2000, eval_every=999999,
             windows=[(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8)]),
    ],
)
