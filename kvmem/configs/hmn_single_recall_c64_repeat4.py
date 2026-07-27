"""
Ablation of `hmn_single_recall_c64.py` — identical in every way (same
architecture, seed, curriculum, 100000 steps) except `repeat_batch=4`: each
sampled batch gets 4 consecutive gradient steps before a fresh batch is
drawn, instead of a fresh batch every step. Same total optimizer-step count
and LR schedule as the baseline, so `loss`/`eval_mean` vs. step is directly
comparable between `logs/hmn_single_recall_c64/train.log` (baseline) and
`logs/hmn_single_recall_c64_repeat4/train.log` (this config) — the only
variable under test is whether repeating a batch a few times before
resampling speeds up or slows down convergence.

Baseline (`hmn_single_recall_c64`) convergence, for comparison: first hit
val eval_mean=100.0% at step 60000 (loss=0.0266), and held 100.0% at every
subsequent eval (70000/80000/90000/100000) — no dips, converged cleanly by
step 60000 with the remaining 40000 steps a pure plateau.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_single_recall_c64_repeat4.py --device mps
"""

hp = dict(
    d=64, n_layers=8, n_heads=4, V=274,
    block_type='single_attn',
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=100000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True,
    rmsnorm=True,
    name='hmn_single_recall_c64_repeat4', seed=48,
    repeat_batch=4,

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=1, chunk_len=64, n_refine=0, B=6, n_steps=100000, eval_every=10000,
             chain_steps=[(0, 1)]),
    ],
)
