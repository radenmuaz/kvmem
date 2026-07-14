"""
Reproducibility check for the vocab reorder — flow half of the pair (see
hmn_stage0_round0_single_vreorder.py for the vocab-reorder rationale, and
kvmem/hmn.py's vocab section docstring for the mechanics). Identical
hyperparameters to hmn_flow.py in every respect except warm-starting from
`hmn_stage0_round0_single_vreorder`'s checkpoint instead of the original
`hmn_stage0_round0_single` one (new vocab IDs mean the two checkpoint
families are not cross-compatible — an intentional clean break).

Compare final numbers directly against hmn_flow.py's recorded run (see
CLAUDE.md's flow progress table) — a close match at equivalent checkpoints
confirms the reorder is a genuine no-op on trainability/final results.

Run (only after hmn_stage0_round0_single_vreorder.py finishes — never two
jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_flow_vreorder.py \
        --pretrained kvmem/logs/hmn_stage0_round0_single_vreorder/checkpoints/stage0_best.pt \
        --device mps
"""

hp = dict(
    d=64, n_layers=8, n_heads=4, V=274,
    block_type='single_attn',
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=160000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True,
    rmsnorm=True,
    name='hmn_flow_vreorder', seed=49,

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        dict(n_chunks=4, chunk_len=16, n_refine=0, B=6, n_steps=160000, eval_every=10000,
             chain_steps=[(0, 2), (1, 3), (2, 4)], flow=True),
    ],
)
