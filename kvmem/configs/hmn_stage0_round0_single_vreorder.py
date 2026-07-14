"""
Reproducibility check for the vocab reorder (see kvmem/hmn.py's vocab
section docstring): chat tags now occupy IDs 256-261 (fixed, small, never
expected to grow) and HMN_STATE_0 starts at 262 (pure tail region, grows
freely with hp['V']) — this SUPERSEDES the original kvmem/data.py-ported
ordering (STATE before tags, with 6 dead legacy padding slots between them),
which is what `hmn_stage0_round0_single.py`/`hmn_flow.py` were trained
under. This is a pure token-ID relabeling — smoke-tested to produce a
byte-identical sequence LENGTH/mask shape to the pre-reorder layout
(chunk_positions_flow(4,16,8,8,[(0,2),(1,3),(2,4)]) still gives L=236) — so
this run exists ONLY to confirm the reorder doesn't change trainability or
final numbers, not because the architecture changed. Same hyperparameters
as hmn_stage0_round0_single.py in every other respect.

Trained from scratch (new vocab IDs mean the old solo/flow checkpoints'
embedding rows no longer align — this is an intentional clean break, not a
bug; see docs/HMN_RECIPE.md).

Compare final numbers directly against hmn_stage0_round0_single.py's
recorded result (val per-span MEAN=94.4%, best 97.2% at step 150000,
test=100%, loss=0.017) — a close match confirms the reorder is a genuine
no-op on trainability, not just on paper.

Run:
    python3 -m kvmem.hmn --config kvmem/configs/hmn_stage0_round0_single_vreorder.py \
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
    name='hmn_stage0_round0_single_vreorder', seed=48,

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        dict(n_chunks=4, chunk_len=16, n_refine=0, B=6, n_steps=160000, eval_every=10000,
             chain_steps=[(0, 2)]),
    ],
)
