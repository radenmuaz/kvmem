"""
Stage 0 / `solo` — single chain-step, round 0 only, no chain (per
design-experiment-which-use-atomic-kay.md's "New staging" section). Establishes
basic state-compression on the current vocab before adding any accumulation
(relay) complexity — mirrors this project's own proven "simplest case first"
bootstrap principle. No relay region exists in this stage's layout at all
(nothing to relay from with only one chain step).

single_attn block type (this project's default going forward), n_layers=8
(double the paired dual_attn n_layers=4, matching total attention-op count —
see hmn_dualattn_nc4_iq.py / hmn_singleattn_nc4_iq.py precedent in this
scratchpad). Scale matches this project's proven nc4/chunk_len=16 primitive
(one chain step spanning 2 chunks = 32B recall unit).

Vocab: chat tags occupy IDs 256-261 (fixed, small, never expected to grow),
STATE occupies the tail starting at 262 (pure append-growth region) — see
kvmem/hmn.py's vocab section docstring for the full layout and rationale.
V=274 (256 data bytes + 6 chat tags + 12 reserved STATE ids).

Sanity bar (design plan section 5): high single-chain-step round-0 match% —
compare loosely against this project's historical single-window IQ baselines
(e.g. ~100% dual-attn single-window IQ result) as a "did this at least learn
the basic task" floor, not a strict target.

Run:
    python3 -m kvmem.hmn --config kvmem/configs/hmn_single_recall.py \
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
    name='hmn_single_recall', seed=48,

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
