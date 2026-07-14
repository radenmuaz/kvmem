"""
Stage 1 — multiple chain steps, round 0 only, WITH chain (STATE_QUEUE),
warm-started from Stage 0 (per design-experiment-which-use-atomic-kay.md's
"New staging" section). The actual chain-memory test: each chain step's
round-0 STATE computation reads the previous chain step's injected
STATE_QUEUE_in (h_inject, see train()'s `stage.get('chain')`-dispatched
sequential path). Still n_refine=0 (IR rounds + chain is explicitly out of
scope for this immediate retrain, deferred per the plan).

chain_steps=[(0,2),(1,3),(2,4)] matches the historical 3-window stitch
schedule at n_chunks=4, chunk_len=16 (50% overlap, 32B recall unit per chain
step). chain=True is required for this schedule since chain steps 1 and 2
each have a STATE_QUEUE_in region (chain step 0 does not — nothing to chain
from yet).

Warm-started from Stage 0's checkpoint via _pretrained_ckpt (set by
--pretrained on the CLI, per train()'s existing partial-load-by-shape-prefix
logic — Stage 0's vocab/dims are identical here so this should load byte-
identical, no growing tensors expected).

Validation (design plan section 5, part 2 — the actual chain-memory test, not
just per-chain-step accuracy): per-chain-step recall alone doesn't prove
STATE_QUEUE carried anything forward (each chain step can solve its own span
locally regardless of chaining, since direct cross-chain-step attention stays
blocked per Rule 3b). The real test is recovering an EARLIER chain step's span
from the LAST chain step's round-0 recall — only reachable via the
accumulated STATE_QUEUE chain. Not yet automated in train()'s eval loop as of
this config; run as a follow-up probe once Stage 1 converges.

Run:
    python3 -m kvmem.hmn --config kvmem/configs/hmn_stage1_round0_chained.py \
        --pretrained kvmem/logs/hmn_stage0_round0_single/checkpoints/stage0_best.pt \
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
    name='hmn_stage1_round0_chained', seed=49,

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        dict(n_chunks=4, chunk_len=16, n_refine=0, B=6, n_steps=160000, eval_every=10000,
             chain_steps=[(0, 2), (1, 3), (2, 4)], chain=True),
    ],
)
