"""
Stage `flow` — alternative to Stage `relay`'s STATE_QUEUE/h_inject relay.
Identical hyperparameters to relay (kvmem/configs/hmn_stage1_round0_chained.py)
in every respect except the cross-chain-step channel itself, so the two are
a direct, apples-to-apples comparison of LEARNING MECHANISM, not information
budget (both are single-hop, both carry exactly one state_len-wide vector
forward).

relay: h_inject forcibly COPIES chain step i-1's final STATE into chain step
i's input, .detach()ed (truncated BPTT — no gradient signal that STATE must
be useful to a FUTURE chain step, only to its own recall loss). Needs one
sequential forward pass per chain step (train()'s `if is_chained:` branch).

flow: chain step i's own round-0 STATE row gets a narrow, single-hop
ATTENTION PERMISSION to read chain step i-1's STATE columns directly
(chunk_mask_fb_flow's relay exception) — the model LEARNS what to
preserve via ordinary gradient descent, full gradient flow across chain
steps, no forced copy. Resolved entirely by mask permissions within ONE
packed-sequence forward pass — no sequential orchestration needed, so this
reuses the same cheap fast path as non-chained stages (train()'s
`if not is_chained:` branch, since `flow=True` always keeps `is_chained`
False — see the stage dispatch in train()).

Exact generated layout differs from relay's only in that there's no separate
STATE_QUEUE_in region — chain step i's own STATE serves double duty as both
"this chain step's recall register" and "the thing chain step i+1 reads."
See chunk_positions_flow / chunk_mask_fb_flow docstrings for full mechanics,
and the smoke test verifying the single-hop exception (chain step 2 sees
chain step 1's STATE but NOT chain step 0's, directly) before this was
queued.

Warm-started from Stage `solo`'s checkpoint (same d/n_layers/vocab as relay,
so this should load byte-identical, no growing tensors).

Run (only after Stage `relay` finishes — never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_flow.py \
        --pretrained kvmem/logs/hmn_stage0_round0_single/checkpoints/stage0_best.pt \
        --device mps

Compare against relay's final numbers directly (same chain_steps schedule,
same step budget) — the headline result is per-chain-step match% at
convergence, plus whether chain step 2 (which needs a 2-hop path: step0 ->
step1 -> step2) shows a CLEARER improvement here than under relay, which
would be direct evidence the gradient-flow fix matters.
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
    name='hmn_flow', seed=49,

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
