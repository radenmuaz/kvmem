"""
`hmn_recall_queue.py` — training config for Stage `hop` (renamed from
`hmn_flow.py`; the underlying mechanism is still called `hop` throughout
the codebase — `chunk_positions_hop`/`chunk_mask_fb_hop`/`hops` —
this filename just describes the TASK/schedule, matching
`hmn_single_recall.py`'s naming convention, not the architecture).

Attention-based relay, single-hop STATE-to-STATE permission
(`chunk_mask_fb_hop`'s relay exception) instead of a forced feature-vector
copy. Chain step i's own round-0 STATE row gets a narrow attention
permission to read chain step i-1's STATE directly — the model LEARNS what
to preserve via ordinary full-gradient backprop, no forced copy, no
`.detach()`-truncated BPTT. Resolved entirely by mask permissions within
ONE packed-sequence forward pass — no sequential per-chain-step
orchestration needed (cheaper per step than the original `h_inject`-based
relay design, since deleted — see CLAUDE.md's "Deleted mechanisms").

Warm-started from `solo`'s checkpoint (same d/n_layers/vocab, so this
should load byte-identical, no growing tensors).

**Known result, measured not assumed**: the run recorded under this exact
config (same architecture/hyperparameters/warm-start-from-solo) previously
converged well — val/test STITCHED=88.1%/85.7%, loss=0.603, massively
outperforming the deleted `h_inject`-based relay design on every metric
(chain step 2 test 70.8% vs. that design's 12.5%). That original checkpoint
was deleted along with other old-vocab artifacts. A SUBSEQUENT run under
this same config (during the vocab-reorder reproducibility check) converged
NOTABLY WORSE — val/test STITCHED=71.4%/71.4%, loss plateaued flat around
1.84-1.86 from step 50000 onward, never breaking out like the first run
did. The mask/relay mechanism itself was independently verified correct in
both cases (byte-identical mask regardless of vocab ID relabeling,
confirmed via direct comparison) — the discrepancy is attributed to
warm-start sensitivity (two `solo` checkpoints can both hit ~100% on
solo's own near-trivial task while differing enough in underlying weight
configuration to matter for `hop`'s harder relay-learning objective), not
a code defect. Re-running this config is not guaranteed to reproduce either
prior result exactly — treat any single run's outcome as one sample from a
distribution with real spread, not a deterministic number.

`hops=1` (explicit): the relay exception's lookback window defaults to
0 (no relay at all — opt-in required, see chunk_mask_fb_hop's docstring)
as of this session's `hops` generalization, so this MUST be set
explicitly to reproduce the single-hop behavior every result above was
measured against. hops>1 (attend back N chain steps, not just 1) is
now buildable but untested — a natural next experiment given the
recovery-probe's clean repeat_query failure, orthogonal to `weave_mix`
(which tests trajectory-shape generalization, not relay depth).

Run:
    python3 -m kvmem.hmn --config kvmem/configs/hmn_recall_queue.py \
        --pretrained kvmem/logs/hmn_single_recall/checkpoints/stage0_best.pt \
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
    name='hmn_recall_queue', seed=49,

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        dict(n_chunks=4, chunk_len=16, n_refine=0, B=6, n_steps=160000, eval_every=10000,
             chain_steps=[(0, 2), (1, 3), (2, 4)], hops=1),
    ],
)
