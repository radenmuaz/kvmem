"""
`hmn_tpu_sanity_w25.py` — TPU port sanity check, NOT the scale-up target
(that's `hmn_tpu_recall1024_flat.py`, `L` up to 2128). This config reuses
`hmn_notags_w25.py`'s curriculum shape (single-chunk recall, chunk_len
8/16/32/64, anchor-swept warmup — `_grid` copied verbatim from that file)
because its `L` stays small (~30-170 across all 4 stages, vs. Run A's
1200-2200) — a much cheaper, much faster-to-compile way to exercise the
SAME model architecture/hyperparameters Run A uses (`d=128, n_layers=16,
n_heads=8`, `rope=False`, `state_len=4`, `state_vocab_size=1`,
`grad_checkpoint='block'`, `bucket_lengths=True`) on real TPU hardware
before committing to Run A's much slower compile.

**Batch size deliberately maximized, not tuned for a real training
schedule** — `token_budget`/`attn_sq_budget` set generously (`4_000_000`/
`500_000_000`, vs. Run A's `131_072`/`125_000_000`) and the per-stage `B`
cap raised to 4096, specifically to stress-test the TPU path (bucketing,
per-bucket B derivation, grad_checkpoint, autocast, eval-replica) at the
largest batch sizes the short-L stages here can plausibly need, not because
these are the right batch sizes for a real run. `n_steps` cut to a few
thousand per stage (vs. `hmn_notags_w25.py`'s 480000-720000) — this is a
smoke test, not a training run to convergence.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_tpu_sanity_w25.py --device tpu
"""
import math


def _grid(chunk_len, warmup_lens, n_anchors, min_recall_len, rb_token, weight=1.0, min_warmup_frac=0.0):
    """Verbatim copy of hmn_notags_w25.py's own `_grid` — see that file for
    the full docstring. Generates weave_mix entries for one length: `n_anchors`
    evenly-spaced query_start values per (chunk_len, warmup_len) pair."""
    if min_warmup_frac > 0:
        min_wl = math.ceil(chunk_len * min_warmup_frac)
        bad = [wl for wl in warmup_lens if wl < min_wl]
        assert not bad, (
            f'_grid(chunk_len={chunk_len}, ...): warmup_lens {bad} are below the '
            f'min_warmup_frac={min_warmup_frac} floor ({min_wl}) — pass an already-'
            f'filtered warmup_lens list instead of relying on a runtime filter')
    entries = []
    for wl in warmup_lens:
        max_start = chunk_len - min_recall_len - wl
        if max_start < 0:
            continue
        if n_anchors == 1 or max_start == 0:
            starts = [0]
        else:
            starts = sorted(set(round(i * max_start / (n_anchors - 1)) for i in range(n_anchors)))
        for s in starts:
            entries.append(dict(weight=weight, dsl=f'E({chunk_len}) Q(0,1,{s},{wl}) {rb_token}'))
    return entries


hp = dict(
    d=128, n_layers=16, n_heads=8, V=271,
    block_type='single_attn',
    # lr_max reverted to hmn_notags_w25.py's ORIGINAL 1e-4 (was 6e-4, carried over
    # unexamined from the large-batch Run A config's √-scaled value) — the 6e-4 run
    # dropped fast to loss~2.5 by step 1000 then plateaued/oscillated 2.2-2.9 for the
    # next 4000 steps (match=2.3% at the first eval, step 5000) instead of continuing
    # to converge. Too-high LR for this batch size (B=16, unchanged from the original
    # recipe) is the likely cause; 1e-4 is the value that recipe actually converged
    # under (see CLAUDE.md's chunk_len ladder results).
    lr_max=1e-4, wd=1e-5,
    warmup_steps=1000, log_every=200,
    rope=False,
    null_kv=True,
    rmsnorm=True,
    # grad_checkpoint='block' RE-ENABLED — the earlier NaN (loss=NaN from step 1 with
    # 'block', clean with False, see this file's own git history / CLAUDE.md's TPU
    # port entry) was isolated to torch_xla.utils.checkpoint not reapplying bf16
    # autocast during backward recompute, now fixed in kvmem/hmn.py's `_ckpt` (routes
    # XLA tensors through stock PyTorch's reentrant checkpoint instead, which does
    # reapply autocast correctly). This run verifies that fix directly.
    grad_checkpoint='block',
    name='hmn_tpu_sanity_w25_fp32', seed=48,
    # DIAGNOSTIC: testing whether bf16 autocast (default) is specifically slowing
    # NoPE's convergence — the bf16 NoPE run (hmn_tpu_sanity_w25.py) showed match%
    # 21.7% (step 5000) -> 17.4% (step 10000), a wobble/slowdown that COULD be bf16
    # precision hurting the causal-depth counting mechanism NoPE relies on to
    # address STATE slots (see CLAUDE.md's state_vocab_size=1 rationale — no
    # per-slot token signal, position recoverable only through exact depth
    # counting, plausibly more precision-sensitive than RoPE's smooth continuous
    # rotations). no_autocast=True forces fp32 for a direct comparison, same seed/
    # curriculum/lr otherwise.
    no_autocast=True,

    adaptive=True,
    adapt_signal='val_match',
    adapt_temp=1.0,
    adapt_ema_alpha=0.5,
    adapt_floor=0.05,

    state_len=4, state_vocab_size=1,  # matches hmn_tpu_recall1024_flat.py's own
                                       # settings — this config sanity-checks THAT
                                       # architecture, not hmn_notags_w25's original
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    # TPU/XLA support — see kvmem/hmn.py's _bucket_ceilings/_pad_mask_to and
    # hmn_tpu_recall1024_flat.py's own docstring for the mechanism. Budgets set
    # generously here (see module docstring) to maximize batch size / stress-test
    # the port, not tuned for a real training schedule.
    bucket_lengths=True,
    max_shape_buckets=8,
    token_budget=4_000_000,
    attn_sq_budget=500_000_000,

    # B values matched to hmn_notags_w25.py's ORIGINAL small batch sizes (16/12/6/4),
    # not this file's earlier stress-test B=4096 — the goal here shifted from "exercise
    # the TPU port at max batch" (already proven) to "does the port actually converge to
    # a real match%, matching the recipe's own tuned hyperparameters, on tpu2/v6e."
    # n_steps bumped from the earlier 3000/stage smoke-test budget to a real (if still
    # short of the original 480k-720k) convergence attempt, feasible given v6e's speed.
    curriculum=[
        dict(n_chunks=1, chunk_len=8, B=16, n_steps=30000, eval_every=5000,
             weave_mix=_grid(8, [2, 3, 4], n_anchors=4, min_recall_len=4, rb_token='B8',
                             min_warmup_frac=0.25)),

        dict(n_chunks=1, chunk_len=16, B=12, n_steps=40000, eval_every=5000,
             weave_mix=(
                 _grid(16, [4, 6, 8], n_anchors=4, min_recall_len=4, rb_token='B8',
                      min_warmup_frac=0.25)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5,
                        min_warmup_frac=0.25)
             )),

        dict(n_chunks=1, chunk_len=32, B=6, n_steps=30000, eval_every=5000,
             weave_mix=(
                 _grid(32, [8, 12, 16], n_anchors=4, min_recall_len=4, rb_token='B16',
                      min_warmup_frac=0.25)
                 + _grid(16, [4, 6, 8], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5,
                        min_warmup_frac=0.25)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5,
                        min_warmup_frac=0.25)
             )),

        dict(n_chunks=1, chunk_len=64, B=4, n_steps=40000, eval_every=5000,
             weave_mix=(
                 _grid(64, [16, 24], n_anchors=4, min_recall_len=4, rb_token='B16',
                      min_warmup_frac=0.25)
                 + _grid(32, [8, 12, 16], n_anchors=2, min_recall_len=4, rb_token='B16', weight=0.5,
                        min_warmup_frac=0.25)
                 + _grid(16, [4, 6, 8], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5,
                        min_warmup_frac=0.25)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5,
                        min_warmup_frac=0.25)
             )),
    ],
)
