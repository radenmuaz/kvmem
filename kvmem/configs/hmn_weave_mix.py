"""
Stage `weave` — training run, using the `weave_mix` dispatch built this
session (`kvmem/hmn.py`'s `if 'weave_mix' in stage:` branch, `stage['weave_mix']`
key). Trains on a UNIFORM mix of the three trajectory shapes flagged as
train-mix candidates in docs/HISTORY.md §4c:
  - `batch`              — encode everything, then query in schedule order
                            (the exact shape `hop` was trained on).
  - `stream`              — interleaved encode/query (encode 2 chunks, query,
                            encode 1 more, query, ...).
  - `interleave_delayed` — encode everything, then query in REVERSED order.
`repeat_query`/`long_hop_recovery`/`decay_curve` are deliberately excluded —
they're held-out generalization probes (`kvmem/eval_weave.py`), and
`weave_mix`'s dispatch code actively rejects them with an AssertionError if
passed here, not just by omission.

Why this matters, concretely: `hop`'s chain-memory recovery probe
(`eval_weave.py --patterns repeat_query`, run against `hop`'s finished
checkpoint) FAILED CLEANLY — first occurrence of a re-queried span scored
100%, the repeated occurrence (reachable only through the accumulated
relay chain, per the single-hop mask rule) scored 0.0% across all 3 test
sequences. The leading candidate explanation (see CLAUDE.md's `hop`
results section) isn't that the relay fails to preserve information across
hops — it's that `hop` was NEVER TRAINED on any trajectory shape besides
its own fixed 3-query schedule, so it has no exposure to handling a
repeated/reordered query at all, independent of what its STATE actually
contains. `weave_mix` training is the direct test of that hypothesis: if
generalization was the missing ingredient, training on VARIED orderings
(without ever training on `repeat_query` itself, which stays held out)
should improve `repeat_query`/`long_hop_recovery` zero-shot performance
after this run, without needing to touch the underlying relay mechanism.

**Warm-started from `hmn_single_recall_c64`'s checkpoint, NOT `hop`'s** —
`hmn_routing_4to1_state` (`solo`) is treated as an archived experiment (no
checkpoint for it exists on disk) and is no longer the warm-start source
for this track; `hmn_single_recall_c64` is a same-architecture
(d/n_layers/n_heads/state_len/V unchanged), single-chunk/no-relay
checkpoint that already exists and reached 100% val match, so it plays
the same "clean, unambiguous base to build from" role `solo` used to.
`hmn_recall_queue`'s (`hop`'s) own checkpoint is NOT used either — it
converged NOTABLY WORSE than the run its own recovery-probe result (0.0%
on `repeat_query`, cited above) was measured against — val/test
STITCHED=71.4%/71.4%, loss=1.851, vs. the original measurement's
88.1%/85.7%, loss=0.603 (see `hmn_recall_queue.py`'s docstring and
CLAUDE.md's reproducibility-check section) — warm-starting from that
weaker checkpoint would make this stage's results hard to interpret
cleanly. Worth keeping in mind when interpreting results: a
`repeat_query`/`long_hop_recovery` failure here could mean either
"generalization training didn't fix it" (the original hypothesis) or
"the relay itself never got learned well enough in the first place" —
`hop`'s own per-pattern val numbers during this run (`batch` specifically,
byte-shape-identical to what `hop` trained on) are the way to tell those
apart: if `batch` converges to `hop`-like accuracy but `repeat_query`
still fails, that isolates the generalization-gap hypothesis cleanly.

n_chunks=4, chunk_len=16, window_chunks=2 match `hop`'s own convention
(same 32-byte, 2-chunk query span) so warm-started weights transfer
cleanly and `batch` here is byte-shape-identical to what `hop` trained on.

Running now (launched ahead of the `squeeze` pair in the original queue
order, per explicit instruction — `squeeze` was paused mid-run at the time).
Once done, re-run `eval_weave.py --patterns repeat_query,long_hop_recovery,decay_curve`
against this checkpoint and compare directly against `hop`'s 0.0% result.

Pretrained checkpoint is set as a DEFAULT directly in this config's `hp`
dict (`_pretrained_ckpt`, read by `train()` at `kvmem/hmn.py:2294` if
present) rather than requiring `--pretrained` on every invocation — the
CLI flag (`kvmem/hmn.py`'s `main()`) still overrides it if passed, so this
doesn't remove the option to warm-start from something else ad hoc.

Run:
    python3 -m kvmem.hmn --config kvmem/configs/hmn_weave_mix.py --device mps
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
    name='hmn_weave_mix', seed=50,
    _pretrained_ckpt='logs/hmn_single_recall_c64/checkpoints/stage0_best.pt',

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=4, chunk_len=16, window_chunks=2, B=6, n_steps=160000, eval_every=10000,
             weave_mix=[
                 dict(weight=1.0, pattern='batch'),
                 dict(weight=1.0, pattern='stream'),
                 dict(weight=1.0, pattern='interleave_delayed'),
             ]),
    ],
)
