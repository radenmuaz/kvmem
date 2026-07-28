"""
`hmn_single_recall_c64_relpos.py` — redo of `hmn_single_recall_c64.py` FROM
SCRATCH under `kvmem/hmn_relpos.py`'s alternative positional mechanism:
`rope=False`, `relpos_enabled=True` — NO RoPE at all, replaced with a
QUERY-SIDE, content-dependent learned bias (`Linear(d, relpos_k*n_heads)`
applied to the query's own hidden state) added directly into the SDPA
attn_mask at exactly the "d steps back" relative positions (d=1..relpos_k,
default 2). No other relative distance gets any signal at all — the
minimal possible positional mechanism, motivated by the same shortcut
`kvmem/probe_positional_shortcut.py` measured (a query slot's position,
not its content, determined which STATE got recalled) but attacking it
from the opposite direction of `hmn_single_recall_c64_dualrope.py`'s
dual-clock RoPE: instead of scoping WHERE the distance signal applies
(macro vs local clock), this removes essentially ALL distance signal
except the one relation genuinely needed for coherent local byte
generation. Originally a fixed learned constant per (head, distance) —
Shaw-et-al.-style relative position embeddings — replaced by this
content-dependent version (renamed accordingly, `relpos_shaw` ->
`relpos_enabled`) chosen specifically for KV-cache friendliness: query-side
needs zero extra cached state, since the current query's hidden state is
already being freshly computed every decode step regardless of caching.

Verified before trusting (see kvmem/hmn_relpos.py's own additions): a
targeted numeric check with the bias manually cranked way up showed
attention collapsing to ~100% on the targeted "d steps back" column for an
interior row (both the original constant version and the k>1 window
generalization), and a separate check confirmed the query-side gate
produces genuinely different bias values across different training
examples in the same batch (content-dependent, not a shared constant).

Not warm-started from anything (same reasoning as the RoPE original and
`_dualrope` sibling — a from-scratch positional mechanism can't inherit
weights trained under a different one). This config: single chunk,
`E1 Q(0,1)`, establishing a base checkpoint under the new mechanism before
testing the actual multi-query case in `hmn_weave_c64_relpos.py`.

Run (never two jobs at once):
    python3 -m kvmem.hmn_relpos --config kvmem/configs/hmn_single_recall_c64_relpos.py --device mps
"""

hp = dict(
    d=64, n_layers=8, n_heads=4, V=274,
    block_type='single_attn',
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=100000, cosine_T_mult=1,
    rope=False, null_kv=True,
    relpos_enabled=True,
    rmsnorm=True,
    name='hmn_single_recall_c64_relpos', seed=48,

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=1, chunk_len=64, B=6, n_steps=100000, eval_every=10000,
             weave_mix=[dict(weight=1.0, dsl='E1 Q(0,1)')]),
    ],
)
