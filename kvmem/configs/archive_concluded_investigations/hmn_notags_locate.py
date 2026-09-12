"""
`hmn_notags_locate.py` — clone of `hmn_locate_nope_curriculum_dense.py`.
Same architecture, same curriculum, same adaptive/early_stop machinery,
same DSL strings as the dense NoPE curriculum config. Runs against
`kvmem.hmn`, which is now (post promotion, see CLAUDE.md/docs/HISTORY.md
§15) the chat-tag-free, opcode+shared-STATE-alphabet design natively —
`V=271` (256 bytes + 3 opcodes + 12 reserved shared STATE values), no
separate module needed:
  - No `HMN_SRC_OPEN/CLOSE`/`HMN_QUERY_OPEN/CLOSE`/`HMN_RESPONSE_OPEN/CLOSE`
    chat tags are ever emitted — E/S/Q/R region boundaries are inferred from
    content-type (byte vs. STATE-ID) and position alone, not explicit
    delimiters.
  - NLL loss covers the warmup region too (`w0:c1`), not just the response.

Question this asks: was the NoPE curriculum's success (stage0 early-stopped
at 81.7%, stage1 reaching 79.6% before regressing — see CLAUDE.md/
docs/HISTORY.md §14) relying on the chat tags to mark region boundaries,
or can the model infer E/S/Q/R structure from content/position alone with
no scaffolding at all? A companion result to the architecture/dataset
diagnostics in `kvmem/probe_signal_propagation.py`.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_notags_locate.py --device mps
"""


def _grid(chunk_len, warmup_lens, n_anchors, min_recall_len, rb_token, weight=1.0):
    """Generates weave_mix entries for one length: `n_anchors` evenly-spaced
    query_start values per (chunk_len, warmup_len) pair (deduped/clamped
    when the valid range is too small to fit them all distinctly — e.g.
    chunk_len=8 naturally has very few valid anchors). Also used for
    REHEARSAL (see below) by passing a smaller `n_anchors` (fewer anchors,
    same warmup_lens as the introducing stage) and `weight=0.5`.
    warmup_len is embedded in the DSL string itself via Q(...)'s 4th arg
    (Q(s,e,w,wl), kvmem/hmn_notags.py's parse_traj_dsl) rather than a
    separate dict key."""
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
    d=64, n_layers=8, n_heads=4, V=271,
    block_type='single_attn',
    lr_max=1e-4, lr_min=1e-6, wd=1e-5,
    warmup_steps=500, log_every=500,
    lr_schedule='cosine_restarts',
    cosine_T0=180000, cosine_T_mult=1,
    rope=False,
    null_kv=True,
    rmsnorm=True,
    name='hmn_notags_locate', seed=48,

    adaptive=True,
    adapt_signal='val_match',
    adapt_temp=1.0,
    adapt_ema_alpha=0.5,
    adapt_floor=0.05,

    state_len=8, state_vocab_size=2,
    warmup_len=8,  # stage-level fallback default — unused here, every entry's DSL sets its own wl
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=1, chunk_len=8, B=16, n_steps=480000, eval_every=24000, early_stop_mean=80.0,
             weave_mix=_grid(8, [2, 3, 4], n_anchors=4, min_recall_len=4, rb_token='B8')),

        dict(n_chunks=1, chunk_len=16, B=12, n_steps=720000, eval_every=36000, early_stop_mean=80.0,
             weave_mix=(
                 _grid(16, [2, 3, 4, 6, 8], n_anchors=4, min_recall_len=4, rb_token='B8')
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5)
             )),

        dict(n_chunks=1, chunk_len=32, B=6, n_steps=480000, eval_every=24000, early_stop_mean=80.0,
             weave_mix=(
                 _grid(32, [2, 4, 6, 8, 12, 16], n_anchors=4, min_recall_len=4, rb_token='B16')
                 + _grid(16, [2, 3, 4, 6, 8], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5)
             )),

        dict(n_chunks=1, chunk_len=64, B=4, n_steps=720000, eval_every=36000, early_stop_mean=80.0,
             weave_mix=(
                 _grid(64, [2, 4, 8, 12, 16, 24], n_anchors=4, min_recall_len=4, rb_token='B16')
                 + _grid(32, [2, 4, 6, 8, 12, 16], n_anchors=2, min_recall_len=4, rb_token='B16', weight=0.5)
                 + _grid(16, [2, 3, 4, 6, 8], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5)
             )),
    ],
)
