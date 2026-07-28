"""
`hmn_locate_nope_curriculum_dense.py` — denser variant of
`hmn_locate_nope_curriculum.py` (same idea: `rope=False`, zero positional
information, `traj_locate_and_continue`-style locate-and-continue task,
curriculum growing `src_len` 8->16->32->64 with rehearsal of previous
lengths). Two changes from the base config:

1. **~4x more diverse `(warmup_len, query_start)` grid per stage.** The
   base config used a fixed 2-anchor scheme (`query_start=0` and
   `query_start=max_valid` only) with 2-4 `warmup_len` values per stage.
   This config uses `_grid(chunk_len, warmup_lens, n_anchors=4, ...)`
   (defined below, generates entries programmatically rather than
   hand-typing dozens of dicts) with 4 evenly-spaced anchors per
   `(chunk_len, warmup_len)` pair AND more `warmup_len` values per stage
   (added intermediate values like 3, 6, 12, 24 alongside the original
   2/4/8/16/... doubling sequence). Net effect, entry counts per stage
   (new-length entries only, before rehearsal): 6/20/24/24 for
   stage0-3, vs the base config's 3/6/8/8 — roughly 3-4x (stage0 is
   naturally capped by how few distinct anchors even fit in an 8-byte
   source, so it undershoots the multiplier a bit).

2. **Adaptive per-stage early stopping** (`early_stop_mean`, new
   `kvmem/hmn.py` feature, general — not specific to this config): each
   stage now also sets `early_stop_mean=80.0` — if val MEAN reaches 80%
   at ANY eval within a stage, that stage ends immediately and training
   moves to the next curriculum stage, instead of burning through the
   rest of that stage's step budget. `n_steps` (same tripled budget as
   the base config: 60k/90k/120k/180k) remains the hard cap if the
   threshold is never reached — this is a ceiling swap, not a step-count
   change: stages that genuinely need the full budget still get it,
   stages that converge faster don't waste the remainder.

3. **Rehearsal now ALSO uses `_grid`, at ~50%+ anchor density** (the base
   config's rehearsal was 1-2 hand-picked entries per past length — a
   much thinner sample than what that length was actually introduced
   with). Each rehearsal call reuses the SAME `warmup_lens` the
   introducing stage used, but with `n_anchors=2` (half of the
   introducing stage's `n_anchors=4`) and `weight=0.5`. Measured
   coverage (`n_anchors=2` grid size / `n_anchors=4` grid size): len=8
   -> 5/6 (83%, naturally capped — an 8-byte source has too few valid
   anchors to halve cleanly), len=16 -> 10/20 (50%), len=32 -> 12/24
   (50%) — every past length gets at least half of its own introducing
   stage's anchor diversity rehearsed, not just a couple of fixed
   points. This does make each stage's total mix noticeably bigger
   (stage1=25, stage2=39, stage3=51 entries, vs. the base config's
   8/11/12) — an explicit, accepted tradeoff for thorough rehearsal
   coverage over a small/fast mix.

Trained FROM SCRATCH (not warm-started), same as the base config.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_locate_nope_curriculum_dense.py --device mps
"""


def _grid(chunk_len, warmup_lens, n_anchors, min_recall_len, rb_token, weight=1.0):
    """Generates weave_mix entries for one length: `n_anchors` evenly-spaced
    query_start values per (chunk_len, warmup_len) pair (deduped/clamped
    when the valid range is too small to fit them all distinctly — e.g.
    chunk_len=8 naturally has very few valid anchors). Also used for
    REHEARSAL (see below) by passing a smaller `n_anchors` (fewer anchors,
    same warmup_lens as the introducing stage) and `weight=0.5`.
    warmup_len is embedded in the DSL string itself via Q(...)'s 4th arg
    (Q(s,e,w,wl), kvmem/hmn.py's parse_traj_dsl) rather than a separate
    dict key — this is what makes each entry's val log label actually
    unique: previously every (start=0, *) entry across different
    warmup_lens shared the identical dsl string (warmup_len lived in a
    same-named dict key invisible to the log label), so e.g.
    'E(16) Q(0,1,0) B8' printed 6x per eval with 6 different match% and
    no way to tell them apart."""
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
    d=64, n_layers=8, n_heads=4, V=274,
    block_type='single_attn',
    lr_max=1e-4, lr_min=1e-6, wd=1e-5,
    warmup_steps=500, log_every=500,
    lr_schedule='cosine_restarts',
    cosine_T0=180000, cosine_T_mult=1,
    rope=False,
    null_kv=True,
    rmsnorm=True,
    name='hmn_locate_nope_curriculum_dense', seed=48,

    # Adaptive weave_mix reweighting (kvmem/hmn.py, merged from hmn_adaptive_trainer.py) —
    # added after hmn_locate_nope_curriculum's stage1 showed a real failure mode: the
    # hardest entry (E(16) Q(0,1,0) B8, warmup_len=2) landed at match=9.5% while easier
    # entries in the same stage hit 66-100%, with no reweighting to compensate. adapt_signal
    # defaults to 'val_match', which is exactly what's needed here — struggling entries get
    # upweighted toward more training share automatically instead of a fixed base_weight=1.0
    # for every entry in the (much larger, 25-51 entry) dense grid regardless of difficulty.
    adaptive=True,
    adapt_signal='val_match',
    adapt_temp=1.0,
    adapt_ema_alpha=0.5,
    adapt_floor=0.05,

    state_len=8, state_vocab_size=2,
    warmup_len=8,  # stage-level fallback default — unused here, every entry's DSL sets its own W<n>
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    curriculum=[
        # stage0: introduce len=8 (dense grid), no rehearsal yet
        # n_steps quadrupled 120000->480000 (eval_every kept at the same 5% fraction) — the
        # doubled budget still wasn't enough to trigger early_stop_mean=80.0, so this makes
        # the cap deliberately generous: if the model CAN reach 80% MEAN on this mix, it now
        # has enough steps to do it and stop early rather than being cut off first.
        dict(n_chunks=1, chunk_len=8, B=16, n_steps=480000, eval_every=24000, early_stop_mean=80.0,
             weave_mix=_grid(8, [2, 3, 4], n_anchors=4, min_recall_len=4, rb_token='B8')),

        # stage1: introduce len=16 (dense grid), rehearse len=8 at ~50%+ anchor density
        # n_steps quadrupled 180000->720000 (eval_every kept at the same 5% fraction), same
        # reasoning as stage0.
        dict(n_chunks=1, chunk_len=16, B=12, n_steps=720000, eval_every=36000, early_stop_mean=80.0,
             weave_mix=(
                 _grid(16, [2, 3, 4, 6, 8], n_anchors=4, min_recall_len=4, rb_token='B8')
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5)
             )),

        # stage2: introduce len=32 (dense grid), rehearse len=16 and len=8 at ~50%+ anchor density
        # n_steps quadrupled 120000->480000, same reasoning.
        dict(n_chunks=1, chunk_len=32, B=6, n_steps=480000, eval_every=24000, early_stop_mean=80.0,
             weave_mix=(
                 _grid(32, [2, 4, 6, 8, 12, 16], n_anchors=4, min_recall_len=4, rb_token='B16')
                 + _grid(16, [2, 3, 4, 6, 8], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5)
             )),

        # stage3: introduce len=64 (dense grid), rehearse len=32/16/8 at ~50%+ anchor density
        # n_steps quadrupled 180000->720000, same reasoning.
        dict(n_chunks=1, chunk_len=64, B=4, n_steps=720000, eval_every=36000, early_stop_mean=80.0,
             weave_mix=(
                 _grid(64, [2, 4, 8, 12, 16, 24], n_anchors=4, min_recall_len=4, rb_token='B16')
                 + _grid(32, [2, 4, 6, 8, 12, 16], n_anchors=2, min_recall_len=4, rb_token='B16', weight=0.5)
                 + _grid(16, [2, 3, 4, 6, 8], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5)
                 + _grid(8, [2, 3, 4], n_anchors=2, min_recall_len=4, rb_token='B8', weight=0.5)
             )),
    ],
)
