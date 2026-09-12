"""
`hmn_locate_nope_curriculum.py` — the most extreme end of the positional-
shortcut fix spectrum: `rope=False`, no `dual_rope`/`rope_state_scale`/
`relpos_enabled` either — literally ZERO positional information anywhere
in the model (`MHAttention.forward` never calls `apply_rope` at all when
`self.rope` is False — confirmed this needs no new code, purely a config
setting). Combined with `traj_locate_and_continue` (`kvmem/hmn.py`): the
model must locate a real ground-truth excerpt wherever it sits in the
source and continue from there, with NO positional cue available at all —
purely content-addressed attention, forced by construction rather than
by hoping a scaling/windowing trick makes position unreliable enough.

Idea being tested: can a model with zero position information learn this
task at all via curriculum alone (small, easy shapes first, growing to
harder ones), i.e. does content-only attention have a learnable path to
"find this excerpt and continue," or does it need SOME positional
scaffolding (even the minimal `relpos` local window) to bootstrap.

**`E(len)` DSL syntax** (`kvmem/hmn.py` — new since this config was first
drafted): chunk_len is embedded directly in each entry's DSL string itself
(`E(64)` = "one chunk of exactly 64 bytes") rather than an external
per-entry `chunk_len=` config key. `E1`/`E4` etc. are unchanged (still "n
same-length chunks using the stage default") — `E(len)`'s parens are a
distinct, backward-compatible syntax form. This is what makes the
rehearsal design below practical: every entry in a stage's `weave_mix`
carries its own length in the string itself, so old and new lengths can
sit side by side in one mix with no bookkeeping mismatch risk.

**Curriculum** (4 stages, `src_len` doubling 8->16->32->64). Each stage
introduces one NEW (harder) length with a handful of `(warmup_len,
query_start)` combinations (2 anchors per valid pair — `query_start=0`
baseline and `query_start=max_valid`, the hardest case), same grid logic
as `hmn_single_recall_c64_locate.py`. `warmup_len` starts at a floor of 2
and grows with the stage; `min_recall_len=4`.

**Step budgets tripled** (20k/30k/40k/60k -> 60k/90k/120k/180k,
`cosine_T0` tripled to match 180000) after a first attempt showed clear
per-stage underfitting — loss and per-entry `traj_loss` were still
declining, not plateaued, when each stage's original budget ran out
(observed directly: stage1 was still dropping 2.0-2.9 with no sign of
flattening at its old 30000-step cutoff before this change).

**Rehearsal — the new part**: every stage AFTER the first also mixes in a
small number of entries at PREVIOUSLY-introduced lengths, so training on
longer sources doesn't let the model forget what it learned on shorter
ones (a real risk with any strictly-sequential curriculum — nothing in
stage3 would otherwise ever show the model an 8-byte source again, 60000
steps after stage0 ended). Kept deliberately light (1-2 entries per past
length, not full re-inclusion of every earlier entry) to avoid the mix
size exploding stage over stage while still giving each old length
periodic refresh:
  stage0 (len=8, introduce):  3 entries, no rehearsal (nothing to rehearse yet)
  stage1 (len=16, introduce): 6 new + 2 rehearsal (len=8)             = 8 entries
  stage2 (len=32, introduce): 8 new + 2 rehearsal (len=16) + 1 (len=8) = 11 entries
  stage3 (len=64, introduce): 8 new + 2 rehearsal (len=32) + 1 (len=16) + 1 (len=8) = 12 entries

`repeat_batch` (B8/B16 per DSL token) scaled by each ENTRY's OWN length
difficulty (short lengths keep B8 even when rehearsed inside a later,
mostly-B16 stage) — NoPE is expected to converge more slowly than any
position-assisted mechanism (nothing to bootstrap from), so extra
gradient steps per sampled batch matter more here than in prior configs.

Trained FROM SCRATCH (not warm-started) — RoPE-trained weights have no
reason to transfer meaningfully into a zero-position architecture, unlike
`rope_state_scale`/`dual_rope` which only change what position values feed
an existing RoPE mechanism.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_locate_nope_curriculum.py --device mps
"""

hp = dict(
    d=64, n_layers=8, n_heads=4, V=274,
    block_type='single_attn',
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=500,
    lr_schedule='cosine_restarts',
    cosine_T0=180000, cosine_T_mult=1,  # matches the longest (final) stage — see hmn_weave_c64.py's
                                        # own docstring for why a single shared T0 does this correctly.
                                        # Tripled (60000->180000) alongside every stage's n_steps below —
                                        # first attempt showed clear underfitting per stage, not a plateau
                                        # (loss/traj_loss still declining when each stage's step budget ran out)
    rope=False,  # NoPE — no yarn needed either, only matters when rope=True
    null_kv=True,
    rmsnorm=True,
    name='hmn_locate_nope_curriculum', seed=48,

    state_len=8, state_vocab_size=2,
    warmup_len=8,  # stage-level default, overridden per-entry below
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    curriculum=[
        # stage0: introduce len=8, no rehearsal (nothing to rehearse yet)
        dict(n_chunks=1, chunk_len=8, B=16, n_steps=60000, eval_every=3000,
             weave_mix=[
                 dict(weight=1.0, dsl='E(8) Q(0,1,0,2) B8'),
                 dict(weight=1.0, dsl='E(8) Q(0,1,2,2) B8'),
                 dict(weight=1.0, dsl='E(8) Q(0,1,0,4) B8'),
             ]),

        # stage1: introduce len=16, rehearse len=8
        dict(n_chunks=1, chunk_len=16, B=12, n_steps=90000, eval_every=4500,
             weave_mix=[
                 dict(weight=1.0, dsl='E(16) Q(0,1,0,2) B8'),
                 dict(weight=1.0, dsl='E(16) Q(0,1,10,2) B8'),
                 dict(weight=1.0, dsl='E(16) Q(0,1,0,4) B8'),
                 dict(weight=1.0, dsl='E(16) Q(0,1,8,4) B8'),
                 dict(weight=1.0, dsl='E(16) Q(0,1,0,8) B8'),
                 dict(weight=1.0, dsl='E(16) Q(0,1,4,8) B8'),
                 # rehearsal (len=8)
                 dict(weight=0.5, dsl='E(8) Q(0,1,0,2) B8'),
                 dict(weight=0.5, dsl='E(8) Q(0,1,2,2) B8'),
             ]),

        # stage2: introduce len=32, rehearse len=16 and len=8
        dict(n_chunks=1, chunk_len=32, B=6, n_steps=120000, eval_every=6000,
             weave_mix=[
                 dict(weight=1.0, dsl='E(32) Q(0,1,0,2)  B16'),
                 dict(weight=1.0, dsl='E(32) Q(0,1,26,2)  B16'),
                 dict(weight=1.0, dsl='E(32) Q(0,1,0,4)  B16'),
                 dict(weight=1.0, dsl='E(32) Q(0,1,24,4)  B16'),
                 dict(weight=1.0, dsl='E(32) Q(0,1,0,8)  B16'),
                 dict(weight=1.0, dsl='E(32) Q(0,1,20,8)  B16'),
                 dict(weight=1.0, dsl='E(32) Q(0,1,0,16) B16'),
                 dict(weight=1.0, dsl='E(32) Q(0,1,12,16) B16'),
                 # rehearsal (len=16, len=8)
                 dict(weight=0.5, dsl='E(16) Q(0,1,0,4) B8'),
                 dict(weight=0.5, dsl='E(16) Q(0,1,8,4) B8'),
                 dict(weight=0.5, dsl='E(8)  Q(0,1,0,2) B8'),
             ]),

        # stage3: introduce len=64, rehearse len=32, len=16, len=8
        dict(n_chunks=1, chunk_len=64, B=4, n_steps=180000, eval_every=9000,
             weave_mix=[
                 dict(weight=1.0, dsl='E(64) Q(0,1,0,2)  B16'),
                 dict(weight=1.0, dsl='E(64) Q(0,1,58,2)  B16'),
                 dict(weight=1.0, dsl='E(64) Q(0,1,0,4)  B16'),
                 dict(weight=1.0, dsl='E(64) Q(0,1,56,4)  B16'),
                 dict(weight=1.0, dsl='E(64) Q(0,1,0,8)  B16'),
                 dict(weight=1.0, dsl='E(64) Q(0,1,52,8)  B16'),
                 dict(weight=1.0, dsl='E(64) Q(0,1,0,16) B16'),
                 dict(weight=1.0, dsl='E(64) Q(0,1,44,16) B16'),
                 # rehearsal (len=32, len=16, len=8)
                 dict(weight=0.5, dsl='E(32) Q(0,1,0,8) B16'),
                 dict(weight=0.5, dsl='E(32) Q(0,1,20,8) B16'),
                 dict(weight=0.5, dsl='E(16) Q(0,1,0,4) B8'),
                 dict(weight=0.5, dsl='E(8)  Q(0,1,0,2) B8'),
             ]),
    ],
)
