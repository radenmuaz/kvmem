"""
`hmn_tpu_recall1024_flat.py` — Run A of the TPU scale-up experiment (see
CLAUDE.md's scale-up entry and `/Users/muaz/.claude/plans/dazzling-waddling-
widget.md` for the full plan). Goal: a model that recalls a 1024-byte source
byte-exactly given a warmup excerpt anchored at ANY index, trained from
scratch on `tpu1` (a single v5e chip, `v5litepod-1`) at a batch size two
orders of magnitude above anything this project has run before (prior work:
`hmn_stitch_src1024_anchor.py`, B<=6, 165K params, MPS).

**Flat, not curriculum** — deliberately. `hmn_notags_w25` / `hmn_notags_
weave_anchor` / `hmn_stitch_src1024_anchor` built a curriculum ladder only
because MPS could not afford B>16 at this L (~1200-2200). With ~100x the
throughput that assumption is worth retesting directly: this config attacks
the 1024-byte target with NO curriculum, NO `_grid`/`_grid_shapes`/`_grid_
stitch` sweeps, NO rehearsal entries, NO adaptive reweighting, NO `_pretrained_
ckpt` (which also sidesteps the V=274/V=271 warm-start hazard flagged in
CLAUDE.md entirely). If this fails against the criterion below, Run B
(`hmn_tpu_recall1024_curr.py`, not yet built) reintroduces the curriculum
ladder using the same port.

**Model**: d=128, n_layers=16, n_heads=8 -> 1,118,208 params (verified via
build_model) — deeper AND wider than every prior config's d=64/n_layers=8,
`d=128` = exactly one v5e MXU tile (128x128).

**Position encoding**: NoPE (`rope=False`) — the `w25`/`weave_anchor`
lineage's own setting (this config went RoPE -> NoPE -> RoPE -> NoPE across
planning; NoPE is final). The measured head-to-head elsewhere
(`hmn_notags_w25` vs `hmn_notags_w25_rope`, CLAUDE.md's "Positional
shortcut" entry) showed RoPE converging faster/higher, and
`hmn_stitch_src1024_anchor` (the closest prior config to this target) used
RoPE — so this is a deliberate departure from both, not an oversight; if
Run A's convergence looks position-encoding-limited, revisit RoPE. The
anchor sweep below (8 evenly-spaced anchors across the full 1024-byte span)
is the mechanism this project uses to defeat the positional shortcut
(`kvmem/probe_positional_shortcut.py`'s finding that fixed-anchor queries get
resolved by attention position, not warmup content) — that fix is orthogonal
to the position encoding, so it applies here regardless of RoPE vs NoPE.
`L_train`/`L_max` are irrelevant under `rope=False` (no RoPE/YaRN frequency
calibration needed) and left unset.

**STATE register**: `state_len=4`, `state_vocab_size=1` — smaller `state_len`
than every prior result on record (`state_len=8`), AND `state_vocab_size=1`
(every prior config used `=2`). Nominal per-position STATE capacity is
`state_len * 2 * d * n_layers`; d and n_layers both doubling vs. the 165K
baseline already multiplies that 4x, so state_len=4 here carries 2x the
payload state_len=8 did at the old size. `state_vocab_size=1` means
`_cyclic_state_ids` (`kvmem/hmn.py:123`) emits the SAME value token at every
STATE slot — under NoPE, position within a STATE block is then recoverable
ONLY through causal depth (the model must count), which IS the deliberate
ablation this run tests: does NoPE's causal structure alone suffice to
address STATE slots, with no per-slot token signal at all? `V=271` is
UNCHANGED (no vocab constant edits) — the 12 reserved STATE value ids stay
free, so stepping back up to `state_vocab_size=2`/`4` later is a config-only
change.

**Length bucketing**: `bucket_lengths=True` — the 8 anchors below produce 8
distinct sequence lengths (1232-2128, `wl` doesn't move L — see the plan
doc's derivation: L = 1104 + 1024 - anchor, independent of warmup_len), each
its own XLA compile; `max_shape_buckets=8` keeps every one of them as its
own bucket (no padding waste at all here — this mix is small enough that
bucketing is exact, not lossy). `token_budget=131072` derives B=32-64 per
bucket (memory-bound: B*Lb roughly constant, matching the plan doc's
throughput-table reasoning) — conservative given no empirical HBM profiling
exists yet for this model size; raise it once gate 5 (end-to-end TPU smoke
test) confirms headroom.

Run (never two jobs at once — this is the only job intended to run on tpu1):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_tpu_recall1024_flat.py --device tpu

**Failure criterion, fixed in advance** (see the plan doc) — Run A has
failed if, at step 150000: val MEAN < 60% AND the 20k-step rolling-average
loss shows no downward trend over the preceding 50k steps. Either alone is
not enough (see `hmn_single_recall_c128`'s "undertrained, not capacity-
limited" lesson in CLAUDE.md). On that trigger, build Run B
(`hmn_tpu_recall1024_curr.py`) reintroducing the curriculum ladder.
"""

# 8 anchors spanning the full 1024-byte source (0, 128, ..., 896) x
# warmup_len in {32, 64} = 16 entries, uniform weight. L is independent of
# warmup_len (see docstring), so this is 8 distinct sequence lengths, each
# exercised at two different warmup/response-length splits — verified via
# kvmem.hmn.parse_traj_dsl + chunk_positions_traj directly:
#   anchor=  0  wl=32/64  L=2128
#   anchor=128  wl=32/64  L=2000
#   anchor=256  wl=32/64  L=1872
#   anchor=384  wl=32/64  L=1744
#   anchor=512  wl=32/64  L=1616
#   anchor=640  wl=32/64  L=1488
#   anchor=768  wl=32/64  L=1360
#   anchor=896  wl=32/64  L=1232
_ANCHORS = [0, 128, 256, 384, 512, 640, 768, 896]
_WARMUP_LENS = [32, 64]

_WEAVE_MIX = [
    dict(weight=1.0, dsl=f'E(64) E15 Q(0,16,{a},{wl})')
    for a in _ANCHORS
    for wl in _WARMUP_LENS
]

hp = dict(
    d=128, n_layers=16, n_heads=8, V=271,
    block_type='single_attn',
    lr_max=6e-4, lr_min=1e-6, wd=0.0001,
    warmup_steps=2000, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=200000, cosine_T_mult=1,
    rope=False,
    null_kv=True,
    rmsnorm=True,
    # Model-depth gradient checkpointing (HMNModel's own, NOT the separate segment_
    # checkpoint/forward_granularity mechanism) — MANDATORY at this L, not just a
    # tuning knob. Without it, gate 5 (kvmem/gate_check.py) measured a real HBM OOM
    # on tpu1: 52.85G requested vs 15.75G available at B=64/Lb=1232 — autograd was
    # retaining the O(B*H*Lb^2) attention score matrix for all 16 layers
    # simultaneously (the arithmetic matches almost exactly: 64*8*1232^2*16 layers
    # *4 bytes/fp32 = 52.8G). 'block' checkpoints each SingleAttnBlock, so backward
    # recomputes one layer's activations at a time instead of retaining all 16 —
    # expected peak drops ~16x on that term. Re-verify via gate 5 after any change
    # to d/n_layers/B/L here, this isn't a one-time-safe setting.
    grad_checkpoint='block',
    name='hmn_tpu_recall1024_flat', seed=51,
    # No _pretrained_ckpt — from scratch, sidesteps the V=274/V=271 warm-
    # start hazard (CLAUDE.md) entirely.

    state_len=4, state_vocab_size=1,
    warmup_len=32,  # stage-level fallback default — unused, every entry's DSL sets its own wl
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,
    repeat_batch=1,  # deliberately NOT re-enabled by default here — its documented win
                      # (weave_mix_accum_rnn: 44.8%->53.7%) was measured at B=2 against a
                      # genuine training-loss plateau; ablate only if Run A itself plateaus.

    # TPU/XLA support (see kvmem/hmn.py's _bucket_ceilings/_pad_mask_to and this
    # file's own docstring) — opt-in, off by default, every other config unaffected.
    # max_shape_buckets/attn_sq_budget REVISED DOWN (2026-07-30) after a real gate-5
    # OOM on tpu2/v6e: `Used 51.70G of 31.25G hbm` at the Lb=1744/B=32 bucket. The
    # original 125_000_000 target (below) assumed grad_checkpoint='block' bounds peak
    # attention memory to ONE recomputed layer (~4GB) — but the actual OOM'd memory
    # dump showed multiple ~2.9-3GB fp32 attention-score buffers alive simultaneously
    # (XLA's own HLO rematerialization creating remat/remat_compressed/remat_
    # uncompressed copies) PLUS 8 large buckets (Lb=1232..2128) all compiled within
    # one stage — total usage (51.7G) is suspiciously close to the UNCHECKPOINTED
    # all-16-layers-at-once formula, matching the original bug-3 OOM's own math almost
    # exactly. Whether that's checkpointing not fully saving memory here, or 8
    # buckets' compiled-executable memory not being released between each other, was
    # not conclusively isolated — cutting both levers (fewer simultaneous buckets,
    # smaller B per bucket) rather than resolving which mechanism dominates.
    bucket_lengths=True,
    max_shape_buckets=4,  # was 8 — halves the number of large compiled graphs resident at once
    token_budget=131072,
    # Second, independent B ceiling scaling as B*Lb^2 (not just B*Lb) — see
    # grad_checkpoint's comment above for why: even WITH checkpointing, the
    # O(B*H*Lb^2) attention-matrix term for the one actively-recomputed layer
    # still dominates at these L, and token_budget alone (a pure B*Lb proxy)
    # doesn't capture that. Cut to 1/4 of the original 125_000_000 (which
    # targeted ~4GB/bucket and still OOM'd) — ~1GB/bucket target, empirically
    # re-verify via gate 5 before trusting this number for a real run either.
    attn_sq_budget=31_000_000,

    curriculum=[
        dict(n_chunks=16, chunk_len=64, B=64, n_steps=200000, eval_every=10000,
             hops=-1,  # single query, op_idx=0 is always relay-exempt — matches
                       # traj_suffix's own reasoning, hops is unused here but set
                       # explicitly (not left at the default) for clarity
             weave_mix=_WEAVE_MIX),
    ],
)
