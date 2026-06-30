"""
Curriculum stages 0+1 for chunk feedback-SRS (see plan:
.claude/plans/synchronous-weaving-otter.md, also summarized in CLAUDE.md).

Stage 0 (iq_windowed): recall-only (no feedback) over srs_schedule_depth2
  (halves + full) — teaches slot compression + windowed/full recall.
Stage 1 (ir_winrefine): ONE global IQ read of the full source, followed by
  2 chained argmax-refine turns targeting a sampled window (half or full,
  weighted {0.4,0.4,0.2}) — the genuinely novel mechanism (feedback-refine on
  a sub-span of a larger encoded context). Replays stage 0's iq_windowed at
  30% to avoid forgetting plain windowed recall.

Run (one shot, both stages, random init):
    python -m kvmem.train_hmn_chunk \
        --config configs/hmn_chunk_curric.py \
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_curric', seed=42,

    slot_len=2, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,

    eval_file='datasets/suratalkauthar.txt',

    curriculum=[
        dict(n_chunks=2, chunk_len=128, depth=2, B=2, n_steps=12000, eval_every=3000,
             traj_mix=[dict(type='iq_windowed', weight=1.0)],
             eval_traj='iq_windowed'),
        dict(n_chunks=2, chunk_len=128, depth=2, B=2, n_steps=12000, eval_every=3000,
             traj_mix=[dict(type='ir_winrefine', weight=0.7),
                       dict(type='iq_windowed', weight=0.3)],
             eval_traj='ir_winrefine'),
    ],
)
