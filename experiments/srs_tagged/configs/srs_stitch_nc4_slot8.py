"""
Stitched-window SRS: same scale as srs_depth2_nc4_slot8 (64B, nc=4, slot_len=8)
but replaces the atomic full-span block (0,4) with the proven `ir_local`
window geometry — three OVERLAPPING 32B windows (stride=16B, 50% overlap):

  windows = [(0,2), (1,3), (2,4)]   (chunk-index tuples, chunk_len=16)
    win A: bytes  0-31
    win B: bytes 16-47   <- 16B overlap with A
    win C: bytes 32-63   <- 16B overlap with B

Each window still gets its own local IQ + 2 chained argmax-IR turns (same
mechanism, same chunk_mask_fb nochain rule, same tag-per-window addressing
as srs_depth2_nc4_slot8 — <query_a/b/c> keyed by position in `windows`).

The difference is EVAL, not training: `eval_mode='stitch'` in the curriculum
stage switches from per-span GT-seeded decode to
`ar_decode_srs_stitched_tagged` (experiments/srs_tagged/stitch_decode.py) —
only window A's warmup is seeded from ground truth; window B's warmup is
seeded from window A's own decoded output (bytes 16-24, which fall inside
A's 8-31 output range), and window C's warmup from window B's own decoded
output. Full 64B coverage emerges from the chain, never from one large
single-shot block. See docs/SRS_RECIPE.md "Stitching vs atomic full-span"
for why this replaces the nc8 atomic scale-up in the experiment queue.

No new vocab/positions code needed — chunk_positions_srs_tagged already
takes an arbitrary `schedule` list; this config just passes overlapping
windows instead of srs_schedule_depth2's disjoint halves+full-span.

Training batches use TEACHER-FORCED ground-truth warmup per window (same as
every other config in this project) — chaining only matters at AR-decode
eval time, matching ir_local's own convention.

Warm-started from srs_depth2_nc4_slot8's best checkpoint once that run
finishes (same vocab/dims, direct transfer — the two half-windows (0,2) and
(2,4) already exist in muscle memory from that run; only window B (1,3) and
the stitched-chain eval behavior are new).

Run (only after srs_depth2_nc4_slot8 finishes — never two jobs at once):
    caffeinate -i python3 -m experiments.srs_tagged.train \\
        --config experiments/srs_tagged/configs/srs_stitch_nc4_slot8.py \\
        --pretrained experiments/srs_tagged/logs/srs_depth2_nc4_slot8/checkpoints/stage0_best.pt \\
        --device mps
    tail -f experiments/srs_tagged/logs/srs_stitch_nc4_slot8/train.log

Success bar: stitched full-sequence match% (the STITCHED_MEAN row) >=90% on
both val and test — directly comparable to srs_depth2_nc4_slot8's failed
span(0,4) atomic block (val 3.6%, test 8.9% at step 5000) as the thing being
replaced.
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256, V=282,
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=60000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='srs_stitch_nc4_slot8', seed=48,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=2.0,
    val_n_seqs=3,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        dict(n_chunks=4, chunk_len=16, n_refine=2, B=6, n_steps=60000, eval_every=5000,
             windows=[(0, 2), (1, 3), (2, 4)], eval_mode='stitch'),
    ],
)
