"""
**SUPERSEDED — do not launch.** srs_depth2_nc4_slot8 (same atomic full-span
design as this config, just at 64B) confirmed the full-span (0,4) block as
the clear bottleneck at every checkpoint (e.g. step 5000: val span(0,4)=3.6%
vs span(0,2)=100%, span(2,4)=65.3%; test span(0,4)=8.9% vs 79.2%/4.2%).
Doubling n_chunks here would only make that single-shot decode LONGER
(112-byte output instead of 56), compounding the exact mechanism that's
already failing. Replaced by experiments/srs_tagged/configs/srs_stitch_nc4_slot8.py
(stitched overlapping windows, no atomic full-span block) — validate that
at 64B first, then its own nc=8 follow-up will double window COUNT, not
window SIZE. See docs/SRS_RECIPE.md "Stitching vs atomic full-span".

--- Original docstring below, kept for the record ---

Second true-SRS scale step: 128B coverage (up from 64B), same proven per-chunk
ratio (chunk_len=16, slot_len=8 -> 2x compression, unchanged from every prior
successful experiment). Only n_chunks doubles (4->8) — per the project's own
MDL principle (docs/MDL_MODEL_SIZE.md: "more chunks, not bigger chunks, is the
correct scaling axis"), and per the compute-cost analysis in docs/SRS_RECIPE.md
§ Scaling roadmap (doubling n_chunks at fixed chunk_len costs ~3x compute, not
the ~161x a chunk_len=256 jump would have cost for 16x more coverage).

Schedule (srs_schedule_depth2(8) = [(0,4),(4,8),(0,8)]):
  span (0,4): first 64B half  -> <query_a>
  span (4,8): second 64B half -> <query_b>
  span (0,8): full 128B       -> <query_c>

Test: datasets/suratalfatihah.txt (562B) via load_chunks_padded — at nc=8/
chunk_len=16 this covers the first 16 bytes of each of 8 line-groups = 128
bytes total, same truncation caveat as the 64B run (not yet full-surah
coverage — see docs/SRS_RECIPE.md's methodology-gap note). Report results
precisely: "test=100%" here means 100% of the 128-byte excerpt actually used.

Queued to launch immediately after srs_depth2_nc4_slot8.py finishes (never
two training jobs at once) — warm-start from ITS best checkpoint once that
run completes and its result is written up.

Estimated cost: L~1764 (vs 996 for the 64B run), ~3.1x more compute per step,
~1.3 it/s, ~13 hours for 60k steps — confirmed acceptable time budget.

Run:
    caffeinate -i python3 -m experiments.srs_tagged.train \\
        --config experiments/srs_tagged/configs/srs_depth2_nc8_slot8.py \\
        --pretrained experiments/srs_tagged/logs/srs_depth2_nc4_slot8/checkpoints/stage0_best.pt \\
        --device mps
    tail -f experiments/srs_tagged/logs/srs_depth2_nc8_slot8/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256, V=282,
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=60000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='srs_depth2_nc8_slot8', seed=49,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=2.0,
    val_n_seqs=3,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        dict(n_chunks=8, chunk_len=16, depth=2, n_refine=2, B=6, n_steps=60000, eval_every=5000),
    ],
)
