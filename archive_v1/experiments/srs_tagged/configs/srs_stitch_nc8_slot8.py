"""
Stitching scale-up step 2: 128B coverage (nc=8, up from nc=4's 64B), via MORE
CHAINED WINDOWS (7 overlapping 32B windows) rather than a bigger atomic block —
this is the corrected "128B milestone" that srs_depth2_nc8_slot8.py (SUPERSEDED,
atomic full-span) was meant to reach, now via the mechanism that actually won
the 64B comparison (see docs/SRS_RECIPE.md "Stitching vs atomic full-span":
srs_stitch_nc4_slot8 held a perfect 100%/100% sweep steps 35000-60000, the
atomic run's full-span block never exceeded 69.6% test).

windows = [(0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8)]   (chunk_len=16, stride=16B)
  win A-G, chained: only win A's warmup seeded from GT, every later window's
  warmup comes from the previous window's own decoded output (stitch_decode.py).

Needs 4 new window-identity tags (D,E,F,G) beyond the nc4 run's A/B/C — added to
experiments/chat_tags/vocab.py (HMN_QUERY_D..G, HMN_TAG_VOCAB_SIZE_V3=290) and
experiments/chat_tags/positions.py's _SRS_SPAN_TAGS (now 7 entries). Warm-started
from srs_stitch_nc4_slot8's best checkpoint via the existing partial-load-by-shape
logic in train.py, which grows special_embed's row count automatically (282->290)
and copies the overlapping A/B/C rows directly — D-G start randomly initialized.

Compute: L=1694 (vs 742 for nc4's 3 windows) — measured via
chunk_positions_srs_tagged(8,16,8,8,windows,n_refine=2). Estimated ~1.1-1.3 it/s
(quadratic falloff from nc4's 5.9 it/s at L=742), ~13-15hr for 60k steps — fits
within a 24hr autonomous budget alongside monitoring/eval time.

Test: datasets/suratalfatihah.txt via load_chunks_padded at n_chunks=8,
chunk_len=16 -> 128 bytes of real text (up from 64B at nc=4) — same truncation
caveat as before (not the whole 562-byte surah), report precisely.

Success bar: STITCHED_MEAN >=90% val AND test, ideally sustained 100% like the
nc4 result, before considering any further scale-up.

Run:
    caffeinate -i python3 -m experiments.srs_tagged.train \\
        --config experiments/srs_tagged/configs/srs_stitch_nc8_slot8.py \\
        --pretrained experiments/srs_tagged/logs/srs_stitch_nc4_slot8/checkpoints/stage0_best.pt \\
        --device mps
    tail -f experiments/srs_tagged/logs/srs_stitch_nc8_slot8/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256, V=290,
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=60000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='srs_stitch_nc8_slot8', seed=48,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=2.0,
    val_n_seqs=3,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        dict(n_chunks=8, chunk_len=16, n_refine=2, B=6, n_steps=60000, eval_every=5000,
             windows=[(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8)],
             eval_mode='stitch'),
    ],
)
