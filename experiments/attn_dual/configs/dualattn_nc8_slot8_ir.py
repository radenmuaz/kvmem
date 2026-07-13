"""
Dual-attn (no MLP), rmsnorm=True, at 128B scale (nc=8, 7 chained windows) —
tests whether RMSNorm fixes window G's "IR2 destroys IR1's gain" failure
(documented in docs/SRS_RECIPE.md "Stitching vs atomic full-span" and the
srs_stitch_nc8_slot8/_continue results), using dual-attn+RMSNorm — the
architecture just confirmed as the working choice after the full 2x2
matched-depth comparison (docs/SRS_RECIPE.md "Full 2x2 matched-depth
comparison": dual-attn+RMSNorm's IR result was ~3-4x better than either
standard-arch variant at matched depth).

Warm-started from dualattn_nc4_slot8_ir's checkpoint (94.0%/94.6%, the
strongest dual-attn result so far) rather than training from scratch —
grows special_embed for the D-G window tags (V 282->290) via the existing
partial-load-by-shape logic in experiments/attn_dual/train.py (same
mechanism proven for the standard-arch nc4->nc8 vocab growth earlier this
session). Moderate step budget (60000, matching the original nc8 atomic
run's budget) since this is a targeted fix test, not a full matched-depth
replication — if RMSNorm resolves window G within this budget, extend;
if not, this at least gives a real signal within a bounded budget rather
than committing to another 260k-step run upfront.

First attempt (B=6, no chunk_attn) crashed with MPS OOM after step 5000 — DualAttnModel
has TWO attention sublayers per block (vs one in the standard architecture), roughly
doubling attention-map memory at this L=1694, and chunk_attn (the existing memory-only
row-chunked-attention flag) was never wired into experiments/attn_dual/model.py at all.
Fixed: chunk_attn now wired through (build_dualattn_model, train.py's hp_model dict),
set to 256 here (matching standard-arch nc8 configs); B lowered 6->3 for extra margin.

Run:
    caffeinate -i python3 -m experiments.attn_dual.train \\
        --config experiments/attn_dual/configs/dualattn_nc8_slot8_ir.py \\
        --pretrained experiments/attn_dual/logs/dualattn_nc4_slot8_ir/checkpoints/stage0_end.pt \\
        --device mps
    tail -f experiments/attn_dual/logs/dualattn_nc8_slot8_ir/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, V=290,
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=60000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True,
    rmsnorm=True, chunk_attn=256,
    name='dualattn_nc8_slot8_ir', seed=48,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=2.0,
    val_n_seqs=3,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        dict(n_chunks=8, chunk_len=16, n_refine=2, B=3, n_steps=60000, eval_every=5000,
             windows=[(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8)]),
    ],
)
