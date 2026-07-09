"""
First true-SRS experiment: depth-2 spaced-repetition schedule (halves -> full)
at the proven 64B/nc=4/slot_len=8 scale, using chat-tags' two validated fixes
(span-specific query tags, wrong-token-weighted IR loss) from the start rather
than discovering they're needed the hard way again.

Schedule (srs_schedule_depth2(4) = [(0,2),(2,4),(0,4)]):
  span (0,2): first 32B half  -> <query_a>
  span (2,4): second 32B half -> <query_b>
  span (0,4): full 64B        -> <query_c>
Each span gets its own local IQ + 2 chained argmax-IR turns, in ONE sequence
(L=902), reusing chunk_mask_fb's existing nochain rule (Rule 3b) to keep spans
architecturally independent (verified via smoke test — no cross-span leak).

Warm-started from the chat-tags wrong-token-weighted-loss checkpoint (97.2%
mean, Win C 91.7%) — same vocab/dims, direct transfer, no growth needed. The
two 32B half-spans partially transfer (same content shape the model already
knows); the new 64B full-span block is a genuinely new output length (56B,
vs 24B everywhere in the chat-tags series) and starts from scratch.

Val: make_test_sequences(64) (same convention as every prior experiment).
Test: datasets/suratalfatihah.txt padded to (4,16) via load_chunks_padded —
the held-out real-text eval the ORIGINAL (pre-chat-tags) SRS configs used
(configs/hmn_chunk_srs_ir.py) but chat-tags never wired in.

Success bar: each of the 3 spans reaches >=90% match on BOTH val and test,
matching the chat-tags bar. If this succeeds, extend to the full srs_schedule
(7 spans, singles->pairs->full) as the actual multi-session SRS scaling test.

See docs/SRS_RECIPE.md § "Resuming true SRS" for full design rationale.

Run:
    caffeinate -i python3 -m experiments.srs_tagged.train \\
        --config experiments/srs_tagged/configs/srs_depth2_nc4_slot8.py \\
        --pretrained experiments/chat_tags/logs/chat_tags_slot8_wrongtok_ablation/checkpoints/stage0_best.pt \\
        --device mps
    tail -f experiments/srs_tagged/logs/srs_depth2_nc4_slot8/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256, V=282,
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=60000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='srs_depth2_nc4_slot8', seed=48,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=2.0,
    val_n_seqs=3,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        dict(n_chunks=4, chunk_len=16, depth=2, n_refine=2, B=6, n_steps=60000, eval_every=5000),
    ],
)
