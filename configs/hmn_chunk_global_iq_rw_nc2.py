"""
Global IQ with random warmup offset, nc=2 (32B source).

Goal: test attention targeting hypothesis. The IQ SLOT must compress all 32B
(attends to both enc_block SLOTs), then recall 16 bytes starting at a random
offset X in [0, 12]. Forces the model to learn position-invariant full-source
encoding and selective recall from arbitrary starting positions.

Contrast with ir_local win(0,2) nc=2: that also reads both enc_blocks, but
always warmup = bytes[0:8] and out = bytes[8:32] (fixed position). Here:
  - warmup_len=4  (shorter warmup seed)
  - out_len=16    (partial source recall)
  - X ~ uniform[0, 12]  → warmup=src[X:X+4], target=src[X+4:X+20]

No IR refinement (n_refine=0) — prove IQ-only global recall first.

From: logs/hmn_chunk_local_32/checkpoints/stage0_end.pt  (81.9%, IQ only)
NOT stage 2 — that has IR training which changes SLOT behavior.

Traj mix:
| weight | nc | warmup_len | out_len | warmup offset | SLOT pos |
|--------|----|-----------:|-------:|--------------|----------|
|   1.0  |  2 |          4 |     16 | random[0,12] |       40 |

Eval: match% averaged over offsets X=0,4,8,12 (4 positions).
Success bar: mean match >= 50% across offsets.

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_global_iq_rw_nc2.py \\
        --pretrained logs/hmn_chunk_local_32/checkpoints/stage0_end.pt \\
        --device mps
    tail -f logs/hmn_chunk_global_iq_rw_nc2/train_status.log
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500, log_every=1000,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='hmn_chunk_global_iq_rw_nc2', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=4,
    use_actual_argmax=False,  # no IR turns
    val_n_seqs=3,
    mask_nochain=False,

    curriculum=[
        dict(
            n_chunks=2, chunk_len=16, B=8, n_steps=50000, eval_every=10000,
            traj_mix=[
                dict(type='iq_global_rw', weight=1.0, n_chunks=2, out_len=16),
            ],
            eval_traj='iq_global_rw',
        ),
    ],
)
