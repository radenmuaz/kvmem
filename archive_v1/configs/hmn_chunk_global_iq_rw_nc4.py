"""
Global IQ with random window training, nc=4 (64B source).

ONE global SLOT always at enc_end (position 80). At each training step a
random chunk-aligned window (0,2)/(1,3)/(2,4) is sampled: warmup fills
that window's first 8B, output fills the next 24B. Eval reports each window
separately in the standard val/iq_global_rw[win(a,b)_ncX]/MEAN format.

Why this is different from ir_local stitch:
  ir_local stitch: 3 separate SLOTs at positions 80/244/408.
  Independent eval (win C): SLOT must be at 80 → position mismatch → fail.
  iq_global_rw: ONE SLOT always at 80, warmup byte selects window.
  No position mismatch between training and eval.

nc=4 chosen so there are 3 valid windows to randomly sample:
  X=0  → warmup=src[0:8],  out=src[8:32]   tag: win(0,2)
  X=16 → warmup=src[16:24], out=src[24:48]  tag: win(1,3)
  X=32 → warmup=src[32:40], out=src[40:64]  tag: win(2,4)

SLOT position: 80 (same as ir_local independent eval for nc=4).

From: logs/hmn_chunk_local_32/checkpoints/stage0_end.pt (81.9% IQ only, nc=2)
Starting from IQ checkpoint: SLOT compression learned, extend to global recall.

Success bar: all win(a,b) >= 50% before comparing with vlen_ext results.

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_global_iq_rw_nc4.py \\
        --pretrained logs/hmn_chunk_local_32/checkpoints/stage0_end.pt \\
        --device mps
    tail -f logs/hmn_chunk_global_iq_rw_nc4/train_status.log

Traj mix:
| weight | nc | warmup_len | out_len | warmup offsets (random) | SLOT pos |
|--------|----|-----------|---------|------------------------|----------|
|   1.0  |  4 |         8 |      24 | random {0,16,32}       |       80 |
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500, log_every=1000,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='hmn_chunk_global_iq_rw_nc4', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=False,  # IQ only
    val_n_seqs=3,
    mask_nochain=False,

    curriculum=[
        dict(
            n_chunks=4, chunk_len=16, B=8, n_steps=50000, eval_every=10000,
            traj_mix=[
                dict(type='iq_global_rw', weight=1.0, n_chunks=4, window_chunks=2),
            ],
            eval_traj='iq_global_rw',
        ),
    ],
)
