"""
Stage 4 stitch-only: 128B src, 7 overlapping 32B windows.
Pure stitch training. mask_nochain=True (full prior rec_block blackout — corrected v5 fix).

Rationale:
  Stage 3 v5 (64B, corrected nochain) must pass with win1/win2 independent ≥40% AND
  stitch ≥50% before running this stage. Update --pretrained to v5 end checkpoint.

  mask_nochain=True blocks IQ SLOT from ALL prior rec_block tokens (SLOT+warmup+output).
  With this fix, pure stitch training suffices — independence is enforced by the mask.

Sequence lengths:
  all-7-windows: enc(8×20=160) + 7×164 = 160+1148 = 1308 tokens
  B=4 → 4×1308=5232 tokens/batch (proven safe for MPS)

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_local_128_stitch.py \\
        --pretrained logs/hmn_chunk_local_64_v5/checkpoints/stage0_end.pt \\
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_local_128_stitch', seed=42,

    mask_nochain=True,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    curriculum=[
        dict(
            n_chunks=8, chunk_len=16, B=4, n_steps=80000, eval_every=10000,
            traj_mix=[
                # pure stitch only: 100% of steps on all-7-windows
                # goal: establish strong 7-window stitch quality (target ≥60%)
                dict(type='ir_local', weight=1.0,
                     windows=[(0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8)], n_refine=2),
            ],
            eval_traj='ir_local',
        ),
    ],
)
