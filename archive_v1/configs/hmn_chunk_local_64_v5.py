"""
Stage 3 v5: 64B, 3 windows, mask_nochain=True (corrected — full prior rec_block blackout).

v4 failure analysis:
  mask_nochain in v4 only blocked IQ SLOT rows from prior rec_block SLOT tokens.
  The model chained through prior OUTPUT tokens instead: window 1's IQ SLOT read
  window 0's recalled output bytes 16-31 (the 50% overlap region), making
  independent per-window recall impossible despite the partial block.
  Result: win1=0%, win2=12.5% independent; stitch=54% (only works because window 0
  fills in the overlap region, not because window 1 encodes independently).

v5 fix:
  Rule 3b now blocks IQ SLOT rows from ALL tokens in prior rec_blocks:
  SLOT, warmup, argmax, AND output. Every window is forced to encode from
  enc-block SLOTs only. Chaining through output tokens is architecturally impossible.

Start: stage2 end (87.5% single-window) — clean slate, no wrong chaining to unlearn.

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_local_64_v5.py \\
        --pretrained logs/hmn_chunk_local_32_stage1/checkpoints/stage0_end.pt \\
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_local_64_v5', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    mask_nochain=True,  # v5: blocks ALL prior rec_block tokens (SLOT+warmup+output)

    curriculum=[
        dict(
            n_chunks=4, chunk_len=16, B=8, n_steps=80000, eval_every=10000,
            traj_mix=[
                dict(type='ir_local', weight=1.0,
                     windows=[(0,2),(1,3),(2,4)], n_refine=2),
            ],
            eval_traj='ir_local',
        ),
    ],
)
