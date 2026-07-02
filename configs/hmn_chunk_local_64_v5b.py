"""
Stage 3 v5b: 64B, 3 windows. Fine-tune per-window independence from v5 end.

v5 result: stitch=89.5%, win0=99.5%, win1=0%, win2=13% independent.
Stitch works because win1/win2 encode correctly at their STITCH positions.
Independent eval fails because win1/win2's IQ SLOT is at a different absolute
position (RoPE) when evaluated without window 0 preceding them.

Fix: add single-window trajectories for win1 and win2. These teach the model
to encode at the shorter (independent) positions too. Safe this time because:
- v5's corrected mask already enforces independence in stitch (no chaining)
- Adding singles doesn't conflict with stitch — both use enc-SLOT-only encoding
- Same mix as v3 (stitch×3 + win1×1 + win2×1) but v3 failed due to wrong mask

Win0 skipped — already independent, no position ambiguity (no prior windows).

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_local_64_v5b.py \\
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
    name='hmn_chunk_local_64_v5b', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    mask_nochain=True,

    curriculum=[
        dict(
            n_chunks=4, chunk_len=16, B=8, n_steps=80000, eval_every=10000,
            traj_mix=[
                # stitch (weight=3.0): 60% of steps — maintain stitch quality
                dict(type='ir_local', weight=3.0,
                     windows=[(0,2),(1,3),(2,4)], n_refine=2),
                # win1 independence (weight=1.0): 20% — teaches encoding at short position
                dict(type='ir_local', weight=1.0, windows=[(1,3)], n_refine=2),
                # win2 independence (weight=1.0): 20% — same
                dict(type='ir_local', weight=1.0, windows=[(2,4)], n_refine=2),
            ],
            eval_traj='ir_local',
        ),
    ],
)
