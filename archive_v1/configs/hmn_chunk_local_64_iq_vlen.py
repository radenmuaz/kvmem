"""
Fallback: IQ-only multi-window vlen (no IR refinement).

Use if vlen (IQ+IR) fails to achieve win C nc=4 independence.

Assumption: model must remember one-shot with no refinement.
Prove IQ-only stitch + independence first, then add IR on top.

Success bar before adding IR:
  - All windows nc=4 independent >= 60%
  - Stitch nc=4 >= 60%
Only then run hmn_chunk_local_64_vlen.py (n_refine=2) on top of this checkpoint.

From: stage 1 IQ checkpoint (logs/hmn_chunk_local_32/checkpoints/stage0_end.pt, 81.9%)
NOT stage 2 — stage 2 has IR training baked in, incompatible with IQ-only objective.

Sequences are 3x shorter than IQ+IR (no IR turns):
  stitch nc=4: 80 enc + 3x36 IQ = 188 tokens (vs 572 with n_refine=2)
  win nc=4 single: 80 enc + 36 IQ = 116 tokens
  win nc=8 single: 160 enc + 36 IQ = 196 tokens

Traj mix (eval order = traj_mix order):
  1. stitch nc=4      weight=2.0  SLOT@80/244/408
  2. win A nc=2       weight=1.0  SLOT@40   (independent eval position)
  3. win A nc=4       weight=0.5  SLOT@80
  4. win A nc=8       weight=0.5  SLOT@160  (bridge)
  5. win B nc=4       weight=1.0  SLOT@80   (independent eval position)
  6. win B nc=8       weight=0.5  SLOT@160  (bridge)
  7. win C nc=4       weight=1.0  SLOT@80   (independent eval position)
  8. win C nc=8       weight=0.5  SLOT@160  (bridge)

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_local_64_iq_vlen.py \\
        --pretrained logs/hmn_chunk_local_32/checkpoints/stage0_end.pt \\
        --device mps 2>/dev/null >> logs/hmn_chunk_local_64_iq_vlen/train.log &
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_local_64_iq_vlen', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    mask_nochain=True,

    curriculum=[
        dict(
            n_chunks=8, chunk_len=16, B=8, n_steps=80000, eval_every=10000,
            traj_mix=[
                dict(type='ir_local', weight=2.0,
                     windows=[(0,2),(1,3),(2,4)], n_refine=0, n_chunks=4),

                dict(type='ir_local', weight=1.0, windows=[(0,2)], n_refine=0, n_chunks=2),
                dict(type='ir_local', weight=0.5, windows=[(0,2)], n_refine=0, n_chunks=4),
                dict(type='ir_local', weight=0.5, windows=[(0,2)], n_refine=0, n_chunks=8),

                dict(type='ir_local', weight=1.0, windows=[(1,3)], n_refine=0, n_chunks=4),
                dict(type='ir_local', weight=0.5, windows=[(1,3)], n_refine=0, n_chunks=8),

                dict(type='ir_local', weight=1.0, windows=[(2,4)], n_refine=0, n_chunks=4),
                dict(type='ir_local', weight=0.5, windows=[(2,4)], n_refine=0, n_chunks=8),
            ],
            eval_traj='ir_local',
        ),
    ],
)
