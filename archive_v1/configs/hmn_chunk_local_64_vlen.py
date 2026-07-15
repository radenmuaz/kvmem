"""
Stage 3 vlen: 64B, variable-context fixed-window recall.

Problem addressed:
  v5 achieves stitch=91.3% but win1=0%, win2=0% in independent eval.
  v5b (singles training) improves win1 to ~33% by step 20k but win2 stays at 0%.
  Root cause: the model learned RoPE-position-dependent encoding — win1's IQ SLOT
  at stitch position ~244 differs from independent position ~80; win2 is worse.

Fix (vlen approach):
  Train the same 32B window recall from variable-length source contexts.
  For window (1,3), train with n_chunks in {4, 8}: IQ SLOT lands at ~80 and ~160.
  The model must learn to encode from enc_block SLOTs regardless of absolute position.
  Independent eval (nc=2 for win0, nc=4 for win1/win2) becomes part of the training
  distribution — no position OOD.

Key: per-traj n_chunks overrides the curriculum-level n_chunks.
Each batch uses ONE traj's n_chunks → fixed sequence length per batch.
Output is always one 32B window (warmup=8 + out=24).

From: stage2 end (87.5% single-window) — clean slate preferred over v5b
      (v5b may have partially learned position-dependent stitch patterns that
       conflict with vlen's position-invariance goal).

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_local_64_vlen.py \\
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
    name='hmn_chunk_local_64_vlen', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    mask_nochain=True,

    curriculum=[
        dict(
            # n_chunks=8 is the max context used by any traj (for val seq generation)
            n_chunks=8, chunk_len=16, B=8, n_steps=80000, eval_every=10000,
            traj_mix=[
                # Stitch (nc=4, all-3-windows): maintain chaining quality
                dict(type='ir_local', weight=2.0,
                     windows=[(0,2),(1,3),(2,4)], n_refine=2, n_chunks=4),

                # win(0,2) at nc=2: IQ SLOT@40 — this IS the independent eval position
                dict(type='ir_local', weight=1.0, windows=[(0,2)], n_refine=2, n_chunks=2),
                # win(0,2) at nc=4: IQ SLOT@80 — also seen in stitch
                dict(type='ir_local', weight=0.5, windows=[(0,2)], n_refine=2, n_chunks=4),
                # win(0,2) at nc=8: IQ SLOT@160 — extra position diversity
                dict(type='ir_local', weight=0.5, windows=[(0,2)], n_refine=2, n_chunks=8),

                # win(1,3) at nc=4: IQ SLOT@80 — the independent eval position
                dict(type='ir_local', weight=1.0, windows=[(1,3)], n_refine=2, n_chunks=4),
                # win(1,3) at nc=8: IQ SLOT@160 — seen in larger stitch
                dict(type='ir_local', weight=0.5, windows=[(1,3)], n_refine=2, n_chunks=8),

                # win(2,4) at nc=4: IQ SLOT@80 — the independent eval position
                dict(type='ir_local', weight=1.0, windows=[(2,4)], n_refine=2, n_chunks=4),
                # win(2,4) at nc=8: IQ SLOT@160 — seen in larger stitch
                dict(type='ir_local', weight=0.5, windows=[(2,4)], n_refine=2, n_chunks=8),
            ],
            eval_traj='ir_local',
        ),
    ],
)
