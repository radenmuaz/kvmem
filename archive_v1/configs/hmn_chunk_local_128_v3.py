"""
Stage 4 v3: 128B src, 7 overlapping 32B windows.
Targeted independence fine-tuning after pure stitch (hmn_chunk_local_128_stitch.py).

Rationale:
  Phase A (128_stitch): pure stitch → strong 7-window stitch baseline established.
  Phase B (this config): add independence for windows 1..6 while keeping stitch dominant.

  win0=(0,2): first window, no prior window to chain from → already independent.
  win1=(1,3)..win6=(6,8): can attend to prior windows' SLOT tokens → may chain.

  Mix: stitch×3 + 6 problematic singles×1 each (33% stitch, ~11% each window 1..6).
  Skip win0 — it's already independent like v1's window 0.

Sequence lengths:
  all-7-windows (stitch): enc(8×20=160) + 7×164 = 1308 tokens
  single-window: enc(160) + 164 = 324 tokens
  B=4 → max 4×1308=5232 tokens/batch (safe for MPS)

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_local_128_v3.py \\
        --pretrained logs/hmn_chunk_local_128_stitch/checkpoints/stage0_end.pt \\
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_local_128_v3', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    curriculum=[
        dict(
            n_chunks=8, chunk_len=16, B=4, n_steps=80000, eval_every=10000,
            traj_mix=[
                # stitch (weight=3.0): 33% of steps — keep stitch as primary objective
                dict(type='ir_local', weight=3.0,
                     windows=[(0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8)], n_refine=2),
                # win1..win6 independence (weight=1.0 each): 11% each
                # skip win0=(0,2) — it's already independent (no prior window to chain from)
                dict(type='ir_local', weight=1.0, windows=[(1,3)], n_refine=2),
                dict(type='ir_local', weight=1.0, windows=[(2,4)], n_refine=2),
                dict(type='ir_local', weight=1.0, windows=[(3,5)], n_refine=2),
                dict(type='ir_local', weight=1.0, windows=[(4,6)], n_refine=2),
                dict(type='ir_local', weight=1.0, windows=[(5,7)], n_refine=2),
                dict(type='ir_local', weight=1.0, windows=[(6,8)], n_refine=2),
            ],
            eval_traj='ir_local',
        ),
    ],
)
