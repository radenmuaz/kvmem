"""
Curriculum stage 2 for chunk feedback-SRS — the actual target architecture
from CLAUDE.md (full depth-2 SRS, per-span local IQ+IR pair). Since stage 1
(hmn_chunk_curric) already rehearsed windowed refinement at both the half
(128B) and full (256B) span sizes, this stage is "generalize the proven
windowed-refine mechanism to the SRS-scheduled sequencing of multiple spans
in one pass" — not a new scale or window. Replays ir_winrefine and
iq_windowed to retain both prior skills.

Run (separate launch, loads stage 1's checkpoint):
    python -m kvmem.train_hmn_chunk \
        --config configs/hmn_chunk_srs_ir.py \
        --pretrained logs/hmn_chunk_curric/checkpoints/stage1_end.pt \
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_srs_ir', seed=42,

    slot_len=2, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,

    eval_file='datasets/suratalkauthar.txt',

    curriculum=[
        dict(n_chunks=2, chunk_len=128, depth=2, B=2, n_steps=20000, eval_every=10000,
             traj_mix=[dict(type='ir_srs', weight=0.6),
                       dict(type='ir_winrefine', weight=0.25),
                       dict(type='iq_windowed', weight=0.15)],
             eval_traj='ir_srs'),
    ],
)
