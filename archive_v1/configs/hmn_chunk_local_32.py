"""
Local-refine curriculum, stages 1+2: fixed 32B src, single window, random init.

Stage 1: IQ only (n_refine=0), 32 in / 32 out, no feedback.
Stage 2: IQ + 2-step argmax-refine IR, single window = the proven
         hmn_feedback_32_ir mechanism exactly (100% match k=0..12 there).

n_chunks=2, chunk_len=16 -> 32B src, windows=[(0,2)] (the whole source,
one window, no overlap needed yet — overlap only matters once a stage
spans more than one 32B window, see hmn_chunk_local_64.py).

slot_len=4 and step counts (50k IQ / 80k IR) match the proven
hmn_feedback_32_iq/_ir recipe exactly — a first attempt at slot_len=2,
8k/8k steps undertrained badly (stage 1 eval BPB diverged from ~10 to ~19
while train loss kept falling, the classic premature-feedback-before-IQ-
is-solid failure CLAUDE.md warns about: "IQ pretraining required before
IR").

Run (random init):
    python -m kvmem.train_hmn_chunk \
        --config configs/hmn_chunk_local_32.py \
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_local_32', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=2, chunk_len=16, B=8, n_steps=50000, eval_every=5000,
             traj_mix=[dict(type='ir_local', weight=1.0, windows=[(0, 2)], n_refine=0)],
             eval_traj='ir_local'),
        dict(n_chunks=2, chunk_len=16, B=8, n_steps=80000, eval_every=10000,
             traj_mix=[dict(type='ir_local', weight=1.0, windows=[(0, 2)], n_refine=2)],
             eval_traj='ir_local'),
    ],
)
