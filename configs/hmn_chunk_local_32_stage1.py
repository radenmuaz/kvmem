"""
Local-refine curriculum, stage 2 ONLY (IR refine), resuming from stage 1's
(IQ-only) checkpoint — relaunched after fixing a real masking bug in
chunk_mask_fb (kvmem/train_hmn_chunk.py).

Bug: IR turns' SLOT_A/argmax/SLOT_B rows were blocked from the raw encoding
chunks but NOT from earlier rec_blocks' own c0:c1 output region sitting in
the same concatenated sequence. During training that region is always
ground truth (teacher-forced), so the model could shortcut by attending
straight to it instead of routing through the intended argmax-copy
bottleneck (am0:am1) — train loss collapsed to ~0 while eval AR-decode
(where that region is the model's own greedy output, not GT) stayed at
0% match with BPB climbing 28->43 over stage 1. Compared against the
proven hmn_feedback_32_ir mechanism (train_hmn_feedback.py), which runs
each turn as a fully separate short sequence with no such leak possible.
Fixed by also blocking SLOT_A/argmax/SLOT_B from every rec_block's c0:c1.

Stage 1 (IQ-only, no IR blocks) is unaffected by the bug -- its checkpoint
is reused as-is rather than retraining from scratch.

Run (resume from stage 0's existing checkpoint):
    python -m kvmem.train_hmn_chunk \
        --config configs/hmn_chunk_local_32_stage1.py \
        --pretrained logs/hmn_chunk_local_32/checkpoints/stage0_end.pt \
        --device mps
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=2000,
    log_every=200,
    rope=True, yarn=True, null_kv=True, compile=False,
    chunk_attn=256,
    name='hmn_chunk_local_32_stage1', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=2, chunk_len=16, B=8, n_steps=80000, eval_every=10000,
             traj_mix=[dict(type='ir_local', weight=1.0, windows=[(0, 2)], n_refine=2)],
             eval_traj='ir_local'),
    ],
)
