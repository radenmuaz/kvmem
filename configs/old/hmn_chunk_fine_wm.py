"""
Config B with warmup: 8 chunks × 16→32 bytes, slot_len=1, warmup_len=8.
Matches feedback arch's warmup_len=8 / out_len=24 pattern.
Pretrained from the full IQ+IR checkpoint (100% single-sequence model).

warmup_len=8 means each recall block: [SLOT×1][warmup:8][out:span_len-8]
Total block length unchanged (slot_len + span_len), but output is 8 bytes shorter.
NTP loss on output positions only; warmup provided as ground-truth cue.

Run:
    python -m kvmem.train_hmn_chunk \
        --config configs/hmn_chunk_fine_wm.py \
        --pretrained logs/hmn_feedback_32_ir/checkpoints/stage0_end.pt \
        --device mps
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    eval_every=5000, log_every=500,
    rope=True, yarn=True, null_kv=True, compile=False,
    name='hmn_chunk_fine_wm', seed=42,

    slot_len=1, slot_count=2,
    warmup_len=8,
    ir_turns=2, noise_p=0.5,
    eval_file='datasets/suratalfatihah.txt',

    curriculum=[
        # Stage 0: IQ only (no SRS), warmup=8
        dict(n_chunks=8, chunk_len=16, use_srs=False, B=8, n_steps=50000),
        # Stage 1: Full SRS + warmup
        dict(n_chunks=8, chunk_len=16, use_srs=True,  B=8, n_steps=80000),
        # Stage 2: 32-byte chunks
        dict(n_chunks=8, chunk_len=32, use_srs=True,  B=8, n_steps=80000),
    ],
)
