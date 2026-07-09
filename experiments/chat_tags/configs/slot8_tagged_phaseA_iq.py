"""
Phase A — chat-tags plumbing sanity check.

Short IQ-only run (n_refine=0) on the iq_global_rw_tagged trajectory
(<src>/<mem>/<query>/<response> boundary tokens wrapping the untagged
iq_global_rw layout). Goal: confirm tokenization/mask/decode are correct —
teacher-forced convergence should look comparable to the untagged IQ-only
baseline (configs/hmn_chunk_global_iq_rw_nc4_slot8_ext.py, 44.0% best over
80k steps) within the first few thousand steps. Not a fair end-state
comparison — this is plumbing verification before committing to the full
staged Phase B run.

See /Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md.

Run:
    python3 -m experiments.chat_tags.train \\
        --config experiments/chat_tags/configs/slot8_tagged_phaseA_iq.py \\
        --device mps
    tail -f experiments/chat_tags/logs/chat_tags_slot8_phaseA_iq/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256,
    lr_max=3e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=500,
    lr_schedule='cosine_restarts',
    cosine_T0=8000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True, compile=False,
    name='chat_tags_slot8_phaseA_iq', seed=42,

    slot_len=8, slot_count=2,
    warmup_len=8,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=4, chunk_len=16, window_chunks=2, n_refine=0,
            B=8, n_steps=8000, eval_every=1000),
    ],
)
