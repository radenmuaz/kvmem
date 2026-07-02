"""
Stage 3 v4: 64B, 3 windows, mask_nochain=True.

Root cause of v1/v2/v3 failures: chunk_mask_fb only blocked IQ SLOT rows from
source chunks — NOT from prior rec_block SLOT tokens. Window 1's IQ SLOT could
freely attend to window 0's IQ SLOT (chaining), making independent per-window
recall structurally impossible regardless of training distribution.

Fix: mask_nochain=True adds Rule 3b — each IQ SLOT is also blocked from ALL
prior rec_block SLOT tokens. Every window is forced to encode from enc-block
SLOTs only. Chaining is architecturally prevented.

With this fix, pure stitch training should produce independently-encodable
windows by construction — no mixed training needed. All windows learn the
same task (encode my 32B span from enc-block SLOTs → slot → recall).

Training: pure stitch only (all-3-windows). No singles needed — independence
is enforced by the mask, not the training distribution.

From stage 2 end (single 32B window, 87.5%) — the nochain mask has no effect
on single-window training, so the pretrained weights are fully compatible.

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_local_64_v4.py \\
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
    name='hmn_chunk_local_64_v4', seed=42,

    slot_len=4, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    mask_nochain=True,

    curriculum=[
        dict(
            n_chunks=4, chunk_len=16, B=8, n_steps=80000, eval_every=10000,
            traj_mix=[
                # Pure stitch only — independence is enforced by mask_nochain,
                # not by per-window single training. No mixed training needed.
                dict(type='ir_local', weight=1.0,
                     windows=[(0,2),(1,3),(2,4)], n_refine=2),
            ],
            eval_traj='ir_local',
        ),
    ],
)
