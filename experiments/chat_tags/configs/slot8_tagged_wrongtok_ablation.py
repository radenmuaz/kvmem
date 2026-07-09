"""
Ablation 1/4 of the IR-refinement loss redesign queue (see docs/FEEDBACK_RESULTS.md
§ IR-refinement loss redesign): wrong-token-weighted loss.

Motivation: B4's IR turns sometimes DEGRADE quality rather than correct it (e.g.
up_counter: IQ=100% -> IR1=75% -> IR2=25%). Current loss is pure per-position NLL,
identical weight whether the fed-back argmax at that position was already correct
("leave alone" — easy) or wrong ("actively fix" — the actual correction task).
Diffuses gradient away from where it's needed.

Fix: w_i = 1 + alpha * 1[argmax_i != gt_i], applied to each IR block's own output
NLL at the position aligned with where its own fed-back argmax was wrong. No
architecture change — same model as B4, same tags, same traj_mix. Warm-started
from B4's best checkpoint (same architecture = no confound, unlike the DenseNet-KV
comparison which had a from-scratch vs warm-started mismatch).

alpha=2.0 chosen as a moderate first value (wrong positions get 3x the gradient
weight of already-correct positions) — not yet ablated across alpha values.

Run:
    caffeinate -i python3 -m experiments.chat_tags.train \\
        --config experiments/chat_tags/configs/slot8_tagged_wrongtok_ablation.py \\
        --pretrained experiments/chat_tags/logs/chat_tags_slot8_phaseB4_windowtags/checkpoints/stage0_best.pt \\
        --device mps
    tail -f experiments/chat_tags/logs/chat_tags_slot8_wrongtok_ablation/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256, V=282,
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=60000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='chat_tags_slot8_wrongtok_ablation', seed=47,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=2.0,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=4, chunk_len=16, window_chunks=2, B=8, n_steps=60000, eval_every=5000,
            traj_mix=[
                dict(weight=1.0, n_refine=2),
                dict(weight=0.5, n_refine=0),
                dict(weight=2.0, n_refine=2, warmup_x_fixed=0),
                dict(weight=1.0, n_refine=0, warmup_x_fixed=0),
                dict(weight=1.0, n_refine=2, warmup_x_fixed=16),
                dict(weight=0.5, n_refine=0, warmup_x_fixed=16),
                dict(weight=1.0, n_refine=2, warmup_x_fixed=32),
                dict(weight=0.5, n_refine=0, warmup_x_fixed=32),
            ]),
    ],
)
