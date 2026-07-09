"""
DenseNet-KV ablation vs chat-tags Phase B4 baseline.

Same recipe as experiments/chat_tags/configs/slot8_tagged_phaseB4_windowtags.py
exactly (same dims, same traj_mix with window-specific query tags, same single
clean cosine decay, same step budget) — the ONLY variable changed is the model
architecture: DenseSlotKVModel (experiments/densenet_kv/model.py) lets each
layer's SLOT-position KV accumulate across depth (layer i+1 attends to layers
1..i's SLOT KV concatenated, not just its own), vs the standard single-layer
attention every other position (and the whole B4 model) uses.

Trained from scratch (different weight connectivity from B4 — DenseSlotKVAttention
has the same parameter SHAPES as standard MHAttention per layer, but the forward
computation graph differs enough that transplanting B4's weights isn't a like-for-
like warm start; a fresh run keeps the comparison clean).

Comparison metric requested: convergence SPEED (steps to reach a given Win C
match%), not just final ceiling — same eval protocol/schedule as B4 (eval_every=5000)
so per-checkpoint trajectories are directly comparable.

See docs/SRS_RECIPE.md § direction 6 for the full design rationale.

Run:
    caffeinate -i python3 -m experiments.densenet_kv.train \\
        --config experiments/densenet_kv/configs/slot8_densekv_windowtags.py \\
        --device mps
    tail -f experiments/densenet_kv/logs/densenet_kv_slot8_windowtags/train.log
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256, V=282,
    lr_max=2e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=80000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True,
    name='densenet_kv_slot8_windowtags', seed=46,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,

    curriculum=[
        dict(n_chunks=4, chunk_len=16, window_chunks=2, B=8, n_steps=80000, eval_every=5000,
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
