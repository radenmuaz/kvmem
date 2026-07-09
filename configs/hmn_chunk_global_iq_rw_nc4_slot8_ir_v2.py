"""
IR + IQ quality + heavy Win A oversample.

Two problems diagnosed from slot8_ir per-turn eval:
  1. IQ=0% for all windows — IQ block has is_clean=False when n_refine>0, so IQ
     output never contributes to loss. Model learns to treat IQ as a pure compression
     stage and lets IR do all recall work.
  2. Win A stuck at ~3% — IQ encodes first 32B poorly; IR1 has nothing to work with.

Fix 1 (IQ quality): add iq_global_rw (n_refine=0) entries to traj_mix. Those steps
  have is_clean=True on the IQ block, forcing the model to maintain one-shot recall.

Fix 2 (Win A): heavily oversample warmup_x_fixed=0 (bytes 0-31). Both IQ-only and
  IR entries are oversampled so Win A gets loss signal in both training modes.

Traj mix (total weight 4.5):
| weight | type           | warmup       | share | purpose                    |
|--------|----------------|--------------|-------|----------------------------|
|   1.0  | iq_global_rw_ir | uniform      |  22%  | IR, all windows            |
|   0.5  | iq_global_rw    | uniform      |  11%  | IQ quality, all windows    |
|   2.0  | iq_global_rw_ir | X=0 (Win A)  |  44%  | heavy Win A IR             |
|   1.0  | iq_global_rw    | X=0 (Win A)  |  22%  | direct Win A IQ recall     |

Win A = 66% of all steps. IQ loss = 33% of all steps.

Eval uses first IR trajectory (n_refine=2) → per-turn logging shows IQ/IR1/IR2.

From: logs/hmn_chunk_global_iq_rw_nc4_slot8_ir/checkpoints/stage0_best.pt
      (step 50k, 51.9% — Win B 93%, Win C 60%, Win A 3%)

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_global_iq_rw_nc4_slot8_ir_v2.py \\
        --pretrained logs/hmn_chunk_global_iq_rw_nc4_slot8_ir/checkpoints/stage0_best.pt \\
        --device mps
    tail -f logs/hmn_chunk_global_iq_rw_nc4_slot8_ir_v2/train.log
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=20000,
    cosine_T_mult=2,
    cosine_cycle_warmup=500,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='hmn_chunk_global_iq_rw_nc4_slot8_ir_v2', seed=42,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,
    mask_nochain=False,

    curriculum=[
        dict(
            n_chunks=4, chunk_len=16, B=8, n_steps=100000, eval_every=5000,
            traj_mix=[
                # IR uniform — maintains Win B/C IR quality
                dict(type='iq_global_rw_ir', weight=1.0, n_chunks=4,
                     window_chunks=2, n_refine=2),
                # IQ-only uniform — forces IQ one-shot recall quality
                dict(type='iq_global_rw', weight=0.5, n_chunks=4,
                     window_chunks=2),
                # IR Win A heavy — teaches IR refinement for bytes 0-31
                dict(type='iq_global_rw_ir', weight=2.0, n_chunks=4,
                     window_chunks=2, n_refine=2, warmup_x_fixed=0),
                # IQ-only Win A heavy — direct one-shot recall for bytes 0-31
                dict(type='iq_global_rw', weight=1.0, n_chunks=4,
                     window_chunks=2, warmup_x_fixed=0),
            ],
            eval_traj='iq_global_rw',
        ),
    ],
)
