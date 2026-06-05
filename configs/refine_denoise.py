"""
Exp 3 — Refine Stage A: denoising self-correction.

Sequence per example:
  <x:16><z:7><h:1>          ← single encoding block
  <q:4><r:8>                 ← draft turn: noisy recall (no loss)
  <q:4><y:8>                 ← final turn: corrected recall (loss here)

The model sees a corrupted first attempt (<r>) and must produce a
better second attempt (<y>). Draft noise: p ~ U(0.05, 0.8).

Tags:
  <r>/<r>  = refinement/draft (internal chain-of-thought, not user-facing)
  <q>/<y>  = user-facing query/value (final output only)

Primary metric:
  val_ref_bpb  — NLL on final <y> given noisy <r> context
  val_bpb      — standard single-turn baseline (no correction context)
  gap          — val_bpb - val_ref_bpb: positive = correction helps

null_kv=True: always; 1.5-2x faster convergence.
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, d_ff=256,
    lr_max=3e-4, wd=0.001, warmup_steps=500,
    eval_every=2000, log_every=500,
    ocd=False, ocd_prob=0.0, tf_warmup=0,
    rope=True, yarn=True, grok=False,
    stablemax=False, eval_offset=0.25, grad_clip=10.0,
    null_kv=True,
    compile=False,
    name='refine_denoise',
    seed=42,
    curriculum=[dict(
        seg_len=16, slot_len=1, warmup_len=4, out_len=8,
        latent_len=7, mem_window=-1,
        n_blocks=1,
        recall_from=0,
        recall_froms=0,
        mode='ref',
        n_draft_turns=1,
        noise_schedule=[(0.05, 0.8)],
        B=16,
        n_steps=80000,
        dataset_size=-1,  # infinite stream
    )],
    eval_configs=[(1, 0)],
)
