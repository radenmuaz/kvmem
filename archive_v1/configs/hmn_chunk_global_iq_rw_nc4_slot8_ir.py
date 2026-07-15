"""
IR refinement stage on top of global IQ slot8 — target: Win A high recall.

Motivation: slot8_ext IQ-only achieved Win B/C high recall but Win A stagnated
at BPB~1.3 (up), 1.9 (odd), 2.6 (down). IR turns proved critical for 32B recall
(hmn_feedback_32_ir lifted 0%→100%). Adding n_refine=2 IR turns lets the model
refine its initial IQ output across all 3 windows, including the hard Win A.

Architecture (nc=4, slot_len=8, warmup_len=8, n_refine=2):
  enc_end = 96
  IQ:  SLOT×8 | warmup×8 | out×24     (pos 96-135,  L=40)
  IR1: SLOT_A×8 | am×24 | SLOT_B×8 | warmup×8 | out×24   (pos 136-207, L=72)
  IR2: SLOT_A×8 | am×24 | SLOT_B×8 | warmup×8 | out×24   (pos 208-279, L=72)
  Total L = 280

Random warmup offset X shared across IQ + IR turns per example (same window).
Eval: fixed chunk-aligned windows {0,16,32} as before.

From: logs/hmn_chunk_global_iq_rw_nc4_slot8_ext/checkpoints/stage0_best.pt
      (step 45k, 44.0% val_mean — best IQ-only checkpoint)

Traj mix:
| weight | nc | n_refine | warmup offsets (train) | warmup offsets (eval) | SLOT pos |
|--------|----|----------|------------------------|----------------------|----------|
|   1.0  |  4 |    2     | uniform [0, 32]        | {0, 16, 32}          |       96 |

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_global_iq_rw_nc4_slot8_ir.py \\
        --pretrained logs/hmn_chunk_global_iq_rw_nc4_slot8_ext/checkpoints/stage0_best.pt \\
        --device mps
    tail -f logs/hmn_chunk_global_iq_rw_nc4_slot8_ir/train.log
"""

hp = dict(
    train_fn='fb',
    d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
    lr_max=3e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=2000, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=20000,
    cosine_T_mult=2,
    cosine_cycle_warmup=500,
    rope=True, yarn=True, null_kv=True, compile=False, chunk_attn=256,
    name='hmn_chunk_global_iq_rw_nc4_slot8_ir', seed=42,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=True,
    val_n_seqs=3,
    mask_nochain=False,

    curriculum=[
        dict(
            n_chunks=4, chunk_len=16, B=8, n_steps=100000, eval_every=5000,
            traj_mix=[
                dict(type='iq_global_rw_ir', weight=1.0, n_chunks=4,
                     window_chunks=2, n_refine=2),
            ],
            eval_traj='iq_global_rw',
        ),
    ],
)
