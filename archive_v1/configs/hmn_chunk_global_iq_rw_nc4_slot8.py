"""
Global IQ random-window — slot_len=8 (double SLOT capacity), trained from scratch.

Motivation: slot_len=4 has a hard 8/24-byte structural ceiling confirmed across
all training runs (ext 100k, ext2 300k, s1_restart 200k, vlen_init 50k).
Root cause: the 4-token IQ SLOT can encode within-chunk bytes (chunk 1 tail, 8B)
but lacks capacity for cross-chunk transitions (chunk 2, 16B).

slot_len=8 doubles the capacity at both enc level (per-chunk: 8 SLOT tokens per
16B chunk → 2 tok/byte vs 0.25 tok/byte at slot_len=4) and IQ level (global IQ
SLOT: 8 tokens to compress 2 chunks instead of 4).

Sequence layout (nc=4, chunk_len=16, slot_len=8):
  enc_block[k]: src[16k:16k+16] (16 tok) + SLOT×8 (8 tok) = 24 tok
  enc_end = 96
  IQ recall: SLOT×8 (pos 96-103) | warmup×8 (104-111) | out×24 (112-135)
  Total L = 136

Must train from scratch — different token layout vs slot_len=4 checkpoints.

Traj mix:
| weight | nc | warmup offsets (train) | warmup offsets (eval) | SLOT pos |
|--------|----|------------------------|----------------------|----------|
|   1.0  |  4 | uniform [0, 32]        | {0, 16, 32}          |       96 |

Run:
    caffeinate -i python3 -m kvmem.train_hmn_chunk \\
        --config configs/hmn_chunk_global_iq_rw_nc4_slot8.py \\
        --device mps
    tail -f logs/hmn_chunk_global_iq_rw_nc4_slot8/train.log
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
    name='hmn_chunk_global_iq_rw_nc4_slot8', seed=42,

    slot_len=8, slot_count=2,
    warmup_len=8,
    use_actual_argmax=False,
    val_n_seqs=3,
    mask_nochain=False,

    curriculum=[
        dict(
            n_chunks=4, chunk_len=16, B=8, n_steps=100000, eval_every=5000,
            traj_mix=[
                dict(type='iq_global_rw', weight=1.0, n_chunks=4, window_chunks=2),
            ],
            eval_traj='iq_global_rw',
        ),
    ],
)
