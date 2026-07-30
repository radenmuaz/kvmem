"""
`hmn_tpu_sanity_w25_ablate_1bucket.py` — DIAGNOSTIC, narrower than
`hmn_tpu_sanity_w25_ablate.py`. That config (8 buckets, 17 entries, B=4096
then B=64, rope=True/state_vocab_size=2) still hit loss=NaN by step 19 in
both batch sizes — ruling out batch size as the cause, leaving one untested
variable: that run mixes 8 DIFFERENT bucket shapes, each triggering its own
XLA compile mid-run, interleaved with already-compiled steps. This config
isolates to exactly ONE padded bucket (Lb=21, the same one inspected in the
local CPU repro that never reproduced NaN: 3 entries with real L=21/19/21 —
`E(8) Q(0,1,0,2) B8`, `E(8) Q(0,1,2,2) B8`, `E(8) Q(0,1,0,3) B8`) — testing
whether real padding ALONE (no other buckets, no dynamic mid-run recompiles)
reproduces the NaN on actual XLA hardware, or whether multi-bucket compile
switching is a necessary ingredient.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_tpu_sanity_w25_ablate_1bucket.py --device tpu
"""

hp = dict(
    d=128, n_layers=16, n_heads=8, V=271,
    block_type='single_attn',
    lr_max=6e-4, wd=1e-5,
    warmup_steps=200, log_every=5,
    rope=True,
    null_kv=True,
    rmsnorm=True,
    grad_checkpoint='block',  # restored — already ruled out as bug 5's cause
    no_autocast=True,  # forces fp32 on TPU — testing bf16-on-XLA as the actual trigger
    name='hmn_tpu_sanity_w25_ablate_1bucket', seed=48,

    state_len=4, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    bucket_lengths=True,
    max_shape_buckets=8,
    token_budget=4_000_000,
    attn_sq_budget=500_000_000,

    curriculum=[
        dict(n_chunks=1, chunk_len=8, B=64, n_steps=100, eval_every=100,
             weave_mix=[
                 dict(weight=1.0, dsl='E(8) Q(0,1,0,2) B8'),  # real L=21
                 dict(weight=1.0, dsl='E(8) Q(0,1,2,2) B8'),  # real L=19 -> padded to 21
                 dict(weight=1.0, dsl='E(8) Q(0,1,0,3) B8'),  # real L=21
             ]),
    ],
)
