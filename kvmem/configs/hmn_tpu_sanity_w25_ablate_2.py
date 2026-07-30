"""
`hmn_tpu_sanity_w25_ablate_2.py` — DIAGNOSTIC, corrects a bucket-math error in
`hmn_tpu_sanity_w25_ablate_1bucket.py` (that config's 3 entries had only 2
DISTINCT real lengths, and with `max_shape_buckets=8 >= 2`,
`_bucket_ceilings`'s `n <= max_buckets` early-return gave each length its
own EXACT bucket — zero padding, despite the file's name/docstring claiming
otherwise). Every conclusion drawn from that config ("padding causes bug 5")
is therefore UNVERIFIED — what was actually tested was `rope=True,
state_vocab_size=2` on a 3-entry subset with NO padding, which also NaN'd,
so rope/vocab aren't it either, but padding was never actually exercised.

This config forces GENUINE padding: 3 entries with 3 DISTINCT real lengths
(L=19, L=20, L=21 — `Q(0,1,2,2)`, `Q(0,1,1,2)`, `Q(0,1,0,2)`), `max_shape_
buckets=1` so `_bucket_ceilings` is FORCED to merge all three into one
padded bucket (Lb=21, two rows genuinely padded) regardless of the n<=max_
buckets shortcut. Uses the ORIGINAL failing settings (`rope=False,
state_vocab_size=1`, matching `hmn_tpu_sanity_w25.py`'s actual stage 1) —
not the rope=True/vocab=2 ablation — to test the real, original padding
hypothesis cleanly for the first time.

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_tpu_sanity_w25_ablate_2.py --device tpu
"""

hp = dict(
    d=128, n_layers=16, n_heads=8, V=271,
    block_type='single_attn',
    lr_max=6e-4, wd=1e-5,
    warmup_steps=200, log_every=5,
    rope=False,
    null_kv=True,
    rmsnorm=True,
    grad_checkpoint='block',
    name='hmn_tpu_sanity_w25_ablate_2', seed=48,

    state_len=4, state_vocab_size=1,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    bucket_lengths=True,
    max_shape_buckets=1,  # FORCES all 3 distinct lengths into one padded bucket
    token_budget=4_000_000,
    attn_sq_budget=500_000_000,

    curriculum=[
        dict(n_chunks=1, chunk_len=8, B=64, n_steps=100, eval_every=100,
             weave_mix=[
                 dict(weight=1.0, dsl='E(8) Q(0,1,0,2) B8'),  # real L=21
                 dict(weight=1.0, dsl='E(8) Q(0,1,1,2) B8'),  # real L=20 -> padded to 21
                 dict(weight=1.0, dsl='E(8) Q(0,1,2,2) B8'),  # real L=19 -> padded to 21
             ]),
    ],
)
