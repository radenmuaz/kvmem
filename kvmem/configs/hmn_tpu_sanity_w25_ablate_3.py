"""
`hmn_tpu_sanity_w25_ablate_3.py` — the negative control for `hmn_tpu_sanity_
w25_ablate_2.py`. IDENTICAL 3 entries (real L=19/20/21), identical settings
(`rope=False, state_vocab_size=1, grad_checkpoint='block'`, bf16 autocast
enabled — the exact combination that NaN'd in ablate_2), but `max_shape_
buckets=3` instead of `1` — each of the 3 distinct real lengths gets its OWN
exact bucket (`_bucket_ceilings`'s `n <= max_buckets` early return), so NO
padding occurs at all. If this trains cleanly (finite loss) where ablate_2
NaN'd, padding — specifically, a bucket containing rows genuinely shorter
than its ceiling — is confirmed as bug 5's cause, for the first time with a
real positive/negative pair (every earlier attempt at this control
accidentally had zero padding on BOTH sides, see ablate_1bucket.py's own
docstring for that mistake).

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_tpu_sanity_w25_ablate_3.py --device tpu
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
    name='hmn_tpu_sanity_w25_ablate_3', seed=48,

    state_len=4, state_vocab_size=1,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    bucket_lengths=True,
    max_shape_buckets=3,  # each distinct L gets its own EXACT bucket -> zero padding
    token_budget=4_000_000,
    attn_sq_budget=500_000_000,

    curriculum=[
        dict(n_chunks=1, chunk_len=8, B=64, n_steps=100, eval_every=100,
             weave_mix=[
                 dict(weight=1.0, dsl='E(8) Q(0,1,0,2) B8'),  # real L=21
                 dict(weight=1.0, dsl='E(8) Q(0,1,1,2) B8'),  # real L=20
                 dict(weight=1.0, dsl='E(8) Q(0,1,2,2) B8'),  # real L=19
             ]),
    ],
)
