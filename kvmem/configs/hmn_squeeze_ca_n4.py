"""
Stage `squeeze` — dedicated compression-capacity experiment, structured-data
arm (per docs/HMN_RECIPE.md §10). Paired with hmn_squeeze_random_n4.py (the
control, identical config except data_kind='random') — a high match% here
alone proves nothing; the gap between this run and the control is the actual
compression evidence.

Single-register layout (n_chunks=1, chain_steps=[(0,1)]) isolates the
capacity question to exactly one encoding-block STATE. chunk_len=32 is 2x
Stage `solo`'s proven near-ceiling length (16 bytes at state_len=8 already
reaches ~94-97% on pure random bytes) — chosen to sit past where raw
memorization should fail for random bytes, while staying within the
theoretical 8/target_bits=4x compression ceiling for target_bits=2.0.

n_layers=4 (not solo/relay's 8) as a deliberate MDL-order choice (broaden
distribution -> simplify algorithm -> grow model size LAST) — starting
smaller reduces the risk that eval_compression.py's state_ablation_gate
would need to catch (weight-based memorization instead of genuine STATE
compression). Escalate to n_layers=6/8 only if this fails to reach
near-ceiling match%, not built preemptively. From scratch — n_layers=4
doesn't match solo/relay's state_dict shapes, no warm start possible.

Run (only after Stage `relay` finishes and the chain-memory recovery probe
runs — never two jobs at once; run hmn_squeeze_random_n4.py FIRST as the
simpler control):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_squeeze_ca_n4.py --device mps

Verify with:
    python3 -m kvmem.eval_compression --ckpt kvmem/logs/hmn_squeeze_ca_n4/checkpoints/stage0_best.pt --device mps --kinds ca
"""

hp = dict(
    d=64, n_layers=4, n_heads=4, V=274,
    block_type='single_attn',
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=160000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True,
    rmsnorm=True,
    name='hmn_squeeze_ca_n4', seed=48,

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    data_kind='ca',
    data_target_bits=2.0,

    curriculum=[
        dict(n_chunks=1, chunk_len=32, n_refine=0, B=6, n_steps=160000, eval_every=10000,
             chain_steps=[(0, 1)]),
    ],
)
