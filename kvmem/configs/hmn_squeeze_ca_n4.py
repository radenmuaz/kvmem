"""
Stage `squeeze` — dedicated compression-capacity experiment, structured-data
arm (per docs/HISTORY.md §10). Paired with hmn_squeeze_random_n4.py (the
control, identical config except data_kind='random') — a high match% here
alone proves nothing; the gap between this run and the control is the actual
compression evidence.

Single-register layout (n_chunks=1, chain_steps=[(0,1)]) isolates the
capacity question to exactly one encoding-block STATE.

chunk_len=96 (CORRECTED, was 32 — measured, not assumed): the original
chunk_len=32 (2x solo's proven ~128-bit near-ceiling length) was chosen
expecting it to strain raw capacity for random bytes, but
hmn_squeeze_random_n4.py's actual run DISPROVED that assumption —
converged to ~100% val, loss~0 by step 70000/160000 on pure random
(genuinely incompressible) 256-bit content, at n_layers=4. Since squeeze's
whole point is "does structured data (this config) beat the random
control at the SAME chunk_len," a control that already saturates near 100%
leaves no room for this run to show an advantage — the comparison becomes
uninformative exactly where it needs to be decisive. chunk_len=96 (768
bits, 3x the point that just saturated, 6x solo's original calibration) is
a first escalation estimate, not a precisely-derived number — escalate
further (or reduce, if this overshoots into near-0% territory) based on
what hmn_squeeze_random_n4.py's rerun actually shows before trusting this
run's result. target_bits=2.0 stays a PER-BYTE quantity independent of
chunk_len, so it's unchanged — at chunk_len=96 that's 192 bits of true
information vs. random's 768 bits, a 4x theoretical compression ceiling
either way.

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
        dict(n_chunks=1, chunk_len=96, n_refine=0, B=6, n_steps=60000, eval_every=5000,
             chain_steps=[(0, 1)]),
    ],
)
