"""
Stage `squeeze` — dedicated compression-capacity experiment, deliberately
sized to land INSIDE the "sweet spot" window (docs/HISTORY.md §10, and the
2026-07-15 design discussion): the only chunk_len range where a success/
failure result actually means something, because it's the only range where
success is both REQUIRED to involve real compression (KV-cache smaller than
the raw file, so trivial byte-for-byte copying cannot fit) AND still
INFORMATION-THEORETICALLY POSSIBLE (KV-cache bigger than the data's true
Shannon content, so exact reconstruction isn't mathematically ruled out).
Below the window, success is trivial (raw copying fits). Above it, success
is impossible for ANY algorithm, not just this one (Shannon's source coding
theorem — you cannot losslessly represent H bits of entropy in fewer than H
bits, regardless of how good the compressor is).

**This config deliberately does NOT aim for the tight edge of that window
(near the impossibility boundary) — the goal here is "must not trivial
copy," not "best achievable compression rate."** Comfortable headroom
against the impossibility bound was chosen on purpose, so an imperfect (but
genuine) compression algorithm still has room to succeed.

**Exact accounting** (state_len=2, d=8, n_layers=4, chunk_len=1024,
data_target_bits=2.0 — all exact via gen_markov's closed-form entropy
calibration, entropy_tol=0.02 bits/byte):
    raw_bits    = chunk_len * 8         = 8192 bits (the uncompressed file)
    true_bits   = chunk_len * 2.0       = 2048 bits (the Shannon-optimal
                                           minimum any lossless method could
                                           ever achieve)
    kv_bits     = state_len * n_layers * 2(K,V) * d * 32(fp32)
                = 2 * 4 * 2 * 8 * 32    = 4096 bits (the full KV-cache view
                                           of STATE's nominal capacity — see
                                           nominal_capacity_accounting in
                                           kvmem/eval_compression.py)

    true_bits (2048) < kv_bits (4096) < raw_bits (8192)  <-  the sweet spot
    kv_bits / true_bits = 2.0x   (comfortable headroom, NOT the tight edge)
    kv_bits / raw_bits  = 0.5x   (STATE must beat 2x compression to succeed
                                   at all — trivial copying is impossible)

Contrast with the earlier hmn_squeeze_markov_n4.py (chunk_len=96,
state_len=8, d=64): there, kv_bits/true_bits was 682.7x — so much slack
that even a non-compressing model could nominally "fit," making any
success/failure result uninformative about genuine compression. THIS
config is the first squeeze variant where the capacity argument is actually
load-bearing by construction, not just checked after the fact.

**n_layers=4 kept (not shrunk further)** deliberately, per the earlier
depth-tension discussion: fewer layers would tighten the KV-cache-view
ceiling further (good for rigor) but risks not having enough computational
depth to reach even a comfortable target (bad — conflates "STATE too small"
with "model too shallow to learn the algorithm at all"). d=8/state_len=2
already does the "shrink the bottleneck" work; depth is left at a level
matched to the rest of this project's squeeze configs for comparability.

**Measured throughput** (this exact architecture, chunk_len=1024,
L=2058, B=6, MPS, dense attention chunk_attn=0): 2.373 it/s -> the full
60000-step budget is ~7.0 hours — tractable in one sitting, unlike the
chunk_len=4096 attempt (measured at 0.009 it/s / ~80 days at B=1, and
out-of-memory at B>=2 even with chunked attention). n_params=5,304
(fp32: 169,728 bits, ~20.7 KB) — smaller than any other config in this
project, by design (the whole point is state_len/d, not raw model size).

Paired with hmn_squeeze_random_n4.py (data_kind='random') as the control —
same reasoning as every other squeeze config: a high match% here alone
proves nothing, the GAP between this run and the random control (evaluated
at the SAME chunk_len/state_len/d) is the actual compression evidence. The
random control was NOT resized for this experiment (still chunk_len=96) —
running it at chunk_len=1024 too would be needed for a apples-to-apples
comparison at this specific capacity point; flagged as a follow-up, not
built here (this file is the compressible arm only).

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_squeeze_sweetspot_n4.py --device mps

Verify with (Diagnostic 0 reports the exact bit accounting above, computed
from this checkpoint's own hp/model rather than hand-typed):
    python3 -m kvmem.eval_compression --ckpt kvmem/logs/hmn_squeeze_sweetspot_n4/checkpoints/stage0_best.pt --device mps --kinds markov
"""

hp = dict(
    d=8, n_layers=4, n_heads=2, V=274,
    block_type='single_attn',
    lr_max=1.5e-4, lr_min=1e-6, wd=0.001,
    warmup_steps=500, log_every=1000,
    lr_schedule='cosine_restarts',
    cosine_T0=60000, cosine_T_mult=1,
    rope=True, yarn=True, null_kv=True,
    rmsnorm=True,
    name='hmn_squeeze_sweetspot_n4', seed=48,

    state_len=2, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    data_kind='markov',
    data_target_bits=2.0,

    curriculum=[
        dict(n_chunks=1, chunk_len=1024, n_refine=0, B=6, n_steps=60000, eval_every=5000,
             chain_steps=[(0, 1)]),
    ],
)
