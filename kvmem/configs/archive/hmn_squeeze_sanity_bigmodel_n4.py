"""
Sanity check for `hmn_squeeze_sweetspot_n4.py` — IDENTICAL dataset (same
chunk_len=1024, n_chunks=1, chain_steps=[(0,1)], data_kind='markov',
data_target_bits=2.0, warmup_len=8) but a GENEROUS-capacity model
(state_len=8, d=8, n_layers=4, n_heads=2). Reuses `sweetspot`'s own tiny,
fast `d=8, n_layers=4` architecture (not `solo`'s d=64/n_layers=8 — that
version was measured too slow, 0.553 it/s / ~30 hours for 60000 steps, and
was cut down for exactly that reason) but with state_len bumped 2 -> 8, so
the only thing that changed relative to `sweetspot` is STATE width, not the
whole architecture or compute cost.

**Purpose**: isolate whether a failure/weak result on
hmn_squeeze_sweetspot_n4.py is really about the tight STATE bottleneck (the
thing that config is designed to test), or whether something else is
broken at this chunk_len/sequence length independent of capacity — e.g. a
masking bug, RoPE-length-generalization issue, or training instability
that only shows up at L~2000+ (nothing in this project has trained a
single-register recall task this long before; `solo`/`hop` both use much
shorter per-block sequences). This config gives the SAME task a comfortable
capacity margin (kv_bits/true_bits=8.0x, see `nominal_capacity_accounting`
in kvmem/eval_compression.py) — not `solo`-architecture-sized overkill
(128x), just enough headroom that STATE clearly isn't the binding
constraint, while keeping compute cost close to `sweetspot`'s own (same
d=8/n_layers=4, only state_len differs, and state_len barely affects L or
per-step cost). If THIS config also fails to recall well, the sweetspot
config's result can't be attributed to STATE being too small; something
else is confounding it and needs to be found before trusting either
config's outcome. If this config succeeds cleanly, it establishes a clean
reference ceiling to compare the sweetspot run against.

**Deliberately NOT a compression test** — 8x nominal headroom means a high
match% here says nothing about whether genuine compression is happening.
It's purely a "does basic recall work on this dataset shape at all" check.
Compare hmn_squeeze_sweetspot_n4.py's result against THIS run's ceiling,
not against 100%.

Measured throughput (this exact architecture, chunk_len=1024, L=2070, B=6,
MPS, dense attention): 2.770 it/s -> 60000 steps ~6.02 hours (5x faster
than the earlier solo-sized d=64/n_layers=8 version's ~30.1 hours, for the
same dataset and nearly the same capacity-margin purpose). Kept at
n_steps=60000 to match hmn_squeeze_sweetspot_n4.py's own budget exactly,
now that it's affordable, for a fair same-steps comparison. n_params: see
build output at run time (small — same architecture as sweetspot's
5,304-param model, just state_len 2->8).

Run (never two jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_squeeze_sanity_bigmodel_n4.py --device mps --log-dir kvmem/logs
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
    name='hmn_squeeze_sanity_bigmodel_n4', seed=48,

    state_len=8, state_vocab_size=2,
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
