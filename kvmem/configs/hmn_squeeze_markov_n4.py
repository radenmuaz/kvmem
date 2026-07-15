"""
Stage `squeeze` — dedicated compression-capacity experiment, structured-data
arm (per docs/HISTORY.md §10). Paired with hmn_squeeze_random_n4.py (the
control, identical config except data_kind='random') — a high match% here
alone proves nothing; the gap between this run and the control is the actual
compression evidence.

**Switched from data_kind='ca' to data_kind='markov'** (this file replaces
the earlier hmn_squeeze_ca_n4.py, never trained, so nothing is lost by the
swap). Reason: `gen_ca`'s target_bits calibration is zlib measure-and-search
— seed-dependent and imprecise (flagged in its own docstring). `gen_markov`'s
is an EXACT closed-form bisection against the true stationary-distribution
entropy rate (`kvmem/structured_data.py:gen_markov`, `entropy_tol=0.02`
bits/byte) — no zlib involved at all. For a "how much did the model actually
compress" claim to mean anything, the TRUE entropy of the input has to be a
trustworthy number, not an estimate with its own unknown error bar stacked
on top of the model's measured bits/byte. `gen_markov` is exactly what
`kvmem/structured_data.py`'s own docstring recommends "when PRECISE
target_bits calibration matters more than generator diversity" — which is
the entire point of this experiment.

**Compression-rate accounting — computed in code, not hand-typed**:
`kvmem.eval_compression.nominal_capacity_accounting(model, hp)` derives
input/STATE/model-weight sizes in bits directly from hp/the built model
(chunk_len, state_len, d, n_layers, n_params, target_bits) — it can't drift
out of sync with this config the way a hand-written number in a docstring
can. Also wired into `eval_compression.py`'s CLI as "Diagnostic 0" (runs
first, pure arithmetic, no forward pass needed):
    python3 -c "
    from kvmem.eval_compression import nominal_capacity_accounting
    from kvmem.hmn import build_model, load_config
    import torch
    hp = load_config('kvmem/configs/hmn_squeeze_markov_n4.py')
    hp_model = dict(V=hp['V'], d=hp['d'], n_layers=hp['n_layers'], n_heads=hp['n_heads'],
                    block_type=hp['block_type'], rope=hp['rope'], yarn=hp['yarn'],
                    null_kv=hp['null_kv'], rmsnorm=hp['rmsnorm'])
    model = build_model(hp_model, torch.device('cpu'))
    print(nominal_capacity_accounting(model, hp))"

At this config's exact values (chunk_len=96, state_len=8, d=64, n_layers=4,
n_params=99,776, target_bits=2.0), that call reports: input_raw=768 bits,
input_true=192 bits (exact — `gen_markov`'s closed-form entropy, see
above), STATE residual-view=16,384 bits (21.3x raw / 85.3x true), STATE
full-KV-cache-view=131,072 bits (170.7x raw / 682.7x true), MODEL
weights=3,192,832 bits (4,157x raw / 16,629x true).

**What these numbers mean — and don't**: every nominal fp32 ceiling here
vastly exceeds even the RAW (uncompressed) input, let alone the true
content — so nominal bit-counting only proves "physically possible," never
"actually used this way," at any level (STATE or weights). This is why the
earlier `chunk_len=32->96` correction (see `hmn_squeeze_random_n4.py`) had
to be found EMPIRICALLY (watch the control run's actual match%/loss) rather
than derived from a capacity calculation — no nominal-bit argument would
predict chunk_len=32 saturating while chunk_len=96 doesn't, since both are
far below every ceiling above. The ~16,600x weight-capacity headroom over
true content is exactly the confound this experiment has to rule out:
`gen_markov`'s "fresh transition matrix every call" discipline
(`kvmem/structured_data.py`'s "fresh parameters per call" requirement,
unchanged from the earlier `gen_ca` version) means the generating rule is
NEVER repeated across training examples, so the model's FFN/attention
WEIGHTS structurally cannot memorize "the" Markov chain the way they could
if a fixed rule were reused — the only channel that varies per-example is
STATE, forcing genuine per-example compression by construction, not just
checked for after the fact. `state_ablation_gate` (`eval_compression.py`,
Diagnostic 1) is still the diagnostic that verifies this empirically
(measures the ACTUAL bits/byte gap when STATE is ablated — the effective-
capacity number nominal fp32 counts cannot substitute for); this section is
the reason to expect it to pass, not a replacement for running it.

Single-register layout (n_chunks=1, chain_steps=[(0,1)]) isolates the
capacity question to exactly one encoding-block STATE.

chunk_len=96 (unchanged from the corrected hmn_squeeze_ca_n4.py value — see
hmn_squeeze_random_n4.py's docstring for the full chunk_len=32->96
correction history): chosen so the random control shows genuine capacity
pressure (not saturated near 100%), which is the precondition for this run's
comparison against it to be informative at all.

Run (only after hmn_squeeze_random_n4.py's control run finishes — never two
jobs at once):
    python3 -m kvmem.hmn --config kvmem/configs/hmn_squeeze_markov_n4.py --device mps

Verify with:
    python3 -m kvmem.eval_compression --ckpt kvmem/logs/hmn_squeeze_markov_n4/checkpoints/stage0_best.pt --device mps --kinds markov
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
    name='hmn_squeeze_markov_n4', seed=48,

    state_len=8, state_vocab_size=2,
    warmup_len=8,
    use_actual_argmax=True,
    wrong_token_weight=0.0,
    val_n_seqs=3,

    data_kind='markov',
    data_target_bits=2.0,

    curriculum=[
        dict(n_chunks=1, chunk_len=96, n_refine=0, B=6, n_steps=60000, eval_every=5000,
             chain_steps=[(0, 1)]),
    ],
)
