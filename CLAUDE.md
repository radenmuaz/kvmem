# kvmem

Fast-weight language model — HashMemNet (HMN). **Current focus: `kvmem/hmn.py`**, a from-scratch consolidated single-file rewrite (single-attn blocks + shared chat-tag vocab + bounded cross-chain-step relay), replacing the prior multi-file `kvmem/`+`experiments/` stack. All prior code/docs/checkpoints are preserved verbatim under `archive_v1/` (old `kvmem/`, old `experiments/`, and `archive_v1/CLAUDE_v1.md` — the previous version of this file) — nothing was deleted, just superseded. `archive_v1/` code still runs standalone via `PYTHONPATH=archive_v1`.

**Why the rewrite**: a design review caught that the old chat-tag vocabulary assigned a separate tag token per window (`HMN_QUERY_A_OPEN`..`_G_OPEN`) — backwards for a chat-formatted design (a real LLM reuses the same role tokens every turn; turn identity comes from position, not a turn-numbered vocab entry). Fixing it required retraining anyway, so this was also the point to build in bounded, persistent chain memory from the start instead of bolting it on later. Full rationale, worked examples, and every original naming decision: [`/Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md`](/Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md) (the approved plan — read this first for historical *why*; some of its terminology, notably `STATE_QUEUE`/`h_inject`, describes a mechanism since superseded — see below).

---

## Terminology (read this before the code — "step" alone is retired, it was overloaded 4 ways)

| Concept | Term |
|---|---|
| token index within the packed sequence | **position** |
| schedule position in a chained schedule (old: "window A/B/C/G") | **chain step** (always two words) |
| byte range being recalled (e.g. `span=(0,2)`) | **span** |
| initial vs refine pass within one chain step | **round** — round 0 = initial, round *k*>0 = refine (unified into one `_emit_round(round_idx, ...)`, not two separate block types) |
| SGD/optimizer iteration (`global_step`/`local_step`) | **training step** |
| chat_tags-style weighted trajectory sampling | **trajectory** |
| the compressed per-chunk/per-round register (old: "SLOT") | **STATE** (`HMN_STATE_0..N-1`, `state_len`, `state_vocab_size`, `_cyclic_state_ids`) |
| bounded cross-chain-step memory channel | **relay** — currently the `hop` mechanism (single-hop attention permission, see below); the original `STATE_QUEUE`/`h_inject` design (forced feature-vector copy) is deleted, see "Deleted mechanisms" |

**Vocab layout** (reordered this session — chat tags now come first, STATE occupies the tail): `HMN_SRC_OPEN/CLOSE`=256/257, `HMN_QUERY_OPEN/CLOSE`=258/259, `HMN_RESPONSE_OPEN/CLOSE`=260/261 (three generic pairs, reused identically at every chain step, no per-position variants, no `<mem>` wrapper — STATE regions are self-identifying via their placeholder tokens). `HMN_STATE_0`=262 onward — the only region expected to grow, so growth is always a pure tail-append (`hp['V']` must be bumped accordingly past `state_vocab_size=12`, which is free under the default `HMN_TAG_VOCAB_SIZE=274`). See `kvmem/hmn.py`'s vocab-section docstring for the full mechanics and why this ordering replaced the original one.

**Masking rule names** (renamed this session from a legacy "Rule 2/3/3b/3b'/4a/4b/5-8" numbering to descriptive names — see `kvmem/hmn.py`'s `chunk_mask_fb`/`chunk_mask_fb_hop`/`chunk_mask_fb_traj`): **encoding isolation** (encoding STATE_k blocked from other chunks), **chunk blackout** (recall STATE blocked from all raw chunks), **nochain blackout** (recall STATE blocked from all prior chain steps' content — the core invariant), **relay exception** (the `hops`-controlled carve-out from the nochain blackout — see below), **warmup/output bottleneck** (round-0 warmup/output rows restricted to own content only), **refine feedback isolation** (a refine round's `state`/argmax/`feedback_state` blocked from chunks + other outputs), **refine output bottleneck** (refine warmup/output rows restricted to own content only), **enc-chain window** (`enc_hops`-controlled bounded window over the ENCODING-CHUNK sequence itself, orthogonal to `hops`/the relay exception — see below).

**`enc_hops`/`hop_drop_prob` (added 2026-07-31, `chunk_mask_fb_traj` in both `kvmem/hmn.py` and `kvmem/hmn_jax.py`, identical mechanism in both)**: `hops` governs relay between QUERY OPS and is a no-op for any single-query-per-entry trajectory (the suffix-recall/`recall1024` family), since `op_idx` is always 0 there and `op_idx==0` is unconditionally exempt from any `hops` bound — the one query gets PERMANENT, UNBOUNDED attention to every encoded chunk's STATE regardless of `hops`. `enc_hops=N` (default -1 = fully unchanged legacy behavior) generalizes the same "bounded N-back window, back=1 never dropped" relay concept to the ENCODING-CHUNK sequence itself: chunk k's own STATE computation may attend to at most the previous N chunks' STATE (raw-byte encoding isolation is untouched — only cross-chunk STATE-to-STATE visibility is windowed), and any op that would otherwise get permanent `is_any_enc_state` access is windowed the same way against the LAST N chunks instead. `hop_drop_prob` (per-stage, curriculum-annealable) then independently drops each back-distance 2..N at every TRAINING step (back=1 never dropped) — LayerDrop-style stochastic regularization along the chunk/time axis rather than the depth axis, intended to discourage the model from leaning on a dense all-chunks-at-once shortcut. `enc_active_backs` (dict: chunk index -> active back-distance set, or `'query'`) is the mechanism both the dropout sampler (`_sample_enc_active_backs`) and combinatorial per-hop-size eval (`val/weave/hopcombo/S=...`, `hp['eval_combinatorial_hops']`) build on. Verified via direct mask-matrix inspection (regression check that `enc_hops=-1` is byte-identical to before the param existed, chunk-to-chunk window correctness, query window correctness, raw-byte isolation preserved, dropout sampler never drops back=1) in BOTH `kvmem/hmn_jax.py` and `kvmem/hmn.py` independently — the torch port is a line-for-line mirror of the JAX version, not a re-derivation. **Known limitation, not yet built**: incompatible with `bucket_lengths`/`forward_granularity` (asserted, not silently wrong) in both files — bucketing's shared-Lb padding and segmented forward's independent KV-cached groups both need integration work that hasn't been done.

**`hops` semantics** (`chunk_mask_fb_hop`/`chunk_mask_fb_traj`, redesigned 2026-07-15 — no separate flag, `hops` alone controls both the relay window width and whether recurrent mode is on, since those are the same question): **`0` is invalid** (raises `ValueError`) — **`-1` (default)** is unbounded/routing-style: every chain step/op sees the union of ALL earlier chain steps'/ops' own STATE, AND keeps permanent unrestricted attention to every encoding-pass STATE directly (this is what `hmn_routing_4to1_state.py`'s single-chain-step case already does, generalized across multiple chain steps) — **`N>=1`** is a genuine bounded recurrence: the union of only the last N chain steps'/ops' STATE, AND every chain step/op past the first (chain step/op 0 is always exempt — the entry point, no predecessor to relay from, same role as an RNN's `h_0=f(x_0)`) is ADDITIONALLY blocked from every encoding-pass STATE directly, leaving the relay window as its ONLY channel for anything beyond its own local query/warmup — literally `h_t=f(h_{t-1..t-N}, x_t)`, not routing with an optional bonus channel. All four combinations (hops=0 invalid, hops=-1, hops=1, hops=2) verified by direct mask-matrix inspection for both `chunk_mask_fb_hop` (chain-step path) and `chunk_mask_fb_traj` (op_idx path, `stream` pattern). This closes a real gap: encoding-pass access used to remain permanently open regardless of `hops`, making the relay optional rather than load-bearing — `hop`'s pre-2026-07-15 measured results (below) were produced under that leaky version and may not reproduce under this corrected masking. `hmn_recall_queue.py` (already sets `hops=1`) and `hmn_weave_mix_accum_rnn.py` (new, sets `hops=1` explicitly) now exercise the corrected, genuinely-recurrent masking automatically — no separate `_accum_rnn`-suffixed hop config is needed (an earlier `hmn_accum_rnn.py` was built with a since-removed `strict_accumulation` flag and deleted once it became identical to `hmn_recall_queue.py`).

**Refine-round STATE naming**: a refine round has two STATE registers, both filled with placeholder tokens like every other STATE region — `state` (built first, functionally identical to every other STATE register: encoding STATE, round-0 STATE) and `feedback_state` (built after incorporating the round's argmax feedback — the register that actually seeds the fresh `<query>`/generation, and what `hop`'s relay reads as "that chain step's last-round STATE"). `feedback_state` also has its own dedicated tail vocab family (`HMN_FEEDBACK_STATE_FAMILY`, `kvmem/hmn.py`) — a distinct, role-based placeholder alphabet reused identically at every refine round (same pattern as `<query>`/`<response>`, not a position-indexed tag) — added since no `n_refine>0` experiment has ever been trained, so there was no checkpoint compatibility to preserve.

---

## Architecture in plain terms

**The task** (unchanged from every prior architecture in this project): memorize a byte sequence, then recall it from a short seed (warmup), byte-exact.

**Block types** — one unified `HMNModel` class, selected via `block_type` hp:

| `block_type` | Structure | Role |
|---|---|---|
| `attn_mlp` | `x = x + attn(norm1(x)); x = x + ffn(norm2(x))` | standard architecture, for comparison |
| `dual_attn` | `x = x + attn1(norm1(x)); x = x + attn2(norm2(x))` (paired, no MLP) | kept as an available ablation option (byte-identical port of the prior architecture) |
| `single_attn` | `x = x + attn(norm(x))` (one attn, one norm, no MLP) | **the default going forward** — same block repeated `n_layers` times; use `n_layers` = 2× the equivalent `dual_attn` config to match total attention-op count |

**Cross-chain-step relay (`hop`)**: each chain step after the first gets its own round-0 STATE row a narrow **attention permission** (the relay exception, width controlled by `hops` — see the `hops` semantics entry above) to read the last `hops` preceding chain steps' own last-round STATE directly — resolved entirely by mask permissions within one packed-sequence forward pass, no sequential per-chain-step orchestration. This replaced the original `STATE_QUEUE`/`h_inject` design (see "Deleted mechanisms" below), which forced a `.detach()`'d feature-vector copy instead of a learned attention path. The nochain blackout (nothing in the mask lets one chain step attend directly into another's raw content) still holds — the relay exception is the *only* sanctioned cross-chain-step channel, scoped to the STATE row alone (never warmup/response rows). At `hops>=1` this is also the ONLY channel period (encoding-pass access is blocked for chain steps past the first); at `hops=-1` (default) it's additive on top of permanent encoding-pass access.

**Deleted mechanisms**: the original `STATE_QUEUE`/`h_inject` relay (`chunk_positions_chained`, `HMNModel.forward`'s `h_inject` param, `train()`'s `chain=True` sequential per-chain-step training loop) was deleted after `hop` (the attention-permission alternative) was shown to massively outperform it — see [`docs/RESULTS_LOG.md`](docs/RESULTS_LOG.md). `kvmem/configs/hmn_stage1_round0_chained.py` (the deleted mechanism's config) has also been removed — nothing in the codebase can execute a `chain=True` stage anymore.

---

## Results

Full experiment ledger (every run's outcome, chronological) lives in [`docs/RESULTS_LOG.md`](docs/RESULTS_LOG.md) — this section only tracks the current best-known config, not history.

**Current best config**: `state_vocab_size=1` (Experiment 1, 2026-09-06 — beat `state_vocab_size=4` on 3 of 4 stages, decisively at the hardest one). Roadmap stage 1 (single-step encode/decode) is passed at the real target architecture (`d=128, n_layers=16, n_heads=8`, ~1.12M params). Roadmap stage 2 (multi-chunk stitch, `hmn_tpu_recall1024_jax_incremental.py`) is in progress — see `docs/RESULTS_LOG.md` for live per-stage numbers.

---

## Structured-data track (queued, not yet used in training)

**Why**: genuine compression (zip/gzip-style, exploiting statistical redundancy) cannot emerge from training on the max-entropy random bytes used everywhere else in this project — Shannon's source coding theorem makes such data literally incompressible, so there's no redundancy for `STATE` to learn to exploit. Random-byte training only teaches raw lossless storage density and the addressing algorithm, not compression. Getting emergent compression requires structured/compressible training data.

**`kvmem/structured_data.py`** implements nine generator families (plus one documented placeholder, `gen_template_repeat`), each sampling **fresh random parameters per call** (required, not optional — a fixed rule across all examples lets the model bake it into static weights instead of encoding anything into `STATE`, the same FFN-as-static-knowledge failure mode this project's `dual_attn` design already avoids elsewhere), organized by which real compressor family they're built to exercise (see `docs/HISTORY.md` §8 for full detail and `LANGUAGE.md` for the generative-hierarchy framing that motivated the newer six):
- `gen_chaotic_logistic` / `gen_fractal_midpoint` — continuous dynamical systems (logistic map / midpoint-displacement fractal), byte-quantized. Weak, quantization-lossy structure — kept for ablation, not the default.
- `gen_ca` — 1D cellular automaton, random rule table + initial condition. **Default for general use** — discrete-native, exactly reproducible, huge tunable rule space.
- `gen_markov` — order-1 Markov chain over the full 256-byte alphabet, **exact closed-form entropy-rate calibration** (bisection against the true stationary-distribution entropy, no measure-and-search). **Measured finding**: `measure_bits_per_byte` (zlib) is *not* a valid sanity check for this generator — DEFLATE's Huffman stage codes against marginal/global frequency, not the previous byte, so it's structurally blind to order-1 conditional structure even when that structure is real and exactly calibrated (empirically confirmed: zlib stayed ~7-8 bits/byte across target_bits=1/2/4/6, while direct empirical order-1 conditional entropy tracked correctly at ~0.85/1.6/2.8/3.6). This is exactly the kind of structure a context-conditional (attention-based) model like this project's own architecture CAN exploit, even though zlib can't see it.
- `gen_iid_skewed` — i.i.d. bytes from a skewed (Zipf-like) marginal distribution. The deliberate "control case zlib CAN see" pairing against `gen_markov` — exact closed-form entropy, and zlib tracks it closely (verified).
- `gen_run_length` — RLE/LZ77-visible: fresh byte + geometric-length run, repeated. Second "zlib-visible" control case, different mechanism (literal repeats, not marginal skew) — approximate closed-form calibration, verified tracking zlib well.
- `gen_markov_order_k` — generalizes `gen_markov` to context length > 1, small alphabet (`K`/`order` tunable, meta-state space `K^order`) for tractability, same exact bisection-on-entropy approach.
- `gen_match_distance` — parametrized LZ77-style generator (match probability + match-DISTANCE range + match-length, not a fixed phrase vocabulary): **required a mid-implementation fix** — an initial single-byte-copy-per-event version was nearly invisible to zlib (DEFLATE needs a 3+ byte match to encode one at all; isolated single-byte copies almost never chain into that by chance), fixed by emitting genuine multi-byte match runs, after which zlib tracked target_bits closely. **Recovery-probe contamination warning**: exact byte repetition means a model could "recover" a matched byte via simple positional copying — do not use for the chain-memory recovery probe without accounting for this.
- `gen_mixed_order` — stochastically blends order-0/1/3 components per position (CTW/PPM-exploitable — defeats any single fixed context length). Least precisely calibrated of the nine (component-wise calibration is only an approximation of the true switched-process entropy) — flagged honestly, not overclaimed.
- `gen_template_repeat` — **placeholder, `NotImplementedError`**, fixed-vocabulary phrase-repetition design documented but not built (superseded in practice by `gen_match_distance`'s more general, already-implemented approach).

**`target_bits` parameter**: all generators accept `target_bits` — desired bits/byte of TRUE compressibility. Calibration precision varies by generator and is documented per-function: exact closed-form bisection for `gen_markov`/`gen_iid_skewed`/`gen_markov_order_k`; approximate closed-form bisection for `gen_run_length`/`gen_match_distance`/`gen_mixed_order`; `measure_bits_per_byte` (zlib) measure-and-search for `gen_chaotic_logistic`/`gen_fractal_midpoint`/`gen_ca`, with the known seed-dependent imprecision documented in each of those three's own docstrings (unchanged from before this session).

**Caution before using this for the chain-memory recovery probe specifically**: structured data risks contaminating that probe — a model could "recover" an earlier chain step's content by inferring the generating rule from its own visible span, without touching the relay at all. Keep the recovery probe on pure random data first; structured data is queued as a separate, later question (does bounded `STATE` capacity effectively increase when content is compressible), not a replacement for the current validation.


**First ablation run result** (markov/ca vs random, 2026-08-01): moved to [`docs/RESULTS_LOG.md`](docs/RESULTS_LOG.md) — not a clean win either direction, open question, queued for a matched-step-count rerun.

---

## TPU access

**`TPU.md` and `TPU_WORKFLOW.md` (repo root) are the user's own scratchpad — do not edit, restructure, or archive their content.** They hold personal gcloud/ssh notes (including cross-project ones) in whatever shorthand form is convenient for the user; treat them as read-only reference, not as project docs to prune or fold into the files below.

**Current path: JAX/Flax NNX (`kvmem/hmn_jax.py`), exclusively.** `torch_xla` was tried first and abandoned — real, reproducible, never-root-caused instability (a data-dependent NaN independent of every hyperparameter tried, and a separate RoPE-at-long-length NaN that persisted through every fix) that JAX simply doesn't reproduce on the identical computation. `kvmem/hmn.py`'s own torch_xla-specific plumbing (`bucket_lengths`, `device_str='tpu'`) still exists but is dead/frozen code — do not extend it. Full forensic history of both the torch_xla saga and the JAX port: [`docs/TPU_JAX_PORT.md`](docs/TPU_JAX_PORT.md).

**Operational how-to** (gcloud commands, persistent SSH, tmux conventions, common failure modes): [`docs/tpu_setup.md`](docs/tpu_setup.md) + [`docs/tpu_direct_ssh.md`](docs/tpu_direct_ssh.md). Available TRC queued-resource nodes: [`TPU.md`](TPU.md).

**Standing rule: never create/recreate a TPU VM directly via `gcloud ... tpu-vm create`** — this project's TPU access is via TRC (TPU Research Cloud) free-tier quota, tracked through `gcloud compute tpus queued-resources`, NOT the direct `tpu-vm create` API (that bills normally instead of drawing on TRC quota). If a node is preempted, report it and wait — do not self-provision a replacement, regardless of how much autonomy is otherwise granted for the training work itself.

**Mindset shift now that TPU compute is no longer scarce**: with multiple queued v4-8 nodes available in parallel, prefer throwing more parallel TPU-hours at an ablation or a bridging-curriculum-stage question over the older MPS/CPU-era discipline of very carefully rationed single-run curricula. Run redundant copies across nodes to hedge against spot preemption rather than babysitting one run.

**URGENT, unresolved as of 2026-09-09: every `kvmem_jax` TPU run this session has used only 1 of a v4-8 node's 4 chips, and MFU on that one chip is almost certainly very low.** Direct evidence from the `[util:...]` log lines every run already prints: `chip0: duty_cycle=100%` but `chip1/chip2/chip3: duty_cycle=0.0%, HBM=0.00/30.75GiB` — the other 3 chips sit completely idle the entire run, and HBM usage on the one active chip is only ~7-9% of capacity (2.2-2.7 GiB of 30.75 GiB). `duty_cycle=100%` only means the chip's pipeline isn't stalled waiting on the host — it says nothing about how efficiently that compute is used, and at this model's scale (~1.1M params, `B=32-64`, `L`~700-1200) the matmuls are almost certainly far too small to saturate a v4 chip's systolic array (~197 TFLOPS bf16 peak), meaning real MFU is likely in the low single digits. This is why runs that "should" be fast (a ~1M-param model) have been taking many hours per stage. **Fix already exists in code, just never enabled for these runs**: `hp['data_parallel']=True` (`kvmem_jax/hmn_jax.py`, `jax.jit`+`Mesh`/`NamedSharding`, previously verified working elsewhere in this project) spreads the batch across all 4 chips — batch size must be divisible by 4. This alone should give close to a 4x wall-clock speedup; it does not by itself fix the small-matmul-shape MFU problem on top of that (a separate, unaddressed question — larger per-chip batch via `data_parallel` may help some of that too, since bigger matmuls are usually more MXU-efficient, but this hasn't been measured). **TODO, not yet done: retrain the recall1024/STATE-capacity lineage with `data_parallel=True` enabled** — every config launched so far (`hmn_oneshotrecall_jax*.py`, `hmn_recall1024_jax_oneshotrecall*.py`, the `_bucket_*` variants) ran single-chip; none of their reported wall-clock times or resource usage should be read as representative of what this workload actually needs once fixed.

---

## Roadmap (recall1024, staged by capability — set 2026-07-31, not by fixed step budgets)

Each stage is a genuine capability gate, not a scheduled step count — advance only once the
current stage actually converges; insert whatever intermediate stage is needed if it doesn't
(see the third bullet). Supersedes any earlier fixed n_steps-per-stage curriculum plan for this
line of work (`hmn_tpu_recall1024_jax_curriculum_staged.py`, the original `hmn_tpu_recall1024_
jax_hopdrop.py` design) where the prior stage's own convergence wasn't actually verified first.

**Live status (2026-09-06)**: **stage 1 PASSED** (`hmn_tpu_sanity_w25_rope_jax.py`, `state_vocab_size=1`,
best=85.5% at the final chunk_len=64 stage — see Experiment 1 in `docs/RESULTS_LOG.md`). **Stage 2 IN
PROGRESS** (`kvmem/configs/hmn_tpu_recall1024_jax_incremental.py`, warm-started from stage 1's winner,
running on two TPU nodes as redundant copies — see Experiment 2 in `docs/RESULTS_LOG.md` for live
per-stage numbers). The stage-1/2 descriptions below are the target-shape design, not yet updated
per-run — check `docs/RESULTS_LOG.md` for what's actually happened.

**Update (2026-09-10, run now complete)**: the `chunk_len` 16→32 collapse (stage 2's own failure
point) got a real, reproducible partial fix — a STATE-capacity ablation (`kvmem_jax` fork:
`state_len=8`, `state_vocab_size=2`, MLP still disabled) hit **27.5-28.6% at chunk_len=32** across two
independent runs, vs. 2.4-4.6% for both the original no-MLP and MLP-variant attempts at the same
stage — roughly 6-10x better. But the improvement did NOT hold going further: both runs ran their
full step budget to completion (260,000 steps, no crash, no early stop) and finished at **14.7-15.8%**
once `chunk_len=64` and the bounded `enc_hops=4` relay were reached — well below every stage's own
early-stop gate. See `docs/RESULTS_LOG.md` for the full ablation trail and the hopcombo breakdown.
**This result's own wall-clock/step-budget numbers are still suspect** — see the URGENT single-chip/
MFU note above (neither run ever used more than 1 of 4 chips); a real retrain with
`data_parallel=True` is a TODO before treating 14.7-15.8% as any kind of ceiling rather than an
artifact of running out of step budget on a badly-underutilized setup.

1. **Single-step encode/decode** — **CORRECTED 2026-07-31 (huge-step-back)**: `hmn_tpu_sanity_
   w25_rope_jax_nocurr.py`'s no-curriculum, all-lengths-mixed-in-one-stage design ran its full
   200000-step budget on `v6e-4-2` and finished at only **best=17.0% val MEAN** (well below
   `early_stop_mean=80.0`) — the qualitative eyeball (`log_qualitative_eyeball`, new this session)
   showed the model getting 1-2 bytes right after warmup then collapsing into repeating a single
   byte (`'''''''`, `]]]`) rather than tracking the sequence, a genuine partial-learning-then-
   collapse pattern, not a clean pass. **Reverted to `hmn_tpu_sanity_w25_rope_jax.py`** — a direct
   JAX port of the ALREADY-PROVEN torch curriculum (`hmn_tpu_sanity_w25_rope.py`/`hmn_notags_w25_
   rope`'s own staged design: `chunk_len` 8 → 16 → 32 → 64, each stage gated by `early_stop_
   mean=80.0` before advancing, each later stage rehearsing earlier lengths at `weight=0.5`), at
   the same `d=128, n_layers=16, n_heads=8` (~1.1M params) architecture the `_nocurr` attempt
   used — the actual fix is restoring curriculum STAGING, not a model-size change (that part was
   already correct). One encode chunk, one decode/recall query per entry; proves the target-sized
   model can do the easy task at all, one length at a time with a real convergence gate, before
   ever touching multi-chunk chaining.
2. **Multi-step encode, single-step decode (`stitch`)** — built on the `hmn_tpu_recall1024_jax_
   hopdrop.py` mechanism (`enc_hops`/`hop_drop_prob`, bounded+stochastically-dropped encoding-
   chain window — see the "Masking rule names"/`enc_hops` entries near the top of this file), not
   a plain fixed-window suffix recall. Two requirements, corrected 2026-07-31 (supersedes the
   original wording of this stage):
   - **Anchor placement GENERALIZED**: the warmup anchor may sit on ANY chunk (not clustered near
     the end, which is what let the model find the "near-end anchor easy" positional shortcut
     documented throughout this file), and the supervised response must cover **two consecutive
     chunks or more** counted from the anchor's own chunk — UNLESS the anchor's chunk is the last
     chunk (nothing left to recall past it, a genuine single-chunk-remainder case). Targets the
     "near-start anchor hard, near-end anchor easy" failure pattern by construction: every anchor
     position gets a genuinely multi-chunk recall requirement, not just the near-end ones that
     happen to have a long tail regardless of anchor choice.
   - **`hop=1`-usable but NOT `hop=1`-only**: the final trained model must work correctly at
     `hops=1` (the minimal/most-constrained relay window — only the immediately preceding chunk's
     STATE) since that's the cheapest/most scalable inference mode, but must ALSO generalize to
     wider `hops` windows with STRICTLY BETTER accuracy as more context becomes available, not
     brittleness where only the exact hop count trained on works (or worse, where widening the
     window HURTS). This is precisely what `eval_combinatorial_hops`/`val/weave/hopcombo/S=...`
     was built to verify — a passing stage 2 should show hopcombo MEAN monotonically
     non-decreasing as `|S|` grows from `{1}` up to the full `{1..enc_hops}` window, not flat or
     inverted. `hop_drop_prob`'s own back=1-never-dropped design already trains toward this
     (every step sees at least the hop=1 case, wider windows are the "bonus" the model must learn
     to actually use, not just tolerate) — this requirement makes that training-time property an
     explicit, checked EVAL criterion for calling stage 2 done, not just an implicit hope.
3. **Bridging stages, inserted as needed** — if stage 2 doesn't converge jumping directly from
   stage 1, add whatever intermediate task/trajectory shape is required (e.g. a 2-chunk stitch
   stage before 4/8/16-chunk, a narrower `enc_hops` window before widening it, a smaller anchor
   grid before the dense one) — this roadmap is deliberately NOT a fixed pre-specified ladder;
   the stage-2 entry above is the target shape, not a promise that it's reachable in one hop from
   stage 1. Diagnose the specific failure (per-entry breakdown, not just aggregate MEAN — see the
   "positional-shortcut-to-content-addressing trade-off" entry above for why) before deciding
   what bridging stage to insert.

---

## Key Principles

- `null_kv=True` always — 1.5–2× faster, better bpb
- Mask convention: `0.0`=attend, `-1e9`=blocked (additive bias for `F.scaled_dot_product_attention`)
- Round 0 (initial) before refine rounds — always required for the feedback mechanism
- The nochain blackout (each chain step's round-0 STATE blocked from ALL tokens in prior rec_blocks) is what makes chain steps independently trainable, and the relay exception (`hop`'s `hops`-controlled attention permission, `chunk_mask_fb_hop`/`chunk_mask_fb_traj`) is the sole sanctioned carve-out from it. Do not weaken either without understanding the consequences — the nochain blackout is what keeps chain steps from leaking raw content forward.
- **Always verify masking changes against the actual attention-mask matrix, not just "does it run"** — a smoke test that completes without crashing says nothing about whether the intended access pattern is actually being enforced (the `hop` encoding-pass leak ran and trained "successfully" for the whole time it was unintentionally leaky). Check specific (row, col) blocks directly, per chain step/op, before trusting a masking change.
- **Report precisely, never round up** — state exactly what was measured (e.g. a padded/truncated excerpt), not the whole file
- **Verify infra before trusting it mid-run**: always check process liveness (`ps -p <pid>`) explicitly on every wake, not just log content — a silently-exited process produces no new log lines
- **Flag suspiciously long/slow jobs instead of silently waiting on them** — a background job with no output for a long stretch is not automatically "still working," but it's also not automatically "stuck": check CPU-time growth (`ps -o pid,time,%cpu`) to distinguish a live-but-slow process from a hung one, and separately sanity-check the workload's own cost model (e.g. full-recompute/`nokv`-style decode is `O(out_len·L²)` — a real production-scale `L` can be 100-1000x more expensive per call than the small shape a script was first verified against) before assuming a stall. If the estimated cost is impractical, don't just let it keep running — kill it and rescope (fewer entries, shorter/prefix-only cases, or a cheaper proxy) rather than wait hours for a check that could've taken minutes. Also route any long-running remote job's stdout to a real file (`nohup ... > file.log 2>&1 &`, then `disown`), never to a bare `&`-backgrounded shell inside an SSH command — the multiplexed channel's pipe can outlive the reader, and a process that eventually tries to write to a dead pipe risks a crash (`BrokenPipeError`) with nothing captured.
- **Never run two training jobs at once**
- **Editing `kvmem/hmn.py` while a training job is running is safe** — Python has already loaded the module into the live process's memory; on-disk edits don't affect it. Verified multiple times this session (deleting `h_inject`, the vocab reorder) without disrupting an in-progress run.

---

## Docs

| What | Where |
|------|-------|
| **`docs/HMN_RECIPE.md`** — quickstart + current-state-only reference (model architecture, the E/S/Q trajectory DSL, the relay, val/test mechanics) for a newcomer with zero context | [`docs/HMN_RECIPE.md`](docs/HMN_RECIPE.md) |
| **`docs/HISTORY.md`** — the full narrative: every design decision, terminology evolution, the deleted `relay`/`STATE_QUEUE` mechanism, the vocab reorder, structured-data track detail, compression diagnostics design, a classical/non-DNN alternatives discussion | [`docs/HISTORY.md`](docs/HISTORY.md) |
| **`docs/RESULTS_LOG.md`** — the append-only experiment ledger going forward: every training run's outcome, chronological, including the pre-TPU torch results and the live Experiment 1/2 roadmap numbers | [`docs/RESULTS_LOG.md`](docs/RESULTS_LOG.md) |
| **`docs/TPU_JAX_PORT.md`** — the full TPU porting forensics: the abandoned `torch_xla` saga (bugs 1-7, never root-caused instability) and the from-scratch JAX/Flax NNX port that replaced it, now the only supported TPU path | [`docs/TPU_JAX_PORT.md`](docs/TPU_JAX_PORT.md) |
| **`docs/tpu_setup.md`** + **`docs/tpu_direct_ssh.md`** — day-to-day TPU operational how-to: gcloud commands, persistent SSH, tmux conventions, common failure modes | [`docs/tpu_setup.md`](docs/tpu_setup.md), [`docs/tpu_direct_ssh.md`](docs/tpu_direct_ssh.md) |
| **`docs/HMN_WALKTHROUGH.md`** — train & eval walkthrough, c64 → weave pipeline | [`docs/HMN_WALKTHROUGH.md`](docs/HMN_WALKTHROUGH.md) |
| **`kvmem_jax/`** — a deliberately non-DRY, standalone fork of `kvmem/hmn_jax.py` (not imported by or importing from `kvmem/`), created 2026-09-06 (originally `kvmem_mlp/`, renamed) to test capacity-increasing hypotheses at the roadmap-stage-2 `chunk_len=16→32` collapse — added real `block_type='attn_mlp'` support (refuted as the fix) and the STATE-capacity ablation (`state_len=8`/`state_vocab_size=2`): a real 6-10x win at chunk_len=32 (27.5-28.6% vs. 2.4-4.6% collapse) that decayed again by the final stage (14.7-15.8% at chunk_len=64 + bounded `enc_hops` relay, run complete as of 2026-09-10, no early stop reached) — plus the first-ever `n_refine>0` training run in this project. Own `configs/`/`logs/` subdirectories. **Ran entirely single-chip (`data_parallel` never enabled) — see the URGENT MFU note above; a retrain is a TODO before treating the final 14.7-15.8% as a real ceiling rather than an artifact of an exhausted step budget on a badly-underutilized setup.** Full trail: `docs/RESULTS_LOG.md`'s "`kvmem_jax` fork" entry. |
| The rewrite plan (original design/approval record — every naming decision, worked `STATE_QUEUE` example predating the `hop` mechanism, why each choice was made) | [`/Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md`](/Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md) |
| Current implementation | [`kvmem/hmn.py`](kvmem/hmn.py) (single file), active configs in [`kvmem/configs/`](kvmem/configs/) — 12 configs left active as of 2026-09-06 (the recall1024/Experiment-1-2 lineage, the local torch recipes `hmn_notags_w25(_rope).py`/`hmn_tpu_sanity_w25_rope.py`, the still-open structured-data ablation, `hmn_single_recall_c64.py`, `hmn_squeeze_markov_n4.py`) — everything else moved to [`kvmem/configs/archive/`](kvmem/configs/archive/) (pre-existing archive), [`kvmem/configs/archive_recall1024_superseded/`](kvmem/configs/archive_recall1024_superseded/) (earlier recall1024/stitch attempts superseded by the incremental-curriculum path), or [`kvmem/configs/archive_concluded_investigations/`](kvmem/configs/archive_concluded_investigations/) (positional-shortcut/NoPE/anchor investigations whose findings are already in `docs/HISTORY.md`), structured-data generators in [`kvmem/structured_data.py`](kvmem/structured_data.py), compression diagnostics in [`kvmem/eval_compression.py`](kvmem/eval_compression.py), trajectory-generalization diagnostics in [`kvmem/eval_weave.py`](kvmem/eval_weave.py), positional-shortcut diagnostics in [`kvmem/probe_positional_shortcut.py`](kvmem/probe_positional_shortcut.py) (behavioral swap test, `batch`/`interleave_delayed` shapes) and [`kvmem/probe_mechanistic_addressing.py`](kvmem/probe_mechanistic_addressing.py) (attention-mass + gradient-saliency counterpart, uses `MHAttention.capture_attn` in `hmn.py`); the suffix-recall/stitch design's own single-query shape has its own pair, [`kvmem/probe_stitch_content_addressing.py`](kvmem/probe_stitch_content_addressing.py) (behavioral swap test) and [`kvmem/probe_stitch_mechanistic_addressing.py`](kvmem/probe_stitch_mechanistic_addressing.py) (mechanistic counterpart, adapted for `hops=-1` routing) |
| `kvmem/hmn.py` dated snapshots (`hmn_v1_backup.py` through `hmn_v4_backup.py` — pre-cleanup draft, post-cleanup, pre-DSL/repeat_batch/stitch feature work, and the pre-promotion old tagged design respectively) were **deleted** (not archived) once the promoted `kvmem/hmn.py` stabilized — they were pure diffing artifacts, never imported by anything, and their content is superseded by the current file plus this doc's own narrative (Results section, `docs/HISTORY.md` §15). `archive_v1/` remains the actual archival record for pre-rewrite code/docs. | — |
| Everything from before the rewrite (dual-attn discovery, RMSNorm, stitching, `juz1.txt` design, MDL theory, all prior architecture history — code AND docs) | [`archive_v1/`](archive_v1/) — old `kvmem/`, old `experiments/`, old `docs/` (`SRS_RECIPE.md`, `EARLY_ARCHITECTURE_HISTORY.md`, `MDL_MODEL_SIZE.md`, etc. all moved here, `docs/` at the repo root is a fresh start for this rewrite going forward) |
| Previous version of this file | [`archive_v1/CLAUDE_v1.md`](archive_v1/CLAUDE_v1.md) |
| Test set | [`datasets/suratalfatihah.txt`](datasets/suratalfatihah.txt) |
| `juz1` scaling target (not yet used in training) | [`datasets/juz1.txt`](datasets/juz1.txt) |
