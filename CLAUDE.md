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

**Masking rule names** (renamed this session from a legacy "Rule 2/3/3b/3b'/4a/4b/5-8" numbering to descriptive names — see `kvmem/hmn.py`'s `chunk_mask_fb`/`chunk_mask_fb_hop`/`chunk_mask_fb_traj`): **encoding isolation** (encoding STATE_k blocked from other chunks), **chunk blackout** (recall STATE blocked from all raw chunks), **nochain blackout** (recall STATE blocked from all prior chain steps' content — the core invariant), **relay exception** (the single-hop carve-out from the nochain blackout that `hop` uses), **warmup/output bottleneck** (round-0 warmup/output rows restricted to own content only), **refine feedback isolation** (a refine round's `state`/argmax/`feedback_state` blocked from chunks + other outputs), **refine output bottleneck** (refine warmup/output rows restricted to own content only).

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

**Cross-chain-step relay (`hop`)**: each chain step after the first gets its own round-0 STATE row a narrow, single-hop **attention permission** (the relay exception) to read the immediately preceding chain step's own last-round STATE directly — resolved entirely by mask permissions within one packed-sequence forward pass, no sequential per-chain-step orchestration. This replaced the original `STATE_QUEUE`/`h_inject` design (see "Deleted mechanisms" below), which forced a `.detach()`'d feature-vector copy instead of a learned attention path. The nochain blackout (nothing in the mask lets one chain step attend directly into another's raw content) still holds — the relay exception is the *only* sanctioned cross-chain-step channel, scoped to the STATE row alone (never warmup/response rows).

**Deleted mechanisms**: the original `STATE_QUEUE`/`h_inject` relay (`chunk_positions_chained`, `HMNModel.forward`'s `h_inject` param, `train()`'s `chain=True` sequential per-chain-step training loop) was deleted after `hop` (the attention-permission alternative) was shown to massively outperform it — see "Results" below. `kvmem/configs/hmn_stage1_round0_chained.py` (the deleted mechanism's config) has also been removed — nothing in the codebase can execute a `chain=True` stage anymore.

---

## Results

- **`solo`** (`kvmem/configs/hmn_single_recall.py`) — one chain step, round 0 only, no relay. **Done**: 160000/160000 steps, val per-span MEAN=94.4% (best 97.2% at step 150000), test=100%, loss=0.017 — matches the historical ~100% single-window initial-round ceiling.
- **`relay`** (config file and logs both deleted — the now-removed `STATE_QUEUE`/`h_inject` mechanism) — three chain steps (`[(0,2),(1,3),(2,4)]`), warm-started from `solo`. **Done, everything removed** (old vocab, superseded mechanism). Final numbers (preserved here since the run itself is gone): val MEAN=45.8% (STITCHED=44.6%), test MEAN=48.6% (STITCHED=44.6%). Chain step 2 (the 2-hop case) closed at 12.5%/12.5%, never exceeding its step-90000 peak of 11.1%/25.0% across the final 70000 steps despite loss continuing to decline (1.448→1.051). Motivated `hop`.
- **`hop`** (`kvmem/configs/hmn_recall_queue.py`, the attention-permission relay) — identical hyperparameters/schedule to `relay`, warm-started from `solo`. **Done, run twice, with a real discrepancy between the two runs** (see below) — the checkpoint currently on disk is from the SECOND run, weaker than what the recovery-probe result below was measured against.
  - **First run** (logs since deleted, old vocab): val = 100.0%/95.8%/72.2% (STITCHED=88.1%), test = 100.0%/95.8%/70.8% (STITCHED=85.7%), loss=0.603, best checkpoint 88.7%. Massively outperformed `relay` on every metric (chain step 2 test 70.8% vs relay's 12.5%, 5.7x) — strong evidence the gradient-flow fix (full backprop vs. `.detach()`-truncated copy) matters. This is the run the recovery-probe result immediately below was measured against.
  - **Second run** (current checkpoint, post vocab-reorder — same config, warm-started from the reordered-vocab `solo`): val/test STITCHED=71.4%/71.4%, loss=1.851 — substantially worse, loss plateaued flat around 1.84-1.86 from step 50000 onward, never broke out like the first run did. Mask/relay mechanism independently verified correct in both runs (byte-identical mask regardless of vocab ID relabeling); the discrepancy is attributed to warm-start sensitivity, not a code defect — two `solo` checkpoints can both hit ~100% on solo's own near-trivial task while differing enough in underlying weight configuration to matter for `hop`'s much harder relay-learning objective. Re-running `hop` is not guaranteed to reproduce either result exactly.
- **Chain-memory recovery probe** (`eval_weave.py --patterns repeat_query`, run against `hop`'s FIRST-run checkpoint, since deleted) — **failed cleanly**. Query span (0,2)→(1,3)→(2,4)→re-query (0,2): first occurrence 100% (trivial), repeated occurrence (reachable only through the accumulated 3-hop relay chain, since direct attention back to chunk 0/1 is blocked) **0.0% across all 3 test sequences** — complete, not partial, failure. Caveat: `repeat_query` is a trajectory shape `hop` was never trained on (only the fixed 3-query schedule), so this could reflect either "the relay doesn't preserve information across 3 hops" or "the model can't generalize to this novel trajectory shape at all" — the total (not gradual) failure leans toward the latter, but this test alone can't cleanly separate the two. `long_hop_recovery` (n_chunks=8) scored near-zero including first occurrences — the known length-extrapolation confound (trained at L=236), not an additional signal. This motivated `weave_mix` (below) — re-running the probe against a `weave_mix`-trained checkpoint is the direct follow-up test.
- **Vocab reorder** — mechanism verified correct (mask byte-identical regardless of vocab ID relabeling, batch construction confirmed correct under the new IDs). Chat tags now occupy IDs 256-261 (fixed, small), STATE occupies the tail from 262 (pure append-growth). Old-vocab logs/checkpoints (original `solo`/`relay`/`hop`) have been deleted, and the `_vreorder`-suffixed configs/logs were renamed to drop that suffix now that the reordered vocab is simply *the* vocab (no more old-vocab comparison to distinguish against) — `kvmem/configs/hmn_single_recall.py` and `hmn_recall_queue.py` (renamed from `hmn_flow.py`) are now the reordered-vocab versions. See the `hop` entry above for the reproducibility-check numbers themselves (kept there, not duplicated here).
- **`squeeze`** (`kvmem/configs/hmn_squeeze_ca_n4.py` + `hmn_squeeze_random_n4.py`) — mid-run (paused). Dedicated compression-capacity test (CA-structured vs. random-byte paired control). See `docs/HISTORY.md` §10 for the full design rationale, including the `chunk_len` capacity-pressure correction.
- **`weave`** — fully built: position/mask/DSL/decode/eval harness (`chunk_positions_traj`/`chunk_mask_fb_traj`/`ar_decode_traj_nokv`/`kvmem/eval_weave.py`) plus the `weave_mix` training-loop dispatch into `train()` (`stage['weave_mix']`, trains on `batch`/`stream`/`interleave_delayed`, rejects the test-only patterns with an assertion, verified via smoke test). **Queued to run**: `kvmem/configs/hmn_weave_mix.py`, uniform mix of the three train-mix patterns, warm-started from `hop`'s checkpoint — directly tests whether the recovery probe's clean `repeat_query` failure was a generalization gap (never trained on varied orderings) rather than a relay-mechanism failure, without touching the relay itself. Runs after `squeeze` finishes (never two jobs at once).

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

---

## Key Principles

- `null_kv=True` always — 1.5–2× faster, better bpb
- Mask convention: `0.0`=attend, `-1e9`=blocked (additive bias for `F.scaled_dot_product_attention`)
- Round 0 (initial) before refine rounds — always required for the feedback mechanism
- The nochain blackout (each chain step's round-0 STATE blocked from ALL tokens in prior rec_blocks) is what makes chain steps independently trainable, and the relay exception (`hop`'s single-hop attention permission, `chunk_mask_fb_hop`/`chunk_mask_fb_traj`) is the sole sanctioned carve-out from it. Do not weaken either without understanding the consequences — the nochain blackout is what keeps chain steps from leaking raw content forward.
- **Report precisely, never round up** — state exactly what was measured (e.g. a padded/truncated excerpt), not the whole file
- **Verify infra before trusting it mid-run**: always check process liveness (`ps -p <pid>`) explicitly on every wake, not just log content — a silently-exited process produces no new log lines
- **Never run two training jobs at once**
- **Editing `kvmem/hmn.py` while a training job is running is safe** — Python has already loaded the module into the live process's memory; on-disk edits don't affect it. Verified multiple times this session (deleting `h_inject`, the vocab reorder) without disrupting an in-progress run.

---

## Docs

| What | Where |
|------|-------|
| **`docs/HMN_RECIPE.md`** — quickstart + current-state-only reference (model architecture, the E/S/Q trajectory DSL, the relay, val/test mechanics) for a newcomer with zero context | [`docs/HMN_RECIPE.md`](docs/HMN_RECIPE.md) |
| **`docs/HISTORY.md`** — the full narrative: every design decision, terminology evolution, the deleted `relay`/`STATE_QUEUE` mechanism, the vocab reorder, structured-data track detail, compression diagnostics design, a classical/non-DNN alternatives discussion | [`docs/HISTORY.md`](docs/HISTORY.md) |
| The rewrite plan (original design/approval record — every naming decision, worked `STATE_QUEUE` example predating the `hop` mechanism, why each choice was made) | [`/Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md`](/Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md) |
| Current implementation | [`kvmem/hmn.py`](kvmem/hmn.py) (single file), configs in [`kvmem/configs/`](kvmem/configs/), structured-data generators in [`kvmem/structured_data.py`](kvmem/structured_data.py), compression diagnostics in [`kvmem/eval_compression.py`](kvmem/eval_compression.py), trajectory-generalization diagnostics in [`kvmem/eval_weave.py`](kvmem/eval_weave.py) |
| Everything from before the rewrite (dual-attn discovery, RMSNorm, stitching, `juz1.txt` design, MDL theory, all prior architecture history — code AND docs) | [`archive_v1/`](archive_v1/) — old `kvmem/`, old `experiments/`, old `docs/` (`SRS_RECIPE.md`, `EARLY_ARCHITECTURE_HISTORY.md`, `MDL_MODEL_SIZE.md`, etc. all moved here, `docs/` at the repo root is a fresh start for this rewrite going forward) |
| Previous version of this file | [`archive_v1/CLAUDE_v1.md`](archive_v1/CLAUDE_v1.md) |
| Test set | [`datasets/suratalfatihah.txt`](datasets/suratalfatihah.txt) |
| `juz1` scaling target (not yet used in training) | [`datasets/juz1.txt`](datasets/juz1.txt) |
