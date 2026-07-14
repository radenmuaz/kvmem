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
| IQ vs IR pass within one chain step | **round** — round 0 = IQ, round *k*>0 = IR (unified into one `_emit_round(round_idx, ...)`, not two separate block types) |
| SGD/optimizer iteration (`global_step`/`local_step`) | **training step** |
| chat_tags-style weighted trajectory sampling | **trajectory** |
| the compressed per-chunk/per-round register (old: "SLOT") | **STATE** (`HMN_STATE_0..N-1`, `state_len`, `state_vocab_size`, `_cyclic_state_ids`) |
| bounded cross-chain-step memory channel | **relay** — currently the `flow` mechanism (single-hop attention permission, see below); the original `STATE_QUEUE`/`h_inject` design (forced feature-vector copy) is deleted, see "Deleted mechanisms" |

**Vocab layout** (reordered this session — chat tags now come first, STATE occupies the tail): `HMN_SRC_OPEN/CLOSE`=256/257, `HMN_QUERY_OPEN/CLOSE`=258/259, `HMN_RESPONSE_OPEN/CLOSE`=260/261 (three generic pairs, reused identically at every chain step, no per-position variants, no `<mem>` wrapper — STATE regions are self-identifying via their placeholder tokens). `HMN_STATE_0`=262 onward — the only region expected to grow, so growth is always a pure tail-append (`hp['V']` must be bumped accordingly past `state_vocab_size=12`, which is free under the default `HMN_TAG_VOCAB_SIZE=274`). See `kvmem/hmn.py`'s vocab-section docstring for the full mechanics and why this ordering replaced the original one.

**Masking rule names** (renamed this session from a legacy "Rule 2/3/3b/3b'/4a/4b/5-8" numbering to descriptive names — see `kvmem/hmn.py`'s `chunk_mask_fb`/`chunk_mask_fb_flow`/`chunk_mask_fb_traj`): **encoding isolation** (encoding STATE_k blocked from other chunks), **chunk blackout** (recall STATE blocked from all raw chunks), **nochain blackout** (recall STATE blocked from all prior chain steps' content — the core invariant), **relay exception** (the single-hop carve-out from the nochain blackout that `flow` uses), **warmup/output bottleneck** (round-0 warmup/output rows restricted to own content only), **IR feedback isolation** (STATE_A/argmax/STATE_B blocked from chunks + other outputs), **IR output bottleneck** (IR warmup/output rows restricted to own content only).

---

## Architecture in plain terms

**The task** (unchanged from every prior architecture in this project): memorize a byte sequence, then recall it from a short seed (warmup), byte-exact.

**Block types** — one unified `HMNModel` class, selected via `block_type` hp:

| `block_type` | Structure | Role |
|---|---|---|
| `attn_mlp` | `x = x + attn(norm1(x)); x = x + ffn(norm2(x))` | standard architecture, for comparison |
| `dual_attn` | `x = x + attn1(norm1(x)); x = x + attn2(norm2(x))` (paired, no MLP) | kept as an available ablation option (byte-identical port of the prior architecture) |
| `single_attn` | `x = x + attn(norm(x))` (one attn, one norm, no MLP) | **the default going forward** — same block repeated `n_layers` times; use `n_layers` = 2× the equivalent `dual_attn` config to match total attention-op count |

**Cross-chain-step relay (`flow`)**: each chain step after the first gets its own round-0 STATE row a narrow, single-hop **attention permission** (the relay exception) to read the immediately preceding chain step's own last-round STATE directly — resolved entirely by mask permissions within one packed-sequence forward pass, no sequential per-chain-step orchestration. This replaced the original `STATE_QUEUE`/`h_inject` design (see "Deleted mechanisms" below), which forced a `.detach()`'d feature-vector copy instead of a learned attention path. The nochain blackout (nothing in the mask lets one chain step attend directly into another's raw content) still holds — the relay exception is the *only* sanctioned cross-chain-step channel, scoped to the STATE row alone (never warmup/response rows).

**Deleted mechanisms**: the original `STATE_QUEUE`/`h_inject` relay (`chunk_positions_chained`, `HMNModel.forward`'s `h_inject` param, `train()`'s `chain=True` sequential per-chain-step training loop) was deleted after `flow` (the attention-permission alternative) was shown to massively outperform it — see "Results" below. `kvmem/configs/hmn_stage1_round0_chained.py` (the deleted mechanism's config) is kept as a historical record; its `chain=True` key is now a harmless no-op if re-run.

---

## Results

- **`solo`** (`kvmem/configs/hmn_stage0_round0_single.py`) — one chain step, round 0 only, no relay. **Done**: 160000/160000 steps, val per-span MEAN=94.4% (best 97.2% at step 150000), test=100%, loss=0.017 — matches the historical ~100% single-window IQ ceiling.
- **`relay`** (`kvmem/configs/hmn_stage1_round0_chained.py`, the now-deleted `STATE_QUEUE`/`h_inject` mechanism) — three chain steps (`[(0,2),(1,3),(2,4)]`), warm-started from `solo`. **Done**: 160000/160000 steps. Final: val MEAN=45.8% (STITCHED=44.6%), test MEAN=48.6% (STITCHED=44.6%). Chain step 2 (the 2-hop case) closed at 12.5%/12.5%, never exceeding its step-90000 peak of 11.1%/25.0% across the final 70000 steps despite loss continuing to decline (1.448→1.051) — ambiguous whether the `.detach()`-truncated gradient was capping this or more steps would have helped. Motivated `flow`.
- **`flow`** (`kvmem/configs/hmn_flow.py`, the attention-permission relay) — identical hyperparameters/schedule to `relay`, warm-started from the same `solo` checkpoint. **Done**: 160000/160000 steps. Final: val = 100.0%/95.8%/72.2% (STITCHED=88.1%), test = 100.0%/95.8%/70.8% (STITCHED=85.7%), loss=0.603, best checkpoint 88.7%. **Massively outperforms `relay`** on every metric: chain step 1 test 95.8% vs relay's 37.5% (2.5x), chain step 2 test 70.8% vs relay's 12.5% (5.7x) — strong evidence the gradient-flow fix (full backprop vs. `.detach()`-truncated copy) matters. Progress climbed cleanly and monotonically from step 10000 (val 84.7/59.7/13.9) through a plateau around 70.8% on chain step 2 (steps 60k-110k), then broke out again to 72-73% by step 120k-150k while chain step 1 climbed to 95.8%.
- **Chain-memory recovery probe** (`eval_weave.py --patterns repeat_query`, run against `flow`'s checkpoint) — **fails cleanly**. Query span (0,2)→(1,3)→(2,4)→re-query (0,2): first occurrence 100% (trivial), repeated occurrence (reachable only through the accumulated 3-hop relay chain, since direct attention back to chunk 0/1 is blocked) **0.0% across all 3 test sequences** — complete, not partial, failure. Caveat: `repeat_query` is a trajectory shape `flow` was never trained on (only the fixed 3-query schedule), so this could reflect either "the relay doesn't preserve information across 3 hops" or "the model can't generalize to this novel trajectory shape at all" — the total (not gradual) failure leans toward the latter, but this test alone can't cleanly separate the two. Does not undermine `flow`'s strong same-schedule result above. `long_hop_recovery` (n_chunks=8) scored near-zero including first occurrences — the known length-extrapolation confound (trained at L=236), not an additional signal.
- **Vocab reorder reproducibility check** (`kvmem/configs/hmn_stage0_round0_single_vreorder.py` → `hmn_flow_vreorder.py`) — **running**. Chat tags now occupy 256-261 (fixed) and STATE occupies the tail from 262 (pure append growth) — smoke-tested to produce a byte-identical layout/length to the pre-reorder vocab, so this pair exists purely to confirm the relabeling doesn't change trainability or final numbers. `solo`/`flow`-equivalent checkpoints under the old vocab are kept until this is confirmed matching.
- **`squeeze`** (`kvmem/configs/hmn_squeeze_ca_n4.py` + `hmn_squeeze_random_n4.py`) — designed and queued, not yet run. Dedicated compression-capacity test (CA-structured vs. random-byte paired control). See `docs/HMN_RECIPE.md` §10.
- **`weave`** — position/mask/DSL/decode/eval harness all built (`chunk_positions_traj`/`chunk_mask_fb_traj`/`ar_decode_traj_nokv`/`kvmem/eval_weave.py`); the `weave_mix` training-loop dispatch into `train()` is not yet wired. See `docs/HMN_RECIPE.md` §4c.

---

## Structured-data track (queued, not yet used in training)

**Why**: genuine compression (zip/gzip-style, exploiting statistical redundancy) cannot emerge from training on the max-entropy random bytes used everywhere else in this project — Shannon's source coding theorem makes such data literally incompressible, so there's no redundancy for `STATE` to learn to exploit. Random-byte training only teaches raw lossless storage density and the addressing algorithm, not compression. Getting emergent compression requires structured/compressible training data.

**`kvmem/structured_data.py`** implements three generator families, each sampling **fresh random parameters per call** (required, not optional — a fixed rule across all examples lets the model bake it into static weights instead of encoding anything into `STATE`, the same FFN-as-static-knowledge failure mode this project's `dual_attn` design already avoids elsewhere):
- `gen_chaotic_logistic` — logistic map, random `r` in the chaotic regime
- `gen_fractal_midpoint` — 1D midpoint-displacement fractal, random Hurst exponent
- `gen_ca` — 1D cellular automaton, random rule table + random initial condition

**Recommendation: `gen_ca` (cellular automata) is the default**, confirmed empirically (byte-histogram entropy on a smoke test: chaotic=7.15 bits, fractal=7.13 bits — both nearly as high as pure-random's 8-bit max, since byte quantization washes out most of their structure; CA=2.87 bits — genuine, strong redundancy). CA is also discrete-native (no quantization ambiguity, unlike the two continuous-valued generators), exactly reproducible from pure integer ops, and has an enormous, easily-tunable rule space (`k_states`/`radius` control complexity directly). The other two stay implemented for a future ablation, not deleted.

**`target_bits` parameter**: all three generators (and `generate_structured_chunks`) accept `target_bits` — desired bits/byte of TRUE compressibility, measured via `measure_bits_per_byte` (min of raw-zlib and delta-then-zlib compressed size/byte — NOT marginal byte-histogram entropy, which misses sequential structure entirely: `"AAAABBBB"` and `"ABABABAB"` have identical histograms but very different compressibility). Calibration works via search (two-phase coarse-then-refine for the continuous generators' scalar knob; rejection sampling over full rule configs for CA, whose rule space isn't a scalar). **Known limitation, measured not assumed**: calibration accuracy varies a lot by generator and is seed-dependent — the logistic map's bifurcation structure is fractal/discontinuous enough that the same `target_bits=5.0` call lands anywhere from ~1 to ~5+ bits/byte depending on RNG seed even with `n_trials=60`; CA's rule-space distribution is bimodal/sparse in the middle (only ~3% of random k=2,r=1 rules land in a 1.5-2.5 band). Fractal calibrates most reliably of the three. Treat `target_bits` as "bias the search toward roughly this neighborhood," not a precise dial — a real fix would need precomputed lookup tables (e.g. a bifurcation diagram for the logistic map), not implemented, flagged as a genuine gap for whoever extends this next.

**Caution before using this for the chain-memory recovery probe specifically**: structured data risks contaminating that probe — a model could "recover" an earlier chain step's content by inferring the generating rule from its own visible span, without touching the relay at all. Keep the recovery probe on pure random data first; structured data is queued as a separate, later question (does bounded `STATE` capacity effectively increase when content is compressible), not a replacement for the current validation.

---

## Key Principles

- `null_kv=True` always — 1.5–2× faster, better bpb
- Mask convention: `0.0`=attend, `-1e9`=blocked (additive bias for `F.scaled_dot_product_attention`)
- Round 0 (IQ) before IR rounds — always required for the feedback mechanism
- The nochain blackout (each chain step's round-0 STATE blocked from ALL tokens in prior rec_blocks) is what makes chain steps independently trainable, and the relay exception (`flow`'s single-hop attention permission, `chunk_mask_fb_flow`/`chunk_mask_fb_traj`) is the sole sanctioned carve-out from it. Do not weaken either without understanding the consequences — the nochain blackout is what keeps chain steps from leaking raw content forward.
- **Report precisely, never round up** — state exactly what was measured (e.g. a padded/truncated excerpt), not the whole file
- **Verify infra before trusting it mid-run**: always check process liveness (`ps -p <pid>`) explicitly on every wake, not just log content — a silently-exited process produces no new log lines
- **Never run two training jobs at once**
- **Editing `kvmem/hmn.py` while a training job is running is safe** — Python has already loaded the module into the live process's memory; on-disk edits don't affect it. Verified multiple times this session (deleting `h_inject`, the vocab reorder) without disrupting an in-progress run.

---

## Docs

| What | Where |
|------|-------|
| **`docs/HMN_RECIPE.md`** — the primary detailed doc for the current architecture (terminology, IQ/IR unification, relay mechanics, staging/results, structured-data track, compression diagnostics, a classical/non-DNN alternatives discussion) | [`docs/HMN_RECIPE.md`](docs/HMN_RECIPE.md) |
| The rewrite plan (original design/approval record — every naming decision, worked `STATE_QUEUE` example predating the `flow` mechanism, why each choice was made) | [`/Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md`](/Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md) |
| Current implementation | [`kvmem/hmn.py`](kvmem/hmn.py) (single file), configs in [`kvmem/configs/`](kvmem/configs/), structured-data generators in [`kvmem/structured_data.py`](kvmem/structured_data.py), compression diagnostics in [`kvmem/eval_compression.py`](kvmem/eval_compression.py), trajectory-generalization diagnostics in [`kvmem/eval_weave.py`](kvmem/eval_weave.py) |
| Everything from before the rewrite (dual-attn discovery, RMSNorm, stitching, `juz1.txt` design, MDL theory, all prior architecture history — code AND docs) | [`archive_v1/`](archive_v1/) — old `kvmem/`, old `experiments/`, old `docs/` (`SRS_RECIPE.md`, `EARLY_ARCHITECTURE_HISTORY.md`, `MDL_MODEL_SIZE.md`, etc. all moved here, `docs/` at the repo root is a fresh start for this rewrite going forward) |
| Previous version of this file | [`archive_v1/CLAUDE_v1.md`](archive_v1/CLAUDE_v1.md) |
| Test set | [`datasets/suratalfatihah.txt`](datasets/suratalfatihah.txt) |
| `juz1` scaling target (not yet used in training) | [`datasets/juz1.txt`](datasets/juz1.txt) |
