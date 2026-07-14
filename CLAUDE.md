# kvmem

Fast-weight language model — HashMemNet (HMN). **Current focus: `kvmem/hmn.py`**, a from-scratch consolidated single-file rewrite (single-attn blocks + shared chat-tag vocab + `STATE_QUEUE` chain memory), replacing the prior multi-file `kvmem/`+`experiments/` stack. All prior code/docs/checkpoints are preserved verbatim under `archive_v1/` (old `kvmem/`, old `experiments/`, and `archive_v1/CLAUDE_v1.md` — the previous version of this file) — nothing was deleted, just superseded. `archive_v1/` code still runs standalone via `PYTHONPATH=archive_v1`.

**Why the rewrite**: a design review caught that the old chat-tag vocabulary assigned a separate tag token per window (`HMN_QUERY_A_OPEN`..`_G_OPEN`) — backwards for a chat-formatted design (a real LLM reuses the same role tokens every turn; turn identity comes from position, not a turn-numbered vocab entry). Fixing it required retraining anyway, so this was also the point to build in bounded, persistent chain memory (`STATE_QUEUE`) from the start instead of bolting it on later. Full rationale, worked examples, and every naming decision: [`/Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md`](/Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md) (the approved plan — read this first to understand *why* things are named the way they are).

---

## Terminology (read this before the code — "step" alone is retired, it was overloaded 4 ways)

| Concept | Term |
|---|---|
| token index within the packed sequence | **position** |
| schedule position in a `STATE_QUEUE`-chained schedule (old: "window A/B/C/G") | **chain step** (always two words) |
| byte range being recalled (e.g. `span=(0,2)`) | **span** |
| IQ vs IR pass within one chain step | **round** — round 0 = IQ, round *k*>0 = IR (unified into one `_emit_round(round_idx, ...)`, not two separate block types) |
| SGD/optimizer iteration (`global_step`/`local_step`) | **training step** |
| chat_tags-style weighted trajectory sampling | **trajectory** |
| the compressed per-chunk/per-round register (old: "SLOT") | **STATE** (`HMN_STATE_0..3`, `state_len`, `state_vocab_size`, `_cyclic_state_ids`) |
| bounded memory carried chain-step-to-chain-step | **STATE_QUEUE** — reuses `HMN_STATE_0..3` directly (no separate token family, no wrapper tag — position/mask already disambiguate it, and `h_inject` overwrites its embedding before any block runs anyway) |

**Vocab is now flat and shared**: `HMN_SRC_OPEN/CLOSE`, `HMN_QUERY_OPEN/CLOSE`, `HMN_RESPONSE_OPEN/CLOSE` — three generic pairs, reused identically at every chain step, no per-position variants. `<mem>` wrapper tags were dropped entirely (redundant — STATE regions are already self-identifying via their reserved placeholder tokens, unlike `<src>`/`<query>`/`<response>` which wrap genuine byte-valued content that needs disambiguation). `HMN_TAG_VOCAB_SIZE = 274`.

---

## Architecture in plain terms

**The task** (unchanged from every prior architecture in this project): memorize a byte sequence, then recall it from a short seed (warmup), byte-exact.

**Block types** — one unified `HMNModel` class, selected via `block_type` hp:

| `block_type` | Structure | Role |
|---|---|---|
| `attn_mlp` | `x = x + attn(norm1(x)); x = x + ffn(norm2(x))` | standard architecture, for comparison |
| `dual_attn` | `x = x + attn1(norm1(x)); x = x + attn2(norm2(x))` (paired, no MLP) | kept as an available ablation option (byte-identical port of the prior architecture) — no longer required for checkpoint compatibility since this is a from-scratch retrain |
| `single_attn` | `x = x + attn(norm(x))` (one attn, one norm, no MLP) | **the default going forward** — same block repeated `n_layers` times; use `n_layers` = 2× the equivalent `dual_attn` config to match total attention-op count |

**Chain memory (`STATE_QUEUE`)**: each chain step after the first carries a bounded memory region (default `M=1`, one `state_len`-wide block) from the previous chain step's own last-round STATE, injected via `h_inject` (overrides the embedding at fixed positions before the first transformer block runs — pre-existing mechanism, reused unmodified). The nochain invariant (nothing in the mask lets one chain step attend directly into another's content) still holds — `STATE_QUEUE` is the *only* sanctioned cross-chain-step channel, and it's a feature-vector injection, not an attention path. This makes a chained stage need one sequential forward pass per chain step (instead of the old "2 passes for the whole packed sequence" trick) — an explicit, accepted cost increase, only for stages with `chain=True`.

**Current staging** (retrained from scratch on the new vocab — old numbers don't carry over, vocab changed):
- **Stage 0** (`kvmem/configs/hmn_stage0_round0_single.py`) — one chain step, round 0 only, no `STATE_QUEUE`. Establishes basic state-compression before any accumulation complexity. **Done**: 160000/160000 steps, final val per-span MEAN=94.4% (best checkpoint 97.2% at step 150000), test=100%, loss=0.017 — matches the historical ~100% single-window IQ ceiling, sanity bar cleared.
- **Stage 1** (`kvmem/configs/hmn_stage1_round0_chained.py`) — three chain steps (`[(0,2),(1,3),(2,4)]`), round 0 only, `chain=True`, warm-started from Stage 0. **Running** (started this session, ~10 it/s with intermittent stalls from MPS sleep/wake cycling despite `caffeinate` — same known issue as before, not a code bug). Progress table (per-chain-step val/test match%; chain step 0 has no `STATE_QUEUE`, steps 1-2 do):

  | training step | span(0,2) val/test | span(1,3) val/test | span(2,4) val/test | loss |
  |---|---|---|---|---|
  | 10000 | 87.5% / 100.0% | 18.1% / 37.5% | 1.4% / 8.3% | 2.965 |
  | 20000 | 91.7% / 87.5% | 23.6% / 29.2% | 0.0% / 0.0% | 2.871 |
  | 30000 | 97.2% / 100.0% | 23.6% / 33.3% | 0.0% / 4.2% | 2.829 |
  | 40000 | 90.3% / 100.0% | 26.4% / 37.5% | 1.4% / 4.2% | 2.766 |
  | 50000 | 98.6% / 100.0% | 29.2% / 41.7% | 2.8% / 12.5% | 2.175 |
  | 60000 | 98.6% / 100.0% | 26.4% / 45.8% | 8.3% / 16.7% | 1.938 |

  Chain step 0 (no memory dependency, inherited from warm start) has been strong throughout. Chain step 1 climbed steadily. Chain step 2 was stuck near 0% through step 40000, then broke out starting ~step 42000-49000 (loss dropped 2.71→2.18 in that window) and has climbed on every checkpoint since — the first real evidence the 2-hop `STATE_QUEUE` relay is being learned, not stalled. Still needs the actual chain-memory-specific validation below before calling it confirmed.
  - **Note on eval metrics**: `val/srs/STITCHED_MEAN` is only meaningful when the `chain_steps` schedule covers the *entire* `n_chunks*chunk_len` source (true for Stage 1's 3-chain-step schedule, NOT true for Stage 0's single-chain-step schedule — there it's capped at ~42.9% by construction since chunks 2-3 are never decoded; watch `span/MEAN` instead for single-chain-step configs).
  - **Zero-shot stitching is already exercised by every eval, not a separate test**: `ar_decode_srs_stitched_tagged_nokv` seeds only chain step 0's warmup from ground truth; every later chain step's warmup comes from the model's own previously-decoded bytes. The per-chain-step numbers above already reflect end-to-end zero-shot decode chaining (same proven mechanism as the pre-rewrite architecture) — this is distinct from and doesn't test `STATE_QUEUE` specifically (see next point).
- **Chain-memory recovery probe** (queued, not yet built): the actual test of whether `STATE_QUEUE` carries anything useful — ask the *last* chain step's round-0 recall to reach an *earlier* chain step's span (not its own). Per-chain-step accuracy alone doesn't prove this; each chain step can solve its own span locally. `STATE_QUEUE` is a **single-hop relay, not an accumulating buffer** (M=1 — chain step *i*'s `STATE_QUEUE_in` comes only from chain step *i-1*'s own last STATE, never *i-2* directly); for information to survive multiple hops, each intermediate chain step must implicitly fold it into its own bottlenecked STATE. See `chunk_positions_chained`'s docstring in `kvmem/hmn.py` for the full mechanics.
- IR rounds (`n_refine>0`) with chain, larger `M`, and the sparse block-attention memory-bank generalization are all deferred to later stages — see the plan doc.

**Known caveat to watch for**: the chained training loop uses `.detach()` on the injected `STATE_QUEUE` feature (truncated BPTT across chain steps) — the model gets no direct gradient signal for "make this STATE useful to a *future* chain step," only for its own chain step's recall loss. If the chain-recovery probe fails, this is a candidate explanation before concluding the mechanism doesn't work at all.

---

## Structured-data track (queued, not yet used in training)

**Why**: genuine compression (zip/gzip-style, exploiting statistical redundancy) cannot emerge from training on the max-entropy random bytes used everywhere else in this project — Shannon's source coding theorem makes such data literally incompressible, so there's no redundancy for `STATE` to learn to exploit. Random-byte training only teaches raw lossless storage density and the addressing algorithm, not compression. Getting emergent compression requires structured/compressible training data.

**`kvmem/structured_data.py`** implements three generator families, each sampling **fresh random parameters per call** (required, not optional — a fixed rule across all examples lets the model bake it into static weights instead of encoding anything into `STATE`, the same FFN-as-static-knowledge failure mode this project's `dual_attn` design already avoids elsewhere):
- `gen_chaotic_logistic` — logistic map, random `r` in the chaotic regime
- `gen_fractal_midpoint` — 1D midpoint-displacement fractal, random Hurst exponent
- `gen_ca` — 1D cellular automaton, random rule table + random initial condition

**Recommendation: `gen_ca` (cellular automata) is the default**, confirmed empirically (byte-histogram entropy on a smoke test: chaotic=7.15 bits, fractal=7.13 bits — both nearly as high as pure-random's 8-bit max, since byte quantization washes out most of their structure; CA=2.87 bits — genuine, strong redundancy). CA is also discrete-native (no quantization ambiguity, unlike the two continuous-valued generators), exactly reproducible from pure integer ops, and has an enormous, easily-tunable rule space (`k_states`/`radius` control complexity directly). The other two stay implemented for a future ablation, not deleted.

**`target_bits` parameter**: all three generators (and `generate_structured_chunks`) accept `target_bits` — desired bits/byte of TRUE compressibility, measured via `measure_bits_per_byte` (min of raw-zlib and delta-then-zlib compressed size/byte — NOT marginal byte-histogram entropy, which misses sequential structure entirely: `"AAAABBBB"` and `"ABABABAB"` have identical histograms but very different compressibility). Calibration works via search (two-phase coarse-then-refine for the continuous generators' scalar knob; rejection sampling over full rule configs for CA, whose rule space isn't a scalar). **Known limitation, measured not assumed**: calibration accuracy varies a lot by generator and is seed-dependent — the logistic map's bifurcation structure is fractal/discontinuous enough that the same `target_bits=5.0` call lands anywhere from ~1 to ~5+ bits/byte depending on RNG seed even with `n_trials=60`; CA's rule-space distribution is bimodal/sparse in the middle (only ~3% of random k=2,r=1 rules land in a 1.5-2.5 band). Fractal calibrates most reliably of the three. Treat `target_bits` as "bias the search toward roughly this neighborhood," not a precise dial — a real fix would need precomputed lookup tables (e.g. a bifurcation diagram for the logistic map), not implemented, flagged as a genuine gap for whoever extends this next.

**Caution before using this for the chain-memory recovery probe specifically**: structured data risks contaminating that probe — a model could "recover" an earlier chain step's content by inferring the generating rule from its own visible span, without touching `STATE_QUEUE` at all. Keep the recovery probe on pure random data first; structured data is queued as a separate, later question (does bounded `STATE` capacity effectively increase when content is compressible), not a replacement for the current validation.

---

## Key Principles (still apply from the prior architecture)

- `null_kv=True` always — 1.5–2× faster, better bpb
- Mask convention: `0.0`=attend, `-1e9`=blocked (additive bias for `F.scaled_dot_product_attention`)
- Round 0 (IQ) before IR rounds — always required for the feedback mechanism
- Rule 3b (nochain masking) is what makes chain steps independently trainable in the old (non-chained) design AND what STATE_QUEUE's cross-chain-step channel is scoped around — do not weaken it without understanding the consequences
- **Report precisely, never round up** — state exactly what was measured (e.g. a padded/truncated excerpt), not the whole file
- **Verify infra before trusting it mid-run**: always check process liveness (`ps -p <pid>`) explicitly on every wake, not just log content — a silently-exited process produces no new log lines
- **Never run two training jobs at once**

---

## Docs

| What | Where |
|------|-------|
| **`docs/HMN_RECIPE.md`** — the primary detailed doc for the current architecture (terminology, IQ/IR unification, STATE_QUEUE mechanics, staging/results, structured-data track, compression diagnostics) | [`docs/HMN_RECIPE.md`](docs/HMN_RECIPE.md) |
| The rewrite plan (original design/approval record — every naming decision, worked STATE_QUEUE example, why each choice was made) | [`/Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md`](/Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md) |
| Current implementation | [`kvmem/hmn.py`](kvmem/hmn.py) (single file), configs in [`kvmem/configs/`](kvmem/configs/), structured-data generators in [`kvmem/structured_data.py`](kvmem/structured_data.py), compression diagnostics in [`kvmem/eval_compression.py`](kvmem/eval_compression.py) |
| Everything from before the rewrite (dual-attn discovery, RMSNorm, stitching, `juz1.txt` design, MDL theory, all prior architecture history — code AND docs) | [`archive_v1/`](archive_v1/) — old `kvmem/`, old `experiments/`, old `docs/` (`SRS_RECIPE.md`, `EARLY_ARCHITECTURE_HISTORY.md`, `MDL_MODEL_SIZE.md`, etc. all moved here, `docs/` at the repo root is a fresh start for this rewrite going forward) |
| Previous version of this file | [`archive_v1/CLAUDE_v1.md`](archive_v1/CLAUDE_v1.md) |
| Test set | [`datasets/suratalfatihah.txt`](datasets/suratalfatihah.txt) |
| `juz1` scaling target (not yet used in training) | [`datasets/juz1.txt`](datasets/juz1.txt) |
