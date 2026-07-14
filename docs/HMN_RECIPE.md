# HMN Recipe — `kvmem/hmn.py` (current architecture)

This is the primary detailed doc for the current implementation, written the
same session as the rewrite itself. For everything before this rewrite
(dual-attn discovery, RMSNorm, byte-level stitching, `juz1.txt` scaling
design, MDL theory, the full HMN v3/chunk-memorization/chat-tags history),
see [`archive_v1/docs/`](../archive_v1/docs/) — nothing was deleted, this
folder is a fresh start for documenting the current design going forward.
The original design/approval record for this rewrite (every naming decision,
worked examples, why each choice was made) is the plan file at
`/Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md` — this
doc restates and extends that record for permanence inside the repo.

---

## 1. Why this rewrite happened

The prior architecture's chat-tag vocabulary assigned a **separate tag token
per window** (`HMN_QUERY_A_OPEN` .. `HMN_QUERY_G_OPEN`, one pair per schedule
position, capped at 7 and requiring vocab growth beyond that). This is
backwards for a chat-formatted design — a real LLM reuses the same
`<|user|>`/`<|assistant|>` role tokens on every turn; turn identity comes
from *position* (RoPE) and *content*, never from a turn-numbered vocabulary
entry. Fixing this required retraining anyway (the vocabulary layout itself
changes), so this was also the point to build in bounded, persistent **chain
memory** (`STATE_QUEUE`) from the start, rather than bolt it on to a design
that had already proven itself on fully-independent windows.

## 2. Terminology

The word "step" was overloaded four different ways across the prior session's
discussions. Fixed vocabulary, used consistently in code and docs from here on:

| Concept | Term |
|---|---|
| token index within the packed sequence | **position** |
| schedule position in a `STATE_QUEUE`-chained schedule (old: "window A/B/C/G") | **chain step** (always two words, never bare "step") |
| byte range being recalled (e.g. `span=(0,2)`) | **span** |
| IQ vs IR pass within one chain step | **round** — round 0 = IQ, round *k*>0 = IR |
| SGD/optimizer iteration | **training step** |
| chat_tags-style weighted trajectory sampling | **trajectory** |
| compressed per-chunk/per-round register (old: "SLOT") | **STATE** (`HMN_STATE_0..3`, `state_len`, `state_vocab_size`, `_cyclic_state_ids`) |
| bounded memory carried chain-step-to-chain-step | **STATE_QUEUE** |

**Vocab is flat and shared**: `HMN_SRC_OPEN/CLOSE`, `HMN_QUERY_OPEN/CLOSE`,
`HMN_RESPONSE_OPEN/CLOSE` — three generic pairs, reused identically at every
chain step, no per-position variants. `HMN_TAG_VOCAB_SIZE = 274`.

**`<mem>` wrapper tags were dropped entirely** — a STATE region's content is
always the fixed `_cyclic_state_ids()` placeholder tokens (reserved IDs never
used anywhere else in the vocabulary), already unambiguous region markers on
their own. Wrapping them in boundary tags would add tokens with zero
additional information the model couldn't already get from the placeholder
IDs. This is different from `<src>`/`<query>`/`<response>`, which wrap
genuine byte-valued content that's indistinguishable from any other byte
region without an explicit tag — those tags stay. The rule: tag content that
needs disambiguation, skip tags for content that's already self-identifying.

**`STATE_QUEUE` reuses `HMN_STATE_0..3` directly — no separate token family.**
Position/mask already disambiguate `STATE_QUEUE_in` from a step's own fresh
`STATE` the same way encoding-pass STATE and recall-round STATE already share
identical placeholder tokens without confusion today. And `h_inject`
overwrites the `STATE_QUEUE_in` region's embedding immediately after
`_embed()`, before any transformer block runs — its placeholder identity is
never even seen by the model.

**Cyclic STATE fill**: `state_vocab_size` (not `state_len`) controls how many
*distinct* placeholder IDs exist; when `state_vocab_size < state_len`, the
alphabet repeats periodically (`_cyclic_state_ids(state_len=8,
state_vocab_size=2)` → `[S0,S1,S0,S1,S0,S1,S0,S1]`).

## 3. IQ/IR unification

Round 0 (IQ) is the `n_refine=0` special case of round *k*>0 (IR), not a
separate block type. Checking the actual mask rules: IQ's STATE and IR's
STATE_A/argmax/STATE_B are both blocked from raw chunks and other
rec-blocks' output by the same kind of rule — the only structural difference
is round 0 has no argmax-feedback segment, because there's no prior round
yet to feed back from. `_emit_round(round_idx, has_queue_in)` implements
both: `round_idx == 0` skips the argmax/STATE_A prefix; `round_idx > 0` adds
it. The underlying mask rules (Rule 3b nochain, Rules 5-8) are unchanged —
this is a naming/API unification, not new masking logic.

## 4. `STATE_QUEUE` mechanics

**Layout**: each chain step after the first gets a `STATE_QUEUE_in` region
of width `M*state_len` (default `M=1`) immediately before that chain step's
round-0 STATE region.

**Masking**: `STATE_QUEUE_in` joins the chain step's own "own content" set
(same treatment as STATE/warmup/response) — folded into the existing Rule
4a/4b unions, no new rule. Still fully blocked from raw chunk content and
other chain steps' regions (Rule 3b unchanged) — the *only* channel for
cross-chain-step information is the injected feature vector, never an
attention path.

**Data flow (`h_inject`)**: for chain step *i* > 0:
1. Run chain step *i-1*'s forward pass with `return_features=True`.
2. Extract the residual-stream slice at chain step *i-1*'s **last round's**
   own STATE positions (round 0's `sl0/sl1` if that chain step had
   `n_refine=0`, else the final IR round's `slb0/slb1`).
3. Run chain step *i*'s forward pass with
   `h_inject={(queue0, queue1): <that slice>}`.

This breaks the old "2 forward passes for the whole packed sequence" trick
for chained stages — chain step *i*'s input genuinely depends on chain step
*i-1*'s computed output, not just its decoded bytes, so a chained stage needs
one sequential forward pass per chain step. Non-chained stages (`chain`
absent/`False`) keep the old fast path unchanged.

**`STATE_QUEUE` is a single-hop relay, not an accumulating buffer.** `M=1`
means chain step *i*'s `STATE_QUEUE_in` comes ONLY from chain step *i-1*'s
own last STATE — never from *i-2* directly. There is no separate "older
states" store to mask or discard: raw content from chain steps older than
*i-1* is already fully blocked by Rule 3b regardless of `STATE_QUEUE` (that
invariant predates this mechanism). For information from chain step *i-2* to
reach chain step *i*, chain step *i-1* must have implicitly folded it into
its own single `state_len`-wide STATE when producing its own output — there
is no guarantee this happens; it's exactly what the chain-memory recovery
probe (§7) is designed to test.

**Known gradient-flow caveat**: `h_inject`'s value is `.detach()`ed
(truncated BPTT across chain steps) — the model gets no direct gradient
signal for "make this STATE useful to a *future* chain step," only for its
own chain step's recall loss. If the recovery probe fails, this is a
candidate explanation before concluding the mechanism doesn't work at all.

## 5. Block types

One `HMNModel` class, selected via `block_type`:

| `block_type` | Structure | Role |
|---|---|---|
| `attn_mlp` | `x = x + attn(norm1(x)); x = x + ffn(norm2(x))` | standard architecture, comparison baseline |
| `dual_attn` | `x = x + attn1(norm1(x)); x = x + attn2(norm2(x))` (paired, no MLP) | available ablation option, byte-identical port of the prior architecture — no longer required for checkpoint compatibility since this is a from-scratch retrain |
| `single_attn` | `x = x + attn(norm(x))` (one attn, one norm, no MLP) | **the default** — same block repeated `n_layers` times; use `n_layers` = 2× the equivalent `dual_attn` config's `n_layers` to match total attention-op count |

## 6. Current staging and results

- **Stage 0** (`kvmem/configs/hmn_stage0_round0_single.py`) — one chain step,
  round 0 only, no `STATE_QUEUE`. **Done**: 160000/160000 steps, final val
  per-span MEAN=94.4% (best checkpoint 97.2% at step 150000), test=100%,
  loss=0.017 — matches the historical ~100% single-window IQ ceiling.
- **Stage 1** (`kvmem/configs/hmn_stage1_round0_chained.py`) — three chain
  steps (`[(0,2),(1,3),(2,4)]`), round 0 only, `chain=True`, warm-started
  from Stage 0. **In progress** — see `CLAUDE.md` for the live progress table
  (updated as the run continues; this doc is not re-edited per checkpoint).
  Chain step 0 (no `STATE_QUEUE` dependency) strong from the start. Chain
  step 1 climbed steadily. Chain step 2 stayed near zero through step 40000,
  then broke out (loss dropped sharply in the step ~42000-49000 window) and
  has climbed on most checkpoints since — early evidence the 2-hop relay is
  being learned, not stalled, but not yet confirmed by the recovery probe.
- IR rounds with chain, larger `M`, and the sparse block-attention
  memory-bank generalization are deferred to later stages.

**Eval metric note**: `STITCHED_MEAN` is only meaningful when the
`chain_steps` schedule covers the *entire* `n_chunks*chunk_len` source (true
for Stage 1, NOT true for Stage 0's single-chain-step schedule, where it's
capped at ~42.9% by construction since the untested chunks are never
decoded). Watch `span/MEAN` for single-chain-step configs.

**Zero-shot stitching is already exercised by every eval, not a separate
test.** `ar_decode_srs_stitched_tagged_nokv` seeds only chain step 0's
warmup from ground truth; every later chain step's warmup comes from the
model's own previously-decoded bytes. This is the same proven byte-level
stitching mechanism from the prior architecture — distinct from and doesn't
test `STATE_QUEUE` specifically (see §7).

## 7. Chain-memory recovery probe (queued, not yet built)

Per-chain-step recall accuracy alone does NOT prove `STATE_QUEUE` carried
anything forward — each chain step can solve its own span locally from
encoding-block STATEs, regardless of chaining. The real test: run the *last*
chain step's round-0 recall on a query that requires recovering an *earlier*
chain step's span — something only reachable via the accumulated
`STATE_QUEUE` chain, since direct cross-chain-step attention stays blocked
(Rule 3b). Not yet implemented.

## 8. Structured-data track (`kvmem/structured_data.py`)

**Motivation**: genuine compression (zip/gzip-style, exploiting statistical
redundancy) cannot emerge from training on the max-entropy random bytes used
everywhere else in this project — Shannon's source coding theorem makes such
data literally incompressible, so there's no redundancy for `STATE` to learn
to exploit. Random-byte training only teaches raw lossless storage density
and the addressing algorithm, not compression.

**Three generator families**, each sampling fresh random parameters per call
(required — a fixed rule across all examples would let the model bake it
into static weights instead of `STATE`, the same FFN-as-static-knowledge
failure mode the `dual_attn` design already avoids elsewhere):
- `gen_chaotic_logistic` — logistic map, random `r`.
- `gen_fractal_midpoint` — 1D midpoint-displacement fractal, random Hurst exponent.
- `gen_ca` — 1D cellular automaton, random rule table + initial condition.
  **Recommended default** — discrete-native (no quantization ambiguity
  unlike the two continuous generators), exactly reproducible from pure
  integer ops, enormous tunable rule space. Confirmed empirically: raw
  byte-histogram entropy for chaotic/fractal is ~7.1-7.15 bits (nearly max,
  since byte quantization washes out their structure at the histogram
  level); CA is 2.87 bits (genuine redundancy).

**`target_bits`**: all three generators accept a target bits/byte
(calibrated via `measure_bits_per_byte` — zlib on raw AND delta-encoded
bytes, min of the two, since raw zlib badly underestimates smooth-signal
compressibility). Calibration reliability varies and is **measured, not
assumed**: fractal calibrates most reliably (two-phase coarse-then-refine
search over its scalar Hurst knob); chaotic is seed-dependent (the logistic
map's bifurcation structure is fractal/discontinuous — bits/byte can jump
from ~1.5 to ~6 between r=3.63 and r=3.64); CA's rule-space distribution is
bimodal/sparse in the middle (only ~3% of random k=2,r=1 rules land in a
1.5-2.5 bits/byte band). Treat `target_bits` as "bias toward roughly this
neighborhood," not a precise dial — a real fix would need precomputed
lookup tables, not implemented.

## 9. Compression-quality diagnostics (`kvmem/eval_compression.py`)

Test-time methodology for separating "genuinely compresses in `STATE`" from
"memorized via weights" or "hasn't learned anything," using information
theory rather than raw loss numbers. Checkpoint-agnostic — works on any
`kvmem/hmn.py` checkpoint, including one trained only on random bytes (the
compression-sensitivity curve, §9.4, is specifically a zero-shot OOD test).

Run in order — each gates the next:

1. **`state_ablation_gate`** — overrides the encoding-pass STATE region with
   noise via `h_inject`, compares teacher-forced bits/byte with vs without.
   A large gap confirms recall genuinely depends on STATE (necessary, not
   sufficient, for any later compression claim). Smoke-tested against
   Stage 0's checkpoint: gap of 26.37 bits/byte (0.01 normal → 26.38
   ablated) — clean confirmation the ablation mechanism and the model's
   STATE-dependence both work.
2. **`floor_comparison`** — `L_model` vs. chance floor (~8 bits/byte) and
   zlib's practical floor on the same sequence (underfitting check).
3. **`train_test_gap`** — approximate overfitting check: variance of
   `L_model` across multiple held-out (never-generated-before) rule/seed
   draws at fixed `target_bits`. High variance suggests inconsistent
   generalization; this script has no access to actual training-loss logs,
   so it can't do a literal train-vs-test comparison — documented as
   approximate, not overclaimed.
4. **`compression_sensitivity_curve`** — sweeps `target_bits` at FIXED
   `chunk_len`/`state_len` (the checkpoint's own trained layout, no
   extrapolation confound) — the main "is compression happening" signal.
   Smoke-tested against Stage 0: flat at ~0.01-0.05 bits/byte across every
   `target_bits` level including pure random. **This flat result is
   informative, not a null result** — Stage 0 already achieves near-100%
   raw memorization at `chunk_len=16`, so there's no capacity pressure for
   compression to matter yet; the curve would only show sensitivity once
   raw capacity is exceeded (see §9.5).
5. **`sweep_max_recallable_length`** (optional) — varies `chunk_len` itself
   to find the longest chunk still recalled near-perfectly, separately for
   random and structured content, giving a direct Effective Compression
   Ratio (`ECR = max_len_structured / max_len_random`, vs. theoretical
   ceiling `8/target_bits`). Flagged as needing chunk_len extrapolation the
   model wasn't necessarily trained for — treat with more caution than 1-4.
   Reports two capacity normalizations so results are comparable across
   model sizes: `capacity_bits_per_state_token` (empirical bits/byte stored
   for incompressible content, divided by `state_len`) and
   `capacity_bits_per_million_params` (same numerator, divided by total
   model parameter count) — the latter exists because a bigger model could
   show higher raw capacity purely from more weight-capacity to hard-code
   patterns, independent of whether `STATE` itself improved (the same
   concern `docs/MDL_MODEL_SIZE.md`'s archived analysis raises about model
   size tracking algorithm complexity, not being a free source of apparent
   capability).

**Next step for this track**: none of Stage 0/1 were trained on structured
data, so diagnostics 3/4 are currently zero-shot-only tests of whatever OOD
generalization happens to exist. A real test of learned compression needs a
dedicated training stage using `kvmem/structured_data.py`, likely at a
`chunk_len` chosen specifically to exceed Stage 0/1's proven raw-capacity
ceiling (so compression has to matter for recall to succeed at all) — not
yet built, next in queue after the chain-memory recovery probe.
