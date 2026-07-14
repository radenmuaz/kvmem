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

**Stage names** — short, descriptive, tied to what each stage mechanically
does rather than a bare index (numbers alone don't convey what changed
between them, and the old "IQ"/"IR" naming is retired per the round
unification in §3):

| Old | Name | What it tests |
|---|---|---|
| Stage 0 | **`solo`** | one chain-step, nothing to relay yet — the bootstrap case |
| Stage 1 | **`relay`** | multi chain-step, `STATE_QUEUE` single-hop relay via `h_inject` (forced copy, detached gradient) |
| (new, §4b) | **`flow`** | same single-hop relay, but via a learned attention permission instead of a forced copy — full gradient flow, cheaper training too |
| (deferred) | **`refine`** | IR rounds added on top of `relay`'s/`flow`'s chaining |
| (deferred) | **`bank`** | generalizes the single-slot queue into a proper memory bank |
| (new, §10) | **`squeeze`** | does `STATE` genuinely compress compressible content, not just store it |

Already-running/finished work keeps its existing file/log names as-is (config
files, log directories) to avoid touching anything mid-run — this table is
the canonical name mapping. New work uses the new names directly, starting
with `squeeze` (§10).

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
it. The underlying mask rules (the nochain blackout, IR feedback isolation) are unchanged —
this is a naming/API unification, not new masking logic.

## 4. `STATE_QUEUE` mechanics

**Layout**: each chain step after the first gets a `STATE_QUEUE_in` region
of width `M*state_len` (default `M=1`) immediately before that chain step's
round-0 STATE region.

**Masking**: `STATE_QUEUE_in` joins the chain step's own "own content" set
(same treatment as STATE/warmup/response) — folded into the existing warmup-
bottleneck/output-bottleneck unions, no new rule. Still fully blocked from
raw chunk content and other chain steps' regions (the nochain blackout
unchanged) — the *only* channel for cross-chain-step information is the
injected feature vector, never an attention path.

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
*i-1* is already fully blocked by the nochain blackout regardless of
`STATE_QUEUE` (that invariant predates this mechanism). For information from chain step *i-2* to
reach chain step *i*, chain step *i-1* must have implicitly folded it into
its own single `state_len`-wide STATE when producing its own output — there
is no guarantee this happens; it's exactly what the chain-memory recovery
probe (§7) is designed to test.

**Known gradient-flow caveat**: `h_inject`'s value is `.detach()`ed
(truncated BPTT across chain steps) — the model gets no direct gradient
signal for "make this STATE useful to a *future* chain step," only for its
own chain step's recall loss. If the recovery probe fails, this is a
candidate explanation before concluding the mechanism doesn't work at all.

## 4b. Stage `flow` — attention-based relay, no forced copy (designed, queued, code built and smoke-tested)

Direct alternative to `relay`'s `h_inject` copy, built specifically to fix
the gradient-flow caveat above. Instead of forcibly overwriting chain step
*i*'s input with chain step *i-1*'s extracted STATE, `flow` grants chain
step *i*'s own round-0 STATE row a narrow, single-hop **attention
permission**: it can read chain step *i-1*'s STATE columns directly (still
blocked from *i-1*'s warmup/response/raw chunks, and still blocked from
*i-2* or earlier — same single-hop information budget as `relay`, M=1 in
`STATE_QUEUE` terms). The model *learns* what to preserve via ordinary
gradient descent — full gradient flow across chain steps, no `.detach()`,
no forced copy.

**Implementation** (`kvmem/hmn.py`):
- `chunk_positions_flow` — same overall shape as `chunk_positions_chained`
  but never allocates a separate `STATE_QUEUE_in` region; chain step *i*'s
  own STATE serves double duty as both its own recall register and the
  thing chain step *i+1* reads.
- `chunk_mask_fb_flow` — copy of `chunk_mask_fb` with exactly one change:
  the nochain blackout's `prior_all` (which blocks a round-0 STATE row from
  all prior chain steps' content) gets a carved-out exception (the relay
  exception) for the immediately preceding chain step's own last-round STATE
  columns specifically. Kept as a fully separate function rather than
  modifying the proven `chunk_mask_fb` in place.
- `train()` dispatch: `stage['flow']=True` (mutually exclusive with
  `stage['chain']`) builds the layout via the functions above and — since
  the relay is now resolved entirely by mask permissions within one forward
  pass — always routes through the existing fast, non-sequential training
  path (no `h_inject` orchestration loop needed at all). This also means
  `flow` should be *cheaper per step* than `relay`, not just
  gradient-cleaner — a real efficiency side benefit, not just a correctness
  fix.

**Verified before queueing** (smoke test, `n_chunks=4, chunk_len=8,
state_len=4, chain_steps=[(0,2),(1,3),(2,4)]`): chain step 1's STATE row
sees chain step 0's STATE columns directly (visible) but not its
warmup/response (blocked); chain step 2's STATE row sees chain step 1's
STATE (visible) but NOT chain step 0's STATE directly (blocked — single-hop
preserved, must relay through chain step 1); encoding isolation and the
relay exception being scoped to the STATE row only (not warmup/
response rows) both hold. Full `train()` path (3 steps, tiny model) ran
end-to-end without error.

**Config**: `kvmem/configs/hmn_flow.py` — identical hyperparameters to
`relay` (same `d`, `n_layers`, `chain_steps`, step budget, warm-started from
`solo`) so the two are a direct, apples-to-apples comparison of *learning
mechanism* (copy-and-detach vs. learned-attention-with-full-gradient), not
information budget (both single-hop, both one `state_len`-wide channel).

**Queued, to run immediately once `relay` finishes** (never two jobs at
once). Headline comparison once both are done: per-chain-step match% at
convergence, and specifically whether chain step 2 (the 2-hop case) shows a
clearer/faster improvement under `flow` than under `relay` — direct evidence
the gradient-flow fix matters, if so.

**Deferred cleanup, conditional on that comparison**: if `flow` matches or
beats `relay`, delete the `h_inject`-relay path entirely — `HMNModel.
forward`'s `h_inject` parameter, `train()`'s `chain=True` sequential
per-chain-step training loop, `chunk_positions_chained`'s `STATE_QUEUE_in`
allocation, and `chunk_mask_fb`'s queue-related mask rules. Not done
preemptively — this is exactly the comparison that determines whether it's
warranted; if `flow` underperforms for some reason, `h_inject` stays as the
working mechanism instead.

## 4c. Stage `weave` — arbitrary interleaved trajectories (built, test harness complete, training dispatch not yet wired)

`solo`/`relay`/`flow` all train on exactly one fixed rhythm: encode
everything, then query in a fixed order. That's one point in a much larger
space of possible encode/query interleavings, and the project's own vision
(reading a document incrementally, answering questions as you go) needs the
model to work correctly under others too. `weave` generalizes
`chunk_positions_flow` to arbitrary interleaved operation sequences via a
compact DSL, and `kvmem/eval_weave.py` provides the corresponding test
harness (zero-shot against any existing checkpoint).

**Primitive vocabulary — three orthogonal operations** (revised from an
initial two-op design after review — see below):
- `E` — ingest one chunk's raw bytes only (`<src>chunk</src>`). Emits no
  STATE by itself.
- `S` — emit one `state_len`-wide STATE region. Its ROLE is determined
  entirely by adjacency, not by the token itself: if an unclaimed `E`
  immediately precedes it, this IS that chunk's own encoding-STATE (encoding
  isolation — sees its own chunk's raw bytes, blocked from every other chunk's raw
  bytes, not part of the relay chain — same treatment the shared encoding
  pass always had). Otherwise it's a **no-op relay hop** — blocked from all
  raw chunks, single-hop relay-only visibility into the immediately
  preceding `Q`-or-bare-`S`'s own STATE, no local recall target,
  `is_clean=False` (contributes nothing to loss directly — gradient reaches
  it only through whatever LATER op depends on it).
- `Q(s,e)` — query/recall span `[s,e)` (unchanged from `flow`).

**Why `E`+`S` instead of one bundled "encode" op, and no separate `N` op
type**: an earlier draft of this design had `E` implicitly bundle its own
STATE emission (matching `chunk_positions_chained`/`chunk_positions_flow`'s
existing convention) and a separate `N` (no-op) op type for relay-only hops.
Collapsing these into `E`+`S`+`Q` is a strict simplification, not just
renaming: it makes the compression step a first-class, visible thing in the
operations list (an `E` MUST be immediately followed by `S`, asserted by
`chunk_positions_traj`, not silently implied), and recognizes that a no-op
IS just "an `S` with no preceding unclaimed `E`" — the exact same STATE-
emission primitive, not a fourth kind of thing. Fewer primitives, same
expressiveness.

**Is a no-op operation useful at all?** Yes, decided deliberately, not
assumed: a no-op isolates *pure relay decay rate* from *recall-accuracy-at-
each-hop*, which `repeat_query`/`long_hop_recovery`'s intermediate hops
conflate (a failure there could mean the relay lost information OR that
hop's own local recall task failed for unrelated reasons — two different
things). A no-op has no local task to fail at, so a chain like `Q(0,2) S S
S S Q(0,2)` isolates decay cleanly, and — since no-ops carry no extra
warmup/response tokens and contribute no extra loss terms — they're cheap to
stretch arbitrarily far, letting `traj_decay_curve` test much longer hop
counts than adding more real queries could afford.

**Trajectory DSL** (compact string notation, one call to `parse_traj_dsl`
builds any pattern — see `kvmem/hmn.py`'s grammar comment for full details):
`E`/`E<n>` (ingest, `E<n>` expands to `n` explicit `E S` pairs, never a
bundled op), `S`/`S<n>` (emit state — role determined by adjacency, per
above), `Q(s,e)` (query span). Every named pattern below is one DSL string:

| Pattern | DSL | Trains on? |
|---|---|---|
| `batch` (= `relay`/`flow`'s fixed rhythm) | `E4 Q(0,2) Q(1,3) Q(2,4)` | yes |
| `stream` (query as soon as dependencies are met) | `E2 Q(0,2) E S Q(1,3) E S Q(2,4)` | yes |
| `interleave_delayed` (queries in shuffled/non-monotonic order) | `E4 Q(2,4) Q(1,3) Q(0,2)` | yes |
| `repeat_query` (same span queried twice) | `E4 Q(0,2) Q(1,3) Q(2,4) Q(0,2)` | **test-only** |
| `long_hop_recovery` (= `repeat_query` at larger `n_chunks`, e.g. 8) | same shape, more chunks | **test-only** |
| `decay_curve` (pure no-op decay, cheap to stretch) | `E2 Q(0,2) S4 Q(0,2)` | test-only, or train-mix at low hop counts |

`batch`/`stream`/`interleave_delayed` are TRAIN-mix candidates (same
generalization principle already used for `warmup_x_dist='uniform'` in the
old architecture — train on varied conditions, not one fixed rhythm — now
applied to *operation order*). `repeat_query`/`long_hop_recovery` must stay
test-only: training on them would defeat their purpose as generalization
probes (they specifically test whether the checkpoint works *beyond*
whatever rhythm it was trained on).

**Implementation** (`kvmem/hmn.py`): `chunk_positions_traj` (position
builder, `enc_blocks` keyed by chunk_idx since `E`/`S` pairs can occur in
any order interspersed with `Q` ops), `chunk_mask_fb_traj` (mask — same
encoding-isolation/chunk-blackout/relay-exception/IR-feedback-isolation logic as `chunk_mask_fb_flow`, generalized to group by
`op_idx`, the i-th `Q`-or-bare-`S` op, instead of chain-step span, since the
same span can recur and `noop` blocks have no span at all),
`ar_decode_traj_nokv` (a NEW decode function — `ar_decode_srs_stitched_
tagged_nokv` cannot handle `'noop'` blocks: it unconditionally unpacks
`rb['span']` and only branches on `'iq'/'ir'`, both of which crash on a
`'noop'` block's `span=None` and missing fields; also drops BPB, whose
"last block per contiguous same-span run" grouping assumes each span
appears at most once, which `repeat_query`/`interleave_delayed` violate —
`match_pct` is the metric this diagnostic actually needs).

**Verified correct** via direct mask-inspection smoke tests before use:
single-hop relay boundary holds for both `Q`-to-`Q` and `S`(no-op)-to-`S`
chains; the relay exception is scoped to the STATE row only (not warmup/
response); encoding isolation still holds correctly for a claimed `S`.
All 4 non-test-only DSL strings cross-checked to reproduce their Python
constructor's operations list exactly (`parse_traj_dsl` is now the primary
implementation — the named constructors are thin wrappers around DSL
strings, not independent logic).

**Test harness** (`kvmem/eval_weave.py`): zero-shot diagnostics against any
existing checkpoint (works because none of `solo`/`relay`/`flow` were
trained on interleaved trajectories at all). Headline signal: compare a
span's first-occurrence match% against a later, repeated occurrence's
match% — a large drop is direct evidence of information loss as the relay
moves forward. Smoke-tested against `solo`'s checkpoint: `batch`/
`repeat_query` (same `n_chunks=4` as `solo`'s training) correctly show
100%→0% (`solo` has zero relay capability, as expected — it was only ever
trained on a single chain-step). One real bug caught and fixed during this
testing: `turn_match_pcts` only has entries for non-`'noop'` rec_blocks, so
`spans` must be filtered the same way before zipping them together, or the
lists silently misalign.

**Known confound for `decay_curve` specifically**: it only encodes
`window_chunks` chunks (not the checkpoint's full trained `n_chunks`), so a
zero-shot eval against `solo`/`relay`/`flow` produces a much shorter total
sequence (`L=156` vs. `solo`'s trained `L=236`) — a checkpoint scoring 0%
there is showing a length-extrapolation failure, not necessarily decay.
Confirmed empirically (`solo` scores 100%→0% correctly on `batch`/
`repeat_query`, which match its trained length, but 0%→0% uninformative on
`decay_curve`). Only trust `decay_curve` results from a checkpoint actually
trained at/near that length.

**Not yet built**: the `train()` dispatch to actually train on the
`batch`/`stream`/`interleave_delayed` mix (a `weave_mix`-style stage key,
reusing the fast non-sequential path the same way `flow` does — no
`h_inject` orchestration needed since the relay is resolved by mask
permissions within one forward pass). Position/mask/decode/DSL/test-harness
are all complete; only the training-loop wiring remains.

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
  from Stage 0. **In progress, near completion (~step 156000/160000)** — see
  `CLAUDE.md` for the live progress table (updated as the run continues;
  this doc is not re-edited per checkpoint). Chain step 0 (no `STATE_QUEUE`
  dependency) held 100% val from step 70000-130000, dipped to 91.7% at step
  150000 (test still 95.8%, single data point, not yet a trend). Chain step
  1 has been flat/noisy in the 23-37% band since step 70000. Chain step 2
  (the 2-hop case `STATE_QUEUE` actually has to prove) broke out of near-zero
  around step 42000-49000 but has since oscillated in a roughly 4-25% band
  without a clean sustained improvement — step 90000's 25.0% test peak was
  not exceeded through step 150000. Loss kept declining steadily throughout
  even as chain step 2's match% plateaued, suggesting the remaining training
  is mostly refining calibration on already-learned trajectories rather than
  acquiring new 2-hop capability. Final verdict awaits step 160000 and the
  recovery probe below.
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
(the nochain blackout). Not yet implemented.

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

## 10. `squeeze` — dedicated compression-capacity experiment (designed, queued, not yet run)

`solo`/`relay` were never trained on structured data, so §9's diagnostics
3/4 against those checkpoints are zero-shot-only tests of whatever OOD
generalization happens to exist (and the smoke test showed exactly that: a
flat, uninformative curve, because `solo` already has enough raw capacity to
memorize `chunk_len=16` losslessly regardless of compressibility — no
capacity pressure existed to reveal a compression benefit). `squeeze` is a
dedicated stage designed to actually test it, using the `data_kind`/
`data_target_bits` hooks now wired into `make_batch_tagged`/`train()`
(`kvmem/hmn.py`) — `data_kind='random'` (default, unchanged) or
`'chaotic'`/`'fractal'`/`'ca'`, sampling fresh structured chunks per batch
item via `kvmem/structured_data.py`.

**Design**:
- **Single-register layout** (`n_chunks=1`, `chain_steps=[(0,1)]`) — isolates
  the capacity question to exactly one encoding-block STATE, rather than
  conflating two STATEs contributing to one recall the way `solo`'s
  `span=(0,2)` layout does. Matches `eval_compression.py`'s
  `sweep_max_recallable_length` internals, which already use this same
  single-chunk pattern.
- **`chunk_len=32`** — 2× `solo`'s proven near-ceiling length (16 bytes at
  `state_len=8` already achieves ~94-97% on pure random bytes, i.e. already
  near the edge of raw capacity) — chosen to be comfortably past where raw
  memorization should fail for random bytes, while staying within the
  theoretical compression ceiling for the chosen `target_bits` (`8/2=4×`,
  so up to ~64 bytes would be the theoretical ceiling at `target_bits=2.0`;
  32 is a conservative first test point, not the maximum stretch).
- **`data_kind='ca', data_target_bits=2.0`** — cellular automata (the
  recommended default generator, see §8), calibrated toward 2 bits/byte true
  compressibility (a 4× theoretical compression ceiling vs. raw storage).
- **Paired control, not a single run**: `squeeze_ca` (structured data) AND
  `squeeze_random` (`data_kind='random'`, otherwise IDENTICAL config) must
  both be trained and compared — a high match% on `squeeze_ca` alone proves
  nothing without the matched random-byte control showing a clear failure at
  the same `chunk_len`/`state_len`/model size. The gap between them is the
  actual compression evidence.
- **Model size — start small, escalate only if needed (MDL order: broaden
  distribution → simplify algorithm → grow model size LAST)**: `n_layers=4`
  (not `solo`'s `n_layers=8`) as the first attempt, specifically because a
  larger model risks the weight-based-memorization contamination
  `state_ablation_gate` (§9.1) exists to catch — starting smaller reduces
  that risk before assuming a bigger model is needed. `d=64` kept unchanged
  (no prior ablation evidence it's oversized, unlike `n_layers` which was
  set to match `dual_attn`'s effective depth, not chosen for minimality).
  Escalate to `n_layers=6` or `8` as a follow-up ONLY if `n_layers=4` fails
  to reach near-ceiling match% on `squeeze_ca` — not built preemptively.
- **From scratch, not warm-started**: `squeeze`'s `n_layers=4` doesn't match
  `solo`/`relay`'s `n_layers=8` state_dict shapes, so no warm start is
  possible for the first attempt regardless.

**Configs** (written, not yet run — `kvmem/configs/hmn_squeeze_ca_n4.py`,
`kvmem/configs/hmn_squeeze_random_n4.py`):
```python
hp = dict(
    d=64, n_layers=4, n_heads=4, V=274, block_type='single_attn',
    rope=True, yarn=True, null_kv=True, rmsnorm=True,
    state_len=8, state_vocab_size=2, warmup_len=8,
    curriculum=[dict(n_chunks=1, chunk_len=32, n_refine=0, B=6,
                     n_steps=160000, eval_every=10000,
                     chain_steps=[(0, 1)])],
    data_kind='ca', data_target_bits=2.0,   # 'random' + no target_bits for the control run
)
```

**Verification once run**: `eval_compression.py`'s full diagnostic 1-4 suite
against both checkpoints, plus the direct paired comparison (`squeeze_ca`
match% vs. `squeeze_random` match% at the identical `chunk_len=32`) as the
headline result. `state_ablation_gate` on `squeeze_ca` specifically is the
check that rules out "the smaller model just memorized CA rules into
weights" before trusting any compression claim.

**Next in queue** (after Stage `relay` finishes and the chain-memory recovery
probe runs): `squeeze_ca_n4` + `squeeze_random_n4` paired runs (never two
jobs at once — sequential, `squeeze_random_n4` first since it's the simpler
control and faster to rule in/out).

## 11. Exploratory: could this work without DNN/SGD at all? (discussion, not a build)

Raised as a design-space sanity check, not a proposal to change course — this
project's mandate is the DNN/SGD architecture above. Recorded here because
the reasoning is directly relevant to *why* the neural approach earns its
keep, which is useful context for anyone reconsidering scope later.

**The task decomposes into three pieces, each independently solved
classically, decades before neural LMs existed:**

1. **Bounded, streaming compression of a byte sequence into fixed-size state.**
   Classical answer: **PPM (Prediction by Partial Matching)** or **Context
   Tree Weighting (CTW)**. Both maintain a bounded, streaming context model
   and produce, at every timestep, a genuine normalized probability
   distribution over the next byte — `P(next_byte | context) =
   count[context][next_byte] / sum(count[context][*])`, refined by smoothing
   (Kneser-Ney, Good-Turing) or, for CTW, an exact closed-form Bayesian
   mixture over every possible context-order pruning, recomputed recursively
   per symbol. No gradient descent — pure counting and a fixed recursive
   formula.
2. **Query-based exact recall of a specific earlier span.** Classical
   answer: **content-defined chunking + per-chunk compression + a
   hash-indexed manifest** — the real pattern production backup tools
   (restic, zpaq) already use. A hash-map lookup from span identity to
   stored/compressed chunk is *zero-error* addressing, strictly better than
   attention-based addressing, which can fail. Bounded *lossy* sketches
   (Count-Min Sketch, Bloom filters, reservoir sampling) are the classical
   analogue when the manifest itself must be size-capped rather than
   growing without bound.
3. **Generation: producing plausible continuations, not just recall,
   including a genuine per-timestep probability distribution and sampling
   (not just deterministic/argmax output).** Already solved by (1) — the
   *same* per-timestep distribution used for lossless compression via
   arithmetic coding (encode the *actual* observed byte into a sub-interval
   sized `-log2(P(byte))` bits) can instead be *sampled from* (draw a
   uniform random number, walk the CDF) to generate novel continuations.
   Compression and generation are duals of the same underlying model, not
   separate mechanisms. This is not hypothetical: Shannon's 1948 paper
   demonstrated sampled pseudo-English text from letter-frequency n-gram
   tables, decades before neural LMs — and every modern sampling trick
   (temperature, top-k, nucleus/top-p) operates on "a probability
   distribution over the vocabulary," agnostic to whether that distribution
   came from a softmax or from count-based frequency estimation.

**The one genuine gap in the classical toolkit**, and the actual reason
gradient-trained distributed representations earn their keep: raw count
tables give *zero* credit to a *similar-but-not-identical* context — two
contexts differing by one byte are unrelated cells in the table, so pushing
context order up runs straight into `256^N` sparsity. Classical partial
answers exist and are worth naming precisely rather than dismissing:
- **Context Tree Weighting** — Bayesian blending *across context orders*
  (not across similar-but-distinct contexts at the same order), provably
  near-optimal, zero heuristics, zero training loop. Its internal weighted
  mixture is itself a bounded, streaming sufficient statistic — structurally
  analogous to this project's own `STATE`, carriable chunk-to-chunk the same
  way `STATE_QUEUE` carries a fixed-width vector forward.
- **Locality-sensitive hashing** (SimHash/MinHash) — the closest classical
  analogue to embedding-space similarity: a *fixed*, non-learned hash
  function maps similar inputs to the same or overlapping buckets with high
  probability, so indexing counts by LSH bucket instead of exact context
  gives automatic partial credit to near-miss contexts, no training
  required.
- **Context-mixing compressors** (the PAQ/cmix family, state-of-the-art
  general-purpose lossless compression) — blend many simple predictors via a
  small **online logistic mixer**: tens of scalar confidence weights, one
  layer, updated with a simple per-symbol delta rule. This is technically
  *learned*, so it sits in a gray zone rather than "purely classical" — but
  it learns *which predictor to trust right now*, not a distributed content
  representation, and is categorically simpler than backprop through a deep
  stack.

**What DNN + SGD training actually contributes, net of all the above**: not
compression (solved), not query addressing (solved, and classically
zero-error rather than attention's fallible-but-flexible version), not
generation-with-sampling (solved, same distribution used both ways). The
genuine addition is **not having to hand-pick the algorithm/content-type in
advance** — a human choosing PPM order, CA rule-family assumptions, or LSH
feature functions per content type, versus a model that meta-learns "how to
update state" as a general, discovered procedure during training. This ties
directly to this project's own STATE-compression framing: the bet is that
gradient descent discovers something closer to CTW-style order-blending or
LSH-style similarity-sharing *automatically*, across arbitrary content types,
rather than requiring a human to pick the right classical scheme per case.
