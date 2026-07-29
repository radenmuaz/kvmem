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
| schedule position in a chained schedule (old: "window A/B/C/G") | **chain step** (always two words, never bare "step") |
| byte range being recalled (e.g. `span=(0,2)`) | **span** |
| IQ vs IR pass within one chain step | **round** — round 0 = IQ, round *k*>0 = IR |
| SGD/optimizer iteration | **training step** |
| chat_tags-style weighted trajectory sampling | **trajectory** |
| compressed per-chunk/per-round register (old: "SLOT") | **STATE** (`HMN_STATE_0..N-1`, `state_len`, `state_vocab_size`, `_cyclic_state_ids`) |
| bounded cross-chain-step memory channel | **relay** — currently the `hop` mechanism (learned single-hop-or-more attention permission, `hops` hyperparameter, §4b). The original `STATE_QUEUE`/`h_inject` design (forced feature-vector copy) is **deleted** — see §4's historical note. |

**Stage names** — short, descriptive, tied to what each stage mechanically
does rather than a bare index (numbers alone don't convey what changed
between them, and the old "IQ"/"IR" naming is retired per the round
unification in §3):

| Old | Name | What it tests | Status |
|---|---|---|---|
| Stage 0 | **`solo`** | one chain-step, nothing to relay yet — the bootstrap case | Done |
| Stage 1 | **`relay`** | multi chain-step, `STATE_QUEUE` single-hop relay via `h_inject` (forced copy, detached gradient) | Done, then **deleted** (mechanism removed from the codebase after `hop` was shown to substantially outperform it) |
| §4b | **`hop`** | same single-hop relay (generalized to N-hop via `hops`), but via a learned attention permission instead of a forced copy — full gradient flow, cheaper training too | Done — the sole surviving relay mechanism |
| (deferred) | **`refine`** | IR rounds added on top of `hop`'s chaining | Not started |
| (deferred) | **`bank`** | generalizes the single-slot relay into a proper memory bank | Not started |
| §10 | **`squeeze`** | does `STATE` genuinely compress compressible content, not just store it | Mid-run (capacity-pressure correction applied, see §10) |
| §4c | **`weave`** | arbitrary interleaved encode/query trajectories, tests generalization beyond one fixed schedule | Built (position/mask/DSL/decode/eval/training dispatch); queued run not yet started |

Config file names now match this table directly (`hmn_single_recall.py`,
`hmn_recall_queue.py`, `hmn_squeeze_*.py`, `hmn_weave_mix.py`) — the earlier
`hmn_stage0_round0_single.py`/`hmn_flow.py`/`hmn_stage1_round0_chained.py`
names have all been renamed or removed.

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

**Historical note (superseded by `hop`, kept for context)**: the original
design had `STATE_QUEUE` reuse `HMN_STATE_0..3` directly rather than a
separate token family — position/mask disambiguated `STATE_QUEUE_in` from a
step's own fresh `STATE`, and `h_inject` overwrote that region's embedding
immediately after `_embed()`, before any transformer block ran. `hop`
doesn't need any of this: it never allocates a separate relay region at
all (see §4b) — a chain step's own STATE serves double duty as both its own
recall register and the thing the next chain step reads, so the "which
token family" question this note originally answered no longer applies.

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
yet to feed back from. `_emit_round(round_idx)` implements
both: `round_idx == 0` skips the argmax/STATE_A prefix; `round_idx > 0` adds
it. The underlying mask rules (the nochain blackout, IR feedback isolation) are unchanged —
this is a naming/API unification, not new masking logic.

## 4. `STATE_QUEUE` mechanics (historical — mechanism deleted, kept for context on why `hop` is designed the way it is)

**This entire section describes the original `relay` mechanism, which no
longer exists in the codebase.** `HMNModel.forward`'s `h_inject` parameter,
`chunk_positions_chained`, `train()`'s `chain=True` sequential per-chain-step
training loop, and `chunk_mask_fb`'s queue-related mask rules were all
deleted after `hop` (§4b) was shown to substantially outperform this design
(see §6's Results). Reading this section is useful only to understand *why*
`hop` is shaped the way it is — every mechanic described below is gone.

**Layout** (as it was): each chain step after the first got a
`STATE_QUEUE_in` region of width `M*state_len` (default `M=1`) immediately
before that chain step's round-0 STATE region.

**Masking** (as it was): `STATE_QUEUE_in` joined the chain step's own "own
content" set (same treatment as STATE/warmup/response) — folded into the
existing warmup-bottleneck/output-bottleneck unions, no new rule. Still
fully blocked from raw chunk content and other chain steps' regions (the
nochain blackout, which `hop` still relies on unchanged) — the *only*
channel for cross-chain-step information was the injected feature vector,
never an attention path.

**Data flow (`h_inject`, as it was)**: for chain step *i* > 0:
1. Run chain step *i-1*'s forward pass with `return_features=True`.
2. Extract the residual-stream slice at chain step *i-1*'s **last round's**
   own STATE positions (round 0's `sl0/sl1` if that chain step had
   `n_refine=0`, else the final IR round's `slb0/slb1`).
3. Run chain step *i*'s forward pass with
   `h_inject={(queue0, queue1): <that slice>}`.

This broke the "2 forward passes for the whole packed sequence" trick for
chained stages — chain step *i*'s input genuinely depended on chain step
*i-1*'s computed output, not just its decoded bytes, so a chained stage
needed one sequential forward pass per chain step. This sequential-pass cost
is exactly what `hop` eliminated (§4b) — `hop` resolves everything within
one packed-sequence forward pass instead.

**`STATE_QUEUE` was a single-hop relay, not an accumulating buffer.** `M=1`
meant chain step *i*'s `STATE_QUEUE_in` came ONLY from chain step *i-1*'s
own last STATE — never from *i-2* directly. There was no separate "older
states" store to mask or discard: raw content from chain steps older than
*i-1* was already fully blocked by the nochain blackout regardless of
`STATE_QUEUE` (that invariant predates this mechanism and still holds for
`hop`). For information from chain step *i-2* to reach chain step *i*, chain
step *i-1* had to have implicitly folded it into its own single
`state_len`-wide STATE when producing its own output — there was no
guarantee this happened; that's exactly what the chain-memory recovery
probe (§7) tests, and `hop`'s own single-hop (`hops=1`) result on that probe
was a clean failure (§7).

**The gradient-flow caveat this section originally flagged, now resolved**:
`h_inject`'s value was `.detach()`ed (truncated BPTT across chain steps) —
the model got no direct gradient signal for "make this STATE useful to a
*future* chain step," only for its own chain step's recall loss. `hop` (§4b)
is the direct fix — full gradient flow, no `.detach()`, no forced copy — and
was measured to substantially outperform this design as a result (§6).

## 4b. Stage `hop` — attention-based relay, no forced copy (done, `hops` generalization also done)

Direct alternative to `relay`'s `h_inject` copy, built specifically to fix
the gradient-flow caveat above. Instead of forcibly overwriting chain step
*i*'s input with chain step *i-1*'s extracted STATE, `hop` grants chain
step *i*'s own round-0 STATE row a narrow, single-hop **attention
permission**: it can read chain step *i-1*'s STATE columns directly (still
blocked from *i-1*'s warmup/response/raw chunks, and still blocked from
*i-2* or earlier — same single-hop information budget as `relay`, M=1 in
`STATE_QUEUE` terms). The model *learns* what to preserve via ordinary
gradient descent — full gradient flow across chain steps, no `.detach()`,
no forced copy.

**Implementation** (`kvmem/hmn.py`):
- `chunk_positions_hop` — same overall shape as `chunk_positions_chained`
  but never allocates a separate `STATE_QUEUE_in` region; chain step *i*'s
  own STATE serves double duty as both its own recall register and the
  thing chain step *i+1* reads.
- `chunk_mask_fb_hop` — copy of `chunk_mask_fb` with exactly one change:
  the nochain blackout's `prior_all` (which blocks a round-0 STATE row from
  all prior chain steps' content) gets a carved-out exception (the relay
  exception) for the immediately preceding chain step's own last-round STATE
  columns specifically. Kept as a fully separate function rather than
  modifying the proven `chunk_mask_fb` in place.
- `train()` dispatch: any `stage['chain_steps']` stage now unconditionally
  builds its layout via `chunk_positions_hop`/`chunk_mask_fb_hop` (the old
  `chunk_positions_chained`/`h_inject`-based path and its `stage['chain']`
  key are deleted entirely — there is no alternative dispatch to be
  "mutually exclusive" with anymore). The relay's lookback window is
  controlled by `stage['hops']` (default 0 — no relay exception at all,
  opt-in required; `hops=1` is the originally-designed, verified single-hop
  behavior; `hops>1` is a later generalization, see below). Resolved
  entirely by mask permissions within one forward pass — always routes
  through the fast, non-sequential training path, no `h_inject`
  orchestration loop needed at all. This also means `hop` is *cheaper per
  step* than the deleted `relay` mechanism, not just gradient-cleaner — a
  real efficiency side benefit, not just a correctness fix.

**Verified before the mask-permission mechanism was trusted** (smoke test,
`n_chunks=4, chunk_len=8, state_len=4, chain_steps=[(0,2),(1,3),(2,4)]`):
chain step 1's STATE row sees chain step 0's STATE columns directly
(visible) but not its warmup/response (blocked); chain step 2's STATE row
sees chain step 1's STATE (visible) but NOT chain step 0's STATE directly
at `hops=1` (blocked — single-hop preserved, must relay through chain step
1); encoding isolation and the relay exception being scoped to the STATE
row only (not warmup/response rows) both hold. Full `train()` path (3
steps, tiny model) ran end-to-end without error. The `hops` generalization
(N-hop, not just single-hop) was verified separately afterward — see below.

**Config**: `kvmem/configs/hmn_recall_queue.py` (renamed from `hmn_flow.py`)
— identical hyperparameters to the deleted `relay` mechanism's config (same
`d`, `n_layers`, `chain_steps`, step budget, warm-started from `solo`),
`hops=1` set explicitly (the parameter's default is `0` — no relay at all,
opt-in required, see below).

**Result, measured**: `hop` massively outperforms `relay` on every metric —
chain step 1 test 95.8% vs relay's 37.5% (2.5x), chain step 2 (the 2-hop
case, the actual test of whether the relay carries anything) test 70.8% vs
relay's 12.5% (5.7x). Full numbers in `CLAUDE.md`'s Results section (this
doc isn't re-edited per checkpoint). This result is what motivated deleting
the `h_inject`-relay path entirely: `HMNModel.forward`'s `h_inject`
parameter, `train()`'s `chain=True` sequential per-chain-step training loop,
`chunk_positions_chained`'s `STATE_QUEUE_in` allocation, and
`chunk_mask_fb`'s queue-related mask rules are all gone from the codebase.

**Caveat worth carrying forward**: `hop` was run TWICE under this same
config (once before, once after the vocab reorder) and the two runs
produced notably different results (88.1%/85.7% STITCHED vs. 71.4%/71.4%,
see `CLAUDE.md`) — attributed to warm-start sensitivity from `solo`, not a
code defect (the mask itself was independently verified byte-identical
across both). Treat any single `hop` run's outcome as one sample from a
distribution with real spread, not a deterministic number.

**`hops` generalization (built after the comparison above, untested in
training)**: `chunk_mask_fb_hop`'s single "chain step *i-1* only" lookup was
generalized to a `hops` parameter — the relay exception now unions the last
`hops` chain steps' own STATE ranges, not just the immediately preceding
one. `hops=0` (the default) disables the relay exception entirely
(equivalent to `solo`'s no-relay case for any multi-chain-step schedule);
`hops=1` reproduces the exact behavior every result above was measured
against; `hops>1` (attend back N chain steps at once) is implemented and
verified via direct mask inspection (0/1/2/3-hop semantics all confirmed
correct) and a full training smoke test, but has never actually been
trained to convergence — a natural next experiment given `hop`'s clean
`repeat_query` failure (§7), orthogonal to `weave_mix` (§4c, which tests
trajectory-shape generalization, not relay depth).

## 4c. Stage `weave` — arbitrary interleaved trajectories (fully built: position/mask/DSL/decode/eval harness/training dispatch; queued run not yet started)

`solo`/`relay`/`hop` all train on exactly one fixed rhythm: encode
everything, then query in a fixed order. That's one point in a much larger
space of possible encode/query interleavings, and the project's own vision
(reading a document incrementally, answering questions as you go) needs the
model to work correctly under others too. `weave` generalizes
`chunk_positions_hop` to arbitrary interleaved operation sequences via a
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
- `Q(s,e)` — query/recall span `[s,e)` (unchanged from `hop`).

**Why `E`+`S` instead of one bundled "encode" op, and no separate `N` op
type**: an earlier draft of this design had `E` implicitly bundle its own
STATE emission (matching `chunk_positions_chained`/`chunk_positions_hop`'s
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
| `batch` (= `relay`/`hop`'s fixed rhythm) | `E4 Q(0,2) Q(1,3) Q(2,4)` | yes |
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
encoding-isolation/chunk-blackout/relay-exception/IR-feedback-isolation logic as `chunk_mask_fb_hop`, generalized to group by
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
existing checkpoint (works because none of `solo`/`relay`/`hop` were
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
zero-shot eval against `solo`/`relay`/`hop` produces a much shorter total
sequence (`L=156` vs. `solo`'s trained `L=236`) — a checkpoint scoring 0%
there is showing a length-extrapolation failure, not necessarily decay.
Confirmed empirically (`solo` scores 100%→0% correctly on `batch`/
`repeat_query`, which match its trained length, but 0%→0% uninformative on
`decay_curve`). Only trust `decay_curve` results from a checkpoint actually
trained at/near that length.

**Built and verified**: the `train()` dispatch to train on the
`batch`/`stream`/`interleave_delayed` mix — a `stage['weave_mix']` key (list
of `{weight, pattern[, n_chunks, window_chunks]}` dicts), dispatched via a
new top-level `if 'weave_mix' in stage:` branch, structurally parallel to
the existing `traj_mix` branch but built on `chunk_positions_traj`/
`chunk_mask_fb_traj` instead. Reuses the fast non-sequential path (one
`model()` forward pass per training step — no `h_inject` orchestration,
since the relay is resolved by mask permissions) the same way `hop` does.
`repeat_query`/`long_hop_recovery`/`decay_curve` are explicitly **rejected
with an `AssertionError`** if passed to `weave_mix` (not silently allowed)
— training on them would defeat their purpose as held-out generalization
probes. `n_refine` is fixed at 0 (no argmax-feedback IR support for weave
patterns yet, matching every existing weave usage). Required a small fix to
`make_batch_tagged`, which previously crashed on `'noop'` rec_blocks
(unconditionally unpacked `rb['span']`, `None` for noops) — it now fills a
noop block's STATE region with placeholder ids and skips everything else
(no warmup/response fields exist on a noop). Verified end-to-end via a tiny
CPU smoke test (5 SGD steps, mixed `batch`/`stream`/`interleave_delayed`
weights) and a rejection test confirming `repeat_query` raises correctly.

## 5. Block types

One `HMNModel` class, selected via `block_type`:

| `block_type` | Structure | Role |
|---|---|---|
| `attn_mlp` | `x = x + attn(norm1(x)); x = x + ffn(norm2(x))` | standard architecture, comparison baseline |
| `dual_attn` | `x = x + attn1(norm1(x)); x = x + attn2(norm2(x))` (paired, no MLP) | available ablation option, byte-identical port of the prior architecture — no longer required for checkpoint compatibility since this is a from-scratch retrain |
| `single_attn` | `x = x + attn(norm(x))` (one attn, one norm, no MLP) | **the default** — same block repeated `n_layers` times; use `n_layers` = 2× the equivalent `dual_attn` config's `n_layers` to match total attention-op count |

## 6. Current staging and results

Full numbers live in `CLAUDE.md`'s Results section (updated per checkpoint
during a run; this doc summarizes rather than duplicates). Status as of this
writing:

- **`solo`** (`kvmem/configs/hmn_single_recall.py`) — one chain step, round
  0 only, no relay. **Done**: 160000/160000 steps, val/test 94.4%/100%
  (later reproducibility run: 100%/100%) — matches the historical ~100%
  single-window IQ ceiling.
- **`relay`** (config and logs deleted, the `STATE_QUEUE`/`h_inject`
  mechanism, §4) — three chain steps, warm-started from `solo`. **Done,
  then removed from the codebase.** Final numbers preserved in `CLAUDE.md`:
  val/test STITCHED=44.6%/44.6%, chain step 2 (2-hop case) capped at
  12.5%/12.5%. Motivated `hop`.
- **`hop`** (`kvmem/configs/hmn_recall_queue.py`, §4b) — same schedule,
  warm-started from `solo`. **Done, run twice** with a real discrepancy
  between runs (warm-start sensitivity, not a bug — see §4b and
  `CLAUDE.md`). Best measured result: val/test STITCHED=88.1%/85.7%, chain
  step 2 test 70.8% — 5.7x `relay`'s result on the exact metric that
  matters.
- **Chain-memory recovery probe** (§7) — run against `hop`'s best-measured
  checkpoint. **Failed cleanly**: 0.0% on `repeat_query` across all 3 test
  sequences. Motivated `weave_mix` (§4c) as the direct follow-up.
- **`squeeze`** (§10) — mid-run, capacity-pressure correction applied
  (`chunk_len` bumped 32→96 after the first attempt at `chunk_len=32`
  turned out to have no capacity pressure at all).
- **`weave_mix`** (§4c) — built and queued, not yet run.
- IR rounds with relay, larger `hops`, and the sparse block-attention
  memory-bank generalization are all still deferred.

**Eval metric note**: `STITCHED_MEAN` is only meaningful when the
`chain_steps` schedule covers the *entire* `n_chunks*chunk_len` source (true
for `relay`/`hop`'s 3-chain-step schedule, NOT true for `solo`'s
single-chain-step schedule, where it's capped at ~42.9% by construction
since the untested chunks are never decoded). Watch `span/MEAN` for
single-chain-step configs.

**Zero-shot stitching is already exercised by every eval, not a separate
test.** `ar_decode_srs_stitched_tagged_nokv` (and its `hop`/`weave`
equivalents) seed only the first chain step's warmup from ground truth;
every later chain step's warmup comes from the model's own
previously-decoded bytes. This is the same proven byte-level stitching
mechanism from the prior architecture — distinct from and doesn't test the
relay mechanism specifically (see §7).

## 7. Chain-memory recovery probe (built and run — result: fails cleanly)

Per-chain-step recall accuracy alone does NOT prove the relay carried
anything forward — each chain step can solve its own span locally from
encoding-block STATEs, regardless of chaining. The real test: run a *later*
chain step's round-0 recall on a query that requires recovering an
*earlier* chain step's span — something only reachable via the accumulated
relay chain, since direct cross-chain-step attention stays blocked (the
nochain blackout).

**Implemented as `kvmem/eval_weave.py --patterns repeat_query`**: query
span (0,2), then (1,3), then (2,4), then re-query (0,2). Per the single-hop
mask rule, that final re-query can only reach span (0,2)'s content through
the accumulated relay chain (Q(0,2)→Q(1,3)→Q(2,4)→Q(0,2)), never by direct
attention back to chunk 0/1's raw bytes.

**Result, run against `hop`'s best-measured checkpoint**: first occurrence
100% (trivial), repeated occurrence **0.0% across all 3 test sequences** —
complete, not partial, failure. Caveat: `repeat_query` is a trajectory
shape `hop` was never trained on (only its fixed 3-query schedule), so this
could reflect either "the relay doesn't preserve information across 3
hops" or "the model can't generalize to this novel trajectory shape at
all, independent of what its STATE contains" — the clean, total (not
gradual) failure leans toward the latter, but this test alone can't
cleanly separate the two. This is exactly what motivated `weave_mix`
(§4c) — training on varied trajectory orderings (never `repeat_query`
itself, which stays held out) directly tests the generalization-gap
hypothesis without touching the relay mechanism.

## 8. Structured-data track (`kvmem/structured_data.py`)

**Motivation**: genuine compression (zip/gzip-style, exploiting statistical
redundancy) cannot emerge from training on the max-entropy random bytes used
everywhere else in this project — Shannon's source coding theorem makes such
data literally incompressible, so there's no redundancy for `STATE` to learn
to exploit. Random-byte training only teaches raw lossless storage density
and the addressing algorithm, not compression.

**Nine generator families** (plus a documented placeholder,
`gen_template_repeat`), each sampling fresh random parameters per call
(required — a fixed rule across all examples would let the model bake it
into static weights instead of `STATE`, the same FFN-as-static-knowledge
failure mode the `dual_attn` design already avoids elsewhere). Organized by
which real compressor family they exercise, following `LANGUAGE.md`'s
generative-hierarchy framing (character frequencies → Markov chains → word/
phrase distributions → long-range repetition — this project's generators
map fairly directly onto that hierarchy's early/middle levels):
- `gen_chaotic_logistic` — logistic map, random `r`.
- `gen_fractal_midpoint` — 1D midpoint-displacement fractal, random Hurst exponent.
- `gen_ca` — 1D cellular automaton, random rule table + initial condition.
  **Recommended default for general use** — discrete-native (no
  quantization ambiguity unlike the two continuous generators), exactly
  reproducible from pure integer ops, enormous tunable rule space.
  Confirmed empirically: raw byte-histogram entropy for chaotic/fractal is
  ~7.1-7.15 bits (nearly max, since byte quantization washes out their
  structure at the histogram level); CA is 2.87 bits (genuine redundancy).
- `gen_markov` — order-1 Markov chain, full 256-byte alphabet. **Recommended
  when precise calibration matters more than diversity** — exact
  closed-form entropy-rate bisection, no measure-and-search. **Important
  measured finding**: `measure_bits_per_byte` (zlib) does NOT validate this
  generator — DEFLATE's Huffman stage codes against marginal/global byte
  frequency, not the preceding byte, so it's structurally blind to order-1
  conditional structure even when exactly calibrated (confirmed: zlib
  stayed ~7-8 bits/byte across target_bits=1-6, while a direct empirical
  order-1 conditional-entropy estimate tracked correctly). This is exactly
  the kind of structure a context-conditional model (this project's own
  attention-based architecture) CAN exploit, even though zlib can't see it.
- `gen_iid_skewed` — i.i.d. bytes, skewed (Zipf-like) marginal distribution.
  The deliberate "control case zlib CAN see," paired against `gen_markov` —
  exact closed-form entropy, zlib tracks it closely (verified).
- `gen_run_length` — fresh byte + geometric-length run, repeated.
  RLE/LZ77-visible (a second, differently-mechanismed zlib-visible control
  case). Approximate closed-form calibration, verified tracking zlib well.
- `gen_markov_order_k` — generalizes `gen_markov` to context length > 1,
  small alphabet (`K`/`order` tunable) for tractability — same exact
  bisection-on-entropy approach, just over a meta-state space.
- `gen_match_distance` — parametrized LZ77-style generator (match
  probability + match-distance range + match-length, not a fixed phrase
  vocabulary — a more controllable realization of what `gen_template_repeat`
  was aiming for). **Required a mid-implementation fix, measured not
  assumed**: an initial single-byte-copy-per-event version was nearly
  invisible to zlib, because DEFLATE needs a 3+ byte match to encode one at
  all and isolated single-byte copies almost never chain into that length
  by chance (verified: p=0.0-0.7 all measured ~7.7-8.0 bits/byte despite a
  smooth decline being expected). Fixed by emitting genuine multi-byte
  match runs; zlib then tracked target_bits closely. **Recovery-probe
  contamination warning**: exact byte repetition means a model could
  "recover" a matched byte via simple positional copying — do not use for
  the chain-memory recovery probe without accounting for this.
- `gen_mixed_order` — stochastically blends order-0/1/3 components per
  position (CTW/PPM-exploitable structure — defeats any single fixed
  context length, see §11's CTW discussion). Least precisely calibrated of
  the nine (component-wise calibration only approximates the true
  switched-process entropy) — flagged honestly.
- `gen_template_repeat` — placeholder, raises `NotImplementedError`.
  Fixed-vocabulary phrase-repetition design documented in its own
  docstring but not built, superseded in practice by `gen_match_distance`'s
  more general, already-implemented approach.

**`target_bits`**: calibration precision varies by generator, documented
per-function. Exact closed-form bisection: `gen_markov`, `gen_iid_skewed`,
`gen_markov_order_k`. Approximate closed-form bisection: `gen_run_length`,
`gen_match_distance`, `gen_mixed_order`. `measure_bits_per_byte` (zlib)
measure-and-search, with documented seed-dependent imprecision: `gen_chaotic_logistic`,
`gen_fractal_midpoint`, `gen_ca` — fractal calibrates most reliably of
these three (two-phase coarse-then-refine over its scalar Hurst knob);
chaotic is seed-dependent (bits/byte can jump from ~1.5 to ~6 between
r=3.63 and r=3.64); CA's rule-space distribution is bimodal/sparse in the
middle (only ~3% of random k=2,r=1 rules land in a 1.5-2.5 bits/byte band).

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

## 10. `squeeze` — dedicated compression-capacity experiment (mid-run, `chunk_len` corrected once already)

`solo`/`hop` were never trained on structured data, so §9's diagnostics 3/4
against those checkpoints are zero-shot-only tests of whatever OOD
generalization happens to exist (and the smoke test showed exactly that: a
flat, uninformative curve, because `solo` already has enough raw capacity to
memorize `chunk_len=16` losslessly regardless of compressibility — no
capacity pressure existed to reveal a compression benefit). `squeeze` is a
dedicated stage designed to actually test it, using the `data_kind`/
`data_target_bits` hooks wired into `make_batch_tagged`/`train()`
(`kvmem/hmn.py`) — `data_kind='random'` (default, unchanged) or any of the
now nine generators in `kvmem/structured_data.py` (§8), sampling fresh
structured chunks per batch item.

**Design**:
- **Single-register layout** (`n_chunks=1`, `chain_steps=[(0,1)]`) — isolates
  the capacity question to exactly one encoding-block STATE, rather than
  conflating two STATEs contributing to one recall the way `solo`'s
  `span=(0,2)` layout does. Matches `eval_compression.py`'s
  `sweep_max_recallable_length` internals, which already use this same
  single-chunk pattern.
- **`chunk_len=96` (CORRECTED from an initial `chunk_len=32`, measured not
  assumed)**: the first attempt used `chunk_len=32` (2× `solo`'s proven
  ~128-bit near-ceiling length), expecting it to strain raw capacity for
  random (incompressible) content. It didn't — `squeeze_random` converged
  to ~100% val, loss~0 by step 70000/160000 on pure random 256-bit content
  at `n_layers=4`, meaning the paired comparison would have been
  uninformative (no capacity pressure for `squeeze_ca` to show an advantage
  against). Killed that run once the saturation was clear, bumped
  `chunk_len` to 96 (768 bits, 3x the point that saturated), and reduced
  `n_steps` to 60000 (the original run's own eval curve converged well
  before 160000 steps, so a shorter budget should be enough to see the
  trend). `target_bits=2.0` for `squeeze_ca` is unchanged (a per-byte
  quantity independent of `chunk_len` — 192 bits of true information vs.
  random's 768 bits at `chunk_len=96`, still a 4x theoretical compression
  ceiling). The rerun (`chunk_len=96`) showed genuine capacity pressure
  early (2.7%-4.2% match through step 15000, vs. the old config's rapid
  saturation) before being paused mid-run per user request — see `CLAUDE.md`
  for exact numbers.
- **`data_kind='ca', data_target_bits=2.0`** — cellular automata (the
  recommended default generator for general use, see §8), calibrated toward
  2 bits/byte true compressibility (a 4× theoretical compression ceiling vs.
  raw storage).
- **Paired control, not a single run**: `squeeze_ca` (structured data) AND
  `squeeze_random` (`data_kind='random'`, otherwise IDENTICAL config) must
  both be trained and compared — a high match% on `squeeze_ca` alone proves
  nothing without the matched random-byte control showing a clear failure at
  the same `chunk_len`/`state_len`/model size. The gap between them is the
  actual compression evidence. `squeeze_random` runs first (simpler,
  establishes the raw-capacity floor).
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
  `solo`/`hop`'s `n_layers=8` state_dict shapes, so no warm start is
  possible regardless.

**Configs** (`kvmem/configs/hmn_squeeze_ca_n4.py`,
`kvmem/configs/hmn_squeeze_random_n4.py`), current state:
```python
hp = dict(
    d=64, n_layers=4, n_heads=4, V=274, block_type='single_attn',
    rope=True, yarn=True, null_kv=True, rmsnorm=True,
    state_len=8, state_vocab_size=2, warmup_len=8,
    curriculum=[dict(n_chunks=1, chunk_len=96, n_refine=0, B=6,
                     n_steps=60000, eval_every=5000,
                     chain_steps=[(0, 1)])],
    data_kind='ca', data_target_bits=2.0,   # 'random' + no target_bits for the control run
)
```

**Verification once run**: `eval_compression.py`'s full diagnostic 1-4 suite
against both checkpoints, plus the direct paired comparison (`squeeze_ca`
match% vs. `squeeze_random` match% at the identical `chunk_len=96`) as the
headline result. `state_ablation_gate` on `squeeze_ca` specifically is the
check that rules out "the smaller model just memorized CA rules into
weights" before trusting any compression claim.

**Status**: `squeeze_random` is mid-run (paused, not finished — see
`CLAUDE.md` for the exact step/numbers). `squeeze_ca` and `weave_mix` (§4c)
are both queued behind it, in that order (never two jobs at once).

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
  analogous to this project's own `STATE`, carried chunk-to-chunk the same
  way the relay (`hop`) carries a fixed-width vector's worth of information
  forward.
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

## 12. `hmn_adaptive_trainer.py` — adaptive weave_mix reweighting, and a discovered positional-shortcut bug

**Motivation**: `hmn_weave_c64` mixes `batch`/`stream`/`interleave_delayed` at a
fixed uniform weight/repeat_batch. `stream` reached 42-49% match while `batch`/
`interleave_delayed` sat at 8-20% across multiple runs — a fixed uniform mix has
no way to notice that split and lean into it. `kvmem/hmn_adaptive_trainer.py`
(cloned from the 2026-07-27 `hmn_v3_backup.py` snapshot) adds exactly that: the
`weave_mix` branch only (all other branches byte-identical to `hmn.py`) tracks
each trajectory's own difficulty (`ema_loss`, a per-trajectory EMA of train loss,
or `last_match`, the raw eval-time match%) and recomputes each trajectory's
sampling weight at every eval step via `_adapt_reweight`/`_temp_softmax_rescale`
— softmax over difficulty normalized to the mix's own mean, at a configurable
temperature (`adapt_temp`, lower = more aggressive). `repeat_batch` adaptation
was deliberately disabled (kept as one feedback signal at a time — weight only
— rather than two coupled ones; the scaling line is commented, not deleted).
The branch also uses a fixed-LR-after-linear-warmup schedule instead of cosine
decay, since a decaying LR fights a training signal that keeps shifting as
reweighting moves effort around.

**First run** (`adapt_signal='train_loss'`, `adapt_temp=1.0` implicit default):
stage0 final MEAN=25.0% (best 25.5%) vs. the non-adaptive baseline's 24.2%
(best 25.1%) — essentially a wash on the aggregate, but a real redistribution:
`interleave_delayed` improved +5.6pp (15.2%->20.8%) at the cost of `stream`
dropping -2.4pp (49.1%->46.7%), matching the mechanism's design intent. Caveat:
this comparison also differs in LR schedule (baseline cosine-decays by step
80000, adaptive holds flat), so the redistribution can't be attributed to
reweighting alone. **`batch` never improved in either run** (8.3% vs 7.4%,
both near-random) despite `_adapt_reweight` giving it the *highest* weight of
the three at several intermediate evals — sampling more of it didn't help,
the first sign something structural (not just "needs more samples") was wrong.

**Aggressiveness knob**: at T=1.0 (softmax temperature), weights barely moved
off ~1/3 each for the loss spreads actually observed (e.g. `[3.68,2.92,4.59]`
-> weights `[0.32,0.27,0.41]`). Added `adapt_temp` (lower = more aggressive,
T->0 approaches one-hot on the single hardest trajectory) after an earlier,
since-replaced `adapt_power` exponent design — replaced because the user asked
for "a softmax temperature-like constant" specifically. At T=0.2, the same
spread reallocates to `[0.22,0.09,0.69]`.

**T=0.2 run revealed a real oscillation/thrashing failure mode**, live: at
step 50000, `interleave_delayed` had been pushed to weight=0.96 (from a prior
adaptation) yet scored 0.0% match at that eval, while the *starved* `batch`
(weight=0.02) scored 42.0%. The mechanism then swung hard the other way —
by step 56997, `batch`'s own EMA loss had risen to 8.23 (worst of the three,
from lack of practice) and its weight climbed to 0.93, echoing the same
instability. This is exactly the "no anti-oscillation memory" gap flagged
before this run: weight is recomputed fresh from the latest snapshot every
eval with no momentum across reweighting *events* themselves, so an
overcorrection at one eval can set up an overcorrection in the opposite
direction at the next.

**The real culprit, found via `kvmem/probe_positional_shortcut.py`**: `traj1`
(`batch`, `E2 Q(0,1) Q(1,2)`) and `traj3` (`interleave_delayed`,
`E2 Q(1,2) Q(0,1)`) share a byte-identical `E2` encode prefix (same two STATE
registers, same positions, same mask) and diverge only in query order. Probed
directly: encode two independent random 64-byte chunks as usual, then feed
query slot 1 (which normally recalls chunk index 0 — the STATE built
*first*, i.e. farther in absolute position from the query) chunk 1's own real
warmup bytes instead of chunk 0's. Result (n=8 trials, `hmn_weave_c64_adaptive`
stage0_best.pt): swapped-warmup generation matched chunk 0's true continuation
at 91.1% and chunk 1's true continuation at only 0.4% — **the model completely
ignored the warmup content it was given and defaulted to whatever that query
slot normally recalls.** This is pure position-addressed retrieval, not
content-addressed — confirmed, not just hypothesized. It directly explains why
`traj1`/`traj3` share a genuine training conflict: the SAME query-slot position
is pulled toward "attend to STATE0" in `batch` and "attend to STATE1" in
`interleave_delayed`, and since the model isn't reading warmup content at all,
there's no content-based way to reconcile the two — mixing them (which
`weave_mix` already does) doesn't fix this, since the shortcut isn't "always
guess a fixed order," it's "ignore content entirely and use position," which
mixing the two orders doesn't punish enough to unlearn.

**Why RoPE is implicated, and why ALiBi/T5-relative-bias would NOT fix this**:
RoPE (`apply_rope`, `kvmem/hmn.py`) gives every attention pair a smooth
distance-based prior via absolute position (`torch.arange(offset, offset+L)`
fed into the rotation, uniformly for every token including STATE/tag/
structural positions, no different treatment for the one legitimate
cross-block channel — the query's attention into its own STATE). ALiBi (Press
et al. 2021) and T5's relative-position-bucket bias both drop the sin/cos
rotation machinery but are STILL monotonic nearer-is-preferred distance
decay — swapping RoPE for either would likely reproduce the same shortcut,
since the shortcut isn't about rotation specifically, it's about *any*
distance signal existing on that attention hop at all. NoPE (no positional
encoding, Haviv et al. 2022; Kazemnejad et al. 2023 NeurIPS found it actually
generalizes to longer sequences *better* than RoPE/ALiBi) is the class of fix
that would actually remove the shortcut, if scoped to just the STATE-
addressing attention (give STATE/structural positions a fixed/canonical
position id so their distance from any later query is constant regardless of
trajectory-specific query ordering, while leaving normal RoPE in place for
in-block sequential byte generation, which genuinely needs relative order).
CoPE (Contextual Position Encoding, Meta 2024 — position counters gated by
content rather than raw index) is the more principled long-term answer but a
larger new mechanism to build, not a drop-in swap. **Not yet implemented** —
scoped as the next step, pending a decision to spend a from-scratch (or
partially-warm-startable, TBD) training run to test it, since it changes a
mechanism every existing checkpoint was trained under.

**Randomizing query order was considered and rejected as insufficient**,
given the probe result: `weave_mix` already mixes `batch` and
`interleave_delayed` (i.e., across the training distribution, "query slot 1"
already recalls chunk index 0 half the time and index 1 the other half) —
this is already a form of order randomization at the trajectory-mix level,
and it demonstrably has NOT fixed the shortcut (the probe was run against a
checkpoint trained under exactly this mix). The shortcut survives random
order-mixing because it isn't a memorized fixed correlation between position
and chunk-index, it's a complete absence of content-based addressing — a
strictly stronger problem that only removing the distance signal itself (not
just decorrelating it from one fixed pattern) can fix.

## 13. Three positional-shortcut fixes attempted — one abandoned mid-design, one shipped a real bug and got fixed, one failed outright

Direct continuation of §12's diagnosis. Three separate mechanisms were built
to remove the query-slot positional shortcut; this section is the record of
what actually happened to each, in the order they were tried.

### 13a. Dual-clock RoPE (`dual_rope`) — abandoned before completing a full run

First attempt: `apply_rope_dual`/`_dual_positions` (`kvmem/hmn.py`) — two
separate position clocks, `pos_state` (advances only at STATE-emission
events, frozen everywhere else — so any query following the same encoding
pass sees an identical macro-distance to every STATE regardless of query
order) and `pos_local` (resets to 0 at the start of every encode/query
block, preserving genuine local byte-order). **Caught a real bug before
trusting it**: the first draft also advanced `pos_state` on a query's OWN
recall-STATE row (built into every query for potential relay-chain use,
even when no relay chain is actually in play), silently recreating the same
query-order-dependent value the whole mechanism was built to remove — found
via direct numerical inspection (`batch`'s two queries showed macro values
2 and 3; should have been identical), not by training and observing failure.
Fixed by excluding a query's own STATE row from `state_starts` entirely
(only `enc_blocks`' STATE counts) — re-verified afterward: both queries in
both `batch`/`interleave_delayed` then showed identical distance-to-
STATE0/STATE1 regardless of order.

**Abandoned anyway**, before any training run, once a further design
discussion surfaced the STATE register's own cyclic token IDs
(`_cyclic_state_ids`: with the default `state_vocab_size=2`, only 2 distinct
IDs cycle through all `state_len=8` slots) as a second, independent source
of ambiguity the reset-heavy two-clock design would need to keep getting
right on top of the query-order fix — judged too much surface area for a
mechanism that had already produced one bug. Superseded by 13b before a
single training run was launched under it.

### 13b. `rope_state_scale` — shipped with a real bug, caught by direct log comparison, then fixed

Single-clock design (`_scaled_state_positions`, `kvmem/hmn.py`): every
non-STATE token keeps its ordinary real position (identical to plain RoPE),
while STATE-region tokens' real index gets divided by `state_scale` (e.g.
1e6) — chosen specifically to avoid 13a's reset-based bug class entirely
(no per-block-type bookkeeping to get wrong).

**First version had a real bug, found by comparing against the original
`hmn_single_recall_c64` baseline's own logs at matched step counts** — not
by inspecting the position math in isolation (which looked correct: cross-
query distance did collapse to numerically negligible, verified offline
before trusting). The bug was that dividing the ENTIRE real index by
`state_scale` also crushed the WITHIN-region spacing: real STATE0 slot
positions 66-73 (native spacing 7) collapsed to a spacing of ~7e-6 after
scaling — six orders of magnitude smaller, on top of `state_vocab_size=2`'s
own content ambiguity, destroying BOTH channels a model needs to
disambiguate individual STATE slots. This wasn't caught by the position-math
verification alone; it only showed up as `hmn_single_recall_c64_scaledrope`
training far worse than the baseline at matched steps (loss stuck ~2.9 vs.
baseline's 0.03, val stuck at best=3.0% vs. baseline's 100%) — a
single-chunk task with no cross-query ambiguity to fix at all, so the
failure couldn't be explained by "the shortcut is still there," only by
something breaking ordinary recall outright.

**Fixed**: `pos[i] = (i - s0) + s0 / state_scale` for a STATE region
starting at `s0` — within-region spacing stays native/unscaled (slot k's
position is still exactly `k` more than slot k-1's), only the region's
overall baseline (`s0`, where it sits in the whole sequence) gets
compressed. Re-verified both properties simultaneously before trusting:
within-STATE0 slot spacing came back to exactly `7.0` (matching the real
7-position range), and cross-query invariance was still intact (`batch`'s
and `interleave_delayed`'s same query slot see identical distance-to-
STATE0/STATE1 to 6 decimal places, unaffected by the fix).

**Result after the fix, `hmn_single_recall_c64_scaledrope` (rerun from
scratch)**: tracked the ORIGINAL baseline's own convergence curve almost
exactly through the first half of training (loss=1.560 at step 20000 vs.
baseline's 1.5525 — essentially identical), then decelerated somewhat
relative to baseline in the second half (loss=0.630 at step 80000 vs.
baseline's ~0.01-0.02 by that point) but kept climbing to a final
best-val=42.3% — far short of baseline's 100%, but two orders of magnitude
better than the broken first version's 3.0%, and with no sign of being
stuck (still improving when the stage ended). `hmn_weave_c64_scaledrope`
(the actual `batch`/`interleave_delayed` test) is warm-started from this
checkpoint and running as of this writing — see the live section below for
current numbers, not yet resolved.

### 13c. `relpos` (`kvmem/hmn_relpos.py`) — redesigned mid-flight, then failed outright

Separate, independent mechanism: no RoPE at all, replaced with a learned
bias baked directly into the SDPA `attn_mask` at exactly the "d steps back"
relative positions (`relpos_k` distances, default 2) — motivated by the
same probe result but attacking it from the opposite direction (remove
essentially all distance signal except a small fixed local window, rather
than scoping WHERE the signal applies as in 13a/13b).

**First version** (originally named `relpos_shaw`, since it was literally
Shaw et al. 2018-style relative position embeddings): a single learned
constant per `(head, distance)` — same bias value added regardless of what
token content was actually involved. Verified mechanically correct (a
targeted test with the bias cranked to an extreme value showed attention
collapsing ~100% onto the intended "d steps back" column).

**Redesigned to a query-side content-dependent gate** after discussion of
whether input-dependence would help: `Linear(d, relpos_k*n_heads)` applied
to the QUERY's own current hidden state (not the key/attended-to token, and
not a combined query-key dot product) — chosen specifically for KV-cache
friendliness, since the query's own hidden state is already being freshly
computed every decode step regardless of caching, unlike a key-side or
combined version which would need an entirely new cached tensor threaded
through `past_kv`/`return_kv` alongside K/V. This required the attention
mask to gain a genuine per-example batch dimension (`(B,H,L,L_kv)` instead
of the shared `(1,1,L,L_kv)` broadcast every other mechanism in this
codebase uses), since content-dependent bias varies per training example
where the old constant/permission mask didn't. Verified before trusting:
batched forward ran cleanly and the computed gate values differed
across different examples in the same batch (confirming genuine
content-dependence, not an accidentally-shared constant). Renamed
`relpos_shaw` -> `relpos_enabled` at this point, since the mechanism was no
longer Shaw's design.

**Result: failed outright.** `hmn_single_recall_c64_relpos` (single chunk,
no cross-query ambiguity at all — the same trivial task 13b's base
checkpoint reached 100% on, and the broken 13b bug still reached loss~2.9)
finished its full 100000-step run at only **best=2.4%**, comparable to (very
slightly worse than) 13b's diagnosed-broken 3.0% — loss never dropped below
~3.5-3.7 for the entire run. Unlike 13b's bug, no code defect has been
found in `relpos`'s implementation to explain this — the mechanism was
verified mechanically correct at both the fixed-constant and
content-dependent-gate stages. The most likely reading, not yet
investigated further: restricting position information to a k=2-token
local window may simply be insufficient signal for THIS task's genuinely
local-order-dependent needs (byte-by-byte generation coherence), as
distinct from the STATE-addressing shortcut it was built to remove — i.e.
the mechanism may have successfully killed the shortcut while also being
too weak to support ordinary recall, the same FAILURE MODE as 13b's first
buggy version (breaks basic recall) but via a genuinely different cause
(insufficient local signal vs. corrupted slot disambiguation).

**Triggered a pre-authorized queue-reordering rule**: given `relpos`'s
stage0 finished with best-val under a ~10% threshold (matching the
"comparable to 13b's diagnosed-failure level" criterion), the training
queue was reordered live — the `relpos` chain (including the
already-auto-started `hmn_weave_c64_relpos`) was killed before wasting
compute warm-starting from a broken base, and the fixed 13b `scaledrope`
queue was promoted to run immediately instead of waiting its turn. `relpos`
is not currently queued to resume; revisiting it (larger `relpos_k`? a
key-side or combined gate despite the KV-cache cost? diagnosing why local-
window-only position broke basic recall?) is unstarted follow-up work, not
abandoned by a deliberate decision the way 13a was.

### Current state (live, as of this writing)

`hmn_weave_c64_scaledrope` (13b's fix) running, stage0 (n_chunks=2,
80000 steps), warm-started from the 42.3%-best-val base checkpoint:

| step | batch | stream | interleave_delayed | MEAN | loss |
|---|---|---|---|---|---|
| 10000 | 3.0% | 9.2% | 2.4% | 4.9% | 4.75 |
| 20000 | 6.0% | 8.9% | 0.9% | 5.3% | 4.29 |
| 30000 | 6.8% | 11.6% | 0.6% | 6.3% | 3.88 |
| 40000 | 6.2% | 13.1% | 1.2% | 6.8% | 3.51 |

Not yet resolved — `stream` climbing slowly as expected, `batch` roughly
flat, `interleave_delayed` (the shape most directly implicated by the
shared-`E2`-prefix conflict) still barely off its floor. Historical
ceiling for `batch`/`interleave_delayed` under every prior mechanism was
8-20%; `stream` historically reached 42-49% by the end of a full run. Stage0
has 40000 steps left, then stage1 (n_chunks=4, 160000 steps) is the harder,
more definitive test. This section should be updated once stage0/stage1
finish with the actual resolution, not left as a live snapshot.

**Re-ran `kvmem/probe_positional_shortcut.py` against this in-progress
checkpoint** (`stage0_last.pt`, ~step 40000/80000, not converged) — first
had to fix two real bugs the script itself had: it never passed any
position argument to `model(...)` at all (would have silently tested the
model under plain sequential positions rather than the `rope_state_scale`
scheme it was actually trained under — meaningless result), and it built
the permission mask with the default `hops=-1` instead of the actual
trained `hops=1`. Both fixed (mirroring the same position/mask
construction `train()`/`ar_decode_traj_nokv` use), verified to import and
run cleanly. Result at this mid-training snapshot: baseline match=5.8%,
swap-test match vs. the swapped-in content=2.2%, vs. the slot's usual
content=7.1% — a 4.9pp gap, under the script's own 10pp threshold, so
reported as `INCONCLUSIVE` rather than a clean win. But contrast the GAP
SIZE against the original RoPE checkpoint's result (91.1% position / 0.4%
content, a 90.7pp gap): the position-preference magnitude has collapsed by
roughly an order of magnitude relative to the original gap, even before
training has converged. Suggestive that the fix is genuinely reducing the
shortcut's dominance rather than just moving aggregate match% around, but
not conclusive on its own — both numbers are low simply because the model
hasn't finished training yet, and the swap test hasn't been re-run against
a converged checkpoint. Re-running this probe once stage0/stage1 actually
finish is the natural follow-up for a definitive answer.

### 14. `hmn_locate_nope_curriculum_dense` — is the architecture or the dataset a limiting factor?

Two independent diagnostic questions asked while `hmn_locate_nope_curriculum_dense`
was mid-run (`lr_max=1e-4`, stage0 already at MEAN=59.9% by step 72000, stage1
in progress) — both answered `NO`, with the actual reproducible script
preserved at **`kvmem/probe_signal_propagation.py`** (`--mode signal` /
`--mode ambiguity`).

**Is the 8-layer/d=64/n_heads=4/NoPE architecture too limiting** (vanishing
gradients, degenerate attention that never sharpens, exploding activations)?
Compared a fresh random-init model against `stage0_best.pt` (both `single_attn`,
`rope=False`, `null_kv=True`, `rmsnorm=True`) on per-layer activation norm,
attention entropy, and gradient norm, plus a short 300-step CPU training run.
**Two real bugs found and fixed in the diagnostic itself before trusting any
result** (per this project's own house rule — verify infra before trusting
it): (1) first run evaluated `stage0_best.pt` (trained only on `chunk_len=8`)
against a `chunk_len=16` batch it had never seen, making the pretrained model
look worse than random init (loss 17.6 vs 5.5) — fixed by testing on one of
stage0's own trajectories instead; (2) the attention-entropy normalization
divided by `log(n_valid)`, and rows with exactly one attendable position
(`n_valid=1`, common for early causal rows) have `log(1)=0` — any epsilon
clamp on that denominator blew a near-zero numerator up into the millions
(`attn_entropy=1753215` in the first run) — fixed by excluding `n_valid<2`
rows from the normalized average rather than clamping.

**Result, post-fix**: gradient norm is LARGER in early layers than late ones
(layer0=0.52→layer7=0.10 at random init; layer0=9.49→layer7=2.45 pretrained)
— the healthy, expected residual-network pattern (skip connections carry
gradient to early layers nearly unattenuated), not vanishing gradients.
Activation norm grows across depth (0.3→30.1) but each block re-normalizes
its own input via RMSNorm before attention, so this is normal transformer
behavior, not instability. Attention entropy at random init is uniformly
diffuse (0.73-0.90 every layer); after training, layer 1 sharpens to 0.30 —
real evidence the architecture DOES learn decisive, content-based attention
under NoPE, not permanently-uniform. **Verdict: no evidence of an
architectural/signal-propagation ceiling** — the slow real-run convergence
(tens of thousands of steps even at the easiest stage) looks like the
expected cost of learning genuine content-only locate, not a capacity limit.

**Is `data_kind='random'` (uniform i.i.d. bytes) itself a limiting factor** —
i.e., does the training data contain genuine ambiguity (the TRUE warmup
excerpt's exact byte string recurring elsewhere in the same chunk, making
"locate this" unsolvable in principle for that sample)? Measured empirically
per `(chunk_len, warmup_len)` pair used across this project's locate configs:
under 0.1% even at the shortest tested `warmup_len=2`, `chunk_len=64`
(cross-checked against the birthday-bound approximation for "does ANY
duplicate window-pair exist in the chunk," ~3% — consistent once accounting
for the two being different questions: only 2 of ~63 windows are the
colliding pair even when one exists, so P(the specific TRUE excerpt is one of
them) ≈ 2/63 × 3% ≈ 0.1%). **Verdict: the dataset is not a meaningful
limiting factor** — if anything, uniform random (max-entropy, incompressible)
data is close to the best-case distribution for this task, since structured/
repetitive data would introduce MORE genuine duplicate substrings, not fewer.
The near-garbage qualitative output seen at `warmup_len=2` (documented in
CLAUDE.md's positional-shortcut/qualitative-decode entries) is better
explained as a learning-difficulty issue — the exact-match attention
algorithm is harder to learn from only 2 bytes of signal — than a
data-imposed ceiling.

### 15. The STATE-role ambiguity, opcode tokens, and end-of-turn STATE redesign — promoted to `kvmem/hmn.py`

A design conversation (fully implemented, ported, and promoted to be the
new `kvmem/hmn.py` — see "Status" at the end) that started from a real gap
in the chat-tag-free
fork and ended up redesigning how STATE emissions work generally, in a way
that also resolves a genuine architectural redundancy present in `hmn.py`
too, not just the fork.

**The original gap**: without chat tags, an encode chunk's claim-STATE and
a query's own round-0 recall-STATE use the exact same token family
(`_cyclic_state_ids(..., family=0)`) — there is no vocab-level signal
distinguishing "this STATE just claimed an encode chunk" from "this STATE
is a fresh query's blank register." The only available signal is the
local order pattern (raw bytes precede an encode-claim; raw bytes follow a
query's STATE) — real, but weaker than what `HMN_QUERY_OPEN` used to give
for free. Separately: bare no-op relay hops (`S<n>` tokens) ALSO share
that same family in `hmn.py` itself, tags or no tags — chat tags never
disambiguated *that* case in the original file either.

**First proposed fix, then generalized**: give no-op hops their own STATE
family (a third family alongside regular/feedback). Then generalized
further into a **factorized opcode + shared value alphabet**: instead of
N separate families (each with its own private `state_vocab_size`-sized
alphabet), use ONE opcode token per STATE emission (`update`/`noop`/
`feedback` — 3 values) followed by `state_len` value tokens drawn from a
SINGLE shared alphabet reused across all three roles. Vocab cost: `3 +
state_vocab_size` extra IDs instead of `3 × state_vocab_size` — smaller,
but the real benefit is representational: the model learns what "value 0
vs value 1" means ONCE, shared across every role, instead of relearning it
independently per family. Given how step/data-constrained this whole
project is, that's a real sample-efficiency argument, not just a
vocab-size nicety.

**The double-state redundancy** (found by asking "isn't this redundant"
about the query's own STATE existing *separately* from the encode's claim-
STATE): a query's round-0 STATE is a genuinely separate emission from the
chunk's claim-STATE it reads from — is that ever necessary, or just
architectural uniformity? Precise answer, derived from `_relay_source`
(reads `prev_rb['sl0':'sl1']` — literally *this op's own* STATE, not
whatever it read): a query's own STATE emission is **needed whenever a
later op relays from it** — true under any `hops` setting, because it's
what lets a chain accumulate (`state_t=f(state_{t-1},x_t)`) rather than
stay static. It's **provably redundant only for the terminal op** (the
last, or only, op in a chain) — nothing downstream ever reads its STATE,
so warmup/response could attend directly to whatever single source it
would have read, with zero behavioral difference. For every current
`weave_mix` locate config (`n_chunks=1`, one query, nothing after it) —
trivially the terminal op of a length-1 chain — this redundancy is real
and applies right now.

**The `hops=-1` vs `hops=1` distinction that motivated going further**: the
query-STATE's *aggregation* role (reading from MULTIPLE priors at once)
only exists under `hops=-1` (unbounded/routing) — under `hops=1` there's
only ever one accessible prior, so "aggregation" doesn't apply. But the
STATE emission is still not redundant for non-terminal ops under `hops=1`
— it's the transition-function OUTPUT itself, the actual `h_t=f(h_{t-1},
x_t)`, which is what the next op needs to relay from. Masking alone can't
substitute for this (masking controls which EXISTING positions attend to
which; it can't manufacture a value that was never computed and written
into the sequence) — so "just merge the hop into a single state via
correct masking" works ONLY for the terminal op, not in general.

**The redesign this converged on**: keep exactly one STATE block per op
(never two), but reposition it to the END of a query's turn instead of the
start, and let warmup/response attend DIRECTLY to whatever `hops` makes
available (no intermediate "pre-filter" register they're forced through
first):
```
current:  [STATE][warmup][response]      — STATE computed first, a forced pre-filter
redesign: [warmup][response][STATE]      — warmup/response attend directly to hops-permitted
                                            priors; STATE computed LAST, purely as the
                                            chain-continuation handoff for the next op
```
This isn't a new shape invented from nothing — it's exactly what refine
rounds already do for `slb` (the STATE built *after* incorporating argmax
feedback, handed forward as what the next round reads), generalized to
every op instead of only refine rounds. Mechanically: `S` already claims
the immediately-preceding unclaimed `E` (encode-STATE) or falls back to a
bare no-op hop — this redesign extends the SAME claim logic so `S` can
ALSO claim an immediately-preceding, just-finished `Q` (building its
end-of-turn STATE from whatever `hops` permits plus its own warmup+
response), making a terminal query (no trailing `S`) and a non-terminal
query (`Q(...) S`) an *explicit, visible* distinction in the DSL itself,
not an implicit one buried in `_relay_source` logic.

**Worked trajectories, verified by hand before any code was written** (per
this project's own house rule — verify masking changes against the actual
matrix, not just "does it run"), using `chunk_len=4, state_len=2,
warmup_len=1`, `OP_U`=update opcode:
- **Single recall** (`E,S,Q` — terminal): `[c0 c1 c2 c3][OP_U sA0 sA1][w r0 r1 r2]` — ONE STATE block total (the query's own is omitted — terminal, nothing relays from it). L=11.
- **Batch** (`E1 E2 Q1 Q2`, `Q1` has a successor): `[c0][OP_U sA][c1][OP_U sB][w1 r1][OP_U sQ1][w2 r2]` — `Q1` gets an end-of-turn STATE (needed, `Q2` relays from it); `Q2` (terminal) doesn't. `Q1` (op_idx=0, exempt) sees BOTH `sA` and `sB` directly — both already exist causally, since batch groups encodes before any query.
- **Stream** (`E1 Q1 E2 Q2`, same op roles, interleaved order): `[c0][OP_U sA][w1 r1][OP_U sQ1][c1][OP_U sB][w2 r2]` — same block set, reordered. `Q1` only ever sees `sA` (`sB` doesn't exist yet causally — nothing to do with `hops`). Under `hops=1`, `Q2` is blocked from `sA` AND `sB` directly, relying solely on `sQ1` — computed *before chunk1 even existed*. **This reproduces, from the redesigned mechanism, the exact structural bottleneck already observed empirically**: `hmn_weave_mix_accum_rnn_repeat8`'s persistent weak spot at "op1" (4.2-8.3% match, the one op that stays weak in every pattern) is precisely this case — the first non-exempt op under `hops=1`, structurally starved of any channel to its own chunk's compressed content when the ordering is stream-like. Confirms the redesign doesn't hide or change this known limitation, only removes the redundant pre-filter step around it.
- **3-query chain** (`E1 Q1 E2 Q2 E3 Q3`) isolates where `hops=1` and `hops=2` actually diverge — identical everywhere except `Q3`'s row (the first op with two distinct priors to bound between): `hops=1` gives `Q3` only `sQ2`; `hops=2` gives `Q3` both `sQ2` AND `sQ1` — a genuine "skip connection" to two hops back, still bounded (unlike `hops=-1`, which would also include every encoding-pass STATE regardless of distance).

**Refine rounds — a real boundary problem found, and why it's refine-only**:
worked through a hypothetical refine trajectory (ground-truth round-0 feed,
model's own argmax fed back, a `feedback`-opcode STATE, repeat, final
update) and found a genuine ambiguity `chunk_positions_traj`'s existing
design doesn't have: a refine round's argmax feedback (`am`, the model's
OWN argmax predictions read off the previous round's response logits, per
`use_actual_argmax=True`) sits directly adjacent to that previous round's
response (`r`) — BOTH raw byte-range tokens (0-255), with no vocab-level
or structural cue marking where one ends and the other begins. This is
NOT the same situation as `E`→STATE or STATE→`warmup` transitions (raw
bytes vs. STATE-range IDs 259+ are already vocab-distinguishable) — it's
the one place two *different-role* regions share the *same* token range
with nothing between them.

**Why this doesn't also apply to encode/query boundaries**: given the
discipline "always add a trailing `S` after `E` or `Q` unless it's truly
the last op in the whole trajectory," every op's content is immediately
followed by a STATE commit (vocab-distinguishable), so two ops' raw
content is never directly adjacent. Framed precisely: **encode is a
special case of query** — both are "process content, then commit an
update to STATE," differing only in that encode's content is pure
ground-truth (a chunk) while a query's is ground-truth-prefix (`warmup`)
+ free generation (`response`). Refine's `r`→`am` transition is different
in kind: it happens *inside* one op's own refine elaboration, BEFORE any
commit — the "always commit before the next thing" discipline doesn't
protect it, because nothing commits between them.

**Fix, refine-only**: move the `feedback` opcode to PRECEDE the argmax
content instead of following it — `[r0][OP_F][am0][sF0 sF1][w1][r1]`
rather than `[r0][am0][OP_F sF0 sF1][w1][r1]`. `OP_F` now announces
"feedback content starts here" before the ambiguous raw bytes arrive; the
STATE values (`sF`) still need no separate marker since raw-byte→STATE-ID
remains vocab-distinguishable on its own. Deliberately scoped to refine
only — encode/query commit-boundaries stay marker-free by the trailing-`S`
discipline, not by adding more opcodes.

**Status**: fully implemented, ported, and PROMOTED. Round-0 (`chunk_
positions_traj`/`chunk_mask_fb_traj`/`make_batch_tagged`/`ar_decode_traj_
nokv`) was implemented and verified first (`S` claims either a pending `E`
or a just-finished `Q`, opcode-prefixed `1+state_len`-wide STATE blocks;
`chunk_mask_fb_traj` rewritten around a single positive `allowed_state`-
based allowlist per op instead of the old three-part chunk-blackout/
nochain-blackout/bottleneck logic), directly verified against ALL FOUR
hand-derived trajectories above (single-recall, batch, stream, 3-query
hops=1-vs-2) via a standalone script checking exact block-to-block mask
permissions — every assertion passed, including the subtle `hops=1`-vs-
`hops=2` divergence and the stream+`hops=1` structural bottleneck
reproducing exactly as predicted from first principles.

Refine-round support (the `feedback`-opcode-before-content boundary fix
worked out above) was then added: `OP_FEEDBACK` placed before the argmax
content (`opf0`), a `'refine'`-type rec_block's `sl0/sl1` holds the
feedback STATE values (exactly `state_len` wide, no opcode prefix — the
opcode already sits at `opf0`), and a separate `end_sl0/end_sl1` holds the
optional end-of-turn STATE (only on the last round, if claimed by a
trailing `'S'`). `_fill_argmax_fb` needed no changes (already field-name
compatible).

All three remaining position-builder paths were then ported: `chunk_
positions_hop`/`chunk_mask_fb_hop` (the `chain_steps` path) reimplemented
as thin wrappers delegating to `chunk_positions_traj`/`chunk_mask_fb_traj`
(chain_steps is structurally just a batch-style ops list with a trailing
`'S'` after every step but the last); `chunk_positions_iq_global_rw_
tagged`/`chunk_mask_fb` (the legacy global-window `traj_mix` path) ported
as its own dedicated function, since its fixed-size sliding-window
`out_len` shape genuinely differs from `chunk_positions_traj`'s "through
the end of span" query handling; `chunk_positions_stitch`/`make_batch_
stitch`/`ar_decode_stitch` (the `stitch_mix` path) ported as its own
dedicated function, with `ar_decode_stitch`'s old "cache the closing tag"
KV-cache step replaced by a general `_sweep_known(seg_start, seg_end)`
helper (there's no more closing tag to key off since tags are gone). Each
path was individually smoke-tested (train + eval decode, and argmax
injection where relevant) before moving to the next.

**Two critical bugs were caught during PRE-PROMOTION verification** (proactively
checking the named trajectory-pattern constructors before trusting them,
rather than relying on the earlier ad-hoc DSL-string smoke tests, which had
always included an explicit trailing `'S'` by hand):
1. `traj_batch`/`traj_stream`/`traj_interleave_delayed`/`traj_repeat_query`
   never inserted a trailing `'S'` between consecutive queries. Under the
   OLD chat-tag design every query automatically got a pre-filter STATE
   row; under the new design a query's STATE is optional and must be
   explicitly claimed by a trailing `'S'` — without it, every query in
   these four patterns became terminal (no end-of-turn STATE at all),
   silently breaking the relay chain for every config using `batch`/
   `stream`/`interleave_delayed` (`hmn_weave_mix.py`, `hmn_recall_queue.py`,
   `hmn_weave_mix_accum_rnn.py`, etc.) had promotion happened before this
   was caught. Reproduced directly (`AssertionError: op_idx=0 has no
   end-of-turn STATE ... but a later op is trying to relay from it`).
   Fixed by joining query strings with `' S '` (inserting `'S'` after every
   query except the last) in all four constructors.
2. `_relay_source` mishandled `'noop'`-type blocks: its type check was
   `('sl0','sl1') if prev_rb['type']=='initial' else ('end_sl0','end_sl1')`
   — a `'noop'` block (which always stores its relayed STATE in `sl0/sl1`,
   having no separate always-present feedback-values field to conflict
   with) incorrectly fell into the `else` branch, raising `KeyError:
   'end_sl0'` on `traj_decay_curve` (which legitimately produces bare `'S'`
   noop blocks in its relay chain). Fixed by checking `prev_rb['type'] in
   ('initial', 'noop')` vs. `'refine'`.

Both fixes re-verified via direct mask-matrix construction across `batch`/
`stream`/`interleave_delayed`/`repeat_query`/`decay_curve` at `hops ∈
{-1, 1, 2}` — all build successfully with no crash — AND via a real
end-to-end training-loop smoke test (`weave_mix` dispatch, `pattern=
'batch'/'stream'/'interleave_delayed'`, `hops=1`, 6 steps) showing sane
loss (~5.53-5.57, the random-guess baseline) with no crashes across
training AND validation/decode. A separate direct mask-permission check
confirmed the relay itself is load-bearing as intended: under `hops=1`,
a second query's warmup can see its predecessor's end-of-turn STATE
(`ALLOW`) but is correctly blocked from every encoding-pass STATE directly
(`BLOCK`, `BLOCK`).

**The archive+promote step has now happened**: `kvmem/hmn.py` (the old
tagged design, `V=274`) was archived verbatim to `kvmem/hmn_v4_backup.py`,
and `kvmem/hmn_notags.py`'s content was promoted to be the new `kvmem/
hmn.py` (module docstring rewritten for its new role as the primary file;
`kvmem/hmn_notags.py` no longer exists as a separate file). `hmn_v4_backup.py`
and the earlier `hmn_v1-v3_backup.py` diffing snapshots were later deleted
entirely as part of a cleanup pass (pure diffing artifacts, never imported
by anything — their content is superseded by the current file plus this
doc's own narrative). `kvmem/configs/
hmn_notags_w25.py` and `hmn_notags_locate.py` were updated to invoke
`kvmem.hmn` directly (no separate module needed) and verified to still
load cleanly with `V=271`.

**Caveat not yet resolved**: every config with `_pretrained_ckpt` set
(`hmn_single_recall_c128`, `hmn_weave_c64*`, `hmn_weave_mix*`, `hmn_
weave_mix_accum_rnn*`, `hmn_stitch_src1024`, etc. — roughly a dozen
configs) warm-starts from a checkpoint trained under the OLD tagged
vocab (`V=274`, tags at IDs 256-261). Loading it into the new opcode
vocab (`V=271`, opcodes at 256-258) via the existing shape-mismatch-
tolerant loader (`kvmem/hmn.py`'s `_pretrained_ckpt` handling in `train()`)
would silently reinterpret old tag-token embeddings as new opcode-token
embeddings at the SAME IDs — wrong, not just stale, since the loader only
checks tensor shape, not semantic vocab compatibility. Do not warm-start
any of those configs from pre-promotion checkpoints without addressing
this (either retraining the warm-start source from scratch under the new
vocab, or writing a loader that explicitly skips/re-inits the tag-ID rows
rather than trusting shape-match alone). Configs that train from scratch
(`hmn_notags_w25`, `hmn_notags_locate` — no `_pretrained_ckpt` key) are
unaffected and were used to verify the promotion (see CLAUDE.md's Results
section for the training-run status).

### 16. Post-promotion config cleanup, RoPE-vs-NoPE head-to-head, anchor-variation attack on the positional shortcut, and mechanistic verification tooling

**Config cleanup.** Once the promotion (§15) stabilized and `hmn_notags_w25`
started producing real results, the active `kvmem/configs/` directory was
pruned: 19 completed/superseded configs (`hmn_recall_queue`, `hmn_routing_
4to1_state`, `hmn_weave_mix*`, all `*_dualrope`/`*_scaledrope`/`*_relpos`
variants, `hmn_single_recall_c128`/`_repeat4`/`_locate`, the `hmn_squeeze_
random_n4`/`_sanity_bigmodel_n4`/`_sweetspot_n4` variants, `hmn_weave_c64_
adaptive`, `hmn_accum_rnn_sanity`) were `git mv`'d into `kvmem/configs/
archive/` — their numeric results are already preserved in CLAUDE.md/this
doc and in `logs/`, so archiving the config file itself loses nothing.
Three configs were explicitly kept in place as future experiment bases per
request: `hmn_weave_c64.py`, `hmn_stitch_src1024.py`, `hmn_squeeze_markov_
n4.py`. Separately, the `hmn_v1_backup.py`–`hmn_v4_backup.py` dated
snapshots (pure diffing artifacts, see CLAUDE.md's docs table) were deleted
outright.

**A confirmed correctness bug + a deprecation, found during a code review
of `kvmem/hmn.py`** (requested explicitly as a review, not part of any
feature work): `_dual_positions` (`dual_rope`) and `_scaled_state_positions`
(`rope_state_scale`) both still referenced stale `sla0`/`sla1`/`slb0`/
`slb1` refine-block field names from BEFORE the opcode/end-of-turn-STATE
redesign — confirmed via direct reproduction to crash (`KeyError: 'sla0'`)
on any refine trajectory, and further investigation while fixing this
found a second, more severe latent bug: both functions ALSO crashed (or,
worse, silently miscomputed) on the ordinary case of a *terminal* query
(`sl0=None`), because they assumed STATE always precedes warmup/response —
true under the old pre-filter-STATE layout, false since the end-of-turn-
STATE redesign. Fixed properly with per-op origin tracking (mirroring the
`op_first_w0` pattern already used in `chunk_mask_fb_traj`), verified
across terminal/relay-chain/refine cases. Per user instruction, both
`dual_rope` and `rope_state_scale` are now marked **deprecated** in their
own docstrings (all three attempted positional-shortcut fixes — dual_rope,
rope_state_scale, relpos — either failed or were abandoned, see §12-13) —
kept correct rather than deleted, but flagged not to build new work on.
A third finding, `_iter_forward_segments`/`_forward_segmented`
(`forward_granularity`), turned out to be broken in EVERY case under the
current design (not just refine) — `start = rb['sl0']` is wrong whether
`sl0` is `None` (terminal, crashes) or present (non-terminal, `sl0 > c1`
now, so `seg_start > seg_end` silently). Rather than patch this without
the segment-tiling verification the project's own masking-change rule
requires, it now raises `NotImplementedError` immediately — honest
disabled-state until someone re-derives and re-verifies it properly.

**`hmn_notags_w25` vs `hmn_notags_w25_rope` — a live, step-aligned RoPE-
vs-NoPE comparison.** `hmn_notags_w25_rope` (identical config to `hmn_
notags_w25` — same 4-stage curriculum, same anchor-varying single-chunk
recall — except `rope=True`) was run specifically to answer: does RoPE
help or hurt under the new opcode/no-tags design, now that NoPE is the
default. Result, unambiguous: RoPE converges dramatically faster and
higher than NoPE at every single directly-comparable stage/step, no
exceptions. Early-stop thresholds (80% val MEAN) that NoPE never reached
in stages 0/1/3 (manually stopped at 65.3% best in stage3, never
converged) were cleared by RoPE within a fraction of the steps: stage0
early-stopped at 91.7% (step 48000, vs NoPE's 59.7% at the same step),
stage1 at 81.5% (step 36000, vs 53.2%), stage2 at 90.0% (step 72000, vs
62.3%), and stage3 itself early-stopped at 80.2% (step 288000) — the one
stage NoPE never converged on at all. One transient dip (stage3, step
180000: RoPE dropped to 24.9% while NoPE was 62.5% at that step) recovered
fully by the next eval (79.1%) and was not a sustained regression — plausibly
adaptive-reweighting noise, not instability. This is a genuinely
surprising result worth flagging honestly: the earlier `hops=1`
positional-shortcut investigation (§12-13) was all conducted UNDER RoPE
(the old tagged design's default), and every fix attempted THERE (dual_rope,
rope_state_scale, relpos) tried to change the RoPE mechanism itself —
but this head-to-head suggests RoPE alone (under the NEW opcode/no-tags/
end-of-turn-STATE design) is a strong convergence aid, not the source of
the problem it was once blamed for. That reframes the positional-shortcut
question entirely (see below).

**`hmn_notags_weave_anchor(_rope)` — attacking the positional shortcut via
anchor variation instead of position-encoding changes.** Given the RoPE
result above, the natural next question is whether the ORIGINAL `batch`/
`interleave_delayed` positional-shortcut ceiling (§12, all three RoPE-
mechanism fixes failed to move it) can instead be fixed by never letting a
query's warmup land at a single fixed position — the same idea `hmn_
notags_w25`'s own `_grid` anchor sweep already validated for single-chunk
recall, generalized to `chunk_positions_traj`'s multi-query `Q(s,e,
warmup_start,warmup_len)` DSL token. New helper `_grid_shapes` (in both
`hmn_notags_weave_anchor.py` and its `_rope` sibling) generates, for each
of `batch`/`stream`/`interleave_delayed`, a sweep across `chunk_len` in
{8,16,32,64} x several non-zero-biased anchor offsets — built on top of
`hmn_weave_c64.py`'s two-stage difficulty ramp (`hops=1`, nc=2/wc=1 then
nc=4/wc=2), the exact shape the original positional-shortcut ceiling was
measured at. Caught and fixed one real bug in the first draft before it
shipped: `_e_block` initially emitted bare `E{n}` tokens which silently
ignore the swept `chunk_len` (only `E(len)` actually sets it via `parse_
traj_dsl`) — every entry would have trained at the stage-default `chunk_
len=64` regardless of the intended sweep. Fixed and re-verified (`dsl_
chunk_len` confirmed to vary correctly across entries) before queuing.

Two variants were built and QUEUE-REORDERED per explicit request (RoPE
first, since it was clearly winning): `hmn_notags_weave_anchor_rope.py`
(rope=True, warm-started from `hmn_notags_w25_rope`'s checkpoint) now runs
BEFORE `hmn_notags_weave_anchor.py` (NoPE, warm-started from `hmn_notags_
w25`'s checkpoint) — both same-vocab warm-starts, safe under the
`_pretrained_ckpt` caveat above.

**Positional-shortcut probe — two real bugs found and fixed, plus CLI
extensions to make it usable against an anchor-swept checkpoint.**
`kvmem/probe_positional_shortcut.py` predates the opcode redesign and
crashed immediately (`ValueError: could not broadcast ... shape (8,) into
shape (9,)`) — it wrote `state_len`-wide STATE blocks where the actual
layout is `1+state_len` wide (opcode + values). Fixed (write `HMN_OP_
UPDATE` at `sl0`, values at `sl0+1:sl1`, matching `make_batch_tagged`'s own
convention). After that fix, the FIRST corrected run still came back with
a near-zero "baseline" sanity check (0.2-0.4% — the model apparently
couldn't even recall its own correct warmup), which turned out to be a
second bug: the probe hardcoded `chunk_len=64` and read `warmup_len` from
`hp['warmup_len']` — the stage-level FALLBACK, never actually trained on
for a `_grid_shapes`-style config where every entry sets its own
`warmup_len` via the DSL. Worse, `traj_batch(2,1)` (used to build the test
trajectory) always defaults `warmup_start=0`, silently locking the probe
to the one anchor value the whole anchor-variation design exists to avoid
over-representing — and empirically the weakest/most-undertrained corner
at `chunk_len=64` specifically (longest possible response). Added
`--chunk-len`/`--warmup-len`/`--anchor` CLI overrides and replaced the
`traj_batch(2,1)` call with an equivalent hand-built `ops` list that
threads `anchor` through each `Q`'s 3rd argument. With overrides picking
combinations actually trained (chunk_len 8/16/32, several anchors,
`stage0_best.pt` of `hmn_notags_weave_anchor_rope`), the probe finally ran
cleanly: **content-addressed at every converged length** (0-3% match to
the wrong/"usual" chunk vs. 73-100% match to the correct swapped content —
no sign of the old shortcut). `chunk_len=64` came back inconclusive
because baseline recall itself hasn't converged there yet (0.4%) — not
evidence of anything, just insufficient training so far at that length.

**Mechanistic verification, beyond the purely behavioral swap test.** A
correct swap-test OUTPUT doesn't by itself prove the network's internal
computation actually reads the swapped STATE — it could in principle be
right for an unrelated reason that happens to correlate in this
construction. Two additions close that gap:
- `kvmem/hmn.py`: `MHAttention` gained an opt-in diagnostic flag,
  `capture_attn` (unset/`False` everywhere by default — checked via
  `getattr(self, 'capture_attn', False)`, so it changes nothing about
  training/eval/decode unless explicitly set). When enabled it forces the
  attention module through its ALREADY-EXISTING manual-softmax branch
  (previously only reachable via `logit_cap`/`attn_temp`) instead of fused
  SDPA — mathematically identical, just not opaque — and stashes `self.
  last_attn_probs` (B,H,Lq,Lkv) for inspection after the call.
- `kvmem/probe_mechanistic_addressing.py` (new): teacher-forces the swap-
  test continuation with chunk1's TRUE bytes (the content-addressed
  hypothesis as the target, not whatever the model would freely generate),
  then checks two independent internal signals: (1) per-layer attention
  mass from the response rows onto STATE0's columns (the "usual"/wrong
  slot) vs STATE1's columns (the swapped-in, correct content); (2)
  gradient saliency at the embedded input (before layer 0, via a forward-
  pre-hook + `retain_grad()`) — L2 norm and signed input×gradient, backprop
  from the teacher-forced NLL loss.

Run at two lengths against `hmn_notags_weave_anchor_rope`'s `stage0_
best.pt` (chunk_len=32/warmup_len=8/anchor=0 and chunk_len=16/warmup_len=4/
anchor=0), both gave a clean, consistent story: layer 1 concentrates
85-94% of attention mass onto STATE1 (vs 6-15% onto STATE0); gradient L2
norm at STATE1 is 2.7-3x that at STATE0, with the input×gradient sign
flipping as expected (positive at STATE1 — increasing it helps the
correct-token log-prob — negative at STATE0). Layers 6-7 show ~zero direct
attention to either STATE region, consistent with layer 1 having already
pulled the needed value into the residual stream, with later layers
refining the local prediction rather than re-reading STATE. **This
mechanistically confirms** the earlier purely-behavioral result — the
network's internal computation genuinely traces back to the swapped
content, not a coincidental output match.

**Status as of this writing**: `hmn_notags_weave_anchor_rope` is training
(stage1, the harder nc=4/wc=2 shape); `chunk_len=64`'s recall needs to
improve further (currently ~10% mid-stage1, well below a useful testing
threshold) before the probe (behavioral + mechanistic) can be re-run at
that length to close the one open question. `hmn_notags_weave_anchor`
(NoPE) is queued behind it.
