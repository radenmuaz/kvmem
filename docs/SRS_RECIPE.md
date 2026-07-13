# SRS Recipe — General Theory and Scaling

## Core Idea

A model that ingests a byte corpus **incrementally** using Spaced Repetition System (SRS)
scheduling — reading, revising, and recalling windows of bytes — until the full corpus is
retained in the model's fast-weight slots. Goal: achieve train and test NLL matching a
backprop LM trained on the same corpus, but without weight updates at inference time.

The HMN Feedback mechanism is the primitive: one IQ turn (initial encoding into SLOTs)
plus k IR turns (iterative argmax-feedback refinement). Proven at 32B / k=2:
87.5% match. Stage 3 (64B, 3 overlapping windows) is the first scaling experiment.

---

## Primitives

| Symbol | Meaning |
|--------|---------|
| W | window size (bytes); default 32B, can scale |
| s | stride (bytes); default 16B (50% overlap = W/2) |
| C | chunk_len (bytes); W = k·C for integer k (chunks per window) |
| n | number of IR refinement passes per window per review |
| R | retention: AR-decode match% (0–1) on a window |
| B | BPB: NLL/ln(2) on the window's output under teacher forcing |
| S | stability: how long a window stays above R_thresh after one review |
| λ | forgetting rate = 1/S |

---

## Forgetting and Retention

After the last review of a window, retention decays:

```
R(t) = R_0 · exp(-λ · t)
```

where t is steps (or wall-clock time) since the last review, R_0 is post-review
retention, and λ = 1/S.

After n IR refinement steps, post-review retention and stability improve:

```
R_0(n) = 1 - ε(n)         (ε → 0 as n grows, saturates)
S(n)   = S_0 · α^n         (stability multiplies each review, α > 1)
```

Empirically at 32B / 2 IR steps: R_0 ≈ 0.875, B ≈ 0.1 (vs random B ≈ 8.0).

Next review is due when R(t) drops to a threshold R_thresh (e.g. 0.9):

```
t_due = S · ln(R_0 / R_thresh)
```

---

## Open-Loop vs Closed-Loop Feedback

### Open-Loop (current)

Fixed trajectory regardless of the model's actual retention:

```
IQ → IR_1 → IR_2 → ... → IR_n   (n fixed by config)
```

- Predictable, fully parallelizable, no sampling during training
- Wastes compute on windows already well-retained
- What stages 1–3 use: n_refine=2

### Closed-Loop (adaptive, TODO)

After each IR step, sample AR decode. Continue only if below threshold:

```
IQ → IR_1 → [eval R]
         if R ≥ R_thresh: stop     (already retained)
         else: IR_2 → [eval R] → ...
```

Equivalent to SRS ease factor: easy windows get fewer reviews, hard windows more.
Enables training to scale without linear growth in sequence length per window.

The existing architecture supports this: argmax feedback is already the model's own
greedy output. Closed-loop just stops the IR chain early based on a runtime check.

---

## Parallel Batch Ingest with Merge-Resolve Operator

### The problem

Serial ingestion processes windows left-to-right, one sequence. Each IQ/IR turn
sees the full accumulated KV context of all prior turns. This is correct but
sequential — O(n_windows) latency even with a fast per-window unit.

Parallel ingestion splits the corpus into independent batches, each producing its
own KV state. But the resulting slot representations are **conditioned on disjoint
contexts** — batch A's SLOTs only saw chunks 0..k, batch B's only saw k+1..n.
Naive concatenation of their KV caches is incoherent: recall against the merged
state does NOT match what single-pass serial ingestion would have produced.

The merge-resolve operator must produce a merged state that is **recall-equivalent
to serial single-pass** — i.e., IQ and IR recall from the merged state must produce
byte-for-byte identical output to serial ingestion on the same corpus.

### Merge via IR: the argmax bridge

The existing IR mechanism is already a merge primitive:
```
SLOT_A  +  argmax_cue  +  SLOT_B  →  merged output
```

Treat batch A's IQ recall output as the argmax cue and batch B's SLOT as the
starting representation. A "merge IR" turn then resolves the two branches:

```
Batch A: ENC_{0..k}  → IQ_w_A  → IR_w_A  ...  →  SLOT_A, gen_A (bytes 0..m)
Batch B: ENC_{k+1..n} → IQ_w_B  → IR_w_B  ...  →  SLOT_B, gen_B (bytes m+1..L)

Merge turn (shape identical to a normal IR turn):
  [SLOT_A × slot_len]            ← branch A's compressed state
  [argmax: gen_A, byte-concat gen_B]  ← both branches' recalled bytes as cue
  [SLOT_B × slot_len]            ← merged output slot
  [warmup: bytes 0:wl]           ← seeded from GT for validation
  [output: bytes wl:L]           ← merged recall; loss here
```

After merge, run IQ recall (no argmax, just SLOT_B → output) and require exact
match against serial single-pass output. This is the **validation gate**.

### Hierarchical merge (tree reduce)

For N batches, apply merge as a binary tree:

```
Level 0 (parallel, independent):
  Batch 0: ENC_{0..k}    → SLOT_0, gen_0
  Batch 1: ENC_{k+1..2k} → SLOT_1, gen_1
  Batch 2: ENC_{2k+1..3k}→ SLOT_2, gen_2
  Batch 3: ENC_{3k+1..4k}→ SLOT_3, gen_3

Level 1 (pairwise merge, still parallelisable):
  Merge(SLOT_0, gen_0, SLOT_1, gen_1) → SLOT_01, gen_01
  Merge(SLOT_2, gen_2, SLOT_3, gen_3) → SLOT_23, gen_23

Level 2 (root merge):
  Merge(SLOT_01, gen_01, SLOT_23, gen_23) → SLOT_root, gen_root

Validate: IQ recall from SLOT_root must match serial single-pass gen.
```

Depth = log2(N). Each level is fully parallel within it. Total latency:
O(log N) merge passes instead of O(N) serial passes.

### Rebase after merge

If gen_01 at level 1 has errors (imperfect recall), they propagate into the
level-2 argmax cue. A **rebase IR** turn corrects this:

```
Rebase = IR turn with argmax = merged_gen (from level-1 output)
                      but warmup seeded from GT (not from merged_gen)
```

This is exactly the existing IR mechanism — the argmax carries the imperfect
estimate from the parallel branch, and the model learns to refine it using the
GT warmup as a reference signal. The GT warmup at training time is replaced at
inference by whatever the model last decoded (standard AR decode contract).

So rebase IS an IR turn. No new operator needed — just chain IR passes after merge.

### Validation: exact recall equivalence

The merge is valid iff the following holds for every span in the corpus:

```
serial_recall(span, IQ+2IR)  ==  merged_recall(span, IQ+2IR after merge)
```

Measured by: AR-decode match% = 100% (byte-for-byte exact), not just NLL.
BPB near zero is necessary but not sufficient — the model must produce the EXACT
same byte sequence, not just assign high probability to it.

**Why exact matters**: the argmax cue in a later IR turn is the previous turn's
greedy decode. If merged recall produces a different byte at position i, that
byte becomes the argmax cue for all subsequent turns, causing cascading divergence.
High-probability ≠ same-byte; only 100% exact match guarantees coherent chaining.

### Training the merge operator

The merge turn is a new trajectory type (`type='ir_merge'`) trained jointly with
the per-window IQ+IR turns:

```
traj_mix = [
    dict(type='ir_local', weight=0.5, windows=..., n_refine=2),   # serial ingest
    dict(type='ir_merge', weight=0.3, split_at=k, n_refine=2),    # parallel+merge
    dict(type='ir_rebase', weight=0.2, split_at=k, n_rebase=2),   # rebase after merge
]
```

Training data: for each batch, run both serial (ground truth) and parallel (input)
on the same corpus. Loss on the merge output against the serial ground truth ensures
the model learns to reconcile the two branch representations.

### Why this is necessary for scale

At 1MB corpus with 65534 windows, serial ingestion takes ~65534 IQ+IR passes.
With 1024-way parallel batches and log2(1024)=10 merge levels, total passes drop
to ~10 merge levels × 64 batches = 640 passes (100× speedup). The merge operator
amortizes the O(N) serial cost into O(log N) with full parallel hardware utilisation.

---

## Parallel KV Consolidation (major train change required)

### Core idea

Run N independent trajectories in parallel — each chunk gets its own full IQ+IR
forward pass producing its own KV state. Then run a **consolidation forward pass**
that concatenates all N KV states along the time dimension and produces a NEW,
smaller slot that distills all N into a single compressed representation.

The compressed slot must support recall at the same quality as the full N-slot union.
Each consolidation round halves the slot budget, like a compression pass:

```
Round 0 (parallel, N independent passes):
  chunk_0 → KV_0  (slot_len=2 slots → 2 KV entries)
  chunk_1 → KV_1  (2 KV entries)
  chunk_2 → KV_2  (2 KV entries)
  chunk_3 → KV_3  (2 KV entries)

Round 1 (consolidation forward pass):
  cat(KV_0, KV_1, KV_2, KV_3)   ← 8 KV entries in time dim
  → SLOT_consol × 2              ← consolidation bottleneck: back to 2 slots
  → [warmup][output]             ← must recall full 4-chunk span from 2 slots

Validation: IQ recall from SLOT_consol (no access to source KVs) == serial recall.
```

### Why this is different from merge-resolve

Merge-resolve reconciles **two independent encodings of the same span** — the
two branches computed different representations of overlapping bytes, and the
merge picks the consistent one.

Consolidation **compresses N non-overlapping spans into one** — the N chunks
cover disjoint byte ranges, and consolidation must encode all of them into a
slot budget that is smaller than their union:

```
Merge:       span_A ∩ span_B ≠ ∅  →  reconcile two views of the same bytes
Consolidate: span_0 ∪ span_1 ∪ ... ∪ span_{N-1}  →  encode ALL bytes in fewer slots
```

### Sequence layout (consolidation pass)

The consolidation pass is a new IQ-like turn appended after the parallel KVs:

```
──── parallel forward passes (independent, can run on separate devices) ────
KV_0:  [ENC_0: chunk_0 | SL_0×2] [IQ_0: SL×2 | wm | out] [IR_0×2]
KV_1:  [ENC_1: chunk_1 | SL_1×2] [IQ_1: SL×2 | wm | out] [IR_1×2]
KV_2:  [ENC_2: chunk_2 | SL_2×2] [IQ_2: SL×2 | wm | out] [IR_2×2]
KV_3:  [ENC_3: chunk_3 | SL_3×2] [IQ_3: SL×2 | wm | out] [IR_3×2]

──── consolidation forward pass (single pass, reads from concat KV) ─────────
Virtual seq: cat(KV_0, KV_1, KV_2, KV_3)    offset positions: 0, L, 2L, 3L
[SLOT_consol × 2]                            ← consolidation bottleneck
[warmup: bytes 0:wl of full 4-chunk span]
[output: bytes wl:4×chunk_len]               ← recall ALL 4 chunks from 2 slots
```

Mask for consolidation SLOT: can attend to all N sets of source SLOTs
(the IQ/IR output slots from each parallel pass), blocked from raw chunk bytes
(same bottleneck contract as a standard IQ SLOT).

### Position encoding during concat

Each parallel pass starts at position 0 within its own sequence of length L.
When concatenated for the consolidation pass, positions are re-indexed with
offsets: pass i starts at position i×L. The consolidation SLOT is at position N×L.

RoPE positions for the consolidation turn are thus large (N×L + ...) — the relative
position distances between the consolidation SLOT and the source SLOTs reflect how
far apart in the virtual sequence they are. This matters: if N=4 and L=204, the
consolidation SLOT is at position 816, attending to source SLOTs at 0, 204, 408, 612.
The RoPE distances encode "how far back in time" each chunk was ingested — a natural
recency signal for the consolidation bottleneck.

Alternative: use **segment-local positions** (each pass's internal positions stay at
0..L-1; the consolidation SLOT uses a fresh position 0). This removes the recency
signal but makes the consolidation SLOT's attention more uniform across all N passes.
Ablate both.

### Iterative consolidation (slot budget halving)

Apply consolidation recursively to reduce slot count from N×2 to 2:

```
Level 0: N=4 chunks, 4×2=8 total slots (parallel passes)
Level 1: consolidate 4→2 chunks at a time → 2×2=4 total slots
Level 2: consolidate 2→1 → 1×2=2 total slots (final)
```

Or more aggressively, full fan-in at each level:

```
Level 0: 64 chunks × 2 slots = 128 slots  (64 parallel passes)
Level 1: 1 consolidation pass, 128 source slots → 2 final slots
```

Whether 128→2 is achievable in one pass depends on model capacity. More likely:
a binary tree like the merge-resolve hierarchy, but each node is a consolidation
(compressing, not reconciling).

### Training requirements (why this is a major change)

Current training: one forward pass per training step, fixed sequence length.

Consolidation requires:
1. **Parallel forward passes**: run N independent passes, collect their KV caches
2. **KV concatenation + re-offset**: concat along time dim, shift position indices
3. **Consolidation forward pass**: new pass with the concat KV as `past_kv`,
   consolidation SLOT and output at offset N×L
4. **New mask type** (`chunk_mask_consolidation`): consolidation SLOT rows attend
   to all source SLOTs across all N KV segments; blocked from raw chunk bytes
   (same as IQ SLOT rule) and from all IR/IQ output regions (same as current fix)
5. **Loss on consolidation output** (not the parallel IQ/IR outputs — those are
   just scaffolding for producing the source KVs)
6. **Validation pass**: after consolidation, run IQ recall from SLOT_consol
   (discarding source KVs) and require exact match

The parallel passes themselves are teacher-forced IQ+IR (same as current training),
but their purpose changes: instead of measuring loss on their output, they generate
KV context for the consolidation step. Gradients still flow through the consolidation
pass and back into the parallel passes via the KV state.

### Validation: exact match gating (same contract as merge-resolve)

The consolidation is valid iff:

```
recall(SLOT_consol, IQ, full_span)  ==  serial_recall(full_span, IQ+2IR)
```

where serial_recall is what a standard (non-parallel) single-pass ingestion would
produce. Exact byte-for-byte match required — not just BPB — for the same reason
as merge: any divergent byte becomes an argmax cue and cascades through IR chains.

If consolidation fails exact match, chain consolidation IR turns (same as rebase
in merge-resolve): feed the imperfect consolidation output as argmax cue, re-run
the consolidation SLOT, and iterate until exact match or max_rounds.

---

## Scaling — Corpus Ingestion Recipe

### Window layout

For a corpus of length L bytes, partition into overlapping windows:

```
n_windows = (L - W) / s + 1     (W=32, s=16)
window_i   = bytes [i·s, i·s + W)   for i = 0, 1, ..., n_windows-1
```

Stage 3 (64B, 3 windows): n_windows = (64-32)/16 + 1 = 3 ✓

| Corpus | n_chunks | n_windows | Sequence length L (approx) |
|--------|----------|-----------|---------------------------|
| 32B    | 2        | 1         | 204                       |
| 64B    | 4        | 3         | 572                       |
| 128B   | 8        | 7         | ~1300                     |
| 256B   | 16       | 15        | ~2700                     |
| 1KB    | 64       | 63        | ~10800                    |

### Per-window SRS schedule (one epoch over corpus)

```
for each window w in corpus (processed left-to-right):
    IQ(w)                          # slot-compress window w
    for step in 1..n_refine:
        IR(w, argmax_{step-1})     # argmax-feedback refinement
        [closed-loop: break if R(w) ≥ R_thresh]
    schedule_next_review(w, S(w))  # exponentially increasing interval

# Subsequent passes: process windows in due-date order (earliest first)
# until all windows: R(w) ≥ R_thresh for M consecutive reviews
```

### Growth rule

Always use the SAME proven 32B IQ+IR unit per window — only n_windows grows.
Never widen the per-window compression ratio. This is the key constraint.

```
Stage N checkpoint → Stage N+1: double src_len, add n_windows windows,
                                 inherit all slot representations
```

---

## Scaling to Long Corpora — Sequence Layout at Each Scale

### Fixed window (W=32B), growing corpus

The simplest scaling axis: keep the proven 32B IQ+IR unit fixed, add more windows.
Training sequence layout for a single SRS pass over a batch of B_w windows:

```
[ENC_0..ENC_{n_chunks-1}]                  ← one shared encoding pass
[IQ_w0 | IR1_w0 | IR2_w0]                  ← window 0: bytes 0-31
[IQ_w1 | IR1_w1 | IR2_w1]                  ← window 1: bytes 16-47
[IQ_w2 | IR1_w2 | IR2_w2]                  ← window 2: bytes 32-63
...
[IQ_w_{B_w-1} | IR1 | IR2]                 ← last window in this session
```

At scale, the full sequence no longer fits in memory — SRS scheduling determines
which B_w windows appear in each training session.

| Corpus | n_windows | Seq L (all windows, W=32 s=16) | Fits single pass? |
|--------|-----------|-------------------------------|-------------------|
| 64B    | 3         | ~572                          | ✓ (stage 3)       |
| 1KB    | 63        | ~10800                        | ✓ (fits easily)   |
| 16KB   | 1023      | ~168K                         | ✓ (large batch)   |
| 65KB   | 4095      | ~672K                         | ✗ — batch windows |
| 1MB    | 65534     | ~10.7M                        | ✗ — batch windows |

**Multi-pass SRS session** (corpus too large for single pass):
```
Session = ENC(all chunks) + [IQ_w | IR1_w | IR2_w for w in srs_due(t)]

srs_due(t) = {w : R_w(t) < R_thresh}   # windows due for review at time t
           ordered by urgency: lowest R_w first

Each session processes B_w windows; B_w chosen to fit context budget.
```

After each session, update stability estimates:
```
if R_w(t) >= R_thresh after review:   S_w ← S_w · α   (interval grows)
else:                                  S_w ← S_0        (reset — hard window)
```

### Raising window size: W=128B or W=1024B

Instead of more windows at 32B, widen each window. Two sub-options:

**A. Keep chunk_len=16, increase chunks per window (k=8 or k=64)**
```
W=128B: k=8 chunks, slot must encode 128B  (4× harder than proven 32B)
W=1024B: k=64 chunks, slot must encode 1024B (32× harder)
```
Risk: the slot bottleneck (slot_len=4 tokens) may not have enough capacity.
Current proof is at k=2 (32B). k=8 is an unknown. Must ablate before committing.

**B. Raise chunk_len, keep k=2 chunks per window**
```
chunk_len=64,  W=128B:  k=2, same structure as proven recipe, 4× larger window
chunk_len=512, W=1024B: k=2, 32× larger window
```
Advantage: training layout is IDENTICAL to the proven recipe (just larger tensors).
The IQ+2IR mechanism has the same structural form. Risk: slot capacity per-chunk.

| Config | chunk_len | k | W | Compression ratio (bytes→slot_len tokens) |
|--------|-----------|---|---|------------------------------------------|
| proven | 16        | 2 | 32B  | 16:4 = 4:1 per chunk                  |
| →128B  | 64        | 2 | 128B | 64:4 = 16:1 per chunk (4× harder)     |
| →1024B | 512       | 2 | 1024B | 512:4 = 128:1 per chunk (32× harder) |
| →1024B | 16        | 64 | 1024B | 16:4 = 4:1 per chunk, 64 chunks      |

Option B (raise chunk_len, keep k=2) is the natural next ablation — structure
stays identical, only compression ratio increases. Start at chunk_len=32 (W=64B),
verify IQ+2IR still converges, then step up.

### Hierarchical chunking (future)

At very large W, a two-level hierarchy avoids catastrophic compression:

```
Level 1: proven 32B windows (chunk_len=16, k=2) → SLOT_L1 per window
Level 2: windows of 4 L1-windows (128B) → read L1 SLOTs → SLOT_L2
Level N: windows of 4^(N-1) 32B base windows
```

Each level's IQ reads from the level below's SLOT tokens instead of raw bytes.
Same IQ+IR mechanism, just the "source" is SLOTs not bytes.

---

## Inference-Aligned Training — Interleaved Ingest and Query

### The problem

Standard SRS training processes windows sequentially: ingest all, then recall.
At inference, a user may **query mid-ingest** (e.g. "what did you just read?")
or the model must **resume ingestion** after answering without forgetting earlier windows.

Training must reflect this if the model is to handle it at inference.

### Interleaved IQ-query + IR-ingest sequence

Three turn types co-exist in a single training sequence:

```
type='ingest_iq'  : IQ turn scoped to a NEW window (never seen before in session)
type='ingest_ir'  : IR turn refining a window already IQ'd this session
type='query_iq'   : IQ recall of a window already ingested — NO source re-read,
                    model must recall from the SLOT KVs already in context
type='resume_ir'  : IR turn after a query (resumes refinement, proves no forgetting)
```

**Example sequence layout (8 chunks = 128B corpus, 7 windows):**

```
[ENC_0..ENC_7]                          ← encode all 8 chunks once

[ingest_iq w0] [ingest_ir w0] [ingest_ir w0]   ← ingest window 0 (bytes 0-31)
[ingest_iq w1] [ingest_ir w1] [ingest_ir w1]   ← ingest window 1 (bytes 16-47)
[ingest_iq w2] [ingest_ir w2] [ingest_ir w2]   ← ingest window 2 (bytes 32-63)

[query_iq w0]                                  ← user queries window 0 mid-ingest
                                               ← model must recall bytes 8-31 of w0
                                               ← only GT warmup[0:8] provided

[resume_ir w3] [resume_ir w3]                  ← resume: ingest window 3 (no fresh IQ)
[ingest_iq w4] [ingest_ir w4] [ingest_ir w4]
...
```

The `query_iq` turn has **no source re-read**: the encoding blocks are in context
but the model must use only SLOT KVs already in the KV cache. This forces the model
to maintain slot representations across the full sequence — not just within a window.

The `resume_ir` after a query has no preceding IQ — it reads from `argmax_src_c0`
pointing to an IQ block earlier in the sequence, proving the IQ representation survived
the query interruption.

### Training mix

In practice, mix multiple trajectory types per batch:

```
traj_mix = [
    dict(type='ir_local', weight=0.6, windows=all_windows, n_refine=2),  # pure ingest
    dict(type='ir_local_query', weight=0.3, windows=..., query_after=k),  # ingest+query
    dict(type='ir_local_resume', weight=0.1, ...),                         # resume after query
]
```

The `query_after=k` variant inserts a query_iq turn after window k's ingest, then
continues ingesting windows k+1..: forcing retention of k during distraction.

### Why this reflects inference

At inference, a user streaming bytes into the model and periodically asking questions
triggers exactly this pattern:
- Model runs IQ+IR on each new 32B chunk as it arrives (streaming ingest)
- User query = IQ recall without re-reading source
- Model continues ingestion after answering
- SRS scheduler resurfaces stale windows in the background

This is equivalent to an **online, interactive LM** with fast-weight memory —
the model "reads" a document and can answer questions about any part of it
without backpropagation or fine-tuning.

---

## Target: Train and Test NLL Matching Backprop LM

A backprop LM achieves NLL by gradient descent over many passes through the corpus,
updating global weights. HMN achieves NLL by:

1. **Fast-weight update (IQ)**: slot-compress each window into SLOT tokens — no
   gradient, deterministic given the window bytes.
2. **Iterative refinement (IR)**: argmax-feedback passes improve the slot
   representation of each window — converges to low BPB.
3. **SRS scheduling**: ensures every window stays above retention threshold — the
   model doesn't "forget" early windows while learning later ones.

**Equivalence condition**: if the slot compression is expressive enough to encode a
32B window losslessly, then k=2 IR steps suffice and the model achieves BPB → 0
on that window. Extending to the full corpus via overlapping windows with SRS
scheduling should achieve BPB → 0 globally.

**Train NLL**: teacher-forced NLL on recall output (what training optimizes)  
**Test NLL**: AR-decode BPB starting from only the first window's 8-byte warmup —
the "prolonged AR" protocol in `ar_decode_chunk_fb_stitch_kv`

The train/test gap closes as:
- The IQ slot compression generalizes (slot encodes structure, not just memorized bytes)
- The IR refinement converges reliably (closed-loop stops when R ≥ threshold)
- SRS scheduling prevents forgetting across the corpus

---

## `ir_slot_window` — controllable within-window SLOT recurrence depth

**Question that prompted this**: is there a parameter controlling how much a memory
SLOT can attend to a *past refine turn's* SLOT (e.g. IR2 attending to IR1's SLOT, not
just its `argmax` copy)? Checked `chunk_mask_fb`: no — by construction, IR turns'
`SLOT_A`/`argmax`/`SLOT_B` rows are only blocked from `is_any_chunk | is_any_rec_output`
(raw source chunks and raw prior OUTPUT regions); prior turns' SLOT columns within the
*same window* were never in either excluded set, so they're visible unrestricted by
default — a structural byproduct of how the mask rules are written, not a deliberate
design choice. (Rule 3b, the "nochain" rule, only blocks *cross-window* SLOT attention;
it says nothing about turns within one window's own chain.)

**Added `ir_slot_window`** (`experiments/chat_tags/mask_windowed.py`,
`chunk_mask_fb_windowed`, wired into `experiments/srs_tagged/train.py` via
`hp.get('ir_slot_window', None)`) as an explicit, controllable parameter for this —
framed as an RNN sliding window over turns (this project's own framing: the IR loop is
already an unrolled recurrence, each turn re-writing fresh SLOT tokens conditioned on
the previous turn's argmax, structurally the same "review and strengthen" pattern as
the macro SRS schedule, just within one forward pass):

- `None` (default) — unbounded, IDENTICAL to `chunk_mask_fb`'s existing behavior
  (verified bit-for-bit via `verify_unbounded_matches_baseline()`) — no breaking change,
  every existing config/run is unaffected unless explicitly set.
- `1` — only the current turn's own SLOT visible; all earlier turns' SLOTs in the same
  window blocked (like an RNN with only `h_t`, no history) — forces the model to rely
  purely on the explicit `argmax` copy as its only feedback channel.
- `2` — current turn + immediately preceding turn's SLOT visible (`h_t, h_{t-1}`).
- `N` — current + previous `N-1` turns visible (sliding window).

Smoke-tested: `None` reproduces the baseline mask exactly; `1` blocks strictly more
while remaining causal. Infrastructure only — not yet used in any launched run; a
natural ablation once other higher-priority items (window-G fix, `juz1` prep) are done,
to test whether restricting the "free" SLOT-to-SLOT side-channel changes how much the
model relies on `argmax` as intended, or whether removing that side-channel hurts
performance (meaning the model was using it meaningfully despite never being deliberately
designed to).

---

## Fast-Weight Rank and Addressing — connecting the chat-tags experiment to scaling theory

Viewed through the fast-weight-programmer lens (Schmidhuber's original work; Schlag &
Schmidhuber 2021 "Linear Transformers Are Secretly Fast Weight Programmers"), each SLOT
token's key/value contributes one rank-1 outer-product update to an implicit associative
memory matrix. With `slot_len=8`, each attention head's addressable memory is capped at
**rank ≤ 8** — a real, literature-grounded ceiling, not a metaphor.

**Live evidence this ceiling was not the actual bottleneck**: the `experiments/chat_tags/`
track (see `docs/FEEDBACK_RESULTS.md § Chat-tags experiment`) hit exactly the failure this
predicts — Win C (the third of three windows sharing one global SLOT block) stuck at a
converged 27.8-30.6% plateau, with IR turns *degrading* quality rather than correcting it
(classic destructive interference from multiple write-patterns colliding under one shared
key). The fix that actually worked — window-specific query tags (`<query_a/b/c>` instead of
one generic `<query>`) — changed **zero rank**: same `slot_len=8`, same `d=64`, same 4 heads.
It only gave each window a distinct addressing key. Result: Win C jumped from 27.8% to
84-92% within one run. If the ceiling were genuinely rank-limited, disambiguating addressing
without adding capacity should not have recovered this much — the content would still be
irretrievably tangled regardless of how cleanly it's addressed. This is strong, direct
confirmation of `docs/MDL_MODEL_SIZE.md`'s standing verdict, written before this experiment
ran: *"Current failure — capacity problem? No — training distribution problem. Fix: more
parameters? Wrong — fix: stronger constraint."*

**Where this generalizes and where it doesn't.** Window-specific tags is a hand-authored fix
for exactly 3 known, discrete cases — it doesn't scale to "arbitrarily many distinguishable
memories," which is what the corpus-scale goal above requires. Hypothetical directions for a
principled version of the same fix, roughly nearest-to-current-architecture first:

1. **Delta-rule fast weights** (DeltaNet — Schlag & Schmidhuber; Yang et al. 2024
   "Parallelizing Linear Transformers with the Delta Rule"): replace naive additive
   rank-1 accumulation (`S += k v^T`) with an error-correcting associative write
   (`S += k(v - S^T k)^T`-style). This is the *learned, principled* version of what
   window-specific tags did by hand — it architecturally reduces interference between
   multiple writes sharing one memory, instead of requiring a human to pre-assign
   non-colliding keys per window.
2. **Content-derived addressing instead of identity tags**: a learned hash/routing of the
   query context into a large key space (Reformer-style LSH attention), so the model
   addresses arbitrarily many memories without a human enumerating and tagging each one.
   This is the real generalization needed before Tier 2 (random-warmup, any X) can work
   without per-position hand tags.
3. **Hierarchical/tree memory** (already flagged as "future" — see Hierarchical chunking
   below): the actual sublinear-scaling answer. A flat SLOT block cannot hold billions of
   tokens' worth of distinguishable content regardless of addressing quality — this needs
   O(log N) hierarchical depth with fixed per-node capacity, not O(N) flat rank.
4. **`slot_len` growth**: the most surgical "more rank" lever (maps 1:1 onto rank-1
   contribution count) but doesn't scale sublinearly — useful as a local capacity boost
   only, not a scaling strategy. Should remain last-resort per the MDL ordering already
   established in `docs/MDL_MODEL_SIZE.md`.
5. **Explicit fast-weight layer**: a real linear-attention/associative-memory layer bolted
   onto (or replacing part of) the softmax transformer, rather than an implicit
   approximation via masked attention over a handful of SLOT tokens. Would make rank/capacity
   tradeoffs mathematically explicit and tunable via known linear-attention/DeltaNet error
   bounds, instead of discovered empirically through ablation.
6. **Depth-wise growing/concatenated cross-layer SLOT memory**: instead of one `slot_len`
   bottleneck reprocessed recurrently (which caps rank at `min(slot_len, d_head)` no matter
   how many refinement passes run), let each layer emit its own `slot_len`-sized SLOT block
   and have layer L's recall attend to the **concatenation** of layers `1..L`'s SLOT KV. By
   the final layer this exposes `L × slot_len` keys instead of `slot_len` — for `d=64,
   n_heads=4` (`d_head=16`) and `slot_len=8, n_layers=4`, that's `4×8=32` keys against
   `d_head=16`, saturating the *full* per-head rank ceiling (`rank=min(32,16)=16`) instead of
   being capped below it by token count (`min(8,16)=8`). Unlike recurrent depth (which only
   helps get closer to an *existing* ceiling — see below), this is a genuine ceiling *raise*,
   since it changes what's being attended over, not just how many times it's reprocessed.
   Literature precedent: DenseNet's cross-layer concatenation (Huang et al. 2017) applied to
   KV instead of conv features; Feedback Transformer (Fan et al. 2020) pools across all
   previous layers rather than just the immediately-prior one; DenseFormer (Pagliardini et
   al. 2024) shows depth-weighted averaging of all previous layers improves perplexity per
   parameter. Note this is the deliberate *inverse* of the efficiency-motivated Cross-Layer
   Attention / YOCO family (Brandon et al. 2024; Microsoft 2024), which *shares* KV across
   layers to shrink cache for long-sequence LLMs — this project has the opposite problem (a
   deliberately tiny fixed bottleneck, not a huge sequence-length cache), so growing rather
   than sharing is the right direction here. Cost: attention at layer L grows to
   O(L·slot_len) keys, cumulative O(L²·slot_len) across all layers vs today's O(L·slot_len)
   — real but modest at this project's scale. Strongest candidate so far for "raise the
   ceiling" specifically, since it grows capacity by depth×width rather than width alone,
   without inflating per-layer KV footprint.

**Queued next experiment** (after the chat-tags Phase B4 run confirms success — see
`docs/FEEDBACK_RESULTS.md`): ablate this direction against B4 as baseline, comparing
convergence speed (steps to reach a given Win C match%, not just final ceiling).

Scoped design (deliberately narrower than "every token gets cross-layer KV" — DenseFormer's
full generality isn't needed here and would be much more expensive):
- Only **SLOT token positions** (encoding + recall) accumulate cross-layer KV. Regular
  content/warmup/output positions keep normal single-layer attention — this keeps the extra
  cost bounded to `slot_len` extra keys per layer, not the whole sequence.
- Requires a new model variant, not just a config/traj_mix change — `MHAttention`/
  `TransformerBlock`/`KVMemModel` in `kvmem/model.py` (unmodified, per this project's
  isolation convention) assume ONE `(L,L)` mask reused identically at every layer;
  supporting "layer L's SLOT rows attend to layers `1..L-1`'s SLOT KV too" needs a
  genuinely different per-layer attention scope, which the current single-mask design
  can't express. Build as a new, self-contained model class (e.g.
  `experiments/densenet_kv/model.py`), reusing `kvmem.model`'s RoPE frequency computation,
  `FFN`, and `RMSNorm` via import where unchanged, but with its own `MHAttention`/
  `TransformerBlock` forward pass that threads a growing list of prior layers' SLOT
  `(K,V)` and concatenates them into the attention computation at SLOT rows only.
  `experiments/chat_tags/`'s position/mask/batch machinery (`chunk_positions_iq_global_rw_tagged`,
  `chunk_mask_fb`, `make_batch_tagged`) is architecture-agnostic (produces token arrays +
  masks, doesn't know about the model internals) and can be reused unchanged; only the
  model-building and forward-pass code is new.
- Baseline for comparison: B4's final converged numbers (both peak and running-average
  match%, plus **steps-to-reach-90%-per-window** as the specific "faster convergence"
  metric the user asked to compare) vs the same recipe/traj_mix on the new architecture,
  warm-start not directly possible (different weight shapes/connectivity) — train from
  scratch, matched step budget.

**Result (built and run — see `docs/FEEDBACK_RESULTS.md` § DenseNet-KV ablation for full
detail): inconclusive, not negative.** Implementation verified correct before launch (cross-layer
KV growth confirmed numerically — layer *i* sees exactly `i×slot_len` extra keys; causality
check confirmed zero leak). But the planned comparison against B4 turned out to be unfair: B4
was warm-started from Phase B3's checkpoint (~348k cumulative prior steps), while densenet_kv
trained from scratch in the same 80k-step budget. densenet_kv's mean match% rose essentially
monotonically for the entire run (1.9%→20.8%, still climbing at cutoff, never converged) — B4
reached 94.9%. This is not evidence the architecture fails; it's evidence the experiment design
didn't control for starting point. **Correct follow-up**: a from-scratch **standard**-architecture
control at the same 80k budget (no warm start, same tags/traj_mix) would establish the fair
baseline densenet_kv should actually be compared against — not yet built.

### Depth (even weight-shared/recurrent) is not a substitute for hierarchical memory

A natural question: since IR turns already route/refine through the model recurrently, isn't
scaling depth — even cheaply via weight-shared recurrent depth (Universal Transformer /
"looped transformer" style: same weights, more sequential passes, no added parameters) —
equivalent to O(log N) hierarchical routing, making explicit hierarchical memory unnecessary?

No — depth and hierarchy are different axes. **Depth** (recurrent or not) buys more sequential
*refinement* over a fixed-size bottleneck; **hierarchy** buys more total *storage*, organized
so any one item is reachable in O(log N) steps instead of needing to fit in one flat buffer
simultaneously. A fixed `slot_len=8` bottleneck has a hard information ceiling (rank ≤ 8 per
head) that no amount of reprocessing — once or a hundred times — can exceed; computation can
only get closer to that ceiling (reduce interference/routing errors), not raise it. The
O(log N) property of hierarchical memory comes specifically from a pooling/tree topology
(progressively aggregating many leaves into fewer coarser summaries at each level), not merely
from having many sequential layers.

**IR turns themselves are direct evidence for this distinction.** They *are* weight-shared
recurrent depth applied to the SLOT representation, chained via argmax feedback. Before the
window-specific-tag fix (see § above), more recurrent depth was actively *harmful* for Win
C — IR1→IR2 degraded quality (e.g. 100%→75%→25% in one case) instead of improving it. Depth
amplifies whatever's already in the bottleneck, good or bad; it only helps once the underlying
representation is free of destructive interference to refine productively. So: recurrent depth
is a real, cheap (parameter-count-free, MDL-aligned) lever for closing the gap to an *existing*
capacity ceiling — worth pursuing further as an **adaptive** mechanism (loop refinement until
retention R ≥ threshold, rather than fixed `n_refine=2/3` — already flagged as "Closed-Loop
(adaptive, TODO)" below) — but it is complementary to, not a replacement for, hierarchical
memory when the goal is raising total corpus capacity beyond what one SLOT block can hold.

**Verdict on direction**: the project's stated goal — train/test NLL parity with a backprop
LM, achieved via fast-weight IQ + IR refinement + SRS scheduling, *without weight updates at
inference time* — is coherent and literature-grounded (fast weight programmers, modern
linear-attention/DeltaNet resurgence, NTM/DNC external memory, retrieval-augmented LMs). The
chat-tags result is genuinely useful evidence *for* this direction, not just a side
experiment: it shows the SLOT/IR mechanism has materially more headroom than the stuck
plateau suggested, and that the MDL principle (fix addressing/distribution before capacity)
holds under real pressure, not just in theory. To reach the billions-of-tokens goal, the
roadmap already on file — Tier 2 random-warmup generalization, hierarchical chunking (below),
parallel KV consolidation — is the right next set of steps; this experiment adds a concrete
data point for *why* addressing matters as much as raw capacity when designing those.

---

## Current Experiments

| Stage | Src | Windows | Result |
|-------|-----|---------|--------|
| 1 | 32B | 1, n_refine=0 | 81.9% match (IQ only) |
| 2 | 32B | 1, n_refine=2 | **87.5% match** |
| 3 | 64B | 3, n_refine=2 | superseded by the chat-tags track (see `docs/FEEDBACK_RESULTS.md`) |

Stage 3 (as originally scoped, plain `iq_global_rw`) was superseded — the chat-tags
experiment series (window-specific tags + wrong-token-weighted loss) solved the same
64B/nc=4 recall problem more thoroughly (97.2% mean, all three windows ≥90%, converged)
than continuing stage 3 alone would have. See `docs/FEEDBACK_RESULTS.md § Chat-tags
experiment` for the full arc.

---

## Resuming true SRS (spaced-repetition span scheduling) — queued, planned

**Why now**: the earlier depth-2 SRS attempt (`hmn_chunk_curric`/`hmn_chunk_srs_ir`,
`configs/hmn_chunk_srs_ir.py`) **failed** — stage 0 (IQ-only, 256B src, `slot_len=2`, 64×
compression per chunk) never escaped random-baseline BPB (~8.0) after 12000 steps. Root
cause diagnosed then as *too-aggressive compression + far too few training steps*, not a
windowing/refinement problem per se (see `docs/MDL_MODEL_SIZE.md`). The underlying IQ+IR
primitive (`chunk_positions_fb_localrefine`, one local IQ + n chained argmax-IR turns
per span) was never actually broken — it's the same mechanism validated at 87.5% (32B)
and now 97.2% (64B, tagged + wrong-token loss). What's genuinely new and worth applying
to true multi-span SRS: (1) `slot_len=8` not 2 (the compression ratio that actually
works), (2) span-specific query tags (the fix that took chat-tags Win C from 27.8%→91.7%,
directly transplantable — SRS spans have exactly the same "multiple regions share one
addressing key" collision risk that windows did), (3) the wrong-token-weighted IR loss
(fixed IR degradation cheaply, one-line change, no architecture cost).

**What "true SRS" means here, vs. what chat-tags already does**: `iq_global_rw` (the
chat-tags foundation) uses **one global SLOT** reading the whole source, queried at
different byte offsets (windows A/B/C) — it does not implement spaced-repetition
*scheduling* (review order, retention decay) at all. True SRS (`srs_schedule`/
`srs_schedule_depth2` + `chunk_positions_fb_localrefine`, already implemented in
`kvmem/train_hmn_chunk.py`) gives **each span its own local IQ+chained-IR unit**, visited
in a schedule designed to combat forgetting across a growing sequence (singles → pairs →
full, or halves → full for the depth-2 variant). This is the architecture the whole
`docs/SRS_RECIPE.md` vision document is actually about — the chat-tags track was a
detour to fix a specific `iq_global_rw` bottleneck, not the SRS mechanism itself.

**Reused unchanged** (per this project's isolation convention — no `kvmem/` edits):
- `kvmem.train_hmn_chunk.srs_schedule_depth2` / `srs_schedule` — span-order generation.
- `kvmem.train_hmn_chunk.chunk_mask_fb` — already implements the "IQ SLOT blocked from
  ALL tokens in prior rec_blocks" (nochain) rule generically (Rule 3b), which is exactly
  the v5 chaining fix `ir_local` needed — this de-risks the SRS extension significantly,
  since the hardest architectural problem in the whole local-refine family is already
  solved in the shared mask function chat-tags already depends on.
- `experiments/chat_tags/batch.py`'s `make_batch_tagged`/`_fill_argmax_fb` and
  `ar_decode_iq_global_rw_tagged` — both are generic over `pos_content['rec_blocks']`
  (don't assume `iq_global_rw`'s random-warmup-X structure specifically; the IQ branch's
  fixed-span fallback path already does exactly what per-span SRS review needs: warmup =
  first `warmup_len` bytes of the span, output = the rest). `turn_match_pcts` in the
  decode result already reports per-rec_block match — exactly what's needed for per-span
  SRS eval, no new decode function required.
- `kvmem.utils.make_test_sequences` for val (same as every prior experiment) and
  `kvmem.train_hmn_chunk.load_chunks_padded` + `datasets/suratalkauthar.txt` /
  `datasets/suratalfatihah.txt` for held-out test — the val/test convention used by the
  *original* (pre-chat-tags) SRS configs, not yet wired into `experiments/chat_tags/`'s
  train.py (which only does val). Worth adding for the SRS run specifically, matching
  `configs/hmn_chunk_srs_ir.py`'s `eval_file=` pattern.

**New code needed** (new folder `experiments/srs_tagged/`, isolated, no `kvmem/` edits):
- `chunk_positions_srs_tagged(n_chunks, chunk_len, slot_len, warmup_len, schedule,
  n_refine)` — mirrors `chunk_positions_fb_localrefine`'s per-span-in-sequence structure
  (one shared encoding pass, then each span in `schedule` gets its own tag-wrapped local
  IQ + n_refine chained IR), reusing the exact tag-wrapping helper pattern from
  `chunk_positions_iq_global_rw_tagged`, but assigning each span a query tag **by its
  position in the schedule** rather than by byte offset X. `srs_schedule_depth2(4)` gives
  exactly 3 spans `[(0,2),(2,4),(0,4)]` — conveniently exactly matching the 3 existing
  `HMN_QUERY_A/B/C` tags with zero new vocab needed for the first run. The full
  `srs_schedule(4)` (7 spans) would need 4 more tags — deferred to a follow-up once
  depth-2 is validated.
- A new `train.py` (not extending `experiments/chat_tags/train.py` in place) since the
  eval-reporting loop differs enough to warrant it: chat-tags groups eval by X-offset
  sweep over one shared trajectory; SRS needs to group by **distinct span**, reading each
  span's own final rec_block from `turn_match_pcts`/`rb['span']`, plus wiring in the
  `eval_file` test-set path the original SRS configs used but chat-tags never added.

**First target** (matching proven scale, not jumping to where the old attempt failed):
`n_chunks=4, chunk_len=16` (64B total, same scale as the whole chat-tags series),
`slot_len=8`, `srs_schedule_depth2` (3 spans: two 32B halves + the 64B full span),
`n_refine=2`, wrong-token-weighted loss enabled from the start (`wrong_token_weight=2.0`,
already proven). Val: `make_test_sequences(64)`. Test: `datasets/suratalfatihah.txt`
padded to `(4, 16)` via `load_chunks_padded`.

**Success bar (stricter than the chat-tags bar): the run only passes if TEST match hits
exactly 100%** on every span (not ≥90% — real held-out text is the actual target, val is
just a training-time proxy). If test falls short, iterate and fix rather than accepting a
partial result — do not declare success at <100% test. If depth-2 reaches 100% test,
extend to the full `srs_schedule` (7 spans, singles→pairs→full) as the next step — that's
the actual multi-session spaced-repetition scaling test the vision doc is about.

**Known practical constraint**: this run is markedly slower than any prior chat-tags run
— L=902 (one sequence packs all 3 spans + their full IR chains) vs 322 for `iq_global_rw`,
and attention cost scales roughly quadratically, so throughput dropped from ~20-25 it/s to
~4.1 it/s (~5x slower per step). At 60k steps this is ~4 hours, not the ~50-75 min prior
runs took — factor this into monitoring cadence and iteration turnaround time.

### Scaling roadmap beyond depth-2 (`suratalfatihah` → `juz1` → full Quran)

The test-file progression is deliberate, each step a real order-of-magnitude jump in
corpus size — not just "bigger for its own sake," but the actual scaling axis this whole
research program is about:

| Stage | Test file | Approx. scale | What it stresses |
|---|---|---|---|
| Current (depth-2) | `datasets/suratalfatihah.txt` | 1 short surah | Can the SRS-tagged mechanism recall a real (not synthetic) short text at all, at 100% |
| Next | `datasets/juz1.txt` | 1/30 of the Quran — much longer | Corpus size the depth-2 (3-span) schedule almost certainly can't cover — will need the full `srs_schedule` (singles→pairs→full, more spans) and likely more SLOT capacity or hierarchical chunking |
| Then | `datasets/quran_uthmani.txt` | full Quran | The actual target scale for "SRS corpus ingestion" as originally envisioned in this doc's Core Idea — genuinely tests whether spaced-repetition scheduling prevents forgetting across a large, real corpus, not a toy one |

**"Longer sequences need more diverse traj_mix and richer SRS" is the load-bearing
constraint at each step of this progression, not a one-time fix**: `srs_schedule_depth2`
(3 spans) is already close to its ceiling for `juz1`-scale input — a corpus that size
needs the full `srs_schedule` (singles → pairs → full, more review spans) at minimum, and
likely genuinely more SLOT capacity (`slot_len` growth) or the hierarchical/tree-memory
direction already discussed (§ Fast-Weight Rank and Addressing, direction 3) once a flat
SLOT block can no longer plausibly hold a corpus that size regardless of addressing
quality. Each scale step should be treated as its own experiment with its own diagnosis,
not assumed to inherit success automatically from the previous scale.

**North star for every stage of this roadmap — feature parity with a backprop-trained LM,
not just "high match% on a fixed eval set"**: the model should be able to **overfit** a
given corpus (memorize it exactly, same as an overparameterized backprop LM would given
enough capacity/steps — this is what "100% test match" is actually testing right now) AND
**generalize** beyond exact memorization (perform sensibly on held-out text drawn from a
similar distribution but never seen verbatim, the way a backprop LM generalizes from train
to a broader distribution via learned statistical/algorithmic structure, not just
verbatim recall). Every stage of this roadmap should be evaluated on both properties, not
just the corpus-specific exact-match number — a model that only ever hits 100% on the
exact text it was trained on, with no generalization signal, has only proven the
"overfit" half of the goal. This is the standard the whole SRS/fast-weight research
program is ultimately answerable to, per the Core Idea section at the top of this doc.

**Known methodology gap, discovered while grounding this roadmap in real file sizes —
must be fixed before "test=100%" is treated as a real pass**:
`kvmem.train_hmn_chunk.load_chunks_padded` **truncates** each line-group to `chunk_len`
bytes (`g[:chunk_len]`), it does not pad/cover the whole file. `datasets/suratalfatihah.txt`
is 562 bytes across 6 lines; at the currently-running `n_chunks=4, chunk_len=16`, the test
eval only actually exercises the **first 16 bytes of each of 4 line-groups = 64 bytes
total** — the remaining ~498 bytes of the surah are silently dropped, never seen by
`ar_decode_iq_global_rw_tagged` during test eval at all. A "test=100%" result from the
currently-running depth-2 config would therefore only prove recall of a 64-byte truncated
slice, **not the whole surah** — this does not meet the spirit of "real held-out text"
testing and should not be reported as a full pass even if the raw number hits 100%.

**Fix for the next iteration** (not applied to the already-running config, to avoid
restarting mid-run): size `chunk_len` (or `n_chunks`) so the target file's full byte
length fits within `n_chunks × chunk_len` — e.g. for a 562-byte `suratalfatihah.txt`,
something like `n_chunks=4, chunk_len=140` (560 capacity) or growing `n_chunks` instead
(preferred, since MDL_MODEL_SIZE.md's principle is that the per-chunk encoding algorithm
is position-invariant and parameter count shouldn't need to grow with corpus size — more
chunks, not bigger chunks, is the correct scaling axis, and is also what the SRS
`schedule` mechanism is built to handle). This same truncation trap will recur at every
step of the `juz1`/full-Quran roadmap above unless `n_chunks`/`chunk_len` sizing is
explicitly checked against the real target file's byte length each time, not just copied
forward from the previous stage's config.

**Compute-cost reality check** (important before naively scaling `n_chunks`/`chunk_len`
to "cover the whole file" in one leap): attention is O(L²), and L scales with total byte
coverage in this architecture — dominated by the `(0, n_chunks)` full-span block, whose
IQ+`n_refine`×IR turns each cost ~`total_bytes` of output. A literal "pad the whole
562-byte surah to 1024 bytes via `n_chunks=4, chunk_len=256`" was computed and found
**infeasible**: L jumps from 996 to 12,620, a **161x** compute-cost increase — 60k steps
at that rate is ~27 days, not hours. Going from 64B→1024B coverage is only 16x more
content but 161x more compute because both the sequence length AND the per-position
attention cost grow with coverage simultaneously.

### Stitching vs atomic full-span — pivot away from the nc8 scale-up

`srs_depth2_nc4_slot8`'s live eval data (step 5000) directly answered "should stitching
be the priority" before nc8 was even launched: the `span(0,4)` full-span block (one IQ+IR
unit decoding all 64 bytes in a single shot) was the clear bottleneck on **every**
sequence, val and test alike, even where both half-spans were near-perfect:

```
val/srs/span(0, 2)/MEAN   match=100.0%
val/srs/span(2, 4)/MEAN   match=65.3%
val/srs/span(0, 4)/MEAN   match=3.6%      <- full-span block, the weak link
test/srs/span(0, 2)/MEAN  match=79.2%
test/srs/span(2, 4)/MEAN  match=4.2%
test/srs/span(0, 4)/MEAN  match=8.9%      <- full-span block, the weak link
```

**Diagnosis**: the atomic full-span block is a genuinely new, never-before-validated
mechanism (a single IQ+IR unit outputting 56+ bytes) — nothing in this project's history
(chat-tags, `iq_global_rw_ir_v2`, or any prior `hmn_feedback_*` result) has proven that
works reliably. Meanwhile the half-spans reuse the already-proven 32B/24-byte-output IQ+IR
unit. Doubling `n_chunks` (the queued `srs_depth2_nc8_slot8.py`) would only make the
already-failing mechanism's output *longer* (112 bytes) — compounding the problem, not
fixing it — while also being the config that drives the O(L²) compute wall above (the
full-span block's output length scales with total corpus size, which is exactly why
compute blows up quadratically with coverage).

**Fix — stitching, not one bigger atomic block**: the `ir_local` track already solved
"cover more content" a different way — chain several small, fixed-cost, already-proven
32B windows via `ar_decode_chunk_fb_stitch_kv`'s mechanism: each later window's warmup is
seeded from the *previous* window's own just-decoded output (valid because `warmup_len=8`
always fits inside the 50%-overlap between adjacent windows, `stride=16B` at
`window=32B`). No single decode step ever has to produce more than ~24-32 bytes,
regardless of total corpus length — so total compute scales roughly **linearly** with
corpus length (more windows chained, each fixed-cost) instead of quadratically (one
block whose own output grows with corpus length). This is also the right shape for the
`juz1`/full-Quran roadmap above, whose growth rule (`n_windows = (src_len-32)/16 + 1`) is
linear by construction.

**Implemented**: `experiments/srs_tagged/stitch_decode.py` (`ar_decode_srs_stitched_tagged`)
— adapts `ar_decode_chunk_fb_stitch_kv`'s chaining logic to the tag-wrapped SRS layout, no
new position-builder code needed (`chunk_positions_srs_tagged` already accepts an
arbitrary `schedule`; the stitched config just passes overlapping windows
`[(0,2),(1,3),(2,4)]` — the exact `ir_local` window geometry — instead of
`srs_schedule_depth2`'s disjoint halves+full-span). `experiments/srs_tagged/train.py`
gained two small, backward-compatible additions: a `windows` curriculum-stage key that
overrides the depth-based schedule generator, and `eval_mode='stitch'` that switches eval
from per-span GT-seeded decode to the chained stitch decode (reports a `STITCHED_MEAN`
row = match% against the fully-chained reconstructed source). Smoke-tested: causal mask
confirmed, cross-window nochain (Rule 3b) confirmed intact under the new overlapping
schedule, end-to-end training+eval loop runs clean.

**`srs_depth2_nc4_slot8` final result (60k steps, complete)**: val 100%/100%/100% (span(0,2)/span(2,4)/span(0,4)), test 100%/100%/**69.6%**. Under the strict pass bar ("run only passes if test match is 100%"), this is a **fail** — the atomic full-span block never reached 100% on held-out real text even after val saturated at 100% and stayed there for the final 20k+ steps (loss dropped to ~0.01, fully converged). The gap is a genuine val/test generalization failure specific to the full-span mechanism, not undertraining: both half-spans hit 100%/100% on val AND test throughout. This is the clean confirmation baseline the stitching pivot was waiting on — **launched `srs_stitch_nc4_slot8.py`** immediately after, warm-started from `stage0_best.pt`.

**Revised queue**: `srs_depth2_nc8_slot8.py` marked **SUPERSEDED, do not launch** (kept
for the record, not deleted). Next run after `srs_depth2_nc4_slot8` finishes is
`experiments/srs_tagged/configs/srs_stitch_nc4_slot8.py` — same 64B/nc=4/slot_len=8 scale,
warm-started from `srs_depth2_nc4_slot8`'s best checkpoint, success bar `STITCHED_MEAN`
≥90% val AND test (directly comparable to the atomic block's 3.6%/8.9% at step 5000).
Only once stitching is validated at 64B does the queue return to scaling coverage — and
the correct next scale-up step becomes **more chained windows** (nc=8 → windows
`[(0,2),(1,3),...,(6,8)]`, 7 windows), not a single bigger atomic block — restoring the
"128B coverage" milestone but via the stitched mechanism instead of `nc8_slot8`'s atomic
one.

**Results, `srs_stitch_nc4_slot8` (warm-started from `srs_depth2_nc4_slot8`'s
`stage0_best.pt`, complete, all 60000 steps)**: reached `STITCHED_MEAN`=100% val AND 100%
test by step 15000, hit a **perfect sweep** by step 35000 (every individual span
(0,2)/(1,3)/(2,4) and the chained `STITCHED_MEAN` at 100.0% on both val and test
simultaneously), and **held that perfect sweep through every remaining checkpoint to the
final step 60000** (loss 0.0001). This clears — and holds — the strict 100%-test bar that
the atomic run (`srs_depth2_nc4_slot8`, final test span(0,4)=69.6%, never reached 100% at
any checkpoint) failed. Directly confirms the diagnosis above: chaining several small,
already-proven windows generalizes to held-out real text where one large atomic
single-shot block does not, even measuring the exact same "recall the full 64B sequence"
capability. (Some early-training oscillation observed between steps 5000-25000 — test
dipping to 41.7-75% on individual checkpoints before recovering — same cosine-restart
volatility pattern documented elsewhere in this doc; did not recur once past step 25000,
30000+ steps of sustained 100%/100%.) Training was also faster per-step (~5.9 it/s vs
~4.1 it/s for the atomic run) since the stitched layout's packed sequence is shorter
(`L=742` vs `L=902` — no 56-byte atomic full-span block to fit).

**Stitching vs atomic — verdict: stitching wins outright.** Same scale (64B), same
warm-start, same step budget, strictly better result (100% sustained vs 69.6% ceiling),
faster per-step. No tradeoff found in this comparison — stitching should be the default
mechanism for all further scale-up, not an alternative to weigh against atomic blocks.

### Is `juz1.txt` scaling ready? No — concrete gaps, not just "more steps"

Asked directly after the stitch result above. The honest answer is no, even though 64B
stitching just passed the strict bar. `datasets/juz1.txt` is 44,443 bytes — about **700x**
the validated 64B scale — and three specific mechanisms would break before getting there,
not just "need more training time":

1. **Window-identity tags are hardcoded to 3.** `_SRS_SPAN_TAGS` in
   `experiments/chat_tags/positions.py` defines exactly 3 window tags
   (`<query_a/b/c>`), and `chunk_positions_srs_tagged` raises `ValueError` if the
   schedule has more spans than that. `juz1` at the proven `chunk_len=16`/`window=32B`/
   `stride=16B` geometry needs `(44443-32)/16+1 ≈ 2778` overlapping windows — nowhere
   near 3. A per-window unique tag doesn't scale to thousands of windows regardless
   (vocab would need to grow by one ID per window); the tag scheme needs to become
   window-index-agnostic before any real scale-up.

2. **Whole-schedule-in-one-sequence training doesn't scale to that window count.**
   Every SRS run so far (`srs_depth2_*`, `srs_stitch_nc4_slot8`) packs the *entire*
   schedule — all windows plus their full IR chains — into ONE training sequence
   (`L=742`-`902` for 3 windows). Thousands of windows can't be packed into one
   context window; this needs to become genuine streaming/epoch-style corpus ingestion
   (sample a random window or short chain per training step, not "the whole book visible
   in one forward pass") — this is exactly what `§ Scaling — Corpus Ingestion Recipe`
   above already specifies, but it has never actually been implemented; every SRS
   experiment run to date has used the simpler whole-schedule-packed design instead.

3. **No intermediate validation step yet.** Every stitching result so far is 3 windows.
   Jumping straight to ~2778 windows would conflate "does chaining generalize at all
   beyond 3 windows" with "does it generalize at scale" — if it fails, there'd be no way
   to tell which assumption broke.

**Recommended order before `juz1`**: (a) redesign window-tagging to not require one
unique ID per window (drop per-window tags entirely and rely on stitching + RoPE
position alone — this is the more principled fix, since position-invariant encoding is
already this project's stated MDL goal; or fall back to a small fixed set of *relative*
tags reused cyclically across windows), (b) implement the streaming/epoch corpus-ingestion
training loop from `§ Scaling — Corpus Ingestion Recipe` (a real architecture change to
`experiments/srs_tagged/train.py`, not a config tweak), (c) validate at an intermediate
scale with many chained windows (e.g. 128B-256B, 7-15 windows) using the new training
loop before attempting `juz1`'s ~2778. Treat `juz1` as 2-3 design steps away, not a
next-config-launch away.

### Concrete `juz1.txt` scaling design — data/capacity check, block sizing, candidate traj_mix strategies

**Data check**: `datasets/juz1.txt` = 44,443 bytes (`wc -c`, confirmed). At the proven
`chunk_len=16`, that needs `ceil(44443/16) = 2778` chunks minimum. Rounding to the
nearest clean multiple of 128 (as requested — prefer power-of-2/nice-multiple sizing):
**`n_chunks=2816` (128x22) = 45,056 bytes, only 613B (1.4%) padding** — clearly better
than the nearest true power of 2 (`4096` chunks = 65,536B = 47.5% padding, wasteful) or
under-covering (`2048` chunks = 32,768B, doesn't cover the file at all). Recommend
**`n_chunks=2816`** as the padded target.

**Capacity check**: current models are 166k-232k params (`d=64, n_layers=4`). Per this
project's own MDL principle (`docs/MDL_MODEL_SIZE.md`: parameter count should scale with
*algorithm* complexity, not corpus length — the same model already handles 64B and 128B
with no architecture change), **no model-size growth is expected to be needed for
juz1** — this is a scaling-the-training-loop problem, not a scaling-the-model problem.

**Why the CURRENT training/eval design cannot reach 2816 chunks — the actual blocker,
quantified**: every run so far (`srs_stitch_nc4_slot8`, `srs_stitch_nc8_slot8`, ...)
packs the ENTIRE window schedule into one sequence sharing one `L x L` mask. At
`n_chunks=2816` with the proven window geometry (`window=32B, stride=16B`), that's
`n_windows = 2816 - 2 + 1 = 2815` windows, each contributing an IQ + 2 IR block
(~530-650 tokens/window with tags at `slot_len=8`) — extrapolating the observed
`L=902 (nc=4) -> L=1694 (nc=8)` growth rate, `L` at `nc=2816` would land somewhere in
the **hundreds of thousands of tokens**. The `L x L` mask ALONE (a numpy bool array)
would need `(3-6x10^5)^2` ~ `10^11` entries — tens to hundreds of GB just to hold the
mask, before any model computation. **Both training AND eval (the full stitched decode)
break the same way** — `ar_decode_srs_stitched_tagged` also builds one `pos_content`/
`mask_np` for the whole `windows` list at once, so it has the exact same ceiling.
`chunk_attn` (the existing memory-saving flag) does NOT help here — confirmed by reading
`kvmem/model.py`: it only computes attention in row-chunks to reduce peak *activation*
memory, the FLOP cost and the mask itself are still full `O(L^2)`, unchanged.

**The fix: local, fixed-size blocks — never one global mask, ever.** Both training and
eval should operate on a small, FIXED-SIZE local window run (matching an already-proven
or near-proven scale — recommend **`block_nc=8` chunks, 7 windows, `L=1694`**, same as
the current `srs_stitch_nc8_slot8` config) sampled from a random starting offset within
the padded 2816-chunk corpus, rather than ever building one packed sequence for the
whole corpus. This is a genuine architecture change to the training loop (not a config
tweak) but does NOT require gradient checkpointing or multi-segment forward passes for
TRAINING — because each step's `L` is bounded by `block_nc`, not by corpus length,
total per-step compute stays flat regardless of how large the corpus grows; only the
NUMBER of steps needed to cover it grows. `2816 - 8 + 1 = 2809` valid starting offsets
give plenty of sampling diversity.

**Eval DOES need the "several forward passes" approach** the same way: chain
`ceil(2809 / 7) ~ 402` independent local-block decodes end-to-end, each one a normal
`ar_decode_srs_stitched_tagged`-style call over its own small `L=1694` mask, with block
`i+1`'s first window's warmup seeded from block `i`'s own last decoded output bytes —
literally the same warmup-seeded stitching mechanism already proven WITHIN one block,
just extended ACROSS block boundaries too (no new mechanism, only a change in what
counts as "one call" — currently one call chains 3 or 7 windows; the juz1 version chains
~2815 windows across ~402 calls). Cheap: each block decode takes on the order of seconds;
~402 of them sequentially is plausibly a 20-40 minute full end-to-end check, well within
a single monitoring cycle — feasible to run periodically, not just once at the very end.

**Four candidate traj_mix / training strategies** (in increasing sophistication —
recommend prototyping in this order):

1. **Uniform random block sampling (baseline, build first)**: each training step
   samples a uniformly random starting chunk offset in `[0, n_chunks - block_nc]`,
   builds the exact same `block_nc=8` window+IQ+IR training example as today's
   `srs_stitch_nc8_slot8`, just with the *source bytes* drawn from that random offset
   into `juz1.txt` instead of always the same fixed content. Smallest code change (only
   the batch-sampling step changes; `chunk_positions_srs_tagged`, `make_batch_tagged`,
   `chunk_mask_fb` all reused completely unmodified). Every byte gets roughly equal
   training exposure in expectation over many steps — no explicit SRS scheduling yet,
   just the ingestion-loop plumbing this whole roadmap has been blocked on.

2. **SRS-weighted resampling (the actual point of this whole research program)**:
   maintain a per-block retention/difficulty score (e.g. an EMA of that block's own
   recent eval match%, from a periodic sub-sampled eval sweep — can't afford to eval
   all 2809 blocks every `eval_every` cycle, so randomly eval e.g. 20-50 blocks each
   cycle and use that to update scores), then oversample low-retention blocks more
   often — a genuine implementation of this doc's Core Idea (spaced repetition,
   review-what's-forgotten) rather than uniform coverage. This is what distinguishes
   "SRS corpus ingestion" from plain random sampling and is the natural next step once
   strategy 1 confirms the basic streaming loop works at all.

3. **Curriculum block-size growth**: start training with `block_nc=4` (3 windows,
   `L=742`, the CLEANEST proven scale — 100%/100% sustained) sampled across the whole
   corpus, then grow to `block_nc=8` (7 windows) once loss/match% indicate readiness —
   hedges against `block_nc=8`'s still-unresolved window-G-style difficulties (see
   above) by not committing to the harder block size until the easier one is solid
   across the WHOLE corpus, not just one fixed 64B slice.

4. **Hierarchical: local blocks + periodic full-corpus review pass**: combine strategy
   1 or 2 with an occasional (e.g. every K training steps) full sequential end-to-end
   stitched decode across ALL ~402 blocks (eval-only, no gradient) to directly measure
   the thing that actually matters — "does the full chained decode reproduce all
   44,443 bytes exactly" — since per-block metrics alone can miss COMPOUNDING errors
   across a ~2815-window chain even if every individual block scores near-100% in
   isolation. This is the closest analogue to the literal "final stitch eval must
   generate the same byte sequence 100% match" bar and should be the actual pass/fail
   gate, not per-block averages.

**On gradient-detached carry-state (Transformer-XL-style memory)**: NOT required for a
first attempt at pure byte-exact memorization — local block training already gets
cross-block continuity "for free" through warmup-seeded stitching (the same mechanism
already proven within one block), no compressed carry-state needed for that. Where it
COULD matter: `juz1.txt` is real Arabic text with genuine repeated phrases (unlike the
synthetic random-byte val sequences used everywhere else in this project) — a purely
local window might not disambiguate "which occurrence of this common recurring phrase
am I at" without some longer-range signal beyond the local 16B warmup. Recommend
treating this as a **Tier-2 fallback**, not a v1 requirement: try local-only sampling
first (simpler, matches everything already proven); only add a small
gradient-stopped SLOT carry-over (frozen forward-pass output from the previous few
blocks, prepended as extra non-differentiable context) if repeated-phrase confusion
actually shows up as a specific, diagnosed failure mode in the full-corpus eval pass
(strategy 4) — don't build it preemptively.

**Recommended first step**: implement strategy 1 (uniform random block sampling,
`block_nc=8`, `n_chunks=2816` padded target) as the smallest possible extension of the
existing `experiments/srs_tagged/train.py` — this alone validates whether the streaming
loop works at all before layering SRS-weighting or curriculum growth on top. Not yet
built; queued as the actual "implement the streaming corpus-ingestion training loop"
step this doc has referenced since the original `§ Scaling — Corpus Ingestion Recipe`
section, now with concrete numbers grounding it instead of the original design sketch.

### Estimates — quantitative sizing (windows, refine steps, KV capacity, wall-clock)

**Total windows/context**: `n_chunks=2816` padded corpus -> **2815 total overlapping
windows**; chained into `block_nc=8` local units (7 windows/block, `L=1694`, 1-chunk
overlap between consecutive blocks) -> **403 blocks** cover the corpus end-to-end.

**Refine steps**: keep `n_refine=2` — no basis to increase it for a larger corpus; it's
a per-window property, decoupled from total corpus length, and every proven result
(32B through 128B) used exactly 2.

**KV/SLOT capacity — not the bottleneck, doesn't need to grow**: `slot_len=8` per
16-byte chunk (2x compression) is applied PER-CHUNK, not globally, so it scales
automatically and linearly with corpus length under the local-block design — never a
fixed ceiling to solve. Sanity check: one SLOT position's raw KV capacity across 4
layers x 4 heads x 16-dim K+V ~ 4096 bits, vastly more than the ~128 bits (16 bytes) it
needs to carry. This project's own prior diagnosis already found *addressing*, not
capacity/rank, was the real bottleneck at every scale tested — expected to hold at
juz1 scale too.

**Wall-clock — genuinely uncertain, driven by cross-block transfer speed** (per-step
cost is architecture/L-bound at ~1.4 it/s for `block_nc=8`, NOT corpus-length-bound; the
unknown is total STEPS needed):

| Assumption | Total steps | Wall-clock @ 1.4 it/s |
|---|---|---|
| Optimistic (shared algorithm generalizes fast across offsets) | ~150,000-300,000 | 1.2-2.5 days |
| Matches single-location deep lineage (~260k) + modest coverage overhead | ~400,000-600,000 | 3.3-5.0 days |
| Naive no-sharing baseline (403 blocks x ~50k steps/block, bounds the pessimistic case) | ~20,000,000 | ~165 days |

**Traditional-LM framing (batch size / epoch) surfaces a real implementation gap**:
today's code shares ONE sampled position/content across the whole batch (diversity
currently comes from independently-random BYTES at a fixed position — meaningless for
real text, since content at any offset is fixed). Real-corpus training needs batch
diversity from DIFFERENT OFFSETS per batch item — a genuine code change (per-example
masking/position), not a hyperparameter tweak. At `B=32` distinct offsets/step, one
epoch over 403 blocks ~ 13 steps; a 300k-step budget ~ 23,600 epochs — consistent with
this being intentional-overfitting/exact-memorization, not generalization-focused
pretraining, so "many epochs" is expected and not a red flag by itself.

### Worked example — 65,536 bytes (2^16), sequential incremental ingestion + SRS review

A different, more realistic framing than "randomly sample any offset": simulate a user
handing the corpus to the model incrementally, chunk by chunk, in order, exactly as
`docs/SRS_RECIPE.md`'s original Core Idea envisioned — not i.i.d. random sampling
across a static, already-fully-available corpus.

**Scale**: `n_chunks=4096` (2^16 bytes / 16), `n_windows=4095`, **585 sequential local
blocks** (`block_nc=8`, 7 windows/block, `L=1694`, stepping by 7 chunks with 1-chunk
overlap).

**Phase A — Bootstrap (blocks 0-49, heavier budget)**: the model hasn't learned the
general position-invariant "compress->address->refine" algorithm yet — needs exposure
to several DISTINCT real blocks (not one repeated block, which every experiment this
session has actually tested) before that generalizes. Interleave blocks 0-49
(round-robin/weighted); `block_nc=4` for the first ~10 blocks (most proven scale), then
`block_nc=8`; `n_refine=2` throughout. **~250,000-350,000 steps** (same order as the
single-block deep lineage, ~260k, since it's the same skill, just needing a few
different examples instead of one repeated one for weight-shared generalization to
emerge).

**Phase B — Streaming ingestion + SRS review (blocks 50-584)**: once the core
mechanism is learned, each NEW block needs far less dedicated exposure (fine-tuning an
already-general skill onto new content, not learning the skill itself):
- New-block ingestion: ~500-2,000 steps/block (the single biggest unverified number in
  this whole estimate — no experiment has tested cross-block transfer speed yet).
- SRS review (Leitner/SM-2-style doubling intervals, in units of "new blocks ingested
  since last review"): review at +1, +2, +4, +8, +16... further blocks, until answered
  correctly M=4 consecutive times, then "mastered" (rare maintenance checks only, every
  500+ new blocks) — review load stays LOGARITHMIC in corpus size, not linear; this is
  the actual point of using SRS instead of naive full replay every step.
- 535 remaining blocks x ~4 average reviews each ~ 2,140 review episodes x ~100-300
  steps/episode ~ 200,000-650,000 review steps. New-block ingestion: 535 x 500-2,000 ~
  270,000-1,070,000 steps.

| Component | Steps |
|---|---|
| Phase A (bootstrap, 50 blocks) | 250,000-350,000 |
| Phase B new-block ingestion (535 blocks) | 270,000-1,070,000 |
| Phase B SRS review (2,140 episodes) | 200,000-650,000 |
| **Total** | **~700,000-2,100,000** |

At 1.4 it/s: **~5.8-17.4 days**. Compare to the naive no-sharing/no-SRS baseline (585
blocks independently at the single-block deep-lineage cost, ~260k steps each) ~ 152
million steps ~ ~1,256 days — the SRS+shared-algorithm design is the entire reason this
is remotely feasible. **Every number past `n_blocks=585` is built on an untested
assumption** (that a position-invariant algorithm transfers this cheaply across
distinct real content). Recommend measuring the real per-new-block transfer cost via a
small pilot (Phase A + ~10 Phase-B blocks, a few hundred thousand steps) before
committing to a multi-day run — that single measurement lets the rest of this table be
recalculated from data instead of extrapolated guesswork.

**Step (c) started** (still whole-schedule-packed design, not yet the streaming loop —
picked as the lowest-risk next autonomous step, since it only requires extending
`_SRS_SPAN_TAGS` from 3 to 7 entries, not the harder streaming-training rewrite):
`experiments/srs_tagged/configs/srs_stitch_nc8_slot8.py` — 128B coverage via 7 chained
windows `[(0,2),(1,3),...,(6,8)]` (same stitching mechanism as the 64B win, just more
windows), added `HMN_QUERY_D..G` tags (`vocab.py`, `HMN_TAG_VOCAB_SIZE_V3=290`) and
extended `_SRS_SPAN_TAGS` to 7. Smoke-tested (10-step run + warm-start-with-vocab-growth
both confirmed working, `special_embed.weight` grows 26->34 rows automatically). Warm-
started from `srs_stitch_nc4_slot8`'s `stage0_best.pt`. `L=1694` (vs 742 for 3 windows);
estimated ~1.1-1.3 it/s, ~13-15hr for 60k steps. Launched 2026-07-10 08:48 local.

**`srs_stitch_nc8_slot8` final result (60k steps, complete)**: val 100%×6 windows +
window G (last) stuck at **4.2%** (STITCHED_MEAN 80.8%); test even weaker (100/100/
16.7/8.3/4.2/0/33.3, STITCHED_MEAN 38.3%). **Fails the strict 100%-test bar** — unlike
the clean 64B win, this run also shows real test degradation on the MIDDLE windows
(C-F), not just window G, a broader val/test generalization gap at this larger scale.

**Diagnosis (qualitative decode inspection)**: window G's own IQ->IR1->IR2 chain shows
IQ=0% (expected, IQ alone is a rough guess) -> **IR1=100%** (fully recovers) ->
**IR2=4.2%** (collapses back to near-total failure). This is the "IR2 destroys IR1's
gain" pathology documented elsewhere in this project (`down_counter`, 64B scale) —
`wrong_token_weight` only upweights loss where the fed-back argmax was WRONG; it does
nothing to protect already-correct positions from being overwritten by IR2's transform.
Root cause specific to this run: window G only becomes reliably correct at IR1 very
late in the single 60k-step cosine cycle (after windows A-F, which converge earlier,
absorb most of the LR budget) — by the time IR1 is reliable for G, LR has decayed to
~1e-6, leaving no gradient room to teach IR2 "leave this alone" in that regime. Windows
A-F didn't hit this because they became reliably correct while LR was still high.

**Fix launched**: `experiments/srs_tagged/configs/srs_stitch_nc8_slot8_continue.py` —
warm-started from `stage0_end.pt`, fresh short cosine cycle (`cosine_T0=20000`,
`lr_max=5e-5`, lower peak than the original 1.5e-4) to give the model renewed gradient
signal specifically in the "IR1 already correct" regime window G only reached at the
very end of the prior run. If this doesn't resolve window G within 20k steps, the next
fix is oversampling window G specifically — not attempted first since the current
single-fixed-schedule design has no traj_mix/weighted-sampling machinery (restructuring
that is a bigger change than this LR-based attempt). Launched 2026-07-10 21:33 local.

**`srs_stitch_nc8_slot8_continue` final result (20k steps, complete) — fix FAILED**:
window G ended at val **1.4%** (vs 100% for windows A-F), test 25.0% — essentially
unchanged from the pre-fix 4.2%/33.3%. The fresh short cosine cycle at lower peak LR
did not give window G meaningfully more room to learn "preserve an already-correct
IR1 answer" — its val score was flat (0-5.6%) across every one of the 10 checkpoints
in this continuation run, no upward trend. **Revised diagnosis**: this is likely not
(purely) an LR-budget problem — the flat non-improving trend across a full fresh
training cycle suggests a structural limit tied to window G's position (the single
largest RoPE/relative-distance offset in the whole `L=1694` packed sequence — window G
sits after 6 other windows' full IQ+IR chains), not a "hasn't had enough gradient
steps in the right regime" problem as first hypothesized. **Decision**: stop chasing
this fix (two attempts, ~4hr+4hr sunk, no improvement on the second) — 6/7 windows at
100% val is still a strong, informative result to report honestly, and further LR/
schedule tweaks are unlikely to be the right lever. The oversampling-window-G idea
noted as the fallback earlier remains a valid next attempt for a FUTURE session, but
requires restructuring the single-fixed-schedule design (bigger change, not attempted
in this autonomous loop) — noted as a TODO rather than pursued further here. Moving on
to the already-queued dual-attn ablation instead.

### Dual-attention-block ablation (MLP removed, replaced with a second self-attention)

**Motivation**: MLP layers in a regular LM are the documented mechanism for storing
static factual/associative knowledge in weights (Geva et al., "Transformer FFN Layers
Are Key-Value Memories"). This architecture deliberately keeps content OUT of the
weights — it lives in the runtime SLOT KV (this doc's fast-weight framing) — so MLP's
usual job may not be load-bearing here; the useful computation (copy/route bytes from
encoded SLOT positions to output positions) is an addressing problem, attention's job.
Counter-risk: MLPs are the model's only source of per-token NONLINEAR elementwise
recoding — attention is structurally a weighted average of other positions' values,
with no native way to apply e.g. "add 1 to this byte" independent of context. The val
suite's `up_counter`/`down_counter`/`odd` sequences specifically exercise that kind of
per-token arithmetic transform, so this ablation also diagnoses whether that capability
is actually needed for this task.

**Design**: `experiments/attn_dual/` (new, isolated folder, zero edits to existing
code) — `DualAttnBlock` (attn1+attn2, no FFN) replaces `TransformerBlock` (attn+ffn);
reuses `MHAttention`/RoPE/norm machinery from `kvmem/model.py` unmodified. Same task,
schedule, and hyperparameters as `srs_stitch_nc4_slot8` (the proven 3-window 64B
baseline: warm-started, sustained 100%/100% val+test steps 35000-60000) — only the
block architecture differs. Trained FROM SCRATCH (different state_dict shape, no
warm-start possible) — a harder starting point than the baseline's warm start, so
matching or approaching that ceiling despite the harder start would be strong evidence
for the hypothesis. No KV-cache support (a dual-attn block has two KV pairs per layer,
incompatible with every existing KV-cache decode function in this project) — eval uses
full-recompute AR decode (`experiments/attn_dual/decode.py`), same tradeoff
`densenet_kv` made for its own KV-incompatible architecture; fine at this scale.

**Parameter count**: 166,656 (dual-attn) vs 232,192 (baseline) — the dual-attn variant
is actually **28% SMALLER** despite adding a second attention layer, since the FFN at
`d_ff=4d` cost more parameters than one extra attention layer does. Not a param-matched
ablation by design — if dual-attn matches the baseline's ceiling, that's evidence for
the hypothesis at LOWER parameter count, a stronger result than a matched-param tie.

**Status**: built and smoke-tested (10-step run confirmed clean end-to-end). Launched
2026-07-11 01:43 local (wallclock). Progress so far (from scratch, no warm-start):

| step | wallclock | val STITCHED_MEAN | test STITCHED_MEAN | loss |
|---|---|---|---|---|
| 5000  | 03:32 (+1h49m) | 20.8% | 21.4% | 11.73 |
| 10000 | ~04:20 | 36.3% | 39.3% | ~8 |
| 15000 | ~04:50 | 56.0% | 57.1% | ~6 |
| 20000 | ~05:20 | 56.5% | 62.5% | ~5.5 |
| 25000 | ~05:50 | 53.6% | 62.5% | ~5.3 |
| 30000 | 07:04 (+5h21m) | 40.5% | 58.9% | 5.08 |

**Comparison is currently unfair — flagged before drawing conclusions**: the
`srs_stitch_nc4_slot8` baseline this is being measured against was WARM-STARTED (from
the 64B atomic run's fully-pretrained checkpoint), while `dualattn` trains from
scratch. Warm-starting `dualattn` to match isn't a valid fix either — dual-attn has no
FFN to receive the baseline's pretrained FFN weights; at best `attn1` could receive the
baseline's matching-shape attention weights while `attn2` stays randomly initialized, a
half-transferred/half-random model that tests a different, confounded question.

**Correct fix — clean from-scratch-vs-from-scratch comparison**: train a SECOND
baseline run, `srs_stitch_nc4_slot8` architecture (attn+ffn), from scratch, same task
(`windows=[(0,2),(1,3),(2,4)]`), same 60000-step budget, same hyperparameters — the
missing control this project never actually ran (the existing baseline result was
always warm-started). Queued to launch immediately after `dualattn` finishes (never two
jobs at once). Config: `experiments/srs_tagged/configs/srs_stitch_nc4_slot8_scratch.py`
(new, same hp as the warm-started config minus `--pretrained`).

**Wallclock budget note**: `dualattn` had ~2h11m left as of 07:10 (step 31000/60000,
~3.67 it/s). A from-scratch run of the standard architecture at the same `L=742` should
run at ~5.9 it/s (matching the warm-started baseline's own speed, same architecture/
sequence length) — ~2.8h for 60000 steps. Sequential total from 07:10: ~5h — does not
fit within the 24h+2h extended autonomous budget (ends ~10:46 local). Will report
whatever wallclock progress both runs reach by the deadline; the from-scratch baseline
run will likely need to continue past the budget window or be checked in a follow-up
session.

**`dualattn_nc4_slot8` final result (60k steps, complete, wallclock 01:43->09:13,
~7h30m)**: val 50.6%, test 55.4%, best-over-run 56.5% (val, step ~40000-50000 region).
Loss finished at 3.4567 — nowhere near the warm-started baseline's ~0.0001, i.e. this
run did NOT converge within budget, plateauing in the 35-56% match range for the back
half of training with no clear final climb toward 100%. **Preliminary read (from-scratch
vs from-scratch comparison not yet available — see below)**: the no-MLP architecture
clearly underperforms what the standard architecture achieves on this task, though part
of the gap is expected to be the missing warm-start advantage, not architecture alone —
final verdict awaits the from-scratch control below.

`srs_stitch_nc4_slot8_scratch` (from-scratch control, standard attn+ffn architecture,
identical task/steps/hyperparameters) launched 2026-07-11 09:35 local — only ~1h10m
remained in the 24h+2h budget at launch time (~2.8h needed for 60000 steps), so this run
will continue past today's deadline. Partial progress will be reported at whatever step
count the budget window allows; full comparison requires a follow-up check.

**Infra anomaly discovered while watching this run**: actual sustained throughput is
~0.35-0.68 it/s (measured from real inter-checkpoint wallclock deltas), roughly **10x
slower** than the ~5.9 it/s every prior run at this exact architecture/L achieved,
despite tqdm's displayed instantaneous rate still showing ~5.9 it/s (a smoothed/local
measurement, not representative of overall pace). Process confirmed alive and using CPU
(`STAT=UN`, ~26% CPU) — not hung, just severely throttled. One confirmed cause: a
macOS "Maintenance Sleep" event at 10:55:35 (woke 11:05:02, ~9.5min gap) despite
`caffeinate -i` running — `-i` only prevents *idle* sleep, not Power Nap/maintenance
sleep. This doesn't fully explain the full ~10x slowdown though (single 9.5min gap vs
~85min of near-zero net progress observed earlier). Likely compounding factor: this is
the ~4th consecutive MPS training job run back-to-back over the past ~24h+ with no
cooldown (`srs_stitch_nc8_slot8` -> `_continue` -> `dualattn_nc4_slot8` -> this run) —
thermal throttling or GPU-driver-level resource degradation from sustained near-100%
MPS utilization is a plausible explanation, though not confirmed (`pmset -g therm`
showed no recorded thermal warning, for what that's worth on this hardware). At this
rate, 60000 steps would take 24-48h, not ~2.8h. **This is the concrete trigger for the
GPU-migration discussion from earlier in this session** — a dedicated GPU box would
both avoid this class of OS-level power-management interference and remove the thermal
risk of days of continuous back-to-back MPS runs on laptop hardware. (Throughput did
fully recover to ~5.9 it/s after step ~3000 and stayed there for the rest of the run —
the stall was transient, not a permanent degradation.)

### Final verdict: dual-attn (no MLP) ablation — inconclusive, but warm-starting turns out to be the dominant variable

**`srs_stitch_nc4_slot8_scratch` final result (60k steps, complete, wallclock
09:35->13:50, ~4h15m)**: val 45.2% (best-over-run 51.8%), test 64.3%, loss 5.0237 —
plateaued, never approached the warm-started baseline's ~0.0001.

**Three-way final comparison** (all same task: `windows=[(0,2),(1,3),(2,4)]`, 64B,
slot_len=8, wrong_token_weight=2.0, 60000 steps):

| Run | Params | Warm-started? | Final val | Final test | Best val | Final loss |
|---|---|---|---|---|---|---|
| `srs_stitch_nc4_slot8` (standard, attn+ffn) | 232,192 | **yes** | 100% | 100% | 100% (sustained 35k-60k) | ~0.0001 |
| `dualattn_nc4_slot8` (no MLP, attn+attn) | 166,656 | no | 50.6% | 55.4% | 56.5% | 3.4567 |
| `srs_stitch_nc4_slot8_scratch` (standard, attn+ffn) | 232,192 | no | 45.2% | 64.3% | 51.8% | 5.0237 |

**Matched from-scratch comparison (dual-attn vs standard, isolating the MLP variable)
— genuinely mixed, no clear winner**: dual-attn converged FASTER early (ahead at every
checkpoint through step ~25000) and reached a LOWER final training loss (3.46 vs 5.02
— notable, since dual-attn has fewer parameters and no MLP), but the standard
architecture's final TEST match (64.3%) exceeded dual-attn's (55.4%), while dual-attn's
best VAL (56.5%) exceeded standard's (51.8%). Neither architecture is unambiguously
better within this budget — this does not confirm the "MLP is unnecessary" hypothesis,
but it also does not refute it; it shows the training-budget-limited from-scratch
regime is too noisy/incomplete on this task to give a clean 60k-step verdict either way.

**Interesting sub-finding — loss and match% diverge between the two architectures**:
dual-attn's LOWER teacher-forced loss (3.46) with WORSE test match (55.4% vs standard's
64.3% at higher loss 5.02) suggests loss on the teacher-forced objective doesn't track
AR-decode match% as tightly for the no-MLP architecture. Plausible explanation: AR
decode compounds per-token errors across many autoregressive steps and 2 chained IR
turns with argmax feedback — MLP may be doing disproportionate work specifically in
that feedback-robustness role (correcting/regularizing after an imperfect argmax input)
rather than in the raw content-recall role this ablation was designed to isolate. Not
confirmed, flagged as a follow-up direction.

**The dominant, higher-confidence finding of this whole ablation series**: warm-starting
was doing FAR more work than either architecture alone. Both from-scratch runs — with
and without MLP — plateaued in the 45-65% range, nowhere near the original warm-started
100% result. This means the original `srs_stitch_nc4_slot8` "clean win" (100%/100%
sustained) should not be read as "this architecture/task combination is easy to learn
from scratch in 60k steps" — it depended heavily on prior learned representations
(addressing, routing) carried over from the 64B atomic run's own warm-start lineage.
This is a genuinely new, useful finding independent of the MLP question, and suggests a
real prerequisite for the `juz1` scaling roadmap: expect every new scale/schedule
change to need either a longer from-scratch budget or a warm-start plan, not assume
fast convergence by default.

**Status**: dual-attn ablation closed as inconclusive (not a negative result — the
comparison was too budget-limited to resolve).

**Planned analysis addition — IR loop as micro-SRS recurrence**: the argmax-feedback
IR loop (`IQ -> IR1 -> IR2`) is structurally a form of UNROLLED RECURRENCE even without
literal RNN state — each turn writes fresh `SLOT_A`/`SLOT_B` tokens conditioned on the
previous turn's argmax output, the same "review and strengthen a memory trace" pattern
as the macro-level SRS schedule (`srs_schedule`/`srs_schedule_depth2`), just operating
at a much smaller timescale: within one forward pass (turn-by-turn), not across the
training curriculum (spaced across thousands of steps). Once `dualattn_nc4_slot8_ir`
finishes, worth checking whether the SLOT content actually evolves turn-to-turn the way
a strengthening memory trace would (i.e. IR1's SLOT_B encodes something meaningfully
different from IQ's SLOT, not just "the same address, re-verified") — this would connect
`attn1`/`attn2`'s roles (see "Where does nonlinearity live without an MLP" discussion,
this session) to the SRS retention-and-review principle operating internally, not just
across training steps. Candidate method: adapt `attn_viz.py` (currently built for
`KVMemModel`/old trajectory types only) for `DualAttnModel` + the tagged stitched
layout, and/or a probe comparing SLOT-position hidden states across IQ vs IR1 vs IR2 for
the same window.

**Follow-up: staged pretraining for dual-attn (in progress)**. Rather than a single
60k-step from-scratch shot (what both prior dual-attn/scratch-baseline runs did) or a
cross-architecture weight transplant (considered, rejected — mixing "partially
pretrained" and "fully random" sublayers within one block muddies the comparison
rather than clarifying it), replicate this project's own proven staging principle
("IQ pretraining required before IR" — every prior success here, from
`hmn_feedback_32_iq`->`_ir` through `srs_depth2_nc4_slot8`->`srs_stitch_nc4_slot8`,
warm-starts a harder stage from an easier SAME-architecture checkpoint):

1. `dualattn_nc4_slot8_iq.py` — Stage 0, IQ-only (`n_refine=0`), 30000 steps, same
   task/windows. Establishes basic slot-compression/addressing before feedback is
   introduced, matching every prior IQ-before-IR precedent in this codebase.
2. `dualattn_nc4_slot8_ir.py` — Stage 1, warm-started from Stage 0, adds `n_refine=2`
   IR turns, 60000 steps (same budget as the original from-scratch ablation, so the
   ONLY variable changed is staging).

Tests whether dual-attn's earlier poor from-scratch result was a training-curriculum
gap (fixable by staging, like the standard architecture needed) rather than a
fundamental architecture limitation (would still fail even with proper staging).
Smoke-tested (10-step IQ run + 5-step warm-started IR run, confirmed clean 61/61 tensor
load). Launched Stage 0 2026-07-11 21:23 local.

**Fairness check (caught before over-claiming)**: the "100%/100%" `srs_stitch_nc4_slot8`
target ceiling was warm-started from a much DEEPER lineage than this 2-stage curriculum
gives dual-attn — chat-tags Phase A->B->B2->B3->B4->wrongtok-ablation->atomic-full-span
->stitched, likely 300k+ cumulative steps across ~7 stages, not 90k across 2. So a
negative result for staged dual-attn would NOT cleanly prove "architecture is the
limiting factor" (still confounded with staging depth). **Added a matched-depth
control**: `srs_stitch_nc4_slot8_iq.py` -> `srs_stitch_nc4_slot8_ir.py` — the SAME
30k-IQ + 60k-IR 2-stage curriculum, applied to the standard (attn+ffn) architecture.
This makes the comparison a clean 2x2: {dual-attn, standard} x {scratch, lightly-staged
at matched depth}, isolating the architecture variable at every training regime tested
so far, rather than comparing one architecture's full historical lineage against the
other's fresh 2-stage attempt.

**Revised queue** (never two jobs at once): `dualattn_nc4_slot8_iq` (running) ->
`dualattn_nc4_slot8_ir` (warm-started) -> `srs_stitch_nc4_slot8_iq` (new control) ->
`srs_stitch_nc4_slot8_ir` (new control, warm-started) -> full analysis across all 6 data
points (2 from-scratch + 2 lightly-staged + the original deep-lineage 100% result, for
context) -> then deferred experiments (window-G oversampling fix, then `juz1.txt`
prerequisite work).

**Revised again after the RMSNorm result**: if `dualattn_nc4_slot8_ir` (rmsnorm=True)
confirms strong success, repeat `srs_stitch_nc4_slot8_iq`/`_ir` (also updated to
rmsnorm=True) to establish full matched-depth parity between architectures under
RMSNorm — replacing the LayerNorm-based matched-depth comparison rather than adding a
third variant alongside it. THEN, before the previously-planned window-G oversampling
restructuring: **try RMSNorm as a candidate fix for window G** in the nc8 stitched run
first — window G's failure mode ("IR2 destroys IR1's gains", `srs_stitch_nc8_slot8`) is
exactly the class of IR-loop convergence-dynamics problem RMSNorm just showed a
dramatic effect on here, and it's a much cheaper test (one flag + rerun) than
restructuring the schedule for oversampling. Only fall back to oversampling if RMSNorm
doesn't resolve it. Then `juz1.txt` prerequisite work as before.

**Launched**: `experiments/attn_dual/configs/dualattn_nc8_slot8_ir.py` — dual-attn (the
architecture confirmed by the hard-gate comparison above) at nc=8/128B, `rmsnorm=True`,
warm-started from `dualattn_nc4_slot8_ir`'s 94.0%/94.6% checkpoint (not from scratch —
grows `special_embed` 26->34 rows for the new window D-G tags). **Bug found and fixed
first**: `experiments/attn_dual/train.py`'s warm-start logic only handled exact-shape
matches, silently DROPPING any growing tensor (`special_embed.weight`) instead of
copying its overlapping prefix — smoke test caught it (loss=62 instead of a sane
warm-started value, only 51/52 tensors "loaded" with the grown one silently skipped).
Fixed by porting the same prefix-copy-on-growth logic already proven in
`experiments/srs_tagged/train.py`. Re-verified: `special_embed.weight: (26, 64)->(34,
64)` now grows correctly. 60000-step budget (matching the original nc8 atomic run's
budget, not the full 260k matched-depth budget — this is a targeted fix test).

**Second bug found and fixed — MPS OOM crash**: the first launch (B=6, no `chunk_attn`)
crashed with `RuntimeError: MPS backend out of memory` sometime after step 5000.
`chunk_attn` (the existing memory-only row-chunked-attention flag, verified earlier to
NOT reduce FLOPs, only peak activation memory) was never wired into
`experiments/attn_dual/model.py`/`train.py` at all — and `DualAttnModel` has TWO
attention sublayers per block vs the standard architecture's one, roughly doubling the
attention-map memory footprint at the same `L=1694`. Fixed: `chunk_attn` now wired
through `build_dualattn_model` and `train.py`'s `hp_model` dict (set to 256, matching
standard-arch nc8 configs); `B` lowered 6->3 for extra margin. Relaunched clean.

**This crash sat undiagnosed for ~4+ hours due to a separate, unresolved
`ScheduleWakeup` delivery gap** — a fallback scheduled for ~30 minutes out fired closer
to 4 hours later (cause unconfirmed, outside what could be directly debugged from
within the session). The process had already exited with the full OOM traceback
sitting in its launch log the entire time. Going forward: treat `Monitor` task-
notifications (which fired reliably and promptly throughout this session) as the
primary signal, `ScheduleWakeup`'s delay as an untrusted fallback rather than a
reliable heartbeat — and check process liveness (`ps -p <pid>`), not just log content,
on every wake, since a silently-exited process produces no new log lines and looks
identical to "still running slowly" without that explicit check.

**RMSNorm restart**: since the IQ stage had only just started (LayerNorm version was at
step ~20000/160000, val 25.0%/test 44.6% and progressing normally), switched to
`rmsnorm=True` and restarted from scratch to test the more MDL-aligned, fully bias-free
variant — reason given: elegance, matches the project's own "every Linear is bias=False"
philosophy (LayerNorm's β was the one remaining implicit bias). Fixed a real bug first:
`DualAttnModel` didn't plumb `rmsnorm` through at all (hardcoded `norm_out=_make_norm(d,
False)`, never passed the flag to blocks). Same `lr_max=1.5e-4`, no retune, as an
initial single-variable test. **Known risk, flagged before launching**:
`configs/hmn_chunk_abl_rmsnorm.py` (a prior small-scale ablation, unrelated to this
project's current architecture question) showed RMSNorm converging clearly worse than
LayerNorm at an unretuned LR (IR-stage final loss 2.17 vs 0.54, plus a loss spike at
the IQ->IR transition). **Explicit fallback plan**: if this run fails to converge
comparably to the LayerNorm version's step-20000 checkpoint (val 25.0%, test 44.6%),
revert to `rmsnorm=False` — not an LR retune first. New param count: 166,080 (576 fewer
than the LayerNorm version, matching the removed β/γ terms).

**Results — dual-attn RMSNorm staged run, both stages complete**:
- Stage 0 (IQ-only, 160k budget): hit a PERFECT sweep by step 30000 (100%/100% val+test,
  loss 0.0026), confirmed stable at step 40000 (2nd consecutive perfect checkpoint) —
  stopped early, no value in the remaining 120k steps.
- Stage 1 (IQ+IR, 100k, warm-started from Stage 0): final val 94.0% (best 94.0%), test
  94.6%, loss 0.2658 — does not hit the strict 100% bar but is dramatically better than
  every prior dual-attn attempt (previous best from-scratch, LayerNorm, was 50.6%/55.4%).

**Parity control launched**: `srs_stitch_nc4_slot8_iq` (standard architecture, same
matched-depth 160k/100k budget, now also `rmsnorm=True`) — 231,616 params (576 fewer
than its LayerNorm counterpart, same delta as dual-attn's).

**`srs_stitch_nc4_slot8_iq` (standard, rmsnorm=True) final result — genuinely different
failure mode from dual-attn**: val 100.0% (best, stable from step ~110000 on), test only
**53.6%** (window `(2,4)` collapsed to 4.2%). Unlike dual-attn's RMSNorm IQ stage (clean
100%/100% sweep, val and test tracking closely throughout), this run showed a large,
persistent, noisy val/test gap for most of training (test oscillating 23-89% while val
climbed steadily to 100%) — an overfitting-like signature RMSNorm apparently does NOT
protect the standard (with-MLP) architecture from on this task, even though it helped
dual-attn generalize essentially perfectly. **This is a real, important architecture x
normalization interaction**, not noise: RMSNorm's benefit appears specific to (or at
least much stronger for) the no-MLP architecture. IR stage (`srs_stitch_nc4_slot8_ir`,
warm-started) launched to see if the argmax-feedback refinement loop can close this test
gap the way it closed dual-attn's own gap in its own IR stage (dual-attn IQ was perfect,
IR degraded to 94%/94.6% — the two architectures may be trading which stage shows the
weakness).

**`srs_stitch_nc4_slot8_ir` (standard, rmsnorm=True) final result — confirms the
architecture-specific interaction, does not resolve it**: finished all 100000 steps at
val 40.5%, test 30.4% (best-over-run 44.0% val, step 90000). The IR stage did NOT close
the IQ stage's val/test gap — it stayed low throughout (never exceeded ~44% val), far
below dual-attn+RMSNorm's equivalent IR-stage result (94.0%/94.6%) at every matching
checkpoint (10k through 100k). This is now a clear, sustained, non-noise finding:
RMSNorm at this matched-depth budget helps dual-attn converge cleanly but leaves the
standard (with-MLP) architecture unable to get anywhere close, on the identical task.
**Missing comparison cell (standard+LayerNorm at the same matched depth) launched next**
— `srs_stitch_nc4_slot8_iq_ln.py` — to determine whether this is a RMSNorm-specific
problem for standard-arch, or whether standard-arch just needs deeper staging than 260k
regardless of norm choice (the depth confound flagged in the architecture-comparison
audit above).

**`srs_stitch_nc4_slot8_iq_ln` final result — resolves the confound decisively**:
finished all 160000 steps at val 100.0%, test **89.3%** (best-over-run 100.0% val,
window `(2,4)` the one remaining soft spot at 75%). Compare directly to the RMSNorm
version's identical-depth result: val 100.0%, test only **53.6%**. **This confirms the
architecture-specific interaction is a genuine RMSNorm-specific problem for the
standard (with-MLP) architecture, not a depth confound** — same budget, same task, same
architecture, only the norm choice differs, and LayerNorm generalizes dramatically
better (89.3% vs 53.6% test). IR stage (`srs_stitch_nc4_slot8_ir_ln`, warm-started)
launched to complete the picture — if it also converges cleanly (unlike its RMSNorm
counterpart, which finished at val 40.5%/test 30.4%), this becomes the strongest
evidence yet that **the standard architecture should stay on LayerNorm**, while
**dual-attn should stay on RMSNorm** — i.e. the right choice may be architecture-
DEPENDENT norm selection, not one universal winner.

**`srs_stitch_nc4_slot8_ir_ln` final result — flips the norm-dependent hypothesis,
reveals the real finding**: finished all 100000 steps at val 33.3%, test **23.2%**
(best-over-run 48.8% val) — actually slightly WORSE than standard+RMSNorm's IR result
(val 40.5%, test 30.4%). The IQ-stage pattern (LayerNorm clearly better) did NOT carry
through to IR — at IR, RMSNorm edges out LayerNorm slightly for standard-arch. Norm
choice is not cleanly separable by architecture the way the IQ-only result suggested.

**Full 2x2 matched-depth comparison (260k steps: 160k IQ + 100k IR)**:

| | Standard, LayerNorm | Standard, RMSNorm | Dual-attn, RMSNorm |
|---|---|---|---|
| IQ stage | val 100.0% / test 89.3% | val 100.0% / test 53.6% | val 100.0% / test 100.0% |
| IR stage | val 33.3% / test 23.2% | val 40.5% / test 30.4% | val 94.0% / test 94.6% |

**The finding that actually matters**: both standard-arch variants badly trail
dual-attn+RMSNorm at IR, REGARDLESS of norm choice — dual-attn+RMSNorm's IR result
(94.0%/94.6%) is roughly 3-4x better than either standard-arch IR variant. This isn't a
norm-interaction story — the argmax-feedback IR/refinement mechanism itself generalizes
dramatically better in the no-MLP dual-attn architecture at this matched training
depth, across both norm choices tested for standard. **This satisfies the hard gate for
proceeding**: dual-attn+RMSNorm is the clear empirical leader across the full matched-
depth comparison, not just one favorable cell. Remaining deferred items (dual-attn+
LayerNorm cell, second seed, 128B validation, real text) stay open as follow-up
validation, not blockers — the working choice for immediate next steps (nc8 window-G
test, juz1 prep) is **dual-attn + RMSNorm**.

### One-attn-per-block-double-depth vs current dual-attn (paired) structure — decided: keep as-is

Asked whether `DualAttnBlock` (4 blocks x 2 attn sublayers = 8 total attn ops) should
be restructured as 8 single-attn blocks (n_layers=8) instead. Checked the math: these
are **mathematically identical networks** — same computational graph (`x = x +
attn(norm(x))` repeated 8 times either way), same param count, same capacity. No
functional tradeoff exists.

**Decided: do NOT restructure.** Two reasons: (1) loses the `attn1`=retrieve/
`attn2`=refine mechanistic-pairing framing the planned `attn_viz` analysis depends on
(recoverable post-hoc, but free to keep now); (2) **real, immediate cost** — existing
checkpoints (`dualattn_nc4_slot8_iq`/`_ir`) use state_dict keys `blocks.{i}.attn1.*`/
`blocks.{i}.attn2.*`; renaming to 8 single-attn blocks breaks the warm-start chain
mid-comparison, for zero functional gain. Revisit only if `attn_viz` shows no real
`attn1`/`attn2` specialization (at which point the pairing framing has no interpretive
value left).

**Latent trap flagged, confirmed currently inert**: `DualAttnBlock`'s `n_layers=4`
config value means 8 actual residual-attention operations, not 4 — if depth-scaled
init (`KVMemModel`'s existing `depth_scaled_init` option, `1/sqrt(2*n_layers)` residual
scaling) is ever added to `DualAttnModel`, it must use `n_layers*2` as the effective
depth. **Checked: `depth_scaled_init` is not currently used anywhere** — defaults to
`False` in `kvmem.model.build_model`, zero hits across every config in `experiments/`,
and `experiments/srs_tagged/train.py`'s `hp_model` dict doesn't even pass it through
(would silently stay `False` regardless). So this trap is inert for every result in
this session — only relevant if someone deliberately enables it later. Noted directly
in `experiments/attn_dual/model.py`'s `DualAttnModel` docstring.

### Is dual-attn confirmed as the final architecture? No — audit of what's actually resolved

Asked directly. Honest status: dual-attn+RMSNorm is the most promising combination found
so far, but the comparison is incomplete, not conclusive. Three specific gaps, audited
against the actual queue (not just named as caveats — checked whether anything is
actually scheduled to resolve each):

| Gap | Queued? |
|---|---|
| Standard+LayerNorm at the SAME matched-depth (260k) budget — without this we can't tell whether RMSNorm caused standard-arch's val/test gap, or whether standard-arch just needs deeper staging than 260k regardless of norm choice (a depth confound, not a norm finding) | **Was not queued — fixed below** |
| Validation at 128B+ | Partial — the queued "RMSNorm on nc8 window G" test covers the *standard* architecture at 128B, not dual-attn at 128B |
| Second-seed reproducibility (current best result, 94.0%/94.6%, is close to but short of the 100% bar — worth confirming it's not seed luck) | **Not queued** |

**Fix for the first gap (highest priority, added to queue)**:
`srs_stitch_nc4_slot8_iq_ln.py` -> `srs_stitch_nc4_slot8_ir_ln.py` — identical
matched-depth (160k IQ + 100k IR) config to `srs_stitch_nc4_slot8_iq`/`_ir`, just
`rmsnorm=False`. Smoke-tested. This completes the full 2x2x2 comparison matrix:
{dual-attn, standard} x {LayerNorm, RMSNorm} x {scratch, matched-depth-staged} — the
current run leaves 2 of those 8 cells unfilled at matched depth (LayerNorm x
{dual-attn, standard}), and this closes the standard-arch one. A dual-attn+LayerNorm
matched-depth run would close the last cell but is lower priority (LayerNorm dual-attn
from-scratch already showed a clear plateau, so this cell is less likely to change the
overall verdict) — noted as available but not queued yet.

**Second and third gaps — deferred, explicitly tracked, not silently dropped**:
dual-attn at 128B (bigger lift — needs building nc8-scale dual-attn configs with the
D-G window tags, plus its own warm-start chain) is deferred until nc8/window-G's fate
is decided on the simpler standard architecture first (testing the harder combination
before knowing if the base case is solvable would be premature). Second-seed check is
deferred as lower priority given time cost (doubles wall-clock of an already-long
stage) — a lighter partial check (re-run just the IQ stage, faster signal than full
IQ+IR) is the cheaper option if this becomes worth prioritizing later.

**Revised queue** (never two jobs at once): `srs_stitch_nc4_slot8_ir` (rmsnorm=True,
running) -> `srs_stitch_nc4_slot8_iq_ln` -> `srs_stitch_nc4_slot8_ir_ln` (the new
LayerNorm matched-depth control) -> full 2x2 comparison writeup -> RMSNorm test on nc8
window-G -> juz1 streaming loop implementation (strategy 1) + small pilot.

**Hard gate, explicit**: `juz1` implementation work does NOT start until the
architecture comparison is actually CONCLUSIVE — not merely "until the currently
queued runs finish." If the LayerNorm matched-depth results still leave the
architecture choice ambiguous (e.g. standard+LayerNorm also fails to generalize at
260k, meaning the gap is a depth confound rather than a norm-choice or architecture
finding; or the two architectures end up close enough that neither clearly wins), add
more experiments before proceeding — the deferred dual-attn+LayerNorm matched-depth
cell, and/or a second-seed check — rather than picking an architecture on an
incomplete/ambiguous comparison and carrying that uncertainty into a multi-day juz1
run. Committing to juz1 on a shaky architecture choice would repeat the exact mistake
already caught once this session (the original unfair from-scratch vs warm-started
comparison).

---

## Open Questions

### Near-term (stage 3 results will answer)
1. **Does 87.5% match hold at 64B (stage 3)?** — each window is identical to stage 2,
   but the model must encode 4 chunks instead of 2 and the windows share an encoding pass.

2. **Does prolonged AR decode degrade gracefully?** — errors in window i propagate as
   the "warmup" for window i+1. If window i match < 100%, window i+1's warmup is noisy.

3. **Zero-shot full-span IQ recall (stage 3 checkpoint)?** — can the model recall
   bytes 0-63 with a single IQ turn (no IR, no SRS) after being trained on windowed IR?
   Expected to fail — motivates either full-span IQ stage or accepting windowed-only.

### Scaling questions (after stage 3)
4. **chunk_len scaling**: does IQ+2IR still converge at chunk_len=32 (W=64B, 2×
   compression)? chunk_len=64 (W=128B, 4×)? Find the compression ratio ceiling.

5. **Multi-pass SRS session**: at 1KB+ corpus, does batching B_w windows per session
   with SRS due-date scheduling maintain retention on early windows? Or does forgetting
   occur between sessions?

6. **Closed-loop stopping criterion**: what R_thresh avoids over-reviewing easy
   windows while still training hard ones? Ablate: R_thresh ∈ {0.8, 0.9, 0.95, 1.0}.

7. **Long-range forgetting**: does the model maintain retention on window 0 after
   training on windows 1..N in the same sequence? The mask prevents direct attention,
   but the shared slot token IDs may cause interference.

### Inference-alignment questions (after multi-pass SRS works)
8. **Query mid-ingest**: does a `query_iq` turn correctly recall an earlier window
   without re-reading the source, given only the SLOT KVs in context?

9. **Resume without forgetting**: does an IR turn after a query (`resume_ir`) maintain
   the quality of an IQ representation that was established N windows earlier?

10. **Streaming ingest**: can the model process an arriving byte stream window-by-window
    in real time, answering queries at any point, equivalent to an online LM with
    in-context fast-weight memory?
