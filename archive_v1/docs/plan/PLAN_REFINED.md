# KV Memory — Refined Plan

Built on top of PLAN_STAGE1.md. Incorporates what was learned from experiments and
proposes a cleaner sequencing of what to try next.

---

## What we know works (May 2026)

- **Single-pass KV recall at seg_len=8**: 100% on held-out deterministic sequences.
- **Scales to seg_len=128 and seg_len=576**: 100% with YaRN + tag scheme `<m></m>`.
- **Slot identity is redundant**: `zeros` slots (YaRN position alone) achieves same 100%.
- **Full suratalfatihah recall**: 562 bytes, 100% CER=0.000 with full AR decode from x_S[0].
- **Tag scheme** `<m>` / `</m>` (3+4 byte markers, no byte restrictions): works.
- **Random-window training** (mid-sequence recall): under test, not confirmed yet.

---

## Architecture as of now

```
[x_S | <m> | slots (N) | </m> | Y]

x_S  : source bytes, any value [0x00, 0xFF]
<m>  : [0x3C, 0x6D, 0x3E]                3 bytes
slots: N × 0x00 (zeros) OR N × (i%256)   N bytes
</m> : [0x3C, 0x2F, 0x6D, 0x3E]          4 bytes
Y    : AR recall target (warmup + output)

L = len(x_S) + 7 + N + len(Y)
```

Positional encoding: **YaRN** (NTK-aware scaled RoPE). Mask: Y cannot see x_S or `<m>`.

---

## Open question: random-window recall

**Status**: training at seg=128, chunk=32 right now (step ~4000/10000).

If this works (generalizes to full-sequence recall on held-out test sequences), then:
- The model learns to recall **any window** of x_S given warmup = x_S[y_start-1]
- Chunked inference works: chain decoded chunks for any-length sequence
- Suratalfatihah can be recalled chunk-by-chunk without 576-token AR

If it fails: full AR decode from x_S[0] remains the working approach.

---

## Two competing design directions

### Direction A: Sequential Continual Memory (new, user-proposed)

Multiple memory blocks, each encoding a corpus chunk, chained sequentially:

```
corpus_1 → <m> slots_1 </m> → corpus_2 → <m> slots_2 </m> → ... → recall
```

Or with interleaved recall windows:

```
corpus_1 → <m> s1 </m> → recall_w1 → corpus_2 → <m> s2 </m> → recall_w2
```

Or with memory accumulation (each new `<m></m>` builds on prior KV state):

```
corpus_1 → <m> s1 </m> → corpus_2 → <m> s2 </m> → recall_w → <m> s3 </m>
```

**What this adds**:
- The model reads multiple source chunks sequentially, accumulating memory across `<m></m>` blocks.
- Recall can query any prior window, not just the current one.
- Incremental ingestion: arbitrary corpus length via chaining, no fixed seg_len limit.

**Prerequisite**: random-window recall must work (the model must recall mid-sequence windows,
not just from x_S[0]).

**Fits current architecture**: `<m></m>` tags are already arbitrary-length markers in the
existing 256-token vocab. No architecture change needed — just longer sequences and
a training regime that includes multiple `<m></m>` blocks per example.

**Compatible with Stage 1 self-correction**: yes. Each `<m></m>` block is a write step;
multiple passes = multiple `<m></m>` blocks. The focused loss from PLAN_STAGE1 naturally
applies: pass 2's `<m></m>` block can be trained to correct positions where pass 1 failed.
Stage 1 loss-only refinement is essentially the 2-block version of Direction A.

---

### Direction B: Stage 1 Self-Correction (existing plan)

Train T sequential memory passes over the same source, with hard-position focused loss:

```
x_S → <m> s_1 </m> → x_S → <m> s_2 </m> → ... → <m> s_T </m> → Y
```

Each pass tries to fix what the previous pass missed. Training loss upweights positions
with high NLL in pass t-1 for pass t's loss.

**What this adds**:
- Better recall quality per byte of source.
- Error correction amortized into weights — no architecture change.
- Pass 1 works at inference for speed; more passes for quality.

**Not yet started** — waiting for Stage 0 recall to work at longer lengths.

---

## Decision: which to try first?

**Recommendation: Direction A (sequential continual memory) before Direction B (self-correction).**

Reasoning:
1. **Direction A is a natural extension of what already works.** The tag scheme and
   random-window recall (if it works) directly enable chaining. Stage 1 requires a
   working Stage 0 as prerequisite — we have that now.

2. **Direction A is prerequisite for real use.** Self-correction (Stage 1) only helps
   if the model can already ingest multiple corpus chunks sequentially. A model that
   can correct one 576-byte pass but can't chain multiple chunks is still limited.

3. **Direction A and B are composable, not competing.** Once chaining works, Stage 1
   self-correction can be applied to each `<m></m>` block independently, using the
   same focused loss from PLAN_STAGE1.

4. **The ideal case for Direction A is still 100% recall.** Stage 1 is the fallback for
   when Direction A doesn't achieve 100% on its own. Try Direction A at 100% first;
   add Stage 1 error correction only if needed.

---

## Proposed sequence

### Step 0 (current): Confirm random-window recall
- Experiment: seg=128, chunk=32, YaRN, zeros slots, 10k steps (~30min)
- Pass criterion: test sequences at seg=128 recall ≥ 95% with chained chunk decode
- If passes → Direction A is viable

### Step 1: Two-chunk sequential memory
- Source: two random 128-byte chunks, two `<m></m>` blocks
- Format: `[c1 | <m> s1 </m> | c2 | <m> s2 </m> | recall_window]`
- Training: recall window is a random window from c1 OR c2 (model must learn which chunk to read)
- Eval: can the model recall from the correct chunk given a warmup token?

### Step 2: Suratalfatihah via chained chunks
- Split 562 bytes into 4×128 + 1×66 chunks
- Train: `[c1 | <m> s1 </m> | c2 | <m> s2 </m> | ... | recall_w]`
- Eval: perfect recall of full surah via chained chunk decoding

### Step 3: Arbitrary corpus length
- Variable number of chunks (2–16)
- YaRN handles length extrapolation
- Eval: can the model recall any window from any chunk?

### Step 4 (Stage 1, if needed): Self-correction
- Apply focused loss (PLAN_STAGE1 §2.6) within each `<m></m>` block
- Each block gets multiple passes, later passes upweight earlier errors
- Enables >100% fidelity per chunk without architectural change

---

## Explicit Role-Tag Scheme (if warmup-anchor fails)

If windowed recall still fails with 16-byte warmup, try semantic role tags:

```
<s> x_S </s> <m> slots </m> <f> warmup_bytes </f> <c> continuation </c>
```

- `<s>` / `</s>` — wraps source (replaces bare x_S prefix)
- `<m>` / `</m>` — wraps KV slots (existing)
- `<f>` / `</f>` — "from": explicit anchor bytes, tells model WHERE in source to start
- `<c>` / `</c>` — "continue": model outputs here

**Why this works better than bare warmup bytes:**
The current scheme relies on the model inferring "these warmup bytes are a position hint" — implicit. With `<f>`, the role is explicit: "find where `<f>...</f>` appears in `<s>...</s>`, then continue from there into `<c>`." The model can learn to attend the `<f>` content against the source KV to locate position, rather than relying on YaRN alone.

**Implementation:** new tag bytes (e.g., `<s>` = `[0x3C, 0x73, 0x3E]`, `</s>` = `[0x3C, 0x2F, 0x73, 0x3E]`, etc.), new mask rules where `<c>` region can see `<f>` region and `<m>` slots but NOT `<s>` source directly.

---

## What Stage 1 plan defers

From PLAN_STAGE1.md, the following are still valid and unchanged:
- Hard-position focused loss (§2.3-2.4) — implement at Step 4
- Monotonicity regularizer (§2.5) — implement at Step 4
- Probe-based outer loop (§3.2) — implement at streaming stage
- Self-evaluation head (Stage 4 of main doc) — replaced by direct probing ✓

The key realization from PLAN_STAGE1: "error correction behavior can be amortized
into trained weights" is correct and remains the design. The refinement: chaining
multiple `<m></m>` blocks IS the multi-pass mechanism, and the focused loss from
Stage 1 is just a training-time loss applied to each block in the chain.

---

## Summary table

| Step | What | Prerequisite | Status |
|------|------|-------------|--------|
| 0 | Random-window recall (seg=128, chunk=32) | — | 🔄 running |
| 1 | Two-chunk sequential memory | Step 0 passes | queued |
| 2 | Full suratalfatihah via chained chunks | Step 1 | queued |
| 3 | Arbitrary corpus length | Step 2 | queued |
| 4 | Self-correction (Stage 1 focused loss) | Step 3 | deferred |

The suratalfatihah 100% full-sequence recall (via single AR decode from x_S[0]) is
already achieved. The remaining goal is arbitrary-window recall and multi-chunk chaining.
