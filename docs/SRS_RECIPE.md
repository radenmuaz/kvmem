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
| W | window size (bytes), fixed at 32B |
| s | stride (bytes), fixed at 16B (50% overlap) |
| C | chunk_len (bytes), 16B — window and stride land on chunk boundaries |
| n | number of review passes (IR turns) for a window |
| R | retention: AR-decode match% (0–1) on a window |
| B | BPB: NLL/ln(2) on the window's output under teacher forcing |
| S | stability: how long a window stays above retention threshold after review |
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

## Current Experiments

| Stage | Src | Windows | Result |
|-------|-----|---------|--------|
| 1 | 32B | 1, n_refine=0 | 81.9% match (IQ only) |
| 2 | 32B | 1, n_refine=2 | **87.5% match** |
| 3 | 64B | 3, n_refine=2 | **in progress** — first SRS scaling experiment |

Stage 3 is the first test of whether the per-window IQ+IR unit composes across
overlapping windows with the prolonged AR eval protocol.

---

## Open Questions

1. **Does 87.5% match hold at 64B (stage 3)?** — each window is identical to stage 2,
   but the model must encode 4 chunks instead of 2 and the windows share an encoding pass.

2. **Does prolonged AR decode degrade gracefully?** — errors in window i propagate as
   the "warmup" for window i+1. If window i match < 100%, window i+1's warmup is noisy.

3. **Zero-shot full-span IQ recall (stage 3 checkpoint)?** — can the model recall
   bytes 0-63 with a single IQ turn (no IR, no SRS) after being trained on windowed IR?
   Expected to fail — motivates either full-span IQ stage or accepting windowed-only.

4. **Closed-loop stopping criterion?** — what R_thresh avoids over-reviewing easy
   windows while still training hard ones? Ablate: R_thresh ∈ {0.8, 0.9, 0.95, 1.0}.

5. **Long-range forgetting?** — does the model maintain retention on window 0 after
   training on windows 1..N? SRS scheduling designed to prevent this, but not yet tested.
