# Multi-Turn Corpus Recall — Experiment Plan

**Gate rule:** only advance when current stage passes. Stop and diagnose before moving on.

---

## Stage 1 — Two blocks, recall from recent  (warm-up)

```
2x<h:1><x:16><z:7><q:4><y:8,from=1>
```

`h_1` has absorbed both chunks. Query targets chunk_1 (recency advantage). If this fails, the `<h>` update step itself is broken and must be fixed before proceeding.

**Config:** `configs/ablate_2b_recent.py`  
**Steps:** 80k  
**Pass:** ≥90% match  
**Fail means:** cross-`<h>` attention update is broken

---

## Stage 2 — Two blocks, recall from earlier  (key test)

```
2x<h:1><x:16><z:7><q:4><y:8,from=0>
```

After seeing chunk_1, the model must still recall chunk_0 from `h_1`. Tests whether the fast-weight update propagates and retains early information.

**Config:** `configs/ablate_2b_old.py`  
**Steps:** 80k  
**Pass:** ≥80% match  
**Fail means:** fast-weight update is lossy for non-recent chunks — consider line numbers or larger slot_len

---

## Stage 3 — Two blocks, content-addressed routing  (main test)

Mix Stage 1 and Stage 2 in same training pool. Model receives random `<q>` warmups from either chunk. Must route to the correct block using anchor content alone, no positional hint.

**Config:** `configs/ablate_2b_mixed.py`  
**Steps:** 160k  
**Pass:** ≥80% on both from=0 and from=1 simultaneously  
**Fail means:** `<h>` cannot content-address two distinct chunks simultaneously — routing is ambiguous

---

## After Stage 3: proceed to Refine Experiment

Do not scale to N>2 blocks until refinement is validated. Refinement tests whether the model can self-correct a failed recall — a prerequisite for robust multi-block retrieval at larger N.

---

# Refine Experiment Plan

**Motivation:** Even a perfect Stage 3 model may fail individual recall attempts. The refine experiment trains the model to detect and correct its own mistakes in a second pass — using the self-correction architecture.

## Setup

Minimal sequence change: two `<q><y>` pairs over the same `<h>`.

```
<x>src</x><z>z</z><h>h</h>
<q>anchor</q><y>attempt_1 [corrupted]</y>
<q>anchor</q><y>attempt_2 [ground truth]</y>
```

- `attempt_1` = corrupted ground truth (denoising: uniform substitution, rate p ~ U(0.05, 0.4))
- `attempt_2` = ground truth
- **Loss: only on `attempt_2`**  — model must correct from `attempt_1`

`attempt_2`'s `<q>` can attend to `attempt_1` causally — it reads the first (noisy) attempt and learns to fix it.

---

## Refine Experiment — Multi-Turn Correction

### Stopping mechanism

After each `</y>`, the model predicts whether another turn follows — this IS part of the NTP loss:
```
<y>attempt_K</y> <q>   ← model predicts <q> → not done, continue
<y>attempt_N</y> [EOS] ← model predicts EOS → done, stop
```

At inference: run until model does not predict `<q>` after `</y>`.

### Sequence structure

Min 2 turns (always), max 3-4:
```
<x>src</x><z>z</z><h>h</h>
<q>anchor</q><y>y_1 [p_high corrupt]</y> <q>   ← always present
<q>anchor</q><y>y_2 [p_low corrupt]</y>  <q>   ← 3 turns if y_1 badly wrong
<q>anchor</q><y>y_3 [ground truth]</y>  [EOS]  ← loss here
```

Minimum 2 enforced so model always experiences both "imperfect state" and "correction" within every training example.

### Noise schedule per turn (descending)

```
turn 1:  p ~ U(0.05, 0.8)    ← varied, diverse errors
turn 2:  p ~ U(0.00, 0.3)    ← closer to correct
turn 3:  p = 0               ← ground truth (or U(0.0, 0.1))
turn 4:  p = 0               ← ground truth (max turns always correct)
```

Number of turns sampled per example: 25% → 2 turns, 50% → 3 turns, 25% → 4 turns.

### Loss

- Intermediate `<y>_1..<y>_{N-1}` positions: **no loss** (input context)
- Final `<y>_N` positions: **NTP loss** (ground truth target)
- `<q>` continuation tokens between turns: **NTP loss** (stopping signal)

### Stages

**Stage A — Denoising (cheap, 1× compute)**
- y_1 = uniform substitution at rate p
- Baseline: does correction improve over y_1?
- Config: `configs/refine_denoise.py`, steps=80k

**Stage B — Parallel sample (systematic errors, 2× compute)**
- y_1 = argmax one forward pass (real model uncertainty)
- 70% parallel sample + 30% teacher-forced (teaches "no correction needed" case)
- Config: `configs/refine_parallel.py`, steps=80k

### Metrics

| Metric | Meaning |
|--------|---------|
| `y_1` match% | quality of first (corrupted) attempt |
| `y_final` match% | quality of last attempt |
| Delta = final - y_1 | improvement from correction |
| Stop accuracy | does model predict EOS after correct attempt? |

**Primary signal:** positive Delta — any improvement confirms the mechanism.  
**Secondary signal:** stop accuracy > 80% — model knows when it's done.

### Implementation notes

- Extend `make_multi_batch` to support multiple `<q><y>` pairs with variable turn counts
- Corruption applied in numpy (zero extra compute for Stage A)
- Stage B: one extra forward pass per batch step
- EOS / continuation signal: needs one new tag token (e.g. `<ok>` ID 266+slot_len+intermed_len)
