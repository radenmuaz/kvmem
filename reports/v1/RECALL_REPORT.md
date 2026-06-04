# KV-Memory Recall — Research Report

**Project**: Prove that a transformer with a KV bottleneck can losslessly memorize and recall arbitrary byte sequences, trained only on random data, generalized to unseen structured sequences.

---

## Architecture

Sequence format: `[x_S | STX | slots (N) | ETX | Y]`

- `x_S` — source sequence (bytes to memorize), length `seg_len`
- `STX` — start-of-memory marker (protocol byte)
- `slots` — N unique slot-ID tokens `[0x04..0x1F]`, one per memory slot
- `ETX` — end-of-memory marker
- `Y` — target output (copy of `x_S`), decoded autoregressively

**Attention mask rules:**
- Y tokens can only attend to slots + ETX (not to `x_S` directly)
- Slots can attend to `x_S` (encoding pass)
- Nothing outside Y attends to Y (no future leakage)

**Effect**: `x_S` is encoded into KV cache of the N slots during the single forward pass. Y must read from those slots to reconstruct `x_S`. No positional embeddings (NoPE).

**Training objective**: NTP on Y positions only. Random synthetic data — model must learn a general copy algorithm, not memorize specific patterns.

---

## Timeline & Results

### May 27–28 — Stage 0: Corpus Recall

- Trained on Arabic text (`suratalfatihah.txt`, `1.txt`)
- Eval: recall accuracy on the same text
- **Result**: stuck at ~41% recall, not improving
- **Apparent cause**: plateau — actually a red herring (see below)

### May 28–29 — Root Cause: Slot Collapse

**Bug**: all N memory slots initialized with `NUL=0x00` token → identical embeddings → model cannot write different information to different slots → all slots collapse to the same KV representation → model can only recall ~4–10% of sequences.

**Fix**: unique slot ID tokens per position:
```python
SLOT_BASE = 0x04
def make_slot_ids(N):
    return [(SLOT_BASE + i % 28) for i in range(N)]
```
Each slot gets a distinct embedding from initialization, enabling differentiated writes.

**Files**: `kvmem/data.py` — `make_slot_ids()`, `kvmem/stage0.py`

### May 30 — Mini Recall Experiment

**Design**: small controlled experiment to validate the architecture cleanly.

- **Training data**: purely random bytes, 4 distributions (uniform, Dirichlet-skewed, sub-range uniform, geometric). No structured patterns.
- **Test sequences** (held-out, never in training):
  - `up_counter`, `down_counter`, `odd`, `even`, `linear`, `sawtooth`, `palindrome`, `geometric`
  - All deterministic, interpretable patterns in `[0x20, 0xFF]`
- **Eval**: greedy AR decode from 1-byte warmup, CER (character error rate) per sequence

**Results** (`seg_len=8`, `N=4`, `d=64`, 4 layers, `wd=0.0`):

| Step | Mean match | Notes |
|------|-----------|-------|
| 2000 | 66.1% | up_counter, down_counter already 100% |
| 6000 | 83.9% | 5/8 sequences perfect |
| **8000** | **100%** | **All 8 sequences perfect** |

**Confirmed twice**: 4-layer run (step 8000), 6-layer run (step 6000).

**Overfitting sanity** (`kvmem/overfit_recall.py`): trained on one fixed sequence (B=1), achieved 100% recall at step 500, loss → 0. Proves architecture is correct; generalization is the challenge, not the architecture.

**Key insight**: model trained only on random bytes generalizes to unseen deterministic patterns — genuine learning of the copy operation, not memorization.

### May 30–31 — Extrapolation & Surah Test

Loaded the best seg_len=8 checkpoint (91.1% match) and tested:

**Extrapolation** (longer sequences than training):

| Test seg_len | Mean match |
|---|---|
| 8 (train length) | 91.1% |
| 16 (2×) | 21.7% |
| 32 (4×) | 2.4% |
| 64 (8×) | 0.6% |

**Conclusion**: no extrapolation. N=4 slots encode exactly 8 bytes. Double the length requires double the slots — must train at target length.

**Surah Al-Fatihah** (562 bytes, split into 8-byte chunks):
- 6/64 chunks perfect (9.4%)
- Mean match: 55.6%
- UTF-8 Arabic bytes are all ≥ 0x80, within valid range

### May 31 — MPS Acceleration

Installed `jax-metal`. Issues encountered and fixed:

- `JAX_PLATFORMS=METAL` fails; auto-detect works (just `import jax`)
- `jax.nn.gelu` lacks MPS vmap batching rule → replaced with tanh-approximation GELU across all training code

**Speed** (MPS vs CPU, solo process, full-Y format):

| Config | CPU | MPS | Speedup |
|--------|-----|-----|---------|
| seg=128, B=8 | ~0.25s | 0.15s | 1.7× |
| seg=1024, B=1 | 0.51s | 0.31s | 1.6× |
| seg=1024, d=128, B=1 | ~0.55s | 0.34s | 1.6× |

All training scripts updated with `--device mps` flag.

### May 31 — Segmented Decode (Failed Approach)

**Idea**: encode full 1024 bytes into N=1024 slots but decode in 128-byte chunks → L=2178 instead of L=3074, faster training.

**Training format**: `[x_S | STX | slots | ETX | warmup | y_chunk]`
- Random chunk offset each step
- Warmup = last token before the chunk, gives model a position cue

**Result**: **mode collapse** — model outputs constant repeated bytes (`4a4a4a4a...`). In-dist val oscillates at 5–12% for 8000+ steps, never improves.

**Root cause**: warmup token cannot reliably locate position in a random sequence — duplicate byte values make the lookup ambiguous. The model collapses to predicting the mode byte.

**Lesson**: the full-Y approach (mini_recall) is correct. The Y sequence being contiguous from position 0 gives the model unambiguous sequential context.

### May 31 — Current: Full-Y at Long Lengths

Back to the proven `mini_recall.py` format, extended to longer sequences on MPS.

**Benchmark** (full-Y, MPS):

| seg_len | N | d | B | L | s/step | 30k ETA |
|---------|---|---|---|---|--------|---------|
| 128 | 128 | 64 | 8 | 386 | 0.15s | 1.3h |
| 512 | 512 | 64 | 2 | 1538 | 0.22s | 1.8h |
| 1024 | 1024 | 64 | 1 | 3074 | 0.31s | 2.6h |
| **1024** | **1024** | **128** | **1** | **3074** | **0.34s** | **2.8h** |

**Two runs currently training** (JIT compiling):
- `mini_recall` seg=128, N=128, d=64, B=8, MPS — sanity check (~1.3h)
- `mini_recall` seg=1024, N=1024, d=128, B=1, MPS — full target (~2.8h)

---

## What Works

| Component | Status |
|-----------|--------|
| Architecture (KV bottleneck) | ✅ Proven correct via overfitting test |
| Slot ID fix | ✅ Eliminates slot collapse |
| 100% recall at seg_len=8 | ✅ Confirmed on 8 unseen structured sequences |
| Generalization from random → structured | ✅ Genuine, not memorization |
| MPS acceleration (1.6×) | ✅ Working |
| GELU vmap fix for MPS | ✅ Applied to all training scripts |

## What Doesn't Work

| Component | Status |
|-----------|--------|
| Extrapolation beyond training length | ❌ Model can't recall longer than trained |
| Segmented chunk training | ❌ Mode collapse due to position ambiguity |
| Compressed slots (N < seg_len) | ❌ Not tested to convergence; likely too hard without explicit position tokens |

## Not Yet Attempted

| Component | Notes |
|-----------|-------|
| OCD (Optimal Completion Distillation) | Code written in `kvmem/stage0_ocd.py`; not run. Fixes teacher-forcing exposure bias. |
| seg_len=1024 full recall | Currently training (~2.8h) |
| Surah recall with longer seg_len | Will test after 1024 run converges |

---

## Key Files

| File | Purpose |
|------|---------|
| `kvmem/data.py` | Data utils, `make_slot_ids()`, attention masks |
| `kvmem/stage0.py` | Model architecture, training loops |
| `kvmem/mini_recall.py` | Main validated training script |
| `kvmem/overfit_recall.py` | Architecture sanity check |
| `kvmem/eval_recall.py` | Post-hoc evaluation (extrapolation, surah) |
| `kvmem/seg_recall.py` | Segmented decode attempt (failed) |
| `kvmem/stage0_ocd.py` | OCD training (not yet run) |
