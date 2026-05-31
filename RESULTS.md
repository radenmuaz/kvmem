# KV Memory — Experimental Results

All experiments use a decoder-only transformer with KV bottleneck.
Sequence format: `[x_S | <m> | slots (N) | </m> | Y]`
Tags: `<m>` = `[0x3C, 0x6D, 0x3E]`, `</m>` = `[0x3C, 0x2F, 0x6D, 0x3E]`
Positional encoding: YaRN (NTK-aware scaled RoPE).
Training data: purely random bytes, 4 distributions. No real text in training.

---

## 1. Recall Results

### 1.1 Slot ID Ablation (seg=128, YaRN, d=64, 4L, B=8, 10k steps)

| Slot style | Tokens in `<m>...<m>` | Match @ 10k |
|---|---|---|
| `seq` | slot i = i % 256 | **100%** |
| **`zeros`** | all 0x00 | **100%** |

**Finding**: YaRN positional encoding alone is sufficient to route each Y token to the correct KV slot. Slot token identity is redundant — the position encodes the address.

### 1.2 Positional Encoding Ablation (seg=128, d=64, 4L, B=8, 10k steps)

| PE | Match @ 10k | Notes |
|---|---|---|
| NoPE | **0.7% (diverged)** | Loss rose 4.18→4.38, mode collapse |
| RoPE | **100%** | Same as YaRN at training length |
| **YaRN** | **100%** | Reference from here on |

**Finding**: positional encoding is essential at seg=128. NoPE cannot route attention across 128 distinct slots without position signal.

### 1.3 Scaling Recall Length (YaRN, zeros slots, d=64, 4L)

| seg_len | N | Steps to 100% | Training time |
|---|---|---|---|
| 8 | 4 | 8,000 | ~28 min |
| 128 | 128 | 10,000 | ~28 min |
| 576 | 576 | 7,000 | ~135 min |

**Finding**: 100% recall achieved at all tested lengths. Convergence is not strongly sensitive to sequence length.

### 1.4 Suratalfatihah Recall (562 bytes)

Model: seg=576, N=576, YaRN, zeros slots, d=64, 4L, step 7000.

**Full AR decode from x_S[0] (no chunking):**
```
full 576 bytes CER = 0.0000   match = 100.00%
orig 562 bytes CER = 0.0000   match = 100.00%
PERFECT: no mismatches!
```

**Finding**: 562-byte Arabic UTF-8 text recalled perfectly. Model trained on random bytes only.

**Chunked decode failure (4×128 + 1×64):**
```
chunk 0 [0:128]    match = 98.4%   ← warmup=x_S[0], matches training
chunk 1 [128:256]  match = 60.2%   ← warmup=x_S[127], never seen in training
chunk 2 [256:384]  match = 38.3%   ← cascading warmup mismatch
chunk 3 [384:512]  match = 42.2%
chunk 4 [512:576]  match = 43.8%
→ mean match = 56.6%
```

Teacher-forced warmup gives identical results — the issue is not chaining errors but the model never being trained to decode from mid-sequence warmup tokens.

---

## 2. Extrapolation Results (zero-shot)

Model: same seg=576 checkpoint (trained for recall only, Y = copy of x_S).

**Setup**: encode full x_S → KV, give warmup = last 4 bytes of x_S, generate 32 bytes beyond x_S.

| Sequence | Match | Observation |
|---|---|---|
| up_counter | 0% | Correct step (+1), wrong phase (offset by ~seg_len bytes) |
| down_counter | 0% | Correct step (−1), wrong phase |
| odd | 25% | Correct step (+2), phase off by ~8 |
| even | 25% | Correct step (+2), phase off by ~8 |
| linear | 25% | Correct step (+4), phase off |
| sawtooth | 25% | Correct step, phase off |
| palindrome | **75%** | Pattern largely correct |
| geometric | **75%** | Pattern largely correct |
| **mean** | **31%** | |

**Key finding**: the model IS encoding the pattern structure, not just positional values. The generated sequences have the correct arithmetic step/ratio but are phase-shifted. The model outputs the pattern as if the warmup bytes are from position ~4 in x_S rather than position 576.

**Why the phase shift**: the warmup bytes `9c 9d 9e 9f` (end of up_counter at seg=576) happen to be identical to bytes from an earlier position in the sequence (due to wrapping at 256). YaRN assigns the Y position its own rotary angle — the model cannot distinguish "warmup is at end of x_S" from "warmup is at mid-x_S" without being trained on extrapolation.

**What this means**: the KV is a semantic encoder of pattern structure, not just a key-value store of (position→value). The model generalizes from random byte training to structured pattern continuation, but needs a position-awareness fix (training with Y = continuation rather than Y = copy) to correct the phase.

**Interpolation** (future work): source = (start, end, ?) tuples; model must fill in the middle. Harder encoding — slot must store start-end-step structure, not just values.

---

## 3. Random-Window Recall (in progress)

Training: seg=128, chunk=32, random y_start, warmup=x_S[y_start-1].
Goal: model learns to recall any 32-byte window of x_S given a 1-byte warmup.

| Step | Loss | Match on test seqs |
|---|---|---|
| 1 | 6.73 | ~1% |
| 4000 | 4.08 | ~2% |
| 9000 | 4.94 | ~2% |

**Final result (10k steps)**: 1.6% match — **failed**. Mode collapse to repeated byte patterns (e.g. `8a8a8a8a...`, `9d9d9d9d...`).

**Conclusion**: warmup = x_S[y_start-1] is ambiguous for random x_S — the same byte value appears at multiple positions. The model has no reliable position anchor and collapses to the mode byte of the distribution.

**Next**: use `seq` slots (i%256) so the unique slot token content provides an additional position cue to disambiguate — then re-test. Alternatively: provide explicit chunk-index token as part of the Y prefix.

---

## 4. Summary

| Capability | Status | Notes |
|---|---|---|
| Single-sequence recall (any length) | ✅ 100% | YaRN + zeros slots |
| Real Arabic text recall (suratalfatihah) | ✅ 100% | 562 bytes, full AR decode |
| Pattern extrapolation (zero-shot) | ⚠️ 31% | Correct structure, wrong phase |
| Random-window recall | ❌ ~2% | Not converging |
| Multi-chunk sequential memory | 🔜 | Requires random-window to work |
| Algorithmic tasks (sort, reverse, etc.) | 🔜 | After multi-chunk |
