# kvmem

Fast-weight language model — HashMemNet (HMN). **Current focus: dual-attention-block architecture (no MLP) + RMSNorm**, discovered this session to dramatically outperform the standard attn+ffn architecture on the proven memorization/recall task. Older architectures (HMN v3, early chunk-memorization iterations, the feedback-only 32B primitive) are archived — see `docs/EARLY_ARCHITECTURE_HISTORY.md`.

**Convention — always include traj_mix / per-span table when launching or describing a run:**
Eval output rows appear in a fixed order with no labels. Without a table it's impossible to match e.g. `val/srs/span(6, 8)/MEAN = 26%` to the right window. Always show step/val/test per checkpoint.

**Read these docs to resume:**

| Priority | Doc | Why |
|----------|-----|-----|
| 1 | [`docs/SRS_RECIPE.md`](docs/SRS_RECIPE.md) | **Everything current**: stitching vs atomic, RMSNorm discovery, architecture comparison, chain memory design, `juz1.txt` scaling design |
| 2 | [`docs/MDL_MODEL_SIZE.md`](docs/MDL_MODEL_SIZE.md) | **Model size theory**: MDL analysis, why parameter count should track algorithm complexity, not corpus length |
| 3 | [`docs/EARLY_ARCHITECTURE_HISTORY.md`](docs/EARLY_ARCHITECTURE_HISTORY.md) | Archived: HMN v3, early chunk-memorization v1-v5 saga, old bug history, generalization-axes planning doc |
| 4 | [`docs/FEEDBACK_RESULTS.md`](docs/FEEDBACK_RESULTS.md) | Chat-tags experiment arc (window-tag addressing fixes, wrong-token-weighted loss) — superseded by dual-attn but the addressing/loss-shaping lessons still apply |
| 5 | [`docs/BOOK.md`](docs/BOOK.md) § 8 | HMN v3 architecture reference (predecessor) |
| 6 | [`docs/kv_dims.md`](docs/kv_dims.md) | KV capacity math, model size, SRS multi-sequence design |

---

## Architecture in plain terms

**The task**: memorize a byte sequence, then recall it from a short seed (warmup), byte-exact.

**One recall unit** (the proven primitive): `IQ` (initial slot-compression) + `IR1`/`IR2` (argmax-feedback refinement):
```
[source bytes] [SLOT×n] [warmup: 8 bytes] → [output: 24 bytes]
then refine 2× using the model's own previous output as feedback (IR turns)
```

**Scaling via overlapping windows**: split source into `chunk_len=16` chunks. Each chunk gets its own local SLOT encoding. Run the 32-byte recall unit on overlapping windows of 2 chunks each (`window=32B`, `stride=16B`, 50% overlap):
```
window A (0,2): recall bytes  0-31   (chunks 0+1)
window B (1,3): recall bytes 16-47   (chunks 1+2)  ← 16B overlap with A
window C (2,4): recall bytes 32-63   (chunks 2+3)  ← 16B overlap with B
```
- **`nc`** = number of chunks. nc=4 → 64B, nc=8 → 128B.
- **`nochain` masking**: every window's SLOT is architecturally blocked from every other window's content (no shared attention path) — windows are fully independent units.
- **Stitching (how full-sequence recall works despite that independence)**: at decode time, window `i+1`'s warmup is seeded from window `i`'s own *just-decoded byte output* (not any internal representation) — a plain byte-level copy, valid because `warmup_len=8` always fits inside the 16B overlap. Only the very first warmup is seeded from ground truth. This is a genuine **zero-shot generalization**: training always uses teacher-forced ground-truth warmup; the chained/decoded-warmup mechanism only exists at eval time and works because each window's own recall accuracy is high enough that decoded ≈ ground truth.
- **Only 2 forward passes needed per training step, regardless of window count**: causal masking computes every position in parallel in one pass; a second pass fills in `argmax` feedback slots (all of them, across all windows, at once) and finalizes. See `docs/SRS_RECIPE.md` for the full mechanics.

---

## Current Status — dual-attn + RMSNorm confirmed as the working architecture

**Full 2×2 matched-depth (260k-step) comparison** (standard attn+ffn vs dual-attn attn+attn, LayerNorm vs RMSNorm):

| | Standard, LayerNorm | Standard, RMSNorm | Dual-attn, RMSNorm |
|---|---|---|---|
| IQ stage | val 100.0% / test 89.3% | val 100.0% / test 53.6% | val 100.0% / test 100.0% |
| IR stage | val 33.3% / test 23.2% | val 40.5% / test 30.4% | **val 94.0% / test 94.6%** |

Both standard-arch variants trail dual-attn+RMSNorm by ~3-4x at IR, regardless of norm choice — the argmax-feedback refinement mechanism itself generalizes far better in the no-MLP architecture at matched training depth. This is the empirical basis for treating **dual-attn + RMSNorm as the working architecture** going forward. Deferred (not blockers): dual-attn+LayerNorm cell, second-seed reproducibility, real-text validation.

**Currently running**: `dualattn_nc8_slot8_ir` (128B, dual-attn+RMSNorm, warm-started from the 94.0%/94.6% nc4 checkpoint) — testing whether RMSNorm resolves the "window G stuck" failure mode that blocked the standard-architecture nc8 attempt. Two bugs found and fixed before this launch stayed clean: (1) warm-start logic silently dropped growing tensors instead of prefix-copying; (2) MPS OOM crash from `chunk_attn` never being wired into `DualAttnModel` (now fixed, `chunk_attn=256, B=3`). Progress table and full incident writeups in `docs/SRS_RECIPE.md`.

**Architecture decisions made this session** (see `docs/SRS_RECIPE.md` for full reasoning on each):
- Keep `DualAttnBlock`'s paired structure (2 attn/block × 4 blocks) rather than flattening to 8 single-attn blocks — mathematically identical network, but flattening would break existing checkpoints for zero gain.
- `rmsnorm=True` reversed an earlier *negative* small-scale finding (`configs/hmn_chunk_abl_rmsnorm.py`) — likely because that test never retuned LR for the norm change. Do not re-litigate that old result as a reason to avoid RMSNorm.
- `grad_checkpoint` implemented for `DualAttnModel` (two granularities), predicted net *slower* on this MPS hardware (checkpointing always adds recompute; the payoff requires batch-size headroom MPS doesn't have) — queued as an empirical speed test, not assumed.

**Next in queue** (never two jobs at once): resolve window G → grad-checkpoint speed test → `juz1.txt` streaming-loop implementation (design complete, not yet built — see `docs/SRS_RECIPE.md § Concrete juz1.txt scaling design`) → chain-memory implementation (new architecture design, see `docs/SRS_RECIPE.md § Chain memory` — bounded, persistent, resumable `CHAIN_SLOT` state, a real departure from today's fully-independent-window design).

**`juz1.txt` (44,443 bytes) scaling — designed, not yet implemented.** Padded target `n_chunks=4096` (true power of 2, unifies every hyperparameter). The current packed-whole-schedule design cannot reach this scale (mask would need ~10¹¹ entries) — fix is local fixed-size blocks chained via the already-proven warmup-seeded stitching mechanism, never one global mask. Full design, cost estimates, and a hybrid random-offset sampler that reproduces SRS's provable-coverage/spaced-review properties without requiring sequential training order: `docs/SRS_RECIPE.md`.

---

## Key Principles

- `null_kv=True` always — 1.5–2× faster, better bpb
- Mask convention: `0.0`=attend, `-1e9`=blocked (additive bias for `F.scaled_dot_product_attention`)
- IQ stage before IR — always required for feedback arch, including dual-attn
- `nochain` masking (every window's SLOT blocked from every other window's content) is what makes windows independently trainable AND what makes zero-shot stitching work — do not weaken this without understanding both consequences
- **`rmsnorm=True` matters, but is not architecture-agnostic**: dramatic win for dual-attn, genuinely mixed for the standard architecture (better at IQ with LayerNorm, better at IR with RMSNorm) — do not assume one norm choice transfers cleanly across architectures
- **Report precisely, never round up.** "test=100%" must mean 100% of whatever was actually tested (e.g. a padded/truncated excerpt), not the whole file. This has mattered concretely more than once this project (`load_chunks_padded` truncation, the from-scratch/warm-started comparison confound, the depth-vs-norm-choice confound) — always check what's actually been measured before stating a result.
- **Verify infra before trusting it mid-run**: `ScheduleWakeup`'s delay is not reliable (one incident: ~4hr late, during which a training run OOM-crashed undetected) — treat `Monitor` task-notifications as the primary signal, always check process liveness (`ps -p <pid>`) explicitly, not just log content, since a silently-exited process produces no new log lines.

### Model size vs task — MDL principle (see [`docs/MDL_MODEL_SIZE.md`](docs/MDL_MODEL_SIZE.md))

- **Parameter count scales with algorithm complexity, not sequence length.** The same ~166-232k model should handle 64B, 128B, and (once the streaming loop exists) `juz1`'s 44KB — the per-chunk encoding algorithm is identical at all scales.
- **Position-dependent encoding = longer MDL description.** Training at multiple scales penalizes position-dependent solutions (higher description length) and forces the model toward position-invariant ones (shorter description) — this is why every architecture here holds `d`/`n_layers` fixed across scale changes rather than growing the model.
- **If a run stalls: do not add parameters first.** Correct order: (1) broaden training distribution, (2) simplify algorithm, (3) increase model size only as last resort. This order held throughout the session — RMSNorm (a training-dynamics fix) and staged warm-starting (a distribution/curriculum fix) resolved problems that looked like they might need more capacity.
- **Dataset is infinite random bytes** (val) — classical overfitting analysis does not apply there. Real text (test, and eventually `juz1`) is the actual generalization test.

---

## Docs

| What | Where |
|------|-------|
| **Everything current**: RMSNorm discovery, architecture comparison, stitching, chain memory, `juz1` design | [`docs/SRS_RECIPE.md`](docs/SRS_RECIPE.md) |
| Archived pre-dual-attn architecture history | [`docs/EARLY_ARCHITECTURE_HISTORY.md`](docs/EARLY_ARCHITECTURE_HISTORY.md) |
| Chat-tags experiment (window-tag addressing, wrong-token-weighted loss) | [`docs/FEEDBACK_RESULTS.md`](docs/FEEDBACK_RESULTS.md), [`experiments/chat_tags/`](experiments/chat_tags/) |
| Dual-attn ablation (current architecture) | [`experiments/attn_dual/`](experiments/attn_dual/) — `model.py`, `train.py`, `decode.py`, `configs/` |
| True SRS / stitched training (current architecture) | [`experiments/srs_tagged/`](experiments/srs_tagged/) — `train.py`, `stitch_decode.py`, `configs/` |
| Model size theory | [`docs/MDL_MODEL_SIZE.md`](docs/MDL_MODEL_SIZE.md) |
| HMN v3 reference book (predecessor) | [`docs/BOOK.md`](docs/BOOK.md) |
| KV capacity + SRS trajectories | [`docs/kv_dims.md`](docs/kv_dims.md) |
| Chunk SRS / local-refine training (legacy, still imported by current code) | [`kvmem/train_hmn_chunk.py`](kvmem/train_hmn_chunk.py) |
| Feedback training (32B primitive) | [`kvmem/train_hmn_feedback.py`](kvmem/train_hmn_feedback.py) |
| HMN v3 mono training (legacy) | [`kvmem/train_hmn_mono.py`](kvmem/train_hmn_mono.py) |
| Test set | [`datasets/suratalfatihah.txt`](datasets/suratalfatihah.txt) |
| `juz1` scaling target (not yet used in training) | [`datasets/juz1.txt`](datasets/juz1.txt) |
