# kvmem

Fast-weight language model — HashMemNet (HMN). Two architectures: v3 (MEM blocks) and **Feedback** (argmax loop, current focus).

**Convention — always include traj_mix table when launching or describing a run:**
Eval output rows appear in traj_mix order with no labels. Without the table it's impossible to match `val/ir_local/MEAN = 26%` to the right window/nc. Format:

| weight | windows | nc | SLOT pos | trains |
|---|---|---|---|---|
| 2.0 | [(0,2),(1,3),(2,4)] | 4 | 80/244/408 | stitch 64B |
| 1.0 | [(0,2)] | 2 | 40 | win A independent |
| 0.5 | [(2,4)] | 8 | 160 | win C bridge |

**Read these docs to resume:**

| Priority | Doc | Why |
|----------|-----|-----|
| 1 | [`docs/SRS_RECIPE.md`](docs/SRS_RECIPE.md) | **Vision**: SRS equations, open/closed-loop, corpus ingestion recipe, scaling to backprop LM NLL |
| 2 | [`docs/FEEDBACK_RESULTS.md`](docs/FEEDBACK_RESULTS.md) | **Key result**: feedback arch achieves 100% k=0..12, why it works, bug history |
| 3 | [`docs/BOOK.md`](docs/BOOK.md) § 8 | HMN v3 architecture reference (predecessor) |
| 4 | [`docs/kv_dims.md`](docs/kv_dims.md) | KV capacity math, model size, SRS multi-sequence design |
| 5 | [`docs/MDL_MODEL_SIZE.md`](docs/MDL_MODEL_SIZE.md) | **Model size theory**: MDL analysis, gradient-descent tax, why vlen = MDL regularization |

---

## Architecture in plain terms

**The task**: memorize a byte sequence, then recall it from a short seed (warmup).

**One recall unit** — the proven primitive (`hmn_feedback_32_ir`, 100% on 32 bytes):
```
[source bytes] [SLOT×4] [warmup: 8 bytes] → [output: 24 bytes]
then refine 2× using model's own previous output as feedback (IR turns)
```

**Scaling up**: split source into 16-byte chunks (`chunk_len=16`). Each chunk gets encoded into 4 SLOT tokens. Run the 32-byte recall unit on overlapping windows of 2 chunks each.

```
source = 64 bytes = 4 chunks of 16 bytes     (nc=4)

[chunk0|SLOT×4] [chunk1|SLOT×4] [chunk2|SLOT×4] [chunk3|SLOT×4]
  enc_block[0]    enc_block[1]    enc_block[2]    enc_block[3]
  positions 0-19  positions 20-39 positions 40-59 positions 60-79

window A (0,2): recall bytes  0-31   (chunks 0+1)
window B (1,3): recall bytes 16-47   (chunks 1+2)  ← 16B overlap with A
window C (2,4): recall bytes 32-63   (chunks 2+3)  ← 16B overlap with B
```

- **nc** = number of chunks. nc=2 → 32B, nc=4 → 64B, nc=8 → 128B.
- **enc_block[k]** = the k-th chunk's token region (raw bytes + SLOT tokens).
- **SLOT tokens** = compressed memory for one chunk. The recall IQ turn reads from these (raw bytes are masked out) to reconstruct the window.
- **Stitch decode**: run windows A→B→C in order. Each window's 8-byte warmup comes from the previous window's decoded output — works because warmup_len=8 < stride=16B, so the warmup always falls inside the 16B overlap that the prior window already generated. Only the very first warmup (bytes 0-7) is seeded from ground truth.

**The position problem (current issue)**: in stitch training, windows always appear in sequence — window C's recall block is at token position ~408 (after A and B). The model learned to read enc_block[2,3] SLOTs from that far position. In independent eval (window C alone, right after enc_blocks), the recall block is at position ~80 and enc_block[3]'s SLOTs are only 1-4 positions away — a distance the model has never seen for window C, so it fails. Window B has the same problem but milder (position shift 244→80 vs 408→80).

**Fix in progress**: v5b adds training examples of B and C in isolation. B is improving; C is stuck because enc_block[3] ends up at distance 1 in isolation — too different from distance ~330 in stitch. **vlen** (ready) trains C at multiple source lengths (nc=4/8) so the recall SLOT lands at positions 80 and 160, forcing position-invariant encoding.

---

## Current Status

**Feedback architecture solved the refinement problem. Active scaling approaches:**
1. **ir_local** (local IQ per window, stitch decode) — chaining bug fixed in v5, running
2. **iq_global_rw + IR** (single global IQ, all windows, argmax feedback IR) — **slot8_ir RUNNING**, Win B at 100%
3. **chat-tags** (`experiments/chat_tags/`, isolated from `kvmem/`) — explicit boundary tokens wrapping the `iq_global_rw` layout. **SOLVED — success bar met**: `slot8_tagged_wrongtok_ablation.py` (window-specific `<query_a/b/c>` tags + wrong-token-weighted IR loss, warm-started from Phase B4) **CONVERGED at 97.2% mean, Win A 100%, Win B 100%, Win C 91.7%** — stable across the final 6 consecutive checkpoints, 6/8 val sequences perfect on Win C. Full arc: Phase A→B (confounded, 60.2%) → B2 (oscillating, 70.8%) → B3 (converged 75.9%, Win C stuck 27.8-30.6%, diagnosed root cause: IR turns *degrading* quality from destructive interference — 3 windows sharing one `<query>` key) → B4 (window-specific tags fix the addressing, converged 94.9%, Win C 84.7% — close but short of bar) → **wrong-token-weighted loss ablation (upweight NLL where the fed-back argmax was wrong — one-line change) closes the remaining gap, 91.7% Win C**. Confirms addressing (not capacity/rank — see `docs/SRS_RECIPE.md § Fast-Weight Rank and Addressing`) and loss-shaping (not architecture) were the two real bottlenecks, cheaper to fix than adding model capacity. One reproducible soft spot remains (`down_counter`: IR1 fixes it, IR2 re-breaks it; `palindrome` is the hardest case throughout the whole series) — margin-loss/error-flag-head/attention-supervised ablations (2-4) documented as available follow-ups but not needed to close this queue. Separately: **DenseNet-style depth-wise cross-layer SLOT-KV concatenation** (`experiments/densenet_kv/`) built and run — architecture verified correct, but ablation vs B4 was **inconclusive** (from-scratch vs warm-started confound, never converged in-budget); a from-scratch standard-architecture control run would give a fair read, not yet built. See `docs/FEEDBACK_RESULTS.md § Chat-tags experiment` / `§ DenseNet-KV ablation` / `§ IR-refinement loss redesign` and the plan at `/Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md`.
4. **True SRS** (`experiments/srs_tagged/`, spans reviewed via `srs_schedule_depth2`, reuses chat-tags' proven fixes) — `srs_depth2_nc4_slot8` (64B, halves+atomic-full-span schedule) **DONE, FAILED strict bar**: val 100%/100%/100%, test 100%/100%/**69.6%** (full-span block never reached 100% test despite full convergence) — confirmed the atomic full-span `(0,4)` block (one IQ+IR unit decoding all 64B single-shot) as the bottleneck and the driver of the O(L²) compute wall; `srs_depth2_nc8_slot8.py` marked SUPERSEDED (would only make the failing mechanism longer). **Pivoted to stitching**: `experiments/srs_tagged/configs/srs_stitch_nc4_slot8.py` reuses the proven `ir_local` overlapping-window chain mechanism (`ar_decode_srs_stitched_tagged`, new `experiments/srs_tagged/stitch_decode.py`) — full-sequence coverage from chaining already-proven 32B windows (linear compute) instead of one long single-shot decode. **SOLVED — success bar met, run complete**: warm-started from the atomic run's checkpoint, reached `STITCHED_MEAN`=100% val+test by step 15000, perfect sweep (every span + stitched chain = 100.0% val AND test) by step 35000, **held that sweep through the final step 60000** (loss 0.0001) — clears and sustains the strict 100%-test bar the atomic run never reached at any checkpoint. Verdict: stitching strictly beats atomic at equal scale/budget (100% sustained vs 69.6% ceiling, also faster per-step) — no tradeoff found, stitching is now the default mechanism for further scale-up. See `docs/SRS_RECIPE.md § Stitching vs atomic full-span`. **`juz1.txt` scaling (44KB, ~700x this scale) confirmed NOT ready** — three concrete gaps (window-tags hardcoded to 3, whole-schedule-in-one-sequence training doesn't scale to ~2778 windows, no intermediate validation step yet), not just "needs more steps"; see `docs/SRS_RECIPE.md § Is juz1.txt scaling ready?` for the recommended 3-step path (window-tag redesign → streaming corpus-ingestion training loop → intermediate 128-256B validation). **Step (c) attempted**: `srs_stitch_nc8_slot8` (128B, 7 chained windows, tags extended to D-G) **FAILED strict bar** — val 100%×6 + window G stuck at 4.2%, test even weaker (broad C-F degradation too, not just G). Diagnosed via qualitative decode: window G's IR1 reaches 100%, IR2 destroys it back to 4.2% ("IR2 destroys IR1's gain" pathology, previously seen at 64B) — root cause is LR decaying to ~0 before window G (last to converge) ever reaches the "IR1 already correct" regime IR2 needs to learn to preserve. Fix attempted: `srs_stitch_nc8_slot8_continue.py`, warm-started, fresh short cosine cycle at lower peak LR — **FAILED**, window G ended flat at 1.4% val (vs 100% for A-F) across the full 20k-step continuation, no improving trend. Revised diagnosis: likely a structural limit tied to window G's extreme position (largest RoPE distance in the packed sequence), not an LR-budget problem. Stopped chasing this fix after two attempts; oversampling window G noted as a future TODO (needs bigger restructuring). See `docs/SRS_RECIPE.md` for full diagnosis.

5. **Dual-attention-block ablation** (`experiments/attn_dual/`, no MLP anywhere — attn+attn per block instead of attn+ffn) — **inconclusive, not negative**. Three-way comparison: warm-started standard baseline 100%/100% sustained (loss ~0.0001); dual-attn from scratch 50.6%/55.4% (loss 3.46, fewer params); standard-from-scratch control 45.2%/64.3% (loss 5.02) — genuinely mixed between the two from-scratch runs (dual-attn faster early + lower loss, standard higher final test match), no clear winner within the 60k-step budget. **Bigger finding**: both from-scratch runs plateaued far below the warm-started 100% — warm-starting was doing far more work than either architecture choice, a new result independent of the MLP question, with implications for the `juz1` roadmap (expect every new scale/schedule to need a warm-start plan, not assume fast from-scratch convergence). See `docs/SRS_RECIPE.md § Final verdict: dual-attn ablation`.

6. **RMSNorm discovery, matched-depth staged re-run, and `juz1` concrete design** — switching `dualattn`'s staged (IQ 160k -> IR 100k) re-run from LayerNorm to RMSNorm produced a dramatic result: IQ stage hit a **perfect 100%/100% sweep by step 30000** (vs LayerNorm's 25%/45% at the same step), IR stage finished at **val 94.0%, test 94.6%** — far better than any prior dual-attn attempt, though short of the strict 100% bar. Reverses an earlier small-scale RMSNorm ablation's negative finding (`configs/hmn_chunk_abl_rmsnorm.py`, noted in Key Principles below — likely because that test reused an untuned LR). **Architecture-specific interaction found**: the SAME RMSNorm switch applied to the standard (with-MLP) architecture showed a genuinely different failure mode — val reached 100% but test stayed stuck at 53.6% (persistent overfitting-like val/test gap that never closed, unlike dual-attn's clean generalizing sweep) — RMSNorm's benefit is not universal, it interacts differently per architecture. Added `ir_slot_window` hyperparameter (`experiments/chat_tags/mask_windowed.py`, default `None`=unbounded/unchanged behavior, verified bit-for-bit) — controls how many prior refine-turns' SLOT tokens a later IR turn can attend to (RNN-framing sliding window), infrastructure only, not yet used in a launched run. **`juz1.txt` (44,443B) scaling given a concrete design**: `n_chunks=2816` padded target (128-multiple, 1.4% padding); the current packed-whole-schedule design would need an `L×L` mask with ~10¹¹ entries (confirmed infeasible for both train AND eval, `chunk_attn` doesn't help — verified it's memory-only, still full `O(L²)` FLOPs); fix is local fixed-size blocks (`block_nc=8`, `L=1694`) chained via the already-proven warmup-seeded stitching mechanism, never one global mask. Four candidate strategies documented (uniform sampling → SRS-weighted resampling → curriculum block growth → hierarchical local+periodic-full-corpus-check), plus a worked sequential-ingestion example (65,536B, 585 blocks, two-phase bootstrap+SRS-review protocol, estimated ~700k-2.1M steps / ~6-17 days, vs a ~1,256-day naive no-sharing baseline) — genuinely uncertain estimates, flagged as needing a small pilot to measure real cross-block transfer speed before committing. Not yet implemented — see `docs/SRS_RECIPE.md § Concrete juz1.txt scaling design` and `§ Worked example`.

7. **Architecture comparison audit + hard gate before `juz1`** — asked directly whether dual-attn is confirmed as the final architecture: **no**. Audited the queue against 3 named gaps (standard+LayerNorm at matched-depth, 128B validation, second-seed reproducibility) and found 2 of 3 weren't actually scheduled to resolve despite being named as caveats. Fixed the highest-priority one: added `srs_stitch_nc4_slot8_iq_ln`/`_ir_ln` (LayerNorm matched-depth control, same 160k+100k budget as the RMSNorm runs) to complete the {dual-attn, standard} x {LayerNorm, RMSNorm} x {scratch, matched-depth} comparison matrix. **Hard gate added**: `juz1` implementation work does not start until the architecture comparison is conclusive, not merely until currently-queued runs finish — if results stay ambiguous, add more experiments (deferred dual-attn+LayerNorm cell, second seed) rather than proceeding on a shaky comparison. Separately decided **not** to restructure `DualAttnBlock` (2 attn/block x 4 blocks) into 8 single-attn blocks — mathematically identical network, zero functional gain, but would break existing checkpoints' state_dict keys mid-comparison; noted a latent `depth_scaled_init` depth-counting trap in the model docstring (`n_layers*2` is the true residual depth) and confirmed it's currently inert — that flag is unused in every config in this project. See `docs/SRS_RECIPE.md § Is dual-attn confirmed as the final architecture?` and `§ One-attn-per-block-double-depth vs current dual-attn`.

8. **Hard gate satisfied — dual-attn+RMSNorm confirmed as the working architecture choice.** Full 2x2 matched-depth (260k step) comparison completed: standard+LayerNorm (IQ 100%/89.3%, IR 33.3%/23.2%), standard+RMSNorm (IQ 100%/53.6%, IR 40.5%/30.4%), dual-attn+RMSNorm (IQ 100%/100%, IR 94.0%/94.6%). **The IQ-stage norm-dependent hypothesis flipped at IR** (RMSNorm slightly beats LayerNorm for standard-arch IR, opposite of IQ) — norm choice isn't cleanly separable by architecture. **The real finding**: both standard-arch variants badly trail dual-attn+RMSNorm at IR regardless of norm choice (~3-4x worse) — the argmax-feedback refinement mechanism itself generalizes far better in the no-MLP architecture at this matched depth. This satisfies the hard gate (clear empirical leader across the full comparison, not one favorable cell) — **dual-attn + RMSNorm is now the working architecture for `juz1` prep and the nc8 window-G test**. Deferred items (dual-attn+LayerNorm cell, second seed, 128B validation, real text) remain open as follow-up validation, not blockers. See `docs/SRS_RECIPE.md § srs_stitch_nc4_slot8_ir_ln final result`.

| Run | Result |
|-----|--------|
| `hmn_feedback_32_ir` | **100% at k=0..4 AND k=0..12 extrapolation** ✓✓ |
| `hmn_chunk_local_32` stage 1 (IQ, 50k) | 81.9% match — solid IQ pretraining ✓ |
| `hmn_chunk_local_32_stage1` (IR, 80k) | **87.5% match** with fixed KV decode ✓ |
| `hmn_chunk_local_64` v1 (64B, pure stitch, 80k) | stitch=76.8%, win0=85.9%, win1=**0%**, win2=13% — chaining failure |
| `hmn_chunk_local_64_v2` (64B, 4-way equal mix, 80k) | **stitch collapsed to 6.2%** — equal mix gave stitch only 25% of steps, insufficient |
| `hmn_chunk_local_64_v3` (64B, targeted fix: stitch×3+win1×1+win2×1, from v1 end) | **failed** — stitch oscillated 58→43→35→45→41%, win2 collapsed at 60k. Killed. |
| `hmn_chunk_local_64_v4` (64B, mask_nochain=True, pure stitch, from stage2 end) | **failed** — win1=0%, win2=12.5% indep; stitch=53.6% best. nochain blocked SLOTs only, model chained through OUTPUT tokens |
| `hmn_chunk_local_64_v5` (64B, mask_nochain=True corrected, pure stitch, from stage2 end) | **running** — v5 fix: IQ SLOT blocked from ALL prior rec_block tokens (SLOT+warmup+output) |
| **Global IQ rw track** | |
| `hmn_chunk_global_iq_rw_nc4_slot4` → ext2 (slot4, IQ only, 150k+ steps) | 18.1% best — Win A BPB ~3.5-4.0, slot capacity bottleneck confirmed |
| `hmn_chunk_global_iq_rw_nc4_wina_ovs` (slot4 + 2× Win A oversample, from ext2) | 18.1% best — Win A BPB 2.128 (new low), 12.5% match peak. Capacity not distribution. |
| `hmn_chunk_global_iq_rw_nc4_slot8` → ext (slot8, IQ only, 80k+45k=125k total) | **44.0% best** — Win B/C improving, Win A BPB=1.283 (plateau) |
| `hmn_chunk_global_iq_rw_nc4_slot8_ir` (slot8 + n_refine=2 IR, from slot8_ext best) | **57.4% best** (step 75k) — Win B 97%, Win C 74%, Win A 3% (IQ=0% throughout — is_clean bug) |
| `hmn_chunk_global_iq_rw_nc4_slot8_ir_v2` (slot8 + IQ fix + Win A oversample, 100k) | **77.8% best** @ step 60k — Win A **100%** ✓, Win B 77.8%, Win C 55.6%. Final (100k): 75.5% |
| All HMN v3 variants (mono, cerb, p2, p4, pinf, tlogit) | ~95% k=0, collapses at k>4 |

### IR vs IQ-only — the decisive comparison

Win B down_counter BPB: **0.350** (IQ-only after 80k steps) → **0.039** (IR after 15k steps) — 9× BPB reduction in 5× fewer steps. IQ-only was plateaued; IR unlocked it in one stage. This mirrors the 32B result where IR lifted 50%→100% match. See `docs/FEEDBACK_RESULTS.md` for full analysis.

### slot8_ir_v2 key findings (see `docs/FEEDBACK_RESULTS.md` for full tables)

- **IQ=0% root cause fixed**: `is_clean=(n_refine==0)` excluded IQ from loss when n_refine>0. Fix: add `iq_global_rw` (n_refine=0) entries at 33% of traj_mix — IQ loss is present, model learns one-shot recall.
- **Win A solved at step 45k** (100% MEAN), stays solved at 100k. IQ training (Win A X=0 at 22% of steps) + IR training (Win A X=0 at 44%) unlocks it.
- **Best checkpoint is cycle 2 end (step 60k)**: cosine_T0=20k → cycle ends at 20k/60k/140k. Cycle 3 restart introduces volatility; Win C collapses at step 70k, never fully recovers.
- **IR2 corrects IR1 regressions**: Win A at step 100k shows IR1=45-62% but IR2=100% — the two-turn chain is qualitatively different from single IR.
- **Win C `up_counter` unsolved** (4.2% at end): IQ reaches 37.5% at step 100k but IR1 destroys it. Root cause: traj_mix gives Win C X=0 only ~0.7% of IR steps (uniform coverage) vs Win A X=0 at 44%. IR learned on other (window, X) pairs applies wrong corrections for Win C at X=0. Fix: add `warmup_x_fixed=16` (Win B) and `warmup_x_fixed=32` (Win C) IR entries to traj_mix. **Source distribution (infinite random bytes) is correct** — EXP1 showed it converges fastest; Win C IQ BPB trends down steadily, confirming algorithm learning not distribution noise. See `docs/FEEDBACK_RESULTS.md § Dataset design`.
- **Ablation queue running**: slot4/slot8/slot12 wina_s0 (IQ-only from scratch) → then slot8 2x model, mixed slot_len, from-scratch 2-stage recipe.

---

## Chunk Memorization — Local-Refine Windowed Architecture (current work)

**Earlier depth-2 SRS curriculum (`hmn_chunk_curric`/`hmn_chunk_srs_ir`) FAILED**: stage 0
(IQ-only, 256B src, 64x slot compression/chunk) never escaped random-baseline BPB (~8.0)
after 12000 steps. Root cause diagnosed as too-aggressive compression + far too few
training steps, not a windowing/refinement problem per se.

**Current approach**: fall back to the ONE proven mechanism (`hmn_feedback_32_ir`:
100% match k=0..12) — IQ once + 2 chained argmax-refine IR turns, scoped to a single
32-byte window — and grow scale by adding MORE overlapping 32-byte windows (fixed
window=32B, fixed stride=16B, 50% overlap) instead of widening compression ratio.
`chunk_len=16` makes 32B windows / 16B stride land exactly on chunk-index boundaries.

**Training script**: [`kvmem/train_hmn_chunk.py`](kvmem/train_hmn_chunk.py) — use `train_fn='fb'` in config.

### Local-refine window unit (`ir_local` trajectory type)
```
chunk_positions_fb_localrefine(n_chunks, chunk_len, slot_len, warmup_len, windows, n_refine)
```
Each `window` in `windows` gets its OWN local IQ turn (recall scoped only to that
window's chunks, not a global full-source read) + `n_refine` chained argmax-IR turns
refining the same window — directly reusing the `hmn_feedback_32_ir` unit, just scoped
per-window instead of always "the whole source". Windows are processed in sequence,
threading one running token offset; `chunk_mask_fb`/`_chunk_make_batch_fb`/
`_fill_argmax_fb` are generic and needed **no changes** to support this.

### Stage recipe — **must match the proven `hmn_feedback_32_iq`/`_ir` recipe**
A slot_len=2 / 8k-step first attempt undertrained badly: stage 1 (IR) eval BPB
diverged 10→19 while train loss kept falling (classic premature-feedback-before-IQ-
is-solid failure). Fix: `slot_len=4` (not 2), IQ stage 50000 steps, IR stage 80000
steps — exactly matching `hmn_feedback_32_iq.py`/`hmn_feedback_32_ir.py`.

`slot_len=4, slot_count=2` in all stages: 4 token positions per slot block, 2 alternating
IDs (258/259). slot_count=4 was ablated but not adopted.

| Stage | n_chunks (src) | windows / traj mix | n_refine | steps | Config | Result |
|---|---|---|---|---|---|---|
| 1 | 2 (32B) | `[(0,2)]` | 0 | 50000 | `hmn_chunk_local_32.py` | 81.9% ✓ |
| 2 | 2 (32B) | `[(0,2)]` | 2 | 80000 | `hmn_chunk_local_32_stage1.py` | **87.5%** ✓ |
| 3 v1 | 4 (64B) | all-3 only | 2 | 80000 | `hmn_chunk_local_64.py` | stitch=76.8%, **win1=0%** — chaining failure |
| 3 v2 | 4 (64B) | equal mix (4-way) from stage2 | 2 | 80000 | `hmn_chunk_local_64_v2.py` | **stitch=6.2%** — too few stitch steps |
| 3 v3 | 4 (64B) | stitch×3+win1×1+win2×1 from v1 | 2 | 80000 | `hmn_chunk_local_64_v3.py` | **failed** — chaining structurally unavoidable, killed at 60k |
| 3 v4 | 4 (64B) | pure stitch, mask_nochain=True (SLOTs only) from stage2 | 2 | 80000 | `hmn_chunk_local_64_v4.py` | **failed** — win1=0% indep, stitch=53.6%. Chained through OUTPUT tokens |
| 3 v5 | 4 (64B) | pure stitch, mask_nochain=True (full blackout) from stage2 | 2 | 80000 | `hmn_chunk_local_64_v5.py` | **running** |
| 4 | 8 (128B) | pure stitch, mask_nochain=True from v5 | 2 | 80000 | `hmn_chunk_local_128_stitch.py` (update pretrained path) | pending |
| 5 | 16 (256B) | stitch×3+win1..win14×1 | 2 | 80000 | `hmn_chunk_local_256.py` from 4b | pending |

**Stage 3 sequence layout** (L=572, slot_len=4, slot_count=2):
```
[ENC_0: src[0:16]  | SLOT×4]   ─┐
[ENC_1: src[16:32] | SLOT×4]    │  one shared encoding pass over all 4 chunks
[ENC_2: src[32:48] | SLOT×4]    │
[ENC_3: src[48:64] | SLOT×4]   ─┘
── window (0,2): bytes 0-31 ──────────────────────────────────────────
[IQ:   SLOT×4 | warmup[0:8]   | out[8:32]  ]                    36 tok
[IR1:  SLOT_A×4 | am[8:32]  | SLOT_B×4 | warmup[0:8]   | out[8:32]  ]  64 tok
[IR2:  SLOT_A×4 | am[8:32]  | SLOT_B×4 | warmup[0:8]   | out[8:32]  ]  64 tok
── window (1,3): bytes 16-47 ─────────────────────────────────────────
[IQ:   SLOT×4 | warmup[16:24] | out[24:48] ]                    36 tok
[IR1:  SLOT_A×4 | am[24:48] | SLOT_B×4 | warmup[16:24] | out[24:48] ]  64 tok
[IR2:  SLOT_A×4 | am[24:48] | SLOT_B×4 | warmup[16:24] | out[24:48] ]  64 tok
── window (2,4): bytes 32-63 ─────────────────────────────────────────
[IQ:   SLOT×4 | warmup[32:40] | out[40:64] ]                    36 tok
[IR1:  SLOT_A×4 | am[40:64] | SLOT_B×4 | warmup[32:40] | out[40:64] ]  64 tok
[IR2:  SLOT_A×4 | am[40:64] | SLOT_B×4 | warmup[32:40] | out[40:64] ]  64 tok
```

Growth rule (fixed window=32B, fixed stride=16B): `n_windows = (src_len-32)/16 + 1`.
128B→7 windows, 256B→15 windows (chunk-idx tuples `(i,i+2)` stepping by 1).

### Eval protocol — full-sequence "prolonged AR" decode (stage 3+, >1 window)
`ar_decode_chunk_fb_stitch_kv` (next to `ar_decode_chunk_fb_kv`): only the very first
window's warmup (first 8 bytes of the whole source) is seeded from ground truth — every
later byte (every later window's warmup AND output) comes from the model's own
previously generated tokens, stitched into one global `(src_len,)` buffer (later windows
overwrite earlier ones in the 16B overlap). Works because warmup_len=8 fits entirely
within the 16B overlap. Wired automatically in `train_chunk_fb`'s eval loop whenever
`eval_traj == 'ir_local'` and the trajectory spans >1 window; single-window stages keep
using the original last-block-only `ar_decode_chunk_fb_kv`. Training-time batch
construction is unaffected (teacher-forced GT warmup per window, unchanged).

### Stage 3 v1 failure — slot chaining dependency (do not reintroduce)

v1 trained ONLY on all-3-windows trajectories. The mask only blocks raw output
regions (`c0:c1`) from later windows — it does NOT block window i's IQ SLOT from
attending to window j<i's IQ/IR SLOT tokens. The model exploited this: window 1's
IQ SLOT read from window 0's SLOT context instead of encoding chunks 1-2 from the
encoding blocks independently. Result:
- Window 0 (independent, no prior window SLOTs in context): 85.9% ✓
- Window 1 (relied on window 0's SLOT): 0.0% in isolation ✗
- Window 2 (relied on windows 0+1's SLOT): 13.0% in isolation ✗
- Stitch (chaining present): 76.8% (windows benefit from prior SLOTs)

**Root cause of v1/v2/v3 failures (architectural)**:
`chunk_mask_fb` only blocked IQ SLOT rows from source chunks — NOT from prior rec_block SLOT tokens.
Window 1's IQ SLOT could freely attend to window 0's IQ SLOT (chaining). No training distribution
could fix this: stitch training reinforces chaining, singles training fights it, model oscillates.

**v4 attempt (failed)**: `mask_nochain=True` blocked IQ SLOT rows from prior SLOT tokens only.
The model found a new chaining path through prior OUTPUT tokens: window 1's IQ SLOT read window 0's
recalled output bytes 16-31 (the 50% overlap region). Result: win1=0%, win2=12.5% independent;
stitch=53.6% (only works because window 0 fills the overlap, not because window 1 encodes it).

**Fix (v5)**: Rule 3b now blocks IQ SLOT rows from ALL tokens in prior rec_blocks — SLOT, warmup,
argmax, AND output. Every window can only attend to enc-block SLOTs. All chaining paths cut off.
See `chunk_mask_fb(..., nochain=True)` in `kvmem/train_hmn_chunk.py`.

**v2 attempt (failed)**: 4-way equal-weight mix from stage 2 checkpoint.
- Stitch got only 25% of steps → stitch collapsed from 76.8% (v1) to 6.2%
- Per-window: win0=13.5%, win1=15.1%, win2=24.5% — independence improved but weak
- Root cause: starting point (stage 2 checkpoint) had no stitch knowledge + too few stitch steps

**Fix (v3)**: Targeted mix starting from v1 checkpoint (76.8% stitch, win0=85.9%):
- Stitch ×3 (60% of steps): preserves v1's strong stitch quality
- Win1 ×1 (20%): fixes window (1,3) — chains in v1, no solo training
- Win2 ×1 (20%): fixes window (2,4) — partial chaining in v1
- Win0 skipped: already independent (no prior window to chain from)

**Staged approach for 4+ windows**: always establish pure stitch first (100% steps),
THEN fine-tune independence. Never mix from a damaged-stitch starting point.
- Stage 4: pure stitch `hmn_chunk_local_128_stitch.py` (from v5 end, update pretrained path)
- Stage 5: `hmn_chunk_local_256.py` (from 128 end)

### Experiment tiers — ordered by difficulty (do not skip ahead)

**Tier 1 — ir_local stitch (current work)**
Prove per-window IQ+IR stitch works reliably at scale: 64B → 128B → 256B.
Success bar: stitch ≥60% AND all windows independently ≥40% before moving up.
Configs: `hmn_chunk_local_64_v5` → `hmn_chunk_local_128_stitch` → `hmn_chunk_local_256`

**Tier 2 — random-warmup continuation (after Tier 1 at ≥128B)**
Goal: "continue from any warmup position X in sequence."
Step 1: zero-shot eval on Tier 1 model — seed warmup at arbitrary X≠0, no retraining.
  Expected: works at window-start positions only.
Step 2 if needed: add randomized warmup offset within window to ir_local training.
  Low cost — same architecture, just change warmup position in batch construction.

**Tier 3 — ir_winrefine: global IQ + per-window IR (hard, defer)**
`chunk_positions_fb_winrefine`: ONE global IQ reads full source, windowed IRs refine per window.
- No chaining problem (all windows share ONE global IQ SLOT, not window-specific SLOTs)
- Natural home for random-warmup: global encoding, seed IR from any X
- BUT: global IQ must compress full source into same small SLOT → was hard before
  (earlier curriculum at 256B+64x compression failed at BPB~8.0)
- Do NOT attempt until Tier 1+2 proven solid at ≥128B

**Parked — ir_stitch** (`chunk_positions_fb_stitch`, configs `hmn_chunk_stitch_a/b/smoke.py`):
Per-window IQ+IR + optional full-span final IQ. Has THE SAME chaining problem as ir_local v1.
Do not revive before ir_winrefine is warranted — it doesn't solve the chaining issue.

### Run commands
```bash
# Stage 3 v5 (64B, mask_nochain=True corrected — full prior rec_block blackout, from stage2 end)
caffeinate -i python3 -m kvmem.train_hmn_chunk \
    --config configs/hmn_chunk_local_64_v5.py \
    --pretrained logs/hmn_chunk_local_32_stage1/checkpoints/stage0_end.pt \
    --device mps

# Stage 4 (128B, pure stitch, mask_nochain=True, from v5 end)
caffeinate -i python3 -m kvmem.train_hmn_chunk \
    --config configs/hmn_chunk_local_128_stitch.py \
    --pretrained logs/hmn_chunk_local_64_v5/checkpoints/stage0_end.pt \
    --device mps

# Stage 5 (256B, 15 windows, from 128 end)
caffeinate -i python3 -m kvmem.train_hmn_chunk \
    --config configs/hmn_chunk_local_256.py \
    --pretrained logs/hmn_chunk_local_128_stitch/checkpoints/stage0_end.pt \
    --device mps

# Qualitative eval (any checkpoint — auto-detects n_chunks and windows from hp)
python3 eval_fb_qual.py --ckpt logs/hmn_chunk_local_64_v5/checkpoints/stage0_end.pt --device mps
python3 eval_fb_qual.py --ckpt logs/hmn_chunk_local_128_stitch/checkpoints/stage0_end.pt --device mps
```

### KV cache decode — why it's valid and what the mask does

The mask is strictly causal: `visible = causal & ~blocked` where `causal = (c <= r)`.
Additional blocking only removes connections — it never adds backward attention. No
position ever attends to a future token.

KV cache inference is valid because:
1. The cache always holds positions `0..L_cached-1` (all causally prior).
2. `seg_mask = full_mask[seg_start:seg_end, :L_cached + seg_len]` is the correct
   submatrix **only when `seg_start == L_cached`** — sequence position of the segment
   equals the cache size. The off-by-one bug broke this invariant; the fix restores it.
3. Given that invariant, row `i` of seg_mask = `full_mask[L_cached+i, :L_cached+seg_len]`,
   which correctly encodes which of the `L_cached+i+1` prior positions token `i` may
   attend to (both the cached portion and the causal-within-segment portion).

The mask is **structurally required** even with KV cache — the architecture has
non-trivial blocking that a plain causal mask cannot express:
- IQ SLOT/warmup/output rows: blocked from source chunk columns
- IR SLOT_A/argmax/SLOT_B rows: blocked from source chunks AND all prior rec_block
  output columns (the GT-leak fix)
- IR warmup/output rows: blocked from everything except own SLOT_B + own warmup/out

Without the mask, IR tokens would freely attend to source chunks and bypass the slot
bottleneck entirely.

### TODO — filtered KV cache for IR inference (no mask needed)

IR tokens are blocked from source chunks and all prior rec outputs. Instead of passing
`seg_mask` with `-1e9` blocking, build a **filtered cache** for IR tokens containing
only the positions they're allowed to attend to:
- Encoding SLOT positions (`sl0..sl1` per enc block) — NOT source chunk positions
- Exclude all prior rec_block output regions (`c0..c1` of any earlier block)
- SLOT_A / argmax / SLOT_B of the current IR block are in the current segment (causal, no cache entry needed)

Equivalent to masking because `-1e9 → exp(-1e9) ≈ 0` after softmax — absent keys give
the same result. RoPE is unaffected (positions baked into keys/queries before caching).
**Condition**: filter must match the mask exactly (train/eval mismatch otherwise).

Speedup: at 64B (4 chunks), source chunks alone are `4×16 = 64` tokens of the
`enc_end = 80` cache — IR's filtered cache is ~half the size immediately, growing
faster than the SLOT portion as src scales up. Not urgent (eval isn't the bottleneck
now), but a clean optimization for a production inference path.

### Known bugs fixed (do not reintroduce)
1. **`chunk_mask_fb` GT leak** (`train_hmn_chunk.py` ~line 467): IR turns' SLOT_A/argmax/SLOT_B
   rows must be blocked from ALL prior rec_block outputs (`is_any_rec_output`), not just
   source chunks. During training those positions hold teacher-forced GT — a direct attention
   path lets the model shortcut, collapsing at AR-decode eval. Fixed by adding `is_any_rec_output`
   union and ORing it into the block condition for IR rows.

2. **KV decode off-by-one** (`ar_decode_chunk_fb_kv` and `ar_decode_chunk_fb_stitch_kv`
   `_decode_segment`): after generating `out_len` tokens, the LAST token was written into
   `tok` but never processed through the model to add its KV to the cache. This left
   `L_cached = c1-1` instead of `c1`, causing all subsequent blocks to receive RoPE positions
   shifted by -1. Eval showed 0% match throughout all 80k training steps despite correct
   training loss; teacher-forced BPB was near-zero (model trained fine, only eval was broken).
   Fixed by adding one extra forward pass after the output loop to cache the last decoded token.

3. **Non-KV decode argmax chaining** (`ar_decode_chunk_fb`): original code used
   `ir_rbs = {span: rb}` dict which only kept the LAST IR block per span, then filled
   its argmax from IQ directly (skipping intermediate IR steps). Fixed by processing
   `rec_blocks` in sequence order with `last_ir_by_span` tracking.

### Metrics
- **val**: `make_test_sequences` split into n_chunks (`val_n_seqs` config knob caps
  how many of the 8 deterministic sequences are used, for faster iteration)
- **test**: specific surah file, padded to (n_chunks, chunk_len), eval-only — currently
  omitted from new configs (no `eval_file`) for faster iteration
- **BPB**: teacher-forced NLL/ln(2) on the final recall block's own output (single
  window) or the full stitched sequence (multi-window, via `ar_decode_chunk_fb_stitch_kv`)
- **match%**: AR greedy exact-match, same scope as BPB above

---

---

## Feedback Architecture

Sequence layout — Turn 0 (IQ):
```
[src: src_len] [SLOT×n] [warmup: wl] [out: ol]
```

Turn t≥1 (IR — argmax feedback):
```
[SLOT_A×n] [argmax_{t-1}: ol] [SLOT_B×n] [warmup: wl] [out: ol]
```

- No MEM_START/MEM_END — only SLOT tokens (IDs 258-259 by default, `slot_count=2`)
- `argmax_{t-1}` = greedy decode of previous turn's output positions (detached)
- Mask: `0.0`=attend, `-1e9`=blocked (additive bias — **critical**, was `0.0/1.0` bug causing NLL→0)
- IQ bottleneck: `warmup`/`out` blocked from `src`
- IR bottleneck: `warmup`/`out` blocked from `SLOT_A` and `argmax` (only `SLOT_B` visible)

**IQ pretraining required** before IR — model must learn slot compression before feedback is meaningful.

### Global IQ with Random Warmup (`iq_global_rw` / `iq_global_rw_ir` traj types)

Alternative to local-refine: ONE global IQ reads all nc enc_blocks, predicts any 32B window (warmup_len=8, out_len=24). Training samples random warmup offset X ∈ [0, src_len-32]; eval uses fixed window-start offsets {0, 16, 32}.

```
[ENC_0: src[0:16] | SLOT×slot_len]   (4×16 + 4×slot_len = enc_end tokens)
[ENC_1: src[16:32] | SLOT×slot_len]
[ENC_2: src[32:48] | SLOT×slot_len]
[ENC_3: src[48:64] | SLOT×slot_len]
[IQ:  SLOT×slot_len | warmup[X:X+8] | out[X+8:X+32]]
[IR1: SLOT_A×slot_len | argmax×24 | SLOT_B×slot_len | warmup[X:X+8] | out]  ← n_refine≥1
[IR2: SLOT_A×slot_len | argmax×24 | SLOT_B×slot_len | warmup[X:X+8] | out]  ← n_refine≥2
```

- **slot_len** is the bandwidth knob: slot_len=4 → 4 tokens compress 32B → capacity bottleneck. slot_len=8 unlocks Win B/C. Win A remains hardest at all slot sizes tested.
- **n_refine=2**: two chained IR turns after IQ. Same warmup offset X for all turns per example.
- **IR vs long IQ**: IQ-only Win B down_counter BPB plateaued at 0.350 after 80k steps. IR reduced it to 0.039 in 15k steps (9× lower, 5× fewer steps). IR is structurally necessary, not just beneficial — the delta-encoding bottleneck through SLOT_B cannot be replicated by more IQ training.
- **Argmax is the right feedback signal** (not soft probabilities): training fills argmax positions with hard GT tokens; eval fills them with model's hard argmax predictions. Train/eval distributions match. Soft feedback (`softmax(logits) @ E`) would create a distribution mismatch and require retraining with Gumbel-softmax.

Run commands:
```bash
# slot8 IR stage (RUNNING)
caffeinate -i python3 -m kvmem.train_hmn_chunk \
    --config configs/hmn_chunk_global_iq_rw_nc4_slot8_ir.py \
    --pretrained logs/hmn_chunk_global_iq_rw_nc4_slot8_ext/checkpoints/stage0_best.pt \
    --device mps
tail -f logs/hmn_chunk_global_iq_rw_nc4_slot8_ir/train.log
```

Traj mix (`slot8_ir`, IQ=0% bug — do not use as reference):
| weight | nc | n_refine | warmup X (train) | warmup X (eval) | SLOT pos | L |
|--------|----|----------|------------------|-----------------|----------|---|
| 1.0 | 4 | 2 | uniform [0,32] | {0, 16, 32} | 96 | 280 |

Traj mix (`slot8_ir_v2`, IQ fix + Win A oversample — current best):
| weight | type | nc | n_refine | warmup X (train) | share | purpose |
|--------|------|----|----------|------------------|-------|---------|
| 1.0 | `iq_global_rw_ir` | 4 | 2 | uniform [0,32] | 22% | IR all windows |
| 0.5 | `iq_global_rw` | 4 | 0 | uniform [0,32] | 11% | IQ-only all windows |
| 2.0 | `iq_global_rw_ir` | 4 | 2 | fixed X=0 | 44% | Win A IR heavy |
| 1.0 | `iq_global_rw` | 4 | 0 | fixed X=0 | 22% | Win A IQ-only heavy |

### Run

```bash
# Stage 0: IQ pretraining from scratch
python -m kvmem.train_hmn_feedback --config configs/hmn_feedback_32_iq.py --device mps

# Stage 1: IR + feedback, pretrained from IQ
python -m kvmem.train_hmn_feedback \
    --config configs/hmn_feedback_32_ir.py \
    --pretrained logs/hmn_feedback_32_iq/checkpoints/stage0_end.pt \
    --device mps

# Quick eval (inline — no --eval-only yet in train_hmn_feedback)
python3 -c "
import torch, numpy as np
from kvmem.model import build_model
from kvmem.train_hmn_feedback import ar_decode_fb
from kvmem.utils import make_test_sequences, cer
device = torch.device('mps')
ckpt = torch.load('logs/hmn_feedback_32_ir/checkpoints/stage0_end.pt', map_location=device)
sd = ckpt['model']
hp = dict(V=268,d=64,n_layers=4,n_heads=4,d_ff=256,rope=True,yarn=True,null_kv=True,compile=False)
model = build_model(hp, device); model.load_state_dict(sd); model.eval()
seqs = make_test_sequences(32)
for hk in [0,1,2,3,4,6,8,10,12]:
    cers = [cer(ar_decode_fb(model,list(x),4,2,list(x[:8]),24,device,k=hk),list(x[8:32])) for x in seqs.values()]
    print(f'k={hk}  match={100*(1-sum(cers)/len(cers)):.1f}%')
"
```

---

## HMN v3 Architecture (predecessor — refinement never worked)

Sequence: `[MEM_0: BLEN][src: src_len][MEM_1: BLEN] ... [MEM_k+1][warmup][out]`
- `BLEN = slot_len + 2` (MEM_START + slots + MEM_END)
- Recall rows blocked from attending to everything except final MEM block

Training: `kvmem/train_hmn_mono.py` — flat_mono, cerb, cum_mean, teacher logit variants.
All variants plateau at ~95% with no monotone improvement across k turns.

---

## Generalization Axes

Current model holds all of these fixed to prevent train/test mismatch. Listed in order of unlock priority.

### Source
- `src_len` — fixed per stage (32B→64B→128B→256B), could be variable length
- Content type — random bytes; real use needs natural text / structured data
- Encoding resolution (`chunk_len`) — fixed at 16B/chunk

### Window geometry
- Window size — fixed at 32B (2 chunks)
- Stride / overlap — fixed at 16B (50%)
- Window boundary alignment — fixed to chunk grid; could be arbitrary byte positions
- Window order during stitch — fixed left→right

### Recall
- **Warmup position** — fixed at window start; Tier 2 goal is any X in [0, src_len-8]
- Warmup length — fixed at 8B
- Query type — always "recall forward from warmup"; could generalize to random-access, fill-in-middle

### Refinement
- **`n_refine` per window** — fixed at 2; could vary per window, per difficulty, or adaptively (closed-loop SRS: R(t) = R₀·exp(-λt))
- Refinement schedule across SRS sessions — currently open-loop; recipe calls for adaptive S(n) = S₀·αⁿ

### Slot / memory
- **`slot_len`** — fixed at 4; could grow (harder content needs more capacity) or shrink/consolidate as retention improves
- `slot_count` — fixed at 2 (IDs 258/259); ablation tests dedicated SLOT_B IDs (260/261)
- Slot scope — currently one slot per window (ir_local); `ir_winrefine` uses one global slot for full source
- **Slot consolidation** — after multiple refines, compress multiple window slots into fewer tokens (SRS "strengthen trace")

### Training distribution
- Single source per batch — SRS recipe needs multi-sequence with independent retention clocks per sequence
- Train/test sequence type — currently identical random bytes; real use has domain shift
- Curriculum order — fixed stage sequence; could be adaptive per-window based on measured retention

### Model
- Model size (`d`, `n_layers`) — fixed at d=64, 4 layers
- Positional encoding range — RoPE+YaRN handles some OOD extension but untested at long range

### Unlock order (planned)
1. Warmup position (Tier 2 — low code cost, same architecture)
2. `n_refine` adaptive per window (closed-loop, needs retention signal)
3. `slot_len` growing/shrinking (consolidation = SRS "strengthen" primitive)
4. Variable `src_len` / natural text (real corpus ingestion)

---

## Key Principles

- `null_kv=True` always — 1.5–2× faster, better bpb
- Mask convention: `0.0`=attend, `-1e9`=blocked (additive bias for `F.scaled_dot_product_attention`)
- IQ stage before IR — always required for feedback arch
- Feedback eval: use inline script above (no `--eval-only` in train_hmn_feedback yet)
- **`rmsnorm=True` already tried and did WORSE, do not re-suggest without an LR retune**: `configs/hmn_chunk_abl_rmsnorm.py` vs `hmn_chunk_abl_baseline.py` (small 32B-scale ablation, 20k+30k steps) — RMSNorm's IR-stage final loss was ~4x higher (2.17 vs 0.54) and showed a loss spike at the IQ→IR transition baseline didn't have. Likely cause: reused LayerNorm's tuned LR (`3e-4`) unchanged — RMSNorm has different gradient/activation scale and needs its own LR sweep to be a fair test, not a straight flag flip.

### Model size vs task — MDL principle (see [`docs/MDL_MODEL_SIZE.md`](docs/MDL_MODEL_SIZE.md))

- **Parameter count scales with algorithm complexity, not sequence length.** The same 231k model should handle 128B and 256B — the per-chunk encoding algorithm is identical at all scales.
- **Current model is ~4–8× the theoretical minimum** for single-window 32B. Overhead is the gradient-descent tax for SGD learnability, not waste.
- **Position-dependent encoding = longer MDL description.** Vlen training is MDL regularization: training at multiple nc values penalizes position-dependent solutions (higher description length) and forces the model toward position-invariant ones (shorter description).
- **If a run stalls: do not add parameters first.** Correct order: (1) broaden training distribution, (2) simplify algorithm (IQ-only fallback), (3) increase model size only as last resort.
- **Dataset is infinite random bytes** — classical overfitting analysis does not apply. The relevant quantity is model description length vs target function description length, not model size vs dataset size.

---

## Docs

| What | Where |
|------|-------|
| **SRS recipe, scaling theory, open/closed-loop** | [`docs/SRS_RECIPE.md`](docs/SRS_RECIPE.md) |
| **Feedback results + global IQ rw + IR evidence** | [`docs/FEEDBACK_RESULTS.md`](docs/FEEDBACK_RESULTS.md) |
| HMN v3 reference book | [`docs/BOOK.md`](docs/BOOK.md) |
| KV capacity + SRS trajectories | [`docs/kv_dims.md`](docs/kv_dims.md) |
| All configs | [`configs/`](configs/) |
| **Chunk SRS / local-refine + global IQ rw training** | [`kvmem/train_hmn_chunk.py`](kvmem/train_hmn_chunk.py) |
| Local-refine configs (active) | `hmn_chunk_local_32.py`, `hmn_chunk_local_32_stage1.py`, `hmn_chunk_local_64_v5.py`, `hmn_chunk_local_128_stitch.py`, `hmn_chunk_local_256.py` |
| **Global IQ rw configs (active)** | `hmn_chunk_global_iq_rw_nc4_slot8_ir_v2.py` (**DONE** 77.8% best), `hmn_chunk_global_iq_rw_nc4_slot4_wina_s0.py` (**RUNNING**), `hmn_chunk_global_iq_rw_nc4_slot8_wina_s0.py`, `hmn_chunk_global_iq_rw_nc4_slot12_wina_s0.py` (queued) |
| Local-refine eval (generic) | [`eval_fb_qual.py`](eval_fb_qual.py) — any stage, auto-detects n_chunks from checkpoint |
| Parked — ir_stitch (chaining problem, do not revive) | [`configs/hmn_chunk_stitch_a.py`](configs/hmn_chunk_stitch_a.py), `_b.py`, `_smoke.py` |
| Parked — ir_winrefine (global IQ, Tier 3) | `chunk_positions_fb_winrefine` in train_hmn_chunk.py |
| **Chat-tags experiment (isolated, no kvmem/ edits — code+logs colocated under `experiments/chat_tags/`)** | [`experiments/chat_tags/`](experiments/chat_tags/) — `vocab.py`, `positions.py`, `batch.py`, `train.py`, `configs/slot8_tagged_phaseA_iq.py` (**DONE**, passed), `configs/slot8_tagged_phaseB_full.py` (**DONE**, 60.2% best, Win C confounded — see status above) |
| Feedback training (32B) | [`kvmem/train_hmn_feedback.py`](kvmem/train_hmn_feedback.py) |
| HMN v3 mono training | [`kvmem/train_hmn_mono.py`](kvmem/train_hmn_mono.py) |
| Test set | [`datasets/suratalfatihah.txt`](datasets/suratalfatihah.txt) |
