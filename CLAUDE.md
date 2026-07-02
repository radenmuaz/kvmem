# kvmem

Fast-weight language model — HashMemNet (HMN). Two architectures: v3 (MEM blocks) and **Feedback** (argmax loop, current focus).

**Read these docs to resume:**

| Priority | Doc | Why |
|----------|-----|-----|
| 1 | [`docs/SRS_RECIPE.md`](docs/SRS_RECIPE.md) | **Vision**: SRS equations, open/closed-loop, corpus ingestion recipe, scaling to backprop LM NLL |
| 2 | [`docs/FEEDBACK_RESULTS.md`](docs/FEEDBACK_RESULTS.md) | **Key result**: feedback arch achieves 100% k=0..12, why it works, bug history |
| 3 | [`docs/BOOK.md`](docs/BOOK.md) § 8 | HMN v3 architecture reference (predecessor) |
| 4 | [`docs/kv_dims.md`](docs/kv_dims.md) | KV capacity math, model size, SRS multi-sequence design |

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

**Feedback architecture solved the refinement problem. Now extending to multi-sequence SRS chunk memorization.**

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
| All HMN v3 variants (mono, cerb, p2, p4, pinf, tlogit) | ~95% k=0, collapses at k>4 |

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

---

## Docs

| What | Where |
|------|-------|
| **SRS recipe, scaling theory, open/closed-loop** | [`docs/SRS_RECIPE.md`](docs/SRS_RECIPE.md) |
| **Feedback results + architecture** | [`docs/FEEDBACK_RESULTS.md`](docs/FEEDBACK_RESULTS.md) |
| HMN v3 reference book | [`docs/BOOK.md`](docs/BOOK.md) |
| KV capacity + SRS trajectories | [`docs/kv_dims.md`](docs/kv_dims.md) |
| All configs | [`configs/`](configs/) |
| **Chunk SRS / local-refine training** | [`kvmem/train_hmn_chunk.py`](kvmem/train_hmn_chunk.py) |
| Local-refine configs (active) | `hmn_chunk_local_32.py`, `hmn_chunk_local_32_stage1.py`, `hmn_chunk_local_64_v3.py`, `hmn_chunk_local_128_stitch.py`, `hmn_chunk_local_128_v3.py`, `hmn_chunk_local_256.py` |
| Local-refine eval (generic) | [`eval_fb_qual.py`](eval_fb_qual.py) — any stage, auto-detects n_chunks from checkpoint |
| Parked — ir_stitch (chaining problem, do not revive) | [`configs/hmn_chunk_stitch_a.py`](configs/hmn_chunk_stitch_a.py), `_b.py`, `_smoke.py` |
| Parked — ir_winrefine (global IQ, Tier 3) | `chunk_positions_fb_winrefine` in train_hmn_chunk.py |
| Feedback training | [`kvmem/train_hmn_feedback.py`](kvmem/train_hmn_feedback.py) |
| HMN v3 mono training | [`kvmem/train_hmn_mono.py`](kvmem/train_hmn_mono.py) |
| Test set | [`datasets/suratalfatihah.txt`](datasets/suratalfatihah.txt) |
