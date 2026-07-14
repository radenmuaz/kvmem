# Early Architecture History (pre-dual-attn)

Archived from `CLAUDE.md` to keep that file focused on the current dual-attn
+ RMSNorm work. Everything here predates the dual-attention-block ablation
and the RMSNorm discovery — kept for reference (bug history, failed
approaches not to reintroduce, foundational mechanisms still relied upon)
but no longer the active line of work.

---

## HMN v3 Architecture (predecessor — refinement never worked)

Sequence: `[MEM_0: BLEN][src: src_len][MEM_1: BLEN] ... [MEM_k+1][warmup][out]`
- `BLEN = slot_len + 2` (MEM_START + slots + MEM_END)
- Recall rows blocked from attending to everything except final MEM block

Training: `kvmem/train_hmn_mono.py` — flat_mono, cerb, cum_mean, teacher logit variants.
All variants plateau at ~95% with no monotone improvement across k turns.

See also `docs/BOOK.md` § 8 for the full HMN v3 reference.

---

## Feedback Architecture — foundational primitive

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

**IQ pretraining required** before IR — model must learn slot compression before feedback is meaningful. This principle carries forward into every later architecture (chat-tags, true SRS, dual-attn).

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
# slot8 IR stage
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

Traj mix (`slot8_ir_v2`, IQ fix + Win A oversample — best in this track):
| weight | type | nc | n_refine | warmup X (train) | share | purpose |
|--------|------|----|----------|------------------|-------|---------|
| 1.0 | `iq_global_rw_ir` | 4 | 2 | uniform [0,32] | 22% | IR all windows |
| 0.5 | `iq_global_rw` | 4 | 0 | uniform [0,32] | 11% | IQ-only all windows |
| 2.0 | `iq_global_rw_ir` | 4 | 2 | fixed X=0 | 44% | Win A IR heavy |
| 1.0 | `iq_global_rw` | 4 | 0 | fixed X=0 | 22% | Win A IQ-only heavy |

### Run (32B primitive)

```bash
# Stage 0: IQ pretraining from scratch
python -m kvmem.train_hmn_feedback --config configs/hmn_feedback_32_iq.py --device mps

# Stage 1: IR + feedback, pretrained from IQ
python -m kvmem.train_hmn_feedback \
    --config configs/hmn_feedback_32_ir.py \
    --pretrained logs/hmn_feedback_32_iq/checkpoints/stage0_end.pt \
    --device mps

# Quick eval (inline — no --eval-only in train_hmn_feedback)
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

## Chunk Memorization — Local-Refine Windowed Architecture (early scaling attempts)

**Earlier depth-2 SRS curriculum (`hmn_chunk_curric`/`hmn_chunk_srs_ir`) FAILED**: stage 0
(IQ-only, 256B src, 64x slot compression/chunk) never escaped random-baseline BPB (~8.0)
after 12000 steps. Root cause diagnosed as too-aggressive compression + far too few
training steps, not a windowing/refinement problem per se.

**Approach that followed**: fall back to the ONE proven mechanism (`hmn_feedback_32_ir`:
100% match k=0..12) — IQ once + 2 chained argmax-refine IR turns, scoped to a single
32-byte window — and grow scale by adding MORE overlapping 32-byte windows (fixed
window=32B, fixed stride=16B, 50% overlap) instead of widening compression ratio.
`chunk_len=16` makes 32B windows / 16B stride land exactly on chunk-index boundaries.
This is the direct ancestor of the later `ir_local` and stitching mechanisms still in
active use.

**Training script**: [`kvmem/train_hmn_chunk.py`](../kvmem/train_hmn_chunk.py) — use `train_fn='fb'` in config.

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
| 3 v5 | 4 (64B) | pure stitch, mask_nochain=True (full blackout) from stage2 | 2 | 80000 | `hmn_chunk_local_64_v5.py` | succeeded — v5 fix: IQ SLOT blocked from ALL prior rec_block tokens (SLOT+warmup+output) |

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
128B→7 windows, 256B→15 windows (chunk-idx tuples `(i,i+2)` stepping by 1). This exact
growth rule is what the later `srs_stitch_nc4/nc8_slot8` configs and the `juz1` scaling
design (`docs/SRS_RECIPE.md`) still use unchanged.

### Eval protocol — full-sequence "prolonged AR" decode (stage 3+, >1 window)
`ar_decode_chunk_fb_stitch_kv` (next to `ar_decode_chunk_fb_kv`): only the very first
window's warmup (first 8 bytes of the whole source) is seeded from ground truth — every
later byte (every later window's warmup AND output) comes from the model's own
previously generated tokens, stitched into one global `(src_len,)` buffer (later windows
overwrite earlier ones in the 16B overlap). Works because warmup_len=8 fits entirely
within the 16B overlap. Wired automatically in `train_chunk_fb`'s eval loop whenever
`eval_traj == 'ir_local'` and the trajectory spans >1 window; single-window stages keep
using the original last-block-only `ar_decode_chunk_fb_kv`. Training-time batch
construction is unaffected (teacher-forced GT warmup per window, unchanged). This is
the direct ancestor of `ar_decode_srs_stitched_tagged` still used today.

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

**Fix (v5, still the active rule today)**: Rule 3b now blocks IQ SLOT rows from ALL tokens in prior
rec_blocks — SLOT, warmup, argmax, AND output. Every window can only attend to enc-block SLOTs. All
chaining paths cut off. See `chunk_mask_fb(..., nochain=True)` in `kvmem/train_hmn_chunk.py` — this
`nochain` rule is unchanged and still load-bearing in every current experiment.

**v2 attempt (failed)**: 4-way equal-weight mix from stage 2 checkpoint.
- Stitch got only 25% of steps → stitch collapsed from 76.8% (v1) to 6.2%
- Per-window: win0=13.5%, win1=15.1%, win2=24.5% — independence improved but weak
- Root cause: starting point (stage 2 checkpoint) had no stitch knowledge + too few stitch steps

**Fix (v3)**: Targeted mix starting from v1 checkpoint (76.8% stitch, win0=85.9%):
- Stitch ×3 (60% of steps): preserves v1's strong stitch quality
- Win1 ×1 (20%): fixes window (1,3) — chains in v1, no solo training
- Win2 ×1 (20%): fixes window (2,4) — partial chaining in v1
- Win0 skipped: already independent (no prior window to chain from)

**Staged approach for 4+ windows lesson**: always establish pure stitch first (100% steps),
THEN fine-tune independence. Never mix from a damaged-stitch starting point. This lesson
carried forward into every later staged recipe (chat-tags Phase A→B, dual-attn IQ→IR).

### Experiment tiers — historical planning (superseded by actual progress)

**Tier 1 — ir_local stitch**: prove per-window IQ+IR stitch works reliably at scale:
64B → 128B → 256B. This tier's goal was later achieved and superseded by the
`srs_stitch_nc4_slot8`/`srs_stitch_nc8_slot8` results (see `docs/SRS_RECIPE.md`).

**Tier 2 — random-warmup continuation**: goal was "continue from any warmup position X."
Superseded — the stitching mechanism's fixed-warmup-position design proved sufficient.

**Tier 3 — ir_winrefine: global IQ + per-window IR (parked, never attempted)**:
`chunk_positions_fb_winrefine`: ONE global IQ reads full source, windowed IRs refine per
window. No chaining problem (all windows share ONE global IQ SLOT). Natural home for
random-warmup. Never attempted — global IQ compressing the full source into one small
SLOT was hard in earlier curricula (BPB~8.0 at 256B+64x compression). Superseded in
spirit by the chain-memory design (`docs/SRS_RECIPE.md § Chain memory`), which solves
the same "shared global memory" goal differently (bounded accumulating CHAIN_SLOT
instead of one global IQ read).

**Parked — ir_stitch** (`chunk_positions_fb_stitch`, configs `hmn_chunk_stitch_a/b/smoke.py`):
Per-window IQ+IR + optional full-span final IQ. Has THE SAME chaining problem as ir_local v1.
Never revived — the atomic full-span mechanism it partially resembles was later tried
directly (`srs_depth2_nc4_slot8`) and confirmed to underperform stitching.

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

### TODO — filtered KV cache for IR inference (no mask needed, still open)

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
in this project so far), but a clean optimization for a production inference path — the
current `experiments/attn_dual/decode.py`'s full-recompute no-KV eval decode is the
inverse tradeoff (simplicity over speed) for a different reason (dual-attn's 2-KV-pair-
per-layer structure doesn't fit the single-pair cache format this note assumes).

### Known bugs fixed (do not reintroduce)
1. **`chunk_mask_fb` GT leak** (`train_hmn_chunk.py` ~line 467): IR turns' SLOT_A/argmax/SLOT_B
   rows must be blocked from ALL prior rec_block outputs (`is_any_rec_output`), not just
   source chunks. During training those positions hold teacher-forced GT — a direct attention
   path lets the model shortcut, collapsing at AR-decode eval. Fixed by adding `is_any_rec_output`
   union and ORing it into the block condition for IR rows. Still active in every mask today.

2. **KV decode off-by-one** (`ar_decode_chunk_fb_kv` and `ar_decode_chunk_fb_stitch_kv`
   `_decode_segment`): after generating `out_len` tokens, the LAST token was written into
   `tok` but never processed through the model to add its KV to the cache. This left
   `L_cached = c1-1` instead of `c1`, causing all subsequent blocks to receive RoPE positions
   shifted by -1. Eval showed 0% match throughout all 80k training steps despite correct
   training loss; teacher-forced BPB was near-zero (model trained fine, only eval was broken).
   Fixed by adding one extra forward pass after the output loop to cache the last decoded token.
   Same fix pattern reused in every later `ar_decode_*_kv` function in this project.

3. **Non-KV decode argmax chaining** (`ar_decode_chunk_fb`): original code used
   `ir_rbs = {span: rb}` dict which only kept the LAST IR block per span, then filled
   its argmax from IQ directly (skipping intermediate IR steps). Fixed by processing
   `rec_blocks` in sequence order with `last_ir_by_span` tracking.

### Metrics (still the convention used everywhere today)
- **val**: `make_test_sequences` split into n_chunks (`val_n_seqs` config knob caps
  how many of the 8 deterministic sequences are used, for faster iteration)
- **test**: specific surah file, padded to (n_chunks, chunk_len), eval-only
- **BPB**: teacher-forced NLL/ln(2) on the final recall block's own output (single
  window) or the full stitched sequence (multi-window)
- **match%**: AR greedy exact-match, same scope as BPB above

---

## Old scaling-track results table

| Run | Result |
|-----|--------|
| `hmn_feedback_32_ir` | **100% at k=0..4 AND k=0..12 extrapolation** ✓✓ |
| `hmn_chunk_local_32` stage 1 (IQ, 50k) | 81.9% match — solid IQ pretraining ✓ |
| `hmn_chunk_local_32_stage1` (IR, 80k) | **87.5% match** with fixed KV decode ✓ |
| `hmn_chunk_local_64` v1 (64B, pure stitch, 80k) | stitch=76.8%, win0=85.9%, win1=**0%**, win2=13% — chaining failure |
| `hmn_chunk_local_64_v2` (64B, 4-way equal mix, 80k) | **stitch collapsed to 6.2%** — equal mix gave stitch only 25% of steps, insufficient |
| `hmn_chunk_local_64_v3` (64B, targeted fix: stitch×3+win1×1+win2×1, from v1 end) | **failed** — stitch oscillated 58→43→35→45→41%, win2 collapsed at 60k. Killed. |
| `hmn_chunk_local_64_v4` (64B, mask_nochain=True, pure stitch, from stage2 end) | **failed** — win1=0%, win2=12.5% indep; stitch=53.6% best. nochain blocked SLOTs only, model chained through OUTPUT tokens |
| `hmn_chunk_local_64_v5` (64B, mask_nochain=True corrected, pure stitch, from stage2 end) | succeeded — v5 fix: IQ SLOT blocked from ALL prior rec_block tokens (SLOT+warmup+output) |
| **Global IQ rw track** | |
| `hmn_chunk_global_iq_rw_nc4_slot4` → ext2 (slot4, IQ only, 150k+ steps) | 18.1% best — Win A BPB ~3.5-4.0, slot capacity bottleneck confirmed |
| `hmn_chunk_global_iq_rw_nc4_wina_ovs` (slot4 + 2× Win A oversample, from ext2) | 18.1% best — Win A BPB 2.128 (new low), 12.5% match peak. Capacity not distribution. |
| `hmn_chunk_global_iq_rw_nc4_slot8` → ext (slot8, IQ only, 80k+45k=125k total) | **44.0% best** — Win B/C improving, Win A BPB=1.283 (plateau) |
| `hmn_chunk_global_iq_rw_nc4_slot8_ir` (slot8 + n_refine=2 IR, from slot8_ext best) | **57.4% best** (step 75k) — Win B 97%, Win C 74%, Win A 3% (IQ=0% throughout — is_clean bug) |
| `hmn_chunk_global_iq_rw_nc4_slot8_ir_v2` (slot8 + IQ fix + Win A oversample, 100k) | **77.8% best** @ step 60k — Win A **100%** ✓, Win B 77.8%, Win C 55.6%. Final (100k): 75.5% |
| All HMN v3 variants (mono, cerb, p2, p4, pinf, tlogit) | ~95% k=0, collapses at k>4 |

### IR vs IQ-only — the decisive comparison

Win B down_counter BPB: **0.350** (IQ-only after 80k steps) → **0.039** (IR after 15k steps) — 9× BPB reduction in 5× fewer steps. IQ-only was plateaued; IR unlocked it in one stage. This mirrors the 32B result where IR lifted 50%→100% match. See `docs/FEEDBACK_RESULTS.md` for full analysis. This is the same IR-necessity finding that later reappeared (much more dramatically) in the dual-attn+RMSNorm results.

### slot8_ir_v2 key findings (see `docs/FEEDBACK_RESULTS.md` for full tables)

- **IQ=0% root cause fixed**: `is_clean=(n_refine==0)` excluded IQ from loss when n_refine>0. Fix: add `iq_global_rw` (n_refine=0) entries at 33% of traj_mix — IQ loss is present, model learns one-shot recall.
- **Win A solved at step 45k** (100% MEAN), stays solved at 100k. IQ training (Win A X=0 at 22% of steps) + IR training (Win A X=0 at 44%) unlocks it.
- **Best checkpoint is cycle 2 end (step 60k)**: cosine_T0=20k → cycle ends at 20k/60k/140k. Cycle 3 restart introduces volatility; Win C collapses at step 70k, never fully recovers.
- **IR2 corrects IR1 regressions**: Win A at step 100k shows IR1=45-62% but IR2=100% — the two-turn chain is qualitatively different from single IR.
- **Win C `up_counter` unsolved** (4.2% at end): IQ reaches 37.5% at step 100k but IR1 destroys it. Root cause: traj_mix gives Win C X=0 only ~0.7% of IR steps (uniform coverage) vs Win A X=0 at 44%. IR learned on other (window, X) pairs applies wrong corrections for Win C at X=0. Fix: add `warmup_x_fixed=16` (Win B) and `warmup_x_fixed=32` (Win C) IR entries to traj_mix. **Source distribution (infinite random bytes) is correct** — EXP1 showed it converges fastest; Win C IQ BPB trends down steadily, confirming algorithm learning not distribution noise. See `docs/FEEDBACK_RESULTS.md § Dataset design`.

This "last window is hardest" pattern (`up_counter` specifically) recurred throughout the project's history — most recently as window G's stuck "IR2 destroys IR1's gain" pathology in `srs_stitch_nc8_slot8` and the current `dualattn_nc8_slot8_ir` window-G fix test.

---

## Generalization Axes (early planning doc — mostly superseded by actual progress)

Written early in the project to enumerate what was held fixed. Kept for historical
context; most axes below have since been explored or superseded by later work (noted
inline where applicable).

### Source
- `src_len` — fixed per stage (32B→64B→128B→256B); NOW variable via the nc4→nc8→juz1 scaling track
- Content type — random bytes; real use needs natural text / structured data — NOW the explicit `juz1.txt` goal
- Encoding resolution (`chunk_len`) — fixed at 16B/chunk, still true today

### Window geometry
- Window size — fixed at 32B (2 chunks), still true today
- Stride / overlap — fixed at 16B (50%), still true today
- Window boundary alignment — fixed to chunk grid; still true today
- Window order during stitch — fixed left→right; still true for training, but the `juz1` random-offset sampling design explores unordered training while preserving ordered eval-time stitching

### Recall
- **Warmup position** — fixed at window start; still true today (Tier 2's "any X" goal was never pursued — turned out unnecessary)
- Warmup length — fixed at 8B, still true today
- Query type — always "recall forward from warmup"; still true today

### Refinement
- **`n_refine` per window** — fixed at 2 in every proven recipe since; adaptive refinement never implemented
- Refinement schedule across SRS sessions — still open-loop; the `juz1` hybrid priority sampler (`docs/SRS_RECIPE.md`) is the closest realization of adaptive/closed-loop scheduling attempted so far

### Slot / memory
- **`slot_len`** — grew from 4 to 8 as scale increased (128B+); slot_count still fixed at 2
- Slot scope — still one slot per window in every proven recipe; the chain-memory design (`docs/SRS_RECIPE.md § Chain memory`) is the first real attempt at cross-window slot consolidation
- **Slot consolidation** — chain-memory's `CHAIN_SLOT` is the first concrete design for this

### Training distribution
- Single source per batch — still true in every synthetic-random-byte run; real-corpus training (`juz1`) will need this to change (flagged in `docs/SRS_RECIPE.md`'s batch-size discussion)
- Train/test sequence type — synthetic random bytes (val) vs real text (test) has been the convention since early on
- Curriculum order — fixed stage sequence in every recipe so far; the `juz1` hybrid priority sampler is the first adaptive-order design

### Model
- Model size (`d`, `n_layers`) — still fixed at d=64, 4 layers across every architecture tried, including dual-attn
- Positional encoding range — RoPE+YaRN, untested at long range until `juz1`
