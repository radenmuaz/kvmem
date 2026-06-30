# kvmem

Fast-weight language model — HashMemNet (HMN). Two architectures: v3 (MEM blocks) and **Feedback** (argmax loop, current focus).

**Read these docs to resume:**

| Priority | Doc | Why |
|----------|-----|-----|
| 1 | [`docs/FEEDBACK_RESULTS.md`](docs/FEEDBACK_RESULTS.md) | **Key result**: feedback arch achieves 100% k=0..12, why it works, bug history |
| 2 | [`docs/BOOK.md`](docs/BOOK.md) § 8 | HMN v3 architecture reference (predecessor) |
| 3 | [`docs/kv_dims.md`](docs/kv_dims.md) | KV capacity math, model size, SRS multi-sequence design |

---

## Current Status

**Feedback architecture solved the refinement problem. Now extending to multi-sequence SRS chunk memorization.**

| Run | Result |
|-----|--------|
| `hmn_feedback_32_ir` | **100% at k=0..4 AND k=0..12 extrapolation** ✓✓ |
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

| Stage | n_chunks (src) | windows (chunk-idx) | n_refine | steps | Config |
|---|---|---|---|---|---|
| 1 | 2 (32B) | `[(0,2)]` | 0 | 50000 | `hmn_chunk_local_32.py` (stage 0) |
| 2 | 2 (32B) | `[(0,2)]` | 2 | 80000 | `hmn_chunk_local_32.py` (stage 1) |
| 3 | 4 (64B) | `[(0,2),(1,3),(2,4)]` | 2 | 80000 | `hmn_chunk_local_64.py`, resumes stage 2 |

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

### Pending follow-up
After stage 3 finishes: eval **zero-shot** full-span (0-64) IQ-only recall (single
window `[(0,4)]`, `n_refine=0`, no training) against stage 3's checkpoint — tests
whether stitching transfers to an explicitly-untrained full-span read. Expected to fail
(motivates either explicit full-span IQ training or accepting windowed-only stitching).

### Parked (not deleted) — overlapping-window **stitching** experiment
`chunk_positions_fb_stitch` + `ir_stitch` trajectory type + configs
`hmn_chunk_stitch_a/b/smoke.py` — global IQ + per-window IR pairs + optional trailing
full-span stitched IQ turn. Larger-scale alternative design, deferred — do not delete.

### Run commands
```bash
# Stage 1+2 (random init)
caffeinate -i python3 -m kvmem.train_hmn_chunk \
    --config configs/hmn_chunk_local_32.py --device mps

# Stage 3 (resumes stage 2's checkpoint)
caffeinate -i python3 -m kvmem.train_hmn_chunk \
    --config configs/hmn_chunk_local_64.py \
    --pretrained logs/hmn_chunk_local_32/checkpoints/stage1_end.pt \
    --device mps
```

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

## Key Principles

- `null_kv=True` always — 1.5–2× faster, better bpb
- Mask convention: `0.0`=attend, `-1e9`=blocked (additive bias for `F.scaled_dot_product_attention`)
- IQ stage before IR — always required for feedback arch
- Feedback eval: use inline script above (no `--eval-only` in train_hmn_feedback yet)

---

## Docs

| What | Where |
|------|-------|
| **Feedback results + architecture** | [`docs/FEEDBACK_RESULTS.md`](docs/FEEDBACK_RESULTS.md) |
| HMN v3 reference book | [`docs/BOOK.md`](docs/BOOK.md) |
| KV capacity + SRS trajectories | [`docs/kv_dims.md`](docs/kv_dims.md) |
| All configs | [`configs/`](configs/) |
| **Chunk SRS / local-refine training** | [`kvmem/train_hmn_chunk.py`](kvmem/train_hmn_chunk.py) |
| Current local-refine configs | [`configs/hmn_chunk_local_32.py`](configs/hmn_chunk_local_32.py), [`configs/hmn_chunk_local_64.py`](configs/hmn_chunk_local_64.py) |
| Parked stitching-experiment configs | [`configs/hmn_chunk_stitch_a.py`](configs/hmn_chunk_stitch_a.py), `_b.py`, `_smoke.py` |
| Feedback training | [`kvmem/train_hmn_feedback.py`](kvmem/train_hmn_feedback.py) |
| HMN v3 mono training | [`kvmem/train_hmn_mono.py`](kvmem/train_hmn_mono.py) |
| Test set | [`datasets/suratalfatihah.txt`](datasets/suratalfatihah.txt) |
