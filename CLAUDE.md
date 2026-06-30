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

## Chunk Memorization — Feedback SRS Architecture (current work)

**Redesigned architecture**: depth-2 SRS with explicit feedback argmax IR turns + chunked attention.

**Training script**: [`kvmem/train_hmn_chunk.py`](kvmem/train_hmn_chunk.py) — use `train_fn='fb'` in config for feedback arch.

### Sequence layout (single forward pass, feedback IR)
```
Encoding:  [chunk_0: C][SLOT×s] ... [chunk_{N-1}: C][SLOT×s]

Per SRS span (depth-2: halves + full):
  Turn 0 (IQ):  [SLOT_0: s][warmup: wl][out_0: ol]           ← initial recall
  Turn 1 (IR):  [SLOT_A: s][argmax_0: ol][SLOT_B: s][warmup: wl][out_1: ol]  ← feedback
```
- `argmax_0` = model's own greedy output from turn 0 (detached, 2 forward passes/step)
- `warmup_len=8`, `ol = span_len - wl`
- Mask: IR warmup/out blocked from everything except own SLOT_B + own warmup/out
- Chunked attention: `chunk_attn=512` to bound peak memory to O(L×chunk)

### Depth-2 SRS schedule
```python
srs_schedule_depth2(N):  # N=2 → [(0,1),(1,2),(0,2)];  N=4 → [(0,2),(2,4),(0,4)]
    halves + full  (3 spans always)
```

### Training queue (RESET — feedback SRS arch)

| # | Config | Size | Test set | Status |
|---|--------|------|----------|--------|
| sanity | `hmn_chunk_sanity` | 256B (2×128) | suratalkauthar.txt (179B) | **RUNNING** (step ~130/20k) |
| full | `hmn_chunk_1024` | 1024B (4×256) | suratalfatihah.txt (556B) | queued if sanity passes |

**Decision**: if sanity BPB improves and test_match > 5% → launch full 1024 config.

### Run commands
```bash
# Sanity (running)
caffeinate -i python3 -m kvmem.train_hmn_chunk \
    --config configs/hmn_chunk_sanity.py \
    --pretrained logs/hmn_feedback_32_ir/checkpoints/stage0_end.pt \
    --device mps

# Full 1024 — after sanity passes
caffeinate -i python3 -m kvmem.train_hmn_chunk \
    --config configs/hmn_chunk_1024.py \
    --pretrained logs/hmn_feedback_32_ir/checkpoints/stage0_end.pt \
    --device mps
```

### Completed runs (old arch, for reference)

| Run | Best BPB | Best test_match | Notes |
|-----|----------|-----------------|-------|
| `hmn_chunk_fine` (baseline, wl=0) | 7.41 | 8.0% | stage 1 best |
| `hmn_chunk_fine_wm` (q1, wl=8) | **7.26** | **10.6%** | stage 1 best, beats baseline |

### Metrics
- **val**: `make_test_sequences` split into n_chunks (8 deterministic sequences, in-distribution)
- **test**: specific surah file, padded to (n_chunks, chunk_len), eval-only, never trained on
- **BPB**: teacher-forced NLL/ln(2) on IR `out_1` of final full-sequence span
- **match%**: AR greedy exact-match on IR `out_1` of final full-sequence span

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
| **Chunk SRS training** | [`kvmem/train_hmn_chunk.py`](kvmem/train_hmn_chunk.py) |
| Feedback training | [`kvmem/train_hmn_feedback.py`](kvmem/train_hmn_feedback.py) |
| HMN v3 mono training | [`kvmem/train_hmn_mono.py`](kvmem/train_hmn_mono.py) |
| Test set | [`datasets/suratalfatihah.txt`](datasets/suratalfatihah.txt) |
