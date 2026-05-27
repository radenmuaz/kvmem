# KV-as-Fast-Weights — Implementation Plan (Revised)

## 1. Project Context

The project builds a transformer in which **new information is absorbed by writing to the KV cache rather than by gradient updates to weights**. Slow MLP weights hold procedural skills (read, write, retrieve, compress); the KV cache holds declarative content. "Training on new data" at deployment becomes a single forward pass — no backprop at inference.

Stages 0 and 1 validate the core primitive on a synthetic Markov-chain task before any real-text or streaming work.

---

## 2. Architectural Decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Architecture | Decoder-only, custom attention mask | Simpler than encoder-decoder |
| Position embeddings | **None (NoPE)** | Arbitrary N; better length generalization |
| Memory tokens | NUL (`0x00`) × N, bracketed by STX/ETX | Position-based identity, not token ID |
| Strength gating | **None** | Removed; mask enforces bottleneck |
| Training target | **NTP on independent continuation** | IPTT (arXiv:2604.06169); no copy shortcut |
| Mask design | Y write-only sink; Y can't see S; cross-Y blocked | Train/inference consistency |
| Vocab | **Byte-level V=256** | Realistic; no augmentation needed |
| Variable N | **Sampled per batch** from N_set | Single model handles any N |
| Optimizer | **AdamW** (default) or **GrokAdamW** (arXiv:2605.01172) | Comparison experiment |

---

## 3. Segment Protocol

### 3.1 Byte registry

| Open | Close | ASCII | Segment | Stage |
|---|---|---|---|---|
| `0x02` | `0x03` | STX / ETX | Memory write region (M) | 0 |
| `0x04` | `0x05` | EOT / ENQ | Continuation / query (Y) | 0 |
| `0x06` | `0x07` | ACK / BEL | SRS rehearsal chunk | 3 |
| `0x08` | `0x09` | BS / HT | Self-eval probe | 4 |

Data bytes constrained to `[0x20, 0xFF]`. Protocol bytes `0x00–0x1F` never appear in data.

### 3.2 Memory block

```
memory_block(N) = [ STX | NUL×N | ETX ]   length N+2
```

Slot identity comes from **causal depth** (NoPE), not token ID. Collapse fix: increase n_layers.

---

## 4. KV Size Rule of Thumb

```
KV_floats = 2 × n_layers × N × d
```

With current model (d=64, n_layers=4):

| N | KV floats | % of 165K params |
|---|---|---|
| 2  | 1,024  | 0.6% |
| 8  | 4,096  | 2.5% |
| 32 | 16,384 | 9.9% |

---

## 5. Current Hyperparameters (stage 0, calibrated)

| Param | Value | Rationale |
|---|---|---|
| V | 256 | byte vocab |
| V_chain | 32 | 32-state Markov; only ~35 tokens ever active |
| L_S | 64 | source length |
| N_set | {2,4,8,16,32} | variable-N training |
| d | 64 | matched to task; 165K total params |
| n_layers | 4 | — |
| n_heads | 4 (d_head=16) | — |
| d_ff | 128 | — |
| alpha | 0.1 | peaked Dirichlet → high MI between x_S and y |
| L_y schedule | [(0,16),(10k,32),(25k,64)] | curriculum |
| B | 64 | batch size |
| lr_max | 3e-4 | — |
| warmup | 500 steps | — |
| n_steps | 20,000 | — |
| loss | 0.1×L_src + L_cont | src auxiliary at 0.1 weight |
| optimizer | adamw \| grokadamw | comparison experiment |

**Why smaller model than original plan:** Only 35 of 256 vocab entries are ever active for V_chain=32. d=128 (854K params) is severely overparameterized for a 32-state chain task. d=64 (165K) is right-sized and trains ~4× faster.

---

## 6. Stage 0 — Single-pass NTP through the KV bottleneck

### 6.1 Sequence layout

```
[ x_S (L_S) | STX | NUL×N | ETX | y (L_y) ]
  S           M_open  M      M_close  Y
```

### 6.2 Attention mask

- S sees S causally
- M sees S and causally earlier M
- Y sees M, ETX, and causal Y — **cannot see S or STX**
- Y is write-only sink (nothing outside Y attends to Y)

### 6.3 Loss

```
total = 0.1 × L_src + L_cont
```

`L_src` auxiliary at 0.1 weight teaches chain-structure reading (needed for KV writing). `L_cont` on Y positions drives the bottleneck. Padding positions excluded via bounded `mask_cont`.

### 6.4 Padding strategy (no JIT retracing)

All tokens padded to `L_max = L_S + 2 + N_max + L_y_max`. Masks padded similarly. `train_step` JIT-compiled with `static_argnums=(N, L_y)` — 15 traces max, then cached.

### 6.5 Eval conditions

For each (N, L_y) in grid:
- **matched**: y continues same chain as x_S
- **cross**: y from different chain
- **uniform**: x_S is random noise

```
gain(N, L_y) = bpt_uniform − bpt_matched   [higher = better KV use]
penalty      = bpt_cross − bpt_uniform      [should be negative or near 0]
SCR          = L_S / N                      [semantic compression ratio]
```

### 6.6 Success criteria

| Criterion | Required |
|---|---|
| `bpt_matched < bpt_uniform` | Yes |
| `bpt_matched` decreasing as N grows | Trend |
| `gain > 0` for N ≥ 4 | Yes |

### 6.7 Post-training test (auto-runs)

1. **Per-line continuation** on `datasets/suratalfatihah.txt` — memorize each of 7 ayat into KV, give 4-byte warmup, complete rest, report byte-match %
2. **Whole-file continuation** — memorize all 7 ayat concatenated, give first line as prompt, generate rest

Validation during training: `datasets/1.txt` byte-match every 1000 steps.

### 6.8 Optimizer comparison

Two sequential runs:
- `--optimizer adamw` (default)
- `--optimizer grokadamw` (SNR-gated AdamW, arXiv:2605.01172, `--grok-rho 0.9`)

Report: loss curves, val byte-match, test byte-match, convergence speed.

---

## 7. Stage 1 — Multi-pass NTP refinement  *(after stage 0 success)*

### 7.1 Sequence layout

```
[ x_S | MB | y^(1) | MB | y^(2) | ... | MB | y^(T) ]
where MB = [ STX | NUL×N | ETX ]   (N+2 tokens)
Total L = L_S + T×(N+2+L_y)
```

Each y^(t) is an **independent fresh continuation** from the same chain terminal state.

### 7.2 Attention mask

- M^(t) sees S and all prior M^(s≤t)
- Y^(t) sees M^(≤t) and ETX of own block — **cannot see S, other Y blocks, or M^(>t)**
- Y is write-only sink globally

### 7.3 Loss — focused multi-pass (from PLAN_STAGE1.md)

**Pass weighting:** `beta_t = t / Σ(1..T)` — linear ramp, later passes weighted more.

**Pass 1:** standard unweighted NTP loss.

**Passes t≥2:** hard-position focused loss:
```
w^(t)_{b,k} = stop_grad( clip( e^(t-1)_{b,k} − tau,  0,  w_max ) )
L_cont,t_focus = Σ (1 + λ_w × w^(t)) × e^(t) / Σ (1 + λ_w × w^(t))
```
Positions where pass t-1 failed get higher weight in pass t's loss. Amortized at training time — model unchanged at inference.

**Monotonicity regularizer (optional):**
```
L_mono = Σ_{t≥2} ReLU( L_cont,t_base − L_cont,t-1_base + gamma )
```

**Total:**
```
L_total = 0.1×L_src
        + λ_cont × L_cont,1_base
        + λ_cont × Σ_{t≥2} beta_t × L_cont,t_focus
        + λ_mono × L_mono
```

**Stage 1 hyperparameters:**
| Param | Value |
|---|---|
| T | sweep {1, 2, 4} |
| lambda_cont | 2.0 |
| lambda_w | 2.0 |
| lambda_mono | 0.1 |
| tau | 0.693 (= log 2) |
| w_max | 3.0 |
| gamma | 0.0 |

### 7.4 Success criteria

| Criterion | Required |
|---|---|
| Monotone `bpt_cont(t+1) ≤ bpt_cont(t)` | Yes |
| `bpt_cont(T=4) < bpt_cont(T=1)` by ≥ 0.1 bits | Yes |
| Truncation: T=1 from T=4 model ≥ dedicated T=1 | Yes |
| Focused loss: pass-1 quality not degraded vs vanilla | Yes |
| `L_mono` regression < 5% of steps at convergence | Sanity |

### 7.5 Ablation

Train A (vanilla multi-pass) vs B (focused multi-pass). Compare at truncation t=1.

---

## 8. Stage 2+ — Probe-based streaming *(not yet)*

At stage 2, ingestion-time ground truth enables direct probing instead of a learned self-eval head:

```python
def probe(model, M, chunk):
    logits = model(chunk, prefix_cache=M)
    return -log_softmax(logits[:-1])[range(len(chunk)-1), chunk[1:]]
```

Scheduling logic: plain Python comparing `current_nll` vs `baseline_nll + delta`. No learned head. No SRS controller. The model stays a pure forward function.

This replaces planned stages 4 (self-eval head) and 5 (SRS controller) with simpler, more tractable alternatives.

---

## 9. File Layout

```
kvmem/
├── data.py       # Markov chain dataset, masks, text file I/O, dataset write/load
├── stage0.py     # Stage 0: single-pass training, N sweep eval, test, optimizer compare
└── stage1.py     # Stage 1: multi-pass, focused loss, T sweep, truncation diagnostic
```

No optax. No flax. JAX + Equinox only.

---

## 10. Stage 0 Go/No-Go for Stage 1

Stage 1 starts when stage 0 achieves:
- `gain(N=8, L_y=32) > 0` on matched vs uniform eval
- Loss curve clearly descends below oracle floor (~1.73 nats for V_chain=32, alpha=0.1)
- Both AdamW and GrokAdamW runs complete; comparison recorded

If stage 0 fails (loss stays near marginal entropy 3.47 nats):
- Check src loss: if src > 2.5 nats, model isn't reading chain structure → increase n_layers or d
- Check gate fraction in GrokAdamW: if < 10% params updated per step, rho too aggressive → lower rho
- Try reconstruction loss (y = x_S, no bottleneck mask) as warmup phase before enabling bottleneck

---

## 11. Acceptance Summary

| Artifact | Done |
|---|---|
| `data.py` — Markov chains, masks, numpy prefetcher, text I/O, .bin/.txt dataset | ✅ |
| `stage0.py` — training loop, AdamW + GrokAdamW, validation, auto-test | ✅ (training) |
| Stage 0 convergence (gain > 0) | ⏳ |
| Stage 0 test on suratalfatihah.txt | ⏳ |
| `stage1.py` — multi-pass, focused loss, T sweep | ⏳ |
| `reports/stage01_summary.md` | ⏳ |
