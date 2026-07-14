# KV Memory Dimensions

## Formula

```
d_head    = d / n_heads
KV_floats = 2 × n_layers × active_slots × d     (K+V, all layers, n_heads cancels)
KV_bytes  = 4 × KV_floats                        (float32)
ratio     = src_bytes / KV_bytes                 (>1 = compression, <1 = expansion)
```

Cell format: `KV_bytes (src:KV ratio)`. Ratio `1:N` means each source byte maps to N KV bytes.

---

## d=128, n_layers=4

`KV_bytes = 4096 × active_slots`

| seg\slots | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| **8** | 4.0KB (1:512.0) | 8.0KB (1:1024.0) | 16.0KB (1:2048.0) | 32.0KB (1:4096.0) | — | — | — | — | — |
| **16** | 4.0KB (1:256.0) | 8.0KB (1:512.0) | 16.0KB (1:1024.0) | 32.0KB (1:2048.0) | 64.0KB (1:4096.0) | — | — | — | — |
| **32** | 4.0KB (1:128.0) | 8.0KB (1:256.0) | 16.0KB (1:512.0) | 32.0KB (1:1024.0) | 64.0KB (1:2048.0) | 128.0KB (1:4096.0) | — | — | — |
| **64** | 4.0KB (1:64.0) | 8.0KB (1:128.0) | 16.0KB (1:256.0) | 32.0KB (1:512.0) | 64.0KB (1:1024.0) | 128.0KB (1:2048.0) | 256.0KB (1:4096.0) | — | — |
| **128** | 4.0KB (1:32.0) | 8.0KB (1:64.0) | 16.0KB (1:128.0) | 32.0KB (1:256.0) | 64.0KB (1:512.0) | 128.0KB (1:1024.0) | 256.0KB (1:2048.0) | 512.0KB (1:4096.0) | — |
| **256** | 4.0KB (1:16.0) | 8.0KB (1:32.0) | 16.0KB (1:64.0) | 32.0KB (1:128.0) | 64.0KB (1:256.0) | 128.0KB (1:512.0) | 256.0KB (1:1024.0) | 512.0KB (1:2048.0) | 1.0MB (1:4096.0) |
| **512** | 4.0KB (1:8.0) | 8.0KB (1:16.0) | 16.0KB (1:32.0) | 32.0KB (1:64.0) | 64.0KB (1:128.0) | 128.0KB (1:256.0) | 256.0KB (1:512.0) | 512.0KB (1:1024.0) | 1.0MB (1:2048.0) |
| **1024** | 4.0KB (1:4.0) | 8.0KB (1:8.0) | 16.0KB (1:16.0) | 32.0KB (1:32.0) | 64.0KB (1:64.0) | 128.0KB (1:128.0) | 256.0KB (1:256.0) | 512.0KB (1:512.0) | 1.0MB (1:1024.0) |

---

## d=256, n_layers=8

`KV_bytes = 16384 × active_slots`

| seg\slots | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| **8** | 16.0KB (1:2048.0) | 32.0KB (1:4096.0) | 64.0KB (1:8192.0) | 128.0KB (1:16384.0) | — | — | — | — | — |
| **16** | 16.0KB (1:1024.0) | 32.0KB (1:2048.0) | 64.0KB (1:4096.0) | 128.0KB (1:8192.0) | 256.0KB (1:16384.0) | — | — | — | — |
| **32** | 16.0KB (1:512.0) | 32.0KB (1:1024.0) | 64.0KB (1:2048.0) | 128.0KB (1:4096.0) | 256.0KB (1:8192.0) | 512.0KB (1:16384.0) | — | — | — |
| **64** | 16.0KB (1:256.0) | 32.0KB (1:512.0) | 64.0KB (1:1024.0) | 128.0KB (1:2048.0) | 256.0KB (1:4096.0) | 512.0KB (1:8192.0) | 1.0MB (1:16384.0) | — | — |
| **128** | 16.0KB (1:128.0) | 32.0KB (1:256.0) | 64.0KB (1:512.0) | 128.0KB (1:1024.0) | 256.0KB (1:2048.0) | 512.0KB (1:4096.0) | 1.0MB (1:8192.0) | 2.0MB (1:16384.0) | — |
| **256** | 16.0KB (1:64.0) | 32.0KB (1:128.0) | 64.0KB (1:256.0) | 128.0KB (1:512.0) | 256.0KB (1:1024.0) | 512.0KB (1:2048.0) | 1.0MB (1:4096.0) | 2.0MB (1:8192.0) | 4.0MB (1:16384.0) |
| **512** | 16.0KB (1:32.0) | 32.0KB (1:64.0) | 64.0KB (1:128.0) | 128.0KB (1:256.0) | 256.0KB (1:512.0) | 512.0KB (1:1024.0) | 1.0MB (1:2048.0) | 2.0MB (1:4096.0) | 4.0MB (1:8192.0) |
| **1024** | 16.0KB (1:16.0) | 32.0KB (1:32.0) | 64.0KB (1:64.0) | 128.0KB (1:128.0) | 256.0KB (1:256.0) | 512.0KB (1:512.0) | 1.0MB (1:1024.0) | 2.0MB (1:2048.0) | 4.0MB (1:4096.0) |

---

## d=64, n_layers=4 — Model Size + KV Budget

```
KV per token per layer: 2 × d × 4 bytes = 2 × 64 × 4 = 512 bytes
KV per token (4 layers): 512 × 4 = 2048 bytes

MEM block (slot_len=S, BLEN=S+2):
  S=0 → BLEN=2 → 4096 bytes KV
  S=1 → BLEN=3 → 6144 bytes KV
  S=2 → BLEN=4 → 8192 bytes KV

Window=2 (2 MEM blocks cached at inference):
  S=1 → 2 × 6144 = 12288 bytes = 12 KB
```

### Model parameter count (d=64, n_layers=4, n_heads=4, d_ff=256, vocab=268)

```
Embedding:          268 × 64                 =  17,152
Per layer:
  LayerNorm ×2:     2 × 2 × 64              =     256
  Wq,Wk,Wv,Wo:      4 × 64×64               =  16,384
  W1: 64×256                                 =  16,384
  W2: 256×64                                 =  16,384
  biases (b1,b2):   256+64                   =     320
  per layer                                  =  49,728
4 layers:           4 × 49,728               = 198,912
Final LayerNorm:    2 × 64                   =     128
LM head (tied):                              =       0

Total params:  216,192
fp32 size:     216,192 × 4 = 864,768 bytes ≈ 845 KB
```

### 1 MB total budget (model + KV, slot_len=1, window=2)

```
Model:    845 KB
KV cache:  12 KB
Total:    857 KB  ← fits under 1 MB
```

### "1 float = 1 byte" capacity check (slot_len=1, window=2)

KV floats available at recall: 2 × 3 × 4 × 128 = 3072 floats (K) + 3072 (V) = 6144 floats
Data to represent (8 × 128 bytes): 1024 bytes
Headroom: 6144 / 1024 = **6×** — sufficient.

---

## SRS Training Trajectory — 8 × 128 bytes in 2048 tokens

Config: `src_len=128, slot_len=1, BLEN=3, out_len=128, warmup_len=0`

```
Turn cost: BLEN + src_len = 131 tokens
Fits in 2048: (N+1) × 131 = 2048 → N = 14 turns
```

SRS schedule (10 inserts + 4 mandatory revisits for oldest sequences):

```
t0:  I_1    insert seq1
t1:  I_2    insert seq2
t2:  R_1    early review (gap=2, high forgetting risk)
t3:  I_3
t4:  R_2    review seq2 (gap=3)
t5:  I_4
t6:  R_1    spaced review (gap=4)
t7:  I_5
t8:  I_6
t9:  R_3    review seq3 (gap=6)
t10: I_7
t11: R_4    review seq4 (gap=6)
t12: I_8
t13: R_1    long review (gap=7, confirms retention)
Q:   Q_x   uniform sample from {1..8}
```

Reviews: seq1=3, seq2/3/4=1, seq5–8=0 (passive survival test).
Sequences without review must survive 6–14 steps via MEM carry-through only — this is the hard compression signal.
