# Model Size vs Task Complexity — Algorithms & MDL Analysis

## The actual function being learned

The model must implement a **universal chunk encoder-decoder**: for *any* 16-byte chunk drawn from {0,...,255}^16 (2^128 possible inputs), compress it into 4 SLOT tokens and decode it back given 8 warmup bytes. This must hold for all 2^128 inputs simultaneously — the model cannot memorize examples, it must learn a systematic algorithm.

This is strictly harder than fitting a dataset. It is closer to learning a **bijective hash function** via gradient descent.

---

## MDL breakdown

| component | bits |
|---|---|
| Model description (231k params × fp16) | ~3.7 Mbits |
| Kolmogorov complexity of a bijection {0,255}^16 → R^256 | ~128 bits |
| Ratio | ~29,000× overhead |

That 29,000× ratio is not waste — it is the **gradient-descent tax**: the function must be reachable by SGD from a random initialization over a smooth loss landscape. Random bijections are not SGD-reachable in any reasonable number of steps; a smooth parametric family large enough to cover the space is required.

---

## Minimum model size for the task

A transformer with d=32, 2 layers (~30k params) can in principle implement the per-chunk encoding:

- Layer 1 attention: SLOT tokens pool over raw chunk bytes
- Layer 2 attention: decoder output attends to SLOTs

So 231k params is roughly **4–8× the theoretical minimum** for a single-window 32B task. This overhead is typical for tasks requiring generalization over exponentially large input spaces under SGD.

---

## The key MDL insight — positional invariance

A position-*dependent* encoding function has **higher description length** than a position-invariant one. It requires more bits to specify:

> "encode chunk k at position p using rule f(k, p)"

vs.

> "encode chunk k using rule f(k)"

The model trained on pure stitch (v5) learned the longer description — SLOT behavior depends on the absolute token position of the enc_block in the sequence. This is a local SGD minimum that happens to fit the training distribution but generalizes poorly to new positions.

**Vlen training is MDL regularization.** Training at nc=2, 4, 8 penalizes position-dependent solutions (they fail at nc=2 if learned at nc=4) and reward position-invariant ones (the same algorithm works at any nc). This is not adding model capacity — it is constraining the SGD search to shorter, more general descriptions.

The bridge mechanism (nc=8 intermediate) provides a gradient path between the stitch position (enc_block[3] at distance ~330) and the independent eval position (enc_block[3] at distance ~1), forcing the model to find a solution that works at both. The position-invariant solution is the unique one that satisfies all constraints simultaneously.

---

## Dataset considerations

The training distribution is i.i.d. uniform random bytes — effectively an infinite dataset. Classical overparameterization analysis (VC dimension, bias-variance tradeoff) does not directly apply because there is no fixed test set to overfit to.

The relevant quantity is not "model size vs dataset size" but **"model description length vs target function description length."** The target function (position-invariant chunk encoder-decoder for arbitrary bytes) has a description length dominated by the bijection complexity (~128 bits/chunk) plus the positional invariance constraint. The model at 231k params provides ample capacity to represent this function; the challenge is SGD finding it.

---

## Scaling argument

For 128B (nc=8, 7 windows), the per-chunk algorithm is identical to the 64B case — more windows, same encoding rule. If the algorithm is truly position-invariant (short MDL description), the same 231k-param model handles it with no increase in size.

Parameter count scales with **algorithm complexity**, not sequence length. The right move when scaling to 128B is not to increase model size — it is to ensure the training distribution covers the new positional regime (pure stitch at nc=8 forces the model to maintain position-invariant encoding at all 7 windows).

This is why staged training is the correct approach: find the short description at 64B, then extend it to 128B. The model's capacity is not the bottleneck.

---

## Practical verdict

| claim | verdict |
|---|---|
| Model is overparameterized vs dataset | Not applicable — dataset is infinite |
| Model is overparameterized vs task algorithm | ~4–8× overhead, typical for SGD learnability |
| Model needs to grow for 128B | No — same algorithm, same size |
| Current failure (Win C nc=4 stuck) is a capacity problem | No — it is a training distribution problem |
| Fix: more parameters | Wrong — fix: stronger position-invariance constraint (vlen) |
| Fix if vlen fails: n_refine=0 IQ-only | Correct fallback — simpler algorithm, shorter description, easier for SGD to find |

If Win C nc=4 stalls despite vlen at 80k, the right response is **not** to add parameters. Options in order of MDL preference:

1. Broaden the bridge: add nc=6, 10, 12 to the traj mix (more constraints → shorter description forced)
2. IQ-only fallback (n_refine=0): removes the IR refinement complexity, simpler algorithm for SGD to find first
3. Increase model size only as a last resort and only if the above both fail

---

## Connection to SRS scaling

At the SRS scale (256B+, many sequences with independent retention clocks), the same analysis holds:

- The per-chunk encoding algorithm is the same at all scales
- The number of parameters does not need to grow with corpus size
- What grows is the context length and the number of SLOT tokens in the sequence — both handled by RoPE+YaRN, not by more weights
- MDL predicts: a model that has found the short position-invariant description at 64B will generalize to 256B+ without retraining from scratch, only curriculum extension

---

## Fractal synthetic data vs real data — training distribution design

### Why natural language is approximately fractal

Natural language has self-similar statistical structure across scales (character → word → sentence → paragraph). Key measurable properties:

| property | random bytes | 1/f synthetic | n-gram synthetic | fractal+n-gram | real text |
|---|---|---|---|---|---|
| Entropy rate | 8 bit/byte | ~4 bit/byte | ~2 bit/byte | ~1.5 bit/byte | ~1 bit/byte |
| Hurst exponent H | 0.5 (white) | ~0.75 | ~0.6 | ~0.75 | ~0.8 |
| Spectral density | flat (white noise) | 1/f^α | roughly 1/f | 1/f | 1/f^α (α≈1) |
| Local n-gram statistics | none | none | ✓ tunable | ✓ | ✓ |
| Hierarchical grammar | none | none | none | none | ✓ |
| Semantic coherence | none | none | none | none | ✓ |

### What fractal generators can cover

- **1/f (pink) noise mapped to bytes**: matches spectral density and Hurst exponent. Byte sequences with correct long-range correlation but no local structure.
- **Iterated Function Systems (IFS)**: tunable Hausdorff dimension, self-similar structure. Matches fractal dimension of text character distributions.
- **Multifractal random measures**: match varying local entropy (dense/sparse information alternation, as in content words vs function words).
- **n-gram Markov chains with power-law transition matrices**: match local byte statistics up to order n. Combined with 1/f envelope → fractal+n-gram row above.
- **Stochastic context-free grammars**: generate hierarchical nested structure — closest to syntactic grammar, but still no semantics.

What fractal data cannot cover: hierarchical semantic coherence, topic-level long-range dependencies, and the mixture-of-domain statistics in real corpora (text/code/tables/math all have different local entropy profiles).

### Model memory: what survives training

| | what it holds | persists? |
|---|---|---|
| Weights | encoding algorithm + distributional bias from training data | yes, permanently |
| SLOT activations | verbatim content of current input | inference only, ephemeral |

The model does **not** verbatim-memorize training sequences in weights — each step sees a fresh sequence, and there is no capacity to store specific inputs in 231k params. What persists is **statistical bias**: the SLOT encoding becomes optimized for the training distribution, compressing high-frequency patterns more efficiently than rare ones.

With random bytes: no distributional bias possible — purely algorithmic weights.
With real text: weights encode English n-gram statistics — eval on the same distribution benefits, out-of-distribution suffers.
With fractal synthetic: controlled intermediate bias, no content memorized, tunable to match target domain statistics.

### Mixing strategy: staged curriculum, not simultaneous mix

Simultaneous mixing of fractal and real data risks the model learning to **distinguish data sources** (they have different BPB profiles) and conditioning the SLOT encoder on source identity rather than content — a shortcut that breaks at deployment.

Recommended staged curriculum:

```
Stage 1: random bytes          → learn distribution-free encoding algorithm
Stage 2: 1/f synthetic         → learn to exploit long-range byte correlations  
Stage 3: n-gram synthetic      → learn to exploit local statistical patterns
Stage 4: real text (held-out)  → adapt to full deployment distribution
```

Each stage fine-tunes from the previous checkpoint. The model progressively shortens its MDL description from "works for all bytes" → "works efficiently for text-like bytes" without ever memorizing specific training content.

If simultaneous mixing is required (e.g., to prevent catastrophic forgetting across stages), weight as:

```
fractal:real = 4:1  (early — algorithm still being established)
fractal:real = 1:1  (mid — distributional specialization)
fractal:real = 1:4  (late — fully adapting to target distribution)
```

### MDL interpretation

A model trained on fractal data tuned to target statistics acquires a **shorter description** of the target domain's encoding function than a model trained on random bytes. The fractal training injects inductive bias (lower expected description length for text-like inputs) without exposing the model to any actual content. This is the principled way to build domain-specific compression without privacy/copyright exposure from real data.
