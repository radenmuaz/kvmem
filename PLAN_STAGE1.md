# Addendum: Loss-Only Iterative Refinement

This document extends `STAGE_IMPLEMENTATION_PLAN.md`. Read that first. This addendum changes the loss function for stage 1 and clarifies the role of ground truth during ingestion. Architecture and mask design are unchanged from the main doc.

---

## 1. Key insight: ground truth is available at ingestion

The original handover treated inference as a "no supervisory signal available" regime. That's wrong for this project.

Two distinct deployment phases need to be separated:

- **Ingestion-time**: the data being absorbed into memory is the input. We have it, by definition. Errors in predicting it from memory are forward-computable.
- **Query-time**: the model is asked to produce continuations of unseen prompts. No ground truth.

The "no gradients at inference" commitment forbids backprop only — not supervisory signals. Using known ground-truth data as forward-pass input is fully consistent with that commitment.

This expands what's possible: at ingestion, the model can self-check by running reconstruction probes against the actual source data. But — and this is the key choice — that self-check can stay outside the model entirely. The transformer doesn't need to compute error tensors or take feedback inputs. Instead, the error-correction behavior can be *amortized into the trained weights* through a training-time loss design. The model stays a standard transformer at every step.

---

## 2. Loss-only refinement objective (replaces §6.4 of main doc)

### 2.1 Motivation

Standard multi-pass training (the current §6.4) applies `L_cont,t` at each pass with weighting `beta_t`. Each pass gets supervision, but there's no explicit pressure for *pass `t` to fix the specific positions where pass `t-1` failed*. This is what the loss-only refinement adds.

The mechanism is **hard-position weighting using stop-gradient on previous-pass errors**. At training time, positions where pass `t-1` had high NLL get extra weight in pass `t`'s loss. The model learns to allocate compute toward correcting prior mistakes — without ever receiving an explicit error tensor as input.

At inference, the model is unchanged. The refinement is a learned forward dynamic.

### 2.2 Per-position NLL

For each pass `t = 1..T`, define per-position NLL on the `Y^(t)` block:
```
e^(t)_{b,k} = -log p_theta( y^(t)_{b,k} | context at position p_y(t) + k - 1 )
              for b ∈ [B], k ∈ [L_y]
```

The standard (unweighted) per-pass continuation loss:
```
L_cont,t_base = (1 / (B · L_y)) · sum_{b,k} e^(t)_{b,k}
```

### 2.3 Hard-position weights

For pass `t ≥ 2`, compute weights from pass `(t-1)`'s NLL, treated as constants:
```
w^(t)_{b,k} = stop_grad( clip( e^(t-1)_{b,k} - tau,  0,  w_max ) )
```

- `tau` (threshold): positions with NLL below this are "easy" and get weight 0. Suggested: `tau = log(2)` ≈ 0.69 nats — positions where the model is at or worse than chance for a binary distinction.
- `w_max` (ceiling): caps the influence of catastrophically wrong positions. Suggested: `w_max = 3`.
- `stop_grad` is essential. Weights are a deterministic function of pass `(t-1)`'s loss values and must not contribute to gradients through pass `(t-1)`'s predictions. Use `jax.lax.stop_gradient` in JAX.

### 2.4 Focused per-pass loss

For pass `t ≥ 2`:
```
L_cont,t_focus = sum_{b,k} (1 + lambda_w · w^(t)_{b,k}) · e^(t)_{b,k}
                 / sum_{b,k} (1 + lambda_w · w^(t)_{b,k})
```

The `1 +` term ensures every position still contributes (preventing easy positions from being ignored and degrading); the weighted bonus reallocates compute toward correcting prior errors.

- `lambda_w` (focus strength): how aggressively to upweight hard positions. Suggested starting value: `lambda_w = 2`. Sweep `{0.5, 1, 2, 4}`.

### 2.5 Monotonicity regularizer (optional but recommended)

Penalize passes that regress on overall NLL compared to the previous pass:
```
L_mono = sum_{t=2..T} ReLU( L_cont,t_base − L_cont,t-1_base + gamma )
```
- `gamma` (margin): small slack tolerated. Start at `gamma = 0` (strict monotonicity).
- `lambda_mono` (weight on this term). Start at `0.1`.

This is a global signal (gives direction; pass `t` shouldn't be worse overall than pass `t-1`), complementing the per-position focus (gives detail; pass `t` should be better specifically where pass `t-1` was weak).

### 2.6 Total loss

```
L_total = L_src
        + lambda_cont · L_cont,1_base                          # pass 1 (no focus, no prior pass)
        + lambda_cont · sum_{t=2..T} beta_t · L_cont,t_focus   # passes 2..T (focused)
        + lambda_mono · L_mono                                  # optional monotonicity
        + lambda_c · L_cap
```

Pass 1 uses the unweighted base loss (no prior pass exists to focus against).

### 2.7 Hyperparameters

Add to the table from main doc §4.5:
```
lambda_cont   = 2.0        # unchanged
lambda_w      = 2.0        # hard-position focus strength
lambda_mono   = 0.1        # monotonicity regularizer
tau           = 0.693      # = log(2), error threshold
w_max         = 3.0        # weight ceiling
gamma         = 0.0        # monotonicity margin (strict)
beta_t        = (t/T) / sum_s (s/T)   # linear ramp, normalized (unchanged)
```

### 2.8 Code (reference implementation)

```python
import jax
import jax.numpy as jnp

def stage1_loss(params, batch, model, T, hp):
    """
    batch fields:
      tokens         (B, L)  full sequence
      mask           (L, L)  attention mask (from make_mask_stage1)
      src_mask       (B, L-1)  source NTP positions
      y_masks        list of T arrays, each (B, L-1), 1 on positions where pass t's NLL applies
      mem_drop_mask  (B, L-1)  positions predicting deterministic memory tokens (drop from loss)
    hp fields: lambda_cont, lambda_w, lambda_mono, tau, w_max, gamma, beta, lambda_c
    """
    logits, strengths = model.apply(params, batch.tokens, batch.mask)

    # Per-position NLL over the L-1 shifted positions
    # logits[i] predicts token at i+1
    nll = -jnp.take_along_axis(
        jax.nn.log_softmax(logits[:, :-1], axis=-1),
        batch.tokens[:, 1:, None],
        axis=-1
    ).squeeze(-1)                                  # (B, L-1)

    # Drop deterministic memory predictions
    nll_kept = nll * (1.0 - batch.mem_drop_mask)

    # Source loss
    L_src = (
        (nll_kept * batch.src_mask).sum() / batch.src_mask.sum().clip(min=1)
    )

    # Per-pass losses
    L_cont = []
    e_prev = None
    for t in range(T):
        y_mask_t = batch.y_masks[t]                # (B, L-1)
        e_t = nll_kept * y_mask_t                  # (B, L-1), zero outside y^(t)
        n_t = y_mask_t.sum().clip(min=1)

        if t == 0 or hp.lambda_w == 0.0:
            # Pass 1 or focus disabled: unweighted base loss
            L_cont.append(e_t.sum() / n_t)
        else:
            # Focused loss: weight positions by pass (t-1)'s error
            w = jax.lax.stop_gradient(
                jnp.clip(e_prev - hp.tau, 0.0, hp.w_max)
            ) * y_mask_t                           # zero outside y^(t-1)'s positions
            # Note: w is indexed against y^(t-1)'s positions; we need to align it
            # with y^(t)'s positions. Since L_y is constant and the within-block 
            # offset matches across passes, we can take the per-(b, k_within) values.
            # Easiest: extract per-block-position arrays directly.
            e_t_block = extract_block(nll_kept, batch.y_masks[t])    # (B, L_y)
            e_prev_block = extract_block(nll_kept, batch.y_masks[t-1])  # (B, L_y)
            w_block = jax.lax.stop_gradient(
                jnp.clip(e_prev_block - hp.tau, 0.0, hp.w_max)
            )
            weights = 1.0 + hp.lambda_w * w_block  # (B, L_y)
            L_cont.append((weights * e_t_block).sum() / weights.sum())

        e_prev = e_t

    L_cont_base = [
        (nll_kept * batch.y_masks[t]).sum() / batch.y_masks[t].sum().clip(min=1)
        for t in range(T)
    ]

    # Monotonicity regularizer
    L_mono = sum(
        jnp.maximum(0.0, L_cont_base[t] - L_cont_base[t-1] + hp.gamma)
        for t in range(1, T)
    )

    # Total
    L_total = (
        L_src
        + hp.lambda_cont * sum(hp.beta[t] * L_cont[t] for t in range(T))
        + hp.lambda_mono * L_mono
        + hp.lambda_c * strengths.mean()
    )
    return L_total, {
        'L_src': L_src,
        'L_cont_base': L_cont_base,
        'L_cont_focus': L_cont,
        'L_mono': L_mono,
    }
```

Helper `extract_block(arr, mask)` collects positions where `mask == 1`, reshaped to `(B, L_y)` — straightforward; the y blocks are equal-sized and contiguous, so a static slice based on `p_y(t)` works.

### 2.9 Inference (unchanged from main doc)

The model has no awareness of the focused loss at inference. Forward passes proceed identically to the original spec. The "refinement" is a property of the learned weights, not of the inference computation. See main doc §6.7.

---

## 3. Probe-based outer loop for stage 2 (replaces parts of §7 main doc)

At stage 2 (streaming), the ingestion-time ground truth lever provides a simple alternative to the self-evaluation head and SRS controller as previously specified. The model remains a pure transformer; all decision logic lives in plain orchestration code.

### 3.1 Probe operation

Given current memory state `M` and a chunk `c`, compute reconstruction quality:
```python
def probe(model, params, M, c):
    """Forward pass: M as prefix, c as queries. Returns per-token NLL on c."""
    logits = model.apply_with_prefix_cache(params, c, prefix_cache=M)
    nll = -log_softmax(logits[:-1])[range(len(c)-1), c[1:]]
    return nll      # shape (L_c - 1,)
```

This is one forward pass, no gradients, no special architecture. Just the same model used in any normal query.

### 3.2 Streaming loop with probe-driven rehearsal

```python
buffer = []                    # list of (chunk, baseline_nll, age)
M = M_null
for c in stream():
    # Ingest
    M = write(model, params, c, M)
    
    # Establish baseline reconstruction quality
    baseline_nll = probe(model, params, M, c).mean()
    buffer.append({'chunk': c, 'baseline_nll': baseline_nll, 'age': 0})
    
    # Periodic interference check
    if step % refresh_period == 0:
        for entry in buffer:
            current_nll = probe(model, params, entry['chunk'], M).mean()
            entry['age'] += 1
            if current_nll > entry['baseline_nll'] + delta_threshold:
                # Interference detected: rehearse
                M = write(model, params, entry['chunk'], M)
                entry['baseline_nll'] = probe(model, params, entry['chunk'], M).mean()
```

Key properties:
- The model is invoked only for `write` and `probe` — both standard forward passes.
- The scheduler logic is plain Python, has access to ground truth (the chunks in the buffer), can use whatever heuristic the experiment calls for.
- No learned self-evaluation head needed. The signal is the real NLL on real data.

### 3.3 What this replaces from the main doc

Stages 4 (self-evaluation head) and 5 (SRS controller) in main doc §7 become much simpler:

- **Original stage 4**: train a head `r_phi(M, c)` to predict reconstruction NLL by regression.  
  **New stage 4**: drop. Use direct probing.

- **Original stage 5**: SRS controller driven by the learned `r_phi`.  
  **New stage 5**: SRS controller driven by direct probing (per code in §3.2 above). The research contribution shifts from "learn a retrievability predictor" to "design good scheduling heuristics on real probe signals" — which is a simpler, more tractable research problem.

Stages 3, 6, 7 in the main doc are unaffected.

---

## 4. Validation additions for stages 0 and 1

Add to main doc §5.6 and §6.6 success criteria:

### 4.1 Stage 0 (no new criteria; loss-only refinement is a stage 1 mechanism)

Stage 0 is single-pass and doesn't benefit from focused loss. Run as specified in main doc.

### 4.2 Stage 1 — additional criteria

| Criterion | Required |
|---|---|
| `L_cont,1_base` (focused-loss-trained model) ≥ `L_cont,1` (vanilla-trained model, same hyperparameters) | Yes — focused training shouldn't hurt pass-1 quality |
| Per-position NLL distribution at pass `T` is more concentrated (lower variance) than at pass `1` | Yes — refinement should narrow the error distribution |
| For top-K hardest positions at pass `1` (highest NLL), pass `T` NLL on those positions improves by ≥ X% more than on the median position | Yes — direct check that focused loss does what it claims |
| `L_mono` term: regression frequency at convergence is below 5% of training steps | Sanity check on monotonicity regularizer |

### 4.3 New ablation to run

Train two stage 1 models on identical data:
- **A**: vanilla multi-pass loss (main doc §6.4).
- **B**: focused multi-pass loss (this addendum §2.6).

Compare on the truncation diagnostic from main doc §6.5. Specifically: at truncation `t = 1` (i.e., using only one pass of memory from a model trained for `T = 4`), is model B meaningfully worse than model A? If yes, focused training has over-specialized to later passes — bad. If they're comparable, focused training is a free win. If B is better even at truncation `t = 1`, focused training has incidentally improved pass-1 quality too (best case).

Report this comparison in the stage 1 summary.

---

## 5. What stays the same

From the main doc, the following are unchanged by this addendum:

- All architectural decisions (decoder-only, NoPE, single marker token, strength gating)
- The sequence layouts for stages 0 and 1
- The attention masks (causal + bottleneck + write-only Y + cross-Y blocked)
- Hyperparameters for model size, training duration, optimizer
- Inference procedure (just forward passes; no error feedback at inference)
- Success criteria for stage 0
- Implementation file structure and dependencies

The focused-loss change is a pure training-time modification. If it doesn't help, removing it returns to the main-doc spec with zero side effects.

---

## 6. Decision summary

| Aspect | Main doc | This addendum |
|---|---|---|
| Architecture | Decoder-only + masking | Unchanged |
| Inference loop | Stack `T` forward passes | Unchanged |
| Training loss | NTP per pass, `beta_t` weighted | NTP per pass + hard-position focus + monotonicity |
| Self-evaluation at deployment | "Future stage 4 work" | Direct probe (forward pass against real data) |
| SRS scheduling | Driven by learned head (future stage 5) | Driven by direct probe signal — Python heuristic |
| Model files / size | As specified | Unchanged |
| Implementation cost | Baseline | +~20 lines of loss code |

The addendum's central claim is that **error-correction during refinement can be amortized into trained weights via a training-time loss design**, requiring no changes to the model's architecture or inference behavior. The model stays a transformer. Inference stays simple. The intelligence lives in the trained weights, where it always has.

The probe-based outer loop for stage 2 makes the same architectural commitment at a coarser granularity: the model is *just* a forward function, the scheduling is *just* Python, and the signal that links them is *just* the actual NLL on actual data.