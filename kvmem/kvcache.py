"""
kvmem/kvcache.py — Blockwise KV-cache training utilities.

Splits each sequence at the last </m> boundary (pos['mc1']):
  prefix = all <m>slots</m><s>src</s> blocks  (encoded once, KV cached)
  suffix = <f>warmup</f><c>output</c>          (attends to cached prefix KV)

This matches inference exactly: encode source blocks once, run multiple recall
queries against cached K/V states without re-encoding the source.

Trade-off vs full-pass:
  Full-pass : exact gradients, exact SDPA computation, O(L²) attention matrix.
  KV-cache  : prefix and suffix computed in separate passes → different float32
              rounding in backward → different local optima. Converges to lower
              bpb but worse exact-match than full-pass on short sequences.

Use full-pass (train.py default) for match% targets.
Use kvcache (this module) for large sequences (seg≥576) where full L×L is
the memory bottleneck, or for inference-aligned fine-tuning.

Functions
---------
blockwise_tf_loss   : two-pass TF loss (prefix KV cached, suffix loss)
ocd_rollout_full    : AR rollout using full forward passes
ocd_rollout_kvcache : AR rollout using prefix KV cache (fast for large L)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# OCD helper
# ---------------------------------------------------------------------------

def _ocd_next_tokens(y_gen: list[int], x_ref: list[int],
                     vocab_size: int = 256) -> np.ndarray:
    """
    Uniform distribution over next tokens minimising edit distance to x_ref
    given already-generated prefix y_gen.
    """
    k = len(y_gen)
    L = len(x_ref)
    dist = np.zeros(vocab_size, dtype=np.float32)
    if L == 0 or k >= L:
        return dist
    if y_gen == x_ref[:k]:
        dist[x_ref[k]] = 1.0
        return dist
    costs = np.empty(L, dtype=np.int32)
    for j in range(L):
        overlap = min(k, j)
        hamm    = sum(y_gen[i] != x_ref[i] for i in range(overlap))
        costs[j] = hamm + abs(k - j)
    min_cost = int(costs.min())
    opts: set[int] = {x_ref[j] for j in range(L) if costs[j] == min_cost}
    p = 1.0 / len(opts)
    for tok in opts:
        dist[tok] = p
    return dist


# ---------------------------------------------------------------------------
# Blockwise TF loss
# ---------------------------------------------------------------------------

def blockwise_tf_loss(model, tokens_np: np.ndarray, pos: dict,
                      mask_t: torch.Tensor, log_probs_fn, device) -> torch.Tensor:
    """
    Two-pass TF loss:
      Pass 1: prefix = tokens[:, :mc1]  →  cache prefix KV (no loss)
      Pass 2: suffix = tokens[:, mc1:]  →  compute loss on <c> region

    Mathematically equivalent to full-pass TF but with different float32
    rounding in backward (different SDPA call graph).

    pos must contain: mc1, c0, c1
    """
    tokens      = torch.tensor(tokens_np, device=device)
    L_p         = pos['mc1']
    prefix_mask = mask_t[:L_p, :L_p]
    suffix_mask = mask_t[L_p:, :]
    _, prefix_kv = model(tokens[:, :L_p], prefix_mask, return_kv=True)
    logits_suf   = model(tokens[:, L_p:], suffix_mask, past_kv=prefix_kv)
    c0s = pos['c0'] - L_p
    c1s = pos['c1'] - L_p
    lp       = log_probs_fn(logits_suf[:, c0s:c1s])
    tgts_c   = tokens[:, pos['c0']+1:pos['c1']+1]
    nll      = -lp.gather(2, tgts_c.unsqueeze(-1)).squeeze(-1)
    return nll.mean()


# ---------------------------------------------------------------------------
# OCD rollouts
# ---------------------------------------------------------------------------

@torch.no_grad()
def ocd_rollout_full(model, tokens_batch: np.ndarray,
                     pos: dict, refs: list[list[int]],
                     mask_t: torch.Tensor, device
                     ) -> tuple[torch.Tensor, np.ndarray]:
    """
    Batched AR rollout using full forward passes (one per generation step).
    B examples run in parallel each step. No KV caching — recomputes full
    sequence every step.
    """
    out_len = pos['c1'] - pos['c0']
    c0      = pos['c0']
    B       = tokens_batch.shape[0]
    tok_t       = torch.tensor(tokens_batch, dtype=torch.long, device=device)
    ocd_targets = np.zeros((B, out_len, 256), dtype=np.float32)
    y_gens      = [[] for _ in range(B)]
    for k in range(out_len):
        logits = model(tok_t, mask_t)
        nbs    = logits[:, c0 + k - 1].argmax(-1).cpu().numpy()
        for b in range(B):
            ocd_targets[b, k] = _ocd_next_tokens(y_gens[b], refs[b])
            y_gens[b].append(int(nbs[b]))
        tok_t[:, c0 + k] = torch.from_numpy(nbs).to(device)
    return tok_t, ocd_targets


@torch.no_grad()
def ocd_rollout_kvcache(model, tokens_batch: np.ndarray,
                        pos: dict, refs: list[list[int]],
                        mask_t: torch.Tensor, device
                        ) -> tuple[torch.Tensor, np.ndarray]:
    """
    KV-cached batched AR rollout.

    Encodes the prefix (<m>slots</m><s>src</s> blocks) once, then generates
    suffix tokens attending to cached prefix KV. O(out_len) suffix passes
    instead of O(out_len) full-sequence passes.

    Requires pos['mc1'] as the prefix split point.
    """
    out_len  = pos['c1'] - pos['c0']
    c0       = pos['c0']
    L_prefix = pos['mc1']
    B        = tokens_batch.shape[0]
    L        = tokens_batch.shape[1]
    tok_t    = torch.tensor(tokens_batch, dtype=torch.long, device=device)

    prefix_mask_t = torch.tensor(
        mask_t.cpu().numpy()[:L_prefix, :L_prefix], device=device)
    _, prefix_kv  = model.encode_prefix(tok_t[:, :L_prefix], prefix_mask_t)

    ocd_targets = np.zeros((B, out_len, 256), dtype=np.float32)
    y_gens      = [[] for _ in range(B)]
    for k in range(out_len):
        suf_len      = (L - L_prefix) - (out_len - k)
        suf_toks     = tok_t[:, L_prefix:L_prefix + suf_len + k + 1]
        full_suf_len = suf_toks.shape[1]
        suf_mask_t   = torch.tensor(
            mask_t.cpu().numpy()[L_prefix:L_prefix+full_suf_len,
                                 :L_prefix+full_suf_len], device=device)
        logits = model.forward_with_prefix_kv(suf_toks, prefix_kv, suf_mask_t)
        nbs    = logits[:, -1].argmax(-1).cpu().numpy()
        for b in range(B):
            ocd_targets[b, k] = _ocd_next_tokens(y_gens[b], refs[b])
            y_gens[b].append(int(nbs[b]))
        tok_t[:, c0 + k] = torch.from_numpy(nbs).to(device)
    return tok_t, ocd_targets
