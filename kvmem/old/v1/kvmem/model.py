"""
kvmem/model.py — PyTorch KV-memory transformer.

Two forward modes:
  1. Full-pass:  model(tokens, mask)
  2. Block-pass: model(prefix, prefix_mask, return_kv=True)  →  prefix_kv
                 model(suffix, suffix_mask, past_kv=prefix_kv, offset=L_prefix)

Flags:
  grad_checkpoint : apply torch.utils.checkpoint per block (depth-wise).
                    Saves residual-stream activations between layers at the
                    cost of one extra forward per block during backward.
                    Does NOT reduce the O(L²) attention matrix — use chunked
                    or flash attention for that.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt


# ---------------------------------------------------------------------------
# RoPE / YaRN
# ---------------------------------------------------------------------------

def rope_freqs(d_head: int, base: float = 10000.0, device=None) -> torch.Tensor:
    i = torch.arange(0, d_head, 2, dtype=torch.float32, device=device)
    return 1.0 / (base ** (i / d_head))


def yarn_freqs(d_head: int, L_train: int, L_max: int,
               base: float = 10000.0,
               beta_fast: int = 32, beta_slow: int = 1,
               device=None) -> torch.Tensor:
    """YaRN NTK-aware scaled RoPE (arXiv:2309.00071)."""
    s     = L_max / L_train
    i     = torch.arange(0, d_head, 2, dtype=torch.float32, device=device)
    inv_f = 1.0 / (base ** (i / d_head))
    wl    = 2 * math.pi / inv_f
    lo, hi = 2 * math.pi * beta_slow, 2 * math.pi * beta_fast
    ramp  = torch.clamp((wl - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (1 - ramp) * inv_f + ramp * (inv_f / s)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """x: (..., H, L, d_head)  freqs: (d_head//2,)  offset: position base."""
    L, dh  = x.shape[-2], x.shape[-1]
    pos    = torch.arange(offset, offset + L, dtype=torch.float32, device=x.device)
    angles = pos[:, None] * freqs[None, :]
    cos_a, sin_a = angles.cos(), angles.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos_a - x2 * sin_a,
                        x1 * sin_a + x2 * cos_a], dim=-1).reshape(x.shape)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class MHAttention(nn.Module):
    def __init__(self, d: int, n_heads: int,
                 rope: bool = False, freqs: torch.Tensor | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d // n_heads
        self.rope    = rope
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d, bias=False)
        if freqs is not None:
            self.register_buffer('freqs', freqs)
        else:
            self.freqs = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                past_kv: tuple | None = None,
                return_kv: bool = False,
                offset: int = 0) -> torch.Tensor | tuple:
        batched = x.dim() == 3
        if not batched:
            x = x.unsqueeze(0)
        B, L, d = x.shape
        H, dh = self.n_heads, self.d_head
        Q = self.W_Q(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)
        K = self.W_K(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)
        V = self.W_V(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)
        if self.rope and self.freqs is not None:
            Q = apply_rope(Q, self.freqs, offset=offset)
            K = apply_rope(K, self.freqs, offset=offset)
        K_cur, V_cur = K, V
        if past_kv is not None:
            K_past, V_past = past_kv
            K = torch.cat([K_past, K], dim=2)
            V = torch.cat([V_past, V], dim=2)
        out = F.scaled_dot_product_attention(Q, K, V,
                                             attn_mask=mask.unsqueeze(0).unsqueeze(0))
        out = out.permute(0, 2, 1, 3).reshape(B, L, d)
        out = self.W_O(out)
        if not batched:
            out = out.squeeze(0)
        if return_kv:
            return out, (K_cur, V_cur)
        return out


class FFN(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.W1(x)
        h = 0.5 * h * (1.0 + torch.tanh(0.7978845608028654 * (h + 0.044715 * h ** 3)))
        return self.W2(h)


class TransformerBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, d_ff: int,
                 rope: bool = False, freqs: torch.Tensor | None = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = MHAttention(d, n_heads, rope=rope, freqs=freqs)
        self.norm2 = nn.LayerNorm(d)
        self.ffn   = FFN(d, d_ff)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                past_kv: tuple | None = None,
                return_kv: bool = False,
                offset: int = 0) -> torch.Tensor | tuple:
        attn_out = self.attn(self.norm1(x), mask,
                             past_kv=past_kv, return_kv=return_kv, offset=offset)
        if return_kv:
            attn_out, kv = attn_out
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        if return_kv:
            return x, kv
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class KVMemModel(nn.Module):
    def __init__(self, V: int, d: int, n_layers: int, n_heads: int,
                 d_ff: int,
                 rope: bool = False, yarn: bool = False,
                 L_train: int = 512, L_max: int = 4096,
                 grad_checkpoint: bool = False):
        super().__init__()
        self.embed           = nn.Embedding(V, d)
        self.norm_out        = nn.LayerNorm(d)
        self.W_out           = nn.Linear(d, V, bias=False)
        self.grad_checkpoint = grad_checkpoint

        freqs = None
        if rope:
            d_head = d // n_heads
            freqs  = (yarn_freqs(d_head, L_train=L_train, L_max=L_max)
                      if yarn else rope_freqs(d_head))

        self.blocks = nn.ModuleList([
            TransformerBlock(d, n_heads, d_ff, rope=rope, freqs=freqs)
            for _ in range(n_layers)
        ])
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'embed' in name or 'W_out' in name:
                nn.init.normal_(p, std=0.02)
            elif p.dim() == 2:
                nn.init.normal_(p, std=math.sqrt(2.0 / p.shape[-1]))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor,
                past_kv: list | None = None,
                return_kv: bool = False,
                offset: int = 0) -> torch.Tensor | tuple:
        """
        tokens   : (B, L) or (L,) int64
        mask     : (L_q, L_kv) — L_kv = L_past + L when past_kv given
        past_kv  : list[n_layers] of (K_past, V_past) — cached prefix KV
        return_kv: return (logits, kv_list) instead of just logits
        offset   : RoPE position offset (= L_past for suffix pass)

        grad_checkpoint=True: checkpoint each block during backward (depth-only).
          Saves residual activations between layers; attention matrix is unchanged.
          Only active when self.training and past_kv is None and not return_kv.
        """
        batched = tokens.dim() == 2
        if not batched:
            tokens = tokens.unsqueeze(0)
        x      = self.embed(tokens)
        kv_out = []
        L_past = past_kv[0][0].shape[2] if past_kv is not None else 0
        _offset = offset if offset else L_past

        for i, block in enumerate(self.blocks):
            pkv = past_kv[i] if past_kv is not None else None
            use_ckpt = (self.grad_checkpoint and self.training
                        and pkv is None and not return_kv)
            if use_ckpt:
                # Checkpoint this block: discard intermediate activations,
                # recompute in backward. mask must not require grad.
                x = _ckpt(block, x, mask, use_reentrant=False)
            else:
                result = block(x, mask, past_kv=pkv,
                               return_kv=return_kv, offset=_offset)
                if return_kv:
                    x, kv_i = result
                    kv_out.append(kv_i)
                else:
                    x = result

        logits = self.W_out(self.norm_out(x))
        if not batched:
            logits = logits.squeeze(0)
        if return_kv:
            return logits, kv_out
        return logits

    # Convenience aliases used by kvcache.py
    def encode_prefix(self, prefix_tokens, prefix_mask):
        """Full forward on prefix, return (logits, kv_list)."""
        return self.forward(prefix_tokens, prefix_mask, return_kv=True)

    def forward_with_prefix_kv(self, suffix_tokens, prefix_kv, suffix_mask):
        """Suffix forward attending to cached prefix KV."""
        L_past = prefix_kv[0][0].shape[2]
        return self.forward(suffix_tokens, suffix_mask,
                            past_kv=prefix_kv, offset=L_past)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(hp: dict, device=None) -> KVMemModel:
    model = KVMemModel(
        V=hp['V'], d=hp['d'], n_layers=hp['n_layers'],
        n_heads=hp['n_heads'], d_ff=hp['d_ff'],
        rope=hp.get('rope', False),
        yarn=hp.get('yarn', False),
        L_train=hp.get('L_train', hp.get('seg_len', 512)),
        L_max=hp.get('L_max', hp.get('seg_len', 512) * 8),
        grad_checkpoint=hp.get('grad_checkpoint', False),
    )
    if device is not None:
        model = model.to(device)
    return model
