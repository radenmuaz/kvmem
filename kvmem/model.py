"""
kvmem/model.py — PyTorch KV-memory transformer.

Replaces old/stage0.py. Uses:
  - torch.nn.functional.scaled_dot_product_attention (Flash Attention on MPS)
  - RoPE / YaRN positional encoding
  - Standard torch.optim.AdamW
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RoPE / YaRN
# ---------------------------------------------------------------------------

def rope_freqs(d_head: int, base: float = 10000.0, device=None) -> torch.Tensor:
    """Standard RoPE inverse frequencies: (d_head//2,)"""
    i = torch.arange(0, d_head, 2, dtype=torch.float32, device=device)
    return 1.0 / (base ** (i / d_head))


def yarn_freqs(d_head: int, L_train: int, L_max: int,
               base: float = 10000.0,
               beta_fast: int = 32, beta_slow: int = 1,
               device=None) -> torch.Tensor:
    """YaRN NTK-aware scaled RoPE frequencies (arXiv:2309.00071)."""
    s     = L_max / L_train
    i     = torch.arange(0, d_head, 2, dtype=torch.float32, device=device)
    inv_f = 1.0 / (base ** (i / d_head))
    wl    = 2 * math.pi / inv_f
    lo, hi = 2 * math.pi * beta_slow, 2 * math.pi * beta_fast
    ramp  = torch.clamp((wl - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (1 - ramp) * inv_f + ramp * (inv_f / s)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings.
    x: (..., H, L, d_head)  — works for both (H,L,dh) and (B,H,L,dh).
    freqs: (d_head//2,)
    """
    L, dh  = x.shape[-2], x.shape[-1]
    pos    = torch.arange(L, dtype=torch.float32, device=x.device)
    angles = pos[:, None] * freqs[None, :]   # (L, dh//2)
    cos_a  = angles.cos()                    # (L, dh//2) — broadcasts over ...H
    sin_a  = angles.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rx1 = x1 * cos_a - x2 * sin_a
    rx2 = x1 * sin_a + x2 * cos_a
    return torch.stack([rx1, rx2], dim=-1).reshape(x.shape)


# ---------------------------------------------------------------------------
# Model
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (L, d) or (B, L, d)   mask: (L, L)
        batched = x.dim() == 3
        if not batched:
            x = x.unsqueeze(0)   # (1, L, d)
        B, L, d = x.shape
        H, dh = self.n_heads, self.d_head
        Q = self.W_Q(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)  # (B, H, L, dh)
        K = self.W_K(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)
        V = self.W_V(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)
        if self.rope and self.freqs is not None:
            Q = apply_rope(Q, self.freqs)   # (B, H, L, dh)
            K = apply_rope(K, self.freqs)
        # mask: (L,L) → (1,1,L,L) broadcasts over (B,H,L,L)
        out = F.scaled_dot_product_attention(Q, K, V,
                                             attn_mask=mask.unsqueeze(0).unsqueeze(0))
        out = out.permute(0, 2, 1, 3).reshape(B, L, d)
        out = self.W_O(out)
        if not batched:
            out = out.squeeze(0)
        return out


class FFN(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # tanh-GELU — avoids any MPS-unsupported ops
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (L, d) or (B, L, d) — LayerNorm and FFN handle both via broadcasting
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class KVMemModel(nn.Module):
    def __init__(self, V: int, d: int, n_layers: int, n_heads: int,
                 d_ff: int,
                 rope: bool = False, yarn: bool = False,
                 L_train: int = 512, L_max: int = 4096):
        super().__init__()
        self.embed    = nn.Embedding(V, d)
        self.norm_out = nn.LayerNorm(d)
        self.W_out    = nn.Linear(d, V, bias=False)

        # RoPE / YaRN frequencies (not a learned parameter)
        freqs = None
        if rope:
            d_head = d // n_heads
            if yarn:
                freqs = yarn_freqs(d_head, L_train=L_train, L_max=L_max)
            else:
                freqs = rope_freqs(d_head)

        self.blocks = nn.ModuleList([
            TransformerBlock(d, n_heads, d_ff, rope=rope, freqs=freqs)
            for _ in range(n_layers)
        ])

        # Weight init: scaled normal
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'embed' in name or 'W_out' in name:
                nn.init.normal_(p, std=0.02)
            elif p.dim() == 2:
                nn.init.normal_(p, std=math.sqrt(2.0 / p.shape[-1]))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        tokens: (L,) int64 → logits (L, V)
             or (B, L) int64 → logits (B, L, V)
        mask: (L, L) float32 — same for all batch elements
        """
        x = self.embed(tokens)   # (L, d) or (B, L, d)
        for block in self.blocks:
            x = block(x, mask)
        return self.W_out(self.norm_out(x))

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
    )
    if device is not None:
        model = model.to(device)
    return model
