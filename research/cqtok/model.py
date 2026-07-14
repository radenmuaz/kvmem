"""
model.py — Causal Transformer with RoPE (PyTorch).

Used as backbone for both byte-level and latent LM.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(
    q: torch.Tensor,   # (B, T, H, d_head)
    k: torch.Tensor,
    theta: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    T, d = q.shape[1], q.shape[-1]
    half = d // 2
    inv_freq = 1.0 / (theta ** (torch.arange(half, device=q.device).float() / half))
    t = torch.arange(T, device=q.device).float()
    freqs = torch.outer(t, inv_freq)                              # (T, half)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)           # (T, d)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2)                           # (1, T, 1, d)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class CausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope_theta: float = 10_000.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        H, d_h = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, T, H, d_h)
        k = self.k_proj(x).view(B, T, H, d_h)
        v = self.v_proj(x).view(B, T, H, d_h)

        q, k = apply_rope(q, k, self.rope_theta)

        # sdp_attention expects (B, H, T, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, d_h)
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        d_h = int(d_model * expansion * 2 / 3)
        d_h = (d_h + 63) // 64 * 64
        self.gate = nn.Linear(d_model, d_h, bias=False)
        self.up   = nn.Linear(d_model, d_h, bias=False)
        self.down = nn.Linear(d_h, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope_theta: float = 10_000.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = CausalAttention(d_model, n_heads, rope_theta)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLUFFN(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CausalTransformer(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, rope_theta: float = 10_000.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, rope_theta) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LatentLM(nn.Module):
    """z_hat (B, T, d_q) → embed → Transformer → h (B, T, d_model)."""

    def __init__(self, d_q: int, d_model: int, n_layers: int, n_heads: int,
                 rope_theta: float = 10_000.0):
        super().__init__()
        self.in_proj     = nn.Linear(d_q, d_model)
        self.transformer = CausalTransformer(d_model, n_layers, n_heads, rope_theta)

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        return self.transformer(self.in_proj(z_hat))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    lm = LatentLM(d_q=18, d_model=128, n_layers=4, n_heads=4).to(dev)
    z = torch.randn(2, 64, 18, device=dev)
    h = lm(z)
    print(f"LatentLM  input={tuple(z.shape)}  output={tuple(h.shape)}  params={lm.param_count():,}")
