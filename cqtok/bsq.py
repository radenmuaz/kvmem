"""
bsq.py — Binary Spherical Quantization (PyTorch).

Encoder: project → L2-normalize → sign (STE) → ±1/sqrt(d_q).
LM head: d_q independent Bernoullis, per-bit BCE loss.

Codebook sizing for K=8 bytes: d_q=18 → 2^18 ≈ 262K codes (compact text).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BSQEncoder(nn.Module):
    def __init__(self, d_in: int, d_q: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_q, bias=False)
        self.d_q = d_q

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        u: (*, d_in)
        Returns:
          z_hat : (*, d_q)   STE-quantized float ±1/sqrt(d_q)
          bits  : (*, d_q)   int64 {0, 1}
        """
        v = self.proj(u)
        v_norm = F.normalize(v, dim=-1)
        b = v_norm.sign()                               # {-1, +1}
        z_hat = v_norm + (b - v_norm).detach()          # STE
        z_hat = z_hat / math.sqrt(self.d_q)
        bits = ((b + 1) / 2).long()                     # {0, 1}
        return z_hat, bits


class BSQLMHead(nn.Module):
    def __init__(self, d_model: int, d_q: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_q)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (*, d_model) → logits: (*, d_q)"""
        return self.linear(h)

    @staticmethod
    def loss(logits: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        """
        logits: (*, d_q)
        bits:   (*, d_q) int64 {0, 1}  — target next-chunk bits
        """
        return F.binary_cross_entropy_with_logits(logits, bits.float())


if __name__ == "__main__":
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    enc  = BSQEncoder(128, 18).to(dev)
    head = BSQLMHead(128, 18).to(dev)

    u = torch.randn(32, 128, device=dev)
    h = torch.randn(32, 128, device=dev)
    z_hat, bits = enc(u)
    loss = BSQLMHead.loss(head(h), bits)

    print(f"BSQ  d_q=18  codebook=2^18={2**18:,}")
    print(f"z_hat={tuple(z_hat.shape)}  ||z||≈{z_hat.norm(dim=-1).mean():.3f}")
    print(f"loss={loss.item():.4f}  (expect ≈ln2={math.log(2):.4f})")
