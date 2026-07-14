"""
fsq.py — Finite Scalar Quantization (PyTorch), L=2 and L=8.

L=2  d_q=18 → 2^18 ≈ 262K codes, per-bit BCE head.
L=8  d_q=6  → 8^6  = 2^18 codes, per-dim 8-way CE head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _fsq_quantize(z_pre: torch.Tensor, L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    z_pre: (*, d_q) raw projections.
    Returns z_hat (STE float) and codes (int64 in {0,...,L-1}).
    """
    if L == 2:
        z_cont = z_pre.tanh() * 0.5                                 # (-0.5, 0.5)
        z_q    = torch.where(z_cont >= 0,
                             torch.full_like(z_cont, 0.5),
                             torch.full_like(z_cont, -0.5))
        z_hat  = z_cont + (z_q - z_cont).detach()                   # STE
        codes  = (z_q + 0.5).long()                                  # {0, 1}
    else:
        half   = (L - 1) / 2.0
        z_cont = z_pre.tanh() * half                                 # (-half, half)
        z_q    = z_cont.round()
        z_hat  = z_cont + (z_q - z_cont).detach()                   # STE
        codes  = (z_hat + half).round().long().clamp(0, L - 1)
    return z_hat, codes


class FSQEncoder(nn.Module):
    def __init__(self, d_in: int, d_q: int, L: int):
        super().__init__()
        assert L >= 2
        self.proj = nn.Linear(d_in, d_q, bias=False)
        self.L = L
        self.d_q = d_q

    @property
    def codebook_bits(self) -> float:
        return self.d_q * math.log2(self.L)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """u: (*, d_in) → z_hat (*, d_q), codes (*, d_q) int64"""
        return _fsq_quantize(self.proj(u), self.L)


class FSQLMHead(nn.Module):
    def __init__(self, d_model: int, d_q: int, L: int):
        super().__init__()
        self.L = L
        self.d_q = d_q
        out = d_q if L == 2 else d_q * L
        self.linear = nn.Linear(d_model, out)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (*, d_model)
        Returns: (*, d_q) for L=2; (*, d_q, L) for L>2
        """
        out = self.linear(h)
        if self.L > 2:
            out = out.unflatten(-1, (self.d_q, self.L))
        return out

    def loss(self, logits: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """
        logits: (*, d_q) for L=2; (*, d_q, L) for L>2
        codes:  (*, d_q) int64
        """
        if self.L == 2:
            return F.binary_cross_entropy_with_logits(logits, codes.float())
        # per-dim CE: flatten to (N, L) vs (N,)
        return F.cross_entropy(logits.reshape(-1, self.L), codes.reshape(-1))


if __name__ == "__main__":
    import math
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    for d_q, L, label in [(18, 2, "FSQ L=2 d_q=18"), (6, 8, "FSQ L=8 d_q=6")]:
        enc  = FSQEncoder(128, d_q, L).to(dev)
        head = FSQLMHead(128, d_q, L).to(dev)
        u = torch.randn(32, 128, device=dev)
        h = torch.randn(32, 128, device=dev)
        z_hat, codes = enc(u)
        loss = head.loss(head(h), codes)
        print(f"{label}  codebook=2^{enc.codebook_bits:.0f}  "
              f"codes={tuple(codes.shape)}  loss={loss.item():.4f}  "
              f"(expect≈ln{L}={math.log(L):.4f})")
