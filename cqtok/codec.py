"""
codec.py — MLP byte encoder and decoder (PyTorch, Phase 1).

ByteEncoder : K bytes → embed → MLP → d_enc
ByteDecoder : d_q floats → MLP → (K, 256) logits
ByteAutoencoder: encoder + quantizer + decoder for Phase 1 standalone training.
"""

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from bsq import BSQEncoder
from fsq import FSQEncoder


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: Sequence[int]):
        super().__init__()
        dims = [d_in, *hidden, d_out]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ByteEncoder(nn.Module):
    """K bytes (int) → byte embeddings → flatten → MLP → d_enc float."""

    def __init__(self, K: int = 8, d_byte: int = 16, d_enc: int = 128,
                 hidden: Sequence[int] = (256,)):
        super().__init__()
        self.K = K
        self.byte_emb = nn.Embedding(256, d_byte)
        self.mlp = MLP(K * d_byte, d_enc, hidden)

    def forward(self, chunks: torch.Tensor) -> torch.Tensor:
        """chunks: (*, K) int64 → (*, d_enc)"""
        emb = self.byte_emb(chunks)          # (*, K, d_byte)
        flat = emb.flatten(-2)               # (*, K*d_byte)
        return self.mlp(flat)


class ByteDecoder(nn.Module):
    """d_q floats → MLP → (K, 256) logits."""

    def __init__(self, d_q: int, K: int = 8, hidden: Sequence[int] = (256,)):
        super().__init__()
        self.K = K
        self.mlp = MLP(d_q, K * 256, hidden)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (*, d_q) → (*, K, 256)"""
        return self.mlp(z).unflatten(-1, (self.K, 256))

    @staticmethod
    def reconstruction_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: (*, K, 256)  targets: (*, K) int64 → scalar mean NLL."""
        return F.cross_entropy(logits.flatten(0, -2), targets.flatten())

    @staticmethod
    def reconstruction_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Fraction of chunks where all K bytes are correct."""
        pred = logits.argmax(-1)                        # (*, K)
        correct = (pred == targets).all(-1).float()     # (*,)
        return correct.mean()


class ByteAutoencoder(nn.Module):
    """Phase 1: ByteEncoder → BSQ/FSQ → ByteDecoder."""

    def __init__(self, byte_enc: ByteEncoder, quant_enc: nn.Module, byte_dec: ByteDecoder):
        super().__init__()
        self.byte_enc  = byte_enc
        self.quant_enc = quant_enc
        self.byte_dec  = byte_dec

    def forward(self, chunks: torch.Tensor):
        """
        chunks: (*, K) int64
        Returns (rec_loss, z_hat, codes, logits).
        """
        u = self.byte_enc(chunks)
        z_hat, codes = self.quant_enc(u)
        logits = self.byte_dec(z_hat)
        rec_loss = ByteDecoder.reconstruction_loss(logits, chunks)
        return rec_loss, z_hat, codes, logits


def make_autoencoder_bsq(d_q: int = 18, K: int = 8) -> ByteAutoencoder:
    return ByteAutoencoder(
        ByteEncoder(K=K, d_byte=16, d_enc=128, hidden=(256,)),
        BSQEncoder(d_in=128, d_q=d_q),
        ByteDecoder(d_q=d_q, K=K, hidden=(256,)),
    )


def make_autoencoder_fsq(d_q: int = 6, L: int = 8, K: int = 8) -> ByteAutoencoder:
    return ByteAutoencoder(
        ByteEncoder(K=K, d_byte=16, d_enc=128, hidden=(256,)),
        FSQEncoder(d_in=128, d_q=d_q, L=L),
        ByteDecoder(d_q=d_q, K=K, hidden=(256,)),
    )


if __name__ == "__main__":
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    for label, ae in [
        ("BSQ  d_q=18",    make_autoencoder_bsq(d_q=18)),
        ("FSQ L=8 d_q=6",  make_autoencoder_fsq(d_q=6, L=8)),
        ("FSQ L=2 d_q=18", make_autoencoder_fsq(d_q=18, L=2)),
    ]:
        ae = ae.to(dev)
        chunks = torch.randint(0, 256, (32, 8), device=dev)
        rec_loss, z_hat, codes, logits = ae(chunks)
        params = sum(p.numel() for p in ae.parameters())
        print(f"{label}  rec={rec_loss.item():.4f}  "
              f"(expect≈{math.log(256):.2f})  params={params:,}")
