"""
bsq.py — Binary Spherical Quantization

Theory (from research/LM.md §1.2.3):
  Encoder: project u → v, L2-normalize → unit sphere, sign → ±1 bits.
  Codebook: implicit, size 2^d_q.
  LM head: d_q independent Bernoullis. Loss: per-bit BCE.

Codebook sizing for K=8 bytes (256-way each):
  Full space = 256^8 = 2^64 bits. Practical text needs far less:
    Arabic/English text entropy ≈ 1–3 bits/byte → K=8 → 8–24 bits.
    Add 8–12 bit safety margin.
  Recommended d_q:
    18  compact text (2^18  ≈ 262K codes)   ← default for Quran ~1.4MB
    24  balanced / general text (2^24 ≈ 16M)
    36  code-heavy / multilingual (2^36 ≈ 68B)

  For this corpus: ~170K chunks of 8 bytes → d_q=18 covers it with headroom.

JAX note: set JAX_PLATFORMS=cpu before import (MPS has no PRNG support).
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx


class BSQEncoder(eqx.Module):
    """
    Projects d_in → d_q, L2-normalizes, then quantizes via sign (STE).

    z_hat = (v / ||v||) + sg(sign(v/||v||) - v/||v||)     [STE]
    normalized to 1/sqrt(d_q) so ||z_hat|| ≈ 1.

    bits ∈ {0, 1}^d_q for BCE loss.
    """

    proj: eqx.nn.Linear

    def __init__(self, d_in: int, d_q: int, *, key: jax.Array):
        self.proj = eqx.nn.Linear(d_in, d_q, use_bias=False, key=key)

    @property
    def d_q(self) -> int:
        return self.proj.out_features

    def __call__(self, u: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        u: (d_in,) or (B, d_in)
        Returns:
          z_hat : same leading dims, (d_q,)  — STE-quantized float ±1/sqrt(d_q)
          bits  : same leading dims, (d_q,)  — int32 in {0, 1}
        """
        batched = u.ndim > 1
        v = jax.vmap(self.proj)(u) if batched else self.proj(u)
        v_norm = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
        b = jnp.sign(v_norm)                                   # {-1, +1}
        z_hat = v_norm + jax.lax.stop_gradient(b - v_norm)    # STE
        z_hat = z_hat / jnp.sqrt(self.d_q)
        bits = ((b + 1) / 2).astype(jnp.int32)                # {0, 1}
        return z_hat, bits


class BSQLMHead(eqx.Module):
    """
    LM head for predicting BSQ codes: linear → d_q logits (one per bit).

    Loss: numerically stable sigmoid BCE (per-bit, averaged).
    """

    linear: eqx.nn.Linear

    def __init__(self, d_model: int, d_q: int, *, key: jax.Array):
        self.linear = eqx.nn.Linear(d_model, d_q, key=key)

    def __call__(self, h: jax.Array) -> jax.Array:
        """h: (d_model,) or (B, d_model) → logits (d_q,) or (B, d_q)"""
        batched = h.ndim > 1
        return jax.vmap(self.linear)(h) if batched else self.linear(h)

    @staticmethod
    def loss(logits: jax.Array, bits: jax.Array) -> jax.Array:
        """
        logits: (..., d_q)
        bits:   (..., d_q) int32 {0, 1}   — target is next-chunk's bits
        Returns: scalar, mean per-bit BCE.
        """
        t = bits.astype(jnp.float32)
        # numerically stable: max(x,0) - x*t + log(1 + exp(-|x|))
        bce = jnp.maximum(logits, 0.0) - logits * t + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        return jnp.mean(bce)


# ---------------------------------------------------------------------------
# Convenience: pack encoder + LM head together for standalone experiments
# ---------------------------------------------------------------------------

class BSQ(eqx.Module):
    """Full BSQ module: encoder quantizer + LM prediction head."""

    encoder: BSQEncoder
    lm_head: BSQLMHead

    def __init__(self, d_in: int, d_q: int, d_model: int, *, key: jax.Array):
        k1, k2 = jax.random.split(key)
        self.encoder = BSQEncoder(d_in, d_q, key=k1)
        self.lm_head = BSQLMHead(d_model, d_q, key=k2)

    def encode(self, u: jax.Array) -> Tuple[jax.Array, jax.Array]:
        return self.encoder(u)

    def predict(self, h: jax.Array) -> jax.Array:
        return self.lm_head(h)

    def prediction_loss(self, h: jax.Array, target_bits: jax.Array) -> jax.Array:
        logits = self.lm_head(h)
        return BSQLMHead.loss(logits, target_bits)


# ---------------------------------------------------------------------------
# Reconstruction accuracy helper (for Phase 1 eval)
# ---------------------------------------------------------------------------

def reconstruction_accuracy(
    z_hat: jax.Array,        # (N, d_q) STE-quantized
    bits_original: jax.Array,  # (N, d_q) int32 {0,1}
) -> jax.Array:
    """
    Fraction of chunks where all d_q bits are reconstructed correctly.
    A chunk is "correct" if its decoder perfectly reproduces it from z_hat.
    Here we use bit-level exact match as a proxy.
    """
    bits_decoded = ((jnp.sign(z_hat) + 1) / 2).astype(jnp.int32)
    per_chunk_correct = jnp.all(bits_decoded == bits_original, axis=-1)
    return jnp.mean(per_chunk_correct.astype(jnp.float32))


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    key = jax.random.PRNGKey(0)

    d_in, d_q, d_model = 256, 18, 128
    bsq = BSQ(d_in=d_in, d_q=d_q, d_model=d_model, key=key)

    B = 32
    key, k1, k2 = jax.random.split(key, 3)
    u = jax.random.normal(k1, (B, d_in))
    h = jax.random.normal(k2, (B, d_model))

    z_hat, bits = bsq.encode(u)

    # shift target by 1 to simulate next-chunk prediction
    target_bits = bits  # in a real loop this would be the next chunk's bits
    loss = bsq.prediction_loss(h, target_bits)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(bsq))

    print(f"d_q={d_q}  codebook size = 2^{d_q} = {2**d_q:,}")
    print(f"K=8 bytes: full space = 2^64, practical coverage at d_q=18: 2^18 = {2**18:,}")
    print(f"z_hat shape: {z_hat.shape}  ||z_hat|| ≈ {float(jnp.mean(jnp.linalg.norm(z_hat, axis=-1))):.3f}")
    print(f"bits range: {int(bits.min())}..{int(bits.max())}")
    print(f"BCE loss: {float(loss):.4f}  (untrained, expect ≈ ln2 ≈ {math.log(2):.4f})")
    print(f"BSQ params: {param_count:,}")
