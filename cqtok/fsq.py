"""
fsq.py — Finite Scalar Quantization (FSQ), with L=2 and L=8 focus.

Theory (from research/LM.md §1.2.2):
  Encoder: project u → d_q scalars, tanh-bound to (-(L-1)/2, (L-1)/2), round (STE).
  Codebook: implicit, size L^d_q.
  LM head: d_q independent categoricals over L levels. Loss: per-dim CE.

Level choices for K=8 bytes (256-way each):
  Full space = 256^8 = 2^64. Practical text needs:
    L=2,  d_q=18  → codebook 2^18  ≈ 262K   ← 2-level FSQ, same size as BSQ d_q=18
    L=8,  d_q=6   → codebook 8^6  = 2^18 ≈ 262K   ← compact, fewer dims
    L=8,  d_q=8   → codebook 8^8  = 2^24 ≈ 16M    ← balanced default
    L=2,  d_q=24  → codebook 2^24 ≈ 16M            ← 2-level, balanced

  L=2 vs L=8 tradeoff:
    L=2:  d_q must be larger (1 bit/dim). Cheaper LM head (BCE instead of 8-way CE).
          Sign-STE is coarser than round-STE.
    L=8:  d_q can be smaller (3 bits/dim). Standard softmax head.
          Rounding STE is milder. Better gradient flow per dim.
  Recommendation: L=8 for the LM head; L=2 as an ablation / when BCE head is preferred.

L=2 special case:
  tanh(u) bounds to (-1, 1). Levels are {-0.5, 0.5} (half-integers).
  round(tanh(u) * 0.5) always = 0 with numpy round. Instead we threshold at 0:
    z_hat = sign(tanh(u)) * 0.5          with STE
  This is equivalent to per-dimension sign quantization without L2-norm (contrast with BSQ).

JAX note: set JAX_PLATFORMS=cpu before import (MPS has no PRNG support).
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import math
from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx


# ---------------------------------------------------------------------------
# Core quantization function (functional, vmappable)
# ---------------------------------------------------------------------------

def _fsq_quantize(z_pre: jax.Array, L: int) -> Tuple[jax.Array, jax.Array]:
    """
    z_pre: (..., d_q) raw projections.
    Returns:
      z_hat  : (..., d_q) float, STE-quantized, values in {-(L-1)/2, ..., (L-1)/2}.
      codes  : (..., d_q) int32, values in {0, ..., L-1}.
    """
    half = (L - 1) / 2.0

    if L == 2:
        # tanh → sign threshold (avoids the rounding-to-zero issue at half-integer levels)
        z_cont = jnp.tanh(z_pre) * 0.5     # ∈ (-0.5, 0.5)
        z_q = jnp.where(z_cont >= 0, 0.5, -0.5)
        z_hat = z_cont + jax.lax.stop_gradient(z_q - z_cont)  # STE
        codes = (z_q + 0.5).astype(jnp.int32)                 # {0, 1}
    else:
        # General: tanh to (-half, half), round (STE)
        z_cont = jnp.tanh(z_pre) * half    # ∈ (-half, half)
        z_q = jnp.round(z_cont)            # levels in {-half, ..., half} (int or half-int)
        z_hat = z_cont + jax.lax.stop_gradient(z_q - z_cont)  # STE
        codes = (z_hat + half).astype(jnp.int32)               # {0, ..., L-1}

    return z_hat, codes


# ---------------------------------------------------------------------------
# Encoder: projection + quantization
# ---------------------------------------------------------------------------

class FSQEncoder(eqx.Module):
    """
    Projects d_in → d_q, then FSQ-quantizes to L levels per dimension.
    """

    proj: eqx.nn.Linear
    L: int = eqx.field(static=True)

    def __init__(self, d_in: int, d_q: int, L: int, *, key: jax.Array):
        assert L >= 2, "L must be ≥ 2"
        self.proj = eqx.nn.Linear(d_in, d_q, use_bias=False, key=key)
        self.L = L

    @property
    def d_q(self) -> int:
        return self.proj.out_features

    @property
    def codebook_bits(self) -> float:
        return self.d_q * math.log2(self.L)

    @property
    def codebook_size(self) -> int:
        return self.L ** self.d_q

    def __call__(self, u: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        u: (d_in,) or (B, d_in)
        Returns:
          z_hat : same leading dims, (d_q,) — STE-quantized float
          codes : same leading dims, (d_q,) — int32 ∈ {0, ..., L-1}
        """
        batched = u.ndim > 1
        z_pre = jax.vmap(self.proj)(u) if batched else self.proj(u)
        return _fsq_quantize(z_pre, self.L)


# ---------------------------------------------------------------------------
# LM head: predict next chunk's FSQ codes
# ---------------------------------------------------------------------------

class FSQLMHead(eqx.Module):
    """
    LM head for predicting FSQ codes.
      L=2  → d_q logits (one per bit, BCE loss)
      L>2  → d_q * L logits reshaped to (d_q, L), per-dim CE loss
    """

    linear: eqx.nn.Linear
    L: int = eqx.field(static=True)

    def __init__(self, d_model: int, d_q: int, L: int, *, key: jax.Array):
        out_features = d_q if L == 2 else d_q * L
        self.linear = eqx.nn.Linear(d_model, out_features, key=key)
        self.L = L

    @property
    def d_q(self) -> int:
        if self.L == 2:
            return self.linear.out_features
        return self.linear.out_features // self.L

    def __call__(self, h: jax.Array) -> jax.Array:
        """
        h: (d_model,) or (B, d_model)
        Returns:
          L=2 : (..., d_q)     — raw logits for Bernoulli
          L>2 : (..., d_q, L)  — raw logits per level
        """
        batched = h.ndim > 1
        out = jax.vmap(self.linear)(h) if batched else self.linear(h)
        if self.L > 2:
            leading = out.shape[:-1]
            out = out.reshape(*leading, self.d_q, self.L)
        return out

    def loss(self, logits: jax.Array, codes: jax.Array) -> jax.Array:
        """
        logits: (..., d_q) for L=2; (..., d_q, L) for L>2
        codes:  (..., d_q) int32 ∈ {0, ..., L-1}
        Returns: scalar mean loss.
        """
        if self.L == 2:
            # BCE: codes ∈ {0,1} → Bernoulli targets
            t = codes.astype(jnp.float32)
            bce = (
                jnp.maximum(logits, 0.0)
                - logits * t
                + jnp.log1p(jnp.exp(-jnp.abs(logits)))
            )
            return jnp.mean(bce)
        else:
            # Per-dim softmax CE: logits (..., d_q, L), targets (..., d_q)
            # Flatten leading dims for simplicity
            flat_logits = logits.reshape(-1, self.L)   # (N*d_q, L)
            flat_codes = codes.reshape(-1)              # (N*d_q,)
            log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
            # gather log-prob of the correct level
            nll = -log_probs[jnp.arange(flat_codes.shape[0]), flat_codes]
            return jnp.mean(nll)


# ---------------------------------------------------------------------------
# Full FSQ module
# ---------------------------------------------------------------------------

class FSQ(eqx.Module):
    """Full FSQ: encoder quantizer + LM prediction head."""

    encoder: FSQEncoder
    lm_head: FSQLMHead

    def __init__(self, d_in: int, d_q: int, L: int, d_model: int, *, key: jax.Array):
        k1, k2 = jax.random.split(key)
        self.encoder = FSQEncoder(d_in, d_q, L, key=k1)
        self.lm_head = FSQLMHead(d_model, d_q, L, key=k2)

    @property
    def L(self) -> int:
        return self.encoder.L

    @property
    def d_q(self) -> int:
        return self.encoder.d_q

    @property
    def codebook_bits(self) -> float:
        return self.encoder.codebook_bits

    @property
    def codebook_size(self) -> int:
        return self.encoder.codebook_size

    def encode(self, u: jax.Array) -> Tuple[jax.Array, jax.Array]:
        return self.encoder(u)

    def predict(self, h: jax.Array) -> jax.Array:
        return self.lm_head(h)

    def prediction_loss(self, h: jax.Array, target_codes: jax.Array) -> jax.Array:
        logits = self.lm_head(h)
        return self.lm_head.loss(logits, target_codes)


# ---------------------------------------------------------------------------
# Reconstruction accuracy helper
# ---------------------------------------------------------------------------

def reconstruction_accuracy(
    codes_decoded: jax.Array,   # (N, d_q) int32 — codes decoded from z_hat
    codes_original: jax.Array,  # (N, d_q) int32 — original encoder codes
) -> jax.Array:
    """Fraction of chunks where all d_q codes match exactly."""
    per_chunk = jnp.all(codes_decoded == codes_original, axis=-1)
    return jnp.mean(per_chunk.astype(jnp.float32))


def codes_from_z_hat(z_hat: jax.Array, L: int) -> jax.Array:
    """Recover integer codes from a STE-quantized z_hat float array."""
    half = (L - 1) / 2.0
    if L == 2:
        return (z_hat >= 0).astype(jnp.int32)
    return (jnp.round(z_hat) + half).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Codebook sizing reference table for K=8 bytes
# ---------------------------------------------------------------------------

def print_sizing_table():
    print("Codebook sizing for K=8 bytes (256-way each), full space = 2^64\n")
    print(f"{'variant':<22} {'d_q':>4} {'L':>4} {'bits':>6} {'codes':>12}  note")
    print("-" * 72)
    configs = [
        ("BSQ",             18,  2, "compact text, this corpus"),
        ("FSQ L=2",         18,  2, "same codebook as BSQ d_q=18"),
        ("FSQ L=8",          6,  8, "same codebook, fewer dims"),
        ("BSQ balanced",    24,  2, "general text at scale"),
        ("FSQ L=2 bal.",    24,  2, "same"),
        ("FSQ L=8 bal.",     8,  8, "same"),
        ("BSQ multilingual",36,  2, "code-heavy / UTF-8 heavy"),
        ("FSQ L=8 multi.",  12,  8, "same"),
        ("Full K=8 space",  64,  2, "all 256^8 combos"),
    ]
    for name, d_q, L, note in configs:
        bits = d_q * math.log2(L)
        size = L ** d_q
        size_str = f"2^{bits:.0f}" if bits == int(bits) else f"2^{bits:.1f}"
        print(f"{name:<22} {d_q:>4} {L:>4} {bits:>6.1f} {size_str:>12}  {note}")


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_sizing_table()
    print()

    key = jax.random.PRNGKey(42)
    d_in, d_model = 256, 128
    B = 32

    for L, d_q, label in [(2, 18, "FSQ L=2 d_q=18"), (8, 6, "FSQ L=8 d_q=6")]:
        fsq = FSQ(d_in=d_in, d_q=d_q, L=L, d_model=d_model, key=key)

        key, k1, k2 = jax.random.split(key, 3)
        u = jax.random.normal(k1, (B, d_in))
        h = jax.random.normal(k2, (B, d_model))

        z_hat, codes = fsq.encode(u)
        loss = fsq.prediction_loss(h, codes)

        param_count = sum(x.size for x in jax.tree_util.tree_leaves(fsq))
        expected_loss = math.log(L)  # untrained random: log(L) nats

        print(f"{label}")
        print(f"  codebook: {fsq.codebook_size:,} codes ({fsq.codebook_bits:.1f} bits)")
        print(f"  z_hat shape: {z_hat.shape},  codes range: {int(codes.min())}..{int(codes.max())}")
        print(f"  loss: {float(loss):.4f}  (untrained expect ≈ ln({L}) = {expected_loss:.4f})")
        print(f"  params: {param_count:,}")
        print()
