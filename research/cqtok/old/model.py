"""
model.py — Causal Transformer with RoPE.

Used as backbone for both byte-level baseline and the latent LM.

JAX note: set JAX_PLATFORMS=cpu before import (MPS has no PRNG support).
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import math
from typing import Sequence

import jax
import jax.numpy as jnp
import equinox as eqx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seq(layer: eqx.nn.Linear, x: jax.Array) -> jax.Array:
    """Apply Linear to (..., in) → (..., out) via matmul (avoids vmap overhead)."""
    out = x @ layer.weight.T
    if layer.bias is not None:
        out = out + layer.bias
    return out


def _rotate_half(x: jax.Array) -> jax.Array:
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rope(
    q: jax.Array, k: jax.Array, positions: jax.Array, theta: float = 10_000.0
) -> tuple[jax.Array, jax.Array]:
    """
    q, k : (T, H, d_head)
    positions : (T,) int
    """
    d = q.shape[-1]
    half = d // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(half, dtype=jnp.float32) / half))
    angles = jnp.outer(positions.astype(jnp.float32), inv_freq)  # (T, half)
    cos = jnp.concatenate([jnp.cos(angles), jnp.cos(angles)], axis=-1)[:, None, :]  # (T,1,d)
    sin = jnp.concatenate([jnp.sin(angles), jnp.sin(angles)], axis=-1)[:, None, :]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class RMSNorm(eqx.Module):
    weight: jax.Array
    eps: float = eqx.field(static=True)

    def __init__(self, d: int, eps: float = 1e-6):
        self.weight = jnp.ones(d)
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight


class CausalAttention(eqx.Module):
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear
    n_heads: int = eqx.field(static=True)
    rope_theta: float = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope_theta: float = 10_000.0,
        *,
        key: jax.Array,
    ):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.q_proj = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k1)
        self.k_proj = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k2)
        self.v_proj = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k3)
        self.o_proj = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k4)
        self.n_heads = n_heads
        self.rope_theta = rope_theta

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (T, d_model)
        T, d = x.shape
        H, d_h = self.n_heads, d // self.n_heads

        q = _seq(self.q_proj, x).reshape(T, H, d_h)
        k = _seq(self.k_proj, x).reshape(T, H, d_h)
        v = _seq(self.v_proj, x).reshape(T, H, d_h)

        q, k = _apply_rope(q, k, jnp.arange(T), self.rope_theta)

        # scores: (H, T, T)
        scores = jnp.einsum("qhd,khd->hqk", q, k) * (d_h ** -0.5)
        causal_mask = jnp.triu(jnp.full((T, T), float("-inf")), k=1)
        scores = scores + causal_mask[None]

        attn = jax.nn.softmax(scores, axis=-1)          # (H, T, T)
        out = jnp.einsum("hqk,khd->qhd", attn, v)       # (T, H, d_h)
        return _seq(self.o_proj, out.reshape(T, d))


class SwiGLUFFN(eqx.Module):
    gate_proj: eqx.nn.Linear
    up_proj: eqx.nn.Linear
    down_proj: eqx.nn.Linear

    def __init__(self, d_model: int, expansion: int = 4, *, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        # 2/3 factor keeps parameter count equivalent to a standard 4x FFN
        d_h = int(d_model * expansion * 2 / 3)
        d_h = (d_h + 63) // 64 * 64   # round up to multiple of 64
        self.gate_proj = eqx.nn.Linear(d_model, d_h, use_bias=False, key=k1)
        self.up_proj   = eqx.nn.Linear(d_model, d_h, use_bias=False, key=k2)
        self.down_proj = eqx.nn.Linear(d_h, d_model, use_bias=False, key=k3)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (T, d_model)
        gate = jax.nn.silu(_seq(self.gate_proj, x))
        up = _seq(self.up_proj, x)
        return _seq(self.down_proj, gate * up)


class TransformerBlock(eqx.Module):
    attn: CausalAttention
    ffn: SwiGLUFFN
    norm1: RMSNorm
    norm2: RMSNorm

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rope_theta: float = 10_000.0,
        *,
        key: jax.Array,
    ):
        k1, k2 = jax.random.split(key)
        self.attn  = CausalAttention(d_model, n_heads, rope_theta, key=k1)
        self.ffn   = SwiGLUFFN(d_model, key=k2)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CausalTransformer(eqx.Module):
    """
    Stack of TransformerBlocks + final RMSNorm.
    Input:  (T, d_model)
    Output: (T, d_model)
    """

    blocks: list
    norm: RMSNorm
    d_model: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        rope_theta: float = 10_000.0,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, n_layers)
        self.blocks = [
            TransformerBlock(d_model, n_heads, rope_theta, key=k) for k in keys
        ]
        self.norm = RMSNorm(d_model)
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads

    def __call__(self, x: jax.Array) -> jax.Array:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def param_count(self) -> int:
        return sum(v.size for v in jax.tree_util.tree_leaves(self))


class LatentLM(eqx.Module):
    """
    Causal LM over a quantized latent sequence.
      z_hat (T, d_q) → embed → Transformer → h (T, d_model)

    The LM head (BSQLMHead / FSQLMHead) lives outside; it predicts
    the next chunk's codes from h.
    """

    in_proj: eqx.nn.Linear        # d_q → d_model
    transformer: CausalTransformer

    def __init__(
        self,
        d_q: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        rope_theta: float = 10_000.0,
        *,
        key: jax.Array,
    ):
        k1, k2 = jax.random.split(key)
        self.in_proj = eqx.nn.Linear(d_q, d_model, use_bias=True, key=k1)
        self.transformer = CausalTransformer(d_model, n_layers, n_heads, rope_theta, key=k2)

    def __call__(self, z_hat: jax.Array) -> jax.Array:
        """z_hat: (T, d_q) → h: (T, d_model)"""
        x = _seq(self.in_proj, z_hat)
        return self.transformer(x)

    def param_count(self) -> int:
        return sum(v.size for v in jax.tree_util.tree_leaves(self))


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    d_q, d_model, n_layers, n_heads, T = 18, 128, 4, 4, 64

    lm = LatentLM(d_q=d_q, d_model=d_model, n_layers=n_layers, n_heads=n_heads, key=key)

    key, k = jax.random.split(key)
    z = jax.random.normal(k, (T, d_q))
    h = lm(z)

    print(f"LatentLM  d_q={d_q} d_model={d_model} L={n_layers} H={n_heads}")
    print(f"  input  z: {z.shape}")
    print(f"  output h: {h.shape}")
    print(f"  params:   {lm.param_count():,}")
    print(f"  h[0,:4]:  {h[0, :4]}")
