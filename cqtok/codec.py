"""
codec.py — Simple MLP byte encoder and decoder for Phase 1.

Encoder: K bytes → embed each byte → flatten → MLP → bottleneck (BSQ or FSQ)
Decoder: bottleneck z_hat → MLP → K * 256 logits → (K, 256)

These are the simplest possible byte encoder/decoder — no SSM, no cross-attention.
Good for validating the bottleneck in isolation (Phase 1 of research/LM.md §5).
Replace with the streaming SSM encoder/decoder in later phases.

JAX note: set JAX_PLATFORMS=cpu before import (MPS has no PRNG support).
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import math
from typing import Tuple, Sequence

import jax
import jax.numpy as jnp
import equinox as eqx


# ---------------------------------------------------------------------------
# Shared MLP building block
# ---------------------------------------------------------------------------

class MLP(eqx.Module):
    """Standard MLP with SiLU activations and optional residual."""

    layers: list

    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden: Sequence[int],
        *,
        key: jax.Array,
    ):
        dims = [d_in, *hidden, d_out]
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
            for i in range(len(dims) - 1)
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = jax.nn.silu(layer(x))
        return self.layers[-1](x)


# ---------------------------------------------------------------------------
# Byte Encoder: K bytes → flat encoding → bottleneck input
# ---------------------------------------------------------------------------

class ByteEncoder(eqx.Module):
    """
    Input: K integer byte values (uint8/int32).
    Output: d_enc floats, passed to BSQ/FSQ encoder.

    Architecture:
      byte_emb : Embedding(256, d_byte)   — learned byte embedding
      mlp      : [K*d_byte] → hidden → d_enc
    """

    byte_emb: eqx.nn.Embedding
    mlp: MLP
    K: int = eqx.field(static=True)

    def __init__(
        self,
        K: int = 8,
        d_byte: int = 16,
        d_enc: int = 128,
        hidden: Sequence[int] = (256,),
        *,
        key: jax.Array,
    ):
        k1, k2 = jax.random.split(key)
        self.byte_emb = eqx.nn.Embedding(256, d_byte, key=k1)
        self.mlp = MLP(K * d_byte, d_enc, hidden, key=k2)
        self.K = K

    @property
    def d_enc(self) -> int:
        return self.mlp.layers[-1].out_features

    def __call__(self, bytes_chunk: jax.Array) -> jax.Array:
        """
        bytes_chunk: (K,) int32 values in [0, 255]
                  or (B, K) for a batch.
        Returns: (d_enc,) or (B, d_enc).
        """
        batched = bytes_chunk.ndim > 1
        if batched:
            # vmap over batch dim
            return jax.vmap(self._single)(bytes_chunk)
        return self._single(bytes_chunk)

    def _single(self, chunk: jax.Array) -> jax.Array:
        # chunk: (K,) int
        embs = jax.vmap(self.byte_emb)(chunk)   # (K, d_byte)
        flat = embs.reshape(-1)                  # (K*d_byte,)
        return self.mlp(flat)                    # (d_enc,)


# ---------------------------------------------------------------------------
# Byte Decoder: bottleneck z_hat → K byte logits
# ---------------------------------------------------------------------------

class ByteDecoder(eqx.Module):
    """
    Input: z_hat (d_q floats from BSQ/FSQ).
    Output: (K, 256) logits — one 256-way distribution per byte position.

    Architecture:
      mlp : d_q → hidden → K * 256
      reshape → (K, 256)

    Training: cross-entropy on ground-truth bytes (one-shot, no masking for Phase 1).
    Upgrade path: add MaskGIT training in Phase 2 by passing masked input + z_hat.
    """

    mlp: MLP
    K: int = eqx.field(static=True)

    def __init__(
        self,
        d_q: int,
        K: int = 8,
        hidden: Sequence[int] = (256,),
        *,
        key: jax.Array,
    ):
        self.mlp = MLP(d_q, K * 256, hidden, key=key)
        self.K = K

    def __call__(self, z_hat: jax.Array) -> jax.Array:
        """
        z_hat: (d_q,) or (B, d_q)
        Returns: (K, 256) or (B, K, 256) logits.
        """
        batched = z_hat.ndim > 1
        if batched:
            return jax.vmap(self._single)(z_hat)
        return self._single(z_hat)

    def _single(self, z: jax.Array) -> jax.Array:
        out = self.mlp(z)                     # (K*256,)
        return out.reshape(self.K, 256)       # (K, 256)

    @staticmethod
    def reconstruction_loss(logits: jax.Array, target_bytes: jax.Array) -> jax.Array:
        """
        logits:       (K, 256) or (B, K, 256)
        target_bytes: (K,) or (B, K) int32 in [0, 255]
        Returns: scalar mean NLL (nats).
        """
        flat_logits = logits.reshape(-1, 256)
        flat_targets = target_bytes.reshape(-1)
        log_probs = jax.nn.log_softmax(flat_logits, axis=-1)
        nll = -log_probs[jnp.arange(flat_targets.shape[0]), flat_targets]
        return jnp.mean(nll)

    @staticmethod
    def decode_greedy(logits: jax.Array) -> jax.Array:
        """logits: (..., K, 256) → (..., K) int32 bytes."""
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)

    @staticmethod
    def reconstruction_accuracy(logits: jax.Array, target_bytes: jax.Array) -> jax.Array:
        """
        Fraction of chunks where all K bytes are predicted correctly.
        logits:       (N, K, 256)
        target_bytes: (N, K) int32
        """
        pred = jnp.argmax(logits, axis=-1)                    # (N, K)
        correct = jnp.all(pred == target_bytes, axis=-1)      # (N,)
        return jnp.mean(correct.astype(jnp.float32))


# ---------------------------------------------------------------------------
# Autoencoder: encoder + quantizer + decoder bundled for Phase 1
# ---------------------------------------------------------------------------

class ByteAutoencoder(eqx.Module):
    """
    Phase 1 autoencoder: ByteEncoder → quantizer (BSQ or FSQ) → ByteDecoder.

    The quantizer is passed in at construction time so this class is agnostic
    to whether BSQ or FSQ is used. The quantizer must implement:
      encode(u) → (z_hat, codes)

    Usage:
        from bsq import BSQEncoder, BSQLMHead
        from fsq import FSQEncoder, FSQLMHead
        ae = ByteAutoencoder(enc, quant_enc, dec)
        loss, z_hat, codes, logits = ae(chunk_bytes)
    """

    byte_enc: ByteEncoder
    quant_enc: eqx.Module   # BSQEncoder or FSQEncoder
    byte_dec: ByteDecoder

    def __init__(
        self,
        byte_enc: ByteEncoder,
        quant_enc: eqx.Module,
        byte_dec: ByteDecoder,
    ):
        self.byte_enc = byte_enc
        self.quant_enc = quant_enc
        self.byte_dec = byte_dec

    def __call__(
        self, chunk_bytes: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """
        chunk_bytes: (K,) or (B, K) int32 in [0, 255]
        Returns:
          rec_loss : scalar — reconstruction NLL
          z_hat    : (d_q,) or (B, d_q) — STE-quantized latent
          codes    : (d_q,) or (B, d_q) int32 — quantized codes
          logits   : (K, 256) or (B, K, 256) — decoder output
        """
        u = self.byte_enc(chunk_bytes)
        z_hat, codes = self.quant_enc(u)
        logits = self.byte_dec(z_hat)
        rec_loss = ByteDecoder.reconstruction_loss(logits, chunk_bytes)
        return rec_loss, z_hat, codes, logits


# ---------------------------------------------------------------------------
# Default hyperparameters for K=8, ~1MB corpus
# ---------------------------------------------------------------------------

def make_autoencoder_bsq(d_q: int = 18, K: int = 8, *, key: jax.Array) -> ByteAutoencoder:
    """BSQ autoencoder, compact text defaults."""
    from bsq import BSQEncoder
    k1, k2, k3 = jax.random.split(key, 3)
    byte_enc = ByteEncoder(K=K, d_byte=16, d_enc=128, hidden=(256,), key=k1)
    quant_enc = BSQEncoder(d_in=byte_enc.d_enc, d_q=d_q, key=k2)
    byte_dec = ByteDecoder(d_q=d_q, K=K, hidden=(256,), key=k3)
    return ByteAutoencoder(byte_enc, quant_enc, byte_dec)


def make_autoencoder_fsq(
    d_q: int = 6, L: int = 8, K: int = 8, *, key: jax.Array
) -> ByteAutoencoder:
    """FSQ autoencoder. Defaults: L=8 d_q=6 (compact, 2^18 codes)."""
    from fsq import FSQEncoder
    k1, k2, k3 = jax.random.split(key, 3)
    byte_enc = ByteEncoder(K=K, d_byte=16, d_enc=128, hidden=(256,), key=k1)
    quant_enc = FSQEncoder(d_in=byte_enc.d_enc, d_q=d_q, L=L, key=k2)
    byte_dec = ByteDecoder(d_q=d_q if L > 2 else d_q, K=K, hidden=(256,), key=k3)
    return ByteAutoencoder(byte_enc, quant_enc, byte_dec)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    K = 8

    for label, ae in [
        ("BSQ  d_q=18", make_autoencoder_bsq(d_q=18, K=K, key=key)),
        ("FSQ  L=8 d_q=6", make_autoencoder_fsq(d_q=6, L=8, K=K, key=key)),
        ("FSQ  L=2 d_q=18", make_autoencoder_fsq(d_q=18, L=2, K=K, key=key)),
    ]:
        key, k = jax.random.split(key)
        B = 32
        chunk_bytes = jax.random.randint(k, (B, K), 0, 256)

        rec_loss, z_hat, codes, logits = ae(chunk_bytes)
        recon_acc = ByteDecoder.reconstruction_accuracy(logits, chunk_bytes)

        param_count = sum(x.size for x in jax.tree_util.tree_leaves(ae))

        print(f"{label}")
        print(f"  rec_loss: {float(rec_loss):.4f} nats  (untrained expect ≈ ln(256)={math.log(256):.2f})")
        print(f"  recon accuracy: {float(recon_acc):.4f} (expect ≈ 0 untrained)")
        print(f"  z_hat: {z_hat.shape},  codes: {codes.shape}")
        print(f"  params: {param_count:,}")
        print()
