"""
lm_train.py — Latent autoregression with re-encoded grounding (Option A-grounded).

Training:
  - Encoder sees ground-truth bytes (teacher-forced) every step.
  - Loss = rec_loss (decoder CE) + pred_loss (LM next-latent CE).
  - LM is causal Transformer; sees full z_hat sequence teacher-forced.

Inference (A-grounded):
  - Sample z_next from LM.
  - Decode z_next → bytes.
  - Re-encode decoded bytes → z_grounded (snaps back to encoder manifold).
  - Feed z_grounded to LM as next input (not the raw sampled z_next).

Usage:
    python data.py --src ../datasets/quran_uthmani.txt --out data/quran
    python lm_train.py --data data/quran --bottleneck bsq

JAX note: set JAX_PLATFORMS=cpu before import (MPS has no PRNG support).
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import json
import math
import time
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from data import ByteDataset
from codec import ByteEncoder, ByteDecoder
from model import LatentLM
from bsq import BSQEncoder, BSQLMHead
from fsq import FSQEncoder, FSQLMHead


# ---------------------------------------------------------------------------
# Full system module
# ---------------------------------------------------------------------------

class LatentARSystem(eqx.Module):
    """
    Complete latent autoregression system:
      ByteEncoder → quantizer (BSQ or FSQ) → LatentLM → LM head
                                           ↘ ByteDecoder
    """

    byte_enc: ByteEncoder
    quant_enc: eqx.Module        # BSQEncoder | FSQEncoder
    byte_dec: ByteDecoder
    lm: LatentLM
    lm_head: eqx.Module          # BSQLMHead | FSQLMHead
    K: int = eqx.field(static=True)

    def __init__(
        self,
        K: int,
        d_byte: int,
        d_enc: int,
        enc_hidden: Tuple[int, ...],
        d_q: int,
        bottleneck: str,          # "bsq" | "fsq2" | "fsq8"
        L: int,                   # FSQ levels (ignored for BSQ)
        d_model: int,
        n_layers: int,
        n_heads: int,
        dec_hidden: Tuple[int, ...],
        *,
        key: jax.Array,
    ):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        self.K = K
        self.byte_enc = ByteEncoder(K=K, d_byte=d_byte, d_enc=d_enc, hidden=enc_hidden, key=k1)
        self.byte_dec = ByteDecoder(d_q=d_q, K=K, hidden=dec_hidden, key=k3)
        self.lm = LatentLM(d_q=d_q, d_model=d_model, n_layers=n_layers, n_heads=n_heads, key=k4)

        if bottleneck == "bsq":
            self.quant_enc = BSQEncoder(d_in=d_enc, d_q=d_q, key=k2)
            self.lm_head = BSQLMHead(d_model=d_model, d_q=d_q, key=k5)
        else:
            self.quant_enc = FSQEncoder(d_in=d_enc, d_q=d_q, L=L, key=k2)
            self.lm_head = FSQLMHead(d_model=d_model, d_q=d_q, L=L, key=k5)

    def encode_chunk(self, chunk: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """chunk: (K,) int → z_hat: (d_q,), codes: (d_q,)"""
        u = self.byte_enc._single(chunk)
        return self.quant_enc(u)

    def decode_chunk(self, z_hat: jax.Array) -> jax.Array:
        """z_hat: (d_q,) → logits: (K, 256)"""
        return self.byte_dec._single(z_hat)

    def param_count(self) -> int:
        return sum(v.size for v in jax.tree_util.tree_leaves(self))


# ---------------------------------------------------------------------------
# Loss (teacher-forced)
# ---------------------------------------------------------------------------

def compute_loss(
    system: LatentARSystem,
    byte_chunks: jax.Array,       # (B, T, K) int32
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Returns (total_loss, rec_loss, pred_loss).

    Teacher-forced encoding:
      Encoder sees ground-truth bytes every step.
      LM sees ground-truth z_hat sequence (not its own predictions).
    """
    B, T, K = byte_chunks.shape

    # --- Encode all chunks (teacher-forced) ---
    # encode_chunk: (K,) → (d_q,), (d_q,)
    # double-vmap: over T then over B
    encode_seq = jax.vmap(system.encode_chunk)          # (T, K) → (T, d_q), (T, d_q)
    encode_batch = jax.vmap(encode_seq)                 # (B, T, K) → ...
    z_hat, codes = encode_batch(byte_chunks)            # (B, T, d_q) each

    # --- Reconstruction loss: decoder sees z_hat, predicts byte_chunks ---
    decode_flat = jax.vmap(system.decode_chunk)         # (B*T, d_q) → (B*T, K, 256)
    logits = decode_flat(z_hat.reshape(B * T, -1))      # (B*T, K, 256)
    targets_flat = byte_chunks.reshape(B * T, K)        # (B*T, K)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    rec_nll = -log_probs[
        jnp.arange(B * T)[:, None],
        jnp.arange(K)[None, :],
        targets_flat,
    ]                                                   # (B*T, K)
    rec_loss = jnp.mean(rec_nll)                        # mean nats per byte

    # --- LM forward (causal, teacher-forced z_hat) ---
    # lm: (T, d_q) → (T, d_model), vmapped over B
    h = jax.vmap(system.lm)(z_hat)                     # (B, T, d_model)

    # --- Prediction loss: h[:, :-1] predicts codes[:, 1:] ---
    # lm_head: (d_model,) → logits; vmapped over B*(T-1)
    h_src = h[:, :-1].reshape(B * (T - 1), -1)         # (B*(T-1), d_model)
    c_tgt = codes[:, 1:].reshape(B * (T - 1), -1)      # (B*(T-1), d_q)

    lm_head_fn = jax.vmap(system.lm_head)
    lm_logits = lm_head_fn(h_src)                       # (B*(T-1), d_q[, L])

    pred_loss = system.lm_head.loss(lm_logits, c_tgt)  # scalar

    total = rec_loss + pred_loss
    return total, (rec_loss, pred_loss)


# ---------------------------------------------------------------------------
# BPB computation
# ---------------------------------------------------------------------------

def compute_bpb(
    system: LatentARSystem,
    val_dataset: ByteDataset,
    T: int,
    n_batches: int = 32,
    rng: np.random.Generator = None,
) -> dict:
    """
    Compute bits-per-byte on val set.
    BPB = total_loss / ln(2)  (both rec and pred contribute).
    Also report them separately.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    K = system.K
    total_loss_acc = rec_acc = pred_acc = 0.0

    for _ in range(n_batches):
        raw = val_dataset.random_batch(8, rng)           # (8, T*K+1)
        chunks_flat = raw[:, : T * K].astype(np.int32)
        chunks = chunks_flat.reshape(8, T, K)
        jchunks = jnp.array(chunks)
        total, (rec, pred) = compute_loss(system, jchunks)
        total_loss_acc += float(total)
        rec_acc += float(rec)
        pred_acc += float(pred)

    n = n_batches
    return {
        "bpb_total": total_loss_acc / n / math.log(2),
        "bpb_rec":   rec_acc / n / math.log(2),
        "bpb_pred":  pred_acc / n / math.log(2),
        "nats_total": total_loss_acc / n,
        "nats_rec":   rec_acc / n,
        "nats_pred":  pred_acc / n,
    }


# ---------------------------------------------------------------------------
# Training step (JIT'd)
# ---------------------------------------------------------------------------

@eqx.filter_jit
def train_step(
    system: LatentARSystem,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    byte_chunks: jax.Array,
) -> Tuple[LatentARSystem, optax.OptState, jax.Array, jax.Array, jax.Array]:
    (total, (rec, pred)), grads = eqx.filter_value_and_grad(
        compute_loss, has_aux=True
    )(system, byte_chunks)

    updates, new_opt_state = optimizer.update(
        grads, opt_state, eqx.filter(system, eqx.is_array)
    )
    new_system = eqx.apply_updates(system, updates)
    return new_system, new_opt_state, total, rec, pred


# ---------------------------------------------------------------------------
# A-grounded inference (greedy, no sampling for now)
# ---------------------------------------------------------------------------

def generate_grounded(
    system: LatentARSystem,
    prompt_bytes: np.ndarray,         # flat byte array
    n_new_chunks: int,
) -> np.ndarray:
    """
    Option A-grounded generation:
      1. Encode prompt chunks (ground-truth, so fully grounded).
      2. For each new chunk:
         a. LM forward on z_history → sample/argmax next codes.
         b. Decode codes → bytes.
         c. Re-encode decoded bytes → z_grounded (snaps back to manifold).
         d. Append z_grounded (not raw z_next) to history.

    Returns flat byte array (prompt + generated).
    """
    K = system.K
    n_prompt = len(prompt_bytes) // K

    # Encode prompt
    z_history = []
    for i in range(n_prompt):
        chunk = jnp.array(prompt_bytes[i * K : (i + 1) * K], dtype=jnp.int32)
        z_hat, _ = system.encode_chunk(chunk)
        z_history.append(z_hat)

    generated_bytes = list(prompt_bytes)

    for _ in range(n_new_chunks):
        # LM forward on history → hidden for last position
        z_seq = jnp.stack(z_history)                        # (T, d_q)
        h = system.lm(z_seq)                                # (T, d_model)
        h_last = h[-1]                                       # (d_model,)

        # LM head: greedy argmax per code dimension
        lm_logits = system.lm_head(h_last)
        if isinstance(system.lm_head, BSQLMHead):
            # BSQ: sigmoid threshold
            bits_pred = (lm_logits > 0).astype(jnp.int32)
            # Reconstruct z_hat from predicted bits: ±1/sqrt(d_q)
            b = (bits_pred.astype(jnp.float32) * 2 - 1)    # {-1,+1}
            z_next = b / jnp.sqrt(system.lm_head.linear.out_features)
        else:
            # FSQ: argmax per dim
            if lm_logits.ndim == 2:                          # (d_q, L)
                codes_pred = jnp.argmax(lm_logits, axis=-1) # (d_q,)
            else:                                            # (d_q,) for L=2
                codes_pred = (lm_logits > 0).astype(jnp.int32)
            L = system.quant_enc.L
            half = (L - 1) / 2.0
            z_next = codes_pred.astype(jnp.float32) - half  # float levels

        # Decode z_next → bytes (greedy)
        dec_logits = system.decode_chunk(z_next)             # (K, 256)
        bytes_pred = jnp.argmax(dec_logits, axis=-1)         # (K,) int

        # Re-encode decoded bytes → z_grounded (A-grounded: snaps to manifold)
        z_grounded, _ = system.encode_chunk(bytes_pred.astype(jnp.int32))

        z_history.append(z_grounded)
        generated_bytes.extend(bytes_pred.tolist())

    return np.array(generated_bytes, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    data_dir: str,
    bottleneck: str = "bsq",
    K: int = 8,
    T: int = 64,                # latent sequence length (= T*K bytes context)
    batch_size: int = 32,
    d_q: int = 18,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    lr: float = 3e-4,
    total_steps: int = 5000,
    warmup_steps: int = 200,
    grad_clip: float = 1.0,
    log_every: int = 100,
    eval_every: int = 500,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    # Load dataset
    train_ds = ByteDataset(str(Path(data_dir) / "train.npy"), seq_len=T * K)
    val_ds   = ByteDataset(str(Path(data_dir) / "val.npy"),   seq_len=T * K)

    with open(Path(data_dir) / "meta.json") as f:
        meta = json.load(f)
    print(f"Dataset: {meta['train_bytes']:,} train bytes, {meta['val_bytes']:,} val bytes")

    # Build system
    L = {"bsq": 2, "fsq2": 2, "fsq8": 8}[bottleneck]
    system = LatentARSystem(
        K=K, d_byte=16, d_enc=128, enc_hidden=(256,),
        d_q=d_q, bottleneck=bottleneck if bottleneck != "fsq2" else "fsq",
        L=L,
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        dec_hidden=(256,),
        key=key,
    )
    print(f"System params: {system.param_count():,}")
    print(f"Bottleneck: {bottleneck}  d_q={d_q}  codebook=2^{d_q if bottleneck in ('bsq','fsq2') else d_q * int(math.log2(L))}")

    # Optimizer: AdamW with cosine LR + warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr * 0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(schedule, weight_decay=0.1),
    )
    opt_state = optimizer.init(eqx.filter(system, eqx.is_array))

    # Training
    t0 = time.time()
    for step in range(1, total_steps + 1):
        raw = train_ds.random_batch(batch_size, rng)           # (B, T*K+1)
        chunks = raw[:, : T * K].reshape(batch_size, T, K).astype(np.int32)
        jchunks = jnp.array(chunks)

        system, opt_state, total, rec, pred = train_step(
            system, opt_state, optimizer, jchunks
        )
        total, rec, pred = float(total), float(rec), float(pred)

        if step % log_every == 0:
            elapsed = time.time() - t0
            lr_now = float(schedule(step))
            print(
                f"step {step:5d}  "
                f"loss={float(total):.4f}  rec={float(rec):.4f}  pred={float(pred):.4f}  "
                f"bpb≈{float(total)/math.log(2):.3f}  "
                f"lr={lr_now:.2e}  {elapsed:.1f}s"
            )

        if step % eval_every == 0:
            metrics = compute_bpb(system, val_ds, T, n_batches=16, rng=rng)
            print(
                f"  [val]  bpb={metrics['bpb_total']:.4f}  "
                f"bpb_rec={metrics['bpb_rec']:.4f}  bpb_pred={metrics['bpb_pred']:.4f}"
            )

    print(f"\nTraining done in {time.time()-t0:.1f}s")
    return system


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default="data/quran", help="data dir from data.py")
    parser.add_argument("--bottleneck",  default="bsq",        choices=["bsq", "fsq2", "fsq8"])
    parser.add_argument("--K",           type=int, default=8,   help="bytes per latent chunk")
    parser.add_argument("--T",           type=int, default=64,  help="latent sequence length")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--d_q",         type=int, default=18)
    parser.add_argument("--d_model",     type=int, default=128)
    parser.add_argument("--n_layers",    type=int, default=4)
    parser.add_argument("--n_heads",     type=int, default=4)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--steps",       type=int, default=5000)
    parser.add_argument("--warmup",      type=int, default=200)
    parser.add_argument("--log_every",   type=int, default=100)
    parser.add_argument("--eval_every",  type=int, default=500)
    parser.add_argument("--seed",        type=int, default=0)
    args = parser.parse_args()

    train(
        data_dir=args.data,
        bottleneck=args.bottleneck,
        K=args.K,
        T=args.T,
        batch_size=args.batch_size,
        d_q=args.d_q,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        lr=args.lr,
        total_steps=args.steps,
        warmup_steps=args.warmup,
        log_every=args.log_every,
        eval_every=args.eval_every,
        seed=args.seed,
    )
