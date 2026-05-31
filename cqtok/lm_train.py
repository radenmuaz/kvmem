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

import argparse
import json
import math
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from tqdm import tqdm

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
# Val eval on a raw text file (suratalfatihah.txt)
# ---------------------------------------------------------------------------

def eval_on_file(
    system: LatentARSystem,
    path: str,
    T: int,
    compute_device,
) -> dict:
    """
    Load a raw text file, chunk into (B, T, K) and compute loss on the full file.
    Returns BPB (total, rec, pred) and nats.
    """
    K = system.K
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8).astype(np.int32)
    n_chunks = len(raw) // K
    T_use = min(T, max(2, n_chunks - 1))        # at least 2 chunks for pred shift
    n_seqs = max(1, n_chunks // T_use)
    arr = raw[: n_seqs * T_use * K].reshape(n_seqs, T_use, K)

    with jax.default_device(compute_device):
        batch = jax.device_put(jnp.array(arr), compute_device)
        total, (rec, pred) = compute_loss(system, batch)

    return {
        "bpb":      float(total) / math.log(2),
        "bpb_rec":  float(rec)   / math.log(2),
        "bpb_pred": float(pred)  / math.log(2),
        "nats":     float(total),
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
    val_file: str = "../datasets/suratalfatihah.txt",
    bottleneck: str = "bsq",
    K: int = 8,
    T: int = 64,
    batch_size: int = 32,
    d_q: int = 18,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    lr: float = 3e-4,
    total_steps: int = 5000,
    warmup_steps: int = 200,
    grad_clip: float = 1.0,
    eval_every: int = 500,
    seed: int = 0,
):
    # --- Devices: random always on CPU, compute on MPS if available ---
    cpu = jax.devices("cpu")[0]
    try:
        compute_device = jax.devices("mps")[0]
        print(f"Compute device: {compute_device}")
    except Exception:
        compute_device = cpu
        print("MPS not available, using CPU")

    rng = np.random.default_rng(seed)

    # --- Dataset ---
    train_ds = ByteDataset(str(Path(data_dir) / "train.npy"), seq_len=T * K)
    with open(Path(data_dir) / "meta.json") as f:
        meta = json.load(f)
    print(f"Train: {meta['train_bytes']:,} bytes  |  Val: {val_file}")

    # --- Build system on CPU (uses random) ---
    L = {"bsq": 2, "fsq2": 2, "fsq8": 8}[bottleneck]
    bn = "bsq" if bottleneck == "bsq" else "fsq"
    codebook_bits = d_q if bottleneck in ("bsq", "fsq2") else d_q * int(math.log2(L))

    with jax.default_device(cpu):
        key = jax.random.PRNGKey(seed)
        system = LatentARSystem(
            K=K, d_byte=16, d_enc=128, enc_hidden=(256,),
            d_q=d_q, bottleneck=bn, L=L,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            dec_hidden=(256,),
            key=key,
        )
    print(f"Params: {system.param_count():,}  |  {bottleneck} d_q={d_q} codebook=2^{codebook_bits}")

    # --- Optimizer (init on CPU, transfer to compute device with model) ---
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr,
        warmup_steps=warmup_steps, decay_steps=total_steps, end_value=lr * 0.1,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adamw(schedule, weight_decay=0.1))
    opt_state = optimizer.init(eqx.filter(system, eqx.is_array))

    # Transfer model and optimizer state to compute device
    system   = jax.device_put(system,    compute_device)
    opt_state = jax.device_put(opt_state, compute_device)

    # --- Training loop ---
    with tqdm(total=total_steps, desc=bottleneck, unit="step") as pbar:
        for step in range(1, total_steps + 1):
            # Build batch on CPU, transfer to compute device
            raw = train_ds.random_batch(batch_size, rng)
            chunks_np = raw[:, : T * K].reshape(batch_size, T, K).astype(np.int32)

            with jax.default_device(compute_device):
                chunks = jax.device_put(jnp.array(chunks_np), compute_device)
                system, opt_state, total, rec, pred = train_step(
                    system, opt_state, optimizer, chunks
                )

            total, rec, pred = float(total), float(rec), float(pred)
            bpb = total / math.log(2)

            pbar.set_postfix(
                loss=f"{total:.4f}",
                bpb=f"{bpb:.3f}",
                rec=f"{rec:.4f}",
                pred=f"{pred:.4f}",
                refresh=False,
            )
            pbar.update(1)

            if step % eval_every == 0:
                m = eval_on_file(system, val_file, T, compute_device)
                tqdm.write(
                    f"  [val step {step}]"
                    f"  bpb={m['bpb']:.4f}"
                    f"  rec={m['bpb_rec']:.4f}"
                    f"  pred={m['bpb_pred']:.4f}"
                )

    return system


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/quran",                     help="data dir from data.py")
    parser.add_argument("--val_file",   default="../datasets/suratalfatihah.txt",  help="raw text file for val eval")
    parser.add_argument("--bottleneck", default="bsq", choices=["bsq", "fsq2", "fsq8"])
    parser.add_argument("--K",          type=int,   default=8)
    parser.add_argument("--T",          type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--d_q",        type=int,   default=18)
    parser.add_argument("--d_model",    type=int,   default=128)
    parser.add_argument("--n_layers",   type=int,   default=4)
    parser.add_argument("--n_heads",    type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--steps",      type=int,   default=5000)
    parser.add_argument("--warmup",     type=int,   default=200)
    parser.add_argument("--eval_every", type=int,   default=500)
    parser.add_argument("--seed",       type=int,   default=0)
    args = parser.parse_args()

    train(
        data_dir=args.data,
        val_file=args.val_file,
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
        eval_every=args.eval_every,
        seed=args.seed,
    )
