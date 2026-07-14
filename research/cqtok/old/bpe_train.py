"""
bpe_train.py — BPE baseline: SentencePiece BPE trained on the corpus + causal Transformer + RoPE.

BPE model is trained directly on datasets/quran_uthmani.txt (or any --src file).
Same Transformer backbone as lm_train.py (model.py).

BPB conversion:
    bpb = nats_per_token / (ln(2) * avg_bytes_per_token)
    avg_bytes_per_token measured on the full corpus.

Usage:
    # Prepare byte-level data (for lm_train baseline comparison)
    python data.py --src ../datasets/quran_uthmani.txt --out data/quran

    # Train BPE model + LM baseline
    python bpe_train.py --src ../datasets/quran_uthmani.txt --out data/quran
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax
import sentencepiece as spm
from tqdm import tqdm

from model import CausalTransformer
from lm_train import setup_run_dir, RunLogger


# ---------------------------------------------------------------------------
# Train SentencePiece BPE on a text file
# ---------------------------------------------------------------------------

def train_bpe(src: str, out_dir: str, vocab_size: int = 1024) -> Path:
    """
    Train a BPE model on src text. Returns path to the .model file.
    Uses byte_fallback=True so every byte is representable (no UNK).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_prefix = str(out / f"bpe_{vocab_size}")

    spm.SentencePieceTrainer.train(
        input=src,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        byte_fallback=True,          # every byte representable → no UNK
        character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        add_dummy_prefix=False,
        logspec="FATAL",             # suppress verbose trainer output
    )
    print(f"BPE model saved: {model_prefix}.model  (vocab={vocab_size})")
    return Path(model_prefix + ".model")


# ---------------------------------------------------------------------------
# Tokenize corpus → int32 arrays, save to disk
# ---------------------------------------------------------------------------

def tokenize_corpus(
    src: str,
    out_dir: str,
    model_path: str,
    val_frac: float = 0.1,
) -> dict:
    sp = spm.SentencePieceProcessor(model_file=model_path)
    text = Path(src).read_text(encoding="utf-8")
    tokens = np.array(sp.encode(text), dtype=np.int32)

    n_val = max(1, int(len(tokens) * val_frac))
    train_tok = tokens[: len(tokens) - n_val]
    val_tok   = tokens[len(tokens) - n_val :]

    out = Path(out_dir)
    np.save(out / "train_bpe.npy", train_tok)
    np.save(out / "val_bpe.npy",   val_tok)

    total_bytes = len(text.encode("utf-8"))
    meta = {
        "model_path": str(Path(model_path).resolve()),
        "vocab_size": sp.vocab_size(),
        "total_tokens": int(len(tokens)),
        "train_tokens": int(len(train_tok)),
        "val_tokens":   int(len(val_tok)),
        "total_bytes":  total_bytes,
        "avg_bytes_per_token": total_bytes / max(1, len(tokens)),
    }
    with open(out / "meta_bpe.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"Tokenized: {total_bytes:,} bytes → {len(tokens):,} tokens  "
        f"(avg {meta['avg_bytes_per_token']:.2f} bytes/token)"
    )
    return meta


# ---------------------------------------------------------------------------
# Token dataset
# ---------------------------------------------------------------------------

class TokenDataset:
    def __init__(self, npy_path: str, seq_len: int):
        self.data = np.load(npy_path, mmap_mode="r")
        self.seq_len = seq_len
        self.n = len(self.data) - seq_len

    def random_batch(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        offsets = rng.integers(0, self.n, size=batch_size)
        return np.stack([self.data[o : o + self.seq_len + 1] for o in offsets])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BPELM(eqx.Module):
    embed: eqx.nn.Embedding
    transformer: CausalTransformer
    head: eqx.nn.Linear

    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, *, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.embed       = eqx.nn.Embedding(vocab_size, d_model, key=k1)
        self.transformer = CausalTransformer(d_model, n_layers, n_heads, key=k2)
        self.head        = eqx.nn.Linear(d_model, vocab_size, use_bias=False, key=k3)

    def __call__(self, tokens: jax.Array) -> jax.Array:
        """tokens: (T,) → logits: (T, vocab_size)"""
        x = jax.vmap(self.embed)(tokens)
        h = self.transformer(x)
        return jax.vmap(self.head)(h)

    def param_count(self) -> int:
        return sum(v.size for v in jax.tree_util.tree_leaves(self))


# ---------------------------------------------------------------------------
# Token → byte-length map
# ---------------------------------------------------------------------------

def build_token_bytes(sp: spm.SentencePieceProcessor) -> np.ndarray:
    """
    Map each vocab id to the number of UTF-8 bytes it represents in decoded text.

    Rules:
      - Byte-fallback tokens <0xNN>   → 1 byte  (they encode a single raw byte)
      - Special tokens pad/unk/bos/eos → 0 bytes (don't contribute to byte count)
      - All other pieces: replace ▁ (word-boundary marker) with a space, then
        count UTF-8 bytes of the resulting string.

    Used for exact BPB:  bpb = sum(nll_i) / (sum(token_bytes_i) * ln 2)
    This is unbiased per-batch; the corpus-average shortcut is only exact corpus-wide.
    """
    specials = {"<pad>", "<unk>", "<s>", "</s>"}
    result = np.zeros(sp.vocab_size(), dtype=np.int32)
    for i in range(sp.vocab_size()):
        piece = sp.id_to_piece(i)
        if piece in specials:
            result[i] = 0
        elif piece.startswith("<0x") and piece.endswith(">"):
            result[i] = 1
        else:
            result[i] = len(piece.replace("▁", " ").encode("utf-8"))
    return result


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------

def _nll_per_token(model: BPELM, tokens: jax.Array) -> jax.Array:
    """tokens: (B, T+1) → per-token NLL (B, T)."""
    inputs  = tokens[:, :-1]
    targets = tokens[:, 1:]
    logits    = jax.vmap(model)(inputs)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    B, T = targets.shape
    return -log_probs[jnp.arange(B)[:, None], jnp.arange(T)[None, :], targets]


def token_nll(model: BPELM, tokens: jax.Array) -> jax.Array:
    """Scalar mean nats/token — used as the training loss."""
    return jnp.mean(_nll_per_token(model, tokens))


def exact_bpb(
    model: BPELM,
    tokens: jax.Array,          # (B, T+1) int32
    token_bytes: jax.Array,     # (vocab_size,) int32 — from build_token_bytes
) -> tuple[jax.Array, jax.Array]:
    """
    Exact BPB: sum(nll) / (sum(byte_lengths_of_targets) * ln 2).
    Also returns mean nats/token for logging.

    This is unbiased per-batch because each predicted token contributes
    proportionally to the bytes it represents.
    """
    nll_mat = _nll_per_token(model, tokens)        # (B, T)
    targets = tokens[:, 1:]                        # (B, T)
    byte_mat = token_bytes[targets]                # (B, T) int32
    total_bytes = jnp.sum(byte_mat)
    bpb = jnp.sum(nll_mat) / (total_bytes * math.log(2))
    nats = jnp.mean(nll_mat)
    return nats, bpb


def eval_file(model, path, sp, token_bytes_np, seq_len, compute_device) -> dict:
    """
    Evaluate on a raw text file using exact per-token byte lengths.
    token_bytes_np: (vocab_size,) int32 numpy array from build_token_bytes.
    """
    text   = Path(path).read_text(encoding="utf-8")
    tokens = np.array(sp.encode(text), dtype=np.int32)
    n_seqs = max(1, len(tokens) // seq_len)
    arr    = tokens[: n_seqs * seq_len].reshape(n_seqs, seq_len)
    batch  = np.concatenate([arr, arr[:, -1:]], axis=1)   # dummy last target

    with jax.default_device(compute_device):
        j_batch = jax.device_put(jnp.array(batch), compute_device)
        j_tbytes = jax.device_put(jnp.array(token_bytes_np), compute_device)
        nats, bpb = exact_bpb(model, j_batch, j_tbytes)

    return {"nats": float(nats), "bpb": float(bpb)}


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

@eqx.filter_jit
def train_step(model, opt_state, optimizer, tokens, token_bytes):
    """
    Returns (model, opt_state, nats, bpb).
    token_bytes: (vocab_size,) int32 — passed each call, treated as dynamic array by JAX.
    """
    (nats, bpb), grads = eqx.filter_value_and_grad(
        exact_bpb, has_aux=True
    )(model, tokens, token_bytes)
    updates, new_opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    return eqx.apply_updates(model, updates), new_opt_state, nats, bpb


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    src: str,
    out_dir: str,
    val_file: str = "../datasets/suratalfatihah.txt",
    vocab_size: int = 1024,
    seq_len: int = 256,
    batch_size: int = 32,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    lr: float = 3e-4,
    total_steps: int = 5000,
    warmup_steps: int = 200,
    grad_clip: float = 1.0,
    eval_every: int = 500,
    seed: int = 0,
    log_dir: str = "logs",
    run_name: str | None = None,
    no_date: bool = False,
    args_dict: dict | None = None,
):
    cpu = jax.devices("cpu")[0]
    try:
        compute_device = jax.devices("mps")[0]
    except Exception:
        compute_device = cpu

    rng = np.random.default_rng(seed)
    out = Path(out_dir)

    # Train BPE model if not already done
    model_path = out / f"bpe_{vocab_size}.model"
    if not model_path.exists():
        model_path = train_bpe(src, out_dir, vocab_size)

    # Tokenize corpus if not already done
    meta_path = out / "meta_bpe.json"
    if not meta_path.exists():
        meta = tokenize_corpus(src, out_dir, str(model_path))
    else:
        with open(meta_path) as f:
            meta = json.load(f)

    sp           = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab        = meta["vocab_size"]
    token_bytes_np = build_token_bytes(sp)   # (vocab_size,) int32, permanent map

    train_ds = TokenDataset(str(out / "train_bpe.npy"), seq_len)

    with jax.default_device(cpu):
        key = jax.random.PRNGKey(seed)
        lm = BPELM(vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads, key=key)

    avg_bpt = token_bytes_np.sum() / max((token_bytes_np > 0).sum(), 1)   # informational only

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr,
        warmup_steps=warmup_steps, decay_steps=total_steps, end_value=lr * 0.1,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adamw(schedule, weight_decay=0.1))
    opt_state = optimizer.init(eqx.filter(lm, eqx.is_array))

    lm        = jax.device_put(lm,        compute_device)
    opt_state = jax.device_put(opt_state, compute_device)
    # token_bytes lives on compute device; JAX treats it as a dynamic array (fixed shape/values)
    token_bytes_jax = jax.device_put(jnp.array(token_bytes_np), compute_device)

    # --- Logging ---
    run_dir = setup_run_dir(log_dir, f"bpe_{vocab_size}", run_name, no_date)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(args_dict or {}, f, indent=2)

    with RunLogger(run_dir) as logger:
        logger.log(f"run_dir:  {run_dir}")
        logger.log(f"device:   {compute_device}")
        logger.log(f"model:    {lm.param_count():,} params  |  vocab={vocab}  avg_bpt(corpus)={meta['avg_bytes_per_token']:.2f}")
        logger.log(f"bpb:      exact per-batch via token→byte map  (not corpus-average approximation)")

        t0 = time.time()
        with tqdm(total=total_steps, desc=f"bpe-{vocab_size}", unit="step") as pbar:
            for step in range(1, total_steps + 1):
                raw = train_ds.random_batch(batch_size, rng)

                with jax.default_device(compute_device):
                    tokens = jax.device_put(jnp.array(raw), compute_device)
                    lm, opt_state, nats, bpb = train_step(
                        lm, opt_state, optimizer, tokens, token_bytes_jax
                    )

                nats, bpb = float(nats), float(bpb)
                pbar.set_postfix(nats=f"{nats:.4f}", bpb=f"{bpb:.3f}", refresh=False)
                pbar.update(1)
                logger.jlog(dict(step=step, nats=nats, bpb=bpb, elapsed=time.time() - t0))

                if step % eval_every == 0:
                    m = eval_file(lm, val_file, sp, token_bytes_np, seq_len, compute_device)
                    msg = f"  [val {step}]  nats={m['nats']:.4f}  bpb={m['bpb']:.4f}"
                    logger.log(msg)
                    logger.jlog(dict(step=step, val=True, **m))

        logger.log(f"done  {time.time()-t0:.0f}s  run_dir={run_dir}")

    return lm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",        default="../datasets/quran_uthmani.txt")
    parser.add_argument("--out",        default="data/quran")
    parser.add_argument("--val_file",   default="../datasets/suratalfatihah.txt")
    parser.add_argument("--vocab_size", type=int,   default=1024)
    parser.add_argument("--seq_len",    type=int,   default=256)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--d_model",    type=int,   default=128)
    parser.add_argument("--n_layers",   type=int,   default=4)
    parser.add_argument("--n_heads",    type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--steps",      type=int,   default=5000)
    parser.add_argument("--warmup",     type=int,   default=200)
    parser.add_argument("--eval_every", type=int,   default=500)
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--log_dir",    default="logs")
    parser.add_argument("--run_name",   default=None, help="override run folder name")
    parser.add_argument("--no_date",    action="store_true", help="don't append timestamp")
    args = parser.parse_args()

    train(
        src=args.src,
        out_dir=args.out,
        val_file=args.val_file,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        lr=args.lr,
        total_steps=args.steps,
        warmup_steps=args.warmup,
        eval_every=args.eval_every,
        seed=args.seed,
        log_dir=args.log_dir,
        run_name=args.run_name,
        no_date=args.no_date,
        args_dict=vars(args),
    )
