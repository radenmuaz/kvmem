"""
bpe_train.py — BPE baseline: SentencePiece trained on corpus + causal Transformer (PyTorch).

BPB is exact per-batch via token→byte map (build_token_bytes), not a corpus-average approximation.

Usage:
    python bpe_train.py --src ../datasets/quran_uthmani.txt --out data/quran_uthmani
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from model import CausalTransformer
from lm_train import setup_run_dir, RunLogger, make_scheduler


# ---------------------------------------------------------------------------
# SentencePiece helpers
# ---------------------------------------------------------------------------

def train_bpe(src: str, out_dir: str, vocab_size: int = 1024) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = str(out / f"bpe_{vocab_size}")
    spm.SentencePieceTrainer.train(
        input=src, model_prefix=prefix, vocab_size=vocab_size,
        model_type="bpe", byte_fallback=True, character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3, add_dummy_prefix=False,
        # logspec="FATAL",
    )
    print(f"BPE model: {prefix}.model  (vocab={vocab_size})")
    return Path(prefix + ".model")


def tokenize_corpus(src: str, out_dir: str, model_path: str, val_frac: float = 0.1) -> dict:
    sp     = spm.SentencePieceProcessor(model_file=model_path)
    text   = Path(src).read_text(encoding="utf-8")
    tokens = np.array(sp.encode(text), dtype=np.int32)
    n_val  = max(1, int(len(tokens) * val_frac))
    train_tok, val_tok = tokens[: len(tokens) - n_val], tokens[len(tokens) - n_val :]
    out    = Path(out_dir)
    np.save(out / "train_bpe.npy", train_tok)
    np.save(out / "val_bpe.npy",   val_tok)
    total_bytes = len(text.encode("utf-8"))
    meta = dict(model_path=str(Path(model_path).resolve()), vocab_size=sp.vocab_size(),
                total_tokens=int(len(tokens)), train_tokens=int(len(train_tok)),
                val_tokens=int(len(val_tok)), total_bytes=total_bytes,
                avg_bytes_per_token=total_bytes / max(1, len(tokens)))
    with open(out / "meta_bpe.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Tokenized: {total_bytes:,} bytes → {len(tokens):,} tokens  "
          f"(avg {meta['avg_bytes_per_token']:.2f} bytes/token)")
    return meta


def build_token_bytes(sp: spm.SentencePieceProcessor) -> np.ndarray:
    """(vocab_size,) int32: UTF-8 byte length each token represents in decoded text."""
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
# Dataset
# ---------------------------------------------------------------------------

class TokenDataset:
    def __init__(self, npy_path: str, seq_len: int):
        self.data    = np.load(npy_path, mmap_mode="r")
        self.seq_len = seq_len
        self.n       = len(self.data) - seq_len

    def random_batch(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        offsets = rng.integers(0, self.n, size=batch_size)
        return np.stack([self.data[o : o + self.seq_len + 1] for o in offsets])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BPELM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int):
        super().__init__()
        self.embed       = nn.Embedding(vocab_size, d_model)
        self.transformer = CausalTransformer(d_model, n_layers, n_heads)
        self.head        = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) int64 → logits: (B, T, vocab_size)"""
        return self.head(self.transformer(self.embed(tokens)))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Loss — exact BPB via token→byte map
# ---------------------------------------------------------------------------

def token_loss_and_bpb(
    model: BPELM,
    tokens: torch.Tensor,       # (B, T+1) int64
    token_bytes: torch.Tensor,  # (vocab_size,) int32
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (mean_nats_per_token, exact_bpb).
    BPB = sum(nll) / (sum(token_bytes_of_targets) * ln2)  — unbiased per-batch.
    """
    inputs, targets = tokens[:, :-1], tokens[:, 1:]      # (B, T)
    logits = model(inputs)                                # (B, T, V)
    nll    = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")
    nll    = nll.reshape(targets.shape)                   # (B, T)

    byte_counts = token_bytes[targets]                    # (B, T) int32
    total_bytes = byte_counts.sum().clamp(min=1)
    bpb  = nll.sum() / (total_bytes.float() * math.log(2))
    nats = nll.mean()
    return nats, bpb


# ---------------------------------------------------------------------------
# Val eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_file(model: BPELM, path: str, sp, token_bytes_np: np.ndarray,
              seq_len: int, device: torch.device) -> dict:
    model.eval()
    text    = Path(path).read_text(encoding="utf-8")
    toks    = np.array(sp.encode(text), dtype=np.int64)
    T_use   = min(seq_len, max(2, len(toks) - 1))
    n_seqs  = max(1, len(toks) // T_use)
    arr     = toks[: n_seqs * T_use].reshape(n_seqs, T_use)
    batch   = torch.from_numpy(np.concatenate([arr, arr[:, -1:]], axis=1)).to(device)
    tb     = torch.from_numpy(token_bytes_np).to(device)
    nats, bpb = token_loss_and_bpb(model, batch, tb)
    model.train()
    return {"nats": nats.item(), "bpb": bpb.item()}


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
    compile_model: bool = False,
    log_dir: str = "logs",
    run_name: str | None = None,
    no_date: bool = False,
    args_dict: dict | None = None,
):
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rng    = np.random.default_rng(seed)
    out    = Path(out_dir)

    model_path = out / f"bpe_{vocab_size}.model"
    if not model_path.exists():
        model_path = train_bpe(src, out_dir, vocab_size)

    meta_path = out / "meta_bpe.json"
    if not meta_path.exists():
        meta = tokenize_corpus(src, out_dir, str(model_path))
    else:
        with open(meta_path) as f:
            meta = json.load(f)

    sp             = spm.SentencePieceProcessor(model_file=str(model_path))
    token_bytes_np = build_token_bytes(sp)
    token_bytes    = torch.from_numpy(token_bytes_np).to(device)
    vocab          = meta["vocab_size"]
    train_ds       = TokenDataset(str(out / "train_bpe.npy"), seq_len)

    model = BPELM(vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads).to(device)
    if compile_model:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    scheduler = make_scheduler(optimizer, warmup_steps, total_steps)

    run_dir = setup_run_dir(log_dir, f"bpe_{vocab_size}", run_name, no_date)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(args_dict or {}, f, indent=2)

    with RunLogger(run_dir) as logger:
        logger.log(f"run_dir: {run_dir}")
        logger.log(f"device:  {device}")
        logger.log(f"model:   {model.param_count():,} params  |  vocab={vocab}  "
                   f"avg_bpt={meta['avg_bytes_per_token']:.2f}  compile={compile_model}")
        logger.log(f"bpb:     exact per-batch via token→byte map")

        model.train()
        t0 = time.time()
        with tqdm(total=total_steps, desc=f"bpe-{vocab_size}", unit="step") as pbar:
            for step in range(1, total_steps + 1):
                raw    = train_ds.random_batch(batch_size, rng)
                tokens = torch.from_numpy(raw.astype(np.int64)).to(device)

                optimizer.zero_grad()
                nats, bpb = token_loss_and_bpb(model, tokens, token_bytes)
                nats.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

                nats_f, bpb_f = nats.item(), bpb.item()
                pbar.set_postfix(nats=f"{nats_f:.4f}", bpb=f"{bpb_f:.3f}", refresh=False)
                pbar.update(1)
                logger.jlog(dict(step=step, nats=nats_f, bpb=bpb_f,
                                 lr=scheduler.get_last_lr()[0], elapsed=time.time() - t0))

                if step % eval_every == 0:
                    m   = eval_file(model, val_file, sp, token_bytes_np, seq_len, device)
                    msg = f"  [val {step}]  nats={m['nats']:.4f}  bpb={m['bpb']:.4f}"
                    logger.log(msg)
                    logger.jlog(dict(step=step, val=True, **m))

        logger.log(f"done  {time.time()-t0:.0f}s  run_dir={run_dir}")

    return model


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
    parser.add_argument("--compile",    action="store_true", help="torch.compile the model")
    parser.add_argument("--log_dir",    default="logs")
    parser.add_argument("--run_name",   default=None)
    parser.add_argument("--no_date",    action="store_true")
    args = parser.parse_args()

    train(
        src=args.src, out_dir=args.out, val_file=args.val_file,
        vocab_size=args.vocab_size, seq_len=args.seq_len, batch_size=args.batch_size,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        lr=args.lr, total_steps=args.steps, warmup_steps=args.warmup,
        eval_every=args.eval_every, seed=args.seed, compile_model=args.compile,
        log_dir=args.log_dir, run_name=args.run_name, no_date=args.no_date,
        args_dict=vars(args),
    )
