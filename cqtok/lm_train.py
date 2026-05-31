"""
lm_train.py — Latent autoregression with re-encoded grounding (PyTorch).

Training (A-grounded):
  - Encoder sees ground-truth bytes every step (teacher-forced).
  - Loss = rec_loss (decoder CE) + pred_loss (LM next-latent CE).

Inference (A-grounded):
  - Sample z_next from LM → decode bytes → re-encode → append z_grounded.

Usage:
    python data.py --src ../datasets/quran_uthmani.txt --out data/quran --val 1100000
    python lm_train.py --data data/quran --bottleneck bsq
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from data import ByteDataset
from codec import ByteEncoder, ByteDecoder
from model import LatentLM
from bsq import BSQEncoder, BSQLMHead
from fsq import FSQEncoder, FSQLMHead


# ---------------------------------------------------------------------------
# Logging (same pattern as before)
# ---------------------------------------------------------------------------

def setup_run_dir(log_base: str, tag: str, run_name: str | None, no_date: bool) -> str:
    ts   = time.strftime("%Y%m%d_%H%M%S")
    stem = run_name if run_name else tag
    name = stem if no_date else f"{stem}_{ts}"
    path = os.path.join(log_base, name)
    os.makedirs(path, exist_ok=True)
    return path


class RunLogger:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self._log_f  = open(os.path.join(run_dir, "train.log"),   "w", buffering=1)
        self._json_f = open(os.path.join(run_dir, "train.jsonl"), "w", buffering=1)

    def log(self, msg: str):
        tqdm.write(msg)
        self._log_f.write(msg + "\n")

    def jlog(self, rec: dict):
        self._json_f.write(json.dumps(rec) + "\n")

    def close(self):
        self._log_f.close()
        self._json_f.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


# ---------------------------------------------------------------------------
# Full system
# ---------------------------------------------------------------------------

class LatentARSystem(nn.Module):
    def __init__(self, K: int, d_q: int, bottleneck: str, L: int,
                 d_model: int, n_layers: int, n_heads: int):
        super().__init__()
        self.K = K
        self.byte_enc  = ByteEncoder(K=K, d_byte=16, d_enc=128, hidden=(256,))
        self.byte_dec  = ByteDecoder(d_q=d_q, K=K, hidden=(256,))
        self.lm        = LatentLM(d_q=d_q, d_model=d_model, n_layers=n_layers, n_heads=n_heads)

        if bottleneck == "bsq":
            self.quant_enc = BSQEncoder(d_in=128, d_q=d_q)
            self.lm_head   = BSQLMHead(d_model=d_model, d_q=d_q)
        else:
            self.quant_enc = FSQEncoder(d_in=128, d_q=d_q, L=L)
            self.lm_head   = FSQLMHead(d_model=d_model, d_q=d_q, L=L)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(system: LatentARSystem, byte_chunks: torch.Tensor):
    """
    byte_chunks: (B, T, K) int64
    Returns (total, rec_loss, pred_loss).
    Teacher-forced: encoder always sees ground-truth bytes.
    """
    B, T, K = byte_chunks.shape

    # Encode all chunks
    u     = system.byte_enc(byte_chunks.reshape(B * T, K))
    z_hat, codes = system.quant_enc(u)                # (B*T, d_q)
    z_hat  = z_hat.reshape(B, T, -1)
    codes  = codes.reshape(B, T, -1)

    # Reconstruction: decoder sees z_hat, predicts bytes
    logits   = system.byte_dec(z_hat.reshape(B * T, -1))     # (B*T, K, 256)
    rec_loss = ByteDecoder.reconstruction_loss(logits, byte_chunks.reshape(B * T, K))

    # LM: causal over z_hat sequence (teacher-forced)
    h = system.lm(z_hat)                                       # (B, T, d_model)

    # Predict codes[:, 1:] from h[:, :-1]
    lm_logits = system.lm_head(h[:, :-1])                     # shift
    pred_loss = system.lm_head.loss(lm_logits, codes[:, 1:])

    return rec_loss + pred_loss, rec_loss, pred_loss


# ---------------------------------------------------------------------------
# Val eval on a raw text file
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_on_file(system: LatentARSystem, path: str, T: int, device: torch.device) -> dict:
    system.eval()
    K   = system.K
    raw = torch.frombuffer(Path(path).read_bytes(), dtype=torch.uint8).long()
    n_chunks = len(raw) // K
    T_use    = min(T, max(2, n_chunks - 1))
    n_seqs   = max(1, n_chunks // T_use)
    arr      = raw[: n_seqs * T_use * K].reshape(n_seqs, T_use, K).to(device)

    total, rec, pred = compute_loss(system, arr)
    system.train()
    return {
        "bpb":      total.item() / math.log(2),
        "bpb_rec":  rec.item()   / math.log(2),
        "bpb_pred": pred.item()  / math.log(2),
        "nats":     total.item(),
    }


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


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
    compile_model: bool = False,
    log_dir: str = "logs",
    run_name: str | None = None,
    no_date: bool = False,
    args_dict: dict | None = None,
):
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(seed)

    train_ds = ByteDataset(str(Path(data_dir) / "train.npy"), seq_len=T * K)
    with open(Path(data_dir) / "meta.json") as f:
        meta = json.load(f)

    L = {"bsq": 2, "fsq2": 2, "fsq8": 8}[bottleneck]
    bn = "bsq" if bottleneck == "bsq" else "fsq"
    codebook_bits = d_q if bottleneck in ("bsq", "fsq2") else d_q * int(math.log2(L))

    system = LatentARSystem(K=K, d_q=d_q, bottleneck=bn, L=L,
                            d_model=d_model, n_layers=n_layers, n_heads=n_heads).to(device)

    if compile_model:
        system = torch.compile(system)

    optimizer = torch.optim.AdamW(system.parameters(), lr=lr, weight_decay=0.1,
                                  betas=(0.9, 0.95))
    scheduler = make_scheduler(optimizer, warmup_steps, total_steps)

    run_dir = setup_run_dir(log_dir, f"lm_{bottleneck}", run_name, no_date)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(args_dict or {}, f, indent=2)

    with RunLogger(run_dir) as logger:
        logger.log(f"run_dir: {run_dir}")
        logger.log(f"device:  {device}")
        logger.log(f"train:   {meta['train_bytes']:,} bytes  |  val: {val_file}")
        logger.log(f"model:   {system.param_count():,} params  |  {bottleneck} d_q={d_q} codebook=2^{codebook_bits}  compile={compile_model}")

        system.train()
        t0 = time.time()
        with tqdm(total=total_steps, desc=bottleneck, unit="step") as pbar:
            for step in range(1, total_steps + 1):
                raw    = train_ds.random_batch(batch_size, rng)
                chunks = torch.from_numpy(
                    raw[:, : T * K].reshape(batch_size, T, K).astype(np.int64)
                ).to(device)

                optimizer.zero_grad()
                total, rec, pred = compute_loss(system, chunks)
                total.backward()
                nn.utils.clip_grad_norm_(system.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

                loss, rec_f, pred_f = total.item(), rec.item(), pred.item()
                bpb = loss / math.log(2)

                pbar.set_postfix(loss=f"{loss:.4f}", bpb=f"{bpb:.3f}",
                                 rec=f"{rec_f:.4f}", pred=f"{pred_f:.4f}", refresh=False)
                pbar.update(1)
                logger.jlog(dict(step=step, loss=loss, bpb=bpb, rec=rec_f, pred=pred_f,
                                 lr=scheduler.get_last_lr()[0], elapsed=time.time() - t0))

                if step % eval_every == 0:
                    m = eval_on_file(system, val_file, T, device)
                    msg = (f"  [val {step}]  bpb={m['bpb']:.4f}"
                           f"  rec={m['bpb_rec']:.4f}  pred={m['bpb_pred']:.4f}")
                    logger.log(msg)
                    logger.jlog(dict(step=step, val=True, **m))

        logger.log(f"done  {time.time()-t0:.0f}s  run_dir={run_dir}")

    return system


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/quran")
    parser.add_argument("--val_file",   default="../datasets/suratalfatihah.txt")
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
    parser.add_argument("--compile",    action="store_true", help="torch.compile the model")
    parser.add_argument("--log_dir",    default="logs")
    parser.add_argument("--run_name",   default=None)
    parser.add_argument("--no_date",    action="store_true")
    args = parser.parse_args()

    train(
        data_dir=args.data, val_file=args.val_file, bottleneck=args.bottleneck,
        K=args.K, T=args.T, batch_size=args.batch_size, d_q=args.d_q,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        lr=args.lr, total_steps=args.steps, warmup_steps=args.warmup,
        eval_every=args.eval_every, seed=args.seed, compile_model=args.compile,
        log_dir=args.log_dir, run_name=args.run_name, no_date=args.no_date,
        args_dict=vars(args),
    )
