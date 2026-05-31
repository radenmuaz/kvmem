"""
data.py — convert any text or binary file to a flat uint8 .npy dataset.

Usage:
    python data.py --src ../datasets/quran_uthmani.txt --out data/quran
    python data.py --src myfile.bin --out data/myfile --mode binary

Writes:
    <out_dir>/train.npy   uint8 array, shape (N_train,)
    <out_dir>/val.npy     uint8 array, shape (N_val,)
    <out_dir>/meta.json   {"total_bytes", "train_bytes", "val_bytes", "val_frac", "mode", "src"}

The .npy files are memory-mappable:
    arr = np.load("data/quran/train.npy", mmap_mode="r")
    batch = arr[offset : offset + seq_len]
"""

import argparse
import json
import math
import os

import numpy as np


def prepare(
    src: str,
    out_dir: str,
    val_frac: float = 0.1,
    encoding: str = "utf-8",
    mode: str = "text",
) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    if mode == "text":
        with open(src, "r", encoding=encoding) as f:
            text = f.read()
        data = np.frombuffer(text.encode(encoding), dtype=np.uint8)
    else:
        with open(src, "rb") as f:
            raw = f.read()
        data = np.frombuffer(raw, dtype=np.uint8)

    total = len(data)
    n_val = max(1, math.floor(total * val_frac))
    n_train = total - n_val

    # Split from the end to keep training data at the start (preserves text order).
    train_data = data[:n_train]
    val_data = data[n_train:]

    train_path = os.path.join(out_dir, "train.npy")
    val_path = os.path.join(out_dir, "val.npy")
    meta_path = os.path.join(out_dir, "meta.json")

    np.save(train_path, train_data)
    np.save(val_path, val_data)

    meta = {
        "src": os.path.abspath(src),
        "mode": mode,
        "encoding": encoding if mode == "text" else None,
        "total_bytes": int(total),
        "train_bytes": int(n_train),
        "val_bytes": int(n_val),
        "val_frac": val_frac,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {train_path}  ({n_train:,} bytes)")
    print(f"Wrote {val_path}    ({n_val:,} bytes)")
    print(f"Wrote {meta_path}")
    return meta


class ByteDataset:
    """Memory-mapped uint8 dataset. Yields contiguous windows of length seq_len."""

    def __init__(self, npy_path: str, seq_len: int):
        self.data = np.load(npy_path, mmap_mode="r")
        self.seq_len = seq_len
        self.n = len(self.data) - seq_len  # number of valid start positions

    def __len__(self) -> int:
        return self.n

    def get_batch(self, offsets: np.ndarray) -> np.ndarray:
        """
        offsets: int array of shape (B,), each in [0, n).
        Returns: uint8 array of shape (B, seq_len + 1).
        The +1 allows caller to split into inputs[:seq_len] and targets[1:].
        """
        return np.stack([self.data[o : o + self.seq_len + 1] for o in offsets])

    def random_batch(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        offsets = rng.integers(0, self.n, size=batch_size)
        return self.get_batch(offsets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source file path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--mode", choices=["text", "binary"], default="text")
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args()

    prepare(args.src, args.out, args.val_frac, args.encoding, args.mode)
