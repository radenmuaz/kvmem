"""
data.py — convert any text or binary file to a flat uint8 .npy dataset.

Split modes (mutually exclusive):
  No boundaries   → train only
  --val N         → train=[0,N)  val=[N,end)
  --val N --test M→ train=[0,N)  val=[N,M)  test=[M,end)

Usage:
    python data.py --src ../datasets/quran_uthmani.txt --out data/quran
    python data.py --src ../datasets/quran_uthmani.txt --out data/quran --val 1100000
    python data.py --src ../datasets/quran_uthmani.txt --out data/quran --val 1100000 --test 1300000
    python data.py --src myfile.bin --out data/myfile --mode binary --val 900000

Writes:
    train.npy            always
    val.npy              if --val given
    test.npy             if --test given
    meta.json            splits, sizes, src path

The .npy files are memory-mappable:
    arr = np.load("data/quran/train.npy", mmap_mode="r")
    batch = arr[offset : offset + seq_len]
"""

import argparse
import json
import os
from typing import Optional

import numpy as np


def prepare(
    src: str,
    out_dir: str,
    val_start: Optional[int] = None,   # byte index where val begins
    test_start: Optional[int] = None,  # byte index where test begins (requires val_start)
    encoding: str = "utf-8",
    mode: str = "text",
) -> dict:
    """
    Load src, split at byte boundaries, save .npy splits.

    val_start / test_start are byte indices into the raw byte array.
    Negative values count from the end (like Python slicing).
    """
    os.makedirs(out_dir, exist_ok=True)

    if mode == "text":
        with open(src, "r", encoding=encoding) as f:
            text = f.read()
        data = np.frombuffer(text.encode(encoding), dtype=np.uint8)
    else:
        with open(src, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)

    total = len(data)

    # Resolve negative indices
    def _resolve(idx: Optional[int]) -> Optional[int]:
        if idx is None:
            return None
        return idx % total  # handles negative: -1 → total-1

    val_start  = _resolve(val_start)
    test_start = _resolve(test_start)

    if test_start is not None and val_start is None:
        raise ValueError("--test requires --val")
    if val_start is not None and test_start is not None and test_start <= val_start:
        raise ValueError(f"test_start ({test_start}) must be > val_start ({val_start})")

    # Build splits
    splits: dict[str, np.ndarray] = {}
    if val_start is None:
        splits["train"] = data
    elif test_start is None:
        splits["train"] = data[:val_start]
        splits["val"]   = data[val_start:]
    else:
        splits["train"] = data[:val_start]
        splits["val"]   = data[val_start:test_start]
        splits["test"]  = data[test_start:]

    meta = {
        "src":         os.path.abspath(src),
        "mode":        mode,
        "encoding":    encoding if mode == "text" else None,
        "total_bytes": int(total),
        "splits":      {k: len(v) for k, v in splits.items()},
        "val_start":   val_start,
        "test_start":  test_start,
    }
    # flat convenience keys used by training scripts
    meta["train_bytes"] = len(splits["train"])
    meta["val_bytes"]   = len(splits.get("val", []))
    meta["test_bytes"]  = len(splits.get("test", []))

    for name, arr in splits.items():
        path = os.path.join(out_dir, f"{name}.npy")
        np.save(path, arr)
        print(f"Wrote {path}  ({len(arr):,} bytes)")

    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")

    return meta


class ByteDataset:
    """Memory-mapped uint8 dataset. Yields contiguous windows of length seq_len."""

    def __init__(self, npy_path: str, seq_len: int):
        self.data = np.load(npy_path, mmap_mode="r")
        self.seq_len = seq_len
        self.n = len(self.data) - seq_len

    def __len__(self) -> int:
        return self.n

    def get_batch(self, offsets: np.ndarray) -> np.ndarray:
        """offsets: (B,) → (B, seq_len+1) uint8"""
        return np.stack([self.data[o : o + self.seq_len + 1] for o in offsets])

    def random_batch(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        offsets = rng.integers(0, self.n, size=batch_size)
        return self.get_batch(offsets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a text/binary file to memory-mappable .npy splits."
    )
    parser.add_argument("--src",      required=True, help="source file")
    parser.add_argument("--out",      required=True, help="output directory")
    parser.add_argument("--val",      type=int, default=None, metavar="N",
                        help="byte index where val split begins")
    parser.add_argument("--test",     type=int, default=None, metavar="M",
                        help="byte index where test split begins (requires --val)")
    parser.add_argument("--mode",     choices=["text", "binary"], default="text")
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args()

    prepare(args.src, args.out, args.val, args.test, args.encoding, args.mode)
