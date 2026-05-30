"""
kvmem/eval_recall.py — Post-training evaluation.

1. Extrapolation test: load a checkpoint trained at seg_len=S,
   test recall on sequences LONGER than S (seg_len = 2S, 4S, ...).
   Uses greedy AR decode with extended mask.

2. Real-text test: split suratalfatihah.txt into seg_len chunks,
   test recall of each chunk independently.

Usage:
    # Extrapolation test on a checkpoint:
    python -m kvmem.eval_recall --ckpt logs/mini_recall_20260530_121836/checkpoints/mini_step8000 \
                                 --seg-len 8 --N 4 --test extrap

    # Surah test (split text into seg-len chunks):
    python -m kvmem.eval_recall --ckpt logs/mini_recall_20260530_121836/checkpoints/mini_step8000 \
                                 --seg-len 8 --N 4 --test surah \
                                 --text datasets/suratalfatihah.txt
"""

from __future__ import annotations

import argparse
import os
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from kvmem.data import DATA_LO, ETX, STX, make_slot_ids, make_mask_stage0
from kvmem.stage0 import build_model, count_params
from kvmem.mini_recall import ar_decode, cer, make_test_sequences


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str, hp: dict):
    """Load model from equinox checkpoint."""
    model = build_model(hp, jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(ckpt_path, model)
    return model


# ---------------------------------------------------------------------------
# Extrapolation test
# ---------------------------------------------------------------------------

def test_extrapolation(model, train_seg_len: int, N: int,
                       test_lengths: list[int] | None = None):
    """
    Test recall on sequences LONGER than training seg_len.
    The model was trained at seg_len=train_seg_len; we test at larger lengths.
    The AR decode uses a mask extended to the test length.
    """
    if test_lengths is None:
        test_lengths = [train_seg_len * 2, train_seg_len * 4, train_seg_len * 8]

    print(f"\n=== Extrapolation Test (trained seg_len={train_seg_len}, N={N}) ===")

    for test_len in test_lengths:
        seqs = make_test_sequences(test_len)
        print(f"\n--- seg_len={test_len} ({test_len/train_seg_len:.0f}x train length) ---")

        results = {}
        for name, x_S in seqs.items():
            warmup = x_S[:1]
            target = x_S[1:]
            try:
                gen = ar_decode(model, x_S, N, warmup, len(target))
                gen_tail = gen[1:]
                c = cer(gen_tail, target)
                match_pct = 100 * (1 - c)
                ok = '✓' if c == 0.0 else '✗'
                print(f"  {ok} {name:<18} match={match_pct:5.1f}%  CER={c:.3f}"
                      f"  gen={bytes(gen_tail).hex()}"
                      f"  ref={bytes(target).hex()}")
                results[name] = c
            except Exception as e:
                print(f"  ✗ {name:<18} ERROR: {e}")
                results[name] = 1.0

        mean_cer = sum(results.values()) / len(results)
        mean_match = 100 * (1 - mean_cer)
        print(f"  → mean CER={mean_cer:.3f}  mean match={mean_match:.1f}%")


# ---------------------------------------------------------------------------
# Real-text (surah) test
# ---------------------------------------------------------------------------

def test_surah(model, seg_len: int, N: int, text_path: str):
    """
    Load a text file as raw bytes, split into seg_len chunks,
    test recall of each chunk with greedy AR decode.
    Only chunks where all bytes are in [DATA_LO, 256) are tested
    (protocol bytes 0x00..0x1F would conflict with slot IDs / STX/ETX).
    """
    with open(text_path, 'rb') as f:
        raw = list(f.read())

    print(f"\n=== Real-text Recall Test ===")
    print(f"  File: {text_path}  ({len(raw)} bytes)")
    print(f"  seg_len={seg_len}  N={N}")

    # Split into chunks
    chunks = []
    for i in range(0, len(raw) - seg_len + 1, seg_len):
        chunk = raw[i:i + seg_len]
        if len(chunk) == seg_len:
            chunks.append((i, chunk))

    print(f"  Chunks: {len(chunks)} × {seg_len} bytes")

    perfect = 0
    total_cer = 0.0
    skipped = 0

    for idx, (offset, chunk) in enumerate(chunks):
        # Check for protocol bytes — skip if any byte < DATA_LO
        if any(b < DATA_LO for b in chunk):
            skipped += 1
            continue

        x_S = list(chunk)
        warmup = x_S[:1]
        target = x_S[1:]

        gen = ar_decode(model, x_S, N, warmup, len(target))
        gen_tail = gen[1:]
        c = cer(gen_tail, target)
        match_pct = 100 * (1 - c)
        ok = '✓' if c == 0.0 else '✗'

        chunk_str = bytes(chunk).decode('utf-8', errors='replace')
        print(f"  {ok} chunk[{offset:3d}:{offset+seg_len:3d}]  match={match_pct:5.1f}%  "
              f"CER={c:.3f}  \"{chunk_str}\"")

        total_cer += c
        if c == 0.0:
            perfect += 1

    tested = len(chunks) - skipped
    if tested > 0:
        mean_cer = total_cer / tested
        mean_match = 100 * (1 - mean_cer)
        print(f"\n  → {perfect}/{tested} chunks perfect  mean CER={mean_cer:.3f}  "
              f"mean match={mean_match:.1f}%  (skipped {skipped} chunks with protocol bytes)")
    else:
        print(f"  → All {len(chunks)} chunks skipped (contain protocol bytes < 0x20)")
        print(f"  NOTE: UTF-8 Arabic text uses bytes in [0x80, 0xFF] range —")
        print(f"        these are all >= DATA_LO=0x20, so they should be testable.")
        print(f"        Re-checking skip logic...")

        # Re-run without skip filter to debug
        for idx, (offset, chunk) in enumerate(chunks[:5]):
            x_S = list(chunk)
            print(f"  chunk[{offset}]: {bytes(chunk).hex()} — min_byte={min(chunk):#04x}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',     required=True, help='Path to equinox checkpoint')
    p.add_argument('--seg-len',  type=int, required=True)
    p.add_argument('--N',        type=int, required=True)
    p.add_argument('--d',        type=int, default=64)
    p.add_argument('--n-layers', type=int, default=4)
    p.add_argument('--d-ff',     type=int, default=None, help='FFN dim (default: d*2)')
    p.add_argument('--test',     choices=['extrap', 'surah', 'both'], default='both')
    p.add_argument('--text',     default='datasets/suratalfatihah.txt')
    p.add_argument('--extrap-lengths', type=int, nargs='+', default=None,
                   help='Test lengths for extrapolation (default: 2x, 4x, 8x train)')
    args = p.parse_args()

    d_ff = args.d_ff if args.d_ff is not None else args.d * 2
    hp = dict(V=256, d=args.d, n_layers=args.n_layers, n_heads=4,
              d_ff=d_ff, seg_len=args.seg_len, N=args.N)

    print(f"Loading checkpoint: {args.ckpt}")
    model = load_checkpoint(args.ckpt, hp)
    pcount = count_params(model)
    print(f"Model: d={args.d}  n_layers={args.n_layers}  params={pcount['total']:,}")

    if args.test in ('extrap', 'both'):
        test_extrapolation(model, args.seg_len, args.N, args.extrap_lengths)

    if args.test in ('surah', 'both'):
        if not os.path.exists(args.text):
            print(f"Text file not found: {args.text}")
        else:
            test_surah(model, args.seg_len, args.N, args.text)


if __name__ == '__main__':
    main()
