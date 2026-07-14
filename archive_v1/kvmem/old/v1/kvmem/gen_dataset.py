"""
kvmem/gen_dataset.py — Pre-generate training dataset to disk.

Separates data generation from training loop. Supports curriculum stages:
  stage 0: full-sequence recall (chunk_len = seg_len)  — proven to work
  stage 1: half-window  (chunk_len = seg_len // 2)
  stage 2: quarter-window (chunk_len = seg_len // 4)
  stage 3: target-window (chunk_len = target)

Each stage is a separate .npy file. Training loads and shuffles from these.

Usage:
    python -m kvmem.gen_dataset \
        --seg-len 128 --N 128 --slot-style seq \
        --n-examples 50000 --out-dir data/curriculum_128 \
        --target-chunk 32
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from kvmem.data import (
    MEM_OPEN, MEM_CLOSE, MEM_OVERHEAD, MEM_OPEN_LEN, MEM_CLOSE_LEN,
    make_slot_ids_tag, _sample_seg,
)


def generate_variable(rng: np.random.Generator, n: int,
                      seg_len: int, N: int, slot_style: str,
                      warmup_len: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """
    Variable-window LM-style dataset.

    Each example: random y_start ~ Uniform[0, seg_len).
    Output Y = x_S[y_start:] (everything from y_start to end).
    Warmup = last warmup_len bytes before y_start (padded at start of seq).

    Format: [x_S | <m> | slots | </m> | warmup(wl) | x_S[y_start:] | padding]
    L_max  = seg_len + MEM_OVERHEAD + N + warmup_len + seg_len  (y_start=0 case)
    Supervision: positions Y_start+warmup_len .. Y_start+warmup_len+(seg_len-y_start)

    Returns:
        tokens:  (n, L_max) int32  — padded with 0
        y_starts:(n,)       int32  — for computing per-example supervision mask
    """
    slot_ids = make_slot_ids_tag(N, slot_style)
    M_start  = seg_len + MEM_OPEN_LEN
    Y_start  = seg_len + MEM_OPEN_LEN + N + MEM_CLOSE_LEN
    L_max    = seg_len + MEM_OVERHEAD + N + warmup_len + seg_len

    tokens   = np.zeros((n, L_max), dtype=np.int32)
    y_starts = np.empty(n, dtype=np.int32)

    for i in range(n):
        seg     = _sample_seg(rng, seg_len)
        y_start = int(rng.integers(0, seg_len))

        w_start = max(0, y_start - warmup_len)
        warmup  = seg[w_start:y_start]
        if len(warmup) < warmup_len:
            warmup = np.concatenate([np.full(warmup_len - len(warmup),
                                             seg[0], dtype=np.int32), warmup])
        out_len = seg_len - y_start   # variable!

        tokens[i, :seg_len]               = seg
        tokens[i, seg_len:M_start]        = MEM_OPEN
        tokens[i, M_start:M_start+N]      = slot_ids
        tokens[i, M_start+N:Y_start]      = MEM_CLOSE
        tokens[i, Y_start:Y_start+warmup_len]              = warmup
        tokens[i, Y_start+warmup_len:Y_start+warmup_len+out_len] = seg[y_start:]
        y_starts[i] = y_start

    return tokens, y_starts


def generate_stage(rng: np.random.Generator, n: int,
                   seg_len: int, N: int, slot_style: str,
                   chunk_len: int, warmup_len: int = 4) -> np.ndarray:
    """
    Generate n examples for one curriculum stage (fixed chunk_len).
    """
    slot_ids = make_slot_ids_tag(N, slot_style)
    M_start  = seg_len + MEM_OPEN_LEN
    Y_start  = seg_len + MEM_OPEN_LEN + N + MEM_CLOSE_LEN

    is_full = (chunk_len >= seg_len)
    if is_full:
        L   = seg_len + MEM_OVERHEAD + N + seg_len
        out = np.empty((n, L), dtype=np.int32)
    else:
        L   = seg_len + MEM_OVERHEAD + N + warmup_len + chunk_len
        out = np.zeros((n, L), dtype=np.int32)

    n_windows = max(1, seg_len - chunk_len)

    for i in range(n):
        seg = _sample_seg(rng, seg_len)
        out[i, :seg_len]          = seg
        out[i, seg_len:M_start]   = MEM_OPEN
        out[i, M_start:M_start+N] = slot_ids
        out[i, M_start+N:Y_start] = MEM_CLOSE

        if is_full:
            out[i, Y_start:] = seg
        else:
            y_start = int(rng.integers(0, n_windows + 1))
            y_end   = min(y_start + chunk_len, seg_len)
            # warmup_len bytes before the window (pad with x_S[0] at start)
            w_start = max(0, y_start - warmup_len)
            warmup  = seg[w_start:y_start]
            if len(warmup) < warmup_len:
                warmup = np.concatenate([np.full(warmup_len - len(warmup),
                                                 seg[0], dtype=np.int32), warmup])
            out[i, Y_start:Y_start+warmup_len]                       = warmup
            out[i, Y_start+warmup_len:Y_start+warmup_len+(y_end-y_start)] = seg[y_start:y_end]

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seg-len',      type=int, required=True)
    p.add_argument('--N',            type=int, required=True)
    p.add_argument('--slot-style',   type=str, default='seq')
    p.add_argument('--n-examples',   type=int, default=50000)
    p.add_argument('--target-chunk', type=int, required=True,
                   help='Final window size (e.g. 32). Stages will be seg, seg//2, seg//4, target.')
    p.add_argument('--warmup-len',   type=int, default=4,
                   help='Warmup context bytes before each window (default 4)')
    p.add_argument('--variable',     action='store_true',
                   help='Generate variable-window LM-style dataset (no curriculum stages)')
    p.add_argument('--out-dir',      type=str, default='data/curriculum')
    p.add_argument('--seed',         type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    seg_len      = args.seg_len
    N            = args.N
    slot_style   = args.slot_style

    if args.variable:
        # Fixed multiples: generate one file per window size, mix during training.
        # window_sizes: powers-of-2 from min_chunk up to seg_len (full recall).
        min_chunk = args.target_chunk
        sizes = []
        s = min_chunk
        while s <= seg_len:
            sizes.append(s)
            s *= 2
        if sizes[-1] != seg_len:
            sizes.append(seg_len)  # always include full

        print(f'Generating multi-size dataset: {args.out_dir}')
        print(f'seg_len={seg_len}  N={N}  slot_style={slot_style}  warmup_len={args.warmup_len}')
        print(f'window sizes: {sizes}  n_examples each: {args.n_examples}')

        for ws in sizes:
            is_full = (ws >= seg_len)
            fname = f'win{ws}.npy'
            path  = os.path.join(args.out_dir, fname)
            data  = generate_stage(rng, args.n_examples, seg_len, N,
                                   slot_style, ws if not is_full else seg_len,
                                   args.warmup_len)
            np.save(path, data)
            print(f'  {fname}: shape={data.shape}  {data.nbytes/1e6:.1f} MB')

        meta = dict(seg_len=seg_len, N=N, slot_style=slot_style,
                    warmup_len=args.warmup_len, n_examples=args.n_examples,
                    mode='multi_size', sizes=sizes, seed=args.seed)
        with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'Done. meta: {args.out_dir}/meta.json')
        return

    target_chunk = args.target_chunk
    # Curriculum stages: full → halving → target
    stages = [seg_len]
    c = seg_len // 2
    while c > target_chunk:
        stages.append(c)
        c = c // 2
    stages.append(target_chunk)

    print(f'Generating curriculum dataset: {args.out_dir}')
    print(f'seg_len={seg_len}  N={N}  slot_style={slot_style}')
    print(f'stages (chunk_len): {stages}')
    print(f'n_examples per stage: {args.n_examples}')

    for chunk_len in stages:
        name = f'stage_chunk{chunk_len}.npy'
        path = os.path.join(args.out_dir, name)
        if os.path.exists(path):
            print(f'  [skip] {name} already exists')
            continue
        L    = seg_len + MEM_OVERHEAD + N + (seg_len if chunk_len >= seg_len else 1 + chunk_len)
        print(f'  generating {name}  chunk_len={chunk_len}  L={L}  shape=({args.n_examples},{L}) ...', end='', flush=True)
        data = generate_stage(rng, args.n_examples, seg_len, N, slot_style, chunk_len, args.warmup_len)
        np.save(path, data)
        print(f' {data.nbytes / 1e6:.1f} MB')

    # Save metadata
    meta = dict(seg_len=seg_len, N=N, slot_style=slot_style,
                stages=stages, n_examples=args.n_examples,
                warmup_len=args.warmup_len, seed=args.seed)
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Done. Metadata written to {args.out_dir}/meta.json')


if __name__ == '__main__':
    main()
