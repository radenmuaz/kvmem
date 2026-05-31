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
import math
import os

import numpy as np

from kvmem.data import (
    MEM_OPEN, MEM_CLOSE, MEM_OVERHEAD, MEM_OPEN_LEN, MEM_CLOSE_LEN,
    make_slot_ids_tag,
)


def _sample_seg(rng: np.random.Generator, seg_len: int) -> np.ndarray:
    dist_type = int(rng.integers(0, 4))
    if dist_type == 0:
        return rng.integers(0, 256, size=seg_len).astype(np.int32)
    elif dist_type == 1:
        alpha = float(rng.uniform(0.05, 1.0))
        p     = rng.dirichlet(np.ones(256) * alpha)
        return rng.choice(256, size=seg_len, p=p).astype(np.int32)
    elif dist_type == 2:
        width = int(rng.integers(4, 129))
        lo    = int(rng.integers(0, 256 - width + 1))
        return rng.integers(lo, lo + width, size=seg_len).astype(np.int32)
    else:
        p_g = float(rng.uniform(0.01, 0.3))
        return np.clip(rng.geometric(p_g, size=seg_len) - 1, 0, 255).astype(np.int32)


def generate_stage(rng: np.random.Generator, n: int,
                   seg_len: int, N: int, slot_style: str,
                   chunk_len: int) -> np.ndarray:
    """
    Generate n examples for one curriculum stage.

    chunk_len == seg_len → full-sequence recall (warmup = x_S[0], Y = x_S[1:])
    chunk_len < seg_len  → random-window recall
      Format: [x_S | <m> | slots | </m> | warmup(1) | window(chunk_len)]
      warmup = x_S[y_start-1] if y_start>0 else x_S[0]
      window = x_S[y_start : y_start+chunk_len]
      L = seg_len + MEM_OVERHEAD + N + 1 + chunk_len

    Returns: (n, L) int32 array.
    """
    slot_ids = make_slot_ids_tag(N, slot_style)
    M_start  = seg_len + MEM_OPEN_LEN
    Y_start  = seg_len + MEM_OPEN_LEN + N + MEM_CLOSE_LEN

    is_full = (chunk_len >= seg_len)
    if is_full:
        L   = seg_len + MEM_OVERHEAD + N + seg_len
        out = np.empty((n, L), dtype=np.int32)
    else:
        L   = seg_len + MEM_OVERHEAD + N + 1 + chunk_len
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
            warmup  = seg[y_start - 1] if y_start > 0 else seg[0]
            out[i, Y_start]                              = warmup
            out[i, Y_start+1:Y_start+1+(y_end-y_start)] = seg[y_start:y_end]

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seg-len',      type=int, required=True)
    p.add_argument('--N',            type=int, required=True)
    p.add_argument('--slot-style',   type=str, default='seq')
    p.add_argument('--n-examples',   type=int, default=50000)
    p.add_argument('--target-chunk', type=int, required=True,
                   help='Final window size (e.g. 32). Stages will be seg, seg//2, seg//4, target.')
    p.add_argument('--out-dir',      type=str, default='data/curriculum')
    p.add_argument('--seed',         type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    seg_len      = args.seg_len
    N            = args.N
    slot_style   = args.slot_style
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
        data = generate_stage(rng, args.n_examples, seg_len, N, slot_style, chunk_len)
        np.save(path, data)
        print(f' {data.nbytes / 1e6:.1f} MB')

    # Save metadata
    import json
    meta = dict(seg_len=seg_len, N=N, slot_style=slot_style,
                stages=stages, n_examples=args.n_examples, seed=args.seed)
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Done. Metadata written to {args.out_dir}/meta.json')


if __name__ == '__main__':
    main()
