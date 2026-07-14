"""
kvmem/eval_surah.py — Windowed recall eval on Surah Al-Fatihah.

Tests the role-tag model's ability to retrieve arbitrary windows of the
full surah from KV memory. Uses raw bytes (no preprocessing — newlines
and all Arabic UTF-8 bytes pass through as-is).

Usage:
    python -m kvmem.eval_surah --ckpt logs/role_20260602_083819/checkpoints/stage4_end.pt
    python -m kvmem.eval_surah --ckpt logs/role_20260602_083819/checkpoints/stage3_end.pt
    python -m kvmem.eval_surah --ckpt <path> --device mps --n-windows 20
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from kvmem.model import build_model
from kvmem.train_role import ar_decode_role, role_positions
from kvmem.utils import cer


def load_surah(path: str, seg_len: int, pad_byte: int = 0x20) -> list[int]:
    """
    Load suratalfatihah.txt as raw bytes, pad/truncate to seg_len.
    No preprocessing — 0x0A newlines and all Arabic UTF-8 bytes are kept.
    """
    raw = open(path, 'rb').read()
    raw = list(raw[:seg_len])
    if len(raw) < seg_len:
        raw = raw + [pad_byte] * (seg_len - len(raw))
    return raw


def eval_surah(ckpt_path: str, surah_path: str, n_windows: int,
               device_str: str, warmup_len: int | None, out_len: int | None):
    device = torch.device(device_str)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hp   = ckpt['hp']

    # Infer stage geometry from checkpoint
    stage_i = ckpt.get('stage', '?')
    curriculum = hp.get('curriculum', [])

    # Find the stage config for this checkpoint's stage index
    if isinstance(stage_i, int) and stage_i < len(curriculum):
        stage_cfg = curriculum[stage_i]
    else:
        # Fall back to last curriculum stage
        stage_cfg = curriculum[-1] if curriculum else {}

    seg_len    = stage_cfg.get('seg_len', hp.get('seg_len', 256))
    N          = stage_cfg.get('N', seg_len)
    wl         = warmup_len if warmup_len is not None else stage_cfg.get('warmup_len', 32)
    ol         = out_len    if out_len    is not None else stage_cfg.get('out_len', 64)
    slot_style = hp.get('slot_style', 'seq')

    print(f'Checkpoint : {ckpt_path}')
    print(f'Stage      : {stage_i}  seg_len={seg_len}  N={N}  warmup={wl}  out={ol}')
    print(f'Slot style : {slot_style}  device={device_str}')
    print()

    # Build and load model
    # Use max curriculum stage for L_train/L_max (same as training)
    max_stage = max(curriculum, key=lambda s: s['seg_len']) if curriculum else stage_cfg
    from kvmem.data import ROLE_OVERHEAD
    L_max_seq = ROLE_OVERHEAD + max_stage['seg_len'] + max_stage.get('N', max_stage['seg_len']) + \
                max_stage.get('warmup_len', 32) + max_stage.get('out_len', 128)
    hp_model = dict(hp, seg_len=max_stage['seg_len'], N=max_stage.get('N', max_stage['seg_len']),
                    L_train=L_max_seq, L_max=L_max_seq * 4)

    model = build_model(hp_model, device)
    sd = ckpt['model']
    # torch.compile wraps keys with '_orig_mod.' prefix — strip it
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    print(f'Model params: {model.count_params():,}')

    # Load surah, pad to seg_len
    x_S = load_surah(surah_path, seg_len)
    raw_len = min(len(open(surah_path, 'rb').read()), seg_len)
    print(f'Surah      : {surah_path}  ({raw_len} bytes, padded to {seg_len})')
    print()

    # Sample n_windows positions uniformly in [wl, seg_len - ol]
    valid_starts = list(range(wl, seg_len - ol + 1))
    if not valid_starts:
        print(f'ERROR: seg_len={seg_len} too small for warmup={wl} + out={ol}')
        return

    rng = np.random.default_rng(0)
    if n_windows >= len(valid_starts):
        positions = valid_starts
    else:
        step = max(1, len(valid_starts) // n_windows)
        positions = valid_starts[::step][:n_windows]

    all_cer  = []
    all_match = []
    print(f'Testing {len(positions)} window positions:')
    print(f'{"pos":>6}  {"match":>7}  {"CER":>6}  gen[:16]         ref[:16]')
    print('-' * 72)

    for y_start in positions:
        y_end   = min(y_start + ol, seg_len)
        actual_ol = y_end - y_start
        warmup  = x_S[max(0, y_start - wl):y_start]
        if len(warmup) < wl:
            warmup = [x_S[0]] * (wl - len(warmup)) + list(warmup)
        target  = x_S[y_start:y_end]

        gen = ar_decode_role(model, x_S, N, slot_style, warmup, actual_ol, device)
        c   = cer(gen, target)
        all_cer.append(c)
        all_match.append(1.0 - c)

        gen_hex = bytes(gen[:8]).hex()
        ref_hex = bytes(target[:8]).hex()
        ok = '✓' if c == 0.0 else '✗'
        print(f'  {ok} {y_start:4d}   {100*(1-c):6.1f}%  {c:6.3f}  {gen_hex}  {ref_hex}')

    mean_cer   = sum(all_cer) / len(all_cer)
    mean_match = sum(all_match) / len(all_match)
    perfect    = sum(1 for c in all_cer if c == 0.0)
    print()
    print(f'Results: mean_match={mean_match*100:.1f}%  mean_CER={mean_cer:.3f}'
          f'  perfect_windows={perfect}/{len(positions)}')
    print()

    if mean_match >= 0.5:
        print('✓ PASS: ≥50% windowed recall on suratalfatihah')
    else:
        print('✗ FAIL: <50% windowed recall')
        print('  Suggested next steps:')
        print('  - Increase steps to 100k per stage (--steps 100000)')
        print('  - Try --drop-close 0.0 (always include </c>)')
        print('  - Try --no-grok (plain AdamW)')

    return mean_match


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',       required=True, help='Checkpoint .pt file')
    p.add_argument('--surah',      default='datasets/suratalfatihah.txt')
    p.add_argument('--n-windows',  type=int, default=20, help='Number of window positions to test')
    p.add_argument('--warmup-len', type=int, default=None, help='Override warmup length')
    p.add_argument('--out-len',    type=int, default=None, help='Override output length')
    p.add_argument('--device',     default='cpu', choices=['cpu', 'mps', 'cuda'])
    args = p.parse_args()

    eval_surah(
        ckpt_path  = args.ckpt,
        surah_path = args.surah,
        n_windows  = args.n_windows,
        device_str = args.device,
        warmup_len = args.warmup_len,
        out_len    = args.out_len,
    )
