"""
kvmem/eval_extrapolation.py — Extrapolation / curve-fitting test.

Tests whether the model encoded the SOURCE PATTERN (not just values) into KV.

Setup:
  - Source x_S: a deterministic pattern sequence of length seg_len
  - Encode full x_S into KV slots (as normal)
  - Warmup: LAST k tokens of x_S  (anchor at the tail — model knows where it is)
  - Generate: n tokens BEYOND x_S  (never seen during training)
  - Compare generated continuation against ground-truth pattern extension

If the model learned the pattern (not just positional copying), it will continue
the sequence correctly. If it only memorized positions, it will output garbage or
repeat the start of x_S.

This is a zero-shot test on any trained checkpoint — no retraining needed.

Usage:
    python -m kvmem.eval_extrapolation \
        --ckpt logs/mini_recall_20260531_185919/checkpoints/mini_step7000.eqx \
        --seg-len 576 --N 576 --d 64 --n-layers 4 \
        --rope --yarn --slot-style zeros \
        --warmup-tail 4 --gen-len 32
"""

from __future__ import annotations

import argparse
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from kvmem.stage0 import build_model, count_params
from kvmem.data import (
    MEM_OPEN, MEM_CLOSE, MEM_OVERHEAD, MEM_OPEN_LEN, MEM_CLOSE_LEN,
    make_slot_ids_tag, make_mask_tag,
)
from kvmem.mini_recall import make_test_sequences, cer


def pattern_continuation(name: str, x_S: list[int], n: int) -> list[int]:
    """
    Return the ground-truth continuation of a named pattern beyond x_S.
    The continuation starts at position len(x_S) in the infinite pattern.
    """
    seg_len = len(x_S)
    V = 256 - 0x20   # 236 usable values in [DATA_LO, 0xFF]

    if name == 'up_counter':
        return [0x20 + (seg_len + i) % V for i in range(n)]

    elif name == 'down_counter':
        return [0x20 + (V - 1 - (seg_len + i) % V) for i in range(n)]

    elif name == 'odd':
        base_odd = 1 if V % 2 == 0 else 0
        return [0x20 + (base_odd + 2 * (seg_len + i)) % V for i in range(n)]

    elif name == 'even':
        return [0x20 + (2 * (seg_len + i)) % V for i in range(n)]

    elif name == 'linear':
        return [0x20 + (4 * (seg_len + i)) % V for i in range(n)]

    elif name == 'sawtooth':
        period = max(4, min(seg_len // 2, V // 4))
        step   = V // period
        return [0x20 + ((seg_len + i) % period) * step for i in range(n)]

    elif name == 'palindrome':
        # Pattern repeats: up half, then down half
        half = seg_len // 2
        full_period = seg_len
        ext = []
        for i in range(n):
            pos = (seg_len + i) % full_period
            if pos < half:
                ext.append(0x20 + (2 * pos) % V)
            else:
                ext.append(0x20 + (2 * (full_period - 1 - pos)) % V)
        return ext

    elif name == 'geometric':
        # Reconstruct full geometric at position seg_len+i by replaying
        val = x_S[-1]
        result = []
        for _ in range(n):
            nxt = int(val * 1.1)
            if nxt > 255:
                nxt = 0x20
            result.append(nxt)
            val = nxt
        return result

    else:
        return []


def ar_decode_extrap(model, x_S: list[int], N: int, slot_style: str,
                     warmup_tail: int, gen_len: int) -> list[int]:
    """
    Extrapolation decode.
    warmup_tail: how many tokens from the END of x_S to use as warmup.
    Generates gen_len tokens after the warmup (i.e., beyond x_S).
    """
    seg_len  = len(x_S)
    slot_ids = make_slot_ids_tag(N, slot_style)
    mem_blk  = MEM_OPEN + slot_ids + MEM_CLOSE

    # Warmup = last warmup_tail tokens of x_S
    warmup = x_S[-warmup_tail:]

    L_y      = warmup_tail + gen_len
    L_full   = seg_len + MEM_OVERHEAD + N + L_y
    mask_jnp = jnp.array(make_mask_tag(seg_len, N, L_y))

    generated = list(warmup)
    for _ in range(gen_len):
        cur    = x_S + mem_blk + generated
        pad_n  = L_full - len(cur)
        padded = jnp.array(cur + [0] * pad_n, dtype=jnp.int32)
        logits = model(padded, mask_jnp)
        nb     = int(jnp.argmax(logits[len(cur) - 1]))
        generated.append(nb)

    return generated[warmup_tail:]   # return only the generated (not warmup)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        required=True)
    p.add_argument('--seg-len',     type=int, required=True)
    p.add_argument('--N',           type=int, required=True)
    p.add_argument('--d',           type=int, default=64)
    p.add_argument('--n-layers',    type=int, default=4)
    p.add_argument('--rope',        action='store_true')
    p.add_argument('--yarn',        action='store_true')
    p.add_argument('--slot-style',  type=str, default='zeros')
    p.add_argument('--warmup-tail', type=int, default=4,
                   help='Last k tokens of source used as warmup (default 4)')
    p.add_argument('--gen-len',     type=int, default=32,
                   help='Tokens to generate beyond source (default 32)')
    args = p.parse_args()

    seg_len = args.seg_len; N = args.N
    hp = dict(V=256, d=args.d, n_layers=args.n_layers, n_heads=4,
              d_ff=args.d * 2, seg_len=seg_len, N=N,
              rope=args.rope, yarn=args.yarn,
              L_train=seg_len + MEM_OVERHEAD + N + seg_len)

    _cpu = jax.devices('cpu')[0]
    with jax.default_device(_cpu):
        model = build_model(hp, jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(args.ckpt, model)
    pc = count_params(model)['total']
    print(f'Checkpoint: {args.ckpt}')
    print(f'Model: d={args.d} layers={args.n_layers} params={pc:,} '
          f'rope={args.rope} yarn={args.yarn}')
    print(f'seg_len={seg_len}  N={N}  slot_style={args.slot_style}')
    print(f'warmup_tail={args.warmup_tail}  gen_len={args.gen_len}')
    print()
    print('=== Extrapolation Test (zero-shot: source pattern → continuation) ===')
    print()

    test_seqs = make_test_sequences(seg_len)
    results = {}

    for name, x_S in test_seqs.items():
        gt_cont = pattern_continuation(name, x_S, args.gen_len)
        if not gt_cont:
            print(f'  [skip] {name}: no continuation defined')
            continue

        gen = ar_decode_extrap(model, x_S, N, args.slot_style,
                               args.warmup_tail, args.gen_len)
        c   = cer(gen, gt_cont)
        match = 100 * (1 - c)
        ok  = '✓' if c == 0.0 else '✗'

        # Show warmup context for interpretability
        warmup_hex = bytes(x_S[-args.warmup_tail:]).hex()
        gt_hex     = bytes(gt_cont).hex()
        gen_hex    = bytes(gen).hex()

        print(f'  {ok} {name:<18} match={match:5.1f}%  CER={c:.3f}')
        print(f'     warmup (tail {args.warmup_tail}): {warmup_hex}')
        print(f'     gen:  {gen_hex}')
        print(f'     ref:  {gt_hex}')
        print()
        results[name] = c

    mean_cer   = float(np.mean(list(results.values())))
    mean_match = 100 * (1 - mean_cer)
    print(f'=== mean CER={mean_cer:.3f}  mean match={mean_match:.1f}% ===')
    print()
    if mean_cer == 0.0:
        print('★ PERFECT EXTRAPOLATION: model encoded patterns, not just values.')
    elif mean_match > 50:
        print('~ Partial extrapolation: some patterns generalized.')
    else:
        print('✗ No extrapolation: model did not encode the pattern beyond recall.')


if __name__ == '__main__':
    main()
