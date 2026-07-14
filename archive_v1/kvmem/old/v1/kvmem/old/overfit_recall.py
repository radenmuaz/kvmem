"""
kvmem/overfit_recall.py — Overfit recall sanity check.

Train on ONE fixed sequence (up_counter) and check if the model can overfit
to recall it perfectly. If it CAN, the architecture works; problem is generalization.
If it CAN'T, the architecture is broken.

Usage:
    python -m kvmem.overfit_recall
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from kvmem.data import DATA_LO, ETX, STX, make_slot_ids, make_mask_stage0
from kvmem.stage0 import (
    KVMemModel,
    build_model,
    count_params,
    init_opt_state,
    adam_update,
    clip_grads,
)
from kvmem.mini_recall import ar_decode, cer, make_test_sequences


def make_recall_token(x_S: list[int], N: int) -> np.ndarray:
    """Build full recall sequence for one x_S."""
    seg_len  = len(x_S)
    slot_ids = make_slot_ids(N)
    seq = x_S + [STX] + slot_ids + [ETX] + x_S   # Y = copy of x_S
    return np.array(seq, dtype=np.int32)


def overfit_test(seg_len: int = 8, N: int = 4, d: int = 64,
                 n_layers: int = 4, n_steps: int = 10_000,
                 lr: float = 1e-3, wd: float = 0.0):
    """
    Overfit one sequence. Goal: perfect recall (loss -> 0, CER = 0.0).
    """
    hp = dict(V=256, d=d, n_layers=n_layers, n_heads=4, d_ff=d*2,
              seg_len=seg_len, N=N)

    key = jax.random.PRNGKey(42)
    key, mkey = jax.random.split(key)
    model     = build_model(hp, mkey)
    opt_state = init_opt_state(model, optimizer='adamw')

    pcount = count_params(model)
    print(f'Model: d={d} n_layers={n_layers} params={pcount["total"]:,}')
    print(f'seg_len={seg_len}  N={N}')

    # Fixed sequence: up_counter
    test_seqs = make_test_sequences(seg_len)
    x_S = test_seqs['up_counter']
    print(f'x_S = {x_S}  ({bytes(x_S).decode("ascii", errors="replace")})')

    tok_np   = make_recall_token(x_S, N)
    L_total  = len(tok_np)
    ETX_pos  = seg_len + 1 + N
    mask_jnp = jnp.array(make_mask_stage0(seg_len, N, seg_len))

    # Single example batch (B=1 — pure overfit)
    tokens_b = jnp.array(tok_np[None], dtype=jnp.int32)   # (1, L_total)

    @jax.jit
    def _step(model, opt_state, step, lr):
        params = eqx.filter(model, eqx.is_array)

        def _loss(m):
            logits = m(tokens_b[0], mask_jnp)  # (L_total, V)
            lp     = jax.nn.log_softmax(logits[:-1], axis=-1)
            tgts   = tokens_b[0, 1:]
            nll    = -lp[jnp.arange(L_total-1), tgts]
            pos       = jnp.arange(L_total - 1)
            Y_end     = ETX_pos + seg_len
            mask_cont = ((pos >= ETX_pos) & (pos < Y_end)).astype(jnp.float32)
            return jnp.sum(nll * mask_cont) / (mask_cont.sum() + 1e-8)

        loss, grads = jax.value_and_grad(_loss)(model)
        grads_arr = eqx.filter(grads, eqx.is_array)
        grads_arr = clip_grads(grads_arr, max_norm=1.0)
        new_params, new_opt = adam_update(
            params, grads_arr, opt_state, lr, wd=wd, step=step)
        deltas    = jax.tree.map(lambda np_, p: np_ - p, new_params, params)
        new_model = eqx.apply_updates(model, deltas)
        return new_model, new_opt, loss

    t0 = time.time()
    for step in range(1, n_steps + 1):
        model, opt_state, loss = _step(model, opt_state, step, lr)

        if step % 500 == 0 or step == 1:
            loss_f = float(loss)
            elapsed = time.time() - t0

            # AR decode from warmup = x_S[0]
            warmup = x_S[:1]
            target = x_S[1:]
            gen    = ar_decode(model, x_S, N, warmup, len(target))
            gen_tail = gen[1:]
            c = cer(gen_tail, target)
            match_pct = 100 * (1 - c)

            status = '★ PERFECT' if c == 0.0 else f'CER={c:.3f}'
            print(f'step={step:5d}  loss={loss_f:.6f}  match={match_pct:.1f}%  {status}  '
                  f'gen={bytes(gen_tail).hex()}  ref={bytes(target).hex()}  {elapsed:.0f}s')

            if c == 0.0:
                print(f'\n✓ Overfitting works! Model can recall at step {step}.')
                return True

    print(f'\n✗ Failed to overfit after {n_steps} steps. Final loss={float(loss):.4f}')
    return False


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--seg-len',  type=int, default=8)
    p.add_argument('--N',        type=int, default=4)
    p.add_argument('--d',        type=int, default=64)
    p.add_argument('--n-layers', type=int, default=4)
    p.add_argument('--steps',    type=int, default=10_000)
    p.add_argument('--lr',       type=float, default=1e-3)
    p.add_argument('--wd',       type=float, default=0.0)
    args = p.parse_args()
    overfit_test(args.seg_len, args.N, args.d, args.n_layers,
                 args.steps, args.lr, args.wd)
