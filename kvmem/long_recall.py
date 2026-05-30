"""
kvmem/long_recall.py — Large-scale recall training.

Train on random byte sequences up to length max_seg_len (default 1024).
N = seg_len (one slot per byte) — no compression bottleneck.
Test on deterministic sequences (up_counter, down_counter, etc.) at
length 512+ — completely held out from training distribution.

Variable-length training: each batch samples a random seg_len from
[min_seg_len, max_seg_len]. The model sees many lengths, forcing it to
learn a general algorithm not tied to one length.

Usage:
    python -m kvmem.long_recall
    python -m kvmem.long_recall --max-seg 256 --test-seg 128 --steps 100000
    python -m kvmem.long_recall --max-seg 1024 --test-seg 512 --d 128 --n-layers 6
"""

from __future__ import annotations

import argparse
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
    save_checkpoint,
    init_opt_state,
    adam_update,
    clip_grads,
)
from kvmem.mini_recall import make_test_sequences, cer


# ---------------------------------------------------------------------------
# Variable-length synthetic batch
# ---------------------------------------------------------------------------

def make_batch(rng: np.random.Generator, B: int,
               seg_len: int) -> np.ndarray:
    """
    Build one batch with a fixed seg_len (N = seg_len).
    Training uses purely random distributions — no structured patterns.

    Sequence format: [x_S (seg_len) | STX | slots (N=seg_len) | ETX | Y=x_S (seg_len)]
    Total length L = 3*seg_len + 2.
    """
    N        = seg_len
    V_full   = 256 - DATA_LO
    L        = seg_len + 2 + N + seg_len   # = 3*seg_len + 2
    out      = np.empty((B, L), dtype=np.int32)
    slot_ids = make_slot_ids(N)

    for i in range(B):
        dist_type = int(rng.integers(0, 5))
        if dist_type == 0:
            # Uniform over full range
            seg = rng.integers(DATA_LO, 256, size=seg_len).astype(np.int32)
        elif dist_type == 1:
            # Dirichlet-skewed (simulates vocabulary frequency)
            alpha = float(rng.uniform(0.05, 1.0))
            p     = rng.dirichlet(np.ones(V_full) * alpha)
            seg   = (rng.choice(V_full, size=seg_len, p=p) + DATA_LO).astype(np.int32)
        elif dist_type == 2:
            # Uniform over random contiguous sub-range
            width = int(rng.integers(4, min(128, V_full + 1)))
            lo    = int(rng.integers(0, V_full - width + 1)) + DATA_LO
            seg   = rng.integers(lo, lo + width, size=seg_len).astype(np.int32)
        elif dist_type == 3:
            # Geometric (exponentially decaying)
            p_g = float(rng.uniform(0.01, 0.3))
            raw = rng.geometric(p_g, size=seg_len) - 1
            seg = (np.clip(raw, 0, V_full - 1) + DATA_LO).astype(np.int32)
        else:
            # Two-cluster mixture (bimodal byte frequencies)
            c1 = int(rng.integers(DATA_LO, 200))
            c2 = int(rng.integers(c1 + 20, min(c1 + 80, 256)))
            w  = float(rng.uniform(0.2, 0.8))
            seg = np.where(
                rng.uniform(size=seg_len) < w,
                np.clip(rng.integers(c1, c1 + 16, size=seg_len), DATA_LO, 255),
                np.clip(rng.integers(c2, c2 + 16, size=seg_len), DATA_LO, 255),
            ).astype(np.int32)

        out[i, :seg_len]               = seg
        out[i, seg_len]                = STX
        out[i, seg_len+1:seg_len+1+N]  = slot_ids
        out[i, seg_len+1+N]            = ETX
        out[i, seg_len+2+N:]           = seg   # Y = exact copy

    return out


# ---------------------------------------------------------------------------
# Greedy AR decode for one sequence
# ---------------------------------------------------------------------------

def ar_decode_long(model: KVMemModel, x_S: list[int],
                   warmup: list[int], max_new: int) -> list[int]:
    """
    Greedy AR decode. N = len(x_S) (one slot per byte).
    Returns full generated sequence including warmup.
    """
    N        = len(x_S)
    seg_len  = N
    mem_blk  = [STX] + make_slot_ids(N) + [ETX]
    L_full   = seg_len + 2 + N + len(warmup) + max_new
    mask_jnp = jnp.array(make_mask_stage0(seg_len, N, len(warmup) + max_new))

    generated = list(warmup)
    for _ in range(max_new):
        cur    = x_S + mem_blk + generated
        pad_n  = L_full - len(cur)
        padded = jnp.array(cur + [0] * pad_n, dtype=jnp.int32)
        logits = model(padded, mask_jnp)
        nb     = int(jnp.argmax(logits[len(cur) - 1]))
        generated.append(nb)

    return generated


# ---------------------------------------------------------------------------
# Eval: test on deterministic sequences of a given length
# ---------------------------------------------------------------------------

def eval_test_seqs(model: KVMemModel, seg_len: int, log_fn) -> float:
    """Evaluate on all held-out deterministic test sequences. Returns mean CER."""
    test_seqs = make_test_sequences(seg_len)
    cers = []
    for name, x_S in test_seqs.items():
        warmup   = x_S[:1]
        target   = x_S[1:]
        gen      = ar_decode_long(model, x_S, warmup, len(target))
        gen_tail = gen[1:]
        c        = cer(gen_tail, target)
        match    = 100 * (1 - c)
        ok       = '✓' if c == 0.0 else '✗'
        log_fn(f'  {ok} {name:<18} match={match:5.1f}%  CER={c:.3f}'
               f'  gen={bytes(gen_tail).hex()[:32]}{"..." if len(gen_tail)>16 else ""}'
               f'  ref={bytes(target).hex()[:32]}{"..." if len(target)>16 else ""}')
        cers.append(c)
    mean_cer = float(np.mean(cers))
    log_fn(f'  → mean CER={mean_cer:.3f}  mean match={100*(1-mean_cer):.1f}%')
    return mean_cer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_long(hp: dict, log_base: str = 'logs'):
    max_seg  = hp['max_seg']
    min_seg  = hp.get('min_seg', 8)
    test_seg = hp['test_seg']
    d        = hp['d']
    n_layers = hp['n_layers']
    B        = hp['B']
    lr_max   = hp['lr_max']
    wd       = hp['wd']
    n_steps  = hp['n_steps']
    eval_every = hp['eval_every']
    seed     = hp.get('seed', 42)

    # Build run dir
    from datetime import datetime
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(log_base, f'long_recall_{ts}')
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    log_path  = os.path.join(run_dir, 'train.log')
    jlog_path = os.path.join(run_dir, 'train.jsonl')
    log_f  = open(log_path,  'w', buffering=1)
    jlog_f = open(jlog_path, 'w', buffering=1)

    def _log(msg):
        print(msg)
        log_f.write(msg + '\n')

    def _jlog(d):
        jlog_f.write(json.dumps(d) + '\n')

    # Save hparams
    with open(os.path.join(run_dir, 'hparams.json'), 'w') as f:
        json.dump(hp, f, indent=2)

    # Build model — N_max = max_seg (enough slots for longest sequence)
    # The model is built with the max possible N; shorter sequences use fewer slots.
    model_hp = dict(
        V=256, d=d, n_layers=n_layers, n_heads=hp.get('n_heads', 8),
        d_ff=hp.get('d_ff', d * 4),
        seg_len=max_seg,   # architecture uses max length; masks handle shorter
        N=max_seg,
    )
    key      = jax.random.PRNGKey(seed)
    key, mk  = jax.random.split(key)
    model    = build_model(model_hp, mk)
    opt_state = init_opt_state(model, optimizer='adamw')

    pcount = count_params(model)
    _log(f'\n=== Long Recall Training | run_dir={run_dir} ===')
    _log(f'  Model: d={d}  n_layers={n_layers}  n_heads={model_hp["n_heads"]}'
         f'  d_ff={model_hp["d_ff"]}  params={pcount["total"]:,}')
    _log(f'  max_seg={max_seg}  min_seg={min_seg}  test_seg={test_seg}  N=seg_len (dynamic)')
    _log(f'  Steps={n_steps}  Batch={B}  lr={lr_max}  wd={wd}')
    _log(f'  Train: RANDOM bytes — 5 distributions, seg_len sampled Unif[{min_seg},{max_seg}]')
    _log(f'  Test (held out): deterministic seqs at seg_len={test_seg}')

    # Warmup schedule
    warmup_steps = hp.get('warmup_steps', 500)

    rng = np.random.default_rng(seed)
    t0  = time.time()

    # Pre-build JIT'd step functions per seg_len bucket to avoid retracing.
    # We bucket seg_lens to powers of 2 to limit number of JIT compilations.
    _step_cache = {}

    def _get_step_fn(seg_len: int):
        if seg_len in _step_cache:
            return _step_cache[seg_len]

        N     = seg_len
        L     = 3 * seg_len + 2
        mask_np = make_mask_stage0(seg_len, N, seg_len)

        @jax.jit
        def _step(model, opt_state, tokens_b, lr):
            params = eqx.filter(model, eqx.is_array)
            mask_j = jnp.array(mask_np)

            def _loss(m):
                # vmap over batch
                def _single(tok):
                    logits = m(tok, mask_j)           # (L, V)
                    lp     = jax.nn.log_softmax(logits[:-1], axis=-1)
                    tgts   = tok[1:]
                    nll    = -lp[jnp.arange(L - 1), tgts]
                    # Only supervise Y positions: ETX_pos..ETX_pos+seg_len
                    ETX_pos = seg_len + 1 + N
                    pos     = jnp.arange(L - 1)
                    mask_y  = ((pos >= ETX_pos) & (pos < ETX_pos + seg_len)).astype(jnp.float32)
                    return jnp.sum(nll * mask_y) / (mask_y.sum() + 1e-8)
                return jnp.mean(jax.vmap(_single)(tokens_b))

            loss, grads = jax.value_and_grad(_loss)(model)
            grads_arr   = eqx.filter(grads, eqx.is_array)
            grads_arr   = clip_grads(grads_arr, max_norm=1.0)
            new_params, new_opt = adam_update(params, grads_arr, opt_state, lr,
                                              wd=wd, step=1)
            deltas    = jax.tree.map(lambda np_, p: np_ - p, new_params, params)
            new_model = eqx.apply_updates(model, deltas)
            return new_model, new_opt, loss

        _step_cache[seg_len] = _step
        return _step

    # Bucket seg_lens to powers of 2 for JIT efficiency
    def _bucket(s):
        b = 1
        while b < s:
            b *= 2
        return min(b, max_seg)

    for step in range(1, n_steps + 1):
        # LR schedule: linear warmup then cosine decay
        if step <= warmup_steps:
            lr = lr_max * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (n_steps - warmup_steps)
            lr = lr_max * (0.5 * (1 + math.cos(math.pi * progress)))
        lr = max(lr, lr_max * 1e-3)

        # Sample a seg_len for this batch
        raw_seg = int(rng.integers(min_seg, max_seg + 1))
        seg_len = _bucket(raw_seg)
        seg_len = max(min_seg, min(seg_len, max_seg))

        # Build batch
        tokens_np = make_batch(rng, B, seg_len)
        tokens_b  = jnp.array(tokens_np)

        # Train step
        step_fn = _get_step_fn(seg_len)
        model, opt_state, loss = step_fn(model, opt_state, tokens_b, lr)

        if step % eval_every == 0 or step == 1:
            loss_f   = float(loss)
            elapsed  = time.time() - t0
            _log(f'\n--- step={step}/{n_steps}  loss={loss_f:.4f}  lr={lr:.2e}'
                 f'  seg={seg_len}  {elapsed:.0f}s ---')
            _jlog(dict(step=step, loss=loss_f, lr=lr, seg_len=seg_len))

            mean_cer = eval_test_seqs(model, test_seg, _log)
            _jlog(dict(step=step, eval_seg=test_seg, mean_cer=mean_cer))

            if mean_cer == 0.0:
                _log(f'\n★ PERFECT RECALL at step {step}! All test sequences 100%.')
                ckpt_path = os.path.join(ckpt_dir, f'long_step{step}')
                save_checkpoint(ckpt_path, model, step, model_hp)
                _log(f'  [ckpt] {ckpt_path}')
                break

        if step % (eval_every * 10) == 0:
            ckpt_path = os.path.join(ckpt_dir, f'long_step{step}')
            save_checkpoint(ckpt_path, model, step, model_hp)
            _log(f'  [ckpt] {ckpt_path}')

    _log(f'\nDone. Total: {time.time()-t0:.0f}s')
    log_f.close()
    jlog_f.close()
    return model, run_dir


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

LONG_HPARAMS = dict(
    max_seg     = 1024,
    min_seg     = 8,
    test_seg    = 512,
    d           = 128,
    n_layers    = 6,
    n_heads     = 8,
    d_ff        = 512,
    B           = 16,         # small batch — sequences are long
    lr_max      = 3e-4,
    wd          = 0.01,
    warmup_steps = 500,
    n_steps     = 200_000,
    eval_every  = 5_000,
    seed        = 42,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--max-seg',    type=int,   default=LONG_HPARAMS['max_seg'])
    p.add_argument('--min-seg',    type=int,   default=LONG_HPARAMS['min_seg'])
    p.add_argument('--test-seg',   type=int,   default=LONG_HPARAMS['test_seg'])
    p.add_argument('--d',          type=int,   default=LONG_HPARAMS['d'])
    p.add_argument('--n-layers',   type=int,   default=LONG_HPARAMS['n_layers'])
    p.add_argument('--n-heads',    type=int,   default=LONG_HPARAMS['n_heads'])
    p.add_argument('--d-ff',       type=int,   default=LONG_HPARAMS['d_ff'])
    p.add_argument('--B',          type=int,   default=LONG_HPARAMS['B'])
    p.add_argument('--lr',         type=float, default=LONG_HPARAMS['lr_max'])
    p.add_argument('--wd',         type=float, default=LONG_HPARAMS['wd'])
    p.add_argument('--steps',      type=int,   default=LONG_HPARAMS['n_steps'])
    p.add_argument('--eval-every', type=int,   default=LONG_HPARAMS['eval_every'])
    p.add_argument('--log-dir',    type=str,   default='logs')
    p.add_argument('--seed',       type=int,   default=LONG_HPARAMS['seed'])
    args = p.parse_args()

    hp = dict(
        max_seg      = args.max_seg,
        min_seg      = args.min_seg,
        test_seg     = args.test_seg,
        d            = args.d,
        n_layers     = args.n_layers,
        n_heads      = args.n_heads,
        d_ff         = args.d_ff,
        B            = args.B,
        lr_max       = args.lr,
        wd           = args.wd,
        n_steps      = args.steps,
        eval_every   = args.eval_every,
        seed         = args.seed,
        warmup_steps = 500,
    )
    train_long(hp, log_base=args.log_dir)
