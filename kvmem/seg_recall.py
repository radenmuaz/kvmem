"""
kvmem/seg_recall.py — Segmented-decode recall training.

Encode the full source (up to 1024 bytes) into N=source_len KV slots,
but decode in fixed-size chunks (e.g. 128 bytes at a time).

This keeps the total sequence length manageable:
    L = src_len + 2 + N + chunk_len   (not 3*src_len + 2)

Example: src_len=1024, chunk_len=128, N=1024
    L = 1024 + 2 + 1024 + 128 = 2178  (vs 3074 for full decode)

Training: for each batch, sample a random src_len and chunk offset,
          supervise only the chunk_len Y tokens at that offset.

Inference: decode all chunks sequentially, concatenate → full recall.

Test: deterministic number sequences at test_src_len (e.g. 512),
      decoded in full via multiple chunk passes.

Usage:
    python -m kvmem.seg_recall
    python -m kvmem.seg_recall --src-len 1024 --chunk 128 --test-len 512
    python -m kvmem.seg_recall --src-len 512  --chunk 64  --test-len 256 --d 128
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime

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
# Mask: encode full src, decode one chunk at offset
# ---------------------------------------------------------------------------

def make_seg_mask(src_len: int, N: int, chunk_len: int) -> np.ndarray:
    """
    Mask for [x_S (src_len) | STX | slots (N) | ETX | y_chunk (chunk_len)].
    Same rules as make_mask_stage0 — Y reads only from M+ETX, not x_S.
    """
    return make_mask_stage0(src_len, N, chunk_len)


# ---------------------------------------------------------------------------
# Batch builder: random src, random chunk offset
# ---------------------------------------------------------------------------

def make_seg_batch(rng: np.random.Generator, B: int,
                   src_len: int, N: int, chunk_len: int,
                   chunk_offset: int | None = None) -> tuple[np.ndarray, int]:
    """
    Build one batch.
    src_len: source sequence length
    N: number of KV slots (can be < src_len for compression)
    chunk_len: number of Y tokens to decode per step
    chunk_offset: which chunk of Y to train on (None = random)

    Returns:
        tokens: (B, src_len + 2 + N + chunk_len)
        offset: the chunk_offset used
    """
    V_full   = 256 - DATA_LO
    L        = src_len + 2 + N + chunk_len
    slot_ids = make_slot_ids(N)

    # Pick offset (which chunk of Y to supervise)
    n_chunks = math.ceil(src_len / chunk_len)
    if chunk_offset is None:
        chunk_offset = int(rng.integers(0, n_chunks))
    y_start = chunk_offset * chunk_len
    y_end   = min(y_start + chunk_len, src_len)
    actual_chunk = y_end - y_start  # may be < chunk_len at last chunk

    out = np.zeros((B, L), dtype=np.int32)

    for i in range(B):
        dist_type = int(rng.integers(0, 5))
        if dist_type == 0:
            seg = rng.integers(DATA_LO, 256, size=src_len).astype(np.int32)
        elif dist_type == 1:
            alpha = float(rng.uniform(0.05, 1.0))
            p     = rng.dirichlet(np.ones(V_full) * alpha)
            seg   = (rng.choice(V_full, size=src_len, p=p) + DATA_LO).astype(np.int32)
        elif dist_type == 2:
            width = int(rng.integers(4, min(128, V_full + 1)))
            lo    = int(rng.integers(0, V_full - width + 1)) + DATA_LO
            seg   = rng.integers(lo, lo + width, size=src_len).astype(np.int32)
        elif dist_type == 3:
            p_g = float(rng.uniform(0.01, 0.3))
            raw = rng.geometric(p_g, size=src_len) - 1
            seg = (np.clip(raw, 0, V_full - 1) + DATA_LO).astype(np.int32)
        else:
            c1 = int(rng.integers(DATA_LO, 200))
            c2 = int(rng.integers(min(c1 + 20, 240), min(c1 + 80, 256)))
            w  = float(rng.uniform(0.2, 0.8))
            seg = np.where(
                rng.uniform(size=src_len) < w,
                np.clip(rng.integers(c1, c1 + 16, size=src_len), DATA_LO, 255),
                np.clip(rng.integers(c2, c2 + 16, size=src_len), DATA_LO, 255),
            ).astype(np.int32)

        # Source
        out[i, :src_len]               = seg
        out[i, src_len]                = STX
        out[i, src_len+1:src_len+1+N]  = np.array(slot_ids, dtype=np.int32)
        out[i, src_len+1+N]            = ETX
        # Y chunk: the slice of seg at this chunk offset
        out[i, src_len+2+N:src_len+2+N+actual_chunk] = seg[y_start:y_end]
        # Remaining Y positions (if last chunk is short) stay 0

    return out, chunk_offset


# ---------------------------------------------------------------------------
# Greedy AR decode — all chunks sequentially
# ---------------------------------------------------------------------------

def ar_decode_seg(model: KVMemModel, x_S: list[int], N: int, chunk_len: int) -> list[int]:
    """
    Decode the full x_S by decoding chunk_len tokens at a time.
    N: number of KV slots (may be < len(x_S) for compressed memory).

    x_S: full source sequence
    chunk_len: tokens per decode step
    Returns: list of length len(x_S) (full recall)
    """
    src_len  = len(x_S)
    slot_ids = make_slot_ids(N)
    mem_blk  = [STX] + slot_ids + [ETX]
    n_chunks = math.ceil(src_len / chunk_len)

    recalled = []
    prev_tok = x_S[0]  # warmup = first source byte

    # Pre-build mask (fixed for all chunks of same size)
    # We decode one token at a time for simplicity and correctness
    # (AR decode: each step adds one token to generated)
    L_full   = src_len + 2 + N + chunk_len + 1  # +1 for warmup
    mask_jnp = jnp.array(make_mask_stage0(src_len, N, chunk_len + 1))

    for chunk_idx in range(n_chunks):
        y_start = chunk_idx * chunk_len
        y_end   = min(y_start + chunk_len, src_len)
        n_gen   = y_end - y_start

        # Decode n_gen tokens greedily
        generated = [prev_tok]
        for _ in range(n_gen):
            cur    = x_S + mem_blk + generated
            pad_n  = L_full - len(cur)
            padded = jnp.array(cur + [0] * pad_n, dtype=jnp.int32)
            logits = model(padded, mask_jnp)
            nb     = int(jnp.argmax(logits[len(cur) - 1]))
            generated.append(nb)

        recalled.extend(generated[1:])  # skip warmup token
        prev_tok = generated[-1]        # last generated = warmup for next chunk

    return recalled


# ---------------------------------------------------------------------------
# Eval on deterministic test sequences
# ---------------------------------------------------------------------------

def eval_test_seqs(model: KVMemModel, test_len: int, N: int, chunk_len: int,
                   log_fn) -> float:
    test_seqs = make_test_sequences(test_len)
    cers = []
    for name, x_S in test_seqs.items():
        gen = ar_decode_seg(model, x_S, N, chunk_len)
        c   = cer(gen, x_S)
        match = 100 * (1 - c)
        ok  = '✓' if c == 0.0 else '✗'
        short = 32
        log_fn(f'  {ok} {name:<18} match={match:5.1f}%  CER={c:.3f}'
               f'  gen={bytes(gen).hex()[:short]}{"..." if len(gen)>16 else ""}'
               f'  ref={bytes(x_S).hex()[:short]}{"..." if len(x_S)>16 else ""}')
        cers.append(c)
    mean_cer = float(np.mean(cers))
    log_fn(f'  → mean CER={mean_cer:.3f}  mean match={100*(1-mean_cer):.1f}%')
    return mean_cer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_seg(hp: dict, log_base: str = 'logs'):
    src_len    = hp['src_len']
    chunk_len  = hp['chunk_len']
    test_len   = hp['test_len']
    d          = hp['d']
    n_layers   = hp['n_layers']
    n_heads    = hp.get('n_heads', 8)
    d_ff       = hp.get('d_ff', d * 4)
    B          = hp['B']
    lr_max     = hp['lr_max']
    wd         = hp['wd']
    n_steps    = hp['n_steps']
    eval_every = hp['eval_every']
    seed       = hp.get('seed', 42)
    warmup_steps = hp.get('warmup_steps', 500)

    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(log_base, f'seg_recall_{ts}')
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    log_f  = open(os.path.join(run_dir, 'train.log'),   'w', buffering=1)
    jlog_f = open(os.path.join(run_dir, 'train.jsonl'), 'w', buffering=1)
    with open(os.path.join(run_dir, 'hparams.json'), 'w') as f:
        json.dump(hp, f, indent=2)

    def _log(msg):
        print(msg); log_f.write(msg + '\n')

    def _jlog(d):
        jlog_f.write(json.dumps(d) + '\n')

    N = hp.get('N', src_len)   # slots; default = src_len (no compression)
    L = src_len + 2 + N + chunk_len

    model_hp = dict(V=256, d=d, n_layers=n_layers, n_heads=n_heads,
                    d_ff=d_ff, seg_len=src_len, N=N)
    key   = jax.random.PRNGKey(seed)
    model = build_model(model_hp, jax.random.split(key)[0])
    opt_state = init_opt_state(model, optimizer='adamw')

    pcount = count_params(model)
    _log(f'\n=== Seg Recall Training | run_dir={run_dir} ===')
    _log(f'  Model: d={d}  n_layers={n_layers}  n_heads={n_heads}'
         f'  d_ff={d_ff}  params={pcount["total"]:,}')
    _log(f'  src_len={src_len}  N={N}  chunk_len={chunk_len}  L={L}')
    _log(f'  test_len={test_len}  n_chunks={math.ceil(src_len/chunk_len)}')
    _log(f'  Steps={n_steps}  B={B}  lr={lr_max}  wd={wd}')
    _log(f'  Train: RANDOM bytes, 5 distributions, chunk sampled randomly each step')

    mask_np = make_seg_mask(src_len, N, chunk_len)
    mask_j  = jnp.array(mask_np)

    @jax.jit
    def _step(model, opt_state, tokens_b, lr, step_i):
        params = eqx.filter(model, eqx.is_array)
        ETX_pos = src_len + 1 + N

        def _loss(m):
            def _single(tok):
                logits = m(tok, mask_j)
                lp     = jax.nn.log_softmax(logits[:-1], axis=-1)
                tgts   = tok[1:]
                nll    = -lp[jnp.arange(L - 1), tgts]
                pos    = jnp.arange(L - 1)
                # Supervise all chunk_len Y positions
                mask_y = ((pos >= ETX_pos) & (pos < ETX_pos + chunk_len)).astype(jnp.float32)
                return jnp.sum(nll * mask_y) / (mask_y.sum() + 1e-8)
            return jnp.mean(jax.vmap(_single)(tokens_b))

        loss, grads = jax.value_and_grad(_loss)(model)
        grads_arr   = eqx.filter(grads, eqx.is_array)
        grads_arr   = clip_grads(grads_arr, max_norm=1.0)
        new_params, new_opt = adam_update(params, grads_arr, opt_state, lr,
                                          wd=wd, step=step_i)
        deltas    = jax.tree.map(lambda np_, p: np_ - p, new_params, params)
        new_model = eqx.apply_updates(model, deltas)
        return new_model, new_opt, loss

    rng = np.random.default_rng(seed)
    t0  = time.time()

    for step in range(1, n_steps + 1):
        # LR schedule
        if step <= warmup_steps:
            lr = lr_max * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (n_steps - warmup_steps)
            lr = max(lr_max * 0.5 * (1 + math.cos(math.pi * progress)), lr_max * 1e-3)

        # Random chunk offset each step
        tokens_np, offset = make_seg_batch(rng, B, src_len, N, chunk_len)
        tokens_b = jnp.array(tokens_np)

        model, opt_state, loss = _step(model, opt_state, tokens_b, lr, step)

        if step % eval_every == 0 or step == 1:
            loss_f  = float(loss)
            elapsed = time.time() - t0
            _log(f'\n--- step={step}/{n_steps}  loss={loss_f:.4f}  lr={lr:.2e}  {elapsed:.0f}s ---')
            _jlog(dict(step=step, loss=loss_f, lr=lr))

            mean_cer = eval_test_seqs(model, test_len, N, chunk_len, _log)
            _jlog(dict(step=step, eval_len=test_len, mean_cer=mean_cer))

            if mean_cer == 0.0:
                _log(f'\n★ PERFECT RECALL at step {step}!')
                ckpt_path = os.path.join(ckpt_dir, f'seg_step{step}')
                save_checkpoint(ckpt_path, model, step, model_hp)
                _log(f'  [ckpt] {ckpt_path}')
                break

        if step % (eval_every * 10) == 0:
            ckpt_path = os.path.join(ckpt_dir, f'seg_step{step}')
            save_checkpoint(ckpt_path, model, step, model_hp)
            _log(f'  [ckpt] {ckpt_path}')

    _log(f'\nDone. Total: {time.time()-t0:.0f}s')
    log_f.close()
    jlog_f.close()
    return model, run_dir


# ---------------------------------------------------------------------------
# Defaults & CLI
# ---------------------------------------------------------------------------

SEG_HPARAMS = dict(
    src_len      = 1024,  # full source length
    N            = 256,   # KV slots (4x compression); set to src_len for no compression
    chunk_len    = 128,   # decode chunk size → L = src_len+2+N+chunk_len = 1410
    test_len     = 512,   # held-out test sequence length
    d            = 64,
    n_layers     = 4,
    n_heads      = 4,
    d_ff         = 256,
    B            = 1,
    lr_max       = 3e-4,
    wd           = 0.01,
    warmup_steps = 500,
    n_steps      = 30_000,
    eval_every   = 5_000,
    seed         = 42,
)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src-len',    type=int,   default=SEG_HPARAMS['src_len'])
    p.add_argument('--N',          type=int,   default=SEG_HPARAMS['N'],
                   help='KV slots (default=256, i.e. 4x compression of src_len=1024)')
    p.add_argument('--chunk',      type=int,   default=SEG_HPARAMS['chunk_len'])
    p.add_argument('--test-len',   type=int,   default=SEG_HPARAMS['test_len'])
    p.add_argument('--d',          type=int,   default=SEG_HPARAMS['d'])
    p.add_argument('--n-layers',   type=int,   default=SEG_HPARAMS['n_layers'])
    p.add_argument('--n-heads',    type=int,   default=SEG_HPARAMS['n_heads'])
    p.add_argument('--d-ff',       type=int,   default=SEG_HPARAMS['d_ff'])
    p.add_argument('--B',          type=int,   default=SEG_HPARAMS['B'])
    p.add_argument('--lr',         type=float, default=SEG_HPARAMS['lr_max'])
    p.add_argument('--wd',         type=float, default=SEG_HPARAMS['wd'])
    p.add_argument('--steps',      type=int,   default=SEG_HPARAMS['n_steps'])
    p.add_argument('--eval-every', type=int,   default=SEG_HPARAMS['eval_every'])
    p.add_argument('--log-dir',    type=str,   default='logs')
    p.add_argument('--seed',       type=int,   default=SEG_HPARAMS['seed'])
    args = p.parse_args()

    hp = dict(
        src_len      = args.src_len,
        N            = args.N,
        chunk_len    = args.chunk,
        test_len     = args.test_len,
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
    train_seg(hp, log_base=args.log_dir)
