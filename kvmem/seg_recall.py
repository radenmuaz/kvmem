"""
kvmem/seg_recall.py — Segmented-decode recall training.

Encode the full source into N KV slots, decode in chunk_len-token windows.

KEY FIX vs prior broken version:
  Training always provides a warmup token (last token before the chunk)
  so the model knows WHERE in the sequence to start decoding.
  Without this, random chunk offset gives the model no localization cue → collapse.

Sequence format:
  [x_S (src_len) | STX | slots (N) | ETX | warmup (1) | y_chunk (chunk_len)]
  Supervise: y_chunk only (not warmup).  L = src_len + 2 + N + 1 + chunk_len.

Inference (ar_decode_seg):
  Chain chunks: warmup of chunk k+1 = last generated token of chunk k.

Sanity checks:
  1. In-distribution val: fresh random sequences (same 5 distributions as training).
     If this fails → the model isn't learning copy at all (architecture/training bug).
  2. OOD test: held-out deterministic sequences (up_counter, etc.).
     If in-dist passes but OOD fails → generalization gap.
  3. Mini mode (--mini): src=128 N=128 chunk=16 — fast convergence sanity check.

Usage:
    python -m kvmem.seg_recall                          # default: src=1024 N=1024 chunk=128
    python -m kvmem.seg_recall --mini                   # src=128 N=128 chunk=16
    python -m kvmem.seg_recall --src-len 1024 --N 1024 --chunk 128 --test-len 512
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
# Mask: src + N slots + warmup(1) + chunk
# ---------------------------------------------------------------------------

def make_seg_mask(src_len: int, N: int, chunk_len: int) -> np.ndarray:
    """
    Mask for [x_S | STX | slots | ETX | warmup | y_chunk].
    warmup is the first Y token — same causal rules apply.
    We treat (warmup + y_chunk) as L_y = 1 + chunk_len in make_mask_stage0.
    """
    return make_mask_stage0(src_len, N, 1 + chunk_len)


# ---------------------------------------------------------------------------
# Batch builder — with warmup token
# ---------------------------------------------------------------------------

def _sample_seg(rng: np.random.Generator, src_len: int) -> np.ndarray:
    """Sample one random source sequence of length src_len in [DATA_LO, 0xFF]."""
    V_full = 256 - DATA_LO
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
    return seg


def make_seg_batch(rng: np.random.Generator, B: int,
                   src_len: int, N: int, chunk_len: int) -> np.ndarray:
    """
    Build training batch with warmup token.

    Format: [x_S | STX | slots | ETX | warmup | y_chunk]
    - warmup = x_S[y_start - 1] for y_start > 0, or x_S[0] for chunk 0
    - y_chunk = x_S[y_start : y_start + chunk_len]
    - Supervise: y_chunk positions only (ETX_pos+1 .. ETX_pos+1+chunk_len)

    L = src_len + 2 + N + 1 + chunk_len
    """
    n_chunks = math.ceil(src_len / chunk_len)
    L        = src_len + 2 + N + 1 + chunk_len
    slot_ids = make_slot_ids(N)
    out      = np.zeros((B, L), dtype=np.int32)

    for i in range(B):
        seg = _sample_seg(rng, src_len)

        # Random chunk offset
        chunk_idx = int(rng.integers(0, n_chunks))
        y_start   = chunk_idx * chunk_len
        y_end     = min(y_start + chunk_len, src_len)

        # Warmup: token just before chunk (wrap to last byte if chunk_idx==0)
        warmup = seg[y_start - 1] if y_start > 0 else seg[0]

        # Y chunk (may be shorter than chunk_len at end of sequence)
        y_chunk = seg[y_start:y_end]

        out[i, :src_len]               = seg
        out[i, src_len]                = STX
        out[i, src_len+1:src_len+1+N]  = np.array(slot_ids, dtype=np.int32)
        out[i, src_len+1+N]            = ETX
        out[i, src_len+2+N]            = warmup
        out[i, src_len+3+N:src_len+3+N+len(y_chunk)] = y_chunk
        # Remaining positions (last chunk shorter than chunk_len) stay 0

    return out


# ---------------------------------------------------------------------------
# In-distribution val batch (same distributions, fresh sequences)
# ---------------------------------------------------------------------------

def make_val_batch(rng: np.random.Generator, B: int,
                   src_len: int, N: int, chunk_len: int) -> tuple[np.ndarray, list]:
    """
    Val batch: sample fresh sequences, always use chunk 0.
    Returns (tokens, list_of_segs) so we can check copy accuracy.
    Warmup = seg[0], target = seg[1:chunk_len+1].
    """
    L        = src_len + 2 + N + 1 + chunk_len
    slot_ids = make_slot_ids(N)
    out      = np.zeros((B, L), dtype=np.int32)
    segs     = []

    for i in range(B):
        seg    = _sample_seg(rng, src_len)
        warmup = seg[0]
        y_tgt  = seg[1:chunk_len+1]

        out[i, :src_len]               = seg
        out[i, src_len]                = STX
        out[i, src_len+1:src_len+1+N]  = np.array(slot_ids, dtype=np.int32)
        out[i, src_len+1+N]            = ETX
        out[i, src_len+2+N]            = warmup
        out[i, src_len+3+N:src_len+3+N+len(y_tgt)] = y_tgt
        segs.append(seg)

    return out, segs


# ---------------------------------------------------------------------------
# Greedy AR decode — full sequence via chained chunks
# ---------------------------------------------------------------------------

def ar_decode_seg(model: KVMemModel, x_S: list[int],
                  N: int, chunk_len: int) -> list[int]:
    """
    Decode full x_S by chaining chunk_len-token windows.

    For each chunk:
      - warmup = last generated token (or x_S[0] for first chunk)
      - generate chunk_len new tokens from [x_S | STX | slots | ETX | warmup | ...]

    Returns list of length src_len (full recall).
    """
    src_len  = len(x_S)
    slot_ids = make_slot_ids(N)
    mem_blk  = [STX] + slot_ids + [ETX]
    n_chunks = math.ceil(src_len / chunk_len)

    # Pre-build mask for (1 warmup + chunk_len) decode window
    L_full   = src_len + 2 + N + 1 + chunk_len
    mask_jnp = jnp.array(make_mask_stage0(src_len, N, 1 + chunk_len))

    recalled = []
    warmup   = x_S[0]  # first chunk warmup = first source byte

    for chunk_idx in range(n_chunks):
        n_gen     = min(chunk_len, src_len - chunk_idx * chunk_len)
        generated = [warmup]

        for _ in range(n_gen):
            cur    = x_S + mem_blk + generated
            pad_n  = L_full - len(cur)
            padded = jnp.array(cur + [0] * pad_n, dtype=jnp.int32)
            logits = model(padded, mask_jnp)
            nb     = int(jnp.argmax(logits[len(cur) - 1]))
            generated.append(nb)

        recalled.extend(generated[1:])   # skip warmup, keep n_gen new tokens
        warmup = generated[-1]           # chain: last token becomes next warmup

    return recalled[:src_len]


# ---------------------------------------------------------------------------
# Eval: in-distribution val (copy accuracy on random sequences)
# ---------------------------------------------------------------------------

def eval_indist(model: KVMemModel, rng: np.random.Generator,
                src_len: int, N: int, chunk_len: int,
                n_seqs: int, log_fn) -> float:
    """
    Evaluate copy accuracy on fresh random sequences (in-distribution).
    Tests the FIRST chunk only (chunk 0) for speed.
    Returns mean CER over n_seqs sequences.
    """
    cers = []
    for _ in range(n_seqs):
        seg    = _sample_seg(rng, src_len)
        warmup = seg[0]
        target = list(seg[1:chunk_len+1])
        # Decode just the first chunk
        gen = ar_decode_seg(model, list(seg), N, chunk_len)
        gen_chunk = gen[1:chunk_len+1]   # skip position 0 (warmup position)
        c = cer(gen_chunk, target)
        cers.append(c)
    mean_cer = float(np.mean(cers))
    match    = 100 * (1 - mean_cer)
    log_fn(f'  [in-dist val] n={n_seqs}  mean CER={mean_cer:.3f}  match={match:.1f}%')
    return mean_cer


# ---------------------------------------------------------------------------
# Eval: OOD test sequences
# ---------------------------------------------------------------------------

def eval_test_seqs(model: KVMemModel, test_len: int, N: int, chunk_len: int,
                   log_fn) -> float:
    test_seqs = make_test_sequences(test_len)
    cers = []
    for name, x_S in test_seqs.items():
        gen  = ar_decode_seg(model, x_S, N, chunk_len)
        c    = cer(gen, x_S)
        match = 100 * (1 - c)
        ok   = '✓' if c == 0.0 else '✗'
        short = 40
        log_fn(f'  {ok} {name:<18} match={match:5.1f}%  CER={c:.3f}'
               f'  gen={bytes(gen).hex()[:short]}...'
               f'  ref={bytes(x_S).hex()[:short]}...')
        cers.append(c)
    mean_cer = float(np.mean(cers))
    log_fn(f'  → OOD mean CER={mean_cer:.3f}  match={100*(1-mean_cer):.1f}%')
    return mean_cer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_seg(hp: dict, log_base: str = 'logs', device: str = 'cpu'):
    src_len      = hp['src_len']
    N            = hp.get('N', src_len)
    chunk_len    = hp['chunk_len']
    test_len     = hp['test_len']
    d            = hp['d']
    n_layers     = hp['n_layers']
    n_heads      = hp.get('n_heads', 4)
    d_ff         = hp.get('d_ff', d * 4)
    B            = hp['B']
    lr_max       = hp['lr_max']
    wd           = hp['wd']
    n_steps      = hp['n_steps']
    eval_every   = hp['eval_every']
    seed         = hp.get('seed', 42)
    warmup_steps = hp.get('warmup_steps', 500)
    val_seqs     = hp.get('val_seqs', 8)   # in-distribution val sequences

    # L includes 1 warmup token before the chunk
    L = src_len + 2 + N + 1 + chunk_len
    ETX_pos = src_len + 1 + N
    # Supervise positions ETX_pos+1 .. ETX_pos+1+chunk_len (skip warmup at ETX_pos)
    Y_start = ETX_pos + 1
    Y_end   = ETX_pos + 1 + chunk_len

    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(log_base, f'seg_recall_{ts}')
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    log_f  = open(os.path.join(run_dir, 'train.log'),   'w', buffering=1)
    jlog_f = open(os.path.join(run_dir, 'train.jsonl'), 'w', buffering=1)
    with open(os.path.join(run_dir, 'hparams.json'), 'w') as f:
        json.dump(hp, f, indent=2)

    def _log(msg): print(msg); log_f.write(msg + '\n')
    def _jlog(d):  jlog_f.write(json.dumps(d) + '\n')

    model_hp = dict(V=256, d=d, n_layers=n_layers, n_heads=n_heads,
                    d_ff=d_ff, seg_len=src_len, N=N)
    _cpu = jax.devices('cpu')[0]
    _dev = jax.devices('mps')[0] if device == 'mps' else _cpu
    with jax.default_device(_cpu):
        key   = jax.random.PRNGKey(seed)
        model = build_model(model_hp, jax.random.split(key)[0])
        opt_state = init_opt_state(model, optimizer='adamw')
    if _dev != _cpu:
        model     = jax.device_put(model, _dev)
        opt_state = jax.device_put(opt_state, _dev)

    pcount = count_params(model)
    compress_str = f'N={N} ({src_len//N}x compress)' if N < src_len else f'N={N} (no compress)'
    import sys
    _log(f'\n=== Seg Recall Training | run_dir={run_dir} ===')
    _log(f'  cmd: {" ".join(sys.argv)}')
    _log(f'  Model: d={d}  n_layers={n_layers}  n_heads={n_heads}'
         f'  d_ff={d_ff}  params={pcount["total"]:,}')
    _log(f'  src_len={src_len}  {compress_str}  chunk_len={chunk_len}  L={L}')
    _log(f'  test_len={test_len}  n_chunks(test)={math.ceil(test_len/chunk_len)}')
    _log(f'  Steps={n_steps}  B={B}  lr={lr_max}  wd={wd}')
    _log(f'  Train: RANDOM bytes, 5 distributions, random chunk per step (with warmup)')
    _log(f'  Val: {val_seqs} fresh random seqs (in-dist) + OOD deterministic seqs')
    _log(f'  Device: {_dev}')

    mask_np = make_seg_mask(src_len, N, chunk_len)
    mask_j  = jax.device_put(jnp.array(mask_np), _dev)

    @jax.jit
    def _step(model, opt_state, tokens_b, lr, step_i):
        params = eqx.filter(model, eqx.is_array)

        def _loss(m):
            def _single(tok):
                logits = m(tok, mask_j)
                lp     = jax.nn.log_softmax(logits[:-1], axis=-1)
                tgts   = tok[1:]
                nll    = -lp[jnp.arange(L - 1), tgts]
                pos    = jnp.arange(L - 1)
                # Supervise y_chunk only (not warmup token)
                mask_y = ((pos >= Y_start) & (pos < Y_end)).astype(jnp.float32)
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

    rng     = np.random.default_rng(seed)
    val_rng = np.random.default_rng(seed + 1)  # separate rng for val (reproducible)
    t0      = time.time()

    for step in range(1, n_steps + 1):
        # LR schedule: warmup then cosine decay
        if step <= warmup_steps:
            lr = lr_max * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (n_steps - warmup_steps)
            lr = max(lr_max * 0.5 * (1 + math.cos(math.pi * progress)), lr_max * 1e-3)

        tokens_np = make_seg_batch(rng, B, src_len, N, chunk_len)
        tokens_b  = jax.device_put(jnp.array(tokens_np), _dev)
        model, opt_state, loss = _step(model, opt_state, tokens_b, lr, step)

        if step % eval_every == 0 or step == 1:
            loss_f  = float(loss)
            elapsed = time.time() - t0
            _log(f'\n--- step={step}/{n_steps}  loss={loss_f:.4f}  lr={lr:.2e}  {elapsed:.0f}s ---')
            _jlog(dict(step=step, loss=loss_f, lr=lr))

            # In-distribution val
            indist_cer = eval_indist(model, val_rng, src_len, N, chunk_len, val_seqs, _log)

            # OOD test (deterministic sequences)
            ood_cer = eval_test_seqs(model, test_len, N, chunk_len, _log)

            _jlog(dict(step=step, indist_cer=indist_cer, ood_cer=ood_cer))

            if ood_cer == 0.0:
                _log(f'\n★ PERFECT OOD RECALL at step {step}!')
                ckpt_path = os.path.join(ckpt_dir, f'seg_step{step}')
                save_checkpoint(ckpt_path, model, step, model_hp)
                _log(f'  [ckpt] {ckpt_path}')
                break

            if indist_cer == 0.0 and step >= eval_every:
                _log(f'  ✓ In-dist perfect. OOD still failing — generalization gap.')

        if step % (eval_every * 10) == 0:
            ckpt_path = os.path.join(ckpt_dir, f'seg_step{step}')
            save_checkpoint(ckpt_path, model, step, model_hp)
            _log(f'  [ckpt] {ckpt_path}')

    _log(f'\nDone. Total: {time.time()-t0:.0f}s')
    log_f.close()
    jlog_f.close()
    return model, run_dir


# ---------------------------------------------------------------------------
# Preset configs
# ---------------------------------------------------------------------------

MINI_HPARAMS = dict(      # fast sanity: src=128 N=128 chunk=16, ~seconds per step on MPS
    src_len=128, N=128, chunk_len=16, test_len=64,
    d=64, n_layers=4, n_heads=4, d_ff=256,
    B=8, lr_max=1e-3, wd=0.01, warmup_steps=200,
    n_steps=20_000, eval_every=2_000, val_seqs=8, seed=42,
)

SEG_HPARAMS = dict(       # main: src=1024 N=1024 chunk=128
    src_len=1024, N=1024, chunk_len=128, test_len=512,
    d=64, n_layers=4, n_heads=4, d_ff=256,
    B=1, lr_max=3e-4, wd=0.01, warmup_steps=500,
    n_steps=30_000, eval_every=5_000, val_seqs=8, seed=42,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mini',       action='store_true', help='Use mini config (src=128)')
    p.add_argument('--src-len',    type=int)
    p.add_argument('--N',          type=int)
    p.add_argument('--chunk',      type=int)
    p.add_argument('--test-len',   type=int)
    p.add_argument('--d',          type=int)
    p.add_argument('--n-layers',   type=int)
    p.add_argument('--n-heads',    type=int)
    p.add_argument('--d-ff',       type=int)
    p.add_argument('--B',          type=int)
    p.add_argument('--lr',         type=float)
    p.add_argument('--wd',         type=float)
    p.add_argument('--steps',      type=int)
    p.add_argument('--eval-every', type=int)
    p.add_argument('--val-seqs',   type=int)
    p.add_argument('--log-dir',    type=str,   default='logs')
    p.add_argument('--seed',       type=int)
    p.add_argument('--device',     type=str,   default='cpu', choices=['cpu', 'mps'],
                   help='Device to train on (default: cpu)')
    args = p.parse_args()

    base = MINI_HPARAMS if args.mini else SEG_HPARAMS
    hp = dict(base)  # copy defaults

    # CLI overrides
    if args.src_len    is not None: hp['src_len']    = args.src_len
    if args.N          is not None: hp['N']          = args.N
    if args.chunk      is not None: hp['chunk_len']  = args.chunk
    if args.test_len   is not None: hp['test_len']   = args.test_len
    if args.d          is not None: hp['d']          = args.d
    if args.n_layers   is not None: hp['n_layers']   = args.n_layers
    if args.n_heads    is not None: hp['n_heads']    = args.n_heads
    if args.d_ff       is not None: hp['d_ff']       = args.d_ff
    if args.B          is not None: hp['B']          = args.B
    if args.lr         is not None: hp['lr_max']     = args.lr
    if args.wd         is not None: hp['wd']         = args.wd
    if args.steps      is not None: hp['n_steps']    = args.steps
    if args.eval_every is not None: hp['eval_every'] = args.eval_every
    if args.val_seqs   is not None: hp['val_seqs']   = args.val_seqs
    if args.seed       is not None: hp['seed']       = args.seed

    train_seg(hp, log_base=args.log_dir, device=args.device)
