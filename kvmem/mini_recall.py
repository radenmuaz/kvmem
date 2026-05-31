"""
kvmem/mini_recall.py — Minimal KV recall sanity check.

Goal: prove the KV bottleneck CAN losslessly memorize a short sequence,
then AR-decode it from a 1-byte warmup.

Test sequences are simple deterministic patterns (NOT random):
  - up_counter:   [32, 33, 34, 35, 36, 37, 38, 39]
  - down_counter: [39, 38, 37, 36, 35, 34, 33, 32]
  - odd:          [33, 35, 37, 39, 41, 43, 45, 47]  (odd bytes from DATA_LO)
  - even:         [32, 34, 36, 38, 40, 42, 44, 46]  (even bytes)
  - linear:       [32, 36, 40, 44, 48, 52, 56, 60]  (step=4)
  - geometric:    [32, 36, 40, 45, 50, 56, 63, 71]  (approx *1.12)
  - sawtooth:     [32, 36, 40, 44, 32, 36, 40, 44]  (repeating pattern)
  - palindrome:   [32, 35, 38, 41, 41, 38, 35, 32]

Training: SYNTHETIC random sequences from the same patterns above,
with random offsets and scales. No real text, no Arabic.

Protocol: [x_S | STX | SLOT_IDs | ETX | x_S]   (Y = copy of x_S)
Eval: greedy AR decode from 1-byte warmup, report CER per sequence.

Usage:
    python -m kvmem.mini_recall [--seg-len 8] [--N 4] [--steps 5000]
                                 [--eval-every 100] [--log-dir logs]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from kvmem.data import (
    DATA_LO, ETX, NUL, STX, make_slot_ids,           # legacy v1
    MEM_OPEN, MEM_CLOSE,                              # tag bytes
    MEM_OPEN_LEN, MEM_CLOSE_LEN, MEM_OVERHEAD,       # tag lengths
    make_slot_ids_tag, make_mask_tag,                 # tag scheme
    make_mask_stage0,
)
from kvmem.stage0 import (
    KVMemModel,
    build_model,
    count_params,
    init_opt_state,
    adam_update,
    clip_grads,
    setup_run_dir,
    save_checkpoint,
)

# ---------------------------------------------------------------------------
# Test sequences (deterministic, interpretable)
# All bytes mapped to [DATA_LO, 0xFF] range
# ---------------------------------------------------------------------------

def make_test_sequences(seg_len: int) -> dict[str, list[int]]:
    """
    Generate deterministic test sequences of length seg_len.
    All bytes strictly in [DATA_LO=0x20, 0xFF] — wraps within this range,
    never into protocol byte territory [0x00, 0x1F].
    """
    V  = 256 - DATA_LO   # 236 usable values
    seqs = {}

    # Up counter: wraps within [DATA_LO, 0xFF]
    seqs['up_counter']   = [DATA_LO + (i % V) for i in range(seg_len)]

    # Down counter: wraps within [DATA_LO, 0xFF]
    seqs['down_counter'] = [DATA_LO + (V - 1 - i % V) for i in range(seg_len)]

    # Odd: step=2, wraps within V
    base_odd = 1 if V % 2 == 0 else 0   # start at offset 1 for odd
    seqs['odd']          = [DATA_LO + (base_odd + 2*i) % V for i in range(seg_len)]

    # Even: step=2, wraps within V
    seqs['even']         = [DATA_LO + (2*i) % V for i in range(seg_len)]

    # Linear step=4, wraps within V
    seqs['linear']       = [DATA_LO + (4*i) % V for i in range(seg_len)]

    # Sawtooth: period = min(seg_len//2, V//4) to keep steps in range
    period = max(4, min(seg_len // 2, V // 4))
    step   = V // period
    seqs['sawtooth']     = [DATA_LO + (i % period) * step for i in range(seg_len)]

    # Palindrome: first half counts up by 2 (mod V), second half mirrors
    half = seg_len // 2
    first_half  = [DATA_LO + (2*i) % V for i in range(half)]
    second_half = list(reversed(first_half))
    extra = [DATA_LO + (2*half) % V] if seg_len % 2 == 1 else []
    seqs['palindrome']   = first_half + extra + second_half

    # Geometric: multiply by ~1.1, wrap within [DATA_LO, 0xFF] on overflow
    geo = [DATA_LO]
    for _ in range(seg_len - 1):
        nxt = int(geo[-1] * 1.1)
        if nxt > 255:
            nxt = DATA_LO
        geo.append(nxt)
    seqs['geometric']    = geo

    return seqs


# ---------------------------------------------------------------------------
# Synthetic training data — UNIFORM RANDOM BYTES ONLY
# Training data must not contain or resemble the held-out test sequences.
# The model must learn the general copy algorithm, not pattern-specific shortcuts.
# ---------------------------------------------------------------------------

def _make_synthetic_batch(rng: np.random.Generator, B: int,
                          seg_len: int, N: int,
                          slot_style: str = 'seq') -> np.ndarray:
    """
    Tag-based training batch. Full byte range [0x00, 0xFF]. Y = exact copy of x_S.

    Format: [x_S | '<m>' | slot_tokens (N) | '</m>' | Y]
    L = seg_len + 3 + N + 4 + seg_len = 2*seg_len + N + 7

    slot_style='zeros': all slot tokens = 0x00 (YaRN position alone routes)
    slot_style='seq':   slot i = i % 256 (position + token identity routes)

    Training data: uniform random bytes over full [0x00, 0xFF]. No restrictions.
    4 distributions: uniform, Dirichlet-skewed, sub-range, geometric.
    """
    slot_ids = make_slot_ids_tag(N, slot_style)
    L        = seg_len + MEM_OVERHEAD + N + seg_len
    M_start  = seg_len + MEM_OPEN_LEN
    Y_start  = seg_len + MEM_OPEN_LEN + N + MEM_CLOSE_LEN
    out      = np.empty((B, L), dtype=np.int32)

    for i in range(B):
        dist_type = int(rng.integers(0, 4))
        if dist_type == 0:
            seg = rng.integers(0, 256, size=seg_len).astype(np.int32)
        elif dist_type == 1:
            alpha = float(rng.uniform(0.05, 1.0))
            p     = rng.dirichlet(np.ones(256) * alpha)
            seg   = rng.choice(256, size=seg_len, p=p).astype(np.int32)
        elif dist_type == 2:
            width = int(rng.integers(4, 129))
            lo    = int(rng.integers(0, 256 - width + 1))
            seg   = rng.integers(lo, lo + width, size=seg_len).astype(np.int32)
        else:
            p_g = float(rng.uniform(0.01, 0.3))
            raw = rng.geometric(p_g, size=seg_len) - 1
            seg = np.clip(raw, 0, 255).astype(np.int32)

        out[i, :seg_len]                = seg
        out[i, seg_len:M_start]         = MEM_OPEN
        out[i, M_start:M_start+N]       = slot_ids
        out[i, M_start+N:Y_start]       = MEM_CLOSE
        out[i, Y_start:]                = seg

    return out


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------

def make_recall_mask(seg_len: int, N: int, slot_style: str = 'seq') -> np.ndarray:
    """Tag-based causal mask: [x_S | <m> | slots | </m> | Y]."""
    return make_mask_tag(seg_len, N, seg_len)


# ---------------------------------------------------------------------------
# AR decode (greedy)
# ---------------------------------------------------------------------------

def ar_decode(model: KVMemModel, x_S: list[int], N: int,
              warmup: list[int], max_new: int,
              slot_style: str = 'seq') -> list[int]:
    """
    Greedy AR decode using tag-based scheme.
    Format: [x_S | '<m>' | slot_tokens | '</m>' | Y]
    All bytes 0x00..0xFF valid in x_S.
    """
    seg_len  = len(x_S)
    L_y      = len(warmup) + max_new
    L_full   = seg_len + MEM_OVERHEAD + N + L_y
    slot_ids = make_slot_ids_tag(N, slot_style)
    mem_block = MEM_OPEN + slot_ids + MEM_CLOSE
    mask_full = jnp.array(make_mask_tag(seg_len, N, L_y))

    generated = list(warmup)
    for _ in range(max_new):
        cur    = x_S + mem_block + generated
        pad_n  = L_full - len(cur)
        padded = jnp.array(cur + [0] * pad_n, dtype=jnp.int32)
        logits = model(padded, mask_full)
        nb     = int(jnp.argmax(logits[len(cur) - 1]))
        generated.append(nb)

    return generated


def cer(pred: list[int], ref: list[int]) -> float:
    """Character Error Rate (edit distance / len(ref))."""
    m, n = len(ref), len(pred)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if ref[i-1] == pred[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j-1], prev[j], dp[j-1])
    return dp[n] / max(m, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_mini(hp: dict, log_base: str = 'logs', device: str = 'cpu'):
    seg_len    = hp['seg_len']
    N          = hp['N']
    B          = hp['B']
    n_steps    = hp['n_steps']
    eval_every = hp['eval_every']
    warmup_n   = hp.get('warmup_n', 1)   # number of warmup bytes for AR eval
    lr_max     = hp['lr_max']
    wd         = hp['wd']

    _cpu = jax.devices('cpu')[0]
    _dev = jax.devices('mps')[0] if device == 'mps' else _cpu
    with jax.default_device(_cpu):
        key = jax.random.PRNGKey(hp['seed'])
        key, mkey = jax.random.split(key)
        model     = build_model(hp, mkey)
        opt_state = init_opt_state(model, optimizer='adamw')
    if _dev != _cpu:
        model     = jax.device_put(model, _dev)
        opt_state = jax.device_put(opt_state, _dev)

    run_dir  = setup_run_dir(log_base, 'mini_recall')
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    log_f  = open(os.path.join(run_dir, 'train.log'),   'w', buffering=1)
    jlog_f = open(os.path.join(run_dir, 'train.jsonl'), 'w', buffering=1)

    def _log(msg):
        tqdm.write(msg)
        log_f.write(msg + '\n')

    def _jlog(rec):
        jlog_f.write(json.dumps(rec) + '\n')

    with open(os.path.join(run_dir, 'hparams.json'), 'w') as f:
        json.dump(hp, f, indent=2)

    slot_style = hp.get('slot_style', 'seq')
    pcount   = count_params(model)
    mask_jnp = jax.device_put(jnp.array(make_recall_mask(seg_len, N, slot_style)), _dev)
    L_total  = seg_len + MEM_OVERHEAD + N + seg_len
    Y_start  = seg_len + MEM_OPEN_LEN + N + MEM_CLOSE_LEN  # first Y token

    test_seqs = make_test_sequences(seg_len)

    import sys
    _log(f'\n=== Mini Recall Training | run_dir={run_dir} ===')
    _log(f'  cmd: {" ".join(sys.argv)}')
    _log(f'  Model: d={hp["d"]}  n_layers={hp["n_layers"]}  params={pcount["total"]:,}')
    _log(f'  seg_len={seg_len}  N={N}  warmup_n={warmup_n}')
    _log(f'  slot_style={slot_style}  format: <m> + slots + </m>  L_total={L_total}')
    _log(f'  Steps={n_steps}  Batch={B}  lr={lr_max}  wd={wd}  device={_dev}')
    _log(f'  Train: RANDOM bytes [0x00..0xFF] — uniform/Dirichlet/sub-range/geometric')
    _log(f'  Test (held out): {list(test_seqs.keys())}')

    rng = np.random.default_rng(hp['seed'] + 1)
    t0  = time.time()

    @jax.jit
    def _step(model, opt_state, tokens, step, lr):
        params = eqx.filter(model, eqx.is_array)

        def _loss(m):
            B_loc = tokens.shape[0]
            L     = tokens.shape[1]
            logits = jax.vmap(lambda tok: m(tok, mask_jnp))(tokens)  # (B, L, V)
            lp     = jax.nn.log_softmax(logits[:, :-1], axis=-1)
            tgts   = tokens[:, 1:]
            nll    = -lp[jnp.arange(B_loc)[:, None], jnp.arange(L-1)[None, :], tgts]
            pos       = jnp.arange(L - 1)
            Y_end     = Y_start + seg_len
            mask_cont = ((pos >= Y_start) & (pos < Y_end)).astype(jnp.float32)
            return jnp.sum(nll * mask_cont[None, :]) / (mask_cont.sum() * B_loc + 1e-8)

        loss, grads = jax.value_and_grad(_loss)(model)
        grads_arr = eqx.filter(grads, eqx.is_array)
        grads_arr = clip_grads(grads_arr, max_norm=hp['grad_clip'])
        new_params, new_opt = adam_update(
            params, grads_arr, opt_state, lr, wd=wd, step=step)
        deltas    = jax.tree.map(lambda np_, p: np_ - p, new_params, params)
        new_model = eqx.apply_updates(model, deltas)
        return new_model, new_opt, loss

    # LR schedule: linear warmup + cosine decay
    def _lr(step):
        w = hp.get('warmup_steps', 200)
        if step < w:
            return lr_max * step / w
        frac = min((step - w) / max(n_steps - w, 1), 1.0)
        return 1e-6 + 0.5 * (lr_max - 1e-6) * (1 + math.cos(math.pi * frac))

    pbar = tqdm(range(1, n_steps + 1), desc='mini_recall', dynamic_ncols=True)

    for step in pbar:
        np_tokens = _make_synthetic_batch(rng, B, seg_len, N, slot_style)
        tokens_b  = jax.device_put(jnp.array(np_tokens), _dev)
        lr        = _lr(step)
        model, opt_state, loss = _step(model, opt_state, tokens_b, step, lr)
        loss_f = float(loss)
        pbar.set_postfix(loss=f'{loss_f:.4f}', lr=f'{lr:.1e}', refresh=False)

        if step % eval_every == 0 or step == 1:
            elapsed = time.time() - t0
            _log(f'\n--- step={step}/{n_steps}  loss={loss_f:.4f}  lr={lr:.2e}  {elapsed:.0f}s ---')
            _jlog(dict(step=step, loss=loss_f, lr=lr, elapsed=elapsed))

            # Eval all test sequences
            all_cer = []
            for name, x_S in test_seqs.items():
                warmup = x_S[:warmup_n]
                target = x_S[warmup_n:]
                gen    = ar_decode(model, x_S, N, warmup, len(target), slot_style)
                gen_tail = gen[warmup_n:]
                c = cer(gen_tail, target)
                all_cer.append(c)
                match_pct = 100 * (1 - c)
                exact = '✓' if c == 0.0 else '✗'
                _log(f'  {exact} {name:15s}  match={match_pct:5.1f}%  CER={c:.3f}  '
                     f'gen={bytes(gen_tail).hex()}  ref={bytes(target).hex()}')
                _jlog(dict(step=step, seq=name, cer=c, match_pct=match_pct,
                           gen=gen_tail, ref=target))

            mean_cer = sum(all_cer) / len(all_cer)
            mean_match = 100 * (1 - mean_cer)
            _log(f'  → mean CER={mean_cer:.3f}  mean match={mean_match:.1f}%')
            _jlog(dict(step=step, mean_cer=mean_cer, mean_match=mean_match))

            if mean_cer == 0.0:
                _log(f'\n★ PERFECT RECALL at step {step}! All test sequences 100% correct.')
                ckpt_path = os.path.join(ckpt_dir, f'mini_step{step}')
                save_checkpoint(ckpt_path, model, step, hp)
                _log(f'  [ckpt] {ckpt_path}')
                break

        if step % (eval_every * 10) == 0:
            ckpt_path = os.path.join(ckpt_dir, f'mini_step{step}')
            save_checkpoint(ckpt_path, model, step, hp)
            _log(f'  [ckpt] {ckpt_path}')

    _log(f'\nDone. Total: {time.time()-t0:.0f}s')
    log_f.close()
    jlog_f.close()
    return model, run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MINI_HPARAMS = dict(
    V          = 256,
    d          = 64,
    n_layers   = 4,
    n_heads    = 4,
    d_ff       = 128,
    seg_len    = 8,
    N          = 4,
    B          = 64,
    lr_max     = 1e-3,
    wd         = 0.0,
    grad_clip  = 1.0,
    warmup_steps = 200,
    n_steps    = 30_000,
    eval_every = 2_000,
    warmup_n   = 1,    # AR eval: 1 byte warmup
    seed       = 42,
    rope       = False,
    yarn       = False,
    slot_style = 'seq',   # 'zeros' or 'seq'
)


def main():
    parser = argparse.ArgumentParser(prog='kvmem.mini_recall')
    parser.add_argument('--seg-len',    type=int,   default=None)
    parser.add_argument('--N',          type=int,   default=None)
    parser.add_argument('--steps',      type=int,   default=None)
    parser.add_argument('--eval-every', type=int,   default=None)
    parser.add_argument('--d',          type=int,   default=None)
    parser.add_argument('--n-layers',   type=int,   default=None)
    parser.add_argument('--B',          type=int,   default=None)
    parser.add_argument('--lr',         type=float, default=None)
    parser.add_argument('--wd',         type=float, default=None)
    parser.add_argument('--warmup-n',   type=int,   default=None,
                        help='AR eval warmup bytes (default 1)')
    parser.add_argument('--rope',       action='store_true')
    parser.add_argument('--yarn',       action='store_true',
                        help='YaRN scaled RoPE (requires --rope)')
    parser.add_argument('--slot-style', type=str, default=None,
                        choices=['zeros', 'seq'],
                        help='zeros: all-zero slot tokens (YaRN-only routing); '
                             'seq: slot i = i%%256 (default)')
    parser.add_argument('--log-dir',    type=str,   default='logs')
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--device',     type=str,   default='cpu', choices=['cpu', 'mps'],
                        help='Device to train on (default: cpu)')
    args = parser.parse_args()

    hp = dict(MINI_HPARAMS)
    if args.seg_len:    hp['seg_len']    = args.seg_len
    if args.N:          hp['N']          = args.N
    if args.steps:      hp['n_steps']    = args.steps
    if args.eval_every: hp['eval_every'] = args.eval_every
    if args.d:
        hp['d']     = args.d
        hp['d_ff']  = args.d * 2
    if args.n_layers:   hp['n_layers']   = args.n_layers
    if args.B:          hp['B']          = args.B
    if args.lr:         hp['lr_max']     = args.lr
    if args.wd:         hp['wd']         = args.wd
    if args.warmup_n:   hp['warmup_n']   = args.warmup_n
    hp['rope']  = args.rope
    hp['yarn']  = args.yarn
    hp['seed']  = args.seed
    if args.slot_style is not None:
        hp['slot_style'] = args.slot_style
    # YaRN L_train = full tag sequence length
    seg = hp['seg_len']; N = hp.get('N', seg)
    hp['L_train'] = seg + MEM_OVERHEAD + N + seg

    train_mini(hp, log_base=args.log_dir, device=args.device)


if __name__ == '__main__':
    main()
