"""
kvmem/stage1.py — Stage 1: Multi-pass NTP refinement through the KV bottleneck.

Sequence layout:
    [ x_S | MB | y^1 | MB | y^2 | ... | MB | y^T ]
    where MB = [ STX | NUL×N | ETX ]  (N+2 tokens)

Each y^t is an independent continuation from x_S's terminal state.
M^t sees S and all prior M^s≤t; Y^t sees only M^≤t and ETX of its own block.

Loss (focused multi-pass, PLAN_STAGE1.md §2):
    L_total = L_src
            + lambda_cont × L_cont,1_base
            + lambda_cont × Σ_{t≥2} beta_t × L_cont,t_focus
            + lambda_mono × L_mono

Usage:
    python -m kvmem.stage1 train [--passes T] [--steps N] [--optimizer adamw|grokadamw]
    python -m kvmem.stage1 eval  --ckpt PATH
    python -m kvmem.stage1 test  --ckpt PATH [--fatihah PATH]
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
from tqdm import tqdm

from kvmem.data import (
    DATA_LO, ETX, NUL, STX,
    BatchPrefetcher,
    make_chain_pool,
    make_mask_stage1,
    _np_walk_chain,
)
# Reuse model, optimizer, checkpoint utilities from stage0
from kvmem.stage0 import (
    KVMemModel,
    build_model,
    count_params,
    init_opt_state,
    lr_schedule,
    clip_grads,
    adam_update,
    grok_adam_update,
    save_checkpoint,
    load_checkpoint,
    setup_run_dir,
    _load_txt_lines,
    _plot_training,
    _decode,
    run_test,
    FATIHAH_PATH,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_HPARAMS = dict(
    V          = 256,
    V_chain    = 8,
    L_S        = 64,
    N_set      = [2, 4, 8, 16, 32],
    # Multi-pass: T passes. Each y^t is an independent fresh continuation.
    T          = 4,
    # Curriculum on L_y
    L_y_schedule = [(0, 16), (10_000, 32), (25_000, 64)],
    d          = 64,
    n_layers   = 4,
    n_heads    = 4,
    d_ff       = 128,
    lambda_cont= 2.0,   # overall continuation weight (PLAN_STAGE1 §2.7)
    lambda_w   = 2.0,   # hard-position focus strength
    lambda_mono= 0.1,   # monotonicity regularizer weight
    tau        = 0.693, # log(2): error threshold for hard-position weights
    w_max      = 3.0,   # weight ceiling
    gamma      = 0.0,   # monotonicity margin (strict)
    B          = 32,    # smaller batch (sequences are T× longer)
    lr_max     = 3e-4,
    lr_min     = 1e-5,
    warmup_steps = 500,
    n_steps    = 30_000,
    grad_clip  = 1.0,
    wd         = 0.01,
    alpha      = 0.05,
    seed       = 42,
    chain_pool_size = 8,
    optimizer  = 'adamw',
    grok_rho   = 0.9,
)


# ---------------------------------------------------------------------------
# Curriculum helper
# ---------------------------------------------------------------------------

def get_L_y(step: int, schedule: list) -> int:
    L_y = schedule[0][1]
    for start, val in schedule:
        if step >= start:
            L_y = val
    return L_y


# ---------------------------------------------------------------------------
# Stage-1 batch generation (numpy, fast)
# ---------------------------------------------------------------------------

def np_make_stage1_one(rng: np.random.Generator, V_chain: int, L_S: int,
                       L_y: int, N: int, T: int, alpha: float,
                       T_mat: np.ndarray | None = None) -> np.ndarray:
    """
    Build one stage-1 sequence:
        [x_S | STX NUL*N ETX | y^1 | STX NUL*N ETX | y^2 | ... | STX NUL*N ETX | y^T]
    Total length: L_S + T*(N+2+L_y)

    If T_mat is given (from pool), use it; otherwise sample fresh.
    """
    if T_mat is None:
        g = rng.gamma(max(alpha, 1e-3), size=(V_chain, V_chain)).astype(np.float32)
        T_mat = g / g.sum(axis=1, keepdims=True)

    start   = rng.integers(0, V_chain)
    x_S_raw = _np_walk_chain(rng, T_mat, int(start), L_S)
    terminal = int(x_S_raw[-1])

    block = N + 2 + L_y
    L     = L_S + T * block
    seq   = np.empty(L, dtype=np.int32)

    seq[:L_S] = (x_S_raw + DATA_LO).astype(np.int32)

    for t in range(T):
        off = L_S + t * block
        seq[off]               = STX
        seq[off+1 : off+1+N]   = NUL
        seq[off+1+N]           = ETX
        y_raw = _np_walk_chain(rng, T_mat, terminal, L_y)
        seq[off+2+N : off+2+N+L_y] = (y_raw + DATA_LO).astype(np.int32)

    return seq


def np_make_stage1_batch(rng: np.random.Generator, B: int, V_chain: int,
                         L_S: int, L_y: int, N: int, T: int, alpha: float,
                         chain_pool: np.ndarray | None = None) -> np.ndarray:
    """Batch of B stage-1 sequences: (B, L_S + T*(N+2+L_y))."""
    block = N + 2 + L_y
    L     = L_S + T * block
    out   = np.empty((B, L), dtype=np.int32)
    for i in range(B):
        if chain_pool is not None:
            k     = int(rng.integers(0, len(chain_pool)))
            T_mat = chain_pool[k]
        else:
            T_mat = None
        out[i] = np_make_stage1_one(rng, V_chain, L_S, L_y, N, T, alpha, T_mat)
    return out


# ---------------------------------------------------------------------------
# Loss function — focused multi-pass
# ---------------------------------------------------------------------------

def _nll_matrix_1d(model: KVMemModel, tokens: jax.Array,
                   mask: jax.Array) -> jax.Array:
    """Per-token NLL for a single sequence: (L-1,)."""
    logits = model(tokens, mask)           # (L, V)
    lp     = jax.nn.log_softmax(logits[:-1], axis=-1)   # (L-1, V)
    return -lp[jnp.arange(len(tokens) - 1), tokens[1:]] # (L-1,)


def loss_fn_stage1(model: KVMemModel, tokens: jax.Array, mask: jax.Array,
                   L_S: int, N: int, L_y: int, T: int,
                   lambda_cont: float, lambda_w: float, lambda_mono: float,
                   tau: float, w_max: float, gamma: float) -> tuple:
    """
    Stage-1 multi-pass focused loss.
    tokens: (B, L_S + T*(N+2+L_y))

    Returns (total_loss, aux_dict).
    """
    B, L = tokens.shape
    block = N + 2 + L_y

    # Per-token NLL for all examples (B, L-1)
    nll = jax.vmap(lambda tok: _nll_matrix_1d(model, tok, mask))(tokens)

    # Source loss: positions 0..L_S-2 predict 1..L_S-1
    pos = jnp.arange(L - 1)
    src_mask = (pos <= L_S - 2).astype(jnp.float32)
    L_src = jnp.mean(
        jnp.sum(nll * src_mask[None, :], axis=-1) / (src_mask.sum() + 1e-8)
    )

    # Per-pass continuation masks and block NLLs
    # Y^t occupies positions L_S + t*block + N+2 .. L_S + t*block + N+1+L_y  (in tokens)
    # In NLL array (shifted by 1): ETX position predicts y[0], so NLL pos = L_S + t*block + N+1
    # y^t NLL positions: ETX_pos .. ETX_pos + L_y - 1  where ETX_pos = L_S + t*block + N+1
    y_starts = [L_S + t * block + N + 1 for t in range(T)]   # ETX predicts y^t[0]

    # Extract per-pass block NLLs: (T, B, L_y)
    def get_block(t):
        s = y_starts[t]
        return nll[:, s:s + L_y]   # (B, L_y)

    blocks = [get_block(t) for t in range(T)]   # list of (B, L_y)

    # Per-pass base loss (unweighted mean)
    L_cont_base = [b.mean() for b in blocks]

    # Focused losses
    L_cont_focus = []
    for t in range(T):
        if t == 0 or lambda_w == 0.0:
            # Pass 1: standard unweighted loss
            L_cont_focus.append(L_cont_base[t])
        else:
            # Focused: weight by pass t-1's per-position error
            e_prev = jax.lax.stop_gradient(blocks[t - 1])  # (B, L_y)
            w = jnp.clip(e_prev - tau, 0.0, w_max)         # (B, L_y)
            weights = 1.0 + lambda_w * w                    # (B, L_y)
            L_cont_focus.append(
                (weights * blocks[t]).sum() / weights.sum()
            )

    # Beta weights: linear ramp β_t = t / Σ(1..T) (1-indexed)
    beta_sum = sum(range(1, T + 1))  # T*(T+1)/2
    betas    = [(t + 1) / beta_sum for t in range(T)]  # [1/S, 2/S, ..., T/S]

    L_cont_total = sum(betas[t] * L_cont_focus[t] for t in range(T))

    # Monotonicity regularizer: penalize regression across passes
    L_mono = sum(
        jnp.maximum(0.0, L_cont_base[t] - L_cont_base[t - 1] + gamma)
        for t in range(1, T)
    ) if T > 1 else jnp.zeros(())

    total = L_src + lambda_cont * L_cont_total + lambda_mono * L_mono

    aux = {
        'L_src':         L_src,
        'L_cont_base':   L_cont_base,
        'L_cont_focus':  L_cont_focus,
        'L_mono':        L_mono,
        'L_cont_pass1':  L_cont_base[0],
        'L_cont_passT':  L_cont_base[-1],
    }
    return total, aux


# ---------------------------------------------------------------------------
# Training step factory
# ---------------------------------------------------------------------------

def make_train_step_stage1(hp: dict):
    """JIT-compiled train step. N, L_y, T are static to avoid retracing."""
    L_S         = hp['L_S']
    lambda_cont = hp['lambda_cont']
    lambda_w    = hp['lambda_w']
    lambda_mono = hp['lambda_mono']
    tau         = hp['tau']
    w_max       = hp['w_max']
    gamma       = hp['gamma']
    grad_clip   = hp['grad_clip']
    optimizer   = hp.get('optimizer', 'adamw')
    use_grok    = (optimizer == 'grokadamw')

    def _loss(model, tokens, mask, N, L_y, T):
        return loss_fn_stage1(
            model, tokens, mask, L_S, N, L_y, T,
            lambda_cont, lambda_w, lambda_mono, tau, w_max, gamma,
        )

    @jax.jit(static_argnums=(4, 5, 6))
    def train_step(model, opt_state, tokens, mask, N, L_y, T, step, lr):
        params = eqx.filter(model, eqx.is_array)
        (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
            model, tokens, mask, N, L_y, T)
        grads_arr = eqx.filter(grads, eqx.is_array)
        grads_arr = clip_grads(grads_arr, grad_clip)
        if use_grok:
            new_params, new_opt = grok_adam_update(
                params, grads_arr, opt_state, lr,
                rho=hp.get('grok_rho', 0.9), wd=hp['wd'],
                step=step, batch_size=hp['B'],
            )
        else:
            new_params, new_opt = adam_update(
                params, grads_arr, opt_state, lr,
                wd=hp['wd'], step=step,
            )
        deltas    = jax.tree.map(lambda np_, p: np_ - p, new_params, params)
        new_model = eqx.apply_updates(model, deltas)
        return new_model, new_opt, loss, aux

    return train_step


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(hp: dict, log_base: str = 'logs'):
    key = jax.random.PRNGKey(hp['seed'])
    key, mkey = jax.random.split(key)

    model     = build_model(hp, mkey)
    opt_state = init_opt_state(model, optimizer=hp.get('optimizer', 'adamw'))

    L_S          = hp['L_S']
    N_set        = hp['N_set']
    L_y_schedule = hp['L_y_schedule']
    T            = hp['T']
    B            = hp['B']
    n_steps      = hp['n_steps']
    alpha        = hp['alpha']
    V_chain      = hp['V_chain']

    run_dir  = setup_run_dir(log_base, 'stage1')
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    raw_log_path   = os.path.join(run_dir, 'train.log')
    train_log_path = os.path.join(run_dir, 'train.jsonl')
    raw_log_f      = open(raw_log_path,   'w', buffering=1)
    train_log_f    = open(train_log_path, 'w', buffering=1)

    def _log(msg):
        tqdm.write(msg)
        raw_log_f.write(msg + '\n')
        raw_log_f.flush()

    def _jlog(record):
        train_log_f.write(json.dumps(record) + '\n')
        train_log_f.flush()

    with open(os.path.join(run_dir, 'hparams.json'), 'w') as f:
        json.dump(hp, f, indent=2)

    # Precompute masks for all (N, L_y) combos; pad to L_max
    L_y_set = sorted({v for _, v in L_y_schedule})
    N_max   = max(N_set)
    L_y_max = max(L_y_set)
    block_max = N_max + 2 + L_y_max
    L_max   = L_S + T * block_max

    mask_cache = {}
    for N in N_set:
        for L_y in L_y_set:
            raw = make_mask_stage1(L_S, N, L_y, T)  # (L_cur, L_cur)
            L_cur = raw.shape[0]
            m = np.full((L_max, L_max), -1e9, dtype=np.float32)
            m[:L_cur, :L_cur] = raw
            mask_cache[(N, L_y)] = m

    train_step_fn = make_train_step_stage1(hp)

    pcount = count_params(model)
    _log(f'\n=== Training stage1 | run_dir={run_dir} ===')
    _log(f'  Params: {pcount["total"]:,}  (embed={pcount["embedding"]:,}, blocks={pcount["blocks"]:,})')
    _log(f'  T={T} passes  N_set={N_set}  L_y curriculum={L_y_schedule}')
    _log(f'  lambda_cont={hp["lambda_cont"]}  lambda_w={hp["lambda_w"]}  lambda_mono={hp["lambda_mono"]}')
    _log(f'  Steps={n_steps}  Batch={B}  Optimizer={hp.get("optimizer","adamw")}')
    _log(f'  Logs  -> {run_dir}/train.log')
    _log(f'  JSONL -> {run_dir}/train.jsonl')

    # Chain pool
    pool_size = hp.get('chain_pool_size', 8)
    rng       = np.random.default_rng(hp['seed'] + 1)
    chain_pool = make_chain_pool(rng, pool_size, V_chain, alpha)
    _log(f'  Chain pool: K={pool_size} chains (alpha={alpha})')

    # Validation file
    VAL_PATH = 'datasets/1.txt'
    val_lines: list[bytes] = []
    if os.path.exists(VAL_PATH):
        try:
            val_lines = _load_txt_lines(VAL_PATH)
            _log(f'  Val file : {VAL_PATH}  ({len(val_lines)} lines)')
        except Exception as e:
            _log(f'  [val] skipping {VAL_PATH}: {e}')

    history: dict[str, list] = {
        'step': [], 'loss': [], 'l_src': [],
        'l_cont': [], 'lr': [], 'L_y': [], 'N': [], 'val_match': [],
    }

    _step_counter = [0]

    def _gen():
        s   = _step_counter[0]
        lyr = get_L_y(s, L_y_schedule)
        n   = int(rng.choice(N_set))
        arr = np_make_stage1_batch(rng, B, V_chain, L_S, lyr, n, T, alpha,
                                   chain_pool=chain_pool)
        # Pad to L_max
        L_cur = arr.shape[1]
        if L_cur < L_max:
            pad = np.zeros((B, L_max - L_cur), dtype=np.int32)
            arr = np.concatenate([arr, pad], axis=1)
        _step_counter[0] += 1
        return (n, lyr, arr)

    prefetcher = BatchPrefetcher(_gen, maxsize=8)

    t0         = time.time()
    log_every  = 100
    val_every  = 1_000
    ckpt_every = 2_000

    pbar = tqdm(range(1, n_steps + 1), desc='stage1', unit='step',
                dynamic_ncols=True, file=sys.stdout)

    for step in pbar:
        N, L_y, np_tokens = prefetcher.get()
        tokens = jnp.array(np_tokens)
        mask   = jnp.array(mask_cache[(N, L_y)])

        lr = lr_schedule(step, hp)
        model, opt_state, loss, aux = train_step_fn(
            model, opt_state, tokens, mask, N, L_y, T, step, lr)

        loss_f  = float(loss)
        l_src_f = float(aux['L_src'])
        l_cont1 = float(aux['L_cont_pass1'])
        l_contT = float(aux['L_cont_passT'])
        l_mono_f = float(aux['L_mono'])

        L_y_vals = sorted({v for _, v in L_y_schedule})
        phase = ['easy', 'med', 'hard'][min(L_y_vals.index(L_y) if L_y in L_y_vals else 0, 2)]

        pbar.set_postfix(
            loss=f'{loss_f:.3f}',
            src=f'{l_src_f:.3f}',
            p1=f'{l_cont1:.3f}',
            pT=f'{l_contT:.3f}',
            N=N, L_y=L_y,
            lr=f'{lr:.1e}',
            refresh=False,
        )

        if step % log_every == 0:
            elapsed = time.time() - t0
            rec = dict(step=step, loss=loss_f, l_src=l_src_f,
                       l_cont_pass1=l_cont1, l_cont_passT=l_contT,
                       lr=lr, L_y=L_y, N=N, phase=phase, elapsed=elapsed,
                       l_cont_base=[float(x) for x in aux['L_cont_base']],
                       l_mono=l_mono_f)
            _jlog(rec)
            _log(f'  step={step:5d}/{n_steps}  L_y={L_y:3d}  N={N:2d}  '
                 f'loss={loss_f:.4f}  src={l_src_f:.4f}  '
                 f'p1={l_cont1:.4f}  pT={l_contT:.4f}  '
                 f'mono={l_mono_f:.4f}  lr={lr:.2e}  [{phase}]  {elapsed:.0f}s')

            history['step'].append(step)
            history['loss'].append(loss_f)
            history['l_src'].append(l_src_f)
            history['l_cont'].append(l_contT)   # track last pass
            history['lr'].append(lr)
            history['L_y'].append(L_y)
            history['N'].append(N)
            history['val_match'].append(None)

        if step % val_every == 0:
            # AR eval using stage-0 style decode (single pass of memory, uses last M^T)
            _N_ar  = N_set[-1]
            _ar_rng = np.random.default_rng(step)
            _ar_correct = _ar_total = 0
            from kvmem.data import make_mask_stage0

            for _t in range(32):
                _ki = int(_ar_rng.integers(0, len(chain_pool)))
                _T  = chain_pool[_ki]
                _start = int(_ar_rng.integers(0, V_chain))
                _x_raw = []
                s = _start
                for _ in range(L_S):
                    s = _ar_rng.choice(V_chain, p=_T[s])
                    _x_raw.append(s)
                _term = _x_raw[-1]
                _y_raw = []
                s = _term
                for _ in range(17):
                    s = _ar_rng.choice(V_chain, p=_T[s])
                    _y_raw.append(s)
                _x_S   = [v + DATA_LO for v in _x_raw]
                _y_true = [v + DATA_LO for v in _y_raw]
                _warmup = _y_true[:1]
                _gen = _decode(model, _x_S, _N_ar, _warmup,
                               max_len=16, temperature=0.0, seed=_t, stop_newline=False)
                _ar_correct += sum(a == b for a, b in zip(_gen[1:], _y_true[1:17]))
                _ar_total   += 16

            ar_acc    = 100 * _ar_correct / _ar_total
            ar_random = 100 / V_chain
            _log(f'  [ar]   step={step:5d}  AR acc={ar_acc:.1f}%  (random={ar_random:.1f}%  N={_N_ar})')
            _jlog(dict(step=step, ar_acc=ar_acc, ar_random=ar_random))

            # NLL on 1.txt (stage-0 mask, last-pass memory)
            if val_lines:
                from kvmem.data import make_mask_stage0 as _ms0
                val_nlls = []
                for vline in val_lines:
                    if len(vline) < 4:
                        continue
                    x_S_v = list(vline)
                    L_S_v = len(x_S_v)
                    L_y_v = min(16, len(x_S_v))
                    mem_v = [STX] + [NUL] * _N_ar + [ETX]
                    y_v   = list(vline[:L_y_v])
                    seq   = x_S_v + mem_v + y_v
                    msk   = jnp.array(_ms0(L_S_v, _N_ar, L_y_v))
                    tok   = jnp.array(seq, dtype=jnp.int32)
                    logits = model(tok, msk)
                    lp     = jax.nn.log_softmax(logits[:-1], axis=-1)
                    ETX_v  = L_S_v + 1 + _N_ar
                    for k in range(L_y_v):
                        pos = ETX_v + k
                        if pos < len(seq) - 1:
                            val_nlls.append(-float(lp[pos, seq[pos + 1]]))
                val_nll = float(np.mean(val_nlls)) if val_nlls else float('nan')
                if history['val_match']:
                    history['val_match'][-1] = val_nll
                _jlog(dict(step=step, val_nll=val_nll))
                _log(f'  [val]  step={step:5d}  1.txt NLL={val_nll:.4f}  (N={_N_ar})')

        if step % ckpt_every == 0 or step == n_steps:
            ckpt_path = os.path.join(ckpt_dir, f'stage1_step{step}')
            save_checkpoint(ckpt_path, model, step, hp)
            _log(f'  [ckpt] {ckpt_path}')

    pbar.close()
    _log(f'\nDone. Total time: {time.time()-t0:.0f}s')
    _log(f'Run dir: {run_dir}')
    train_log_f.close()
    raw_log_f.close()
    return model, run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_ckpt(ckpt_arg: str) -> str:
    if os.path.isfile(ckpt_arg + '.eqx'):
        return ckpt_arg
    ckpt_subdir = os.path.join(ckpt_arg, 'checkpoints')
    if os.path.isdir(ckpt_subdir):
        eqx_files = sorted(
            f[:-4] for f in os.listdir(ckpt_subdir) if f.endswith('.eqx'))
        if eqx_files:
            return os.path.join(ckpt_subdir, eqx_files[-1])
    raise FileNotFoundError(f'Cannot resolve checkpoint from: {ckpt_arg!r}')


def main():
    parser = argparse.ArgumentParser(prog='kvmem.stage1')
    sub    = parser.add_subparsers(dest='cmd', required=True)

    # --- train ---
    p_train = sub.add_parser('train')
    p_train.add_argument('--passes',     type=int,   default=None,
                         help='Number of refinement passes T (default: 4)')
    p_train.add_argument('--steps',      type=int,   default=None)
    p_train.add_argument('--log-dir',    type=str,   default='logs')
    p_train.add_argument('--seed',       type=int,   default=42)
    p_train.add_argument('--optimizer',  type=str,   default='adamw',
                         choices=['adamw', 'grokadamw'])
    p_train.add_argument('--grok-rho',   type=float, default=0.9)
    p_train.add_argument('--lambda-w',   type=float, default=None,
                         help='Hard-position focus strength (default: 2.0)')
    p_train.add_argument('--lambda-mono',type=float, default=None,
                         help='Monotonicity regularizer weight (default: 0.1)')
    p_train.add_argument('--no-focus',   action='store_true',
                         help='Disable focused loss (vanilla multi-pass, lambda_w=0)')

    # --- test ---
    p_test = sub.add_parser('test',
        help='Per-line + whole-file continuation test on a text file.')
    p_test.add_argument('--ckpt', required=True)
    p_test.add_argument('--fatihah', type=str, default=FATIHAH_PATH)
    p_test.add_argument('--mem-size', type=int, default=8)
    p_test.add_argument('--warmup',   type=int, default=4)
    p_test.add_argument('--temp',     type=float, default=0.0)
    p_test.add_argument('--seed',     type=int,   default=0)

    args = parser.parse_args()

    if args.cmd == 'train':
        hp = dict(DEFAULT_HPARAMS)
        hp['seed']      = args.seed
        hp['optimizer'] = args.optimizer
        hp['grok_rho']  = args.grok_rho
        if args.passes:
            hp['T'] = args.passes
        if args.steps:
            hp['n_steps'] = args.steps
        if args.lambda_w is not None:
            hp['lambda_w'] = args.lambda_w
        if args.lambda_mono is not None:
            hp['lambda_mono'] = args.lambda_mono
        if args.no_focus:
            hp['lambda_w'] = 0.0

        model, run_dir = train(hp, log_base=args.log_dir)

        # Auto-test
        if os.path.exists(FATIHAH_PATH):
            print('\n\n' + '═' * 62)
            print(f'AUTO-TEST  suratalfatihah.txt  [stage1]')
            print('═' * 62)
            run_test(model, hp, txt_path=FATIHAH_PATH, N=hp['N_set'][-1],
                     warmup_bytes=4, temperature=0.0)

    elif args.cmd == 'test':
        ckpt  = _resolve_ckpt(args.ckpt)
        key   = jax.random.PRNGKey(0)
        model, hp = load_checkpoint(ckpt, key)
        run_test(model, hp,
                 txt_path=args.fatihah,
                 N=args.mem_size,
                 warmup_bytes=args.warmup,
                 temperature=args.temp,
                 seed=args.seed)


if __name__ == '__main__':
    main()
