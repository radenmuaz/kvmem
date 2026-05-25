"""
kvmem/stage0.py — Stage 0: Single-pass NTP through the KV bottleneck.

Usage:
    python -m kvmem.stage0 train [--baseline] [--steps N] [--ckpt-dir DIR]
    python -m kvmem.stage0 eval  --ckpt PATH [--baseline]
    python -m kvmem.stage0 infer --ckpt PATH [--line N] [--all-lines]
                                  [--mem-size N] [--warmup W] [--temp T]
    python -m kvmem.stage0 infer --ckpt PATH --file PATH
                                  [--mem-size N] [--warmup-bytes W]
                                  [--warmup-text STR] [--temp T]

Single file — all model, training, eval, and inference code lives here.
Imports from kvmem.data only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
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
    build_mask_cache,
    chain_entropy_bits,
    load_fatihah,
    load_text_lines,
    make_batch,
    make_eval_batches,
    make_mask_baseline,
    make_mask_stage0,
    np_make_batch,
    np_make_baseline_batch,
    np_make_eval_batches,
    sample_transition_matrix,
    stationary_distribution,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_HPARAMS = dict(
    V          = 256,       # full byte vocab for embedding
    V_chain    = 224,       # Markov chain states, remapped to [0x20, 0xFF]
    L_S        = 128,       # source length (fixed)
    N_set      = [2, 4, 8, 16, 32],
    # Curriculum: (start_step, L_y)
    L_y_schedule = [(0, 8), (15_000, 32), (35_000, 128)],
    d          = 128,
    n_layers   = 4,
    n_heads    = 4,         # d_head = 32
    d_ff       = 512,
    lambda_cont= 2.0,
    B          = 64,
    lr_max     = 1e-3,
    lr_min     = 1e-5,
    warmup_steps = 1_000,
    n_steps    = 50_000,
    grad_clip  = 1.0,
    wd         = 0.01,
    alpha      = 0.5,       # Dirichlet concentration for chain sampling
    seed       = 42,
)

FATIHAH_PATH = 'datasets/quran_uthmani.txt'


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
# Model
# ---------------------------------------------------------------------------

class MHAttention(eqx.Module):
    W_Q: jax.Array   # (d, d)
    W_K: jax.Array
    W_V: jax.Array
    W_O: jax.Array
    n_heads: int = eqx.field(static=True)
    d_head:  int = eqx.field(static=True)

    def __init__(self, d: int, n_heads: int, key: jax.Array):
        self.n_heads = n_heads
        self.d_head  = d // n_heads
        k1, k2, k3, k4 = jax.random.split(key, 4)
        scale = math.sqrt(2.0 / d)
        self.W_Q = jax.random.normal(k1, (d, d)) * scale
        self.W_K = jax.random.normal(k2, (d, d)) * scale
        self.W_V = jax.random.normal(k3, (d, d)) * scale
        self.W_O = jax.random.normal(k4, (d, d)) * scale

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        # x: (L, d), mask: (L, L) -> (L, d)
        L, d = x.shape
        H, dh = self.n_heads, self.d_head

        Q = (x @ self.W_Q.T).reshape(L, H, dh).transpose(1, 0, 2)  # (H, L, dh)
        K = (x @ self.W_K.T).reshape(L, H, dh).transpose(1, 0, 2)
        V = (x @ self.W_V.T).reshape(L, H, dh).transpose(1, 0, 2)

        attn = (Q @ K.transpose(0, 2, 1)) * (dh ** -0.5) + mask[None]  # (H, L, L)
        attn = jax.nn.softmax(attn, axis=-1)
        out  = (attn @ V).transpose(1, 0, 2).reshape(L, d)              # (L, d)
        return out @ self.W_O.T


class FFN(eqx.Module):
    W1: jax.Array   # (d_ff, d)
    W2: jax.Array   # (d, d_ff)

    def __init__(self, d: int, d_ff: int, key: jax.Array):
        k1, k2 = jax.random.split(key)
        scale1  = math.sqrt(2.0 / d)
        scale2  = math.sqrt(2.0 / d_ff)
        self.W1 = jax.random.normal(k1, (d_ff, d)) * scale1
        self.W2 = jax.random.normal(k2, (d, d_ff)) * scale2

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.gelu(x @ self.W1.T) @ self.W2.T


class TransformerBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn:  MHAttention
    norm2: eqx.nn.LayerNorm
    ffn:   FFN

    def __init__(self, d: int, n_heads: int, d_ff: int, key: jax.Array):
        k1, k2 = jax.random.split(key)
        self.norm1 = eqx.nn.LayerNorm(d)
        self.attn  = MHAttention(d, n_heads, k1)
        self.norm2 = eqx.nn.LayerNorm(d)
        self.ffn   = FFN(d, d_ff, k2)

    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        x = x + self.attn(jax.vmap(self.norm1)(x), mask)
        x = x + jax.vmap(self.ffn)(jax.vmap(self.norm2)(x))
        return x


class KVMemModel(eqx.Module):
    embed:    eqx.nn.Embedding
    blocks:   list
    norm_out: eqx.nn.LayerNorm
    W_out:    jax.Array          # (V, d) untied

    def __init__(self, V: int, d: int, n_layers: int, n_heads: int,
                 d_ff: int, key: jax.Array):
        keys = jax.random.split(key, n_layers + 2)
        self.embed    = eqx.nn.Embedding(V, d, key=keys[0])
        self.blocks   = [TransformerBlock(d, n_heads, d_ff, keys[1 + i])
                         for i in range(n_layers)]
        self.norm_out = eqx.nn.LayerNorm(d)
        scale         = math.sqrt(2.0 / d)
        self.W_out    = jax.random.normal(keys[-1], (V, d)) * scale

    def __call__(self, tokens: jax.Array, mask: jax.Array) -> jax.Array:
        """tokens: (L,) int32, mask: (L, L) -> logits (L, V)"""
        x = jax.vmap(self.embed)(tokens)   # embed each token scalar separately
        for block in self.blocks:
            x = block(x, mask)
        x = jax.vmap(self.norm_out)(x)
        return x @ self.W_out.T

    def hidden(self, tokens: jax.Array, mask: jax.Array) -> jax.Array:
        """Return final hidden states (L, d) for diagnostics."""
        x = jax.vmap(self.embed)(tokens)
        for block in self.blocks:
            x = block(x, mask)
        return jax.vmap(self.norm_out)(x)


def build_model(hp: dict, key: jax.Array) -> KVMemModel:
    return KVMemModel(
        V=hp['V'], d=hp['d'], n_layers=hp['n_layers'],
        n_heads=hp['n_heads'], d_ff=hp['d_ff'], key=key,
    )


def count_params(model: KVMemModel) -> dict:
    arrays = eqx.filter(model, eqx.is_array)
    leaves = jax.tree.leaves(arrays)
    total  = sum(x.size for x in leaves)
    embed  = model.embed.weight.size
    blocks = sum(
        sum(x.size for x in jax.tree.leaves(eqx.filter(b, eqx.is_array)))
        for b in model.blocks
    )
    head   = model.W_out.size + sum(
        x.size for x in jax.tree.leaves(eqx.filter(model.norm_out, eqx.is_array))
    )
    return {'total': total, 'embedding': embed, 'blocks': blocks, 'output_head': head}


# ---------------------------------------------------------------------------
# Optimizer (hand-rolled AdamW, no optax)
# ---------------------------------------------------------------------------

def init_opt_state(model):
    params = eqx.filter(model, eqx.is_array)
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    return (m, v)


def lr_schedule(step: int, hp: dict) -> float:
    step   = float(step)
    w      = hp['warmup_steps']
    lr_max = hp['lr_max']
    lr_min = hp['lr_min']
    n      = hp['n_steps']
    if step < w:
        return lr_max * step / w
    frac = min((step - w) / (n - w), 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * frac))


def clip_grads(grads, max_norm: float = 1.0):
    leaves = jax.tree.leaves(grads)
    norm   = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
    scale  = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree.map(lambda g: g * scale, grads)


def adam_update(params, grads, opt_state, lr: float,
                b1: float = 0.9, b2: float = 0.999,
                eps: float = 1e-8, wd: float = 0.01,
                step: int = 1):
    m, v = opt_state
    m    = jax.tree.map(lambda m_, g: b1 * m_ + (1 - b1) * g,     m, grads)
    v    = jax.tree.map(lambda v_, g: b2 * v_ + (1 - b2) * g ** 2, v, grads)
    bc1  = 1.0 - b1 ** step
    bc2  = 1.0 - b2 ** step
    mh   = jax.tree.map(lambda m_: m_ / bc1, m)
    vh   = jax.tree.map(lambda v_: v_ / bc2, v)
    new_params = jax.tree.map(
        lambda p, mh_, vh_: p - lr * (mh_ / (jnp.sqrt(vh_) + eps) + wd * p),
        params, mh, vh,
    )
    return new_params, (m, v)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _nll_matrix(model: KVMemModel, tokens: jax.Array,
                mask: jax.Array) -> jax.Array:
    """Compute per-token NLL matrix (B, L-1)."""
    B, L   = tokens.shape
    logits = jax.vmap(lambda tok: model(tok, mask))(tokens)      # (B, L, V)
    lp     = jax.nn.log_softmax(logits[:, :-1], axis=-1)         # (B, L-1, V)
    tgts   = tokens[:, 1:]                                        # (B, L-1)
    idx_b  = jnp.arange(B)[:, None]
    idx_t  = jnp.arange(L - 1)[None, :]
    return -lp[idx_b, idx_t, tgts]                                # (B, L-1)


def loss_fn(model: KVMemModel, tokens: jax.Array, mask: jax.Array,
            L_S: int, N: int, lambda_cont: float) -> tuple:
    """
    Stage-0 KV bottleneck loss.
    tokens: (B, L_S + 2 + N + L_y)
    Returns (total_loss, (L_src, L_cont))
    """
    B, L    = tokens.shape
    nll     = _nll_matrix(model, tokens, mask)                    # (B, L-1)
    pos     = jnp.arange(L - 1)
    ETX_pos = L_S + 1 + N

    mask_src  = (pos <= L_S - 2).astype(jnp.float32)
    mask_cont = (pos >= ETX_pos).astype(jnp.float32)

    def wmean(x, m):
        return jnp.sum(x * m[None, :], axis=-1) / (m.sum() + 1e-8)

    L_src  = jnp.mean(wmean(nll, mask_src))
    L_cont = jnp.mean(wmean(nll, mask_cont))
    total  = L_src + lambda_cont * L_cont
    return total, (L_src, L_cont)


def loss_fn_baseline(model: KVMemModel, tokens: jax.Array, mask: jax.Array,
                     L_S: int, lambda_cont: float) -> tuple:
    """
    Backprop baseline loss (no bottleneck — Y sees S directly).
    tokens: (B, L_S + L_y)
    """
    B, L = tokens.shape
    nll  = _nll_matrix(model, tokens, mask)
    pos  = jnp.arange(L - 1)

    mask_src  = (pos <= L_S - 2).astype(jnp.float32)
    mask_cont = (pos >= L_S - 1).astype(jnp.float32)

    def wmean(x, m):
        return jnp.sum(x * m[None, :], axis=-1) / (m.sum() + 1e-8)

    L_src  = jnp.mean(wmean(nll, mask_src))
    L_cont = jnp.mean(wmean(nll, mask_cont))
    total  = L_src + lambda_cont * L_cont
    return total, (L_src, L_cont)


# ---------------------------------------------------------------------------
# Training step (not jitted — jit at call site with closure over hp)
# ---------------------------------------------------------------------------

def make_train_step(hp: dict, baseline: bool = False):
    """Return a jit-compiled train_step function closed over hp."""
    lambda_cont = hp['lambda_cont']
    L_S         = hp['L_S']
    grad_clip   = hp['grad_clip']

    if baseline:
        def _loss(model, tokens, mask, N_unused):
            return loss_fn_baseline(model, tokens, mask, L_S, lambda_cont)
    else:
        def _loss(model, tokens, mask, N):
            return loss_fn(model, tokens, mask, L_S, N, lambda_cont)

    @jax.jit
    def train_step(model, opt_state, tokens, mask, N, step, lr):
        params  = eqx.filter(model, eqx.is_array)
        (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
            model, tokens, mask, N)
        grads_arr = eqx.filter(grads, eqx.is_array)
        grads_arr = clip_grads(grads_arr, grad_clip)
        new_params, new_opt = adam_update(
            params, grads_arr, opt_state, lr,
            wd=hp['wd'], step=step,
        )
        # adam_update returns absolute new params; compute delta for apply_updates
        deltas    = jax.tree.map(lambda np_, p: np_ - p, new_params, params)
        new_model = eqx.apply_updates(model, deltas)
        return new_model, new_opt, loss, aux

    return train_step


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def setup_run_dir(base: str, tag: str) -> str:
    """Create logs/<tag>_<timestamp>/ and return the path."""
    ts  = time.strftime('%Y%m%d_%H%M%S')
    run = os.path.join(base, f'{tag}_{ts}')
    os.makedirs(run, exist_ok=True)
    return run


def save_checkpoint(path: str, model: KVMemModel, step: int, hp: dict):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    eqx.tree_serialise_leaves(path + '.eqx', model)
    with open(path + '.json', 'w') as f:
        json.dump({**hp, 'step': step}, f, indent=2)


def load_checkpoint(path: str, key: jax.Array) -> tuple[KVMemModel, dict]:
    with open(path + '.json') as f:
        hp = json.load(f)
    template = build_model(hp, key)
    model    = eqx.tree_deserialise_leaves(path + '.eqx', template)
    return model, hp


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(hp: dict, baseline: bool = False, log_base: str = 'logs'):
    key = jax.random.PRNGKey(hp['seed'])
    key, mkey = jax.random.split(key)

    model     = build_model(hp, mkey)
    opt_state = init_opt_state(model)

    L_S          = hp['L_S']
    N_set        = hp['N_set']
    L_y_schedule = hp['L_y_schedule']
    B            = hp['B']
    n_steps      = hp['n_steps']
    alpha        = hp['alpha']
    V_chain      = hp['V_chain']

    tag = 'baseline' if baseline else 'stage0'

    # ---- run directory: logs/<tag>_<timestamp>/ ----
    run_dir  = setup_run_dir(log_base, tag)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Streamed log files
    train_log_path = os.path.join(run_dir, 'train.jsonl')
    raw_log_path   = os.path.join(run_dir, 'train.log')
    train_log_f    = open(train_log_path, 'w', buffering=1)   # line-buffered
    raw_log_f      = open(raw_log_path,   'w', buffering=1)

    def _log(msg: str):
        """Write to raw log AND stdout (tqdm-safe)."""
        tqdm.write(msg)
        raw_log_f.write(msg + '\n')
        raw_log_f.flush()

    def _jlog(record: dict):
        """Append one JSON record to train.jsonl."""
        train_log_f.write(json.dumps(record) + '\n')
        train_log_f.flush()

    # Save hparams
    with open(os.path.join(run_dir, 'hparams.json'), 'w') as f:
        json.dump({**hp, 'baseline': baseline}, f, indent=2)

    # Precompute all masks
    L_y_set = sorted({v for _, v in L_y_schedule})
    if baseline:
        mask_cache = {L_y: make_mask_baseline(L_S, L_y) for L_y in L_y_set}
    else:
        mask_cache = build_mask_cache(L_S, N_set, L_y_set)

    train_step_fn = make_train_step(hp, baseline=baseline)

    pcount = count_params(model)
    header = (
        f'\n=== Training {tag} | run_dir={run_dir} ===\n'
        f'  Params: {pcount["total"]:,}  '
        f'(embed={pcount["embedding"]:,}, blocks={pcount["blocks"]:,})\n'
    )
    if not baseline:
        header += '  KV floats: ' + ' | '.join(
            f'N={N}: {2*hp["n_layers"]*N*hp["d"]:,} '
            f'({100*2*hp["n_layers"]*N*hp["d"]/pcount["total"]:.1f}%)'
            for N in N_set) + '\n'
    header += f'  Steps: {n_steps:,}  Batch: {B}  L_y curriculum: {L_y_schedule}\n'
    header += f'  Logs  -> {run_dir}/train.log\n'
    header += f'  JSONL -> {run_dir}/train.jsonl  (tail -f to follow)\n'
    _log(header)

    log_every  = 100
    plot_every = 2_000
    ckpt_every = 10_000

    # rolling history for live plot
    history: dict[str, list] = {'step': [], 'loss': [], 'l_src': [], 'l_cont': [],
                                  'lr': [], 'L_y': [], 'N': []}
    t0   = time.time()
    rng  = np.random.default_rng(hp['seed'] + 1)

    # Prefetch queue: background thread generates numpy batches
    # We generate (N, L_y, batch) tuples; the main thread pulls and forwards to JAX.
    _step_counter = [0]

    def _gen():
        s   = _step_counter[0]
        lyr = get_L_y(s, L_y_schedule)
        n   = N_set[0] if baseline else rng.choice(N_set)
        if baseline:
            arr = np_make_baseline_batch(rng, B, V_chain, L_S, lyr, alpha)
        else:
            arr = np_make_batch(rng, B, V_chain, L_S, lyr, n, alpha)
        _step_counter[0] += 1
        return (n, lyr, arr)

    prefetcher = BatchPrefetcher(_gen, maxsize=8)

    pbar = tqdm(range(1, n_steps + 1), desc=tag, unit='step',
                dynamic_ncols=True, file=sys.stdout)

    for step in pbar:
        N, L_y, np_tokens = prefetcher.get()
        tokens = jnp.array(np_tokens)
        if baseline:
            mask = jnp.array(mask_cache[L_y])
        else:
            mask = jnp.array(mask_cache[(N, L_y)])

        lr = lr_schedule(step, hp)
        model, opt_state, loss, (l_src, l_cont) = train_step_fn(
            model, opt_state, tokens, mask, N, step, lr)

        loss_f   = float(loss)
        l_src_f  = float(l_src)
        l_cont_f = float(l_cont)

        # tqdm postfix (always live)
        phase = 'easy' if L_y == 8 else 'med' if L_y == 32 else 'hard'
        pbar.set_postfix(
            loss=f'{loss_f:.3f}',
            src=f'{l_src_f:.3f}',
            cont=f'{l_cont_f:.3f}',
            N=N, L_y=L_y, phase=phase,
            lr=f'{lr:.1e}',
            refresh=False,
        )

        if step % log_every == 0:
            elapsed = time.time() - t0
            record = dict(step=step, loss=loss_f, l_src=l_src_f,
                          l_cont=l_cont_f, lr=lr, L_y=L_y, N=N,
                          phase=phase, elapsed=elapsed)
            _jlog(record)
            _log(f'  step={step:5d}/{n_steps}  L_y={L_y:3d}  N={N:2d}  '
                 f'loss={loss_f:.4f}  src={l_src_f:.4f}  cont={l_cont_f:.4f}  '
                 f'lr={lr:.2e}  [{phase}]  {elapsed:.0f}s')

            history['step'].append(step)
            history['loss'].append(loss_f)
            history['l_src'].append(l_src_f)
            history['l_cont'].append(l_cont_f)
            history['lr'].append(lr)
            history['L_y'].append(L_y)
            history['N'].append(N)

        if step % plot_every == 0 or step == n_steps:
            _plot_training(history, run_dir, L_y_schedule)

        if step % ckpt_every == 0 or step == n_steps:
            ckpt_path = os.path.join(ckpt_dir, f'{tag}_step{step}')
            save_checkpoint(ckpt_path, model, step, hp)
            _log(f'  [ckpt] {ckpt_path}')

    pbar.close()
    _log(f'\nDone. Total time: {time.time()-t0:.0f}s')
    _log(f'Run dir: {run_dir}')
    train_log_f.close()
    raw_log_f.close()
    return model, run_dir


# ---------------------------------------------------------------------------
# Live training plots
# ---------------------------------------------------------------------------

def _plot_training(history: dict, run_dir: str, L_y_schedule: list):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not history['step']:
        return

    steps   = history['step']
    loss    = history['loss']
    l_src   = history['l_src']
    l_cont  = history['l_cont']
    lr_vals = history['lr']

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(f'Stage 0 Training  (step {steps[-1]:,})', fontsize=13)

    # Phase boundary vertical lines
    phase_colors = {'easy': '#d4e8ff', 'med': '#d4ffd4', 'hard': '#ffd4d4'}
    phase_labels = {8: 'easy', 32: 'med', 128: 'hard'}

    def _phase_spans(ax):
        boundaries = [s for s, _ in L_y_schedule] + [steps[-1]]
        for i, (start, val) in enumerate(L_y_schedule):
            end   = boundaries[i + 1]
            label = phase_labels.get(val, str(val))
            color = phase_colors.get(label, '#eeeeee')
            ax.axvspan(start, end, alpha=0.15, color=color, label=f'L_y={val} ({label})')

    # --- total loss ---
    ax = axes[0, 0]
    _phase_spans(ax)
    ax.plot(steps, loss, lw=1.2, color='tab:blue', label='total loss')
    ax.set_title('Total loss')
    ax.set_xlabel('step')
    ax.set_ylabel('loss (nats)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- src vs cont loss ---
    ax = axes[0, 1]
    _phase_spans(ax)
    ax.plot(steps, l_src,  lw=1.2, color='tab:orange', label='L_src')
    ax.plot(steps, l_cont, lw=1.2, color='tab:green',  label='L_cont')
    ax.set_title('Src vs Cont NLL')
    ax.set_xlabel('step')
    ax.set_ylabel('NLL (nats)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- bpt (cont / log2) ---
    ax = axes[1, 0]
    _phase_spans(ax)
    bpt = [c / math.log(2) for c in l_cont]
    ax.plot(steps, bpt, lw=1.2, color='tab:purple', label='cont bpt')
    ax.axhline(math.log2(256), color='gray', ls='--', lw=0.8, label='uniform (8 bpt)')
    ax.set_title('Continuation bpt')
    ax.set_xlabel('step')
    ax.set_ylabel('bits/token')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- LR ---
    ax = axes[1, 1]
    ax.semilogy(steps, lr_vals, lw=1.2, color='tab:red', label='lr')
    ax.set_title('Learning rate')
    ax.set_xlabel('step')
    ax.set_ylabel('lr')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(run_dir, 'train_curves.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_bpt(model: KVMemModel, tokens: jax.Array, mask: jax.Array,
             L_S: int, N: int, baseline: bool = False) -> float:
    """Compute bits-per-token on continuation region."""
    B, L   = tokens.shape
    nll    = _nll_matrix(model, tokens, mask)         # (B, L-1)
    pos    = jnp.arange(L - 1)

    if baseline:
        mask_c = (pos >= L_S - 1).astype(jnp.float32)
    else:
        ETX_pos = L_S + 1 + N
        mask_c  = (pos >= ETX_pos).astype(jnp.float32)

    cont_nll = jnp.mean(
        jnp.sum(nll * mask_c[None, :], axis=-1) / (mask_c.sum() + 1e-8))
    return float(cont_nll / jnp.log(2.0))


def slot_diversity(model: KVMemModel, tokens_1d: jax.Array,
                   L_S: int, N: int) -> jax.Array:
    """
    Returns (N, N) cosine similarity matrix between memory slot hidden states.
    tokens_1d may be longer than L_S+2+N — we truncate to just the source+memory
    prefix so the mask shape matches.
    Off-diagonal average < 0.9 indicates healthy slot differentiation.
    """
    prefix_len = L_S + 2 + N                  # x_S + STX + NUL*N + ETX
    prefix     = tokens_1d[:prefix_len]        # (L_S+2+N,)
    mask       = jnp.array(make_mask_stage0(L_S, N, 0))
    h          = model.hidden(prefix, mask)    # (L_S+2+N, d)
    M_h        = h[L_S + 1: L_S + 1 + N]     # (N, d)
    norms      = jnp.linalg.norm(M_h, axis=-1, keepdims=True)
    M_n        = M_h / (norms + 1e-8)
    return M_n @ M_n.T                         # (N, N)


def run_eval(model: KVMemModel, hp: dict, baseline: bool = False,
             B_eval: int = 256, key: jax.Array | None = None):
    """Full eval sweep over all (N, L_y) combinations."""
    if key is None:
        key = jax.random.PRNGKey(0)

    L_S      = hp['L_S']
    N_set    = hp['N_set']
    L_y_set  = [v for _, v in hp['L_y_schedule']]
    V_chain  = hp['V_chain']
    alpha    = hp['alpha']

    tag = 'baseline' if baseline else 'stage0'
    print(f'\n=== Eval: {tag} ===')
    print(f'  {"N":>4}  {"L_y":>4}  {"matched":>8}  {"cross":>8}  {"uniform":>8}'
          f'  {"gain":>7}  {"penalty":>8}  {"eta":>6}')

    results = {}
    for L_y in L_y_set:
        N_iter = [None] if baseline else N_set
        for N in N_iter:
            key, ekey = jax.random.split(key)
            if baseline:
                from kvmem.data import make_mask_baseline as _mm
                mask   = jnp.array(_mm(L_S, L_y))
                # Build baseline eval batches: x_S + y, no memory
                batches = _build_baseline_eval(ekey, B_eval, L_S, L_y,
                                               V_chain, alpha)
                N_disp = '—'
                bpt_m  = eval_bpt(model, batches['matched'], mask, L_S, 0,
                                  baseline=True)
                bpt_c  = eval_bpt(model, batches['cross'],   mask, L_S, 0,
                                  baseline=True)
                bpt_u  = eval_bpt(model, batches['uniform'], mask, L_S, 0,
                                  baseline=True)
            else:
                mask   = jnp.array(make_mask_stage0(L_S, N, L_y))
                batches = make_eval_batches(ekey, B_eval, L_S, L_y, N,
                                            V_chain, alpha)
                N_disp = str(N)
                bpt_m  = eval_bpt(model, batches['matched'], mask, L_S, N)
                bpt_c  = eval_bpt(model, batches['cross'],   mask, L_S, N)
                bpt_u  = eval_bpt(model, batches['uniform'], mask, L_S, N)

            gain    = bpt_u - bpt_m
            penalty = bpt_c - bpt_u
            # eta: fraction of gain relative to best possible
            # Use bpt_u as the "no info" baseline; oracle = chain entropy
            # (We don't have oracle here; just report gain/penalty)
            scr     = L_S / N if N is not None else float('inf')
            print(f'  {N_disp:>4}  {L_y:>4}  {bpt_m:>8.4f}  {bpt_c:>8.4f}'
                  f'  {bpt_u:>8.4f}  {gain:>7.4f}  {penalty:>8.4f}  SCR={scr:.0f}')
            key_r = (N, L_y) if not baseline else ('baseline', L_y)
            results[key_r] = dict(bpt_matched=bpt_m, bpt_cross=bpt_c,
                                  bpt_uniform=bpt_u, gain=gain, penalty=penalty)

    return results


def _build_baseline_eval(key, B, L_S, L_y, V_chain, alpha):
    """Build matched/cross/uniform batches without memory tokens."""
    from kvmem.data import (walk_chain, _remap, sample_transition_matrix,
                             DATA_LO)

    def matched(k):
        k0, k1, k2 = jax.random.split(k, 3)
        T  = sample_transition_matrix(k0, V_chain, alpha)
        s  = jax.random.randint(k1, (), 0, V_chain)
        xs = _remap(walk_chain(k1, T, s, L_S), V_chain)
        y  = _remap(walk_chain(k2, T, xs[-1] - DATA_LO, L_y), V_chain)
        return jnp.concatenate([xs, y])

    def cross(k):
        k0, k1, k2, k3 = jax.random.split(k, 4)
        T1 = sample_transition_matrix(k0, V_chain, alpha)
        T2 = sample_transition_matrix(k1, V_chain, alpha)
        s1 = jax.random.randint(k2, (), 0, V_chain)
        s2 = jax.random.randint(k3, (), 0, V_chain)
        xs = _remap(walk_chain(k2, T1, s1, L_S), V_chain)
        y  = _remap(walk_chain(k3, T2, s2, L_y), V_chain)
        return jnp.concatenate([xs, y])

    def uniform(k):
        k0, k1, k2 = jax.random.split(k, 3)
        xs = jax.random.randint(k0, (L_S,), DATA_LO, DATA_LO + V_chain).astype(jnp.int32)
        T  = sample_transition_matrix(k1, V_chain, alpha)
        s  = jax.random.randint(k2, (), 0, V_chain)
        y  = _remap(walk_chain(k2, T, s, L_y), V_chain)
        return jnp.concatenate([xs, y])

    km, kc, ku = jax.random.split(key, 3)
    return {
        'matched': jax.vmap(matched)(jax.random.split(km, B)),
        'cross':   jax.vmap(cross)(jax.random.split(kc, B)),
        'uniform': jax.vmap(uniform)(jax.random.split(ku, B)),
    }


# ---------------------------------------------------------------------------
# Inference: single-verse completion
# ---------------------------------------------------------------------------

def _decode(model: KVMemModel, x_S: list[int], N: int, prompt: list[int],
            max_len: int, temperature: float, seed: int) -> list[int]:
    """
    Autoregressive decode from KV memory.
    x_S: source bytes (the memorized content)
    prompt: initial Y bytes (warmup)
    Grows [x_S | STX | NUL*N | ETX | prompt | ...] token by token.
    """
    mem_block = [STX] + [NUL] * N + [ETX]
    generated = list(prompt)
    key = jax.random.PRNGKey(seed)
    L_S = len(x_S)

    for _ in range(max_len):
        L_y  = len(generated)
        mask = jnp.array(make_mask_stage0(L_S, N, L_y))
        cur  = jnp.array(x_S + mem_block + generated, dtype=jnp.int32)
        logits = model(cur, mask)        # (L, V)
        nxt    = logits[-1]              # (V,)
        if temperature == 0.0:
            nb = int(jnp.argmax(nxt))
        else:
            key, sk = jax.random.split(key)
            nb = int(jax.random.choice(sk, 256,
                                        p=jax.nn.softmax(nxt / temperature)))
        generated.append(nb)
        # Stop at newline
        if nb == 0x0A and len(generated) > len(prompt) + 1:
            break

    return generated


def run_verse_inference(model: KVMemModel, hp: dict,
                        line: int = -1,
                        N: int = 8,
                        warmup_bytes: int = 4,
                        max_len: int = 300,
                        temperature: float = 0.0,
                        seed: int = 0,
                        fatihah_path: str = FATIHAH_PATH):
    """
    Memorize one ayah of Al-Fatihah, give first W bytes as warmup,
    attempt to complete the rest of the verse.
    """
    ayat = load_fatihah(fatihah_path)
    if line == -1:
        line = random.randint(0, 6)

    verse  = ayat[line]
    x_S    = list(verse)
    warmup = list(verse[:warmup_bytes])
    target = verse[warmup_bytes:]

    generated = _decode(model, x_S, N, warmup, max_len, temperature, seed)
    gen_after  = bytes(generated[warmup_bytes:])

    print(f'\n{"=" * 60}')
    print(f'Ayah {line}  (N={N}, warmup={warmup_bytes} bytes, '
          f'SCR={hp["L_S"]}/{N}={hp["L_S"]//N})')
    print(f'  Full verse : {verse.decode("utf-8", errors="replace")}')
    print(f'  Warmup     : {bytes(warmup).decode("utf-8", errors="replace")!r}')
    print(f'  Generated  : {bytes(generated).decode("utf-8", errors="replace")}')
    print(f'  Target tail: {target.decode("utf-8", errors="replace")}')
    if target:
        min_len = min(len(gen_after), len(target))
        matches = sum(a == b for a, b in zip(gen_after, target))
        print(f'  Byte match : {matches}/{min_len} '
              f'({100*matches/max(min_len,1):.1f}%)')


def run_all_verses(model: KVMemModel, hp: dict, N: int = 8,
                   warmup_bytes: int = 4, temperature: float = 0.0,
                   seed: int = 0, fatihah_path: str = FATIHAH_PATH):
    """Run verse inference for all 7 ayat."""
    for line in range(7):
        run_verse_inference(model, hp, line=line, N=N,
                            warmup_bytes=warmup_bytes,
                            temperature=temperature, seed=seed,
                            fatihah_path=fatihah_path)


# ---------------------------------------------------------------------------
# Inference: whole-file memorization
# ---------------------------------------------------------------------------

def run_file_inference(model: KVMemModel, hp: dict,
                       filepath: str,
                       N: int = 8,
                       warmup_text: str = '',
                       warmup_bytes_n: int = 4,
                       max_len: int = 300,
                       temperature: float = 0.0,
                       seed: int = 0):
    """
    Read entire file as x_S, compress into N KV slots,
    then complete from a warmup prompt.

    warmup_text: if non-empty, use as prompt (encoded to UTF-8 bytes).
    warmup_bytes_n: if warmup_text is empty, use first N bytes of file as prompt.
    """
    with open(filepath, 'rb') as f:
        file_bytes = f.read().rstrip(b'\n')

    bad = [hex(b) for b in file_bytes if b < DATA_LO]
    if bad:
        raise ValueError(f'File contains protocol bytes < 0x20: {bad[:5]}')

    x_S = list(file_bytes)
    L_S = len(x_S)

    if warmup_text:
        warmup = list(warmup_text.encode('utf-8'))
    else:
        warmup = list(file_bytes[:warmup_bytes_n])

    print(f'\n{"=" * 60}')
    print(f'File: {filepath}  ({L_S} bytes, N={N}, SCR={L_S}/{N}={L_S//N})')
    print(f'Warmup ({len(warmup)} bytes): '
          f'{bytes(warmup).decode("utf-8", errors="replace")!r}')

    generated = _decode(model, x_S, N, warmup, max_len, temperature, seed)
    gen_text  = bytes(generated).decode('utf-8', errors='replace')
    print(f'Generated: {gen_text}')


# ---------------------------------------------------------------------------
# Plotting (optional — skips gracefully if matplotlib unavailable)
# ---------------------------------------------------------------------------

def _plot_bpt_sweep(results: dict, L_y_set: list, N_set: list,
                    out_path: str = 'reports/stage0_bpt_sweep.png'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [plot] matplotlib not available, skipping')
        return

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig, axes = plt.subplots(1, len(L_y_set), figsize=(5 * len(L_y_set), 4),
                             sharey=False)
    if len(L_y_set) == 1:
        axes = [axes]

    for ax, L_y in zip(axes, L_y_set):
        bpt_m = [results.get((N, L_y), {}).get('bpt_matched', float('nan'))
                 for N in N_set]
        bpt_c = [results.get((N, L_y), {}).get('bpt_cross',   float('nan'))
                 for N in N_set]
        bpt_u = [results.get((N, L_y), {}).get('bpt_uniform', float('nan'))
                 for N in N_set]

        ax.plot(N_set, bpt_m, 'o-', label='matched',  color='tab:blue')
        ax.plot(N_set, bpt_c, 's-', label='cross',    color='tab:red')
        ax.plot(N_set, bpt_u, '^--', label='uniform', color='tab:gray')
        ax.set_xlabel('N (memory slots)')
        ax.set_ylabel('bpt')
        ax.set_title(f'L_y={L_y}')
        ax.legend()
        ax.set_xscale('log', base=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Stage 0: bpt by condition vs N', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  [plot] saved {out_path}')


def _plot_slot_diversity(sim: jax.Array, N: int,
                         out_path: str = 'reports/stage0_slot_diversity.png'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(np.array(sim), vmin=-1, vmax=1, cmap='coolwarm')
    fig.colorbar(im, ax=ax)
    ax.set_title(f'Memory slot cosine similarity (N={N})')
    ax.set_xlabel('slot')
    ax.set_ylabel('slot')
    off_diag = float(jnp.sum(sim) - jnp.trace(sim)) / max(N * (N - 1), 1)
    ax.set_xlabel(f'slot  [off-diag mean={off_diag:.3f}]')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  [plot] saved {out_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_ckpt(ckpt_arg: str) -> str:
    """Accept either a direct ckpt path (no extension) or a run_dir —
    in the latter case, find the latest checkpoint inside it."""
    # Direct path: file exists as-is
    if os.path.isfile(ckpt_arg + '.eqx'):
        return ckpt_arg
    # Run dir: look inside checkpoints/
    ckpt_subdir = os.path.join(ckpt_arg, 'checkpoints')
    if os.path.isdir(ckpt_subdir):
        eqx_files = sorted(
            f[:-4] for f in os.listdir(ckpt_subdir) if f.endswith('.eqx'))
        if eqx_files:
            return os.path.join(ckpt_subdir, eqx_files[-1])
    raise FileNotFoundError(f'Cannot resolve checkpoint from: {ckpt_arg!r}')


def main():
    parser = argparse.ArgumentParser(prog='kvmem.stage0')
    sub    = parser.add_subparsers(dest='cmd', required=True)

    # --- train ---
    p_train = sub.add_parser('train')
    p_train.add_argument('--steps',    type=int,  default=None)
    p_train.add_argument('--log-dir',  type=str,  default='logs',
                         help='Base dir; each run creates logs/<tag>_<ts>/')
    p_train.add_argument('--seed',     type=int,  default=42)
    p_train.add_argument('--baseline', action='store_true')

    # --- eval ---
    p_eval = sub.add_parser('eval')
    p_eval.add_argument('--ckpt', required=True,
                        help='Path to checkpoint (no .eqx) or run_dir')
    p_eval.add_argument('--baseline', action='store_true')
    p_eval.add_argument('--seed', type=int, default=1)
    p_eval.add_argument('--out-dir', type=str, default=None,
                        help='Where to save eval plots (default: same dir as ckpt)')

    # --- infer ---
    p_inf = sub.add_parser('infer')
    p_inf.add_argument('--ckpt', required=True,
                       help='Path to checkpoint (no .eqx) or run_dir')
    p_inf.add_argument('--line',         type=int,   default=-1)
    p_inf.add_argument('--all-lines',    action='store_true')
    p_inf.add_argument('--file',         type=str,   default=None)
    p_inf.add_argument('--mem-size',     type=int,   default=8)
    p_inf.add_argument('--warmup',       type=int,   default=4)
    p_inf.add_argument('--warmup-bytes', type=int,   default=4)
    p_inf.add_argument('--warmup-text',  type=str,   default='')
    p_inf.add_argument('--temp',         type=float, default=0.0)
    p_inf.add_argument('--seed',         type=int,   default=0)
    p_inf.add_argument('--fatihah',      type=str,   default=FATIHAH_PATH)

    args = parser.parse_args()

    if args.cmd == 'train':
        hp = dict(DEFAULT_HPARAMS)
        hp['seed'] = args.seed
        if args.steps:
            hp['n_steps'] = args.steps
        train(hp, baseline=args.baseline, log_base=args.log_dir)

    elif args.cmd == 'eval':
        ckpt = _resolve_ckpt(args.ckpt)
        key  = jax.random.PRNGKey(args.seed)
        model, hp = load_checkpoint(ckpt, key)
        out_dir   = args.out_dir or os.path.dirname(ckpt)
        os.makedirs(out_dir, exist_ok=True)

        results = run_eval(model, hp, baseline=args.baseline, key=key)
        L_y_set = sorted({v for _, v in hp['L_y_schedule']})
        N_set   = hp['N_set']

        if not args.baseline:
            _plot_bpt_sweep(results, L_y_set, N_set,
                            out_path=os.path.join(out_dir, 'bpt_sweep.png'))
            N_div  = 8 if 8 in N_set else N_set[0]
            key, bk = jax.random.split(key)
            sample = make_batch(bk, 1, hp['L_S'], 32, N_div,
                                hp['V_chain'], hp['alpha'])[0]
            sim = slot_diversity(model, sample, hp['L_S'], N_div)
            _plot_slot_diversity(sim, N_div,
                                 out_path=os.path.join(out_dir, f'slot_diversity_N{N_div}.png'))
            od = float(jnp.sum(sim) - jnp.trace(sim)) / max(N_div * (N_div - 1), 1)
            print(f'\n  Slot diversity (N={N_div}): off-diag cosine mean = {od:.3f}',
                  '✓' if od < 0.9 else '✗ COLLAPSE RISK')

    elif args.cmd == 'infer':
        ckpt  = _resolve_ckpt(args.ckpt)
        key   = jax.random.PRNGKey(0)
        model, hp = load_checkpoint(ckpt, key)
        N = args.mem_size

        if args.file:
            run_file_inference(model, hp, filepath=args.file, N=N,
                               warmup_text=args.warmup_text,
                               warmup_bytes_n=args.warmup_bytes,
                               temperature=args.temp, seed=args.seed)
        elif args.all_lines:
            run_all_verses(model, hp, N=N, warmup_bytes=args.warmup,
                           temperature=args.temp, seed=args.seed,
                           fatihah_path=args.fatihah)
        else:
            run_verse_inference(model, hp, line=args.line, N=N,
                                warmup_bytes=args.warmup,
                                temperature=args.temp, seed=args.seed,
                                fatihah_path=args.fatihah)


if __name__ == '__main__':
    main()
