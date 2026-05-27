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
    make_chain_pool,
    make_eval_batches,
    make_mask_baseline,
    make_mask_sanity,
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
    V          = 256,       # full byte vocab (embedding table; only ~35 tokens active)
    V_chain    = 32,        # Markov chain states → data bytes [0x20, 0x3F]
    L_S        = 64,        # source length (bytes to memorize)
    N_set      = [2, 4, 8, 16, 32],
    # Curriculum: (start_step, L_y)
    L_y_schedule = [(0, 16), (10_000, 32), (25_000, 64)],
    # --- Model: medium-small, matched to task scale ---
    # Task: 32-state Markov chain, 64-byte source, up to 64-byte continuation.
    # Only ~35 tokens active. Need enough capacity to learn in-context Markov
    # inference (read x_S, compress chain stats into KV, predict y).
    # ~200K params: enough for in-context learning, small enough to train fast.
    d          = 64,        # model dim  (was 128)
    n_layers   = 4,         # depth
    n_heads    = 4,         # heads (d_head=16)
    d_ff       = 128,       # ff width   (was 512)
    lambda_cont= 1.0,
    B          = 64,
    lr_max     = 3e-4,
    lr_min     = 1e-5,
    warmup_steps = 500,
    n_steps    = 20_000,    # fewer steps needed for smaller model
    grad_clip  = 1.0,
    wd         = 0.01,
    alpha      = 0.1,       # peaked Dirichlet → high MI between x_S and y
    seed       = 42,
    # Chain pool: pre-sample K chains; model learns each chain's stats into weights.
    # KV bottleneck then only needs to encode which chain (not full T_mat).
    # K=16 is tractable: enough variety to test KV, small enough to learn in 20k steps.
    # Set to 0 to disable (sample fresh chain per example — much harder).
    chain_pool_size = 16,
    # Optimizer: 'adamw' | 'grokadamw'
    optimizer  = 'adamw',
    grok_rho   = 0.9,       # GrokAdamW: EMA decay for squared-deviation state
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
# Optimizer (hand-rolled AdamW + GrokAdamW, no optax)
# ---------------------------------------------------------------------------

def init_opt_state(model, optimizer: str = 'adamw'):
    """
    optimizer: 'adamw' | 'grokadamw'
    AdamW state:     (m, v)
    GrokAdamW state: (m, v, s)   s = squared-deviation EMA
    """
    params = eqx.filter(model, eqx.is_array)
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    if optimizer == 'grokadamw':
        s = jax.tree.map(jnp.zeros_like, params)
        return (m, v, s)
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
    m    = jax.tree.map(lambda m_, g: b1 * m_ + (1 - b1) * g,      m, grads)
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


def grok_adam_update(params, grads, opt_state, lr: float,
                     b1: float = 0.9, b2: float = 0.999, rho: float = 0.9,
                     eps: float = 1e-8, wd: float = 0.01,
                     step: int = 1, batch_size: int = 64):
    """
    SNR-Gated AdamW (arXiv:2605.01172).

    Adds one extra EMA state s tracking squared gradient deviations (g - m_prev)².
    Gate q_k = 1{m_k^2 > s_k/(B-1)}: update only high-SNR parameters.
    Accelerates grokking by filtering noisy gradient components.

    One-line change from AdamW: gate the momentum term before applying update.
    """
    m, v, s = opt_state
    m_prev  = m   # before update, used for deviation

    m = jax.tree.map(lambda m_, g: b1 * m_ + (1 - b1) * g,      m, grads)
    v = jax.tree.map(lambda v_, g: b2 * v_ + (1 - b2) * g ** 2, v, grads)
    # Squared deviation EMA: tracks gradient variance around running mean
    s = jax.tree.map(
        lambda s_, g, mp: rho * s_ + (1 - rho) * (g - mp) ** 2,
        s, grads, m_prev,
    )

    bc1 = 1.0 - b1  ** step
    bc2 = 1.0 - b2  ** step
    bcs = 1.0 - rho ** step
    mh  = jax.tree.map(lambda m_: m_ / bc1, m)
    vh  = jax.tree.map(lambda v_: v_ / bc2, v)
    sh  = jax.tree.map(lambda s_: s_ / bcs, s)

    # SNR gate: 1 where signal (mean²) exceeds noise (variance/B)
    thresh = float(max(batch_size - 1, 1))
    gate   = jax.tree.map(
        lambda mh_, sh_: (mh_ ** 2 > sh_ / thresh).astype(jnp.float32),
        mh, sh,
    )

    new_params = jax.tree.map(
        lambda p, mh_, vh_, q: p - lr * (q * mh_ / (jnp.sqrt(vh_) + eps) + wd * p),
        params, mh, vh, gate,
    )
    return new_params, (m, v, s)


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
            L_S: int, N: int, L_y: int, lambda_cont: float) -> tuple:
    """
    Stage-0 KV bottleneck loss.
    tokens: (B, L_S + 2 + N + L_y)
    Returns (total_loss, (L_src, L_cont))
    """
    B, L    = tokens.shape
    nll     = _nll_matrix(model, tokens, mask)                    # (B, L-1)
    pos     = jnp.arange(L - 1)
    ETX_pos  = L_S + 1 + N
    Y_end    = L_S + 2 + N + L_y   # first pad position

    mask_src  = (pos <= L_S - 2).astype(jnp.float32)
    mask_cont = ((pos >= ETX_pos) & (pos < Y_end)).astype(jnp.float32)

    def wmean(x, m):
        return jnp.sum(x * m[None, :], axis=-1) / (m.sum() + 1e-8)

    L_src  = jnp.mean(wmean(nll, mask_src))
    L_cont = jnp.mean(wmean(nll, mask_cont))
    # Equal src+cont weight: src teaches chain-structure reading (prerequisite
    # for writing useful KV); cont drives the bottleneck. Once src converges
    # below oracle (~1.73 nats) the gradient naturally shifts to cont.
    total  = L_src + L_cont
    return total, (L_src, L_cont)


def loss_fn_baseline(model: KVMemModel, tokens: jax.Array, mask: jax.Array,
                     L_S: int, L_y: int, lambda_cont: float) -> tuple:
    """
    Backprop baseline loss (no bottleneck — Y sees S directly).
    tokens: (B, L_max) padded; real content is L_S + L_y tokens.
    """
    B, L = tokens.shape
    nll  = _nll_matrix(model, tokens, mask)
    pos  = jnp.arange(L - 1)

    mask_src  = (pos <= L_S - 2).astype(jnp.float32)
    mask_cont = ((pos >= L_S - 1) & (pos < L_S + L_y)).astype(jnp.float32)

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
    """Return a jit-compiled train_step function closed over hp.

    N and L_y are static so the loss masks are traced correctly per combo.
    With padded fixed-length tokens (L_max), JAX retraces only when (N, L_y)
    changes — 15 traces max, then cached forever.

    optimizer hp key selects: 'adamw' (default) | 'grokadamw'
    """
    lambda_cont = hp['lambda_cont']
    L_S         = hp['L_S']
    grad_clip   = hp['grad_clip']
    optimizer   = hp.get('optimizer', 'adamw')
    use_grok    = (optimizer == 'grokadamw')

    if baseline:
        def _loss(model, tokens, mask, N_unused, L_y):
            return loss_fn_baseline(model, tokens, mask, L_S, L_y, lambda_cont)
    else:
        def _loss(model, tokens, mask, N, L_y):
            return loss_fn(model, tokens, mask, L_S, N, L_y, lambda_cont)

    # N (arg index 4) and L_y (arg index 5) are Python ints → static for JIT
    @jax.jit(static_argnums=(4, 5))
    def train_step(model, opt_state, tokens, mask, N, L_y, step, lr):
        params  = eqx.filter(model, eqx.is_array)
        (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
            model, tokens, mask, N, L_y)
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
# Text file helpers (used by train validation + test)
# ---------------------------------------------------------------------------

def _load_txt_lines(path: str) -> list[bytes]:
    """Load non-empty lines from a UTF-8 text file as raw bytes.
    Validates no protocol bytes (< 0x20) except those in the data range.
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.rstrip('\n').encode('utf-8') for l in f if l.strip()]
    bad_lines = []
    for i, line in enumerate(lines):
        bad = [hex(b) for b in line if b < DATA_LO]
        if bad:
            bad_lines.append((i, bad[:3]))
    if bad_lines:
        raise ValueError(f'Lines contain protocol bytes: {bad_lines[:3]}')
    return lines


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(hp: dict, baseline: bool = False, sanity: bool = False, log_base: str = 'logs'):
    key = jax.random.PRNGKey(hp['seed'])
    key, mkey = jax.random.split(key)

    model     = build_model(hp, mkey)
    opt_state = init_opt_state(model, optimizer=hp.get('optimizer', 'adamw'))

    L_S          = hp['L_S']
    N_set        = hp['N_set']
    L_y_schedule = hp['L_y_schedule']
    B            = hp['B']
    n_steps      = hp['n_steps']
    alpha        = hp['alpha']
    V_chain      = hp['V_chain']

    tag = 'baseline' if baseline else ('sanity' if sanity else 'stage0')

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
    N_max   = max(N_set)
    L_y_max = max(L_y_set)
    # Fixed padded length: all tokens/masks padded to this shape → single JIT trace
    if baseline:
        L_max = L_S + L_y_max
        mask_cache = {}
        for L_y in L_y_set:
            raw = make_mask_baseline(L_S, L_y)     # (L_S+L_y, L_S+L_y)
            m   = np.full((L_max, L_max), -1e9, dtype=np.float32)
            m[:raw.shape[0], :raw.shape[0]] = raw
            mask_cache[L_y] = m
    else:
        # Both stage0 and sanity use same sequence layout [S|STX|M|ETX|Y]
        L_max = L_S + 2 + N_max + L_y_max
        mask_cache = {}
        mask_fn = make_mask_sanity if sanity else make_mask_stage0
        for N in N_set:
            for L_y in L_y_set:
                raw = mask_fn(L_S, N, L_y)         # (L, L)
                m   = np.full((L_max, L_max), -1e9, dtype=np.float32)
                m[:raw.shape[0], :raw.shape[0]] = raw
                mask_cache[(N, L_y)] = m

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
    optimizer_name = hp.get('optimizer', 'adamw')
    header += f'  Steps: {n_steps:,}  Batch: {B}  L_y curriculum: {L_y_schedule}\n'
    header += f'  Optimizer: {optimizer_name}\n'
    header += f'  Logs  -> {run_dir}/train.log\n'
    header += f'  JSONL -> {run_dir}/train.jsonl  (tail -f to follow)\n'
    _log(header)

    log_every  = 100
    plot_every = 2_000
    ckpt_every = 10_000
    val_every  = 1_000   # validation on 1.txt every N steps

    # Load validation file (1.txt) — real Quran text, fixed across training
    VAL_PATH = 'datasets/1.txt'
    val_lines: list[bytes] = []
    if os.path.exists(VAL_PATH):
        try:
            val_lines = _load_txt_lines(VAL_PATH)
            _log(f'  Val file : {VAL_PATH}  ({len(val_lines)} lines)')
        except Exception as e:
            _log(f'  [val] skipping {VAL_PATH}: {e}')

    # rolling history for live plot
    history: dict[str, list] = {'step': [], 'loss': [], 'l_src': [], 'l_cont': [],
                                  'lr': [], 'L_y': [], 'N': [], 'val_match': []}
    t0   = time.time()
    rng  = np.random.default_rng(hp['seed'] + 1)

    # Pre-sample chain pool: fixed set of K Markov chains.
    # Model learns each chain's statistics into its weights; KV bottleneck
    # only needs to encode which chain from the pool (much easier than full
    # in-context T_mat estimation from scratch every example).
    pool_size = hp.get('chain_pool_size', 64)
    if pool_size > 0 and not baseline:
        chain_pool = make_chain_pool(rng, pool_size, V_chain, alpha)
        _log(f'  Chain pool: K={pool_size} chains (alpha={alpha})')
    else:
        chain_pool = None

    # Prefetch queue: background thread generates numpy batches
    # We generate (N, L_y, batch) tuples; the main thread pulls and forwards to JAX.
    _step_counter = [0]

    def _gen():
        s   = _step_counter[0]
        lyr = get_L_y(s, L_y_schedule)
        n   = N_set[0] if baseline else int(rng.choice(N_set))
        if baseline:
            arr = np_make_baseline_batch(rng, B, V_chain, L_S, lyr, alpha)
        else:
            arr = np_make_batch(rng, B, V_chain, L_S, lyr, n, alpha,
                                chain_pool=chain_pool)
        _step_counter[0] += 1
        return (n, lyr, arr)

    prefetcher = BatchPrefetcher(_gen, maxsize=8)

    pbar = tqdm(range(1, n_steps + 1), desc=tag, unit='step',
                dynamic_ncols=True, file=sys.stdout)

    for step in pbar:
        N, L_y, np_tokens = prefetcher.get()
        # Pad tokens to L_max so JAX sees a fixed shape (single JIT trace)
        L_cur = np_tokens.shape[1]
        if L_cur < L_max:
            pad = np.zeros((B, L_max - L_cur), dtype=np.int32)  # NUL padding
            np_tokens = np.concatenate([np_tokens, pad], axis=1)
        tokens = jnp.array(np_tokens)
        if baseline:
            mask = jnp.array(mask_cache[L_y])
        else:
            mask = jnp.array(mask_cache[(N, L_y)])

        lr = lr_schedule(step, hp)
        model, opt_state, loss, (l_src, l_cont) = train_step_fn(
            model, opt_state, tokens, mask, N, L_y, step, lr)

        loss_f   = float(loss)
        l_src_f  = float(l_src)
        l_cont_f = float(l_cont)

        # tqdm postfix (always live)
        L_y_vals = sorted({v for _, v in L_y_schedule})
        phase = ['easy', 'med', 'hard'][min(L_y_vals.index(L_y) if L_y in L_y_vals else 0, 2)]
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
            history['val_match'].append(None)   # filled in at val_every

        if step % val_every == 0 and val_lines and not baseline:
            # Validation: NLL on 1.txt using a single padded forward pass.
            # Avoids _decode's per-step mask retracing which causes JIT slowdowns.
            N_val = N_set[-1]
            val_nlls = []
            for vline in val_lines:
                if len(vline) < 4:
                    continue
                x_S_v  = list(vline)
                L_S_v  = len(x_S_v)
                L_y_v  = min(16, len(x_S_v))
                mem_v  = [STX] + [NUL] * N_val + [ETX]
                # y = first L_y_v bytes of vline (reconstruction as proxy)
                y_v    = list(vline[:L_y_v])
                seq    = x_S_v + mem_v + y_v
                mask_v = jnp.array(make_mask_stage0(L_S_v, N_val, L_y_v))
                tok_v  = jnp.array(seq, dtype=jnp.int32)
                logits = model(tok_v, mask_v)
                lp     = jax.nn.log_softmax(logits[:-1], axis=-1)
                ETX_v  = L_S_v + 1 + N_val
                for k in range(L_y_v):
                    pos = ETX_v + k
                    if pos < len(seq) - 1:
                        val_nlls.append(-float(lp[pos, seq[pos + 1]]))
            val_nll = float(np.mean(val_nlls)) if val_nlls else float('nan')
            val_pct = val_nll  # report as NLL (not byte-match) for speed
            # Patch last history entry
            if history['val_match']:
                history['val_match'][-1] = val_pct
            val_rec = dict(step=step, val_match=val_pct, val_path=VAL_PATH)
            _jlog(val_rec)
            _log(f'  [val]  step={step:5d}  1.txt NLL={val_nll:.4f}  (N={N_val})')

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
            max_len: int, temperature: float, seed: int,
            stop_newline: bool = True) -> list[int]:
    """
    Autoregressive decode from KV memory. Uses a padded fixed-size mask so
    the model call is JIT-compiled once and reused every step.

    x_S    : source bytes memorized into KV
    prompt : warmup bytes (already given); generation appends after these
    Returns full generated list (including prompt).
    """
    L_S       = len(x_S)
    mem_block = [STX] + [NUL] * N + [ETX]
    # Max sequence length for mask pre-allocation
    L_max     = L_S + 2 + N + max_len + len(prompt)
    mask_full = jnp.array(make_mask_stage0(L_S, N, max_len + len(prompt)))

    generated = list(prompt)
    key       = jax.random.PRNGKey(seed)

    @jax.jit
    def _step(cur_tokens, mask):
        logits = model(cur_tokens, mask)
        return logits[-1]   # (V,)

    for _ in range(max_len):
        L_y   = len(generated)
        # Build padded token array (pad with NUL at end, use mask to ignore)
        cur   = x_S + mem_block + generated
        pad_n = (L_S + 2 + N + max_len + len(prompt)) - len(cur)
        cur_arr = jnp.array(cur + [NUL] * pad_n, dtype=jnp.int32)

        # Slice the pre-built mask to current length, padded to L_max
        L_cur  = len(cur)
        # Use a fresh mask sized exactly to L_cur for correctness
        mask_cur = jnp.array(make_mask_stage0(L_S, N, L_y))
        pad_mask = jnp.full((L_cur, L_cur), -1e9, dtype=jnp.float32)
        # Fast path: just call model with current-length input (no padding)
        cur_jnp = jnp.array(cur, dtype=jnp.int32)
        logit   = model(cur_jnp, mask_cur)[-1]   # (V,)

        if temperature == 0.0:
            nb = int(jnp.argmax(logit))
        else:
            key, sk = jax.random.split(key)
            nb = int(jax.random.choice(sk, 256,
                                       p=jax.nn.softmax(logit / temperature)))
        generated.append(nb)
        if stop_newline and nb == 0x0A and len(generated) > len(prompt) + 1:
            break

    return generated


# ---------------------------------------------------------------------------
# Test: per-line continuation
# ---------------------------------------------------------------------------


def run_test(model: KVMemModel, hp: dict,
             txt_path: str = FATIHAH_PATH,
             N: int = 8,
             warmup_bytes: int = 4,
             max_len: int = 200,
             temperature: float = 0.0,
             seed: int = 0):
    """
    Two tests on a text file:

    TEST 1 — Per-line continuation:
        For each line, memorize it into KV, give first `warmup_bytes` as prompt,
        generate until newline. Report byte-match % vs the true tail.

    TEST 2 — Whole-file continuation:
        Memorize all lines concatenated (joined by newline) into KV.
        Give the first line as prompt, generate the rest. Report byte-match %.
    """
    lines = _load_txt_lines(txt_path)
    n_lines = len(lines)
    sep = '─' * 62

    # ── TEST 1: per-line ──────────────────────────────────────────
    print(f'\n{sep}')
    print(f'TEST 1  Per-line continuation  [{txt_path}]')
    print(f'  N={N}  warmup={warmup_bytes}B  temp={temperature}')
    print(sep)

    total_match = total_target = 0
    for i, verse in enumerate(lines):
        warmup = list(verse[:warmup_bytes])
        target = verse[warmup_bytes:]
        x_S    = list(verse)

        gen = _decode(model, x_S, N, warmup, max_len, temperature, seed,
                      stop_newline=False)
        gen_tail = bytes(gen[warmup_bytes:])

        n_match = sum(a == b for a, b in zip(gen_tail, target))
        n_tot   = max(len(target), 1)
        total_match  += n_match
        total_target += n_tot

        status = '✓' if n_match / n_tot >= 0.5 else '✗'
        warmup_str = bytes(warmup).decode('utf-8', errors='replace')
        gen_str    = gen_tail.decode('utf-8', errors='replace')
        tgt_str    = target.decode('utf-8', errors='replace')
        print(f'  [{i}] {status}  warmup={warmup_str!r}')
        print(f'       gen : {gen_str}')
        print(f'       tgt : {tgt_str}')
        print(f'       match: {n_match}/{len(target)}  '
              f'({100*n_match/n_tot:.0f}%)')

    overall = 100 * total_match / max(total_target, 1)
    print(f'\n  TOTAL byte-match: {total_match}/{total_target}  ({overall:.1f}%)')

    # ── TEST 2: whole-file continuation ──────────────────────────
    print(f'\n{sep}')
    print(f'TEST 2  Whole-file continuation  [{txt_path}]')
    print(f'  N={N}  prompt=first line  temp={temperature}')
    print(sep)

    file_bytes = b'\n'.join(lines)
    bad = [hex(b) for b in file_bytes if b < DATA_LO and b != 0x0A]
    if bad:
        print(f'  [skip] file contains protocol bytes: {bad[:5]}')
        return

    x_S    = list(file_bytes)
    # Prompt = first line (without the trailing newline that would be in x_S)
    warmup = list(lines[0]) + [0x0A]   # include the newline separator
    target = file_bytes[len(lines[0]) + 1:]   # everything after first line+\n

    gen     = _decode(model, x_S, N, warmup, max_len * n_lines, temperature, seed,
                      stop_newline=False)
    gen_tail = bytes(gen[len(warmup):])
    n_match  = sum(a == b for a, b in zip(gen_tail, target))
    n_tot    = max(len(target), 1)

    print(f'  Prompt  : {bytes(warmup).decode("utf-8", errors="replace")!r}')
    print(f'  Target  : {target.decode("utf-8", errors="replace")!r}')
    print(f'  Generated: {gen_tail.decode("utf-8", errors="replace")!r}')
    print(f'  Byte-match: {n_match}/{len(target)}  ({100*n_match/n_tot:.1f}%)')
    print(sep)


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
    p_train.add_argument('--steps',     type=int,   default=None)
    p_train.add_argument('--log-dir',   type=str,   default='logs',
                         help='Base dir; each run creates logs/<tag>_<ts>/')
    p_train.add_argument('--seed',      type=int,   default=42)
    p_train.add_argument('--baseline',  action='store_true')
    p_train.add_argument('--sanity',    action='store_true',
                         help='Sanity check: same layout as stage0 but Y sees S directly (no KV bottleneck). '
                              'Should learn fast; if stage0 fails but sanity succeeds, bottleneck is the constraint.')
    p_train.add_argument('--optimizer', type=str,   default='adamw',
                         choices=['adamw', 'grokadamw'],
                         help='Optimizer: adamw (default) | grokadamw (SNR-gated, arXiv:2605.01172)')
    p_train.add_argument('--grok-rho',  type=float, default=0.9,
                         help='GrokAdamW: EMA decay for squared-deviation state (default 0.9)')

    # --- eval ---
    p_eval = sub.add_parser('eval')
    p_eval.add_argument('--ckpt', required=True,
                        help='Path to checkpoint (no .eqx) or run_dir')
    p_eval.add_argument('--baseline', action='store_true')
    p_eval.add_argument('--seed', type=int, default=1)
    p_eval.add_argument('--out-dir', type=str, default=None,
                        help='Where to save eval plots (default: same dir as ckpt)')

    # --- test ---  (primary qualitative test)
    p_test = sub.add_parser('test',
        help='Per-line + whole-file continuation test on a text file.')
    p_test.add_argument('--ckpt', required=True,
                        help='Path to checkpoint (no .eqx) or run_dir')
    p_test.add_argument('--fatihah',  type=str,   default=FATIHAH_PATH,
                        help='Text file to test on (default: Al-Fatihah)')
    p_test.add_argument('--mem-size', type=int,   default=8)
    p_test.add_argument('--warmup',   type=int,   default=4,
                        help='Warmup bytes for per-line test')
    p_test.add_argument('--temp',     type=float, default=0.0)
    p_test.add_argument('--seed',     type=int,   default=0)

    # --- infer ---  (legacy / flexible)
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
        hp['seed']      = args.seed
        hp['optimizer'] = args.optimizer
        hp['grok_rho']  = args.grok_rho
        if args.steps:
            hp['n_steps'] = args.steps
        model, run_dir = train(hp, baseline=args.baseline,
                               sanity=getattr(args, 'sanity', False),
                               log_base=args.log_dir)
        # Auto-test on suratalfatihah.txt after training completes
        if not args.baseline and not getattr(args, 'sanity', False) and os.path.exists(FATIHAH_PATH):
            print('\n\n' + '═' * 62)
            print(f'AUTO-TEST  suratalfatihah.txt  [{args.optimizer}]')
            print('═' * 62)
            run_test(model, hp, txt_path=FATIHAH_PATH, N=hp['N_set'][-1],
                     warmup_bytes=4, temperature=0.0)

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

    elif args.cmd == 'infer':
        ckpt  = _resolve_ckpt(args.ckpt)
        key   = jax.random.PRNGKey(0)
        model, hp = load_checkpoint(ckpt, key)
        N = args.mem_size

        if args.file:
            # Legacy: whole-file using run_test test 2 path
            lines = _load_txt_lines(args.file)
            file_bytes = b'\n'.join(lines)
            x_S    = list(file_bytes)
            warmup = list(file_bytes[:args.warmup_bytes]) if not args.warmup_text \
                     else list(args.warmup_text.encode('utf-8'))
            gen = _decode(model, x_S, N, warmup, 300, args.temp, args.seed,
                          stop_newline=False)
            print(bytes(gen).decode('utf-8', errors='replace'))
        else:
            run_test(model, hp, txt_path=args.fatihah, N=N,
                     warmup_bytes=args.warmup, temperature=args.temp,
                     seed=args.seed)


if __name__ == '__main__':
    main()
