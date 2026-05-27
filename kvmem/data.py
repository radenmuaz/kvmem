"""
kvmem/data.py — Synthetic byte-level Markov chain dataset.

Vocab: V=256 bytes. Data bytes constrained to [0x20, 0xFF] so they never
collide with protocol bytes 0x00-0x1F (STX, ETX, NUL, etc.).

Sequence layout (stage 0):
    [ x_S (L_S) | STX | NUL*N | ETX | y (L_y) ]
    S = [0, L_S)
    M = [L_S+1, L_S+1+N)    <- inner KV slots
    Y = [L_S+2+N, L_S+2+N+L_y)

Memory tokens are always NUL (0x00). Token *identity* in M does not matter —
the model reads region M by position (timestep), not by token value.
"""

import os
import queue
import threading

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Protocol bytes (never appear in data)
# ---------------------------------------------------------------------------
STX = 0x02   # open memory segment
ETX = 0x03   # close memory segment
NUL = 0x00   # inner memory slot filler

# Data bytes are in [DATA_LO, 256)
DATA_LO = 0x20


# ---------------------------------------------------------------------------
# Markov chain helpers
# ---------------------------------------------------------------------------

def sample_transition_matrix(key: jax.Array, V: int = 256,
                             alpha: float = 0.5) -> jax.Array:
    """
    Sample a random V×V row-stochastic transition matrix.
    Each row ~ Dirichlet(alpha, ..., alpha).
    alpha < 1 => peaked/sparse rows (sharper chain statistics).

    Returns: (V, V) float32, rows sum to 1.
    """
    g = jax.random.gamma(key, alpha, shape=(V, V))
    return g / g.sum(axis=-1, keepdims=True)


def stationary_distribution(T_mat: jax.Array, n_iter: int = 200) -> jax.Array:
    """Approximate stationary dist via power iteration. Returns (V,) float32."""
    V = T_mat.shape[0]
    pi = jnp.ones(V, dtype=jnp.float32) / V
    def step(pi, _):
        pi = pi @ T_mat
        return pi, None
    pi, _ = jax.lax.scan(step, pi, None, length=n_iter)
    return pi / pi.sum()


def chain_entropy_bits(T_mat: jax.Array, pi: jax.Array) -> float:
    """
    Entropy rate H = -sum_i pi_i sum_j T_ij log2(T_ij)  [bits/token].
    Oracle lower bound on bpt for a model with perfect knowledge of T_mat.
    """
    log2_T = jnp.where(T_mat > 0, jnp.log2(jnp.clip(T_mat, 1e-30)), 0.0)
    return float(-jnp.sum(pi[:, None] * T_mat * log2_T))


def walk_chain(key: jax.Array, T_mat: jax.Array,
               start: int, length: int) -> jax.Array:
    """
    Walk Markov chain for `length` steps from `start`.
    Returns (length,) int32 of states in [0, V).

    Note: states are in [0, V). Caller is responsible for remapping to
    data bytes [DATA_LO, 256) if needed (see _remap below).
    """
    V = T_mat.shape[0]
    def step_fn(state, key_):
        next_state = jax.random.choice(key_, V, p=T_mat[state])
        return next_state, next_state
    keys = jax.random.split(key, length)
    _, seq = jax.lax.scan(step_fn, jnp.int32(start), keys)
    return seq.astype(jnp.int32)


def _remap(seq: jax.Array, V_chain: int) -> jax.Array:
    """
    Remap chain states [0, V_chain) to data bytes [DATA_LO, DATA_LO+V_chain).
    Ensures no collision with protocol bytes < 0x20.
    V_chain must be <= 256 - DATA_LO = 224.
    """
    return (seq + DATA_LO).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Single-example builders
# ---------------------------------------------------------------------------

def _make_example(key: jax.Array, V_chain: int, L_S: int, L_y: int,
                  N: int, alpha: float) -> jax.Array:
    """
    Build one sequence: [x_S | STX | NUL*N | ETX | y]
    x_S and y are remapped to [DATA_LO, DATA_LO+V_chain).
    y is an independent continuation from x_S's terminal state.

    Returns (L_S + 2 + N + L_y,) int32.
    """
    k0, k1, k2, k3 = jax.random.split(key, 4)

    T_mat = sample_transition_matrix(k0, V_chain, alpha)
    start = jax.random.randint(k1, (), 0, V_chain)

    x_S_raw = walk_chain(k2, T_mat, start, L_S)
    terminal = x_S_raw[-1]
    y_raw = walk_chain(k3, T_mat, terminal, L_y)

    x_S = _remap(x_S_raw, V_chain)
    y   = _remap(y_raw,   V_chain)

    mem = jnp.full((N,), NUL, dtype=jnp.int32)
    stx = jnp.array([STX], dtype=jnp.int32)
    etx = jnp.array([ETX], dtype=jnp.int32)

    return jnp.concatenate([x_S, stx, mem, etx, y])


def _make_example_cross(key: jax.Array, V_chain: int, L_S: int, L_y: int,
                        N: int, alpha: float) -> jax.Array:
    """
    CROSS condition: x_S and y come from *different* transition matrices.
    The KV memorizes x_S's chain; y needs a different chain -> actively wrong memory.
    """
    k0, k1, k2, k3, k4 = jax.random.split(key, 5)

    T_src = sample_transition_matrix(k0, V_chain, alpha)
    T_y   = sample_transition_matrix(k1, V_chain, alpha)

    start_src = jax.random.randint(k2, (), 0, V_chain)
    start_y   = jax.random.randint(k3, (), 0, V_chain)

    x_S_raw = walk_chain(k4, T_src, start_src, L_S)
    y_raw   = walk_chain(k3, T_y,   start_y,   L_y)

    x_S = _remap(x_S_raw, V_chain)
    y   = _remap(y_raw,   V_chain)

    mem = jnp.full((N,), NUL, dtype=jnp.int32)
    stx = jnp.array([STX], dtype=jnp.int32)
    etx = jnp.array([ETX], dtype=jnp.int32)

    return jnp.concatenate([x_S, stx, mem, etx, y])


def _make_example_uniform(key: jax.Array, V_chain: int, L_S: int, L_y: int,
                           N: int, alpha: float) -> jax.Array:
    """
    UNIFORM condition: x_S is uniform random data bytes; y from a fresh chain.
    Memory encodes noise -> baseline (no useful info in KV).
    """
    k0, k1, k2, k3 = jax.random.split(key, 4)

    # x_S: uniform over data bytes [DATA_LO, DATA_LO+V_chain)
    x_S = jax.random.randint(k0, (L_S,), DATA_LO, DATA_LO + V_chain).astype(jnp.int32)

    T_y = sample_transition_matrix(k1, V_chain, alpha)
    start_y = jax.random.randint(k2, (), 0, V_chain)
    y_raw = walk_chain(k3, T_y, start_y, L_y)
    y = _remap(y_raw, V_chain)

    mem = jnp.full((N,), NUL, dtype=jnp.int32)
    stx = jnp.array([STX], dtype=jnp.int32)
    etx = jnp.array([ETX], dtype=jnp.int32)

    return jnp.concatenate([x_S, stx, mem, etx, y])


# ---------------------------------------------------------------------------
# Batched builders (vmapped)
# ---------------------------------------------------------------------------

def make_batch(key: jax.Array, B: int, L_S: int, L_y: int, N: int,
               V_chain: int = 224, alpha: float = 0.5) -> jax.Array:
    """
    Training batch: B matched examples.
    Returns (B, L_S + 2 + N + L_y) int32.
    V_chain: number of Markov states, remapped to [0x20, 0x20+V_chain).
             Default 224 = all printable+high bytes.
    """
    keys = jax.random.split(key, B)
    return jax.vmap(lambda k: _make_example(k, V_chain, L_S, L_y, N, alpha))(keys)


def make_eval_batches(key: jax.Array, B: int, L_S: int, L_y: int, N: int,
                      V_chain: int = 224, alpha: float = 0.5) -> dict:
    """
    Build all three eval condition batches.
    Returns dict with keys 'matched', 'cross', 'uniform',
    each (B, L_S + 2 + N + L_y) int32.
    """
    k0, k1, k2 = jax.random.split(key, 3)
    keys_m = jax.random.split(k0, B)
    keys_c = jax.random.split(k1, B)
    keys_u = jax.random.split(k2, B)
    return {
        'matched': jax.vmap(
            lambda k: _make_example(k, V_chain, L_S, L_y, N, alpha))(keys_m),
        'cross':   jax.vmap(
            lambda k: _make_example_cross(k, V_chain, L_S, L_y, N, alpha))(keys_c),
        'uniform': jax.vmap(
            lambda k: _make_example_uniform(k, V_chain, L_S, L_y, N, alpha))(keys_u),
    }


# ---------------------------------------------------------------------------
# Stage 1 multi-pass builder
# ---------------------------------------------------------------------------

def make_batch_stage1(key: jax.Array, B: int, T: int, L_S: int, L_y: int,
                      N: int, V_chain: int = 224, alpha: float = 0.5) -> jax.Array:
    """
    Stage 1 layout:
        [x_S | STX NUL*N ETX | y^1 | STX NUL*N ETX | y^2 | ... | STX NUL*N ETX | y^T]
    Each y^t is an independent continuation from x_S's terminal state.
    Returns (B, L_S + T*(N+2+L_y)) int32.
    """
    keys = jax.random.split(key, B)

    def one(k):
        ks = jax.random.split(k, 2 + T)
        T_mat = sample_transition_matrix(ks[0], V_chain, alpha)
        start = jax.random.randint(ks[1], (), 0, V_chain)
        x_S_raw = walk_chain(ks[1], T_mat, start, L_S)
        terminal = x_S_raw[-1]
        x_S = _remap(x_S_raw, V_chain)

        mem = jnp.full((N,), NUL, dtype=jnp.int32)
        stx = jnp.array([STX], dtype=jnp.int32)
        etx = jnp.array([ETX], dtype=jnp.int32)

        parts = [x_S]
        for t in range(T):
            y_raw = walk_chain(ks[2 + t], T_mat, terminal, L_y)
            y = _remap(y_raw, V_chain)
            parts += [stx, mem, etx, y]
        return jnp.concatenate(parts)

    return jax.vmap(one)(keys)


# ---------------------------------------------------------------------------
# Mask builders (numpy, static — precomputed at startup)
# ---------------------------------------------------------------------------

def make_mask_stage0(L_S: int, N: int, L_y: int) -> np.ndarray:
    """
    Attention mask for stage 0.
    Shape (L, L) float32 where L = L_S + 2 + N + L_y.
    0.0 = attend, -1e9 = blocked.

    Rules (on top of causal):
      1. Y is write-only: nothing outside Y can attend to Y positions.
      2. Y cannot see S or STX: forced to read only through M and ETX.
    """
    L       = L_S + 2 + N + L_y
    M_start = L_S + 1        # first NUL slot
    Y_start = L_S + 2 + N   # first y token

    rows = np.arange(L)[:, None]
    cols = np.arange(L)[None, :]

    causal         = cols <= rows
    is_S_or_STX    = cols < M_start       # S + STX (col index)
    is_Y_col       = cols >= Y_start
    is_Y_row       = rows >= Y_start

    block_y_sink   = is_Y_col & ~is_Y_row   # rule 1
    block_y_sees_s = is_Y_row & is_S_or_STX  # rule 2

    blocked = block_y_sink | block_y_sees_s
    visible = causal & ~blocked
    return np.where(visible, 0.0, -1e9).astype(np.float32)


def make_mask_stage1(L_S: int, N: int, L_y: int, T: int) -> np.ndarray:
    """
    Attention mask for stage 1 (multi-pass).
    Shape (L, L) where L = L_S + T*(N+2+L_y).

    Block size per pass = N + 2 + L_y  (STX + NUL*N + ETX + y)
    M^t = inner NUL slots of pass t
    Y^t = continuation tokens of pass t

    Rules:
      1. Y is write-only: nothing outside Y attends to any Y.
      2. Y^t cannot see S or STX of any block.
      3. Y^t cannot see Y^s for s != t (each y^t is independent).
    """
    block = N + 2 + L_y
    L     = L_S + T * block

    rows = np.arange(L)
    cols = np.arange(L)

    # For each position, determine which pass it belongs to and whether M or Y
    def region(pos):
        in_src = pos < L_S
        offset = np.where(in_src, 0, pos - L_S)
        t      = np.where(in_src, -1, offset // block)        # pass index 0..T-1
        within = np.where(in_src, 0,  offset  % block)
        # within: 0=STX, 1..N=NUL slots, N+1=ETX, N+2..N+1+L_y=y
        is_M   = ~in_src & (within >= 1) & (within <= N)
        is_Y   = ~in_src & (within >= N + 2)
        is_STX = ~in_src & (within == 0)
        return in_src, t, is_M, is_Y, is_STX

    in_src_r, t_r, is_M_r, is_Y_r, is_STX_r = region(rows)
    in_src_c, t_c, is_M_c, is_Y_c, is_STX_c = region(cols)

    causal = cols[None, :] <= rows[:, None]

    # Broadcast
    is_Y_row   = is_Y_r[:, None]
    is_Y_col   = is_Y_c[None, :]
    t_row      = t_r[:, None]
    t_col      = t_c[None, :]
    is_S_col   = in_src_c[None, :]
    is_STX_col = is_STX_c[None, :]

    block_y_sink   = is_Y_col & ~is_Y_row
    block_y_sees_s = is_Y_row & (is_S_col | is_STX_col)
    block_y_cross  = is_Y_row & is_Y_col & (t_row != t_col)

    blocked = block_y_sink | block_y_sees_s | block_y_cross
    visible = causal & ~blocked
    return np.where(visible, 0.0, -1e9).astype(np.float32)


def make_mask_baseline(L_S: int, L_y: int) -> np.ndarray:
    """Standard causal mask for backprop baseline (no bottleneck)."""
    L    = L_S + L_y
    rows = np.arange(L)[:, None]
    cols = np.arange(L)[None, :]
    return np.where(cols <= rows, 0.0, -1e9).astype(np.float32)


def make_mask_sanity(L_S: int, N: int, L_y: int) -> np.ndarray:
    """Sanity-check mask: same sequence layout as stage0 [S|STX|M|ETX|Y]
    but Y can attend to ALL previous tokens (S included) — no KV bottleneck.
    Only rule preserved: Y is write-only sink (nothing outside Y attends to Y).
    If model can learn this but not stage0, the bottleneck is the constraint.
    """
    L      = L_S + 2 + N + L_y
    Y_start = L_S + 2 + N
    rows = np.arange(L)[:, None]
    cols = np.arange(L)[None, :]
    causal     = cols <= rows
    is_Y_col   = cols >= Y_start
    is_Y_row   = rows >= Y_start
    block_sink = is_Y_col & ~is_Y_row   # Y is write-only sink
    visible    = causal & ~block_sink
    return np.where(visible, 0.0, -1e9).astype(np.float32)


def build_mask_cache(L_S: int, N_set: list, L_y_set: list) -> dict:
    """
    Precompute stage-0 masks for all (N, L_y) combinations.
    Returns dict keyed by (N, L_y) -> np.ndarray of shape (L, L).
    Also includes baseline masks keyed by ('baseline', L_y).
    """
    cache = {}
    for N in N_set:
        for L_y in L_y_set:
            cache[(N, L_y)] = make_mask_stage0(L_S, N, L_y)
    for L_y in L_y_set:
        cache[('baseline', L_y)] = make_mask_baseline(L_S, L_y)
    return cache


# ---------------------------------------------------------------------------
# Fast numpy batch generation (CPU-only, no JAX tracing overhead)
# ---------------------------------------------------------------------------

def _np_walk_chain(rng: np.random.Generator, T_mat: np.ndarray,
                   start: int, length: int) -> np.ndarray:
    """Walk a Markov chain for `length` steps using numpy (fast on CPU)."""
    seq = np.empty(length, dtype=np.int32)
    state = start
    for i in range(length):
        state = rng.choice(T_mat.shape[0], p=T_mat[state])
        seq[i] = state
    return seq


def np_make_one(rng: np.random.Generator, V_chain: int, L_S: int,
                L_y: int, N: int, alpha: float) -> np.ndarray:
    """Build one [x_S | STX | NUL*N | ETX | y] sequence using numpy."""
    # Sample transition matrix via Dirichlet (gamma trick)
    g = rng.gamma(alpha, size=(V_chain, V_chain)).astype(np.float32)
    T_mat = g / g.sum(axis=1, keepdims=True)

    start = rng.integers(0, V_chain)
    x_S_raw = _np_walk_chain(rng, T_mat, start, L_S)
    terminal = int(x_S_raw[-1])
    y_raw    = _np_walk_chain(rng, T_mat, terminal, L_y)

    x_S = (x_S_raw + DATA_LO).astype(np.int32)
    y   = (y_raw   + DATA_LO).astype(np.int32)

    seq = np.empty(L_S + 2 + N + L_y, dtype=np.int32)
    seq[:L_S]                    = x_S
    seq[L_S]                     = STX
    seq[L_S+1 : L_S+1+N]        = NUL
    seq[L_S+1+N]                 = ETX
    seq[L_S+2+N:]                = y
    return seq


def np_make_one_cross(rng: np.random.Generator, V_chain: int, L_S: int,
                      L_y: int, N: int, alpha: float) -> np.ndarray:
    """Cross condition: x_S and y from different chains."""
    g1 = rng.gamma(alpha, size=(V_chain, V_chain)).astype(np.float32)
    g2 = rng.gamma(alpha, size=(V_chain, V_chain)).astype(np.float32)
    T1 = g1 / g1.sum(axis=1, keepdims=True)
    T2 = g2 / g2.sum(axis=1, keepdims=True)

    x_S_raw = _np_walk_chain(rng, T1, rng.integers(0, V_chain), L_S)
    y_raw   = _np_walk_chain(rng, T2, rng.integers(0, V_chain), L_y)

    seq = np.empty(L_S + 2 + N + L_y, dtype=np.int32)
    seq[:L_S]             = (x_S_raw + DATA_LO).astype(np.int32)
    seq[L_S]              = STX
    seq[L_S+1:L_S+1+N]   = NUL
    seq[L_S+1+N]          = ETX
    seq[L_S+2+N:]         = (y_raw + DATA_LO).astype(np.int32)
    return seq


def np_make_one_uniform(rng: np.random.Generator, V_chain: int, L_S: int,
                        L_y: int, N: int, alpha: float) -> np.ndarray:
    """Uniform condition: x_S is random bytes, y from fresh chain."""
    g = rng.gamma(alpha, size=(V_chain, V_chain)).astype(np.float32)
    T = g / g.sum(axis=1, keepdims=True)
    x_S   = rng.integers(DATA_LO, DATA_LO + V_chain, size=L_S).astype(np.int32)
    y_raw = _np_walk_chain(rng, T, rng.integers(0, V_chain), L_y)

    seq = np.empty(L_S + 2 + N + L_y, dtype=np.int32)
    seq[:L_S]           = x_S
    seq[L_S]            = STX
    seq[L_S+1:L_S+1+N] = NUL
    seq[L_S+1+N]        = ETX
    seq[L_S+2+N:]       = (y_raw + DATA_LO).astype(np.int32)
    return seq


def make_chain_pool(rng: np.random.Generator, K: int, V_chain: int,
                    alpha: float) -> np.ndarray:
    """Pre-sample K transition matrices. Returns (K, V_chain, V_chain) float32.

    Using a fixed pool of chains makes in-context Markov learning tractable:
    the model learns each chain's statistics into its weights; the KV bottleneck
    must encode which chain from the pool.  K=64 gives variety without making
    in-context estimation impossible.
    """
    pool = np.empty((K, V_chain, V_chain), dtype=np.float32)
    for k in range(K):
        g = rng.gamma(alpha, size=(V_chain, V_chain)).astype(np.float32)
        pool[k] = g / g.sum(axis=1, keepdims=True)
    return pool


def np_make_batch(rng: np.random.Generator, B: int, V_chain: int,
                  L_S: int, L_y: int, N: int, alpha: float = 0.5,
                  chain_pool: np.ndarray | None = None) -> np.ndarray:
    """Fast numpy batch: (B, L_S+2+N+L_y) int32.

    If chain_pool is given (K, V_chain, V_chain), each example picks one chain
    randomly from the pool instead of sampling a fresh chain.  This makes the
    in-context learning task tractable: the model sees repeated statistics and
    can learn each chain into its weights; the KV bottleneck encodes chain ID.
    """
    L = L_S + 2 + N + L_y
    out = np.empty((B, L), dtype=np.int32)
    for i in range(B):
        if chain_pool is not None:
            k = int(rng.integers(0, len(chain_pool)))
            T_mat = chain_pool[k]
            start = int(rng.integers(0, V_chain))
            x_S_raw = _np_walk_chain(rng, T_mat, start, L_S)
            terminal = int(x_S_raw[-1])
            y_raw = _np_walk_chain(rng, T_mat, terminal, L_y)
            seq = np.empty(L, dtype=np.int32)
            seq[:L_S]             = (x_S_raw + DATA_LO).astype(np.int32)
            seq[L_S]              = STX
            seq[L_S+1:L_S+1+N]   = NUL
            seq[L_S+1+N]         = ETX
            seq[L_S+2+N:]        = (y_raw + DATA_LO).astype(np.int32)
            out[i] = seq
        else:
            out[i] = np_make_one(rng, V_chain, L_S, L_y, N, alpha)
    return out


def np_make_eval_batches(seed: int, B: int, V_chain: int,
                         L_S: int, L_y: int, N: int,
                         alpha: float = 0.5) -> dict:
    """Build matched/cross/uniform eval batches using numpy."""
    rng = np.random.default_rng(seed)
    L   = L_S + 2 + N + L_y
    matched = np.empty((B, L), dtype=np.int32)
    cross   = np.empty((B, L), dtype=np.int32)
    uniform = np.empty((B, L), dtype=np.int32)
    for i in range(B):
        matched[i] = np_make_one(rng, V_chain, L_S, L_y, N, alpha)
        cross[i]   = np_make_one_cross(rng, V_chain, L_S, L_y, N, alpha)
        uniform[i] = np_make_one_uniform(rng, V_chain, L_S, L_y, N, alpha)
    return {'matched': matched, 'cross': cross, 'uniform': uniform}


def np_make_baseline_batch(rng: np.random.Generator, B: int, V_chain: int,
                           L_S: int, L_y: int, alpha: float = 0.5) -> np.ndarray:
    """Baseline batch [x_S | y] without memory tokens: (B, L_S+L_y)."""
    L   = L_S + L_y
    out = np.empty((B, L), dtype=np.int32)
    for i in range(B):
        g = rng.gamma(alpha, size=(V_chain, V_chain)).astype(np.float32)
        T = g / g.sum(axis=1, keepdims=True)
        s     = rng.integers(0, V_chain)
        x_raw = _np_walk_chain(rng, T, s, L_S)
        y_raw = _np_walk_chain(rng, T, int(x_raw[-1]), L_y)
        out[i, :L_S] = (x_raw + DATA_LO).astype(np.int32)
        out[i, L_S:] = (y_raw + DATA_LO).astype(np.int32)
    return out


# ---------------------------------------------------------------------------
# Prefetch queue — generates batches in a background thread
# ---------------------------------------------------------------------------

class BatchPrefetcher:
    """
    Generates batches in a background thread and serves them via get().
    Keeps `maxsize` batches ready so JAX never waits on data.
    """
    def __init__(self, gen_fn, maxsize: int = 4):
        """
        gen_fn: callable() -> np.ndarray  (called repeatedly)
        """
        self._q      = queue.Queue(maxsize=maxsize)
        self._gen_fn = gen_fn
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            try:
                batch = self._gen_fn()
                self._q.put(batch, timeout=1.0)
            except Exception:
                pass

    def get(self) -> np.ndarray:
        return self._q.get()

    def close(self):
        self._stop.set()


# ---------------------------------------------------------------------------
# Quran / text file helpers
# ---------------------------------------------------------------------------

def load_text_lines(path: str, n_lines: int | None = None,
                    encoding: str = 'utf-8') -> list[bytes]:
    """
    Load lines from a text file as raw UTF-8 byte strings.
    Strips trailing newline from each line.
    Asserts all bytes >= 0x20 (no protocol byte collision).
    """
    with open(path, 'r', encoding=encoding) as f:
        lines = [l.rstrip('\n').encode('utf-8') for l in f]
    if n_lines is not None:
        lines = lines[:n_lines]
    for i, line in enumerate(lines):
        bad = [hex(b) for b in line if b < DATA_LO]
        assert not bad, f"Line {i} contains protocol bytes: {bad}"
    return lines


def load_fatihah(path: str = 'datasets/quran_uthmani.txt') -> list[bytes]:
    """Load the 7 ayat of Surah Al-Fatihah (lines 0-6)."""
    return load_text_lines(path, n_lines=7)


# ---------------------------------------------------------------------------
# Dataset writing / loading  (byte-level, no tokenizer needed)
# ---------------------------------------------------------------------------

def write_dataset(
    out_path: str,
    n_examples: int,
    V_chain: int,
    L_S: int,
    L_y: int,
    N: int,
    alpha: float = 0.5,
    seed: int = 0,
    mode: str = 'matched',   # 'matched' | 'cross' | 'uniform' | 'baseline'
    verbose: bool = True,
) -> None:
    """
    Generate `n_examples` synthetic sequences and write them to a binary file.

    File format (simple, no header):
        - First 8 bytes: magic b'KVMEM001'
        - Next 4 bytes:  uint32 LE  — n_examples
        - Next 4 bytes:  uint32 LE  — sequence length L per example
        - Then n_examples * L bytes, each byte is a token (uint8).

    Token values:
        0x00      = NUL  (memory slot)
        0x02      = STX  (memory open bracket)
        0x03      = ETX  (memory close bracket)
        0x20-0xFF = data bytes

    Sequence layout per example (mode != 'baseline'):
        [ x_S (L_S) | STX | NUL*N | ETX | y (L_y) ]   length = L_S + 2 + N + L_y

    Baseline layout:
        [ x_S (L_S) | y (L_y) ]    length = L_S + L_y

    Loading back is trivial:
        data = np.fromfile(path, dtype=np.uint8)
        data = data[16:]              # skip 16-byte header
        tokens = data.reshape(n, L)  # each row is one example
    """
    import struct

    rng = np.random.default_rng(seed)

    if mode == 'baseline':
        L = L_S + L_y
    else:
        L = L_S + 2 + N + L_y

    MAGIC = b'KVMEM001'
    header = MAGIC + struct.pack('<II', n_examples, L)
    assert len(header) == 16

    _make = {
        'matched':  np_make_one,
        'cross':    np_make_one_cross,
        'uniform':  np_make_one_uniform,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with open(out_path, 'wb') as f:
        f.write(header)
        for i in range(n_examples):
            if mode == 'baseline':
                # inline: no memory segment
                g = rng.gamma(alpha, size=(V_chain, V_chain)).astype(np.float32)
                T = g / g.sum(axis=1, keepdims=True)
                s     = rng.integers(0, V_chain)
                x_raw = _np_walk_chain(rng, T, s, L_S)
                y_raw = _np_walk_chain(rng, T, int(x_raw[-1]), L_y)
                row   = np.empty(L, dtype=np.uint8)
                row[:L_S] = (x_raw + DATA_LO).astype(np.uint8)
                row[L_S:] = (y_raw + DATA_LO).astype(np.uint8)
            else:
                row = _make[mode](rng, V_chain, L_S, L_y, N, alpha).astype(np.uint8)
            f.write(row.tobytes())
            if verbose and (i + 1) % max(1, n_examples // 10) == 0:
                print(f'  {i+1}/{n_examples}  ({100*(i+1)//n_examples}%)')

    size_kb = os.path.getsize(out_path) / 1024
    if verbose:
        print(f'Written {n_examples} examples × {L} bytes → {out_path}  ({size_kb:.1f} KB)')


def load_dataset(path: str) -> tuple[np.ndarray, dict]:
    """
    Load a dataset written by write_dataset().

    Returns:
        tokens : (n_examples, L) uint8 ndarray
        meta   : dict with keys n_examples, L, magic
    """
    import struct

    with open(path, 'rb') as f:
        raw = f.read()

    magic = raw[:8]
    assert magic == b'KVMEM001', f'Bad magic: {magic!r}'
    n_examples, L = struct.unpack('<II', raw[8:16])
    tokens = np.frombuffer(raw[16:], dtype=np.uint8).reshape(n_examples, L).copy()
    return tokens, {'n_examples': n_examples, 'L': L, 'magic': magic}


def write_dataset_txt(
    out_path: str,
    n_examples: int,
    V_chain: int,
    L_S: int,
    L_y: int,
    N: int,
    alpha: float = 0.5,
    seed: int = 0,
    mode: str = 'matched',
    verbose: bool = True,
) -> None:
    """
    Generate synthetic sequences and write to a plain-text file.

    Encoding: latin-1 (ISO-8859-1).  Every byte value 0x00–0xFF maps to
    exactly one character — no information loss, no multi-byte sequences.
    Each line = one example.  Lines are newline-terminated (0x0A).

    Because data bytes are constrained to [0x20, 0xFF] and protocol bytes
    (NUL=0x00, STX=0x02, ETX=0x03) never appear in the data region, you
    can visually read the printable ASCII portion of each example directly.

    Loading back:
        tokens, meta = load_dataset_txt(path)
        # tokens: (n_examples, L) uint8

    Sequence layout per line (mode != 'baseline'):
        x_S (L_S bytes)  |  STX  |  NUL*N  |  ETX  |  y (L_y bytes)
        visible as:       '\\x02'   '\\x00'*N   '\\x03'

    Baseline layout:
        x_S (L_S bytes)  |  y (L_y bytes)
    """
    _make = {
        'matched':  np_make_one,
        'cross':    np_make_one_cross,
        'uniform':  np_make_one_uniform,
    }

    rng = np.random.default_rng(seed)

    if mode == 'baseline':
        L = L_S + L_y
    else:
        L = L_S + 2 + N + L_y

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with open(out_path, 'w', encoding='latin-1', newline='') as f:
        # Header comment line so the file is self-describing
        f.write(f'# kvmem txt dataset | mode={mode} n={n_examples} '
                f'L_S={L_S} N={N} L_y={L_y} V_chain={V_chain} '
                f'alpha={alpha} seed={seed} L={L}\n')

        for i in range(n_examples):
            if mode == 'baseline':
                g = rng.gamma(alpha, size=(V_chain, V_chain)).astype(np.float32)
                T = g / g.sum(axis=1, keepdims=True)
                s     = rng.integers(0, V_chain)
                x_raw = _np_walk_chain(rng, T, s, L_S)
                y_raw = _np_walk_chain(rng, T, int(x_raw[-1]), L_y)
                row   = np.empty(L, dtype=np.uint8)
                row[:L_S] = (x_raw + DATA_LO).astype(np.uint8)
                row[L_S:] = (y_raw + DATA_LO).astype(np.uint8)
            else:
                row = _make[mode](rng, V_chain, L_S, L_y, N, alpha).astype(np.uint8)

            f.write(row.tobytes().decode('latin-1') + '\n')

            if verbose and (i + 1) % max(1, n_examples // 10) == 0:
                print(f'  {i+1}/{n_examples}  ({100*(i+1)//n_examples}%)')

    size_kb = os.path.getsize(out_path) / 1024
    if verbose:
        print(f'Written {n_examples} examples × {L} bytes → {out_path}  ({size_kb:.1f} KB)')


def load_dataset_txt(path: str) -> tuple[np.ndarray, dict]:
    """
    Load a dataset written by write_dataset_txt().

    Skips comment lines starting with '#'.
    Each remaining line is decoded as latin-1 and converted to uint8.

    Returns:
        tokens : (n_examples, L) uint8  — same layout as load_dataset()
        meta   : dict parsed from the header comment (if present)
    """
    meta: dict = {}
    rows: list[np.ndarray] = []

    with open(path, 'r', encoding='latin-1', newline='') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#'):
                # parse key=value pairs from header
                for tok in line[1:].split():
                    if '=' in tok:
                        k, v = tok.split('=', 1)
                        try:
                            meta[k] = int(v)
                        except ValueError:
                            try:
                                meta[k] = float(v)
                            except ValueError:
                                meta[k] = v
                continue
            if not line:
                continue
            rows.append(np.frombuffer(line.encode('latin-1'), dtype=np.uint8).copy())

    tokens = np.stack(rows, axis=0)
    meta.setdefault('n_examples', len(rows))
    meta.setdefault('L', tokens.shape[1] if rows else 0)
    return tokens, meta


# ---------------------------------------------------------------------------
# CLI  (python -m kvmem.data write ...)
# ---------------------------------------------------------------------------



def _cli():
    import argparse

    p = argparse.ArgumentParser(
        prog='python -m kvmem.data',
        description='Generate and inspect synthetic KV-mem datasets.',
    )
    sub = p.add_subparsers(dest='cmd', required=True)

    # ---- write ----
    w = sub.add_parser('write', help='Generate dataset and write to binary file.')
    w.add_argument('out',            help='Output file path (e.g. data/train.bin)')
    w.add_argument('--n',  type=int, default=10_000,  help='Number of examples (default 10000)')
    w.add_argument('--L-S', type=int, default=128,    dest='L_S', help='Source length (default 128)')
    w.add_argument('--L-y', type=int, default=32,     dest='L_y', help='Continuation length (default 32)')
    w.add_argument('--N',  type=int, default=8,       help='Memory slots (default 8)')
    w.add_argument('--V-chain', type=int, default=224, dest='V_chain',
                   help='Markov states / data vocab size (default 224)')
    w.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha (default 0.5)')
    w.add_argument('--seed', type=int, default=0,     help='RNG seed (default 0)')
    w.add_argument('--mode', choices=['matched', 'cross', 'uniform', 'baseline'],
                   default='matched', help='Example type (default matched)')

    # ---- info ----
    i = sub.add_parser('info', help='Print header info for a binary dataset file.')
    i.add_argument('path', help='Path to .bin file')

    # ---- write-txt ----
    wt = sub.add_parser('write-txt', help='Generate dataset and write to latin-1 text file (one example per line).')
    wt.add_argument('out',            help='Output file path (e.g. data/train.txt)')
    wt.add_argument('--n',  type=int, default=10_000,  help='Number of examples (default 10000)')
    wt.add_argument('--L-S', type=int, default=128,    dest='L_S', help='Source length (default 128)')
    wt.add_argument('--L-y', type=int, default=32,     dest='L_y', help='Continuation length (default 32)')
    wt.add_argument('--N',  type=int, default=8,       help='Memory slots (default 8)')
    wt.add_argument('--V-chain', type=int, default=224, dest='V_chain',
                    help='Markov states / data vocab size (default 224)')
    wt.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha (default 0.5)')
    wt.add_argument('--seed', type=int, default=0,     help='RNG seed (default 0)')
    wt.add_argument('--mode', choices=['matched', 'cross', 'uniform', 'baseline'],
                    default='matched', help='Example type (default matched)')

    # ---- info-txt ----
    it = sub.add_parser('info-txt', help='Print info for a text dataset file.')
    it.add_argument('path', help='Path to .txt file')

    args = p.parse_args()

    def _print_tokens_info(tokens, meta, path):
        print(f'File   : {path}')
        print(f'Shape  : {tokens.shape}  (n_examples × L)')
        for k, v in meta.items():
            if k not in ('n_examples', 'L'):
                print(f'  {k} = {v}')
        print(f'Unique bytes: {sorted(np.unique(tokens).tolist())}')
        row = tokens[0]
        hex_str = ' '.join(f'{b:02x}' for b in row[:32])
        asc_str = ''.join(chr(b) if 0x20 <= b < 0x7f else '.' for b in row[:32])
        print(f'Ex[0][:32] hex: {hex_str}')
        print(f'Ex[0][:32] asc: {asc_str}')

    if args.cmd == 'write':
        write_dataset(
            out_path   = args.out,
            n_examples = args.n,
            V_chain    = args.V_chain,
            L_S        = args.L_S,
            L_y        = args.L_y,
            N          = args.N,
            alpha      = args.alpha,
            seed       = args.seed,
            mode       = args.mode,
        )

    elif args.cmd == 'info':
        tokens, meta = load_dataset(args.path)
        _print_tokens_info(tokens, meta, args.path)

    elif args.cmd == 'write-txt':
        write_dataset_txt(
            out_path   = args.out,
            n_examples = args.n,
            V_chain    = args.V_chain,
            L_S        = args.L_S,
            L_y        = args.L_y,
            N          = args.N,
            alpha      = args.alpha,
            seed       = args.seed,
            mode       = args.mode,
        )

    elif args.cmd == 'info-txt':
        tokens, meta = load_dataset_txt(args.path)
        _print_tokens_info(tokens, meta, args.path)


if __name__ == '__main__':
    _cli()
