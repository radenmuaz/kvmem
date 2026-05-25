"""
Markov chain synthetic dataset for KV-as-Fast-Weights experiments.

Supports arbitrary vocabulary size V (default 256 for byte-level).
Each batch element gets a fresh random transition matrix.

Sequence layout (stage 0):
    [ x_S (L_S) | mem_tokens (N) | y (L_y) ]
    S = [0, L_S)
    M = [L_S, L_S+N)
    Y = [L_S+N, L_S+N+L_y)

Memory tokens are zeros — token identity is irrelevant, only timestep position matters.
"""

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

def sample_transition_matrix(key: jax.Array, V: int, alpha: float = 0.5) -> jax.Array:
    """
    Sample a random V×V row-stochastic transition matrix.
    Each row is drawn from Dirichlet(alpha, ..., alpha).
    Low alpha => sparse/peaky chains; alpha=1 => uniform Dirichlet.

    Args:
        key:   JAX PRNG key
        V:     vocabulary size
        alpha: Dirichlet concentration (default 0.5 → moderately peaked rows)

    Returns:
        T_mat: (V, V) float32 array, rows sum to 1
    """
    # Gamma trick for Dirichlet: Gamma(alpha) normalized
    gammas = jax.random.gamma(key, alpha, shape=(V, V))
    T_mat = gammas / gammas.sum(axis=-1, keepdims=True)
    return T_mat


def stationary_distribution(T_mat: jax.Array, n_iter: int = 1000) -> jax.Array:
    """
    Approximate stationary distribution by power iteration.

    Args:
        T_mat:  (V, V) row-stochastic transition matrix
        n_iter: number of power iterations

    Returns:
        pi: (V,) stationary distribution
    """
    V = T_mat.shape[0]
    pi = jnp.ones(V) / V
    for _ in range(n_iter):
        pi = pi @ T_mat
    return pi


def chain_entropy_bits(T_mat: jax.Array, pi: jax.Array) -> float:
    """
    Entropy rate of the Markov chain in bits/token.
    H = -sum_i pi_i sum_j T_ij log2(T_ij)
    """
    log2_T = jnp.where(T_mat > 0, jnp.log2(jnp.clip(T_mat, 1e-30)), 0.0)
    return float(-jnp.sum(pi[:, None] * T_mat * log2_T))


# ---------------------------------------------------------------------------
# Sequence sampling
# ---------------------------------------------------------------------------

def sample_chain_sequence(
    key: jax.Array,
    T_mat: jax.Array,
    start_state: int,
    length: int,
) -> jax.Array:
    """
    Walk a Markov chain for `length` steps.

    Args:
        key:         JAX PRNG key
        T_mat:       (V, V) row-stochastic transition matrix
        start_state: initial token (int in [0, V))
        length:      number of tokens to generate

    Returns:
        tokens: (length,) int32 array
    """
    def step(state, key_):
        probs = T_mat[state]
        next_state = jax.random.choice(key_, T_mat.shape[0], p=probs)
        return next_state, next_state

    keys = jax.random.split(key, length)
    _, tokens = jax.lax.scan(step, start_state, keys)
    return tokens.astype(jnp.int32)


# ---------------------------------------------------------------------------
# Single-example builders
# ---------------------------------------------------------------------------

def make_example(
    key: jax.Array,
    V: int = 256,
    L_S: int = 96,
    L_y: int = 32,
    N: int = 8,
    alpha: float = 0.5,
) -> dict:
    """
    Build one training example (stage 0 layout):
        tokens = [ x_S | zeros(N) | y ]

    y is an independent continuation from the terminal state of x_S,
    sampled from the same transition matrix.

    Args:
        key:   JAX PRNG key
        V:     vocabulary size
        L_S:   source length
        L_y:   continuation length
        N:     number of memory slots
        alpha: Dirichlet concentration for transition matrix

    Returns:
        dict with keys:
            'tokens':  (L_S + N + L_y,) int32 — full sequence
            'T_mat':   (V, V) float32 — transition matrix used
    """
    k0, k1, k2, k3 = jax.random.split(key, 4)

    T_mat = sample_transition_matrix(k0, V, alpha)

    # Sample x_S: start from uniform random state
    start_src = int(jax.random.randint(k1, (), 0, V))
    x_S = sample_chain_sequence(k2, T_mat, start_src, L_S)

    # Terminal state of source
    terminal = int(x_S[-1])

    # Independent continuation from terminal state
    y = sample_chain_sequence(k3, T_mat, terminal, L_y)

    mem_tokens = jnp.zeros(N, dtype=jnp.int32)
    tokens = jnp.concatenate([x_S, mem_tokens, y])

    return {'tokens': tokens, 'T_mat': T_mat}


# ---------------------------------------------------------------------------
# Batched builders
# ---------------------------------------------------------------------------

def make_batch(
    key: jax.Array,
    B: int,
    V: int = 256,
    L_S: int = 96,
    L_y: int = 32,
    N: int = 8,
    alpha: float = 0.5,
) -> jax.Array:
    """
    Build a batch of B training examples.

    Returns:
        tokens: (B, L_S + N + L_y) int32
    """
    keys = jax.random.split(key, B)
    examples = jax.vmap(
        lambda k: make_example(k, V, L_S, L_y, N, alpha)['tokens']
    )(keys)
    return examples


# ---------------------------------------------------------------------------
# Eval condition builders
# ---------------------------------------------------------------------------

def make_eval_matched(
    key: jax.Array,
    B: int,
    V: int = 256,
    L_S: int = 96,
    L_y: int = 32,
    N: int = 8,
    alpha: float = 0.5,
) -> jax.Array:
    """
    MATCHED: y continues the same chain as x_S.
    Same as standard make_batch — included for clarity.
    """
    return make_batch(key, B, V, L_S, L_y, N, alpha)


def make_eval_cross(
    key: jax.Array,
    B: int,
    V: int = 256,
    L_S: int = 96,
    L_y: int = 32,
    N: int = 8,
    alpha: float = 0.5,
) -> jax.Array:
    """
    CROSS: x_S and y come from *different* transition matrices.
    The memory should encode x_S's chain, but y needs a different chain —
    so the memory actively misleads. bpt_cross should be > bpt_uniform.
    """
    keys = jax.random.split(key, B)

    def one_cross(k):
        k0, k1, k2, k3, k4 = jax.random.split(k, 5)
        T_src = sample_transition_matrix(k0, V, alpha)
        T_y   = sample_transition_matrix(k1, V, alpha)  # different chain

        start_src = int(jax.random.randint(k2, (), 0, V))
        x_S = sample_chain_sequence(k3, T_src, start_src, L_S)

        start_y = int(jax.random.randint(k4, (), 0, V))
        y = sample_chain_sequence(k4, T_y, start_y, L_y)

        mem_tokens = jnp.zeros(N, dtype=jnp.int32)
        return jnp.concatenate([x_S, mem_tokens, y])

    return jax.vmap(one_cross)(keys)


def make_eval_uniform(
    key: jax.Array,
    B: int,
    V: int = 256,
    L_S: int = 96,
    L_y: int = 32,
    N: int = 8,
    alpha: float = 0.5,
) -> jax.Array:
    """
    UNIFORM: x_S is random uniform bytes (no Markov structure).
    y is from a fresh chain. Memory encodes noise, so it shouldn't help much.
    bpt_uniform serves as the baseline.
    """
    keys = jax.random.split(key, B)

    def one_uniform(k):
        k0, k1, k2, k3 = jax.random.split(k, 4)

        # x_S: uniform random bytes
        x_S = jax.random.randint(k0, (L_S,), 0, V).astype(jnp.int32)

        # y: from a fresh chain
        T_y = sample_transition_matrix(k1, V, alpha)
        start_y = int(jax.random.randint(k2, (), 0, V))
        y = sample_chain_sequence(k3, T_y, start_y, L_y)

        mem_tokens = jnp.zeros(N, dtype=jnp.int32)
        return jnp.concatenate([x_S, mem_tokens, y])

    return jax.vmap(one_uniform)(keys)


def make_eval_batches(
    key: jax.Array,
    B: int,
    V: int = 256,
    L_S: int = 96,
    L_y: int = 32,
    N: int = 8,
    alpha: float = 0.5,
) -> dict:
    """
    Build all three eval condition batches.

    Returns:
        dict with keys 'matched', 'cross', 'uniform'
        each is (B, L_S + N + L_y) int32
    """
    k0, k1, k2 = jax.random.split(key, 3)
    return {
        'matched': make_eval_matched(k0, B, V, L_S, L_y, N, alpha),
        'cross':   make_eval_cross(k1, B, V, L_S, L_y, N, alpha),
        'uniform': make_eval_uniform(k2, B, V, L_S, L_y, N, alpha),
    }


# ---------------------------------------------------------------------------
# Stage 1 multi-pass builder
# ---------------------------------------------------------------------------

def make_batch_stage1(
    key: jax.Array,
    B: int,
    T: int,
    V: int = 256,
    L_S: int = 96,
    L_y: int = 32,
    N: int = 8,
    alpha: float = 0.5,
) -> jax.Array:
    """
    Stage 1 sequence layout:
        [ x_S (L_S) | m^N | y^(1) (L_y) | m^N | y^(2) (L_y) | ... | m^N | y^(T) (L_y) ]

    Each y^(t) is an independent continuation from the terminal state of x_S,
    using a fresh PRNG key (same transition matrix, different random walk).

    Returns:
        tokens: (B, L_S + T*(N + L_y)) int32
    """
    keys = jax.random.split(key, B)

    def one_example(k):
        k0, k1, *ky_keys = jax.random.split(k, 2 + T)
        T_mat = sample_transition_matrix(k0, V, alpha)

        start_src = int(jax.random.randint(k1, (), 0, V))
        x_S = sample_chain_sequence(k1, T_mat, start_src, L_S)
        terminal = int(x_S[-1])

        mem_tokens = jnp.zeros(N, dtype=jnp.int32)
        parts = [x_S]
        for t in range(T):
            y_t = sample_chain_sequence(ky_keys[t], T_mat, terminal, L_y)
            parts.append(mem_tokens)
            parts.append(y_t)

        return jnp.concatenate(parts)

    return jax.vmap(one_example)(keys)


# ---------------------------------------------------------------------------
# Convenience: numpy versions for use outside jit
# ---------------------------------------------------------------------------

def make_batch_np(
    seed: int,
    B: int,
    V: int = 256,
    L_S: int = 96,
    L_y: int = 32,
    N: int = 8,
    alpha: float = 0.5,
) -> np.ndarray:
    """NumPy wrapper for make_batch."""
    key = jax.random.PRNGKey(seed)
    return np.array(make_batch(key, B, V, L_S, L_y, N, alpha))


def make_eval_batches_np(
    seed: int,
    B: int,
    V: int = 256,
    L_S: int = 96,
    L_y: int = 32,
    N: int = 8,
    alpha: float = 0.5,
) -> dict:
    """NumPy wrapper for make_eval_batches."""
    key = jax.random.PRNGKey(seed)
    batches = make_eval_batches(key, B, V, L_S, L_y, N, alpha)
    return {k: np.array(v) for k, v in batches.items()}
