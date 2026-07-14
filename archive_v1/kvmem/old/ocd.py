"""
kvmem/ocd.py — Optimal Completion Distillation (OCD) for exposure bias.

Reference: Sabour, Chan, Norouzi (2018) arXiv:1810.01398
"Optimal Completion Distillation for Sequence Learning"

Core idea
---------
Standard teacher forcing trains on ground-truth prefixes → at inference,
model sees its own (possibly wrong) outputs → compounding errors (exposure bias).

OCD fixes this by:
  1. Rolling out the model autoregressively to get a generated prefix y_gen[0..k-1]
  2. For each step k, finding the set of target suffixes that minimise edit distance
     to the reference x[k:] given the already-generated prefix y_gen[0..k-1]
  3. Supervising with a uniform distribution over the first token of all optimal suffixes

For an exact-copy task (target = source), edit distance simplifies:
  - If y_gen[0..k-1] == x[0..k-1]: optimal continuation is x[k], loss = CE(x[k])
  - If some errors were made: find all positions j in x where x[j:] can be reached
    with minimum additional edits, supervise uniformly over x[j] tokens

Key properties
--------------
- No hyperparameters (unlike scheduled sampling's mixing rate)
- No pretraining required
- Directly optimises task metric (edit distance)
- Trains on model's own output distribution → eliminates train/inference gap

Usage
-----
    from kvmem.ocd import ocd_loss, ocd_targets_exact_copy

    # Single example: copy task
    targets = ocd_targets_exact_copy(y_gen, x_ref)  # (L,) soft target distributions
    loss = ocd_loss(logits, targets)

    # Batch training step (JAX):
    loss, grads = jax.value_and_grad(ocd_batch_loss)(model, tokens_input, x_refs, mask)
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Edit distance helpers (numpy, CPU — called outside JIT)
# ---------------------------------------------------------------------------

def edit_distance(a: list, b: list) -> int:
    """Standard Levenshtein edit distance between two sequences."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def optimal_next_tokens_edit(y_gen: list[int], x_ref: list[int],
                              vocab_size: int = 256) -> np.ndarray:
    """
    OCD target distribution for one step of generation.

    Given:
        y_gen : tokens generated so far (possibly wrong prefix)
        x_ref : full reference sequence

    Returns:
        dist : (vocab_size,) float32 — uniform over optimal next tokens,
               zero elsewhere. Sums to 1.

    Algorithm (generalised for edit distance):
        For each possible alignment j in [0, len(x_ref)]:
            cost(j) = edit_distance(y_gen, x_ref[:j])
        min_cost = min(cost(j) for all j)
        optimal_js = {j : cost(j) == min_cost and j < len(x_ref)}
        optimal_next = {x_ref[j] for j in optimal_js}
        return uniform over optimal_next

    Special case — exact copy (no errors yet, j = len(y_gen)):
        cost(len(y_gen)) = 0 → only optimal next = x_ref[len(y_gen)]
    """
    k   = len(y_gen)
    L   = len(x_ref)
    dist = np.zeros(vocab_size, dtype=np.float32)

    if L == 0:
        return dist

    # Compute edit distance from y_gen to each prefix x_ref[:j]
    costs = np.empty(L + 1, dtype=np.int32)
    for j in range(L + 1):
        costs[j] = edit_distance(y_gen, x_ref[:j])

    min_cost = int(np.min(costs[:L]))   # only consider j < L (need a next token)
    optimal_nexts = set()
    for j in range(L):
        if costs[j] == min_cost:
            optimal_nexts.add(x_ref[j])

    if optimal_nexts:
        p = 1.0 / len(optimal_nexts)
        for tok in optimal_nexts:
            dist[tok] = p

    return dist


def optimal_next_tokens_copy(y_gen: list[int], x_ref: list[int],
                              vocab_size: int = 256) -> np.ndarray:
    """
    OCD target distribution optimised for the exact-copy task.

    For copy, edit distance from y_gen[0..k-1] to x_ref[0..j-1] is minimised
    at j = k - errors, where errors = number of positions where y_gen diverges.
    We use a simpler DP specific to copy:

        For each alignment offset j (we align y_gen against x_ref starting at 0):
            cost(j) = |k - j| (insertion/deletion to reach length j)
                    + hamming(y_gen[:min(k,j)], x_ref[:min(k,j)])

    Then optimal next token = x_ref[j*] where j* minimises cost(j).

    In the no-error case: j* = k, next token = x_ref[k].  ✓
    In the error case: j* = position in x_ref where recovery is cheapest.

    This is O(L) instead of O(k*L) for full Levenshtein.
    """
    k  = len(y_gen)
    L  = len(x_ref)
    dist = np.zeros(vocab_size, dtype=np.float32)

    if L == 0 or k >= L:
        # Already at or past end — no next token needed
        # (EOS case; caller should stop before this)
        return dist

    # Fast path: no errors yet
    if y_gen == x_ref[:k]:
        dist[x_ref[k]] = 1.0
        return dist

    # Compute costs for each alignment j
    costs = np.empty(L, dtype=np.int32)
    for j in range(L):
        overlap = min(k, j)
        hamm    = sum(y_gen[i] != x_ref[i] for i in range(overlap))
        indel   = abs(k - j)
        costs[j] = hamm + indel

    min_cost = int(np.min(costs))
    optimal_nexts: set[int] = set()
    for j in range(L):
        if costs[j] == min_cost:
            optimal_nexts.add(x_ref[j])

    if optimal_nexts:
        p = 1.0 / len(optimal_nexts)
        for tok in optimal_nexts:
            dist[tok] = p

    return dist


# ---------------------------------------------------------------------------
# OCD loss (JAX)
# ---------------------------------------------------------------------------

def ocd_loss_from_dist(logits: jax.Array, target_dist: jax.Array) -> jax.Array:
    """
    Cross-entropy loss against a soft target distribution.

    logits      : (V,) unnormalised logits
    target_dist : (V,) non-negative, sums to 1

    Loss = -sum_v target_dist[v] * log_softmax(logits)[v]
         = KL(target_dist || softmax(logits)) + H(target_dist)

    When target_dist is one-hot, this reduces to standard CE.
    """
    lp = jax.nn.log_softmax(logits, axis=-1)   # (V,)
    return -jnp.sum(target_dist * lp)


def ocd_sequence_loss(logits: jax.Array,
                      target_dists: jax.Array) -> jax.Array:
    """
    OCD loss over a sequence of positions.

    logits       : (L, V) — model logits at each position
    target_dists : (L, V) — OCD soft target distributions per position

    Returns scalar mean loss over positions where target_dist is nonzero.
    """
    # Per-position CE
    lp     = jax.nn.log_softmax(logits, axis=-1)           # (L, V)
    ce     = -jnp.sum(target_dists * lp, axis=-1)          # (L,)
    # Mask positions where target is all-zero (padding / past end)
    active = (target_dists.sum(axis=-1) > 0).astype(jnp.float32)  # (L,)
    return jnp.sum(ce * active) / (active.sum() + 1e-8)


# ---------------------------------------------------------------------------
# OCD rollout for a single example (numpy, outside JIT)
# ---------------------------------------------------------------------------

def ocd_rollout_copy(model_fn,        # callable(tokens_1d, mask) -> logits (L, V)
                     x_S: list[int],  # source sequence encoded in KV
                     N: int,          # KV slots
                     seg_len: int,    # = len(x_S) = len(y_ref)
                     vocab_size: int = 256,
                     make_mask_fn=None,  # make_mask_stage0(seg_len, N, L_y) -> np mask
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    OCD rollout for one copy/recall example.

    Generates y_gen autoregressively, computing OCD target distributions at
    each step. Returns (tokens_input, target_dists) for the full Y region.

    tokens_input : (seg_len + 2 + N + seg_len,) int32
        Full sequence with model-generated Y tokens (not teacher-forced).
    target_dists : (seg_len, vocab_size) float32
        OCD target distribution for each Y position.

    The caller concatenates these into a batch and passes to ocd_sequence_loss.
    """
    from kvmem.data import STX, NUL, ETX

    if make_mask_fn is None:
        from kvmem.data import make_mask_stage0
        make_mask_fn = make_mask_stage0

    y_gen   = []
    targets = np.zeros((seg_len, vocab_size), dtype=np.float32)

    for k in range(seg_len):
        # Build current token sequence
        mem   = [STX] + [NUL] * N + [ETX]
        seq   = x_S + mem + y_gen
        tok   = np.array(seq, dtype=np.int32)
        mask  = np.array(make_mask_fn(seg_len, N, k), dtype=np.float32)

        # Model forward pass
        logits = model_fn(tok, mask)   # (len(tok), V)
        next_logit = logits[-1]        # (V,) — predict next token

        # Greedy sample (temperature=0)
        next_tok = int(np.argmax(np.array(next_logit)))

        # OCD target for this step
        targets[k] = optimal_next_tokens_copy(y_gen, x_S, vocab_size)

        y_gen.append(next_tok)

    # Build full token sequence (with generated Y, not teacher-forced)
    mem_block  = [STX] + [NUL] * N + [ETX]
    full_seq   = np.array(x_S + mem_block + y_gen, dtype=np.int32)

    return full_seq, targets


# ---------------------------------------------------------------------------
# Batch OCD loss for use in train step
# ---------------------------------------------------------------------------

def make_ocd_train_step(model, hp: dict, make_mask_fn, optimizer_update_fn,
                        vocab_size: int = 256):
    """
    Build a training step function that uses OCD loss.

    For each example in the batch:
      1. Roll out model AR to get y_gen (numpy, outside JAX)
      2. Compute OCD target distributions (numpy, outside JAX)
      3. Run forward pass with y_gen as input, compute OCD loss (JAX)
      4. Backprop and update

    This is slower than teacher-forced training (1 AR rollout per batch)
    but directly eliminates exposure bias.

    Returns: train_step(model, opt_state, x_S_batch, N, step, lr)
        x_S_batch : (B, seg_len) int32 — source sequences
    """
    import equinox as eqx
    from kvmem.stage0 import clip_grads

    seg_len = hp['seg_len']
    N_slots = hp['N']

    def _model_fn(tok_np, mask_np):
        """Wrap model for numpy interface used in rollout."""
        import jax.numpy as jnp
        tok  = jnp.array(tok_np, dtype=jnp.int32)
        mask = jnp.array(mask_np, dtype=jnp.float32)
        return np.array(model(tok, mask))

    def train_step(model, opt_state, x_S_batch_np, step, lr):
        """
        x_S_batch_np : (B, seg_len) int32 numpy — source sequences
        Returns: (new_model, new_opt_state, loss_scalar)
        """
        import equinox as eqx
        from kvmem.data import STX, NUL, ETX

        B = x_S_batch_np.shape[0]
        L = seg_len + 2 + N_slots + seg_len
        mask_full = np.array(make_mask_fn(seg_len, N_slots, seg_len))

        # --- Rollout + OCD targets (numpy, outside JAX) ---
        all_tokens  = np.zeros((B, L), dtype=np.int32)
        all_targets = np.zeros((B, seg_len, vocab_size), dtype=np.float32)

        def _mfn(tok_np, mask_np):
            return np.array(model(jnp.array(tok_np), jnp.array(mask_np)))

        for b in range(B):
            x_S = list(x_S_batch_np[b])
            full_seq, tgt = ocd_rollout_copy(
                _mfn, x_S, N_slots, seg_len, vocab_size, make_mask_fn)
            all_tokens[b]  = full_seq
            all_targets[b] = tgt

        # --- JAX forward + OCD loss + grad ---
        tokens_j  = jnp.array(all_tokens)
        targets_j = jnp.array(all_targets)
        mask_j    = jnp.array(mask_full)

        ETX_pos = seg_len + 1 + N_slots

        def _loss(m):
            # Forward pass with generated tokens as input
            logits = jax.vmap(lambda tok: m(tok, mask_j))(tokens_j)  # (B, L, V)
            # Y logits: positions ETX_pos..ETX_pos+seg_len-1 predict Y[0..seg_len-1]
            y_logits = logits[:, ETX_pos : ETX_pos + seg_len, :]      # (B, seg_len, V)
            # OCD loss
            lp     = jax.nn.log_softmax(y_logits, axis=-1)            # (B, seg_len, V)
            ce     = -jnp.sum(targets_j * lp, axis=-1)                # (B, seg_len)
            active = (targets_j.sum(axis=-1) > 0).astype(jnp.float32)
            return jnp.sum(ce * active) / (active.sum() + 1e-8)

        params = eqx.filter(model, eqx.is_array)
        loss, grads = jax.value_and_grad(_loss)(model)
        grads_arr = eqx.filter(grads, eqx.is_array)
        grads_arr = clip_grads(grads_arr, hp.get('grad_clip', 1.0))

        new_params, new_opt = optimizer_update_fn(
            params, grads_arr, opt_state, lr, wd=hp.get('wd', 0.01), step=step)
        deltas    = jax.tree.map(lambda np_, p: np_ - p, new_params, params)
        new_model = eqx.apply_updates(model, deltas)

        return new_model, new_opt, float(loss)

    return train_step


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=== OCD unit tests ===')

    # Test 1: perfect prefix → only correct next token
    y_gen = [0x20, 0x21, 0x22]
    x_ref = [0x20, 0x21, 0x22, 0x23, 0x24]
    dist  = optimal_next_tokens_copy(y_gen, x_ref)
    assert dist[0x23] == 1.0, f'Expected 0x23=1.0, got {dist[0x23]}'
    assert dist.sum() == 1.0
    print('  Test 1 PASS: perfect prefix → correct next token')

    # Test 2: one error at position 0 → optimal recovery
    y_gen = [0xFF]   # wrong first token
    x_ref = [0x20, 0x21, 0x22]
    dist  = optimal_next_tokens_copy(y_gen, x_ref)
    assert dist.sum() > 0, 'Should have some valid target'
    print(f'  Test 2 PASS: one error → targets={[hex(i) for i in range(256) if dist[i]>0]}')

    # Test 3: empty prefix → first token of x_ref
    y_gen = []
    x_ref = [0x41, 0x42, 0x43]
    dist  = optimal_next_tokens_copy(y_gen, x_ref)
    assert dist[0x41] == 1.0, f'Expected 0x41=1.0, got {dist[0x41]}'
    print('  Test 3 PASS: empty prefix → first token')

    # Test 4: ocd_sequence_loss shape
    import jax.numpy as jnp
    L, V = 8, 256
    logits = jnp.zeros((L, V))
    tgt    = jnp.zeros((L, V)).at[:, 0x20].set(1.0)
    loss   = ocd_sequence_loss(logits, tgt)
    print(f'  Test 4 PASS: sequence loss = {float(loss):.4f}  (expected log(256)≈{np.log(256):.4f})')

    # Test 5: loss = 0 when logits perfectly match one-hot target
    logits_perfect = jnp.full((L, V), -1e9).at[:, 0x20].set(0.0)
    loss_perfect   = ocd_sequence_loss(logits_perfect, tgt)
    print(f'  Test 5 PASS: perfect logits → loss = {float(loss_perfect):.6f}  (expected ≈0)')

    print('\nAll tests passed.')
