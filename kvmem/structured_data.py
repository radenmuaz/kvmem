"""
kvmem/structured_data.py — structured-random data generators, queued as a
follow-up track to the pure-random-byte val/test data used everywhere else
in this project.

Why this exists: genuine compression (zip/gzip-style, exploiting statistical
redundancy) cannot emerge from training on max-entropy random bytes — Shannon's
source coding theorem makes such data literally incompressible, so there is no
redundancy for STATE to learn to exploit. Pure-random training only teaches
(a) raw lossless storage density and (b) the addressing algorithm — genuinely
useful, but not compression. To get emergent compression, the model needs
data with real structure. See the design discussion in this session (and
docs/SRS_RECIPE.md) for the full reasoning.

Three generator families are implemented. Each samples FRESH random
parameters per call (mirroring how random bytes are already sampled fresh per
batch elsewhere) — this is required, not optional: if the generating rule
were fixed across all training examples, the model would have no incentive to
encode anything into STATE at all (it could just bake the fixed rule into its
static weights, the exact FFN-as-static-knowledge failure mode this project's
dual-attn design already avoids elsewhere). Varying the rule per example forces
"which rule + which state, for THIS instance" to be encoded dynamically.

Selection: **cellular automata (`gen_ca`) is the recommended default** —
see rationale below `generate_structured_chunks`. The other two are kept
implemented for future ablation, not deleted.

Target-entropy calibration: each generator accepts an optional `target_bits`
(desired bits/byte of TRUE compressibility, not just marginal byte-histogram
entropy — see `measure_bits_per_byte`). marginal/unigram histogram entropy is
NOT a faithful compressibility measure: "AAAABBBB" and "ABABABAB" have
identical byte histograms but wildly different compressibility under any real
compressor. `measure_bits_per_byte` uses zlib-compressed size/byte instead,
which does capture sequential/structural redundancy, and is what
`target_bits` calibration actually optimizes against.
"""
from __future__ import annotations

import zlib

import numpy as np


def measure_bits_per_byte(seq: np.ndarray) -> float:
    """
    Practical proxy for TRUE information content (not just marginal
    byte-histogram entropy — see module docstring). Takes the MIN of two
    zlib-compressed-size/byte measurements: raw bytes, and delta-encoded
    bytes (byte[i] - byte[i-1] mod 256).

    Why delta: zlib's LZ77+Huffman only detects LITERAL repeated substrings.
    A smoothly-varying quantized signal (e.g. a fractal: 120,121,123,122,...)
    is highly predictable via local trend but has no exact repeats for LZ77
    to find — raw zlib badly underestimates its compressibility. Delta-coding
    (the same trick PNG filters and audio codecs use) turns smooth trends
    into small, repeating byte values zlib CAN exploit, without penalizing
    sequences that were already repeat-heavy in raw form (min() picks
    whichever transform actually helped).
    """
    arr = np.asarray(seq, dtype=np.uint8)
    if len(arr) == 0:
        return 0.0
    raw = bytes(arr.tolist())
    delta = bytes(np.diff(arr, prepend=np.uint8(0)).astype(np.uint8).tolist())
    bits_raw = 8.0 * len(zlib.compress(raw, level=9)) / len(raw)
    bits_delta = 8.0 * len(zlib.compress(delta, level=9)) / len(delta)
    return min(bits_raw, bits_delta)


def gen_chaotic_logistic(rng: np.random.Generator, n_bytes: int,
                         r_range: tuple[float, float] = (3.6, 4.0),
                         target_bits: float | None = None,
                         n_trials: int = 60, preview_bytes: int = 256) -> np.ndarray:
    """
    Logistic map x_{n+1} = r*x_n*(1-x_n), r sampled per-call. Default range
    (3.6, 4.0) is past the chaos onset (r>~3.57). Each step quantized to a
    byte.

    target_bits: if given, two-phase search (coarse over the WIDER range down
    to the periodic/low-entropy regime, then refine near the coarse winner)
    for an r whose short preview's measure_bits_per_byte is closest to
    target_bits, then generates the full sequence with that r.

    KNOWN LIMITATION (measured across multiple seeds, not assumed): the
    logistic map's bifurcation structure is fractal/discontinuous enough
    (bits/byte can jump from ~1.5 to ~6 between r=3.63 and r=3.64) that even
    this two-phase search lands anywhere from ~1 to ~5+ bits/byte for the
    SAME target_bits=5.0 depending on the RNG seed — n_trials=60 reduces but
    does not eliminate this. Treat target_bits here as "bias the search
    toward roughly this neighborhood," not a precise, seed-independent dial.
    A proper fix would need a precomputed bifurcation-diagram lookup table
    (r -> typical achieved bits/byte, built offline) rather than per-call
    random search — not implemented, flagged as a real gap for whoever picks
    this up next, not swept under the rug.

    Caveat (why this is NOT the recommended default): the map is defined over
    continuous reals, but training data must be exact bytes — quantizing each
    step loses precision that the *next* step's iteration doesn't see (we feed
    back the true float state internally, not the lossy byte), so the byte
    sequence is only asymptotically deterministic from an "oracle" with exact
    float access, not from the quantized bytes alone. Combined with the
    defining sensitivity-to-initial-conditions of chaos, this makes the
    byte-level sequence's "distance" from pure randomness hard to control and
    hard for a small model to actually invert in practice.
    """
    def _run(r: float, n: int) -> np.ndarray:
        x = rng.uniform(0.05, 0.95)
        out = np.empty(n, dtype=np.uint8)
        for i in range(n):
            x = r * x * (1.0 - x)
            out[i] = int(x * 255.0) & 0xFF
        return out

    if target_bits is None:
        r = rng.uniform(*r_range)
        return _run(r, n_bytes)

    # Two-phase search: the logistic map's bifurcation structure is famously
    # fractal/discontinuous (narrow periodic/intermediate-entropy windows
    # scattered unpredictably within the chaotic regime — e.g. bits/byte can
    # jump from ~1.5 to ~6 between r=3.63 and r=3.64), so a single pass of
    # uniform random sampling over the full range rarely lands in a narrow
    # band close to target_bits. Coarse pass finds the best broad region,
    # then a refine pass samples densely around it.
    search_lo, search_hi = 2.5, 4.0  # widened to reach the low-entropy/periodic regime
    coarse_n = max(1, n_trials // 2)
    best_r, best_diff = None, float('inf')
    for _ in range(coarse_n):
        r = rng.uniform(search_lo, search_hi)
        preview = _run(r, min(n_bytes, preview_bytes))
        diff = abs(measure_bits_per_byte(preview) - target_bits)
        if diff < best_diff:
            best_diff, best_r = diff, r

    refine_span = 0.05
    for _ in range(n_trials - coarse_n):
        r = float(np.clip(rng.uniform(best_r - refine_span, best_r + refine_span), search_lo, search_hi))
        preview = _run(r, min(n_bytes, preview_bytes))
        diff = abs(measure_bits_per_byte(preview) - target_bits)
        if diff < best_diff:
            best_diff, best_r = diff, r

    return _run(best_r, n_bytes)


def gen_fractal_midpoint(rng: np.random.Generator, n_bytes: int,
                         hurst_range: tuple[float, float] = (0.3, 0.9),
                         target_bits: float | None = None,
                         n_trials: int = 10, preview_bytes: int = 256) -> np.ndarray:
    """
    1D midpoint-displacement fractal (fractional-Brownian-motion-style):
    recursively bisect, perturb the midpoint by a random amount scaled by
    2^(-H*depth). H (Hurst exponent) controls roughness — sampled per call.

    target_bits: if given, search a wider H range for the value whose short
    preview's measure_bits_per_byte is closest to target_bits, then generate
    the full sequence with that H (endpoints/perturbations stay freshly
    randomized on the final generation).

    Caveat (why this is NOT the recommended default): same continuous-valued
    quantization issue as the chaotic map, plus it's naturally suited to
    smoothly-varying signals (terrain, audio-like) rather than the sharp
    byte-exact recall task this project trains on — awkward fit for the
    existing chunk-based layout without extra design work.
    """
    def _run(H: float, n: int) -> np.ndarray:
        m = 1
        while m < n:
            m *= 2
        m += 1  # m = 2^k + 1, standard midpoint-displacement grid size

        vals = np.zeros(m, dtype=np.float64)
        vals[0] = rng.uniform(0, 1)
        vals[-1] = rng.uniform(0, 1)
        step = m - 1
        scale = 1.0
        while step > 1:
            half = step // 2
            starts = np.arange(0, m - 1, step)
            mids = starts + half
            avg = (vals[starts] + vals[starts + step]) / 2.0
            perturb = rng.uniform(-0.5, 0.5, size=mids.shape) * scale
            vals[mids] = avg + perturb
            scale *= 2.0 ** (-H)
            step = half

        v = vals[:n]
        v = (v - v.min()) / (v.max() - v.min() + 1e-9)
        return (v * 255.0).astype(np.uint8)

    if target_bits is None:
        H = rng.uniform(*hurst_range)
        return _run(H, n_bytes)

    search_lo, search_hi = 0.05, 0.99
    best_H, best_diff = None, float('inf')
    for _ in range(n_trials):
        H = rng.uniform(search_lo, search_hi)
        preview = _run(H, min(n_bytes, preview_bytes))
        diff = abs(measure_bits_per_byte(preview) - target_bits)
        if diff < best_diff:
            best_diff, best_H = diff, H
    return _run(best_H, n_bytes)


def gen_ca(rng: np.random.Generator, n_bytes: int, k_states: int = 2,
          radius: int = 1, width: int = 64,
          target_bits: float | None = None,
          n_trials: int = 80, preview_bytes: int = 256) -> np.ndarray:
    """
    1D cellular automaton: k_states-ary cells, radius-r neighborhood, a
    RANDOM rule table (not a fixed rule like Wolfram Rule 30 — sampled fresh
    per call from the full k_states^(k_states^(2r+1)) rule space), random
    initial condition, wrapped (toroidal) boundary. Evolved for enough
    generations to cover n_bytes, cells packed base-k_states into bytes.

    target_bits: rule space isn't a smooth scalar knob (unlike r/H above), so
    calibration is rejection sampling instead of a range search: try
    n_trials fresh (k_states, rule_table, initial_state) draws (k_states
    itself resampled per trial from a small candidate set, since alphabet
    size strongly bounds achievable entropy), measure each preview's
    measure_bits_per_byte, keep whichever trial is closest to target_bits,
    then regenerate the FULL n_bytes with that exact winning
    (k_states, rule_table, initial_state) — not just a replayed knob, the
    entire configuration, since compressibility here depends on the whole
    rule table, not one parameter.

    KNOWN LIMITATION (measured, not assumed — see structured_data smoke
    tests): the achievable-bits/byte distribution for small-alphabet CA rules
    is bimodal/sparse in the middle, not smoothly spread — at k_states=2,
    radius=1, only ~3% of random rules land in a mid-range band like
    1.5-2.5 bits/byte (most cluster near 0-1 or 3-4). n_trials=80 (default)
    still isn't a guarantee of landing close to an arbitrary target_bits —
    treat CA's target_bits as "get roughly in the neighborhood," not a
    precise dial, unlike gen_chaotic_logistic/gen_fractal_midpoint's
    two-phase search which converges much more reliably (their knob — r or H
    — is a real, if discontinuous, scalar; CA's isn't).

    **Recommended default** (see generate_structured_chunks) — discrete-native
    (cells are exact integers, zero quantization ambiguity, unlike the two
    continuous-valued generators above), byte-exact reproducible from just
    (rule_table, initial_state) with pure integer ops, cheap to vectorize, and
    has an enormous, easily-parametrized rule space (k_states/radius directly
    control complexity — small k_states=2,radius=1 gives the classic 256
    elementary-CA rule space including famously chaotic-looking-but-low-
    -complexity rules like 30/110; larger k_states/radius scale the rule
    space combinatorially for more diversity).
    """
    def _run(k: int, r: int, rule_table: np.ndarray, init_state: np.ndarray, n: int) -> np.ndarray:
        n_neighborhood = 2 * r + 1
        bits_per_cell = max(1, int(np.ceil(np.log2(k))))
        cells_per_byte = max(1, 8 // bits_per_cell) if k <= 256 else 1
        n_gens_needed = int(np.ceil(n * cells_per_byte / width)) + 1

        powers = k ** np.arange(n_neighborhood - 1, -1, -1)
        state = init_state.copy()
        rows = []
        for _ in range(n_gens_needed):
            rows.append(state.copy())
            padded = np.concatenate([state[-r:], state, state[:r]]) if r > 0 else state
            windows = np.stack([padded[i:i + width] for i in range(n_neighborhood)], axis=0)
            idx = (windows * powers[:, None]).sum(axis=0)
            state = rule_table[idx]

        cells = np.concatenate(rows)
        n_needed_cells = n * cells_per_byte
        cells = cells[:n_needed_cells]
        if len(cells) < n_needed_cells:
            cells = np.pad(cells, (0, n_needed_cells - len(cells)))

        # Pack cells_per_byte consecutive k-ary digits into each byte
        # (base-k positional packing): byte = sum(cell_i * k^i).
        cells = cells.reshape(-1, cells_per_byte)
        byte_powers = k ** np.arange(cells_per_byte)
        packed = (cells * byte_powers[None, :]).sum(axis=1)
        return np.clip(packed, 0, 255).astype(np.uint8)[:n]

    def _draw(k: int, r: int):
        n_configs = k ** (2 * r + 1)
        rule_table = rng.integers(0, k, size=n_configs)
        init_state = rng.integers(0, k, size=width)
        return rule_table, init_state

    if target_bits is None:
        rule_table, init_state = _draw(k_states, radius)
        return _run(k_states, radius, rule_table, init_state, n_bytes)

    # Weighted toward k=2/3: empirically (see structured_data smoke tests) these
    # are the only alphabet sizes that show real rule-dependent bits/byte
    # variance at width=64 — k>=4 collapses almost immediately to a fixed
    # high-entropy value regardless of which rule was drawn, wasting trials.
    k_candidates = [2, 2, 2, 3, 3, 4]
    best_cfg, best_diff = None, float('inf')
    for _ in range(n_trials):
        k = int(rng.choice(k_candidates))
        rule_table, init_state = _draw(k, radius)
        preview = _run(k, radius, rule_table, init_state, min(n_bytes, preview_bytes))
        diff = abs(measure_bits_per_byte(preview) - target_bits)
        if diff < best_diff:
            best_diff, best_cfg = diff, (k, radius, rule_table, init_state)
    k, r, rule_table, init_state = best_cfg
    return _run(k, r, rule_table, init_state, n_bytes)


def generate_structured_chunks(rng: np.random.Generator, kind: str,
                               n_chunks: int, chunk_len: int,
                               target_bits: float | None = None) -> np.ndarray:
    """
    Dispatcher matching the (n_chunks, chunk_len) int64 shape the rest of
    kvmem/hmn.py's batch-filling code expects wherever it currently uses
    random bytes (see make_batch_tagged, make_test_sequences).

    kind: 'chaotic' | 'fractal' | 'ca' (recommended default: 'ca' — see
    gen_ca's docstring for the full rationale: discrete-native, exactly
    reproducible, enormous controllable rule-space diversity, no quantization
    ambiguity. The other two are implemented and available for a future
    ablation comparing which generator family the model actually learns
    structure from most readily, but 'ca' is the one to reach for by default.)

    target_bits: desired bits/byte of TRUE compressibility (measure_bits_per_byte,
    zlib-based — NOT marginal byte-histogram entropy, see module docstring),
    e.g. generate_structured_chunks(rng, 'ca', 8, 16, target_bits=2.0) for a
    "2-bit" CA sequence, generate_structured_chunks(rng, 'fractal', 8, 16,
    target_bits=4.0) for a "4-bit" fractal sequence. None (default) skips
    calibration and uses each generator's default parameter range.
    """
    n_bytes = n_chunks * chunk_len
    if kind == 'chaotic':
        b = gen_chaotic_logistic(rng, n_bytes, target_bits=target_bits)
    elif kind == 'fractal':
        b = gen_fractal_midpoint(rng, n_bytes, target_bits=target_bits)
    elif kind == 'ca':
        b = gen_ca(rng, n_bytes, target_bits=target_bits)
    else:
        raise ValueError(f'unknown structured data kind: {kind!r}')
    return b.reshape(n_chunks, chunk_len).astype(np.int64)
