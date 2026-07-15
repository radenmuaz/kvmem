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

Nine generator families are implemented; five more (`gen_template_repeat`,
`gen_chaotic_3d`, `gen_fractal_2d`, `gen_ca_nd`, `gen_symbolic_equation`) are
documented placeholders (each raises `NotImplementedError` with a full
planned-design docstring — see each function). Every implemented generator
samples FRESH random
parameters per call (mirroring how random bytes are already sampled fresh per
batch elsewhere) — this is required, not optional: if the generating rule
were fixed across all training examples, the model would have no incentive to
encode anything into STATE at all (it could just bake the fixed rule into its
static weights, the exact FFN-as-static-knowledge failure mode this project's
dual-attn design already avoids elsewhere). Varying the rule per example forces
"which rule + which state, for THIS instance" to be encoded dynamically.

Selection: **cellular automata (`gen_ca`) is the recommended default** —
see rationale below `generate_structured_chunks`. `gen_markov` is the
recommended choice specifically when PRECISE `target_bits` calibration
matters more than generator diversity (see its docstring — closed-form
entropy rate, no measure-and-search needed). The others are kept
implemented for future ablation, not deleted.

Target-entropy calibration: each generator accepts an optional `target_bits`
(desired bits/byte of TRUE compressibility, not just marginal byte-histogram
entropy — see `measure_bits_per_byte`). marginal/unigram histogram entropy is
NOT a faithful compressibility measure: "AAAABBBB" and "ABABABAB" have
identical byte histograms but wildly different compressibility under any real
compressor. `measure_bits_per_byte` uses zlib-compressed size/byte instead,
which does capture sequential/structural redundancy, and is what
`target_bits` calibration actually optimizes against for `gen_chaotic_logistic`/
`gen_fractal_midpoint`/`gen_ca` (all measure-and-search based). `gen_markov`
is the exception — its entropy RATE has a closed form, so its calibration is
a 1D bisection against the exact theoretical value, not a `measure_bits_per_byte`
search. For `gen_markov` specifically, `measure_bits_per_byte` is NOT a valid
sanity check (verified empirically, see its docstring) — DEFLATE's Huffman
stage codes against marginal/global byte frequency, not the previous byte,
so it is structurally blind to the order-1 CONDITIONAL structure
`gen_markov`'s target_bits actually controls, which is exactly the kind of
structure a context-conditional (attention-based) model can exploit.
"""
from __future__ import annotations

import base64
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


def save_sequence_text(path: str, arr: np.ndarray) -> None:
    """
    Saves ANY numpy array (any dtype/shape — uint8 from a raw gen_* call,
    or int64 from generate_structured_chunks' reshaped output) to a plain-
    ASCII text file via base64, exactly reversible by load_sequence_text
    (byte-identical dtype/shape/values on reload — not just "close").

    ~33% larger on disk than raw binary (base64's 4-ASCII-chars-per-3-bytes
    ratio) in exchange for a portable, diffable, git-friendly, editor-safe
    plain-text format. Zero training-time cost either way — this is only
    ever used to persist/inspect a generated sequence outside the training
    loop; the array gets decoded back to its exact original bytes before
    anything downstream touches it.

    File format (3 lines, all ASCII, human-inspectable except line 3):
      1: dtype name (e.g. "uint8", "int64")
      2: shape, comma-separated (e.g. "8,16")
      3: base64(arr.tobytes())
    """
    with open(path, 'w') as f:
        f.write(str(arr.dtype) + '\n')
        f.write(','.join(str(d) for d in arr.shape) + '\n')
        f.write(base64.b64encode(arr.tobytes()).decode('ascii') + '\n')


def load_sequence_text(path: str) -> np.ndarray:
    """
    Inverse of save_sequence_text — reconstructs a byte-identical array
    (same dtype, same shape, same values) from a file it wrote.
    """
    with open(path, 'r') as f:
        dtype_str = f.readline().strip()
        shape_str = f.readline().strip()
        b64_data = f.readline().strip()
    shape = tuple(int(d) for d in shape_str.split(','))
    raw = base64.b64decode(b64_data)
    arr = np.frombuffer(raw, dtype=np.dtype(dtype_str))
    return arr.reshape(shape)


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


def gen_chaotic_3d(rng: np.random.Generator, n_bytes: int,
                   target_bits: float | None = None, **kwargs) -> np.ndarray:
    """
    PLACEHOLDER — not yet implemented.

    Planned design: the Lorenz system (dx/dt = sigma*(y-x), dy/dt =
    x*(rho-z) - y, dz/dt = x*y - beta*z), NOT three independently-coupled
    1D logistic maps — this is the canonical, well-studied 3D chaotic
    system (a genuine strange attractor with fractal dimension, not just
    three parallel 1D chaotic sequences), numerically integrated via RK4
    at a small fixed timestep, each of (x,y,z) quantized to a byte per
    integration step. `rho` is the natural chaos-strength knob (plays the
    role gen_chaotic_logistic's `r` plays — classic chaotic regime near
    rho=28, sigma=10, beta=8/3, but chaotic across a wider neighborhood).

    "Tuple structure": each timestep emits a genuine (x,y,z) tuple, coupled
    through the ODE's cross terms (x,y,z appear in each other's derivatives)
    — NOT independent per-axis sequences. How the 3 bytes get INTERLEAVED
    into the output stream is a real design decision, not a footnote:
    interleaved (x0,y0,z0,x1,y1,z1,...) keeps same-timestep coordinates
    adjacent, so their coupled correlation stays within a local attention
    window; blocked (x0..xn,y0..yn,z0..zn) would separate them into three
    widely-spaced regions, likely destroying exactly the correlation
    structure that makes this interesting versus three unrelated 1D
    sequences. Planned default: interleaved.

    Same quantization-vs-chaos caveat as gen_chaotic_logistic (see that
    docstring) applies here too, likely worse — sensitivity to initial
    conditions compounds across 3 coupled axes, not just 1, so exact
    long-range extrapolation from a quantized (lossy) observed prefix is
    even less reliable than the 1D case. Numerical RK4 integration itself
    (not yet in this module — nothing here does ODE integration currently)
    is the concrete implementation dependency blocking this.
    """
    raise NotImplementedError(
        'gen_chaotic_3d is a placeholder — see its docstring for the '
        'planned design (Lorenz system, RK4 integration, interleaved '
        '(x,y,z) tuple output) and the quantization/chaos caveats that '
        'apply even more strongly here than in the 1D case.')


def gen_fractal_2d(rng: np.random.Generator, n_bytes: int,
                   target_bits: float | None = None, **kwargs) -> np.ndarray:
    """
    PLACEHOLDER — not yet implemented.

    Planned design: the diamond-square algorithm, the standard 2D
    generalization of 1D midpoint displacement (same Hurst-exponent
    roughness knob, now recursively bisecting a (2^k+1) x (2^k+1) grid via
    alternating "diamond" and "square" midpoint-perturbation steps instead
    of 1D bisection). Flattened to a 1D byte stream via ROW-MAJOR raster
    scan — this is the whole point of doing this generalization at all
    (see below), not an incidental detail.

    Why this is the highest-priority of the three N-D placeholders (see
    the "generalize to N-D" discussion this session): row-major flattening
    of a 2D grid creates a STRUCTURAL, non-arbitrary long-range dependency
    — byte i correlates with byte (i - grid_width) (the cell directly
    above it in the grid), not an arbitrary synthetic match-distance like
    gen_match_distance's. Setting chunk_len < grid_width would force a
    single grid ROW to span multiple chunks, making correct row-dependent
    prediction require genuine cross-chunk relay — a sharper, more
    structurally-motivated stress test for the `hop` relay mechanism than
    anything currently in this module. This is the concrete reason to
    build this generator before gen_chaotic_3d/gen_ca_nd.

    "Tuple structure": planned as CORRELATED MULTI-LAYER fields (e.g.
    height + moisture + temperature, the classic procedural-terrain
    pattern — each layer its own diamond-square grid, but sharing the same
    perturbation randomness/scale schedule so the layers correlate) rather
    than a single scalar per grid cell. Natural fit for "tuple" here,
    unlike gen_chaotic_3d's genuinely-coupled-ODE tuple or gen_ca_nd's
    independent-channel-layers tuple — three different "what does tuple
    mean" answers for three different generator families, each documented
    on its own placeholder rather than forcing one shared convention.

    Flattening-strategy caveat, to resolve at implementation time not
    glossed over: row-major is simplest but a space-filling curve
    (Hilbert/Z-order) would better preserve 2D locality in BOTH directions
    (row-major only preserves horizontal adjacency well; vertical
    neighbors end up grid_width apart) — row-major is the right starting
    choice specifically BECAUSE that asymmetry is the intended,
    controllable long-range-dependency mechanism, not a flaw to fix.
    """
    raise NotImplementedError(
        'gen_fractal_2d is a placeholder — see its docstring for the '
        'planned design (diamond-square algorithm, row-major flattening '
        'as the deliberate source of a structural long-range dependency '
        'at distance=grid_width, correlated multi-layer tuple output) — '
        'the highest-priority of the three N-D generalization placeholders.')


def gen_ca_nd(rng: np.random.Generator, n_bytes: int,
             n_dims: int = 4, target_bits: float | None = None,
             **kwargs) -> np.ndarray:
    """
    PLACEHOLDER — not yet implemented.

    Planned design: generalizes gen_ca to an N-dimensional grid (default
    n_dims=4). Cellular automata generalize cleanly to N dimensions in
    principle (Conway's Life is the famous 2D case), but rule-table size
    is the binding tractability constraint, and it is dimension-sensitive
    in a way that MUST be handled explicitly, not left as a footnote:
    - a MOORE neighborhood (all cells within radius r along every axis,
      including diagonals) has (2r+1)^n_dims cells — at n_dims=4, r=1,
      that's 3^4=81 neighbors, making even k_states=2 give 2^81 possible
      rules, completely intractable to rejection-sample.
    - a VON NEUMANN neighborhood (axis-aligned only, no diagonals) has
      2*n_dims*r+1 cells — at n_dims=4, r=1, that's 9 neighbors, keeping
      k^9 tractable (512 rules at k=2 — same order of magnitude gen_ca
      already samples successfully at 1D/r=1).
    PLANNED: default to von Neumann neighborhoods for n_dims >= 3,
    matching gen_ca's existing k=2/3-weighted rejection-sampling approach
    for target_bits calibration (rule space isn't a scalar knob here
    either) — Moore neighborhoods should require an explicit opt-in with a
    loud warning about the combinatorial blowup, not be the default.

    Flattening: N nested-axis raster scan (outermost to innermost
    dimension) — same row-major-style choice as gen_fractal_2d, but with
    N-1 distinct "gap sizes" simultaneously (nearest-axis neighbor = 1
    byte back, next axis = size_along_that_axis bytes back, etc.),
    creating genuinely MULTI-SCALE long-range dependencies without any
    hand-designed match-distance knob — a natural realization of
    LANGUAGE.md's Level 6 "long-range dependencies at multiple scales"
    discussion (its chapter-title/table-of-contents example), IF this
    generator is ever needed for that specifically; gen_fractal_2d is the
    lower-cost generator to reach for first for the single-scale version
    of this same idea.

    "Tuple structure": planned as multiple INDEPENDENTLY-evolving CA
    layers (each its own rule table, own initial condition) interleaved
    per cell — like RGB channels — rather than one CA whose rule table
    reads a coupled multi-channel neighborhood (which would multiply the
    rule-table size by (channels)^(neighbors), compounding the
    tractability problem this function already has to manage carefully
    for the spatial dimensionality alone).

    n_dims=4 default is not privileged — 2D/3D CA (Conway's-Life-style and
    its 3D generalization) are also unbuilt and would need this same
    machinery; exposed here as a single N-dimensional implementation
    rather than separate 2D/3D/4D functions once built.
    """
    raise NotImplementedError(
        'gen_ca_nd is a placeholder — see its docstring for the planned '
        'design (von-Neumann-neighborhood default for tractable rule-table '
        'size at n_dims>=3, N-nested-axis raster flattening for '
        'multi-scale long-range structure, independent-layer tuple output) '
        '— lowest-priority of the three N-D generalization placeholders '
        'given rule-table tractability is a genuinely hard constraint here.')


def gen_markov(rng: np.random.Generator, n_bytes: int,
              target_bits: float | None = None,
              temperature_range: tuple[float, float] = (0.1, 8.0),
              entropy_tol: float = 0.02, max_bisect_iters: int = 40) -> np.ndarray:
    """
    Order-1 Markov chain over the full byte alphabet (K=256): sample a
    random base transition matrix (Dirichlet(alpha=1) per row — uniform
    over the probability simplex, fresh per call, same "fresh parameters
    per call" discipline as every other generator here), then temper it
    row-wise by a single scalar temperature T (P(j|i) ∝ base(j|i)^(1/T))
    to control the chain's entropy RATE. T→0 sharpens rows toward one-hot
    (entropy→0); T→∞ flattens rows toward uniform (entropy→log2(256)=8
    bits); entropy rate is monotonic in T.

    Why this generator exists (see LANGUAGE.md's maximum-entropy framing
    and its Level-2 Markov-chain discussion): unlike
    gen_chaotic_logistic/gen_fractal_midpoint/gen_ca, a Markov chain's
    entropy rate has a CLOSED FORM —
        H = -sum_i pi_i * sum_j P(j|i) * log2(P(j|i))
    where pi is the stationary distribution (power iteration on the
    transition matrix — cheap at K=256, converges in well under
    max_bisect_iters * a few hundred small matmuls). Because entropy rate
    is monotonic in T, target_bits calibration is a 1D BISECTION SEARCH
    against this exact theoretical value — no zlib measurement, no
    generating and re-measuring candidate sequences, no seed-dependent
    variance. This directly avoids the "known limitation" flagged in every
    other generator's docstring (calibration accuracy is seed-dependent
    and imprecise because their entropy can only be estimated
    empirically). Calibration here converges to within `entropy_tol` bits
    of target_bits (default 0.02), far tighter than what measure-and-search
    achieves for the other three.

    IMPORTANT, measured not assumed: `measure_bits_per_byte` (zlib/DEFLATE)
    is NOT a valid sanity check here, unlike for the other three
    generators — verified empirically (target_bits=1/2/4/6 gave zlib
    readings of 5.6/7.3/8.2/8.2, essentially flat and uninformative).
    DEFLATE's Huffman stage codes against GLOBAL/marginal byte frequency
    (same code for a byte regardless of what preceded it) plus LZ77
    literal-substring matching — neither sees ORDER-1 CONDITIONAL
    structure (byte i's distribution given byte i-1) unless it also
    happens to skew the marginal distribution or create literal repeats,
    which a per-row-tempered transition matrix generally does not (the
    stationary distribution over all 256 bytes stays close to uniform even
    as row-conditional entropy drops sharply — confirmed empirically:
    marginal entropy stayed ~7-8 bits across all four target_bits above,
    while the actual empirical order-1 conditional entropy tracked
    target_bits correctly: ~0.85/1.6/2.8/3.6 bits for targets 1/2/4/6).
    This is not a bug in the calibration — target_bits here genuinely
    controls the theoretical entropy a CONTEXT-CONDITIONAL model can
    exploit (exactly what this project's own attention-based HMN
    architecture is, much closer to a PPM/order-k model than to DEFLATE),
    it just means zlib is the wrong instrument to verify it with. Use a
    direct empirical order-1 conditional-entropy estimate instead if a
    sanity check is needed.

    target_bits=None: skip calibration, sample T uniformly from
    temperature_range (mirrors the other generators' "unset -> random
    default range" behavior).
    """
    K = 256

    def _temper(base: np.ndarray, T: float) -> np.ndarray:
        logp = np.log(np.clip(base, 1e-300, None)) / T
        logp -= logp.max(axis=1, keepdims=True)  # numerical stability
        p = np.exp(logp)
        return p / p.sum(axis=1, keepdims=True)

    def _stationary(P: np.ndarray, n_iters: int = 300, tol: float = 1e-9) -> np.ndarray:
        pi = np.full(K, 1.0 / K)
        for _ in range(n_iters):
            pi_next = pi @ P
            if np.abs(pi_next - pi).sum() < tol:
                return pi_next
            pi = pi_next
        return pi

    def _entropy_rate(P: np.ndarray) -> float:
        pi = _stationary(P)
        row_h = -(P * np.log2(np.clip(P, 1e-300, None))).sum(axis=1)
        return float((pi * row_h).sum())

    def _sample_sequence(P: np.ndarray, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.uint8)
        state = int(rng.integers(0, K))
        cdfs = np.cumsum(P, axis=1)
        u = rng.random(n)
        for i in range(n):
            nxt = int(np.searchsorted(cdfs[state], u[i]))
            nxt = min(nxt, K - 1)
            out[i] = nxt
            state = nxt
        return out

    base = rng.dirichlet(np.ones(K), size=K)

    if target_bits is None:
        T = rng.uniform(*temperature_range)
        return _sample_sequence(_temper(base, T), n_bytes)

    lo, hi = 1e-3, 100.0
    mid = 1.0
    for _ in range(max_bisect_iters):
        mid = (lo + hi) / 2.0
        h = _entropy_rate(_temper(base, mid))
        if abs(h - target_bits) < entropy_tol:
            break
        if h < target_bits:
            lo = mid
        else:
            hi = mid
    return _sample_sequence(_temper(base, mid), n_bytes)


def gen_template_repeat(rng: np.random.Generator, n_bytes: int,
                        target_bits: float | None = None, **kwargs) -> np.ndarray:
    """
    PLACEHOLDER — not yet implemented.

    Planned design (LANGUAGE.md's Level 5, "hierarchical repetition"):
    sample a small vocabulary of random byte "phrases" (templates), then
    randomly concatenate them to fill n_bytes. Produces genuine
    LZ77-exploitable repeated-SUBSTRING structure — qualitatively
    different from and complementary to the smooth/local structure every
    other generator here produces (chaotic map: smooth trend;
    fractal: smooth trend at multiple scales; CA: local propagated rule;
    Markov: local conditional-frequency structure). Entropy controllable
    via (vocabulary size, template length, concatenation randomness) —
    fewer/longer templates = more compressible; likely has a
    closed-form-ish target_bits calibration in the same spirit as
    gen_markov (vocabulary size directly bounds achievable entropy per
    template choice, roughly log2(vocab_size) bits per template of known
    length), not worked out yet.

    IMPORTANT — recovery-probe contamination risk (see
    docs/HISTORY.md's caution on structured data): a template generator
    is a WORSE case than gen_ca/gen_markov for this. If a template
    literally repeats byte-for-byte, a model could "recover" it via rote
    memorization of that one byte pattern, independent of whether any
    relay/STATE mechanism carried anything at all. Template CONTENT (not
    just structure/parameters) must stay fresh-per-call, matching the same
    discipline already enforced for every other generator's parameters —
    this needs to be a deliberate design constraint here, not just a
    default consequence of "fresh RNG state," since the whole point of
    this generator IS repetition.
    """
    raise NotImplementedError(
        'gen_template_repeat is a placeholder — see its docstring for the '
        'planned design (LANGUAGE.md Level 5, template/phrase repetition) '
        'and the recovery-probe contamination risk that needs resolving '
        'before implementation.')


def gen_symbolic_equation(rng: np.random.Generator, n_bytes: int,
                          target_bits: float | None = None,
                          max_terms: int = 3, **kwargs) -> np.ndarray:
    """
    PLACEHOLDER — not yet implemented.

    Planned design: sample a random SYMBOLIC EQUATION (not just parameters
    within one fixed functional family, unlike every other generator in
    this module) from a small primitive library relevant to physics/
    engineering signals, evaluate it numerically over t=0..n_bytes-1,
    min-max normalize (same trick gen_fractal_midpoint already uses) and
    quantize to bytes. This is a data generator in the spirit of the
    symbolic-regression / neural-symbolic-regression research literature
    (e.g. Lample & Charton's "Deep Learning for Symbolic Mathematics," AI
    Feynman, and follow-on neural-symbolic-regression work that trains
    transformers to recover symbolic form from sampled data) — repurposed
    here as a compressibility source rather than a symbolic-recovery
    training target, but the underlying idea (many different closed-form
    equations, randomized coefficients, evaluated numerically) is the same.

    Why this is a DIFFERENT CATEGORY of compressibility than everything
    else in this module, not just another family: every other generator
    varies PARAMETERS within one fixed functional form (a transition
    matrix, a rule table, a Hurst exponent). This varies the FUNCTIONAL
    FORM ITSELF — one draw might be a damped oscillator, another a sum of
    two incommensurate sine waves, another a power law times an
    exponential. A symbolic equation with k free parameters, evaluated at
    N points, has TRUE information content bounded by roughly
    k * bits_per_parameter, INDEPENDENT OF N — e.g. a 6-parameter damped-
    oscillator equation evaluated at 10,000 points is still only ~192 bits
    of real information (6 params * 32 bits), a Kolmogorov-complexity-
    style compression ratio more extreme than anything the statistical
    (Markov/CA) generators can express. `STATE`'s job under this framing
    isn't "learn the statistics," it's "identify which equation and which
    parameters" — closer to what kvmem/eval_compression.py's diagnostics
    are actually trying to measure than the current generator suite tests.

    This also formalizes, more rigorously than a bespoke suggestion could,
    the "can the model extrapolate a rule rather than just recall it"
    question raised earlier this session (in connection with the
    up_counter/down_counter eval sequences and gen_ca's local, tractably-
    inferable rule structure) — "recover parameters from a partial
    observation and extrapolate correctly" IS the neural-symbolic-
    regression framing, now with an established literature to draw design
    choices from instead of an ad hoc probe.

    Planned primitive library (small, physics/engineering-flavored, not
    an open-ended CAS):
      - oscillatory:       A*sin(w*t + phi),  A*cos(w*t + phi)
      - exponential:       A*exp(k*t)                    (growth/decay)
      - damped oscillator: A*exp(-k*t)*sin(w*t + phi)     (RLC/mechanical)
      - polynomial:        sum_i(A_i * t^i), bounded degree
      - power law:         A*t^p
    Sample a random shallow COMBINATION of `max_terms` primitives (sum or
    product, bounded tree depth — a numerical analogue of LANGUAGE.md's
    Level-4 grammar-production-rule idea, but over functions instead of
    syntax/lexicon), sample each primitive's own coefficients from
    reasonable ranges, evaluate, normalize, quantize.

    target_bits calibration: HARD, flagged honestly rather than assumed
    solvable. The equation-template space is combinatorial/discrete (which
    primitives, how combined, `max_terms`) — same fundamental issue as
    gen_ca's rule space not being a scalar knob — so this would need
    measure-and-search (via measure_bits_per_byte, same delta-then-zlib
    trick gen_chaotic_logistic/gen_fractal_midpoint already use for smooth
    signals) rather than a closed form. Even that measurement is somewhat
    dubious for the same quantization reasons gen_chaotic_logistic's
    docstring already flags — a smooth analytic signal's TRUE information
    content (the Kolmogorov-style argument above) can be far below what
    zlib actually achieves on one quantized, finite realization of it.
    Treat any eventual target_bits calibration here as the LEAST reliable
    of every generator in this module, more so than gen_chaotic_logistic's
    already-documented seed-dependent imprecision — a genuine, currently
    unsolved problem, not swept under the rug.

    Larger implementation dependency than any of the five generators
    already built this session: needs a small expression-tree
    sampler/evaluator (nothing in this module currently represents or
    evaluates a symbolic expression tree — chaotic/fractal/CA are each one
    fixed hand-written recurrence, not a composable primitive library).
    """
    raise NotImplementedError(
        'gen_symbolic_equation is a placeholder — see its docstring for '
        'the planned design (small physics/engineering primitive library, '
        'random shallow equation-tree sampling, connection to the '
        'symbolic-regression research literature) and why its target_bits '
        'calibration is flagged as an unsolved problem, not just an '
        'approximation like gen_run_length/gen_match_distance/gen_mixed_order.')


def gen_iid_skewed(rng: np.random.Generator, n_bytes: int,
                   target_bits: float | None = None,
                   temperature_range: tuple[float, float] = (0.1, 8.0),
                   entropy_tol: float = 0.01, max_bisect_iters: int = 40) -> np.ndarray:
    """
    I.I.D. bytes sampled from a single skewed categorical distribution over
    the 256-byte alphabet (Zipf-like — a random base distribution tempered
    by a scalar T, the same tempering trick gen_markov uses, applied to ONE
    row instead of 256).

    This is the deliberate "control case zlib CAN see," complementary to
    gen_markov's "control case zlib can't." The redundancy here is entirely
    in the MARGINAL/unconditional frequency distribution — exactly what
    DEFLATE's Huffman stage codes against — so measure_bits_per_byte
    should track target_bits reasonably closely here, unlike for
    gen_markov (see that function's docstring for the measured
    counterexample).

    Closed-form entropy: H = -sum_i p_i * log2(p_i) (ordinary Shannon
    entropy of the marginal distribution — no stationary-distribution
    power iteration needed, since samples are independent). target_bits
    calibration is a 1D bisection on T against this exact value, same
    precision guarantee as gen_markov.

    target_bits=None: skip calibration, sample T uniformly from
    temperature_range.
    """
    K = 256
    base = rng.dirichlet(np.ones(K))

    def _temper(T: float) -> np.ndarray:
        logp = np.log(np.clip(base, 1e-300, None)) / T
        logp -= logp.max()
        p = np.exp(logp)
        return p / p.sum()

    def _entropy(p: np.ndarray) -> float:
        return float(-(p * np.log2(np.clip(p, 1e-300, None))).sum())

    if target_bits is None:
        T = rng.uniform(*temperature_range)
        return rng.choice(K, size=n_bytes, p=_temper(T)).astype(np.uint8)

    lo, hi = 1e-3, 100.0
    mid = 1.0
    for _ in range(max_bisect_iters):
        mid = (lo + hi) / 2.0
        h = _entropy(_temper(mid))
        if abs(h - target_bits) < entropy_tol:
            break
        if h < target_bits:
            lo = mid
        else:
            hi = mid
    return rng.choice(K, size=n_bytes, p=_temper(mid)).astype(np.uint8)


def gen_run_length(rng: np.random.Generator, n_bytes: int,
                   target_bits: float | None = None,
                   mean_run_range: tuple[float, float] = (1.0, 64.0),
                   entropy_tol: float = 0.1, max_bisect_iters: int = 30) -> np.ndarray:
    """
    Run-length structure: repeatedly emit a fresh uniform-random byte,
    followed by a run of (run_length - 1) copies of it (run_length ~
    Geometric(p), mean 1/p) — repeat until n_bytes is filled. Pure
    RLE/LZ77-visible redundancy: each run IS a literal repeated substring,
    so DEFLATE's LZ77 stage should detect it directly, unlike gen_markov's
    conditional structure (see that docstring) — this is a second,
    differently-mechanismed "zlib-visible" case alongside gen_iid_skewed
    (Huffman-visible via marginal skew) vs. this one (LZ77-visible via
    literal repeats).

    target_bits calibration: APPROXIMATE closed form (measured, not
    assumed to be bit-precise — treat as directionally reliable, unlike
    gen_markov/gen_iid_skewed's exact bisection). Idealized description
    length per run ~= 8 bits (fresh symbol, uniform over 256) +
    H_geom(p) (entropy of the geometric run-length distribution itself),
    amortized over mean_run_length = 1/p bytes:
        bits_per_byte(mean_run) ~= (8 + H_geom(1/mean_run)) / mean_run
    This is what an idealized RLE-optimal encoder would achieve, not
    necessarily what DEFLATE specifically achieves (it has its own
    framing/quantization overhead) — bisection is on mean_run_length
    against this estimate, not against a measured value.
    """
    def _h_geom(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return float(-((1 - p) * np.log2(1 - p) + p * np.log2(p)) / p)

    def _bits_per_byte(mean_run: float) -> float:
        p = 1.0 / mean_run
        return (8.0 + _h_geom(p)) / mean_run

    def _generate(mean_run: float, n: int) -> np.ndarray:
        p = 1.0 / mean_run
        out = np.empty(n, dtype=np.uint8)
        i = 0
        while i < n:
            sym = rng.integers(0, 256)
            run_len = max(1, min(int(rng.geometric(p)), n - i))
            out[i:i + run_len] = sym
            i += run_len
        return out

    if target_bits is None:
        mean_run = rng.uniform(*mean_run_range)
        return _generate(mean_run, n_bytes)

    lo, hi = mean_run_range
    mid = (lo + hi) / 2.0
    for _ in range(max_bisect_iters):
        mid = (lo + hi) / 2.0
        b = _bits_per_byte(mid)
        if abs(b - target_bits) < entropy_tol:
            break
        if b > target_bits:
            lo = mid  # need longer runs (more compression) to reduce bits
        else:
            hi = mid
    return _generate(mid, n_bytes)


def gen_markov_order_k(rng: np.random.Generator, n_bytes: int,
                       order: int = 2, K: int = 4,
                       target_bits: float | None = None,
                       temperature_range: tuple[float, float] = (0.1, 8.0),
                       entropy_tol: float = 0.02, max_bisect_iters: int = 40) -> np.ndarray:
    """
    Generalizes gen_markov to CONTEXT LENGTH > 1 (next symbol's
    distribution depends on the last `order` symbols, not just the last
    1). Alphabet size K is kept SMALL (default 4) since the meta-state
    space is K^order — same "restrict alphabet for tractability at higher
    order" tradeoff gen_ca already makes for its radius parameter. Symbols
    are packed base-K into output bytes (same positional-packing scheme
    gen_ca uses) since K rarely aligns cleanly to a byte.

    Mechanically: each length-`order` context is a base-K integer
    "meta-state" in [0, K^order). The transition matrix P has shape
    (K^order, K) — a distribution over the NEXT SYMBOL only (never a full
    (K^order, K^order) matrix; the next meta-state is always the
    deterministic shift-and-append of the current context with the
    sampled symbol, so this stays tractable even at order=3-4 with small
    K). Entropy rate and target_bits calibration use the SAME closed-form
    + bisection approach as gen_markov (stationary distribution via power
    iteration over meta-states, fresh-per-call Dirichlet-tempered rows) —
    see gen_markov's docstring for the full rationale; this is a direct
    generalization, not a different technique. target_bits is interpreted
    per OUTPUT BYTE (converted internally to per-symbol via
    cells_per_byte, then capped at log2(K), the alphabet's own ceiling).

    Same measure_bits_per_byte CAVEAT as gen_markov: DEFLATE's Huffman
    stage is blind to conditional structure unless it happens to also skew
    the marginal distribution — not separately re-verified for order>1
    here, but there's no reason to expect zlib behaves differently.

    order/K are STRUCTURAL parameters (not part of target_bits calibration
    — analogous to gen_ca's k_states/radius being separate from its own
    target_bits search). Asserted K**order <= 4096 to keep power iteration
    cheap; raise with care (cost grows as K^order * K per iteration).
    """
    assert K ** order <= 4096, f'K^order={K ** order} too large (meta-state space explodes) — keep K/order small'
    n_ctx = K ** order
    bits_per_cell = max(1, int(np.ceil(np.log2(K))))
    cells_per_byte = max(1, 8 // bits_per_cell)
    max_h = np.log2(K)

    # new_ctx_table[ctx, sym] = the meta-state reached from `ctx` after
    # emitting `sym` (drop the oldest digit, append `sym` — base-K shift).
    drop_base = K ** (order - 1) if order > 1 else 1
    ctx_idx = np.arange(n_ctx)
    sym_idx = np.arange(K)
    new_ctx_table = (ctx_idx[:, None] % drop_base) * K + sym_idx[None, :]

    def _temper(base: np.ndarray, T: float) -> np.ndarray:
        logp = np.log(np.clip(base, 1e-300, None)) / T
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        return p / p.sum(axis=1, keepdims=True)

    def _stationary(P: np.ndarray, n_iters: int = 300, tol: float = 1e-9) -> np.ndarray:
        pi = np.full(n_ctx, 1.0 / n_ctx)
        for _ in range(n_iters):
            contrib = pi[:, None] * P
            pi_next = np.zeros(n_ctx)
            np.add.at(pi_next, new_ctx_table.ravel(), contrib.ravel())
            if np.abs(pi_next - pi).sum() < tol:
                return pi_next
            pi = pi_next
        return pi

    def _entropy_rate(P: np.ndarray) -> float:
        pi = _stationary(P)
        row_h = -(P * np.log2(np.clip(P, 1e-300, None))).sum(axis=1)
        return float((pi * row_h).sum())

    def _sample_symbols(P: np.ndarray, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.int64)
        ctx = int(rng.integers(0, n_ctx))
        cdfs = np.cumsum(P, axis=1)
        u = rng.random(n)
        for i in range(n):
            sym = min(int(np.searchsorted(cdfs[ctx], u[i])), K - 1)
            out[i] = sym
            ctx = int(new_ctx_table[ctx, sym])
        return out

    def _pack(symbols: np.ndarray, n_bytes_needed: int) -> np.ndarray:
        n_needed_cells = n_bytes_needed * cells_per_byte
        if len(symbols) < n_needed_cells:
            symbols = np.pad(symbols, (0, n_needed_cells - len(symbols)))
        symbols = symbols[:n_needed_cells].reshape(-1, cells_per_byte)
        byte_powers = K ** np.arange(cells_per_byte)
        packed = (symbols * byte_powers[None, :]).sum(axis=1)
        return np.clip(packed, 0, 255).astype(np.uint8)

    base = rng.dirichlet(np.ones(K), size=n_ctx)
    n_symbols_needed = n_bytes * cells_per_byte

    if target_bits is None:
        T = rng.uniform(*temperature_range)
        symbols = _sample_symbols(_temper(base, T), n_symbols_needed)
        return _pack(symbols, n_bytes)

    target_per_symbol = min(target_bits / cells_per_byte, max_h)
    lo, hi = 1e-3, 100.0
    mid = 1.0
    for _ in range(max_bisect_iters):
        mid = (lo + hi) / 2.0
        h = _entropy_rate(_temper(base, mid))
        if abs(h - target_per_symbol) < entropy_tol:
            break
        if h < target_per_symbol:
            lo = mid
        else:
            hi = mid
    symbols = _sample_symbols(_temper(base, mid), n_symbols_needed)
    return _pack(symbols, n_bytes)


def gen_match_distance(rng: np.random.Generator, n_bytes: int,
                       target_bits: float | None = None,
                       p_repeat_range: tuple[float, float] = (0.0, 0.9),
                       max_distance_range: tuple[int, int] = (4, 128),
                       mean_match_len_range: tuple[float, float] = (4.0, 32.0),
                       min_match_len: int = 3,
                       entropy_tol: float = 0.3, max_bisect_iters: int = 30) -> np.ndarray:
    """
    LZ77-exploitable structure, directly parametrized by MATCH PROBABILITY,
    MATCH-DISTANCE RANGE, and MATCH-LENGTH — a more direct, more
    controllable realization of what gen_template_repeat's placeholder was
    aiming for (see that docstring). At each "event" (after the first
    min_match_len bytes), with probability p_repeat emit a MATCH: copy a
    run of length L (L ~ Exponential(mean_match_len), floored at
    min_match_len) from `distance` positions back (distance drawn uniformly
    in [1, min(max_distance, position)]); otherwise emit a LITERAL: one
    fresh uniform-random byte. This literally simulates LZ77 match/literal
    tokens, so DEFLATE's LZ77 stage should detect it directly.

    CORRECTNESS NOTE, measured not assumed: an earlier version of this
    generator copied ONE byte per repeat event instead of a multi-byte
    run, which meant zlib essentially never detected the structure —
    DEFLATE requires a MINIMUM MATCH LENGTH of 3 bytes to encode a match
    at all; isolated single-byte "copy from distance d" events almost
    never chain into a 3+-byte matchable run by chance, so the measured
    bits/byte stayed flat near 8 regardless of p until p exceeded ~0.9
    (verified: p=0.0-0.7 all measured ~7.7-8.0 bits/byte despite the old
    formula predicting a smooth decline from 8 to ~6). Emitting genuine
    match RUNS (>= min_match_len) fixes this at the root — matches are now
    always long enough for DEFLATE's LZ77 stage to find, so the achieved
    compression tracks p much more smoothly across its whole range.

    Complementary to gen_run_length (which is the mean_match_len small,
    distance=1 special case): here `max_distance` independently controls
    HOW FAR BACK repeats reach, which run-length structure cannot express.
    This is the most direct lever on this project's own "does structure
    survive across a gap" question (the chain-memory recovery probe's
    concern), applied to raw byte structure rather than model behavior.

    target_bits calibration: APPROXIMATE closed form (bisection on
    p_repeat only; max_distance/mean_match_len are separate structural
    parameters, resampled fresh per call, analogous to gen_ca's k_states
    or gen_markov_order_k's K/order). Per "event", amortized over its
    output bytes:
        avg_bits_per_event(p)  = p*(log2(max_distance) + log2(mean_match_len)) + (1-p)*8
        avg_bytes_per_event(p) = p*mean_match_len + (1-p)*1
        bits_per_byte(p) = avg_bits_per_event(p) / avg_bytes_per_event(p)
    Monotonically decreasing in p — NOT an exact entropy calculation (real
    LZ77 offset/length coding isn't uniform); directionally reliable, same
    honesty caveat as gen_run_length (which uses the identical amortization
    trick for single-byte runs).

    *** RECOVERY-PROBE CONTAMINATION WARNING (see docs/HISTORY.md) ***
    Like gen_template_repeat, this generator's redundancy IS exact byte
    repetition — a model could "recover" a matched byte via simple
    positional copying, independent of whether any relay/STATE mechanism
    carried information. Every call draws fresh randomness (source bytes,
    p_repeat, max_distance, mean_match_len all vary per call), so there is
    no FIXED pattern across training examples to memorize into static
    weights, but WITHIN a single sequence the repeats are real and exact —
    do not use this generator for the chain-memory recovery probe without
    accounting for this; it sharpens the module docstring's general
    caution on structured data, since exact-copy IS the whole point here.
    """
    max_distance = int(rng.integers(*max_distance_range))
    mean_match_len = float(rng.uniform(*mean_match_len_range))

    def _bits_estimate(p: float) -> float:
        cost_match = np.log2(max(max_distance, 2)) + np.log2(max(mean_match_len, 2))
        avg_bytes = p * mean_match_len + (1.0 - p) * 1.0
        avg_bits = p * cost_match + (1.0 - p) * 8.0
        return avg_bits / avg_bytes

    def _generate(p: float, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.uint8)
        i = 0
        while i < n:
            if i >= min_match_len and rng.random() < p:
                L = max(min_match_len, int(round(rng.exponential(mean_match_len))))
                L = min(L, n - i)
                d = int(rng.integers(1, min(max_distance, i) + 1))
                for j in range(L):
                    out[i + j] = out[i + j - d]
                i += L
            else:
                out[i] = rng.integers(0, 256)
                i += 1
        return out

    if target_bits is None:
        p = rng.uniform(*p_repeat_range)
        return _generate(p, n_bytes)

    lo, hi = 0.0, p_repeat_range[1]
    mid = (lo + hi) / 2.0
    for _ in range(max_bisect_iters):
        mid = (lo + hi) / 2.0
        b = _bits_estimate(mid)
        if abs(b - target_bits) < entropy_tol:
            break
        if b > target_bits:
            lo = mid
        else:
            hi = mid
    return _generate(mid, n_bytes)


def gen_mixed_order(rng: np.random.Generator, n_bytes: int,
                    target_bits: float | None = None,
                    K: int = 4, orders: tuple[int, ...] = (0, 1, 3),
                    temperature_range: tuple[float, float] = (0.1, 8.0),
                    entropy_tol: float = 0.02, max_bisect_iters: int = 40) -> np.ndarray:
    """
    CTW/PPM-exploitable structure: at each output position, STOCHASTICALLY
    choose which context order to sample from (mixing weights fresh per
    call), from a small set of independently-calibrated models (default
    orders (0,1,3): a marginal distribution, an order-1 chain, and an
    order-3 chain, all small-alphabet K=4). This specifically defeats any
    FIXED single-order model — no one context length captures all the
    structure — but is exactly the situation Context Tree Weighting (and
    PPM's escape/blending mechanism) is designed for: a Bayesian mixture
    over context orders, weighted by which order is actually informative
    at each position. See docs/HISTORY.md §11's classical-alternatives
    discussion for the CTW background this connects to.

    Implementation: generate a FULL independent n_bytes realization from
    each component order (order 0 via gen_iid_skewed, order>=1 via
    gen_markov_order_k, each individually calibrated to target_bits), then
    pick per-position WHICH component's byte to emit according to
    Dirichlet-sampled mixing weights. Simple, avoids threading shared
    mutable state across differently-ordered submodels, and still
    produces the intended qualitative property (different positions
    genuinely come from different-order sources).

    KNOWN APPROXIMATION, flagged not hidden: the true entropy of a
    stochastically-SWITCHED process is only approximately the weighted
    average of its components' entropies — the actual value is provably
    LOWER, since an adaptive/CTW-aware decoder can partially infer which
    component is active from context (extracting structure a naive
    per-symbol calculation misses). Calibrating each component
    independently to target_bits and mixing lands roughly in the target
    neighborhood but is the LEAST precisely calibrated generator in this
    module — treat target_bits here as the loosest of all seven
    generators' calibration guarantees.

    Visible artifact worth knowing: order-0 samples span the full 256-byte
    range while order>=1 components are restricted to K=4 quantization
    levels packed into bytes — positions drawn from different components
    can look visually/statistically inconsistent within one sequence
    (this is intentional, not a bug — it IS the mixed structure).
    """
    assert len(orders) >= 1
    component_bytes: list[np.ndarray] = []
    for order in orders:
        if order == 0:
            component_bytes.append(gen_iid_skewed(
                rng, n_bytes, target_bits=target_bits,
                temperature_range=temperature_range,
                entropy_tol=entropy_tol, max_bisect_iters=max_bisect_iters))
        else:
            component_bytes.append(gen_markov_order_k(
                rng, n_bytes, order=order, K=K, target_bits=target_bits,
                temperature_range=temperature_range,
                entropy_tol=entropy_tol, max_bisect_iters=max_bisect_iters))

    mix_weights = rng.dirichlet(np.ones(len(orders)))
    choice = rng.choice(len(orders), size=n_bytes, p=mix_weights)
    out = np.empty(n_bytes, dtype=np.uint8)
    for i, comp_bytes in enumerate(component_bytes):
        mask = choice == i
        out[mask] = comp_bytes[mask]
    return out


def generate_structured_chunks(rng: np.random.Generator, kind: str,
                               n_chunks: int, chunk_len: int,
                               target_bits: float | None = None) -> np.ndarray:
    """
    Dispatcher matching the (n_chunks, chunk_len) int64 shape the rest of
    kvmem/hmn.py's batch-filling code expects wherever it currently uses
    random bytes (see make_batch_tagged, make_test_sequences).

    kind: 'chaotic' | 'fractal' | 'ca' | 'markov' | 'iid_skewed' |
    'run_length' | 'markov_k' | 'match_distance' | 'mixed_order'
    (recommended default: 'ca' — see gen_ca's docstring for the full
    rationale: discrete-native, exactly reproducible, enormous
    controllable rule-space diversity, no quantization ambiguity. 'markov'
    is the recommended choice when PRECISE target_bits calibration matters
    more than generator diversity — see gen_markov's docstring,
    closed-form entropy rate, no measure-and-search needed. 'iid_skewed'
    and 'run_length' are the two "zlib CAN see this" control cases
    (marginal-frequency and literal-repeat structure respectively) to pair
    against 'markov'/'ca' as contrast; 'markov_k' generalizes 'markov' to
    context length > 1; 'match_distance' is the parametrized LZ77-style
    generator (see its RECOVERY-PROBE CONTAMINATION WARNING before use);
    'mixed_order' produces CTW/PPM-exploitable structure that defeats any
    single fixed context length. All are implemented and available for
    ablation comparing which generator family the model actually learns
    structure from most readily.)

    target_bits: desired bits/byte of TRUE compressibility. Exact
    (bisection against a closed-form entropy value) for 'markov',
    'iid_skewed', and 'markov_k'; APPROXIMATE (bisection against a
    documented closed-form-ish estimate, not a measured value) for
    'run_length', 'match_distance', and 'mixed_order'; measure_bits_per_byte
    (zlib-based) search for 'chaotic'/'fractal'/'ca' — see each generator's
    own docstring for which regime it's in and why. NOT marginal
    byte-histogram entropy in any case, see module docstring.
    e.g. generate_structured_chunks(rng, 'ca', 8, 16, target_bits=2.0) for a
    "2-bit" CA sequence, generate_structured_chunks(rng, 'markov', 8, 16,
    target_bits=4.0) for a precisely-calibrated "4-bit" Markov sequence.
    None (default) skips calibration and uses each generator's default
    parameter range.
    """
    n_bytes = n_chunks * chunk_len
    if kind == 'chaotic':
        b = gen_chaotic_logistic(rng, n_bytes, target_bits=target_bits)
    elif kind == 'fractal':
        b = gen_fractal_midpoint(rng, n_bytes, target_bits=target_bits)
    elif kind == 'ca':
        b = gen_ca(rng, n_bytes, target_bits=target_bits)
    elif kind == 'markov':
        b = gen_markov(rng, n_bytes, target_bits=target_bits)
    elif kind == 'iid_skewed':
        b = gen_iid_skewed(rng, n_bytes, target_bits=target_bits)
    elif kind == 'run_length':
        b = gen_run_length(rng, n_bytes, target_bits=target_bits)
    elif kind == 'markov_k':
        b = gen_markov_order_k(rng, n_bytes, target_bits=target_bits)
    elif kind == 'match_distance':
        b = gen_match_distance(rng, n_bytes, target_bits=target_bits)
    elif kind == 'mixed_order':
        b = gen_mixed_order(rng, n_bytes, target_bits=target_bits)
    else:
        raise ValueError(f'unknown structured data kind: {kind!r}')
    return b.reshape(n_chunks, chunk_len).astype(np.int64)
