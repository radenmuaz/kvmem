"""
kvmem/data.py — Synthetic byte-level Markov chain dataset.

Vocab: V=256 bytes. Data bytes constrained to [0x20, 0xFF] so they never
collide with protocol bytes 0x00-0x1F (STX, ETX, NUL, etc.).

Sequence layout (stage 0):
    [ x_S (L_S) | STX | SLOT_ID*N | ETX | y (L_y) ]
    S = [0, L_S)
    M = [L_S+1, L_S+1+N)    <- inner KV slots
    Y = [L_S+2+N, L_S+2+N+L_y)

Memory tokens are SLOT_BASE+i for slot i (bytes 0x04..0x04+N-1).
Each slot has a unique token → unique embedding from init → no slot collapse.
The model can differentiate slot 5 from slot 15 by token identity.
"""

import os
import queue
import threading

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Legacy v1 single-byte protocol tokens (kept for backward compat)
# ---------------------------------------------------------------------------
STX = 0x02
ETX = 0x03
NUL = 0x00
SLOT_BASE = 0x04
DATA_LO   = 0x20   # legacy: data restricted to [0x20, 0xFF]


def make_slot_ids(N: int) -> list[int]:
    """Legacy v1 slot IDs in [0x04, 0x20). Wraps mod 28 for N>28."""
    return [(SLOT_BASE + (i % (DATA_LO - SLOT_BASE))) for i in range(N)]


# ---------------------------------------------------------------------------
# Tag-based scheme: multi-byte markers, NO data byte restrictions
# ---------------------------------------------------------------------------
# Data range: ALL 256 bytes [0x00, 0xFF]. The model distinguishes protocol
# from data purely via position (YaRN) and context — not by byte value.
#
# Tags use printable ASCII bytes from the existing 256-token vocab:
#   MEM_OPEN  = '<m>'  = [0x3C, 0x6D, 0x3E]          (3 bytes)
#   MEM_CLOSE = '</m>' = [0x3C, 0x2F, 0x6D, 0x3E]    (4 bytes)
#
# Sequence format:
#   [x_S (L_S) | '<m>' (3) | slot_tokens (N) | '</m>' (4) | Y (L_y)]
#   L = L_S + 3 + N + 4 + L_y = L_S + N + L_y + 7
#
# Slot styles (ablation):
#   'zeros': all N slot tokens = 0x00  → YaRN position alone routes
#   'seq':   slot i = i % 256          → position + token identity route
# ---------------------------------------------------------------------------

MEM_OPEN  = [0x3C, 0x6D, 0x3E]           # '<m>'
MEM_CLOSE = [0x3C, 0x2F, 0x6D, 0x3E]    # '</m>'
MEM_OPEN_LEN  = len(MEM_OPEN)            # 3
MEM_CLOSE_LEN = len(MEM_CLOSE)           # 4
MEM_OVERHEAD  = MEM_OPEN_LEN + MEM_CLOSE_LEN  # 7


def make_slot_ids_tag(N: int, style: str = 'seq') -> list[int]:
    """
    Return N slot tokens for tag-based scheme (legacy, kept for compat).
    Prefer make_mem_slot_ids() / make_latent_slot_ids() for new code.
    """
    if style == 'zeros':
        return [0x00] * N
    elif style == 'seq':
        return [i % 256 for i in range(N)]
    else:
        raise ValueError(f"slot style must be 'zeros' or 'seq', got {style!r}")


# ---------------------------------------------------------------------------
# Learned-embedding token vocab  (v2)
# ---------------------------------------------------------------------------
# DB-style tag vocabulary  (d=data, e=extract, k=key, q=query, v=value)
#
# Data bytes   : IDs 0–255    (all 256 byte values, unchanged)
# Boundary tags: IDs 256–265  (1 token each, never appear in data)
# Key slots    : IDs 266–265+hidden_len    (1 per position — unique dedicated ID)
# Extract slots: IDs 266+hidden_len–…     (1 per position — unique dedicated ID)
#
# V = 256 + 10 + hidden_len + latent_len  (computed by compute_vocab_size)
#
# Sequence layout (source-first, fully causal):
#   <d> data_bytes </x>
#   [<e> extract_0 ... extract_{P-1} </z>]   ← ponder before compression
#   <k> key_0 ... key_{N-1} </h>             ← compressed memory
#   <q> warmup </q>                           ← query anchor
#   <v> output </y>                           ← returned value
#
# Causal access: e sees d; k sees d+e; q/v blocked from d+e (bottleneck via k).
# All boundary tags are 1 token; all lengths = 1.
# ---------------------------------------------------------------------------

# Boundary tag IDs (IDs 256–267)
HIDDEN_OPEN_ID      = 256   # <k>  key/memory open
HIDDEN_CLOSE_ID     = 257   # </h>
INPUT_OPEN_ID       = 258   # <d>  data/source open
INPUT_CLOSE_ID      = 259   # </x>
QUERY_OPEN_ID       = 260   # <q>  query/anchor open  (user-facing)
QUERY_CLOSE_ID      = 261   # </q>
LATENT_OPEN_ID    = 262   # <e>  extract/ponder open
LATENT_CLOSE_ID   = 263   # </z>
OUTPUT_OPEN_ID      = 264   # <v>  value/output open  (user-facing final)
OUTPUT_CLOSE_ID     = 265   # </y>
REFINE_OPEN_ID      = 266   # <r>  refinement/draft open  (internal chain-of-thought)
REFINE_CLOSE_ID     = 267   # </r>

# Base ID for key/extract position tokens (dedicated indexed: unique ID per position)
HIDDEN_SLOT_BASE    = 268   # key slot i → HIDDEN_SLOT_BASE + i
# extract slot j → HIDDEN_SLOT_BASE + hidden_len + j  (hidden_len known at runtime)

N_BOUNDARY_TAGS  = 12   # 10 original + 2 refine tags

# List form for batch builder
HIDDEN_OPEN     = [HIDDEN_OPEN_ID]
HIDDEN_CLOSE    = [HIDDEN_CLOSE_ID]
INPUT_OPEN      = [INPUT_OPEN_ID]
INPUT_CLOSE     = [INPUT_CLOSE_ID]
QUERY_OPEN      = [QUERY_OPEN_ID]
QUERY_CLOSE     = [QUERY_CLOSE_ID]
LATENT_OPEN   = [LATENT_OPEN_ID]
LATENT_CLOSE  = [LATENT_CLOSE_ID]
OUTPUT_OPEN     = [OUTPUT_OPEN_ID]
OUTPUT_CLOSE    = [OUTPUT_CLOSE_ID]
REFINE_OPEN     = [REFINE_OPEN_ID]
REFINE_CLOSE    = [REFINE_CLOSE_ID]

# All tag lengths = 1
HIDDEN_OPEN_LEN = HIDDEN_CLOSE_LEN         = 1
INPUT_OPEN_LEN = INPUT_CLOSE_LEN           = 1
QUERY_OPEN_LEN = QUERY_CLOSE_LEN           = 1
LATENT_OPEN_LEN = LATENT_CLOSE_LEN     = 1
OUTPUT_OPEN_LEN = OUTPUT_CLOSE_LEN         = 1
REFINE_OPEN_LEN = REFINE_CLOSE_LEN         = 1

# Backward-compat aliases (old names → new)
MEM_OPEN_LEN = MEM_CLOSE_LEN     = 1  # <k>
SRC_OPEN_LEN = SRC_CLOSE_LEN     = 1  # <d>
FROM_OPEN_LEN = FROM_CLOSE_LEN   = 1  # <q>
PONDER_OPEN_LEN = PONDER_CLOSE_LEN = 1  # <e>
CONT_OPEN_LEN = CONT_CLOSE_LEN   = 1  # <v>

MEM_OPEN = HIDDEN_OPEN;   MEM_CLOSE = HIDDEN_CLOSE
SRC_OPEN = INPUT_OPEN;  SRC_CLOSE = INPUT_CLOSE
FROM_OPEN = QUERY_OPEN; FROM_CLOSE = QUERY_CLOSE
PONDER_OPEN = LATENT_OPEN; PONDER_CLOSE = LATENT_CLOSE
CONT_OPEN = OUTPUT_OPEN; CONT_CLOSE = OUTPUT_CLOSE

ROLE_OVERHEAD = (HIDDEN_OPEN_LEN + HIDDEN_CLOSE_LEN +
                 INPUT_OPEN_LEN + INPUT_CLOSE_LEN +
                 QUERY_OPEN_LEN + QUERY_CLOSE_LEN +
                 OUTPUT_OPEN_LEN + OUTPUT_CLOSE_LEN)


def compute_vocab_size(hidden_len: int, latent_len: int = 0) -> int:
    """V = 256 (data) + 12 (boundary tags) + hidden_len + latent_len."""
    return 256 + N_BOUNDARY_TAGS + hidden_len + latent_len


def make_hidden_slot_ids(hidden_len: int, cycle_len: int = 0) -> list[int]:
    """
    Return hidden_len slot token IDs.

    cycle_len=0 (default): dedicated indexed — slot i → HIDDEN_SLOT_BASE + i.
      Unique ID per position. Vocab grows with hidden_len. No extrapolation.

    cycle_len=K (K>0): dedicated cyclic — slot i → HIDDEN_SLOT_BASE + (i % K).
      K dedicated IDs cycle over all slots. Fixed vocab regardless of hidden_len.
      Extrapolates to arbitrary hidden_len — model trained with K=8 can infer
      with hidden_len=1024 using the same 8 IDs, RoPE carries absolute position.
      Zero data collision (all IDs > 255). Best design before scaling slot_len.

    TODO: add cycle_len to SeqSpec DSL (e.g. <h:1,cycle=8>) and train.py hp.
          Set cycle_len=8 before scaling slot_len beyond training budget.
    """
    K = cycle_len if cycle_len > 0 else hidden_len
    return [HIDDEN_SLOT_BASE + (i % K) for i in range(hidden_len)]

# backward-compat alias
make_mem_slot_ids = make_hidden_slot_ids


def make_latent_slot_ids(latent_len: int, hidden_len: int) -> list[int]:
    """extract slot j = HIDDEN_SLOT_BASE + hidden_len + j"""
    base = HIDDEN_SLOT_BASE + hidden_len
    return [base + j for j in range(latent_len)]

# backward-compat alias
make_latent_slot_ids = make_latent_slot_ids


def make_mask_role(L_S: int, N: int, L_f: int, L_c: int,
                   active_slots: int = 0) -> np.ndarray:
    """
    Attention mask for the role-tag scheme.

    active_slots: if >0, only the LAST active_slots slots are visible to <f>/<c>.
                  First (N - active_slots) slots still encode x_S but are masked
                  as attention targets. 0 = all slots active (default).

    Layout (absolute positions):
      <s>       [0,                3)
      x_S       [3,                3+L_S)
      </s>      [3+L_S,            7+L_S)
      <m>       [7+L_S,            10+L_S)
      slots     [10+L_S,           10+L_S+N)
      </m>      [10+L_S+N,         14+L_S+N)
      <f>       [14+L_S+N,         17+L_S+N)
      warmup    [17+L_S+N,         17+L_S+N+L_f)
      </f>      [17+L_S+N+L_f,     21+L_S+N+L_f)
      <c>       [21+L_S+N+L_f,     24+L_S+N+L_f)
      output    [24+L_S+N+L_f,     24+L_S+N+L_f+L_c)
      </c>      [24+L_S+N+L_f+L_c, 28+L_S+N+L_f+L_c)

    Mask rules:
      1. <c>/output/</c> are write-only from outside.
      2. <c>/output can attend to: active slots, </m>, <f>/warmup/</f> — NOT x_S or <s>.
      3. <f>/warmup can attend to: active slots, </m> — NOT x_S (forces KV lookup).
      4. slots attend to x_S + prior slots (causal within M).
      5. Standard causal everywhere else.
    """
    src_start   = SRC_OPEN_LEN                              # 3
    src_end     = src_start + L_S                           # 3+L_S
    s_close_end = src_end + SRC_CLOSE_LEN                  # 7+L_S
    m_open_end  = s_close_end + MEM_OPEN_LEN               # 10+L_S
    slot_start  = m_open_end
    slot_end    = slot_start + N                            # 10+L_S+N
    m_close_end = slot_end + MEM_CLOSE_LEN                  # 14+L_S+N
    f_open_end  = m_close_end + FROM_OPEN_LEN               # 17+L_S+N
    f_start     = f_open_end
    f_end       = f_start + L_f                             # 17+L_S+N+L_f
    f_close_end = f_end + FROM_CLOSE_LEN                    # 21+L_S+N+L_f
    c_open_end  = f_close_end + CONT_OPEN_LEN               # 24+L_S+N+L_f
    c_start     = c_open_end
    c_end       = c_start + L_c                             # 24+L_S+N+L_f+L_c
    L           = c_end + CONT_CLOSE_LEN                    # 28+L_S+N+L_f+L_c

    rows = np.arange(L)[:, None]
    cols = np.arange(L)[None, :]
    causal = cols <= rows

    # Region membership
    is_src      = (cols >= SRC_OPEN_LEN) & (cols < s_close_end)    # <s> + x_S + </s>
    is_mem_tag  = ((cols >= s_close_end) & (cols < slot_start)) | \
                  ((cols >= slot_end)    & (cols < m_close_end))    # <m> and </m> always visible
    is_slot     = (cols >= slot_start) & (cols < slot_end)          # slots only
    is_f_region = (cols >= m_close_end) & (cols < f_close_end)      # <f>..warmup..</f>
    is_c_region = (cols >= f_close_end)                              # <c>..output..</c>

    is_c_row    = (rows >= f_close_end)
    is_f_row    = (rows >= m_close_end) & (rows < f_close_end)

    # Rule 1: <c> is write-only
    block_c_sink = is_c_region & ~is_c_row

    # Rule 2: <c> rows cannot see x_S/<s>
    block_c_sees_src = is_c_row & is_src

    # Rule 3: <f> rows cannot see x_S either (forces KV lookup)
    block_f_sees_src = is_f_row & is_src

    # Rule 5: inactive slots (first N-active_slots) are invisible to <f>/<c>
    # <m> and </m> are never blocked regardless of active_slots
    if active_slots > 0 and active_slots < N:
        inactive_end = slot_start + (N - active_slots)
        is_inactive_slot = (cols >= slot_start) & (cols < inactive_end) & ~is_mem_tag
        block_inactive = (is_f_row | is_c_row) & is_inactive_slot
    else:
        block_inactive = np.zeros((L, L), dtype=bool)

    blocked = block_c_sink | block_c_sees_src | block_f_sees_src | block_inactive
    visible = causal & ~blocked
    return np.where(visible, 0.0, -1e9).astype(np.float32)


# ---------------------------------------------------------------------------
# Memory-first multi-block layout  (RNN-style: <m>state</m><s>input</s>)
# ---------------------------------------------------------------------------
#
# Each block: <m>slots</m><s>src</s>
#   Memory tokens come FIRST (like an RNN hidden state).
#   Slots read src NON-CAUSALLY — src follows slots in the sequence, but the
#   mask explicitly allows slots→src attention (bidirectional for that pair).
#
# Multi-block sequence:
#   <m>slots_0</m><s>src_0</s> ... <m>slots_{n-1}</m><s>src_{n-1}</s>
#   <f>warmup</f><c>output</c>
#
# Benefits vs source-first layout:
#   1. Memory prefix is always at position 0 — natural KV cache prefix.
#   2. Recall region (<f><c>) can attend to slots causally (slots are earlier).
#   3. Mirrors inference: cache slot KV once, run many recall queries after.
#
# BLOCK_LEN = MEM_OPEN + slot_len + MEM_CLOSE + SRC_OPEN + seg_len + SRC_CLOSE
#           = 3 + slot_len + 4 + 3 + seg_len + 4  =  slot_len + seg_len + 14
#   (identical to source-first BLOCK_LEN — same total length, different order)
#
# Mask rules (memory-first):
#   1. Slots→src: BIDIRECTIONAL within same block (non-causal override).
#      Slots at positions [sl0, sl1) need to see src at [s0, s1) even though
#      src comes after in sequence position.
#   2. <c> is write-only from outside.
#   3. <f>/<c> cannot attend to any src region.
#   4. Cross-block isolation: slots_i cannot attend to src_j or slots_j (j≠i).
#   5. active_slots>0: only last N slots per block visible to <f>/<c>.
# ---------------------------------------------------------------------------

def _sample_seg(rng: np.random.Generator, seg_len: int) -> np.ndarray:
    """Sample one source segment of `seg_len` bytes using a random distribution."""
    dist_type = int(rng.integers(0, 4))
    if dist_type == 0:
        return rng.integers(0, 256, size=seg_len).astype(np.int32)
    elif dist_type == 1:
        alpha = float(rng.uniform(0.05, 1.0))
        p     = rng.dirichlet(np.ones(256) * alpha)
        return rng.choice(256, size=seg_len, p=p).astype(np.int32)
    elif dist_type == 2:
        width = int(rng.integers(4, 129))
        lo    = int(rng.integers(0, 256 - width + 1))
        return rng.integers(lo, lo + width, size=seg_len).astype(np.int32)
    else:
        p_g = float(rng.uniform(0.01, 0.3))
        return np.clip(rng.geometric(p_g, size=seg_len) - 1, 0, 255).astype(np.int32)


def multi_block_positions(n_blocks: int, seg_len: int, slot_len: int,
                          warmup_len: int, out_len: int,
                          latent_len: int = 0) -> dict:
    """
    Absolute token positions for a multi-block sequence.

    Block layout (fully causal, no overrides):
        <s>src</s> [<p>ponder</p>] <m>slots</m>

    Causal access within each block:
        <p> sees src  (src before p)
        <m> sees src + ponder  (both before m)
        <p> CANNOT see <m>  (m after p — blocked by causal)

    Recall region:  <f>warmup</f> <c>output</c>
        <f>/<c> blocked from src AND ponder (explicit) — bottleneck via slots only.

    latent_len=0: block is just <s>src</s><m>slots</m>  (no ponder tags written)

    Returns a dict with:
      blocks[i]    : {block_start, s0, s1, s_close_end,
                      p_open, p0, p1, p_close_end,  (p* same as s_close_end when latent_len==0)
                      sl0, sl1, mc1}
      recall_start, f0, f1, fc1, c0, c1, L
      mc1  : last block </m> end (kvcache prefix split)
    """
    # Per-block length depends on whether ponder is included
    EXTRACT_BLOCK = (LATENT_OPEN_LEN + latent_len + LATENT_CLOSE_LEN) if latent_len > 0 else 0
    BLOCK_LEN = (INPUT_OPEN_LEN + seg_len + INPUT_CLOSE_LEN +
                 EXTRACT_BLOCK +
                 MEM_OPEN_LEN + slot_len + MEM_CLOSE_LEN)
    blocks = []
    for i in range(n_blocks):
        bs          = i * BLOCK_LEN
        s0          = bs + SRC_OPEN_LEN
        s1          = s0 + seg_len
        s_close_end = s1 + SRC_CLOSE_LEN
        # Optional ponder region inside encoding block
        if latent_len > 0:
            p_open      = s_close_end
            p0          = p_open + PONDER_OPEN_LEN
            p1          = p0 + latent_len
            p_close_end = p1 + PONDER_CLOSE_LEN
        else:
            p_open = p0 = p1 = p_close_end = s_close_end
        sl0         = p_close_end + MEM_OPEN_LEN
        sl1         = sl0 + slot_len
        mc1         = sl1 + MEM_CLOSE_LEN
        blocks.append(dict(block_start=bs,
                           s0=s0, s1=s1, s_close_end=s_close_end,
                           p_open=p_open, p0=p0, p1=p1, p_close_end=p_close_end,
                           sl0=sl0, sl1=sl1, mc1=mc1))
    recall_start = n_blocks * BLOCK_LEN
    f0  = recall_start + FROM_OPEN_LEN
    f1  = f0 + warmup_len
    fc1 = f1 + FROM_CLOSE_LEN
    c0  = fc1 + CONT_OPEN_LEN
    c1  = c0  + out_len
    L   = c1  + CONT_CLOSE_LEN
    return dict(blocks=blocks, recall_start=recall_start,
                f0=f0, f1=f1, fc1=fc1,
                p0=fc1, p1=fc1, pc1=fc1,   # no recall-region ponder
                c0=c0, c1=c1, L=L,
                sl0=blocks[0]['sl0'], sl1=blocks[0]['sl1'],
                s0=blocks[0]['s0'],   s1=blocks[0]['s1'],
                mc1=blocks[-1]['mc1'])


def make_mask_multi(n_blocks: int, seg_len: int, slot_len: int,
                    warmup_len: int, out_len: int,
                    latent_len: int = 0,
                    mem_window: int = -1) -> np.ndarray:
    """
    Attention mask for multi-block sequences. Pure causal — no non-causal overrides.

    Block: <x>src</x> [<z>intermed</z>] <h>slots</h>
      <z> sees src (causal); <h> sees src + z (causal).
      <q>/<y> blocked from src and z — bottleneck via <h> slots only.
      <y> is write-only.

    mem_window: how many <h> states (including itself) each new <h> can attend to.
      0 (default): no limit — all prior <h> visible (full fast-weight accumulation)
      1: isolated — <h>_i sees only its own block, no history
      2: <h>_i sees <h>_{i-1} + itself (1-step Markov update)
      N: <h>_i sees last N <h> states

    Cross-block: <h>_i blocked from src_j, intermed_j of other blocks (j≠i).
    For j within mem_window: <h>_i CAN see <h>_j (fast-weight update).
    For j outside mem_window: <h>_i CANNOT see <h>_j.
    """
    pos    = multi_block_positions(n_blocks, seg_len, slot_len, warmup_len, out_len,
                                   latent_len)
    L      = pos['L']
    blocks = pos['blocks']
    r      = np.arange(L)
    c      = np.arange(L)
    causal  = c[None, :] <= r[:, None]
    blocked = np.zeros((L, L), dtype=bool)

    # Recall region rows
    is_f_row    = (r >= pos['recall_start']) & (r < pos['fc1'])
    is_c_row    = r >= pos['fc1']
    is_c_col    = c >= pos['fc1']
    recall_rows = is_f_row | is_c_row

    # All src and intermed col regions
    is_any_src    = np.zeros(L, dtype=bool)
    is_any_intermed = np.zeros(L, dtype=bool)
    for b in blocks:
        is_any_src     |= (c >= b['s0']) & (c < b['s1'])
        if latent_len > 0:
            is_any_intermed |= (c >= b['p0']) & (c < b['p1'])

    # <y> write-only
    blocked |= is_c_col[None, :] & ~is_c_row[:, None]
    # recall rows blocked from src and intermed
    blocked |= recall_rows[:, None] & is_any_src[None, :]
    blocked |= recall_rows[:, None] & is_any_intermed[None, :]

    # Cross-block rules for <h> slots
    for i, b in enumerate(blocks):
        slot_row = (r >= b['sl0']) & (r < b['sl1'])
        for j, bj in enumerate(blocks):
            if j == i:
                continue
            # Always block cross-block src and intermed
            cross_src_intermed = ((c >= bj['s0']) & (c < bj['s1'])) | \
                                 ((c >= bj['p0']) & (c < bj['p1']))
            blocked |= slot_row[:, None] & cross_src_intermed[None, :]
            # Block cross-block <h> slots outside mem_window
            # j < i: prior blocks — allowed if within window, blocked if outside
            if j < i:
                outside_window = (mem_window != -1) and ((i - j) >= mem_window)
                if outside_window:
                    h_j_col = (c >= bj['sl0']) & (c < bj['sl1'])
                    blocked |= slot_row[:, None] & h_j_col[None, :]
            # j > i: future blocks — already blocked by causal, no action needed

    visible = causal & ~blocked
    return np.where(visible, 0.0, -1e9).astype(np.float32)


def make_mask_old_memory(recall_from: int,
                         n_blocks: int, seg_len: int, slot_len: int,
                         warmup_len: int, out_len: int,
                         latent_len: int = 0,
                         mem_window: int = -1) -> np.ndarray:
    """
    Like make_mask_multi but recall region can only attend to blocks 0..recall_from.
    Blocks recall_from+1..n_blocks-1 are invisible to <q>/<y> — as if the later
    ingestion steps haven't happened yet.

    Used for the temporal margin loss: NLL under old memory (only first recall_from+1
    blocks visible) vs NLL under full memory (all blocks visible). The margin
    loss rewards updates that strictly improve recall.
    """
    pos    = multi_block_positions(n_blocks, seg_len, slot_len, warmup_len, out_len,
                                   latent_len)
    # Start from full mask, then additionally block later <h> slots from recall region
    mask = make_mask_multi(n_blocks, seg_len, slot_len, warmup_len, out_len,
                           latent_len, mem_window)
    blocked_extra = np.zeros(mask.shape, dtype=bool)
    L = pos['L']
    r = np.arange(L)
    c = np.arange(L)
    blocks = pos['blocks']

    is_recall_row = r >= pos['recall_start']
    # Block recall rows from <h> slots of blocks later than recall_from
    for i in range(recall_from + 1, n_blocks):
        b = blocks[i]
        h_col = (c >= b['sl0']) & (c < b['sl1'])
        blocked_extra |= is_recall_row[:, None] & h_col[None, :]

    return np.where((mask == 0.0) & ~blocked_extra, 0.0, -1e9).astype(np.float32)


def make_multi_batch(rng: np.random.Generator, B: int,
                     n_blocks: int, recall_from,
                     seg_len: int, slot_len: int,
                     warmup_len: int, out_len: int,
                     latent_len: int = 0) -> np.ndarray:
    """
    recall_from: int OR list[int].
    If list, each example in the batch independently draws a random recall_from
    from the list — mixed routing batch. All configs must have same sequence
    length (same n_blocks/seg_len/slot_len/warmup_len/out_len).
    """
    if isinstance(recall_from, (list, tuple)):
        recall_froms = list(recall_from)
        choices = rng.integers(0, len(recall_froms), size=B)
        pos = multi_block_positions(n_blocks, seg_len, slot_len, warmup_len, out_len, latent_len)
        L   = pos['L']
        out_arr = np.zeros((B, L), dtype=np.int64)
        for i, rf in enumerate(recall_froms):
            mask = choices == i
            if mask.any():
                sub = make_multi_batch(rng, int(mask.sum()), n_blocks, rf,
                                       seg_len, slot_len, warmup_len, out_len,
                                        latent_len)
                out_arr[mask] = sub
        return out_arr
    """
    Build one multi-block batch. Slot tokens use dedicated indexed scheme — unique ID per position.

    recall_from: which block index (0-based) the <f><c> pair queries.
    n_blocks=1, recall_from=0 is the single-block degenerate case.
    """
    pos        = multi_block_positions(n_blocks, seg_len, slot_len,
                                       warmup_len, out_len, latent_len)
    L          = pos['L']
    slot_ids   = make_hidden_slot_ids(slot_len)
    ponder_ids = make_latent_slot_ids(latent_len, slot_len) if latent_len > 0 else []
    out        = np.zeros((B, L), dtype=np.int64)
    n_win      = max(1, seg_len - out_len)

    for i in range(B):
        segs = [_sample_seg(rng, seg_len) for _ in range(n_blocks)]

        for k, b in enumerate(pos['blocks']):
            seg = segs[k]
            # Block layout: <s>src</s> [<p>ponder</p>] <m>slots</m>
            out[i, b['block_start']:b['s0']]             = SRC_OPEN
            out[i, b['s0']:b['s1']]                      = seg
            out[i, b['s1']:b['s_close_end']]             = SRC_CLOSE
            if latent_len > 0:
                out[i, b['p_open']:b['p0']]              = PONDER_OPEN
                out[i, b['p0']:b['p1']]                  = ponder_ids
                out[i, b['p1']:b['p_close_end']]         = PONDER_CLOSE
            out[i, b['p_close_end']:b['sl0']]            = MEM_OPEN
            out[i, b['sl0']:b['sl1']]                    = slot_ids
            out[i, b['sl1']:b['mc1']]                    = MEM_CLOSE

        seg_r   = segs[recall_from]
        y_start = int(rng.integers(0, n_win + 1))
        y_end   = min(y_start + out_len, seg_len)
        w_st    = max(0, y_start - warmup_len)
        wm      = seg_r[w_st:y_start]
        if len(wm) < warmup_len:
            wm = np.concatenate(
                [np.full(warmup_len - len(wm), seg_r[0], dtype=np.int32), wm])

        rs = pos['recall_start']
        out[i, rs:rs+FROM_OPEN_LEN]                      = FROM_OPEN
        out[i, pos['f0']:pos['f1']]                      = wm
        out[i, pos['f1']:pos['f1']+FROM_CLOSE_LEN]       = FROM_CLOSE
        out[i, pos['fc1']:pos['fc1']+CONT_OPEN_LEN]      = CONT_OPEN
        out[i, pos['c0']:pos['c0']+(y_end-y_start)]      = seg_r[y_start:y_end]
        out[i, pos['c1']:pos['c1']+CONT_CLOSE_LEN]       = CONT_CLOSE

    return out


# ---------------------------------------------------------------------------
# Refine mode — iterative memory correction
#
# Sequence: n_blocks × block  +  warmup  +  n_attempts × (output + correction)  +  final
#
#   [<x>src</x>[<z>z</z>]<h>h</h>] × n_blocks   ← encoding block(s)
#   <r>wm</r>                                      ← warmup anchor, ONCE (like <q>)
#   <y>attempt_1</y>                               ← attempt 1 (noisy gt)
#   [<z>z</z>]<h>h</h>                             ← correction: reads h_prev + attempt_1
#   <y>attempt_2</y>                               ← attempt 2 (less noisy)
#   [<z>z</z>]<h>h</h>
#   ...
#   <y>final</y>                                   ← final (clean gt, loss, trains copy)
#
# n_attempts=0: <r>wm</r><y>final</y> — identical to standard <q>wm</q><y>final</y>
#               (same L, same structure; only tag differs)
#
# Inference: stop after last attempt, don't generate final.
# Stopping criterion: consecutive <y> outputs converge (no new vocab needed).
#
# Mask: all rows after encoding blocked from <x>/<z_enc>. Pure causal otherwise.
# Correction <z>/<h> sees prior <y> causally — no special rules needed.
# ---------------------------------------------------------------------------

def refine_positions(n_attempts: int, n_blocks: int, seg_len: int,
                     slot_len: int, warmup_len: int, out_len: int,
                     latent_len: int = 0) -> dict:
    """
    Positions for the refine sequence.

    n_attempts=0: [encoding] <r>wm</r><y>final</y>  — identical to standard <q><y>, L same
    n_attempts=N: [encoding] <r>wm</r>
                             (<y>attempt_k</y> [<z>z</z><h>h</h>]) × N
                             <y>final</y>

    'attempts' list: attempt turns 0..N-1 (noisy).
    'final': the last clean output (noise=0, trains copy mechanism).
    At inference: stop after last attempt (attempts[-1]), don't generate final.
    """
    base = multi_block_positions(n_blocks, seg_len, slot_len, warmup_len, out_len, latent_len)
    encoding_end = base['recall_start']   # position right after all encoding blocks

    # Warmup: <r>wm</r>
    r_open  = encoding_end
    r0      = r_open + REFINE_OPEN_LEN
    r1      = r0 + warmup_len
    r_close = r1
    rc1     = r_close + REFINE_CLOSE_LEN
    pos_cur = rc1

    # Per-attempt unit: <y>out</y> <z>z</z><h>h</h>
    OUTPUT_UNIT = OUTPUT_OPEN_LEN + out_len + OUTPUT_CLOSE_LEN
    CORR_UNIT   = (LATENT_OPEN_LEN + latent_len + LATENT_CLOSE_LEN +
                   HIDDEN_OPEN_LEN + slot_len + HIDDEN_CLOSE_LEN)

    attempts = []
    for _ in range(n_attempts):
        c0  = pos_cur + OUTPUT_OPEN_LEN
        c1  = c0 + out_len
        cl1 = c1 + OUTPUT_CLOSE_LEN
        # Correction block: <z><h> after output
        p0  = cl1 + LATENT_OPEN_LEN
        p1  = p0  + latent_len
        pc1 = p1  + LATENT_CLOSE_LEN
        sl0 = pc1 + HIDDEN_OPEN_LEN
        sl1 = sl0 + slot_len
        mc1 = sl1 + HIDDEN_CLOSE_LEN
        attempts.append(dict(c0=c0, c1=c1, cl1=cl1, p0=p0, p1=p1, pc1=pc1,
                             sl0=sl0, sl1=sl1, mc1=mc1))
        pos_cur = mc1

    # Copy turn: <y>clean</y>  (no noise — trains copy mechanism, inference stops here)
    fc0  = pos_cur + OUTPUT_OPEN_LEN
    fc1  = fc0 + out_len
    fcl1 = fc1 + OUTPUT_CLOSE_LEN
    pos_cur = fcl1

    # Final correction: <z><h> after copy turn (updates h_final from clean output)
    gp0  = pos_cur + LATENT_OPEN_LEN
    gp1  = gp0 + latent_len
    gpc1 = gp1 + LATENT_CLOSE_LEN
    gsl0 = gpc1 + HIDDEN_OPEN_LEN
    gsl1 = gsl0 + slot_len
    gmc1 = gsl1 + HIDDEN_CLOSE_LEN
    pos_cur = gmc1

    # Post-refine query: <q>wm</q><y>clean</y>  (loss here, must match 100%)
    qr_open = pos_cur
    qr0     = qr_open + QUERY_OPEN_LEN
    qr1     = qr0 + warmup_len
    qrc1    = qr1 + QUERY_CLOSE_LEN
    qc0     = qrc1 + OUTPUT_OPEN_LEN
    qc1     = qc0 + out_len
    qcl1    = qc1 + OUTPUT_CLOSE_LEN
    L       = qcl1

    return dict(
        base=base, attempts=attempts, L=L,
        n_attempts=n_attempts,
        encoding_end=encoding_end,
        r0=r0, r1=r1, rc1=rc1,
        r_open=r_open,
        copy_c0=fc0, copy_c1=fc1, copy_cl1=fcl1,
        final=dict(p0=gp0, p1=gp1, pc1=gpc1, sl0=gsl0, sl1=gsl1, mc1=gmc1),
        query_open=qr_open, qr0=qr0, qr1=qr1, qrc1=qrc1,
        query_c0=qc0, query_c1=qc1, query_cl1=qcl1,
        blocks=base['blocks'],
    )


def make_mask_refine(n_attempts: int, n_blocks: int, seg_len: int,
                     slot_len: int, warmup_len: int, out_len: int,
                     latent_len: int = 0,
                     mem_window: int = -1) -> np.ndarray:
    """
    Attention mask for refine sequences.

    Standard rules (all rows after encoding):
      - blocked from <x> and encoding <z>

    Post-refine query rows (<q><y> at end) — strict bottleneck:
      - can ONLY see final <h> (the last h state) and own <q><y> tokens
      - blocked from: <r> anchor, all <y> attempts, all <z> corrections,
        all encoding <h>, all attempt correction <h>
      - forces recall to route through final <h> only, eliminating copy shortcuts
    """
    pos    = refine_positions(n_attempts, n_blocks, seg_len, slot_len,
                              warmup_len, out_len, latent_len)
    L      = pos['L']
    blocks = pos['blocks']
    enc_end = pos['encoding_end']
    final   = pos['final']

    r = np.arange(L)
    c = np.arange(L)
    causal  = c[None, :] <= r[:, None]
    blocked = np.zeros((L, L), dtype=bool)

    # Source and encoding-latent column masks
    is_any_src     = np.zeros(L, dtype=bool)
    is_any_enc_lat = np.zeros(L, dtype=bool)
    for b in blocks:
        is_any_src     |= (c >= b['s0']) & (c < b['s1'])
        if latent_len > 0:
            is_any_enc_lat |= (c >= b['p0']) & (c < b['p1'])

    # All rows after encoding end blocked from src and encoding-latent
    recall_rows = r >= enc_end
    blocked |= recall_rows[:, None] & is_any_src[None, :]
    blocked |= recall_rows[:, None] & is_any_enc_lat[None, :]

    # Cross-block encoding <h> isolation (same as make_mask_multi)
    for i, b in enumerate(blocks):
        slot_row = (r >= b['sl0']) & (r < b['sl1'])
        for j, bj in enumerate(blocks):
            if j == i:
                continue
            cross = ((c >= bj['s0']) & (c < bj['s1'])) | \
                    ((c >= bj['p0']) & (c < bj['p1']))
            blocked |= slot_row[:, None] & cross[None, :]
            if j < i:
                outside_window = (mem_window != -1) and ((i - j) >= mem_window)
                if outside_window:
                    blocked |= slot_row[:, None] & ((c >= bj['sl0']) & (c < bj['sl1']))[None, :]

    # Post-refine query rows: can only see final <h> and own tokens
    # final <h> region: [final['pc1'], final['mc1'])
    is_query_row = r >= pos['query_open']
    final_h_col  = (c >= final['pc1']) & (c < final['mc1'])   # final <h> open+slots+</h>
    own_col      = c >= pos['query_open']                       # own <q><y> tokens (causal)
    allowed_for_query = final_h_col | own_col
    blocked |= is_query_row[:, None] & ~allowed_for_query[None, :]

    visible = causal & ~blocked
    return np.where(visible, 0.0, -1e9).astype(np.float32)


def extract_multi_batch(tokens_np: np.ndarray, pos: dict, n_blocks: int) -> tuple:
    """
    Extract (segs, wm, tgt) from a make_multi_batch result.

    segs : (B, n_blocks, seg_len) int64  — raw source segments per block
    wm   : (B, warmup_len) int64         — warmup tokens (query anchor)
    tgt  : (B, out_len) int64            — output target tokens

    Used by online_ref trajectory to share segments/targets between the I Q pass
    (for teacher h computation) and the I R Q pass (for correction training).
    """
    B = tokens_np.shape[0]
    seg_len = pos['blocks'][0]['s1'] - pos['blocks'][0]['s0']
    segs = np.zeros((B, n_blocks, seg_len), dtype=tokens_np.dtype)
    for k, b in enumerate(pos['blocks']):
        segs[:, k, :] = tokens_np[:, b['s0']:b['s1']]
    wm  = tokens_np[:, pos['f0']:pos['f1']]
    tgt = tokens_np[:, pos['c0']:pos['c1']]
    return segs, wm, tgt


def make_refine_batch(rng: np.random.Generator, B: int,
                      n_attempts: int, noise_schedule,
                      n_blocks: int, recall_from: int,
                      seg_len: int, slot_len: int,
                      warmup_len: int, out_len: int,
                      latent_len: int = 0,
                      noise_skew: bool = False,
                      segs_batch: np.ndarray | None = None,
                      wm_batch: np.ndarray | None = None,
                      tgt_batch: np.ndarray | None = None) -> np.ndarray:
    """
    Build refine batch.

    n_attempts: number of attempt turns (each with noisy output + correction block).
    noise_schedule: list of length n_attempts; noise level per attempt.
      - float: fixed rate  - (lo, hi): sampled from Uniform(lo, hi) per example
    Final turn: always noise=0 (clean ground truth), trains the copy mechanism.

    n_attempts=0: sequence identical to standard single-pass recall, just <r> tag.

    segs_batch : (B, n_blocks, seg_len) optional — pre-sampled source segments.
                 If provided, skips internal segment sampling.
    wm_batch   : (B, warmup_len) optional — pre-sampled warmup tokens.
    tgt_batch  : (B, out_len) optional — pre-sampled output targets.
                 wm_batch and tgt_batch must be provided together.

    Returns (B, L) int64.
    """
    assert len(noise_schedule) == n_attempts
    pos      = refine_positions(n_attempts, n_blocks, seg_len, slot_len,
                                warmup_len, out_len, latent_len)
    L        = pos['L']
    base     = pos['base']
    attempts = pos['attempts']
    slot_ids = make_hidden_slot_ids(slot_len)
    corr_ponder_ids = make_latent_slot_ids(latent_len, slot_len) if latent_len > 0 else []
    n_win    = max(1, seg_len - out_len)
    out_arr  = np.zeros((B, L), dtype=np.int64)

    for i in range(B):
        if segs_batch is not None:
            segs = [segs_batch[i, k] for k in range(n_blocks)]
        else:
            segs = [_sample_seg(rng, seg_len) for _ in range(n_blocks)]

        # Encoding blocks
        for k, b in enumerate(base['blocks']):
            seg = segs[k]
            out_arr[i, b['block_start']:b['s0']]  = SRC_OPEN
            out_arr[i, b['s0']:b['s1']]            = seg
            out_arr[i, b['s1']:b['s_close_end']]   = SRC_CLOSE
            if latent_len > 0:
                out_arr[i, b['p_open']:b['p0']]    = PONDER_OPEN
                out_arr[i, b['p0']:b['p1']]        = corr_ponder_ids
                out_arr[i, b['p1']:b['p_close_end']] = PONDER_CLOSE
            out_arr[i, b['p_close_end']:b['sl0']]  = MEM_OPEN
            out_arr[i, b['sl0']:b['sl1']]           = slot_ids
            out_arr[i, b['sl1']:b['mc1']]           = MEM_CLOSE

        # Ground truth — use pre-sampled wm/tgt if provided, else sample fresh
        if wm_batch is not None and tgt_batch is not None:
            wm    = wm_batch[i]
            y_ref = tgt_batch[i]
            alen  = len(y_ref)
        else:
            seg_r    = segs[recall_from]
            y_start  = int(rng.integers(0, n_win + 1))
            y_end    = min(y_start + out_len, seg_len)
            alen     = y_end - y_start
            w_st     = max(0, y_start - warmup_len)
            wm       = seg_r[w_st:y_start]
            if len(wm) < warmup_len:
                wm = np.concatenate([np.full(warmup_len - len(wm), seg_r[0], dtype=np.int32), wm])
            y_ref    = seg_r[y_start:y_end]

        # Warmup: <r>wm</r>
        out_arr[i, pos['r_open']:pos['r0']] = REFINE_OPEN
        out_arr[i, pos['r0']:pos['r1']]     = wm
        out_arr[i, pos['r1']:pos['rc1']]    = REFINE_CLOSE

        # Positional noise weights: linear ramp 0→2 so average = 1.0 (preserves mean noise)
        _pos_weights = np.linspace(0.0, 2.0, alen) if (noise_skew and alen > 1) else np.ones(alen)

        # Attempt turns: <y>noisy</y> <z><h>
        for k, att in enumerate(attempts):
            p = noise_schedule[k]
            if isinstance(p, (list, tuple)):
                p = float(rng.uniform(p[0], p[1]))
            y_out = y_ref.copy()
            if p > 0.0:
                per_pos_p = np.clip(p * _pos_weights, 0.0, 1.0)
                nm = rng.random(alen) < per_pos_p
                y_out[nm] = rng.integers(0, 256, size=int(nm.sum())).astype(np.int32)
            out_arr[i, att['c0'] - OUTPUT_OPEN_LEN : att['c0']]     = OUTPUT_OPEN
            out_arr[i, att['c0']:att['c0'] + alen]                   = y_out
            out_arr[i, att['c1']:att['cl1']]                         = OUTPUT_CLOSE
            if latent_len > 0:
                out_arr[i, att['cl1']:att['p0']]                     = PONDER_OPEN
                out_arr[i, att['p0']:att['p1']]                      = corr_ponder_ids
                out_arr[i, att['p1']:att['pc1']]                     = PONDER_CLOSE
            out_arr[i, att['pc1']:att['sl0']]                        = MEM_OPEN
            out_arr[i, att['sl0']:att['sl1']]                        = slot_ids
            out_arr[i, att['sl1']:att['mc1']]                        = MEM_CLOSE

        # Copy turn: <y>clean</y>  (no noise, trains copy mechanism)
        out_arr[i, pos['copy_c0'] - OUTPUT_OPEN_LEN : pos['copy_c0']] = OUTPUT_OPEN
        out_arr[i, pos['copy_c0']:pos['copy_c0'] + alen]               = y_ref
        out_arr[i, pos['copy_c1']:pos['copy_cl1']]                     = OUTPUT_CLOSE

        # Final correction: <z><h>  (updates h_final from clean copy output)
        g = pos['final']
        if latent_len > 0:
            out_arr[i, g['p0'] - LATENT_OPEN_LEN : g['p0']] = PONDER_OPEN
            out_arr[i, g['p0']:g['p1']]                        = corr_ponder_ids
            out_arr[i, g['p1']:g['pc1']]                       = PONDER_CLOSE
        out_arr[i, g['pc1']:g['sl0']]  = MEM_OPEN
        out_arr[i, g['sl0']:g['sl1']]  = slot_ids
        out_arr[i, g['sl1']:g['mc1']]  = MEM_CLOSE

        # Post-refine query: <q>wm</q><y>clean</y>  (loss target, must match 100%)
        out_arr[i, pos['query_open']:pos['qr0']] = QUERY_OPEN
        out_arr[i, pos['qr0']:pos['qr1']]         = wm
        out_arr[i, pos['qr1']:pos['qrc1']]         = QUERY_CLOSE
        out_arr[i, pos['qrc1']:pos['query_c0']]    = OUTPUT_OPEN
        out_arr[i, pos['query_c0']:pos['query_c0'] + alen] = y_ref
        out_arr[i, pos['query_c1']:pos['query_cl1']]       = OUTPUT_CLOSE

    return out_arr


# ---------------------------------------------------------------------------
# Interleaved (int) mode: [block_0 recall_0] [block_1 recall_1] ...
# Each recall_k targets its preceding block_k.
# Later h_k can see prior q/y tokens (interactive: model knows Q&A history).
# Same masking rules as make_mask_multi — just applied to different layout.
# ---------------------------------------------------------------------------

def interleaved_positions(n_blocks: int, seg_len: int, slot_len: int,
                          warmup_len: int, out_len: int,
                          latent_len: int = 0) -> dict:
    """
    Positions for interleaved sequence: N × (block + recall).
    Each sub-unit = <x>src</x>[<z>z</z>]<h>h</h><q>wm</q><y>out</y>
    """
    sub = multi_block_positions(1, seg_len, slot_len, warmup_len, out_len, latent_len)
    unit_len = sub['L']   # length of one (block + recall) unit
    L = unit_len * n_blocks

    units = []
    for k in range(n_blocks):
        base = k * unit_len
        p = {key: sub[key] + base if isinstance(sub[key], (int, np.integer)) else sub[key]
             for key in sub if key != 'blocks'}
        # Adjust single block positions
        b0 = sub['blocks'][0]
        block = {bk: bv + base for bk, bv in b0.items()}
        p['blocks'] = [block]
        p['L'] = unit_len
        units.append(p)

    return dict(units=units, unit_len=unit_len, L=L, n_blocks=n_blocks,
                seg_len=seg_len, slot_len=slot_len, warmup_len=warmup_len,
                out_len=out_len, latent_len=latent_len)


def make_mask_interleaved(n_blocks: int, seg_len: int, slot_len: int,
                          warmup_len: int, out_len: int,
                          latent_len: int = 0,
                          mem_window: int = -1) -> np.ndarray:
    """
    Attention mask for interleaved sequences.
    Same rules as make_mask_multi: q/y blocked from x/z; y write-only;
    cross-block h isolation; h can see prior q/y (interactive).
    """
    pos = interleaved_positions(n_blocks, seg_len, slot_len, warmup_len, out_len, latent_len)
    L = pos['L']
    r = np.arange(L)
    c = np.arange(L)
    causal  = c[None, :] <= r[:, None]
    blocked = np.zeros((L, L), dtype=bool)

    all_x     = np.zeros(L, dtype=bool)
    all_intermed = np.zeros(L, dtype=bool)
    all_y_col = np.zeros(L, dtype=bool)

    for k, unit in enumerate(pos['units']):
        b = unit['blocks'][0]
        all_x |= (c >= b['s0']) & (c < b['s1'])
        if latent_len > 0:
            all_intermed |= (c >= b['p0']) & (c < b['p1'])
        # y write-only
        c0, c1 = unit['c0'], unit['c1']
        all_y_col |= (c >= unit['fc1'])  # c_open onward

        # q/y rows blocked from x/z
        qy_row = r >= unit['recall_start']
        if k < n_blocks - 1:
            next_start = (k + 1) * pos['unit_len']
            qy_row = qy_row & (r < next_start)
        blocked |= qy_row[:, None] & (all_x | all_intermed)[None, :]

        # h cross-block isolation: block h_k from x_j, z_j (j≠k)
        h_row = (r >= b['sl0']) & (r < b['sl1'])
        for j, u2 in enumerate(pos['units']):
            if j == k: continue
            b2 = u2['blocks'][0]
            cross = ((c >= b2['s0']) & (c < b2['s1'])) | \
                    ((c >= b2['p0']) & (c < b2['p1']))
            # also block cross-block h slots outside mem_window
            if j < k:
                outside = (mem_window != -1) and ((k - j) >= mem_window)
                if outside:
                    cross |= (c >= b2['sl0']) & (c < b2['sl1'])
            blocked |= h_row[:, None] & cross[None, :]

    # y write-only: nothing outside y attends to y cols
    # (but later q/y in same or later units can see earlier y causally)
    for k, unit in enumerate(pos['units']):
        y_col = (c >= unit['fc1'])
        if k < n_blocks - 1:
            y_col &= (c < (k + 1) * pos['unit_len'])
        is_y_row_k = (r >= unit['fc1'])
        if k < n_blocks - 1:
            is_y_row_k &= (r < (k + 1) * pos['unit_len'])
        # block non-q/y rows from seeing y_k columns
        non_qy_row = ~is_y_row_k
        # but later units' q/y rows CAN see earlier y (interactive)
        for j in range(k + 1, n_blocks):
            later_qy = (r >= pos['units'][j]['recall_start'])
            if j < n_blocks - 1:
                later_qy &= (r < (j + 1) * pos['unit_len'])
            non_qy_row &= ~later_qy
        blocked |= non_qy_row[:, None] & y_col[None, :]

    visible = causal & ~blocked
    return np.where(visible, 0.0, -1e9).astype(np.float32)


def make_interleaved_batch(rng: np.random.Generator, B: int,
                           n_blocks: int, seg_len: int, slot_len: int,
                           warmup_len: int, out_len: int,
                           latent_len: int = 0,
                           q_count: int = -1) -> tuple:
    """
    Build interleaved batch: full layout [block_k recall_k] × n_blocks.

    q_count: how many blocks get real queries with loss supervision.
      -1 (default): all n_blocks get real queries (pure int mode)
       k in [1,n]: randomly select k blocks to have active recalls;
                   remaining blocks have their recall region filled with
                   structural tokens but y positions zeroed (no loss).

    Returns (tokens, active_c_ranges) where:
      tokens         : (B, L) int64
      active_c_ranges: list of (c0, c1) output ranges with real targets
                       (one per active query per example — same for all B)
    """
    pos      = interleaved_positions(n_blocks, seg_len, slot_len, warmup_len, out_len, latent_len)
    L        = pos['L']
    slot_ids = make_hidden_slot_ids(slot_len)
    intermed_ids = make_latent_slot_ids(latent_len, slot_len) if latent_len > 0 else []
    out_arr  = np.zeros((B, L), dtype=np.int64)
    n_win    = max(1, seg_len - out_len)

    # Decide which blocks have active recalls (same for whole batch, sampled once)
    if q_count == -1 or q_count >= n_blocks:
        active_blocks = list(range(n_blocks))
    else:
        k = max(1, min(q_count, n_blocks))
        active_blocks = sorted(rng.choice(n_blocks, size=k, replace=False).tolist())
    active_c_ranges = [(pos['units'][k]['c0'], pos['units'][k]['c1']) for k in active_blocks]

    for i in range(B):
        segs = [_sample_seg(rng, seg_len) for _ in range(n_blocks)]
        for k, unit in enumerate(pos['units']):
            seg = segs[k]
            b   = unit['blocks'][0]
            out_arr[i, b['block_start']:b['s0']]         = INPUT_OPEN
            out_arr[i, b['s0']:b['s1']]                  = seg
            out_arr[i, b['s1']:b['s_close_end']]         = INPUT_CLOSE
            if latent_len > 0:
                out_arr[i, b['p_open']:b['p0']]          = LATENT_OPEN
                out_arr[i, b['p0']:b['p1']]              = intermed_ids
                out_arr[i, b['p1']:b['p_close_end']]     = LATENT_CLOSE
            out_arr[i, b['p_close_end']:b['sl0']]        = HIDDEN_OPEN
            out_arr[i, b['sl0']:b['sl1']]                = slot_ids
            out_arr[i, b['sl1']:b['mc1']]                = HIDDEN_CLOSE
            # Always write structural recall tokens
            y_start = int(rng.integers(0, n_win + 1))
            y_end   = min(y_start + out_len, seg_len)
            w_st    = max(0, y_start - warmup_len)
            wm      = seg[w_st:y_start]
            if len(wm) < warmup_len:
                wm = np.concatenate([np.full(warmup_len - len(wm), seg[0], dtype=np.int32), wm])
            rs = unit['recall_start']
            out_arr[i, rs:rs+FROM_OPEN_LEN]                    = FROM_OPEN
            out_arr[i, unit['f0']:unit['f1']]                  = wm
            out_arr[i, unit['f1']:unit['f1']+FROM_CLOSE_LEN]   = FROM_CLOSE
            out_arr[i, unit['fc1']:unit['fc1']+CONT_OPEN_LEN]  = CONT_OPEN
            if k in active_blocks:
                # Active recall: target any previously-seen block (random)
                target_k = int(rng.integers(0, k + 1))   # block 0..k
                seg_r    = segs[target_k]
                yr_start = int(rng.integers(0, n_win + 1))
                yr_end   = min(yr_start + out_len, seg_len)
                wr_st    = max(0, yr_start - warmup_len)
                wm_r     = seg_r[wr_st:yr_start]
                if len(wm_r) < warmup_len:
                    wm_r = np.concatenate([np.full(warmup_len - len(wm_r), seg_r[0], dtype=np.int32), wm_r])
                # Overwrite warmup with target-block warmup
                rs_u = unit['recall_start']
                out_arr[i, unit['f0']:unit['f1']] = wm_r
                out_arr[i, unit['c0']:unit['c0']+(yr_end-yr_start)] = seg_r[yr_start:yr_end]
                out_arr[i, unit['c1']:unit['c1']+CONT_CLOSE_LEN]   = CONT_CLOSE
            # Inactive recall: y region stays zeros (masked from loss)

    return out_arr, active_c_ranges


def make_mask_tag(L_S: int, N: int, L_y: int) -> np.ndarray:
    """
    Attention mask for the tag-based scheme.

    Layout:
      src       [0,         L_S)
      MEM_OPEN  [L_S,       L_S+3)
      slots     [L_S+3,     L_S+3+N)
      MEM_CLOSE [L_S+3+N,   L_S+7+N)
      Y         [L_S+7+N,   L_S+7+N+L_y)

    Rules (same semantics as make_mask_stage0):
      1. Y is write-only from outside Y.
      2. Y cannot attend to src or MEM_OPEN — reads only through slots + MEM_CLOSE.
    """
    L        = L_S + MEM_OVERHEAD + N + L_y
    M_start  = L_S + MEM_OPEN_LEN          # first slot token
    Y_start  = L_S + MEM_OPEN_LEN + N + MEM_CLOSE_LEN  # first Y token

    rows = np.arange(L)[:, None]
    cols = np.arange(L)[None, :]

    causal             = cols <= rows
    is_src_or_open     = cols < M_start    # src + MEM_OPEN tokens
    is_Y_col           = cols >= Y_start
    is_Y_row           = rows >= Y_start

    blocked = (is_Y_col & ~is_Y_row) | (is_Y_row & is_src_or_open)
    visible = causal & ~blocked
    return np.where(visible, 0.0, -1e9).astype(np.float32)


# ---------------------------------------------------------------------------
# V2 marker scheme — 2-byte protocol tokens, no data restrictions except 0x00
# ---------------------------------------------------------------------------
# Reserves only 0x00 (MARK) as the protocol prefix byte.
# Data range: [0x01, 0xFF] — all bytes except 0x00.
# Literal 0x00 in data must be escaped as [MARK, MARK] = [0x00, 0x00].
#
# Protocol sequences (2 bytes each, like XML open/close):
#   [0x00, 0x01]     = <MEM>   open memory block
#   [0x00, 0x02]     = </MEM>  close memory block
#   [0x00, 0x03+i]   = slot i  (up to 252 slots for i in 0..251)
#   [0x00, 0x00]     = escaped literal 0x00 in data
#
# Sequence format:
#   [x_S_encoded | 0x00 0x01 | [0x00 0x03]..[0x00 0x03+N-1] | 0x00 0x02 | Y_encoded]
#   L = len(x_S_encoded) + 2 + 2N + 2 + len(Y_encoded)
#   For data with no 0x00 bytes: len(x_S_encoded) = seg_len, L = seg_len + 2N + seg_len + 4
# ---------------------------------------------------------------------------

MARK_V2      = 0x00   # sole reserved prefix byte
MEM_OPEN_V2  = 0x01   # [MARK_V2, MEM_OPEN_V2]  = open
MEM_CLOSE_V2 = 0x02   # [MARK_V2, MEM_CLOSE_V2] = close
SLOT_BASE_V2 = 0x03   # slot i: [MARK_V2, SLOT_BASE_V2 + i]
DATA_LO_V2   = 0x01   # valid data bytes are [0x01, 0xFF]; 0x00 must be escaped

MAX_SLOTS_V2 = 252    # 0x03..0xFE (0xFF reserved for future use)


def encode_v2(data: bytes | list[int]) -> list[int]:
    """Encode raw bytes into V2 format: escape 0x00 → [0x00, 0x00]."""
    out = []
    for b in data:
        if b == MARK_V2:
            out.append(MARK_V2)
            out.append(MARK_V2)
        else:
            out.append(b)
    return out


def decode_v2(tokens: list[int]) -> bytes:
    """Decode V2-encoded token stream back to original bytes."""
    out = []
    i = 0
    while i < len(tokens):
        if tokens[i] == MARK_V2 and i + 1 < len(tokens) and tokens[i+1] == MARK_V2:
            out.append(0x00)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return bytes(out)


def make_slot_ids_v2(N: int) -> list[int]:
    """
    Return flat token list for N slots in V2 format.
    Each slot i is 2 tokens: [MARK_V2, SLOT_BASE_V2 + i].
    Total length: 2*N tokens.
    """
    assert N <= MAX_SLOTS_V2, f"V2 scheme supports up to {MAX_SLOTS_V2} slots, got {N}"
    tokens = []
    for i in range(N):
        tokens.append(MARK_V2)
        tokens.append(SLOT_BASE_V2 + i)
    return tokens


def make_mem_open_v2() -> list[int]:
    return [MARK_V2, MEM_OPEN_V2]


def make_mem_close_v2() -> list[int]:
    return [MARK_V2, MEM_CLOSE_V2]


def make_mask_stage0_v2(L_S: int, N: int, L_y: int) -> np.ndarray:
    """
    Attention mask for V2 (2-byte marker) scheme.

    Sequence layout:
      [x_S (L_S) | 0x00 0x01 | slots (2N) | 0x00 0x02 | Y (L_y)]
       src         MEM_OPEN    slot_tokens   MEM_CLOSE    output

    Positions:
      src:       [0,        L_S)
      MEM_OPEN:  [L_S,      L_S+2)
      slots:     [L_S+2,    L_S+2+2N)
      MEM_CLOSE: [L_S+2+2N, L_S+4+2N)
      Y:         [L_S+4+2N, L_S+4+2N+L_y)

    Rules (same semantics as v1):
      1. Y is write-only from outside Y.
      2. Y cannot see src or MEM_OPEN.
    """
    L        = L_S + 4 + 2 * N + L_y
    M_start  = L_S + 2            # start of slot tokens
    Y_start  = L_S + 4 + 2 * N   # start of Y tokens

    rows = np.arange(L)[:, None]
    cols = np.arange(L)[None, :]

    causal           = cols <= rows
    is_src_or_open   = cols < M_start       # src + MEM_OPEN
    is_Y_col         = cols >= Y_start
    is_Y_row         = rows >= Y_start

    block_y_sink     = is_Y_col & ~is_Y_row    # rule 1
    block_y_sees_src = is_Y_row & is_src_or_open  # rule 2

    blocked = block_y_sink | block_y_sees_src
    visible = causal & ~blocked
    return np.where(visible, 0.0, -1e9).astype(np.float32)


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

    mem = jnp.array(make_slot_ids(N), dtype=jnp.int32)
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

    mem = jnp.array(make_slot_ids(N), dtype=jnp.int32)
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

    mem = jnp.array(make_slot_ids(N), dtype=jnp.int32)
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

        mem = jnp.array(make_slot_ids(N), dtype=jnp.int32)
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
    g = rng.gamma(max(alpha, 1e-3), size=(V_chain, V_chain)).astype(np.float32)
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
    seq[L_S+1 : L_S+1+N]        = make_slot_ids(N)
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
    seq[L_S+1:L_S+1+N]   = make_slot_ids(N)
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
    seq[L_S+1:L_S+1+N] = make_slot_ids(N)
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
        g = rng.gamma(max(alpha, 1e-3), size=(V_chain, V_chain)).astype(np.float32)
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
            seq[L_S+1:L_S+1+N]   = make_slot_ids(N)
            seq[L_S+1+N]         = ETX
            seq[L_S+2+N:]        = (y_raw + DATA_LO).astype(np.int32)
            out[i] = seq
        else:
            out[i] = np_make_one(rng, V_chain, L_S, L_y, N, alpha)
    return out


def np_make_recall_batch(rng: np.random.Generator, B: int,
                         L_S: int, N: int) -> np.ndarray:
    """
    Recall batch: Y = copy of x_S.
    Sequence: [x_S | STX | NUL*N | ETX | x_S]
    x_S is random bytes in [DATA_LO, 0xFF].
    Shape: (B, L_S + 2 + N + L_S).
    """
    L_y = L_S   # Y is a copy of x_S
    L   = L_S + 2 + N + L_y
    out = np.empty((B, L), dtype=np.int32)
    for i in range(B):
        x_S = rng.integers(DATA_LO, 256, size=L_S).astype(np.int32)
        out[i, :L_S]             = x_S
        out[i, L_S]              = STX
        out[i, L_S+1 : L_S+1+N] = make_slot_ids(N)
        out[i, L_S+1+N]         = ETX
        out[i, L_S+2+N:]        = x_S   # Y = exact copy of x_S
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
