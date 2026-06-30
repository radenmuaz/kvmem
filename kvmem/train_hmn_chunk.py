"""
kvmem/train_hmn_chunk.py — Incremental chunk memorization with SRS recall schedule.

All in ONE forward pass per training step.

Sequence layout:
  Encoding:  [chunk_0: C][SLOT×s]  [chunk_1: C][SLOT×s]  ...  [chunk_{N-1}: C][SLOT×s]
  SRS recall: for each span in srs_schedule(N), ir_turns attempts:
    [SLOT×s][out_noisy: span_len]   ← attempt 0: noisy ground truth
    [SLOT×s][out_clean: span_len]   ← attempt 1: clean ground truth (loss here)

Encoding SLOT_k attends to: own chunk_k + all previous SLOT_j (j<k).
Encoding SLOT_k blocked from: all other chunks (cross-block isolation).
Recall SLOT rows: blocked from ALL chunk regions; sees all prior SLOTs + prior recall outputs.
Recall output rows: can only see own SLOT + own prior output (strong bottleneck).

Train set: fully synthetic rng.integers(0, 256) bytes.
Val set:   make_test_sequences split into n_chunks equal chunks.
Test set:  datasets/suratalfatihah.txt — padded to (n_chunks, chunk_len), eval-only.

Usage:
    python -m kvmem.train_hmn_chunk --config configs/hmn_chunk_wide.py --device mps
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from kvmem.model import build_model
from kvmem.data import HMN_SLOT_0, HMN_VOCAB_SIZE
from kvmem.utils import make_test_sequences, cer
from kvmem.train_hmn_mono import _positional_ls_nll, load_config


# ---------------------------------------------------------------------------
# Slot helpers
# ---------------------------------------------------------------------------

def _slot_ids(slot_len: int, slot_count: int = 2) -> list[int]:
    return [HMN_SLOT_0 + (i % slot_count) for i in range(slot_len)]


# ---------------------------------------------------------------------------
# SRS schedule
# ---------------------------------------------------------------------------

def srs_schedule(n_chunks: int) -> list[tuple[int, int]]:
    """
    Returns recall spans (start_incl, end_excl) in SRS order.

    Pattern: first-half singles → first-half pairs → second-half singles →
             second-half pairs → full sequence.

    N=4 → 7 spans; N=8 → 13 spans.
    """
    half = n_chunks // 2
    schedule: list[tuple[int, int]] = []
    for i in range(half):            schedule.append((i, i + 1))       # first-half singles
    for i in range(0, half, 2):      schedule.append((i, i + 2))       # first-half pairs
    for i in range(half, n_chunks):  schedule.append((i, i + 1))       # second-half singles
    for i in range(half, n_chunks, 2): schedule.append((i, i + 2))     # second-half pairs
    schedule.append((0, n_chunks))                                      # full sequence
    return schedule


def srs_schedule_depth2(n_chunks: int) -> list[tuple[int, int]]:
    """
    Depth-2 SRS: halves then full — 3 spans only (no singles/quarters).

    n=2 → [(0,1),(1,2),(0,2)]   (halves = singles for n=2)
    n=4 → [(0,2),(2,4),(0,4)]   (512-byte halves, 1024-byte full)
    """
    half = n_chunks // 2
    return [(0, half), (half, n_chunks), (0, n_chunks)]


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def chunk_positions(n_chunks: int, chunk_len: int, slot_len: int,
                    schedule: list[tuple[int, int]], ir_turns: int = 2,
                    warmup_len: int = 0) -> dict:
    """
    Returns dict:
      enc_blocks[k] = {s0, s1, sl0, sl1}
      rec_blocks[j] = {sl0, sl1, w0, w1, c0, c1, span, span_len, out_len, turn, is_clean}
      enc_end, warmup_len, L

    Recall block layout: [SLOT×slot_len][warmup: warmup_len][out: span_len-warmup_len]
    warmup=0 collapses to the no-cue case.
    Total block length = slot_len + span_len (unchanged by warmup_len).
    """
    enc_block_len = chunk_len + slot_len
    enc_blocks = []
    for k in range(n_chunks):
        s0  = k * enc_block_len
        s1  = s0 + chunk_len
        sl0 = s1
        sl1 = sl0 + slot_len
        enc_blocks.append(dict(s0=s0, s1=s1, sl0=sl0, sl1=sl1))

    enc_end = n_chunks * enc_block_len
    rec_blocks = []
    offset = enc_end
    for span in schedule:
        span_start, span_end = span
        span_len = (span_end - span_start) * chunk_len
        out_len  = span_len - warmup_len
        for turn in range(ir_turns):
            sl0 = offset
            sl1 = sl0 + slot_len
            w0  = sl1
            w1  = w0  + warmup_len
            c0  = w1
            c1  = c0  + out_len
            rec_blocks.append(dict(
                sl0=sl0, sl1=sl1,
                w0=w0, w1=w1,
                c0=c0, c1=c1,
                span=span, span_len=span_len, out_len=out_len,
                turn=turn, is_clean=(turn == ir_turns - 1),
            ))
            offset = c1

    return dict(enc_blocks=enc_blocks, rec_blocks=rec_blocks,
                enc_end=enc_end, warmup_len=warmup_len, L=offset)


def chunk_positions_fb(n_chunks: int, chunk_len: int, slot_len: int,
                       schedule: list[tuple[int, int]],
                       warmup_len: int = 8, with_ir: bool = True) -> dict:
    """
    Positions for feedback-argmax IR layout. Two turn types per span:

    Turn 0 (IQ):  [SLOT_0: s][warmup: wl][out_0: ol]
    Turn 1 (IR):  [SLOT_A: s][argmax_0: ol][SLOT_B: s][warmup: wl][out_1: ol]

    where ol = span_len - warmup_len.

    with_ir=False: emit only the IQ turn per span (is_clean=True on IQ so the
    training loop computes loss on it) — used for the iq_windowed trajectory
    (recall-only curriculum stage, no feedback).

    rec_blocks entries:
      type='iq': {sl0, sl1, w0, w1, c0, c1, span, out_len}
      type='ir': {sla0, sla1, am0, am1, slb0, slb1, w0, w1, c0, c1, span, out_len,
                  argmax_src_c0}  — argmax_src_c0 is the token position to copy
                  generated content from (generalizes "which earlier block fed
                  this turn's argmax cue" beyond just "the same-span IQ block").
    """
    enc_block_len = chunk_len + slot_len
    enc_blocks = []
    for k in range(n_chunks):
        s0  = k * enc_block_len
        s1  = s0 + chunk_len
        sl0 = s1; sl1 = sl0 + slot_len
        enc_blocks.append(dict(s0=s0, s1=s1, sl0=sl0, sl1=sl1))

    enc_end = n_chunks * enc_block_len
    rec_blocks = []
    offset = enc_end

    for span in schedule:
        span_start, span_end = span
        span_len = (span_end - span_start) * chunk_len
        out_len  = span_len - warmup_len

        # Turn 0 (IQ)
        sl0 = offset; sl1 = sl0 + slot_len
        w0  = sl1;    w1  = w0  + warmup_len
        c0  = w1;     c1  = c0  + out_len
        rec_blocks.append(dict(type='iq', sl0=sl0, sl1=sl1, w0=w0, w1=w1,
                               c0=c0, c1=c1, span=span, span_len=span_len,
                               out_len=out_len, is_clean=(not with_ir)))
        iq_c0, iq_c1 = c0, c1
        offset = c1

        if not with_ir:
            continue

        # Turn 1 (IR feedback): [SLOT_A][argmax_0][SLOT_B][warmup][out_1]
        sla0 = offset; sla1 = sla0 + slot_len
        am0  = sla1;   am1  = am0  + out_len      # argmax from turn 0
        slb0 = am1;    slb1 = slb0 + slot_len
        w0   = slb1;   w1   = w0   + warmup_len
        c0   = w1;     c1   = c0   + out_len
        rec_blocks.append(dict(type='ir', sla0=sla0, sla1=sla1,
                               am0=am0, am1=am1,
                               slb0=slb0, slb1=slb1,
                               w0=w0, w1=w1, c0=c0, c1=c1,
                               span=span, span_len=span_len, out_len=out_len,
                               iq_c0=iq_c0, iq_c1=iq_c1,
                               argmax_src_c0=iq_c0,
                               is_clean=True))
        offset = c1

    return dict(enc_blocks=enc_blocks, rec_blocks=rec_blocks,
                enc_end=enc_end, warmup_len=warmup_len, L=offset)


def chunk_positions_fb_stitch(n_chunks: int, chunk_len: int, slot_len: int,
                              warmup_len: int, windows: list[tuple[int, int]],
                              final_iq: bool = True) -> dict:
    """
    Overlapping-window stitching layout: each span in `windows` gets its own
    IQ+argmax-IR pair (same per-span mechanism as chunk_positions_fb,
    reused directly — it only ever indexes spans by chunk range, so
    consecutive overlapping windows work with no changes). Loss is on each
    window's IR turn (clean), matching "splits are IR".

    windows is expected to overlap and cover the full source, e.g. for
    n_chunks=4 (4 chunks covering the source), window=2 chunks, stride=1
    chunk: [(0,2),(1,3),(2,4)].

    final_iq=True appends one trailing IQ-only turn over the FULL span
    (0, n_chunks) after all window turns — tests whether the model can
    reconstruct the entire source start-to-finish after being refined only
    on overlapping sub-windows (the "stitching" capability). Loss is
    computed on this turn too (is_clean=True).
    final_iq=False omits it (ablation: train windows only).
    """
    pos = chunk_positions_fb(n_chunks, chunk_len, slot_len, windows, warmup_len, with_ir=True)
    enc_blocks = pos['enc_blocks']
    rec_blocks = list(pos['rec_blocks'])
    offset = pos['L']

    if final_iq:
        full_span = (0, n_chunks)
        span_len  = n_chunks * chunk_len
        out_len   = span_len - warmup_len
        sl0 = offset; sl1 = sl0 + slot_len
        w0  = sl1;    w1  = w0  + warmup_len
        c0  = w1;     c1  = c0  + out_len
        rec_blocks.append(dict(type='iq', sl0=sl0, sl1=sl1, w0=w0, w1=w1,
                               c0=c0, c1=c1, span=full_span, span_len=span_len,
                               out_len=out_len, is_clean=True))
        offset = c1

    return dict(enc_blocks=enc_blocks, rec_blocks=rec_blocks,
               enc_end=pos['enc_end'], warmup_len=warmup_len, L=offset)


def chunk_positions_fb_winrefine(n_chunks: int, chunk_len: int, slot_len: int,
                                 warmup_len: int, window: tuple[int, int],
                                 n_refine: int = 2) -> dict:
    """
    Windowed-refinement layout: ONE global IQ turn reads the full source,
    followed by `n_refine` chained IR turns that each refine only `window`
    (a chunk-index sub-span of the full source).

    [Enc]
    [IQ:  SLOT][warmup:wl][out: n_chunks*chunk_len - wl]            ← span=(0,n_chunks)
    [IR_1: SLOT_A][argmax: window out-bytes][SLOT_B][warmup:wl][out: window out-bytes]
    [IR_2: SLOT_A][argmax: IR_1's own out]   [SLOT_B][warmup:wl][out: window out-bytes]
    ... up to n_refine IR turns, all targeting the same `window`.

    IR_1's argmax is a byte-slice of the IQ turn's own output (argmax_src_c0 =
    iq_c0 + window_byte_start — see plan doc for the derivation). IR_2+'s
    argmax is a direct copy of the previous IR turn's own output (same
    mechanism chunk_positions_fb already uses turn-to-turn).

    window=(0, n_chunks) degenerates cleanly to "refine the full span."
    """
    full_span = (0, n_chunks)
    pos = chunk_positions_fb(n_chunks, chunk_len, slot_len, [full_span],
                             warmup_len, with_ir=False)
    enc_blocks = pos['enc_blocks']
    rec_blocks = list(pos['rec_blocks'])
    iq_block   = rec_blocks[0]
    iq_block['is_clean'] = False    # no loss on the windowed-refine IQ turn itself
    offset = pos['L']

    win_s, win_e = window
    win_byte_start = win_s * chunk_len
    win_span_len   = (win_e - win_s) * chunk_len
    win_out_len    = win_span_len - warmup_len

    prev_c0 = None
    for step in range(n_refine):
        sla0 = offset; sla1 = sla0 + slot_len
        am0  = sla1;   am1  = am0  + win_out_len
        slb0 = am1;    slb1 = slb0 + slot_len
        w0   = slb1;   w1   = w0   + warmup_len
        c0   = w1;     c1   = c0   + win_out_len

        argmax_src_c0 = (iq_block['c0'] + win_byte_start) if step == 0 else prev_c0

        rec_blocks.append(dict(type='ir', sla0=sla0, sla1=sla1,
                               am0=am0, am1=am1, slb0=slb0, slb1=slb1,
                               w0=w0, w1=w1, c0=c0, c1=c1,
                               span=window, span_len=win_span_len, out_len=win_out_len,
                               argmax_src_c0=argmax_src_c0, is_clean=True))
        prev_c0 = c0
        offset = c1

    return dict(enc_blocks=enc_blocks, rec_blocks=rec_blocks,
                enc_end=pos['enc_end'], warmup_len=warmup_len, L=offset)


def chunk_positions_fb_localrefine(n_chunks: int, chunk_len: int, slot_len: int,
                                   warmup_len: int, windows: list[tuple[int, int]],
                                   n_refine: int = 2) -> dict:
    """
    Per-window local-refine layout: each `window` in `windows` gets its OWN
    local IQ turn (reads/recalls only that window's chunks) followed by
    `n_refine` chained argmax-IR turns refining the same window — i.e. the
    proven `hmn_feedback_32_ir` mechanism (IQ + 2-step argmax-refine),
    applied per-window instead of always to "the whole source".

    Windows are processed in sequence (threading a running `offset`), all
    sharing ONE encoding pass over `n_chunks`. n_refine=0 collapses a
    window's unit to IQ-only (is_clean=True on the IQ block, no IR).

    Differs from chunk_positions_fb_winrefine (parked, larger-scale
    experiment): that one shares a SINGLE global full-span IQ across all
    windows; here every window is fully self-contained (own IQ + own
    refine chain) — required for overlapping windows that don't share a
    single "read everything once" stage.
    """
    enc_block_len = chunk_len + slot_len
    enc_blocks = []
    for k in range(n_chunks):
        s0  = k * enc_block_len
        s1  = s0 + chunk_len
        sl0 = s1; sl1 = sl0 + slot_len
        enc_blocks.append(dict(s0=s0, s1=s1, sl0=sl0, sl1=sl1))

    enc_end = n_chunks * enc_block_len
    offset = enc_end
    rec_blocks = []

    for window in windows:
        win_s, win_e = window
        span_len = (win_e - win_s) * chunk_len
        out_len  = span_len - warmup_len

        # Local IQ — reads/recalls only this window
        sl0 = offset; sl1 = sl0 + slot_len
        w0  = sl1;    w1  = w0  + warmup_len
        c0  = w1;     c1  = c0  + out_len
        iq_block = dict(type='iq', sl0=sl0, sl1=sl1, w0=w0, w1=w1, c0=c0, c1=c1,
                        span=window, span_len=span_len, out_len=out_len,
                        is_clean=(n_refine == 0))
        rec_blocks.append(iq_block)
        offset  = c1
        prev_c0 = iq_block['c0']

        for step in range(n_refine):
            sla0 = offset; sla1 = sla0 + slot_len
            am0  = sla1;   am1  = am0  + out_len
            slb0 = am1;    slb1 = slb0 + slot_len
            ww0  = slb1;   ww1  = ww0  + warmup_len
            cc0  = ww1;    cc1  = cc0  + out_len

            rec_blocks.append(dict(type='ir', sla0=sla0, sla1=sla1,
                                   am0=am0, am1=am1, slb0=slb0, slb1=slb1,
                                   w0=ww0, w1=ww1, c0=cc0, c1=cc1,
                                   span=window, span_len=span_len, out_len=out_len,
                                   argmax_src_c0=prev_c0, is_clean=True))
            prev_c0 = cc0
            offset  = cc1

    return dict(enc_blocks=enc_blocks, rec_blocks=rec_blocks,
               enc_end=enc_end, warmup_len=warmup_len, L=offset)


# ---------------------------------------------------------------------------
# Attention mask
# ---------------------------------------------------------------------------

def chunk_mask(pos: dict) -> np.ndarray:
    """
    Build (L, L) attention mask. Convention: 0.0 = attend, -1e9 = blocked.

    Rules (on top of causal):
      2. Encoding SLOT_k blocked from chunk_j for j != k.
      3. Recall SLOT rows blocked from ALL chunk columns.
      4a. Recall warmup rows blocked from everything except own SLOT + own warmup.
      4b. Recall output rows blocked from everything except own SLOT + own warmup + own output.

    warmup_len=0: w0==w1 → empty ranges → rules 4a/4b collapse to the no-warmup case.
    """
    L = pos['L']
    r = np.arange(L)
    c = np.arange(L)
    causal  = c[None, :] <= r[:, None]
    blocked = np.zeros((L, L), dtype=bool)

    enc_blocks = pos['enc_blocks']
    rec_blocks = pos['rec_blocks']

    is_any_chunk = np.zeros(L, dtype=bool)
    for b in enc_blocks:
        is_any_chunk |= (c >= b['s0']) & (c < b['s1'])

    # Rule 2: encoding SLOT_k rows blocked from chunk_j (j != k)
    for k, b in enumerate(enc_blocks):
        sl_row = (r >= b['sl0']) & (r < b['sl1'])
        for j, bj in enumerate(enc_blocks):
            if j == k:
                continue
            chunk_j = (c >= bj['s0']) & (c < bj['s1'])
            blocked |= sl_row[:, None] & chunk_j[None, :]

    # Rule 3: recall SLOT rows blocked from all chunk columns
    for rb in rec_blocks:
        sl_row = (r >= rb['sl0']) & (r < rb['sl1'])
        blocked |= sl_row[:, None] & is_any_chunk[None, :]

    for rb in rec_blocks:
        own_sl_col  = (c >= rb['sl0']) & (c < rb['sl1'])
        own_wm_col  = (c >= rb['w0'])  & (c < rb['w1'])   # empty when warmup_len=0
        own_out_col = (c >= rb['c0'])  & (c < rb['c1'])

        # Rule 4a: warmup rows — can only see own SLOT + own warmup
        if rb['w0'] < rb['w1']:
            wm_row = (r >= rb['w0']) & (r < rb['w1'])
            blocked |= wm_row[:, None] & ~(own_sl_col | own_wm_col)[None, :]

        # Rule 4b: output rows — can only see own SLOT + own warmup + own output
        out_row = (r >= rb['c0']) & (r < rb['c1'])
        blocked |= out_row[:, None] & ~(own_sl_col | own_wm_col | own_out_col)[None, :]

    visible = causal & ~blocked
    return np.where(visible, 0.0, -1e9).astype(np.float32)


def chunk_mask_fb(pos: dict) -> np.ndarray:
    """
    Mask for feedback-argmax IR layout. Same rules as chunk_mask for encoding
    blocks and IQ turns. Additional rules for IR turns:

    5. SLOT_A rows: blocked from all chunks (like all recall SLOTs).
    6. argmax rows: blocked from all chunks.
    7. SLOT_B rows: blocked from all chunks; sees SLOT_A + argmax causally.
    8. IR warmup/out rows: blocked from everything except own SLOT_B + own warmup/out.
       (Same strong bottleneck as IQ out rows, but SLOT_B is the gate — not SLOT_A or argmax.)
    """
    L = pos['L']
    r = np.arange(L); c = np.arange(L)
    causal  = c[None, :] <= r[:, None]
    blocked = np.zeros((L, L), dtype=bool)

    enc_blocks = pos['enc_blocks']
    rec_blocks = pos['rec_blocks']

    is_any_chunk = np.zeros(L, dtype=bool)
    for b in enc_blocks:
        is_any_chunk |= (c >= b['s0']) & (c < b['s1'])

    # Rule 2: encoding SLOT_k blocked from chunk_j (j≠k)
    for k, b in enumerate(enc_blocks):
        sl_row = (r >= b['sl0']) & (r < b['sl1'])
        for j, bj in enumerate(enc_blocks):
            if j == k: continue
            blocked |= sl_row[:, None] & ((c >= bj['s0']) & (c < bj['s1']))[None, :]

    for rb in rec_blocks:
        if rb['type'] == 'iq':
            # Rule 3 (IQ SLOT): blocked from all chunks
            sl_row = (r >= rb['sl0']) & (r < rb['sl1'])
            blocked |= sl_row[:, None] & is_any_chunk[None, :]
            # Rule 4a: IQ warmup rows — own SLOT + own warmup only
            if rb['w0'] < rb['w1']:
                wm_row = (r >= rb['w0']) & (r < rb['w1'])
                own = (c >= rb['sl0']) & (c < rb['sl1']) | (c >= rb['w0']) & (c < rb['w1'])
                blocked |= wm_row[:, None] & ~own[None, :]
            # Rule 4b: IQ out rows — own SLOT + own warmup + own output
            out_row = (r >= rb['c0']) & (r < rb['c1'])
            own = ((c >= rb['sl0']) & (c < rb['sl1']) |
                   (c >= rb['w0'])  & (c < rb['w1'])  |
                   (c >= rb['c0'])  & (c < rb['c1']))
            blocked |= out_row[:, None] & ~own[None, :]

        else:  # type == 'ir'
            # Rules 5,6,7: SLOT_A, argmax, SLOT_B — all blocked from encoding chunks
            sla_row = (r >= rb['sla0']) & (r < rb['sla1'])
            am_row  = (r >= rb['am0'])  & (r < rb['am1'])
            slb_row = (r >= rb['slb0']) & (r < rb['slb1'])
            blocked |= (sla_row | am_row | slb_row)[:, None] & is_any_chunk[None, :]

            # Rule 8: IR warmup/out rows — only own SLOT_B + own warmup + own output
            wm_row  = (r >= rb['w0'])  & (r < rb['w1'])
            out_row = (r >= rb['c0'])  & (r < rb['c1'])
            own_slb = (c >= rb['slb0']) & (c < rb['slb1'])
            own_wm  = (c >= rb['w0'])   & (c < rb['w1'])
            own_out = (c >= rb['c0'])   & (c < rb['c1'])
            allowed = own_slb | own_wm | own_out
            if rb['w0'] < rb['w1']:
                blocked |= wm_row[:, None] & ~(own_slb | own_wm)[None, :]
            blocked |= out_row[:, None] & ~allowed[None, :]

    visible = causal & ~blocked
    return np.where(visible, 0.0, -1e9).astype(np.float32)


# ---------------------------------------------------------------------------
# Batch builder (training — synthetic random bytes)
# ---------------------------------------------------------------------------

def _chunk_make_batch(rng: np.random.Generator, B: int,
                      n_chunks: int, chunk_len: int,
                      slot_len: int, slot_count: int,
                      schedule: list[tuple[int, int]],
                      ir_turns: int, noise_p: float,
                      pos: dict | None = None) -> np.ndarray:
    """
    Build (B, L) training batch with fully synthetic random bytes.

    Recall attempts: turn < ir_turns-1 → noisy (noise_p fraction replaced).
                     last turn          → clean ground truth.
    """
    if pos is None:
        pos = chunk_positions(n_chunks, chunk_len, slot_len, schedule, ir_turns)
    sids = np.array(_slot_ids(slot_len, slot_count), dtype=np.int64)
    L    = pos['L']
    tok  = np.zeros((B, L), dtype=np.int64)

    # Sample random source bytes
    segs = rng.integers(0, 256, size=(B, n_chunks, chunk_len), dtype=np.int64)

    # Fill encoding blocks
    for k, b in enumerate(pos['enc_blocks']):
        tok[:, b['s0']:b['s1']]   = segs[:, k, :]
        tok[:, b['sl0']:b['sl1']] = sids

    wl = pos.get('warmup_len', 0)

    # Fill recall blocks
    for rb in pos['rec_blocks']:
        tok[:, rb['sl0']:rb['sl1']] = sids
        span_start, span_end = rb['span']
        gt = np.concatenate([segs[:, i, :] for i in range(span_start, span_end)], axis=1)
        # Warmup: always ground truth (first wl bytes of span)
        if wl > 0:
            tok[:, rb['w0']:rb['w1']] = gt[:, :wl]
        # Output: clean GT on last turn, noisy GT on earlier turns
        gt_out = gt[:, wl:]                             # (B, out_len)
        if rb['is_clean']:
            tok[:, rb['c0']:rb['c1']] = gt_out
        else:
            noisy = gt_out.copy()
            nm    = rng.random((B, rb['out_len'])) < noise_p
            nv    = rng.integers(0, 256, size=(B, rb['out_len']), dtype=np.int64)
            noisy[nm] = nv[nm]
            tok[:, rb['c0']:rb['c1']] = noisy

    return tok


def _chunk_make_batch_fb(rng: np.random.Generator, B: int,
                         n_chunks: int, chunk_len: int,
                         slot_len: int, slot_count: int,
                         schedule: list[tuple[int, int]],
                         pos: dict) -> np.ndarray:
    """
    Batch builder for feedback-argmax layout.

    IQ turns: warmup = GT prefix, out = GT suffix (teacher-forced clean).
    IR turns: argmax = GT suffix (teacher-forced, same as GT — upgrade to
              actual model argmax by calling _fill_argmax_fb before pass 2).
    """
    sids = np.array(_slot_ids(slot_len, slot_count), dtype=np.int64)
    wl   = pos['warmup_len']
    L    = pos['L']
    tok  = np.zeros((B, L), dtype=np.int64)
    segs = rng.integers(0, 256, size=(B, n_chunks, chunk_len), dtype=np.int64)

    for k, b in enumerate(pos['enc_blocks']):
        tok[:, b['s0']:b['s1']]   = segs[:, k, :]
        tok[:, b['sl0']:b['sl1']] = sids

    for rb in pos['rec_blocks']:
        span_s, span_e = rb['span']
        gt = np.concatenate([segs[:, i, :] for i in range(span_s, span_e)], axis=1)
        gt_out = gt[:, wl:]   # (B, out_len)

        if rb['type'] == 'iq':
            tok[:, rb['sl0']:rb['sl1']] = sids
            if wl > 0:
                tok[:, rb['w0']:rb['w1']] = gt[:, :wl]
            tok[:, rb['c0']:rb['c1']] = gt_out     # clean GT at out_0

        else:  # 'ir'
            tok[:, rb['sla0']:rb['sla1']] = sids   # SLOT_A
            tok[:, rb['am0']:rb['am1']]   = gt_out  # argmax (teacher-forced = GT)
            tok[:, rb['slb0']:rb['slb1']] = sids   # SLOT_B
            if wl > 0:
                tok[:, rb['w0']:rb['w1']] = gt[:, :wl]
            tok[:, rb['c0']:rb['c1']] = gt_out     # clean GT at out_1

    return tok


def _fill_argmax_fb(tok_np: np.ndarray, logits: torch.Tensor,
                    pos: dict) -> np.ndarray:
    """
    Replace teacher-forced argmax tokens with ACTUAL model predictions
    from pass 1. Call between pass 1 (no_grad) and pass 2 (grad).

    logits: (B, L, V) from pass 1 forward.
    Returns updated tok_np with argmax filled at IR turn am0:am1 positions.
    """
    tok = tok_np.copy()
    for rb in pos['rec_blocks']:
        if rb['type'] == 'ir':
            # logits at positions src_c0-1 .. src_c0-1+out_len predict the
            # source block's own output — works whether the source is the
            # same-span IQ block (chunk_positions_fb) or a byte-sliced /
            # chained earlier block (chunk_positions_fb_winrefine).
            src_c0  = rb['argmax_src_c0']
            out_len = rb['out_len']
            am = logits[:, src_c0-1:src_c0-1+out_len].argmax(-1).cpu().numpy()
            tok[:, rb['am0']:rb['am1']] = am
    return tok


# ---------------------------------------------------------------------------
# AR decode (eval)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ar_decode_chunk(model, chunks_arr: list | np.ndarray,
                    slot_len: int, slot_count: int,
                    schedule: list[tuple[int, int]],
                    mask_t: torch.Tensor, pos: dict,
                    device, valid_mask: np.ndarray | None = None) -> dict:
    """
    Greedy AR decode over the SRS sequence.

    chunks_arr: list of arrays, each shape (chunk_len,) int64, or (n_chunks, chunk_len).
    Returns {bpb, nll, match_pct, decoded_bytes, full_block_gen}.

    BPB/match measured on the last clean full-sequence recall block,
    excluding padded positions (valid_mask=True means real byte).
    """
    if isinstance(chunks_arr, np.ndarray) and chunks_arr.ndim == 2:
        chunks_list = [chunks_arr[k] for k in range(chunks_arr.shape[0])]
    else:
        chunks_list = list(chunks_arr)

    n_chunks  = len(chunks_list)
    chunk_len = len(chunks_list[0])
    sids      = np.array(_slot_ids(slot_len, slot_count), dtype=np.int64)
    L         = pos['L']

    # Build initial token array
    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos['enc_blocks']):
        tok[b['s0']:b['s1']]   = chunks_list[k]
        tok[b['sl0']:b['sl1']] = sids

    wl    = pos.get('warmup_len', 0)
    sids_t = torch.tensor(sids, dtype=torch.long, device=device)
    tok_t  = torch.tensor(tok, dtype=torch.long, device=device)

    # AR decode each recall block in SRS order
    for rb in pos['rec_blocks']:
        tok_t[rb['sl0']:rb['sl1']] = sids_t
        # Fill warmup with ground truth prefix of the span
        if wl > 0:
            span_s, span_e = rb['span']
            gt_span = np.concatenate(chunks_list[span_s:span_e])
            tok_t[rb['w0']:rb['w1']] = torch.tensor(
                gt_span[:wl].astype(np.int64), dtype=torch.long, device=device)
        # AR decode output positions only
        for j in range(rb['out_len']):
            logits = model(tok_t, mask_t)
            nb     = int(logits[rb['c0'] + j - 1].argmax())
            tok_t[rb['c0'] + j] = nb

    # Find last clean full-sequence recall block
    full_rb = None
    for rb in reversed(pos['rec_blocks']):
        if rb['is_clean'] and rb['span'] == (0, n_chunks):
            full_rb = rb
            break
    assert full_rb is not None, 'No clean full-sequence recall block found'

    wl       = pos.get('warmup_len', 0)
    out_len  = full_rb['out_len']

    # Extract generated output (output positions only, after warmup)
    gen    = tok_t[full_rb['c0']:full_rb['c1']].cpu().numpy()          # (out_len,)
    target_full = np.concatenate(chunks_list)                           # (span_len,)
    target = target_full[wl:]                                           # (out_len,)

    # valid_mask applies to the output positions
    if valid_mask is not None:
        vm       = valid_mask.flatten()[wl:wl + out_len]
        gen_v    = gen[:len(vm)][vm]
        target_v = target[:len(vm)][vm]
    else:
        gen_v    = gen
        target_v = target[:out_len]

    match_pct = 100.0 * float(np.sum(gen_v == target_v)) / max(len(target_v), 1)

    # Teacher-forced BPB: fill ALL clean recall positions with ground truth
    tok_tf = tok_t.clone()
    for rb2 in pos['rec_blocks']:
        if rb2['is_clean']:
            span_s, span_e = rb2['span']
            gt = np.concatenate(chunks_list[span_s:span_e])
            if wl > 0:
                tok_tf[rb2['w0']:rb2['w1']] = torch.tensor(
                    gt[:wl].astype(np.int64), dtype=torch.long, device=device)
            tok_tf[rb2['c0']:rb2['c1']] = torch.tensor(
                gt[wl:].astype(np.int64), dtype=torch.long, device=device)

    logits_tf  = model(tok_tf, mask_t)                                  # (L, V)
    # NTP: logits at c0-1..c1-2 predict tokens at c0..c1-1
    # c0-1 = last warmup position (or last SLOT if wl=0) → predicts first output byte
    tgt_tensor = torch.tensor(target[:out_len], dtype=torch.long, device=device)
    lp_full    = F.log_softmax(logits_tf[full_rb['c0']-1:full_rb['c1']-1], dim=-1)
    nll_vals   = -lp_full.gather(1, tgt_tensor.unsqueeze(1)).squeeze(1).cpu().numpy()

    if valid_mask is not None:
        vm_flat  = valid_mask.flatten()[wl:wl + out_len]
        nll_vals = nll_vals[vm_flat[:len(nll_vals)]]

    nll = float(nll_vals.mean())
    bpb = nll / math.log(2)

    return dict(bpb=bpb, nll=nll, match_pct=match_pct,
                decoded_bytes=gen.tolist(), n_valid=len(target_v))


# ---------------------------------------------------------------------------
# KV-cached AR decode  (same semantics, ~L× faster for large sequences)
# ---------------------------------------------------------------------------

def _cat_kv(kv_a: list, kv_b: list) -> list:
    """Concatenate two layer-wise KV caches along the sequence dim (dim=2)."""
    return [(torch.cat([ka, kb], dim=2), torch.cat([va, vb], dim=2))
            for (ka, va), (kb, vb) in zip(kv_a, kv_b)]


@torch.no_grad()
def ar_decode_chunk_kv(model, chunks_arr, slot_len: int, slot_count: int,
                       schedule, mask_np: np.ndarray, pos: dict,
                       device, valid_mask: np.ndarray | None = None) -> dict:
    """
    KV-cached greedy AR decode over the SRS sequence.

    The mask is strictly causal (source-first layout: [chunk][SLOT]),
    so incremental KV caching is correct: each new token's attention only
    reaches positions that are earlier in the sequence.

    Strategy:
      1. Process all encoding blocks as one prefix → cache KV.
      2. For each recall block, process its SLOT + warmup against the cache,
         then generate output tokens one at a time extending the cache.
      3. Each step: O(1 × L_cached × d) attention instead of O(L² × d).

    Speedup: ~3-10× vs ar_decode_chunk for stage-2 sequences (L=1826).
    """
    if isinstance(chunks_arr, np.ndarray) and chunks_arr.ndim == 2:
        chunks_list = [chunks_arr[k] for k in range(chunks_arr.shape[0])]
    else:
        chunks_list = list(chunks_arr)

    n_chunks   = len(chunks_list)
    wl         = pos.get('warmup_len', 0)
    sids       = np.array(_slot_ids(slot_len, slot_count), dtype=np.int64)
    L          = pos['L']
    full_mask  = torch.tensor(mask_np, dtype=torch.float32, device=device)

    # Build initial token array (all non-output positions filled)
    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos['enc_blocks']):
        tok[b['s0']:b['s1']]   = chunks_list[k]
        tok[b['sl0']:b['sl1']] = sids
    for rb in pos['rec_blocks']:
        tok[rb['sl0']:rb['sl1']] = sids
        if wl > 0:
            span_s, span_e = rb['span']
            gt_span = np.concatenate(chunks_list[span_s:span_e])
            tok[rb['w0']:rb['w1']] = gt_span[:wl].astype(np.int64)

    # ── Step 1: process encoding prefix at once ──────────────────────────
    enc_end  = pos['enc_end']
    enc_t    = torch.tensor(tok[:enc_end], dtype=torch.long, device=device)
    enc_mask = full_mask[:enc_end, :enc_end]
    _, kv_cache = model(enc_t, enc_mask, return_kv=True)
    L_cached = enc_end

    # ── Step 2: recall blocks — process SLOT+warmup, then generate outputs ─
    all_gen = {}  # (span, turn) → list[int]
    for rb in pos['rec_blocks']:
        # Process SLOT + warmup (known tokens) against the growing KV cache
        seg_start = rb['sl0']
        seg_end   = rb['c0']     # everything before the output region
        seg_len   = seg_end - seg_start
        seg_t     = torch.tensor(tok[seg_start:seg_end], dtype=torch.long, device=device)
        seg_mask  = full_mask[seg_start:seg_end, :L_cached + seg_len]
        seg_logits, seg_kv = model(seg_t, seg_mask, past_kv=kv_cache,
                                   return_kv=True, offset=L_cached)
        kv_cache  = _cat_kv(kv_cache, seg_kv)
        L_cached += seg_len

        # seg_logits[-1] is at position c0-1 → predicts first output token
        gen = [int(seg_logits[-1].argmax())]
        tok[rb['c0']] = gen[0]

        # Generate remaining output tokens one by one
        for j in range(1, rb['out_len']):
            prev_pos  = rb['c0'] + j - 1          # position of the just-generated token
            prev_t    = torch.tensor([gen[j-1]], dtype=torch.long, device=device)
            prev_mask = full_mask[prev_pos:prev_pos+1, :L_cached + 1]
            prev_logits, prev_kv = model(prev_t, prev_mask, past_kv=kv_cache,
                                         return_kv=True, offset=L_cached)
            kv_cache  = _cat_kv(kv_cache, prev_kv)
            L_cached += 1
            next_tok  = int(prev_logits[-1].argmax())
            gen.append(next_tok)
            tok[rb['c0'] + j] = next_tok

        all_gen[(rb['span'], rb['turn'])] = gen

    # ── Extract result from last clean full-sequence recall block ─────────
    full_rb = None
    for rb in reversed(pos['rec_blocks']):
        if rb['is_clean'] and rb['span'] == (0, n_chunks):
            full_rb = rb
            break
    assert full_rb is not None

    gen         = np.array(all_gen[(full_rb['span'], full_rb['turn'])], dtype=np.int64)
    target_full = np.concatenate(chunks_list)
    target      = target_full[wl:]
    out_len     = full_rb['out_len']

    if valid_mask is not None:
        vm       = valid_mask.flatten()[wl:wl + out_len]
        gen_v    = gen[:len(vm)][vm]
        target_v = target[:len(vm)][vm]
    else:
        gen_v    = gen
        target_v = target[:out_len]

    match_pct = 100.0 * float(np.sum(gen_v == target_v)) / max(len(target_v), 1)

    # Teacher-forced BPB (single full forward, same as ar_decode_chunk)
    tok_tf = torch.tensor(tok, dtype=torch.long, device=device)
    for rb2 in pos['rec_blocks']:
        if rb2['is_clean']:
            span_s, span_e = rb2['span']
            gt = np.concatenate(chunks_list[span_s:span_e])
            if wl > 0:
                tok_tf[rb2['w0']:rb2['w1']] = torch.tensor(
                    gt[:wl].astype(np.int64), dtype=torch.long, device=device)
            tok_tf[rb2['c0']:rb2['c1']] = torch.tensor(
                gt[wl:].astype(np.int64), dtype=torch.long, device=device)

    logits_tf  = model(tok_tf, full_mask)
    tgt_tensor = torch.tensor(target[:out_len], dtype=torch.long, device=device)
    lp_full    = F.log_softmax(logits_tf[full_rb['c0']-1:full_rb['c1']-1], dim=-1)
    nll_vals   = -lp_full.gather(1, tgt_tensor.unsqueeze(1)).squeeze(1).cpu().numpy()

    if valid_mask is not None:
        vm_flat  = valid_mask.flatten()[wl:wl + out_len]
        nll_vals = nll_vals[vm_flat[:len(nll_vals)]]

    nll = float(nll_vals.mean())
    bpb = nll / math.log(2)
    return dict(bpb=bpb, nll=nll, match_pct=match_pct,
                decoded_bytes=gen.tolist(), n_valid=len(target_v))


@torch.no_grad()
def ar_decode_chunk_fb(model, chunks_arr, slot_len: int, slot_count: int,
                       schedule: list[tuple[int, int]], mask_np: np.ndarray,
                       pos: dict, device,
                       valid_mask: np.ndarray | None = None) -> dict:
    """
    Greedy AR decode for feedback-argmax IR layout.

    For each span:
      Turn 0 (IQ): AR-generate out_0 tokens (bottleneck through SLOT_0).
      Turn 1 (IR): fill argmax from turn-0 output, AR-generate out_1 tokens
                   (bottleneck through SLOT_B which sees argmax).

    BPB measured on the final IR turn's out_1 of the full-sequence span.
    """
    if isinstance(chunks_arr, np.ndarray) and chunks_arr.ndim == 2:
        chunks_list = [chunks_arr[k] for k in range(chunks_arr.shape[0])]
    else:
        chunks_list = list(chunks_arr)

    n_chunks  = len(chunks_list)
    wl        = pos['warmup_len']
    sids      = np.array(_slot_ids(slot_len, slot_count), dtype=np.int64)
    L         = pos['L']
    full_mask = torch.tensor(mask_np, dtype=torch.float32, device=device)

    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos['enc_blocks']):
        tok[b['s0']:b['s1']]   = chunks_list[k]
        tok[b['sl0']:b['sl1']] = sids

    tok_t   = torch.tensor(tok, dtype=torch.long, device=device)
    sids_t  = torch.tensor(sids, dtype=torch.long, device=device)
    span_gen = {}   # span → (iq_gen, ir_gen)

    # Process span by span — each span has IQ then IR turn
    iq_rbs = [rb for rb in pos['rec_blocks'] if rb['type'] == 'iq']
    ir_rbs = {rb['span']: rb for rb in pos['rec_blocks'] if rb['type'] == 'ir'}

    for iq_rb in iq_rbs:
        span = iq_rb['span']
        span_s, span_e = span
        gt_span = np.concatenate(chunks_list[span_s:span_e])

        # ── Turn 0 (IQ): AR generate out_0 ──────────────────────────────
        tok_t[iq_rb['sl0']:iq_rb['sl1']] = sids_t
        if wl > 0:
            tok_t[iq_rb['w0']:iq_rb['w1']] = torch.tensor(
                gt_span[:wl].astype(np.int64), dtype=torch.long, device=device)
        for j in range(iq_rb['out_len']):
            logits = model(tok_t, full_mask)
            nb = int(logits[iq_rb['c0'] + j - 1].argmax())
            tok_t[iq_rb['c0'] + j] = nb

        iq_gen = tok_t[iq_rb['c0']:iq_rb['c1']].cpu().numpy()

        # ── Turn 1 (IR): fill argmax from turn 0, AR generate out_1 ─────
        ir_rb = ir_rbs.get(span)
        if ir_rb is None:
            span_gen[span] = (iq_gen, None)
            continue

        tok_t[ir_rb['sla0']:ir_rb['sla1']] = sids_t                # SLOT_A
        tok_t[ir_rb['am0']:ir_rb['am1']]   = tok_t[iq_rb['c0']:iq_rb['c1']]  # argmax = IQ output
        tok_t[ir_rb['slb0']:ir_rb['slb1']] = sids_t                # SLOT_B
        if wl > 0:
            tok_t[ir_rb['w0']:ir_rb['w1']] = torch.tensor(
                gt_span[:wl].astype(np.int64), dtype=torch.long, device=device)
        for j in range(ir_rb['out_len']):
            logits = model(tok_t, full_mask)
            nb = int(logits[ir_rb['c0'] + j - 1].argmax())
            tok_t[ir_rb['c0'] + j] = nb

        ir_gen = tok_t[ir_rb['c0']:ir_rb['c1']].cpu().numpy()
        span_gen[span] = (iq_gen, ir_gen)

    # ── Result: IR output of full-sequence span ──────────────────────────
    full_span = (0, n_chunks)
    iq_gen, ir_gen = span_gen.get(full_span, (None, None))
    final_gen = ir_gen if ir_gen is not None else iq_gen

    target = np.concatenate(chunks_list)[wl:]
    out_len = pos['rec_blocks'][-1]['out_len']

    if valid_mask is not None:
        vm       = valid_mask.flatten()[wl:wl + out_len]
        gen_v    = final_gen[:len(vm)][vm]
        target_v = target[:len(vm)][vm]
    else:
        gen_v    = final_gen[:out_len]
        target_v = target[:out_len]

    match_pct = 100.0 * float(np.sum(gen_v == target_v)) / max(len(target_v), 1)

    # Teacher-forced BPB on IR output
    full_ir_rb = ir_rbs.get(full_span)
    if full_ir_rb is None:
        full_ir_rb = next(rb for rb in reversed(pos['rec_blocks'])
                         if rb['span'] == full_span)
    tok_tf = tok_t.clone()
    tok_tf[full_ir_rb['c0']:full_ir_rb['c1']] = torch.tensor(
        target[:out_len].astype(np.int64), dtype=torch.long, device=device)
    logits_tf  = model(tok_tf, full_mask)
    tgt_tensor = torch.tensor(target[:out_len], dtype=torch.long, device=device)
    lp_full    = F.log_softmax(logits_tf[full_ir_rb['c0']-1:full_ir_rb['c1']-1], dim=-1)
    nll_vals   = -lp_full.gather(1, tgt_tensor.unsqueeze(1)).squeeze(1).cpu().numpy()
    if valid_mask is not None:
        vm_flat  = valid_mask.flatten()[wl:wl + out_len]
        nll_vals = nll_vals[vm_flat[:len(nll_vals)]]
    nll = float(nll_vals.mean())
    bpb = nll / math.log(2)

    return dict(bpb=bpb, nll=nll, match_pct=match_pct,
                decoded_bytes=final_gen.tolist(), n_valid=len(target_v))


@torch.no_grad()
def ar_decode_chunk_fb_kv(model, chunks_arr, slot_len: int, slot_count: int,
                          schedule: list[tuple[int, int]], mask_np: np.ndarray,
                          pos: dict, device,
                          valid_mask: np.ndarray | None = None) -> dict:
    """
    KV-cached greedy AR decode for feedback-argmax IR layout. Same semantics
    as ar_decode_chunk_fb, ~L× faster — see ar_decode_chunk_kv for the
    underlying strategy (the fb mask is built the same source-first/causal
    way, so incremental caching is valid here too).

    Each IQ/IR turn's [SLOT(s) + warmup] prefix is processed as one chunk
    against the growing cache, then output tokens are generated one at a
    time, each extending the cache by one position.
    """
    if isinstance(chunks_arr, np.ndarray) and chunks_arr.ndim == 2:
        chunks_list = [chunks_arr[k] for k in range(chunks_arr.shape[0])]
    else:
        chunks_list = list(chunks_arr)

    n_chunks  = len(chunks_list)
    chunk_len = len(chunks_list[0])
    wl        = pos['warmup_len']
    sids      = np.array(_slot_ids(slot_len, slot_count), dtype=np.int64)
    L         = pos['L']
    full_mask = torch.tensor(mask_np, dtype=torch.float32, device=device)

    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos['enc_blocks']):
        tok[b['s0']:b['s1']]   = chunks_list[k]
        tok[b['sl0']:b['sl1']] = sids

    # ── Step 1: process encoding prefix at once ──────────────────────────
    enc_end  = pos['enc_end']
    enc_t    = torch.tensor(tok[:enc_end], dtype=torch.long, device=device)
    enc_mask = full_mask[:enc_end, :enc_end]
    _, kv_cache = model(enc_t, enc_mask, return_kv=True)
    L_cached = enc_end

    def _decode_segment(seg_start, rb):
        """Process [seg_start:c0) prefix against cache, then AR-generate out_len tokens."""
        nonlocal kv_cache, L_cached
        seg_end = rb['c0']
        seg_len = seg_end - seg_start
        seg_t    = torch.tensor(tok[seg_start:seg_end], dtype=torch.long, device=device)
        seg_mask = full_mask[seg_start:seg_end, :L_cached + seg_len]
        seg_logits, seg_kv = model(seg_t, seg_mask, past_kv=kv_cache,
                                   return_kv=True, offset=L_cached)
        kv_cache  = _cat_kv(kv_cache, seg_kv)
        L_cached += seg_len

        tok[rb['c0']] = int(seg_logits[-1].argmax())
        for j in range(1, rb['out_len']):
            prev_pos  = rb['c0'] + j - 1
            prev_t    = torch.tensor([tok[prev_pos]], dtype=torch.long, device=device)
            prev_mask = full_mask[prev_pos:prev_pos+1, :L_cached + 1]
            prev_logits, prev_kv = model(prev_t, prev_mask, past_kv=kv_cache,
                                         return_kv=True, offset=L_cached)
            kv_cache  = _cat_kv(kv_cache, prev_kv)
            L_cached += 1
            tok[rb['c0'] + j] = int(prev_logits[-1].argmax())

    # Process every rec_block in sequence order. 'iq' blocks decode against
    # their own SLOT; 'ir' blocks copy their argmax cue directly out of `tok`
    # at argmax_src_c0 (causally guaranteed to already be decoded — true for
    # both the same-span IQ source in chunk_positions_fb and the byte-sliced /
    # chained source in chunk_positions_fb_winrefine), then decode their own
    # SLOT_A/argmax/SLOT_B/warmup/out region.
    for rb in pos['rec_blocks']:
        span_s, span_e = rb['span']
        gt_span = np.concatenate(chunks_list[span_s:span_e])

        if rb['type'] == 'iq':
            tok[rb['sl0']:rb['sl1']] = sids
            if wl > 0:
                tok[rb['w0']:rb['w1']] = gt_span[:wl].astype(np.int64)
            _decode_segment(rb['sl0'], rb)
        else:  # 'ir'
            tok[rb['sla0']:rb['sla1']] = sids
            src_c0 = rb['argmax_src_c0']
            tok[rb['am0']:rb['am1']] = tok[src_c0:src_c0 + rb['out_len']]
            tok[rb['slb0']:rb['slb1']] = sids
            if wl > 0:
                tok[rb['w0']:rb['w1']] = gt_span[:wl].astype(np.int64)
            _decode_segment(rb['sla0'], rb)

    # ── Result: last rec_block's own output (its own span, not necessarily
    #    the full sequence — for ir_winrefine the last block is a windowed IR) ─
    final_rb = pos['rec_blocks'][-1]
    span_s, span_e = final_rb['span']
    gt_final  = np.concatenate(chunks_list[span_s:span_e])
    out_len   = final_rb['out_len']
    final_gen = tok[final_rb['c0']:final_rb['c1']]
    target    = gt_final[wl:wl + out_len]

    byte_lo = span_s * chunk_len + wl
    byte_hi = span_e * chunk_len
    if valid_mask is not None:
        vm       = valid_mask.flatten()[byte_lo:byte_hi]
        gen_v    = final_gen[:len(vm)][vm]
        target_v = target[:len(vm)][vm]
    else:
        gen_v    = final_gen[:out_len]
        target_v = target[:out_len]

    match_pct = 100.0 * float(np.sum(gen_v == target_v)) / max(len(target_v), 1)

    # Teacher-forced BPB on the final block's own output (single extra full forward pass)
    tok_tf = torch.tensor(tok, dtype=torch.long, device=device)
    tok_tf[final_rb['c0']:final_rb['c1']] = torch.tensor(
        target[:out_len].astype(np.int64), dtype=torch.long, device=device)
    logits_tf  = model(tok_tf, full_mask)
    tgt_tensor = torch.tensor(target[:out_len], dtype=torch.long, device=device)
    lp_full    = F.log_softmax(logits_tf[final_rb['c0']-1:final_rb['c1']-1], dim=-1)
    nll_vals   = -lp_full.gather(1, tgt_tensor.unsqueeze(1)).squeeze(1).cpu().numpy()
    if valid_mask is not None:
        vm_flat  = valid_mask.flatten()[byte_lo:byte_hi]
        nll_vals = nll_vals[vm_flat[:len(nll_vals)]]
    nll = float(nll_vals.mean())
    bpb = nll / math.log(2)

    return dict(bpb=bpb, nll=nll, match_pct=match_pct,
                decoded_bytes=final_gen.tolist(), n_valid=len(target_v))


@torch.no_grad()
def ar_decode_chunk_fb_stitch_kv(model, chunks_arr, slot_len: int, slot_count: int,
                                 mask_np: np.ndarray, pos: dict, device,
                                 valid_mask: np.ndarray | None = None) -> dict:
    """
    KV-cached greedy AR decode for the ir_local (per-window local-refine)
    layout, "prolonged" across all windows: only the very first window's
    warmup (bytes 0:warmup_len of the whole source) is seeded from ground
    truth. Every later byte — every later window's warmup AND output —
    comes from the model's own previously decoded tokens, stitched into one
    global (src_len,) buffer. Windows overlap by stride and warmup_len
    always fits within the overlap, so by the time window i's warmup is
    needed window i-1's own output has already produced those exact bytes.

    Reports match_pct/bpb against the FULL reconstructed source (bytes
    warmup_len:src_len — the part actually generated, not seeded), not just
    the last window's own output. Later windows overwrite earlier ones in
    the overlap region, matching the prolonged-decode generation order.
    """
    if isinstance(chunks_arr, np.ndarray) and chunks_arr.ndim == 2:
        chunks_list = [chunks_arr[k] for k in range(chunks_arr.shape[0])]
    else:
        chunks_list = list(chunks_arr)

    n_chunks  = len(chunks_list)
    chunk_len = len(chunks_list[0])
    src_len   = n_chunks * chunk_len
    wl        = pos['warmup_len']
    sids      = np.array(_slot_ids(slot_len, slot_count), dtype=np.int64)
    L         = pos['L']
    full_mask = torch.tensor(mask_np, dtype=torch.float32, device=device)
    gt_full   = np.concatenate(chunks_list)

    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos['enc_blocks']):
        tok[b['s0']:b['s1']]   = chunks_list[k]
        tok[b['sl0']:b['sl1']] = sids

    # ── Step 1: process encoding prefix at once ──────────────────────────
    enc_end  = pos['enc_end']
    enc_t    = torch.tensor(tok[:enc_end], dtype=torch.long, device=device)
    enc_mask = full_mask[:enc_end, :enc_end]
    _, kv_cache = model(enc_t, enc_mask, return_kv=True)
    L_cached = enc_end

    def _decode_segment(seg_start, rb):
        nonlocal kv_cache, L_cached
        seg_end = rb['c0']
        seg_len = seg_end - seg_start
        seg_t    = torch.tensor(tok[seg_start:seg_end], dtype=torch.long, device=device)
        seg_mask = full_mask[seg_start:seg_end, :L_cached + seg_len]
        seg_logits, seg_kv = model(seg_t, seg_mask, past_kv=kv_cache,
                                   return_kv=True, offset=L_cached)
        kv_cache  = _cat_kv(kv_cache, seg_kv)
        L_cached += seg_len

        tok[rb['c0']] = int(seg_logits[-1].argmax())
        for j in range(1, rb['out_len']):
            prev_pos  = rb['c0'] + j - 1
            prev_t    = torch.tensor([tok[prev_pos]], dtype=torch.long, device=device)
            prev_mask = full_mask[prev_pos:prev_pos+1, :L_cached + 1]
            prev_logits, prev_kv = model(prev_t, prev_mask, past_kv=kv_cache,
                                         return_kv=True, offset=L_cached)
            kv_cache  = _cat_kv(kv_cache, prev_kv)
            L_cached += 1
            tok[rb['c0'] + j] = int(prev_logits[-1].argmax())

    # Global stitched-byte buffer: filled in as each window's blocks decode.
    # -1 = not yet decoded. Later windows overwrite earlier ones in the
    # overlap region (matches the prolonged-decode generation order).
    decoded_bytes = np.full(src_len, -1, dtype=np.int64)

    for rb in pos['rec_blocks']:
        span_s, span_e = rb['span']
        win_byte_lo = span_s * chunk_len

        if rb['type'] == 'iq':
            tok[rb['sl0']:rb['sl1']] = sids
            if wl > 0:
                tok[rb['w0']:rb['w1']] = (gt_full[win_byte_lo:win_byte_lo + wl]
                                          if win_byte_lo == 0 else
                                          decoded_bytes[win_byte_lo:win_byte_lo + wl])
            _decode_segment(rb['sl0'], rb)
        else:  # 'ir'
            tok[rb['sla0']:rb['sla1']] = sids
            src_c0 = rb['argmax_src_c0']
            tok[rb['am0']:rb['am1']] = tok[src_c0:src_c0 + rb['out_len']]
            tok[rb['slb0']:rb['slb1']] = sids
            if wl > 0:
                tok[rb['w0']:rb['w1']] = (gt_full[win_byte_lo:win_byte_lo + wl]
                                          if win_byte_lo == 0 else
                                          decoded_bytes[win_byte_lo:win_byte_lo + wl])
            _decode_segment(rb['sla0'], rb)

        out_len = rb['out_len']
        decoded_bytes[win_byte_lo + wl:win_byte_lo + wl + out_len] = tok[rb['c0']:rb['c1']]
        if wl > 0:
            decoded_bytes[win_byte_lo:win_byte_lo + wl] = tok[rb['w0']:rb['w1']]

    gen_v    = decoded_bytes[wl:]
    target_v = gt_full[wl:]
    if valid_mask is not None:
        vm       = valid_mask.flatten()[:src_len][wl:]
        gen_v    = gen_v[vm]
        target_v = target_v[vm]
    match_pct = 100.0 * float(np.sum(gen_v == target_v)) / max(len(target_v), 1)

    # ── Teacher-forced BPB: one extra full forward pass with each window's
    #    LAST (most-refined) block's output set to ground truth, then read
    #    off per-byte NLL stitched the same way as decoded_bytes (later
    #    windows overwrite earlier ones in the overlap).
    window_blocks: list[list[dict]] = []
    cur_span = None
    for rb in pos['rec_blocks']:
        if rb['span'] != cur_span:
            window_blocks.append([])
            cur_span = rb['span']
        window_blocks[-1].append(rb)

    tok_tf = tok.copy()
    for blocks in window_blocks:
        rb = blocks[-1]
        span_s, span_e = rb['span']
        gt_span = np.concatenate(chunks_list[span_s:span_e])
        out_len = rb['out_len']
        tok_tf[rb['c0']:rb['c1']] = gt_span[wl:wl + out_len].astype(np.int64)

    tok_tf_t  = torch.tensor(tok_tf, dtype=torch.long, device=device)
    logits_tf = model(tok_tf_t, full_mask)
    lp_full   = F.log_softmax(logits_tf, dim=-1)

    nll_buf = np.full(src_len, np.nan, dtype=np.float64)
    for blocks in window_blocks:
        rb = blocks[-1]
        span_s, span_e = rb['span']
        win_byte_lo = span_s * chunk_len
        out_len     = rb['out_len']
        gt_span     = np.concatenate(chunks_list[span_s:span_e])
        tgt         = gt_span[wl:wl + out_len].astype(np.int64)
        tgt_t       = torch.tensor(tgt, dtype=torch.long, device=device)
        nv = -lp_full[rb['c0']-1:rb['c1']-1].gather(1, tgt_t.unsqueeze(1)).squeeze(1).cpu().numpy()
        nll_buf[win_byte_lo + wl:win_byte_lo + wl + out_len] = nv

    if valid_mask is not None:
        vm  = valid_mask.flatten()[:src_len]
        sel = ~np.isnan(nll_buf) & vm
    else:
        sel = ~np.isnan(nll_buf)
    nll = float(np.nanmean(nll_buf[sel])) if sel.any() else float('nan')
    bpb = nll / math.log(2)

    return dict(bpb=bpb, nll=nll, match_pct=match_pct,
                decoded_bytes=decoded_bytes.tolist(), n_valid=len(target_v))


# ---------------------------------------------------------------------------
# Test-set loader
# ---------------------------------------------------------------------------

def load_chunks_padded(path: str, n_chunks: int,
                       chunk_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load file, split by newline, distribute into n_chunks groups, pad to chunk_len.

    Distribution: lines are divided into n_chunks consecutive groups as evenly as
    possible. First (n_lines % n_chunks) groups get ceil(n_lines/n_chunks) lines,
    the rest get floor. Ensures no empty groups when n_lines >= n_chunks.

    Returns:
      chunks     : (n_chunks, chunk_len) int64
      valid_mask : (n_chunks, chunk_len) bool  — True = real byte, False = pad
    """
    raw     = open(path, 'rb').read()
    lines   = [l for l in raw.split(b'\n') if l]
    n_lines = len(lines)

    # Distribute lines into consecutive groups (ceil/floor even split)
    base  = n_lines // n_chunks
    extra = n_lines % n_chunks      # first `extra` groups get one extra line
    groups: list[bytes] = []
    start = 0
    for gi in range(n_chunks):
        count = base + (1 if gi < extra else 0)
        groups.append(b''.join(lines[start:start + count]))
        start += count

    # Build arrays (truncate to chunk_len if group exceeds it)
    chunks     = np.zeros((n_chunks, chunk_len), dtype=np.int64)
    valid_mask = np.zeros((n_chunks, chunk_len), dtype=bool)
    for k, g in enumerate(groups):
        if g:
            b       = np.frombuffer(g[:chunk_len], dtype=np.uint8).astype(np.int64)
            n_real  = min(len(b), chunk_len)
            chunks[k, :n_real]     = b[:n_real]
            valid_mask[k, :n_real] = True

    return chunks, valid_mask


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_chunk(hp: dict, log_base: str = 'logs', device_str: str = 'cpu'):
    device = torch.device(device_str)
    rng    = np.random.default_rng(hp.get('seed', 42))
    torch.manual_seed(hp.get('seed', 42))

    name     = hp.get('name', 'hmn_chunk')
    log_dir  = os.path.join(log_base, name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    log_file   = open(os.path.join(log_dir, 'train.log'),   'a', buffering=1)
    jsonl_file = open(os.path.join(log_dir, 'train.jsonl'), 'a', buffering=1)

    def _log(msg):
        print(msg)
        print(msg, file=log_file)

    def _jlog(d):
        jsonl_file.write(json.dumps(d) + '\n')

    # Model
    hp_model = dict(
        V=hp.get('V', HMN_VOCAB_SIZE),
        d=hp['d'], n_layers=hp['n_layers'], n_heads=hp['n_heads'], d_ff=hp['d_ff'],
        rope=hp.get('rope', True), yarn=hp.get('yarn', True),
        null_kv=hp.get('null_kv', True), compile=hp.get('compile', False),
    )
    model   = build_model(hp_model, device)
    n_params = sum(p.numel() for p in model.parameters())
    _log(f'Model: {n_params:,} params  device={device}')

    if hp.get('_pretrained_ckpt'):
        ckpt = torch.load(hp['_pretrained_ckpt'], map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        _log(f'Loaded pretrained: {hp["_pretrained_ckpt"]}')

    # Optimizer
    lr_max       = hp.get('lr_max', 3e-4)
    wd           = hp.get('wd', 0.001)
    warmup_steps = hp.get('warmup_steps', 500)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=wd)

    # Eval file
    eval_file   = hp.get('eval_file')
    eval_every  = hp.get('eval_every', 5000)
    log_every   = hp.get('log_every', 500)
    use_stablemax = hp.get('stablemax', False)
    log_probs_fn  = (lambda lg: F.log_softmax(lg, dim=-1))

    curriculum = hp.get('curriculum', [])
    assert curriculum, 'hp must have curriculum list'

    global_step = 0
    t_start     = time.time()

    for stage_i, stage in enumerate(curriculum):
        n_chunks   = stage['n_chunks']
        chunk_len  = stage['chunk_len']
        slot_len   = hp.get('slot_len', 2)
        slot_count = hp.get('slot_count', 2)
        ir_turns   = hp.get('ir_turns', 2)
        warmup_len = hp.get('warmup_len', 0)
        noise_p    = hp.get('noise_p', 0.5)
        B          = stage.get('B', 8)
        n_steps    = stage.get('n_steps', 50000)
        use_srs    = stage.get('use_srs', True)
        ls_max     = hp.get('ls_max', 0.0)

        schedule = srs_schedule(n_chunks) if use_srs else [(0, n_chunks)]
        pos      = chunk_positions(n_chunks, chunk_len, slot_len, schedule, ir_turns, warmup_len)
        mask_np  = chunk_mask(pos)
        mask_t   = torch.tensor(mask_np, dtype=torch.float32, device=device)

        _log(f'\n[stage {stage_i}] n_chunks={n_chunks} chunk_len={chunk_len} '
             f'slot_len={slot_len} ir_turns={ir_turns} use_srs={use_srs} '
             f'n_spans={len(schedule)} L={pos["L"]}  B={B}  steps={n_steps}')

        def _lr(local_step):
            if local_step <= warmup_steps:
                return lr_max * local_step / max(warmup_steps, 1)
            return lr_max

        # Load test chunks once per stage (may change if chunk_len changes)
        test_chunks = test_valid_mask = None
        if eval_file:
            try:
                test_chunks, test_valid_mask = load_chunks_padded(
                    eval_file, n_chunks, chunk_len)
            except Exception as e:
                _log(f'  [test eval disabled: {e}]')

        # Val sequences: make_test_sequences split into n_chunks chunks
        val_seg_len = n_chunks * chunk_len
        val_seqs    = make_test_sequences(val_seg_len)
        val_n_seqs  = hp.get('val_n_seqs')
        if val_n_seqs is not None:
            val_seqs = dict(list(val_seqs.items())[:val_n_seqs])

        pbar = tqdm(range(1, n_steps + 1), desc=f'stage{stage_i}', dynamic_ncols=True)
        for local_step in pbar:
            global_step += 1
            lr = _lr(local_step)
            for pg in opt.param_groups:
                pg['lr'] = lr

            model.train()
            opt.zero_grad()

            tok_np = _chunk_make_batch(
                rng, B, n_chunks, chunk_len, slot_len, slot_count,
                schedule, ir_turns, noise_p, pos=pos)
            tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)

            logits = model(tok_t, mask_t)   # (B, L, V)

            # Loss: NTP on all clean recall output positions only (warmup excluded).
            # logits[:,c0-1] = logit at last warmup token (or last SLOT if wl=0)
            # → predicts tok[:,c0] = first output byte.  Correct for both wl=0 and wl>0.
            nlls = []
            for rb in pos['rec_blocks']:
                if not rb['is_clean']:
                    continue
                lp  = log_probs_fn(logits[:, rb['c0']-1:rb['c1']-1])   # (B, out_len, V)
                tgt = tok_t[:, rb['c0']:rb['c1']]                       # (B, out_len)
                nlls.append(_positional_ls_nll(lp, tgt, ls_max).mean())

            loss = torch.stack(nlls).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_f = float(loss.detach())
            pbar.set_postfix(loss=f'{loss_f:.3f}', lr=f'{lr:.1e}', refresh=False)

            if local_step % log_every == 0:
                _jlog(dict(step=global_step, loss=round(loss_f, 5), lr=lr))

            # Eval
            if local_step % eval_every == 0 or local_step == n_steps:
                model.eval()
                elapsed = time.time() - t_start
                h, m = divmod(int(elapsed), 3600)
                m, s = divmod(m, 60)
                _log(f'\n--- stage={stage_i} step={local_step}/{n_steps}'
                     f'  g={global_step}  loss={loss_f:.4f}  {h:02d}:{m:02d}:{s:02d} ---')

                # Val: deterministic sequences
                val_results = []
                for sname, seq in val_seqs.items():
                    chunks_arr = np.array(
                        [seq[k*chunk_len:(k+1)*chunk_len] for k in range(n_chunks)],
                        dtype=np.int64)
                    r = ar_decode_chunk_kv(model, chunks_arr, slot_len, slot_count,
                                           schedule, mask_np, pos, device)
                    val_results.append(r['match_pct'])
                    _log(f'  val/{sname:<15} BPB={r["bpb"]:.3f}  match={r["match_pct"]:.1f}%')
                val_mean = sum(val_results) / len(val_results)
                _log(f'  val/MEAN               match={val_mean:.1f}%')

                # Test: surah (eval-only)
                if test_chunks is not None:
                    r = ar_decode_chunk_kv(model, test_chunks, slot_len, slot_count,
                                           schedule, mask_np, pos, device,
                                           valid_mask=test_valid_mask)
                    _log(f'  test/surah             BPB={r["bpb"]:.3f}'
                         f'  match={r["match_pct"]:.1f}%'
                         f'  valid_bytes={r["n_valid"]}  [test_bpb/test_match]')
                    _jlog(dict(step=global_step, eval=True,
                               val_mean=round(val_mean, 1),
                               test_bpb=round(r['bpb'], 3),
                               test_match=round(r['match_pct'], 1)))

        # Checkpoint
        ckpt_path = os.path.join(ckpt_dir, f'stage{stage_i}_end.pt')
        torch.save(dict(model=model.state_dict(), hp=hp, hp_model=hp_model,
                        stage=stage_i, step=global_step), ckpt_path)
        _log(f'  [ckpt] {ckpt_path}')

    elapsed = time.time() - t_start
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    _log(f'\nDone. {h:02d}:{m:02d}:{s:02d}')
    log_file.close()
    jsonl_file.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     required=True)
    p.add_argument('--device',     default='cpu')
    p.add_argument('--pretrained', default=None)
    p.add_argument('--log-dir',    default='logs')
    args = p.parse_args()

    hp = load_config(args.config)
    if args.pretrained:
        hp['_pretrained_ckpt'] = args.pretrained

    # Dispatch: hp['train_fn'] = 'fb' routes to train_chunk_fb
    if hp.pop('train_fn', None) == 'fb':
        train_chunk_fb(hp, log_base=args.log_dir, device_str=args.device)
    else:
        train_chunk(hp, log_base=args.log_dir, device_str=args.device)


# ---------------------------------------------------------------------------
# Trajectory builder  (traj_mix support for curriculum stages)
# ---------------------------------------------------------------------------

# Default windows rehearsed by ir_winrefine: both halves + the full span,
# weighted toward the novel sub-full cases (see plan: "decide good window to
# carry over and generalize in srs phase" — train on every span size SRS uses).
_WINREFINE_WEIGHTS = {'half': 0.4, 'full': 0.2}


def _build_trajectories(traj_mix_cfg: list[dict], n_chunks: int, chunk_len: int,
                        slot_len: int, warmup_len: int,
                        schedule: list[tuple[int, int]], device) -> list[dict]:
    """
    Expand a traj_mix config (list of {type, weight}) into a flat list of
    concrete trajectories: {name, window, pos, mask_np, mask_t, has_ir, weight}.

    type='iq_windowed' : chunk_positions_fb(with_ir=False) over `schedule` — no feedback.
    type='ir_srs'       : chunk_positions_fb(with_ir=True) over `schedule` — today's full SRS+IR.
    type='ir_winrefine'  : chunk_positions_fb_winrefine, one variant per window in
                           {halves, full}, weight split half/half/full per _WINREFINE_WEIGHTS.
    type='ir_stitch'     : chunk_positions_fb_stitch — overlapping chunk-index
                           windows (entry['windows'], required) each get their
                           own IQ+IR pair; entry['final_iq'] (default True)
                           appends one trailing full-span IQ-only turn to
                           test start-to-finish stitched reconstruction.
    type='ir_local'      : chunk_positions_fb_localrefine — each window in
                           entry['windows'] gets its OWN local IQ + n_refine
                           (entry['n_refine'], default 2) chained argmax-IR
                           turns, reusing the proven hmn_feedback_32_ir unit
                           per-window instead of once globally.
    """
    half = n_chunks // 2
    windows = [(0, half), (half, n_chunks), (0, n_chunks)]
    win_weight = [_WINREFINE_WEIGHTS['half'], _WINREFINE_WEIGHTS['half'], _WINREFINE_WEIGHTS['full']]

    out = []
    for entry in traj_mix_cfg:
        ttype = entry['type']
        w     = entry['weight']
        if ttype == 'iq_windowed':
            pos = chunk_positions_fb(n_chunks, chunk_len, slot_len, schedule, warmup_len, with_ir=False)
            mask_np = chunk_mask_fb(pos)
            out.append(dict(name=ttype, window=None, pos=pos, mask_np=mask_np,
                            has_ir=False, weight=w))
        elif ttype == 'ir_srs':
            pos = chunk_positions_fb(n_chunks, chunk_len, slot_len, schedule, warmup_len, with_ir=True)
            mask_np = chunk_mask_fb(pos)
            out.append(dict(name=ttype, window=None, pos=pos, mask_np=mask_np,
                            has_ir=True, weight=w))
        elif ttype == 'ir_winrefine':
            for window, ww in zip(windows, win_weight):
                pos = chunk_positions_fb_winrefine(n_chunks, chunk_len, slot_len,
                                                   warmup_len, window, n_refine=2)
                mask_np = chunk_mask_fb(pos)
                out.append(dict(name=ttype, window=window, pos=pos, mask_np=mask_np,
                                has_ir=True, weight=w * ww))
        elif ttype == 'ir_stitch':
            stitch_windows = entry['windows']
            final_iq = entry.get('final_iq', True)
            pos = chunk_positions_fb_stitch(n_chunks, chunk_len, slot_len,
                                            warmup_len, stitch_windows, final_iq=final_iq)
            mask_np = chunk_mask_fb(pos)
            out.append(dict(name=ttype, window=None, pos=pos, mask_np=mask_np,
                            has_ir=True, weight=w))
        elif ttype == 'ir_local':
            local_windows = entry['windows']
            n_refine = entry.get('n_refine', 2)
            pos = chunk_positions_fb_localrefine(n_chunks, chunk_len, slot_len,
                                                 warmup_len, local_windows, n_refine=n_refine)
            mask_np = chunk_mask_fb(pos)
            out.append(dict(name=ttype, window=None, pos=pos, mask_np=mask_np,
                            has_ir=(n_refine > 0), weight=w))
        else:
            raise ValueError(f'unknown traj_mix type: {ttype!r}')

    for t in out:
        t['mask_t'] = torch.tensor(t['mask_np'], dtype=torch.float32, device=device)
    return out


# ---------------------------------------------------------------------------
# Feedback training loop  (depth-2 SRS + feedback argmax IR)
# ---------------------------------------------------------------------------

def train_chunk_fb(hp: dict, log_base: str = 'logs', device_str: str = 'cpu'):
    """
    Training with feedback-argmax IR turns.

    Two forward passes per step:
      Pass 1 (no_grad): compute IQ argmax for each span.
      Pass 2 (grad):    fill argmax into IR turns, compute loss on IR out_1.

    hp extras: use_actual_argmax=True (default) to use pass-1 model output;
               False falls back to teacher-forced GT (single pass).
    """
    device = torch.device(device_str)
    rng    = np.random.default_rng(hp.get('seed', 42))
    torch.manual_seed(hp.get('seed', 42))

    name     = hp.get('name', 'hmn_chunk_fb')
    log_dir  = os.path.join(log_base, name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_file   = open(os.path.join(log_dir, 'train.log'),   'a', buffering=1)
    jsonl_file = open(os.path.join(log_dir, 'train.jsonl'), 'a', buffering=1)

    def _log(msg): print(msg); print(msg, file=log_file)
    def _jlog(d):  jsonl_file.write(json.dumps(d) + '\n')

    hp_model = dict(V=hp.get('V', HMN_VOCAB_SIZE),
                    d=hp['d'], n_layers=hp['n_layers'],
                    n_heads=hp['n_heads'], d_ff=hp['d_ff'],
                    rope=hp.get('rope', True), yarn=hp.get('yarn', True),
                    null_kv=hp.get('null_kv', True), compile=hp.get('compile', False),
                    chunk_attn=hp.get('chunk_attn', 0))
    model    = build_model(hp_model, device)
    n_params = sum(p.numel() for p in model.parameters())
    _log(f'Model: {n_params:,} params  device={device}  chunk_attn={hp_model["chunk_attn"]}')

    if hp.get('_pretrained_ckpt'):
        ckpt = torch.load(hp['_pretrained_ckpt'], map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        _log(f'Loaded: {hp["_pretrained_ckpt"]}')

    lr_max        = hp.get('lr_max', 3e-4)
    wd            = hp.get('wd', 0.001)
    warmup_steps  = hp.get('warmup_steps', 500)
    use_actual_am = hp.get('use_actual_argmax', True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=wd)

    eval_file  = hp.get('eval_file', None)
    eval_every = hp.get('eval_every', 5000)
    log_every  = hp.get('log_every', 500)
    curriculum = hp.get('curriculum', [])
    assert curriculum

    global_step = 0
    t_start     = time.time()

    for stage_i, stage in enumerate(curriculum):
        n_chunks   = stage['n_chunks']
        chunk_len  = stage['chunk_len']
        slot_len   = hp.get('slot_len', 1)
        slot_count = hp.get('slot_count', 2)
        warmup_len = hp.get('warmup_len', 8)
        B          = stage.get('B', 4)
        n_steps    = stage.get('n_steps', 50000)
        depth      = stage.get('depth', 2)
        ls_max     = hp.get('ls_max', 0.0)
        stage_eval_every = stage.get('eval_every', eval_every)

        schedule = srs_schedule_depth2(n_chunks) if depth == 2 else srs_schedule(n_chunks)

        # ── Trajectory mix: each entry is (name, window_or_None, pos, mask_np, weight) ──
        traj_mix_cfg = stage.get('traj_mix')
        if traj_mix_cfg is None:
            traj_mix_cfg = [dict(type='ir_srs', weight=1.0)]
        trajectories = _build_trajectories(traj_mix_cfg, n_chunks, chunk_len,
                                           slot_len, warmup_len, schedule, device)
        traj_weights = np.array([t['weight'] for t in trajectories], dtype=np.float64)
        traj_weights = traj_weights / traj_weights.sum()

        eval_traj_name = stage.get('eval_traj', traj_mix_cfg[0]['type'])
        eval_trajs = [t for t in trajectories if t['name'] == eval_traj_name]

        # Use the dominant (highest-weight) trajectory's L just for logging.
        primary = trajectories[int(np.argmax(traj_weights))]
        _log(f'\n[stage {stage_i}] n_chunks={n_chunks} chunk_len={chunk_len} '
             f'slot={slot_len} wl={warmup_len} depth={depth} '
             f'traj_mix={[(t["name"], t.get("window"), round(w,2)) for t, w in zip(trajectories, traj_weights)]} '
             f'eval_traj={eval_traj_name} L~{primary["pos"]["L"]}  B={B}  steps={n_steps}  '
             f'actual_argmax={use_actual_am}')

        def _lr(s): return lr_max * s / max(warmup_steps, 1) if s <= warmup_steps else lr_max

        test_chunks = test_vm = None
        if eval_file:
            try:
                test_chunks, test_vm = load_chunks_padded(eval_file, n_chunks, chunk_len)
            except Exception as e:
                _log(f'  [test eval disabled: {e}]')

        val_seg_len = n_chunks * chunk_len
        val_seqs    = make_test_sequences(val_seg_len)
        val_n_seqs  = hp.get('val_n_seqs')
        if val_n_seqs is not None:
            val_seqs = dict(list(val_seqs.items())[:val_n_seqs])

        stage_best_val = -1.0

        pbar = tqdm(range(1, n_steps + 1), desc=f'stage{stage_i}', dynamic_ncols=True)
        for local_step in pbar:
            global_step += 1
            lr = _lr(local_step)
            for pg in opt.param_groups: pg['lr'] = lr

            model.train(); opt.zero_grad()

            traj = trajectories[rng.choice(len(trajectories), p=traj_weights)]
            pos, mask_t, has_ir = traj['pos'], traj['mask_t'], traj['has_ir']

            tok_np = _chunk_make_batch_fb(rng, B, n_chunks, chunk_len,
                                          slot_len, slot_count, schedule, pos)
            tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)

            if use_actual_am and has_ir:
                # Pass 1: get argmax cues (no grad)
                with torch.no_grad():
                    logits_1 = model(tok_t, mask_t)
                tok_np = _fill_argmax_fb(tok_np, logits_1, pos)
                tok_t  = torch.tensor(tok_np, device=device, dtype=torch.long)

            # Pass 2 (or only pass if teacher-forced/no-IR): compute loss on clean blocks
            logits = model(tok_t, mask_t)
            nlls = []
            for rb in pos['rec_blocks']:
                if not rb['is_clean']: continue     # IR turns, or IQ turns when with_ir=False
                lp  = F.log_softmax(logits[:, rb['c0']-1:rb['c1']-1], dim=-1)
                tgt = tok_t[:, rb['c0']:rb['c1']]
                nlls.append(_positional_ls_nll(lp, tgt, ls_max).mean())
            loss = torch.stack(nlls).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_f = float(loss.detach())
            pbar.set_postfix(loss=f'{loss_f:.3f}', lr=f'{lr:.1e}', traj=traj['name'], refresh=False)
            if local_step % log_every == 0:
                _jlog(dict(step=global_step, loss=round(loss_f, 5), lr=lr, traj=traj['name']))

            if local_step % stage_eval_every == 0 or local_step == n_steps:
                model.eval()
                elapsed = time.time() - t_start
                h, m = divmod(int(elapsed), 3600); m, s = divmod(m, 60)
                _log(f'\n--- stage={stage_i} step={local_step}/{n_steps}'
                     f'  g={global_step}  loss={loss_f:.4f}  {h:02d}:{m:02d}:{s:02d} ---')

                # Evaluate every variant of the chosen eval_traj (e.g. each
                # window for ir_winrefine), report mean across them.
                eval_val_means = []
                for et in eval_trajs:
                    epos, emask = et['pos'], et['mask_np']
                    tag = et['name'] + (f'[{et["window"]}]' if et.get('window') else '')
                    # ir_local with >1 window: use the full-sequence stitched
                    # "prolonged AR" decode (decode 0..src_len, only the very
                    # first window's warmup seeded from GT). Everything else
                    # (single-window ir_local, ir_srs, ir_winrefine, ...)
                    # keeps the original last-block-only decode.
                    n_local_windows = (len(set(rb['span'] for rb in epos['rec_blocks']))
                                       if et['name'] == 'ir_local' else 1)
                    use_stitch = et['name'] == 'ir_local' and n_local_windows > 1
                    val_results = []
                    for sname, seq in val_seqs.items():
                        chunks_arr = np.array(
                            [seq[k*chunk_len:(k+1)*chunk_len] for k in range(n_chunks)], np.int64)
                        if use_stitch:
                            r = ar_decode_chunk_fb_stitch_kv(model, chunks_arr, slot_len, slot_count,
                                                             emask, epos, device)
                        else:
                            r = ar_decode_chunk_fb_kv(model, chunks_arr, slot_len, slot_count,
                                                     schedule, emask, epos, device)
                        val_results.append(r['match_pct'])
                        _log(f'  val/{tag}/{sname:<15} BPB={r["bpb"]:.3f}  match={r["match_pct"]:.1f}%')
                    vmean = sum(val_results) / len(val_results)
                    eval_val_means.append(vmean)
                    _log(f'  val/{tag}/MEAN               match={vmean:.1f}%')
                val_mean = sum(eval_val_means) / len(eval_val_means)
                _log(f'  val/MEAN (all {eval_traj_name} variants)  match={val_mean:.1f}%')

                if val_mean > stage_best_val:
                    stage_best_val = val_mean
                    best_path = os.path.join(ckpt_dir, f'stage{stage_i}_best.pt')
                    torch.save(dict(model=model.state_dict(), hp=hp, hp_model=hp_model,
                                    stage=stage_i, step=global_step, val_mean=val_mean), best_path)
                    _log(f'  [new best] stage={stage_i} step={local_step} val_mean={val_mean:.1f}% -> {best_path}')

                if test_chunks is not None:
                    et = eval_trajs[0]
                    n_local_windows = (len(set(rb['span'] for rb in et['pos']['rec_blocks']))
                                       if et['name'] == 'ir_local' else 1)
                    if et['name'] == 'ir_local' and n_local_windows > 1:
                        r = ar_decode_chunk_fb_stitch_kv(model, test_chunks, slot_len, slot_count,
                                                         et['mask_np'], et['pos'], device, valid_mask=test_vm)
                    else:
                        r = ar_decode_chunk_fb_kv(model, test_chunks, slot_len, slot_count,
                                                  schedule, et['mask_np'], et['pos'], device, valid_mask=test_vm)
                    _log(f'  test/surah             BPB={r["bpb"]:.3f}'
                         f'  match={r["match_pct"]:.1f}%  valid_bytes={r["n_valid"]}')
                    _jlog(dict(step=global_step, eval=True,
                               val_mean=round(val_mean, 1),
                               test_bpb=round(r['bpb'], 3),
                               test_match=round(r['match_pct'], 1)))

                # Free MPS/CUDA cache after eval to prevent memory fragmentation
                if hasattr(torch, 'mps') and device.type == 'mps':
                    torch.mps.empty_cache()
                elif device.type == 'cuda':
                    torch.cuda.empty_cache()

        ckpt_path = os.path.join(ckpt_dir, f'stage{stage_i}_end.pt')
        torch.save(dict(model=model.state_dict(), hp=hp, hp_model=hp_model,
                        stage=stage_i, step=global_step), ckpt_path)
        _log(f'  [ckpt] {ckpt_path}')

    elapsed = time.time() - t_start
    h, m = divmod(int(elapsed), 3600); m, s = divmod(m, 60)
    _log(f'\nDone. {h:02d}:{m:02d}:{s:02d}')
    log_file.close(); jsonl_file.close()


if __name__ == '__main__':
    main()
