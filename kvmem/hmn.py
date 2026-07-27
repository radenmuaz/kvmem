"""
kvmem/hmn.py — single-file HashMemNet (HMN) training stack: vocab/tag
constants, position/mask builders, batch filling + AR-decode, model
architecture, and the training loop.

See docs/HMN_RECIPE.md for a from-scratch walkthrough and docs/HISTORY.md
for design rationale and prior-mechanism history.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt
from tqdm import tqdm

from kvmem.structured_data import generate_structured_chunks


# =============================================================================
# Vocab / tags
#
# Layout: 256 data bytes, then 6 fixed chat-tag ids (256-261), then STATE
# placeholders (262+) occupying the tail so vocab growth (state_vocab_size)
# is always a pure tail-append — see docs/HISTORY.md for the pre-reorder
# scheme this replaced.
# =============================================================================

DATA_LO = 0x20   # legacy: data restricted to [0x20, 0xFF]

# Generic tag vocabulary, reused identically at every chain step/round — no
# per-step/per-round variants (turn identity comes from position only).
HMN_SRC_OPEN       = 256   # <src>
HMN_SRC_CLOSE      = 257   # </src>
HMN_QUERY_OPEN     = 258   # <query>
HMN_QUERY_CLOSE    = 259   # </query>
HMN_RESPONSE_OPEN  = 260   # <response>
HMN_RESPONSE_CLOSE = 261   # </response>

# First STATE placeholder id — everything up to hp['V'] is STATE alphabet.
HMN_STATE_0 = 262

# 256 bytes + 6 chat tags + 12 reserved STATE ids.
HMN_TAG_VOCAB_SIZE = 274


def _cyclic_state_ids(state_len: int, state_vocab_size: int = 2, family: int = 0) -> list[int]:
    # `family` selects which same-size block of the STATE tail to draw from
    # (family 0 = regular STATE, family 1 = feedback_state's own alphabet).
    assert state_vocab_size >= 1
    base = HMN_STATE_0 + family * state_vocab_size
    return [base + (i % state_vocab_size) for i in range(state_len)]


# family index for a refine round's feedback_state alphabet (see _cyclic_state_ids)
HMN_FEEDBACK_STATE_FAMILY = 1


# =============================================================================
# Position/mask-field builders
# =============================================================================

def chunk_positions_iq_global_rw_tagged(n_chunks: int, chunk_len: int, state_len: int,
                                        warmup_len: int, window_chunks: int = 2,
                                        warmup_x_fixed: int | None = None,
                                        warmup_x_dist: str = 'uniform',
                                        n_refine: int = 0) -> dict:
    """
    Returns dict(pos_content=..., pos_mask=..., tags=[(position, token_id), ...], L=...).

    Sequence (n_refine=0): per chunk k: <src> chunk_k </src> STATE, then
    round 0 (initial): STATE <query> warmup </query> <response> out </response>.
    Each n_refine>0 round appends: state <response> argmax </response>
    feedback_state <query> warmup </query> <response> out </response>.
    """
    enc_blocks_c: list[dict] = []
    enc_blocks_m: list[dict] = []
    tags: list[tuple[int, int]] = []
    offset = 0

    for _ in range(n_chunks):
        tags.append((offset, HMN_SRC_OPEN)); offset += 1
        s0 = offset; s1 = s0 + chunk_len; offset = s1
        tags.append((offset, HMN_SRC_CLOSE)); offset += 1
        sl0 = offset; sl1 = sl0 + state_len; offset = sl1

        enc_blocks_c.append(dict(s0=s0, s1=s1, sl0=sl0, sl1=sl1))
        enc_blocks_m.append(dict(s0=s0 - 1, s1=s1 + 1, sl0=sl0, sl1=sl1))

    enc_end = offset

    out_len = window_chunks * chunk_len - warmup_len

    query_open, query_close = HMN_QUERY_OPEN, HMN_QUERY_CLOSE

    def _emit_round(round_idx: int):
        # round_idx == 0: STATE + <query>/<response>; round_idx > 0: state +
        # argmax + feedback_state + <query>/<response> (refine).
        nonlocal offset
        if round_idx == 0:
            sl0 = offset; sl1 = sl0 + state_len; offset = sl1
            tags.append((offset, query_open)); offset += 1
            w0 = offset; w1 = w0 + warmup_len; offset = w1
            tags.append((offset, query_close)); offset += 1
            tags.append((offset, HMN_RESPONSE_OPEN)); offset += 1
            c0 = offset; c1 = c0 + out_len; offset = c1
            tags.append((offset, HMN_RESPONSE_CLOSE)); offset += 1
            return (dict(sl0=sl0, sl1=sl1, w0=w0, w1=w1, c0=c0, c1=c1),
                    dict(sl0=sl0, sl1=sl1, w0=w0 - 1, w1=w1 + 1, c0=c0 - 1, c1=c1 + 1))
        else:
            sla0 = offset; sla1 = sla0 + state_len; offset = sla1
            tags.append((offset, HMN_RESPONSE_OPEN)); offset += 1
            am0 = offset; am1 = am0 + out_len; offset = am1
            tags.append((offset, HMN_RESPONSE_CLOSE)); offset += 1
            slb0 = offset; slb1 = slb0 + state_len; offset = slb1
            tags.append((offset, query_open)); offset += 1
            w0 = offset; w1 = w0 + warmup_len; offset = w1
            tags.append((offset, query_close)); offset += 1
            tags.append((offset, HMN_RESPONSE_OPEN)); offset += 1
            c0 = offset; c1 = c0 + out_len; offset = c1
            tags.append((offset, HMN_RESPONSE_CLOSE)); offset += 1
            c_fields = dict(sla0=sla0, sla1=sla1, am0=am0, am1=am1, slb0=slb0, slb1=slb1,
                            w0=w0, w1=w1, c0=c0, c1=c1)
            m_fields = dict(sla0=sla0, sla1=sla1, am0=am0 - 1, am1=am1 + 1,
                            slb0=slb0, slb1=slb1, w0=w0 - 1, w1=w1 + 1,
                            c0=c0 - 1, c1=c1 + 1)
            return c_fields, m_fields

    src_len = n_chunks * chunk_len
    x_max = src_len - warmup_len - out_len
    n_windows = n_chunks - window_chunks + 1
    eval_offsets = [i * chunk_len for i in range(n_windows)]
    train_range = (warmup_x_fixed, warmup_x_fixed) if warmup_x_fixed is not None else (0, x_max)
    _dist = 'fixed' if warmup_x_fixed is not None else warmup_x_dist

    initial_c, initial_m = _emit_round(0)
    rw_extra = dict(warmup_train_range=train_range, warmup_x_dist=_dist,
                    warmup_valid_offsets=eval_offsets, window_chunks=window_chunks)
    rec_blocks_c = [dict(type='initial', span=(0, n_chunks), span_len=src_len,
                         out_len=out_len, is_clean=(n_refine == 0), **initial_c, **rw_extra)]
    rec_blocks_m = [dict(type='initial', **initial_m)]

    prev_c0_c = initial_c['c0']
    for _ in range(n_refine):
        refine_c, refine_m = _emit_round(1)
        rec_blocks_c.append(dict(type='refine', span=(0, n_chunks), span_len=src_len,
                                 out_len=out_len, is_clean=True,
                                 argmax_src_c0=prev_c0_c, **refine_c, **rw_extra))
        rec_blocks_m.append(dict(type='refine', **refine_m))
        prev_c0_c = refine_c['c0']

    L = offset

    pos_content = dict(enc_blocks=enc_blocks_c, rec_blocks=rec_blocks_c,
                       enc_end=enc_end, warmup_len=warmup_len, L=L)
    pos_mask = dict(enc_blocks=enc_blocks_m, rec_blocks=rec_blocks_m,
                    enc_end=enc_end, warmup_len=warmup_len, L=L)

    return dict(pos_content=pos_content, pos_mask=pos_mask, tags=tags, L=L)




def chunk_positions_hop(n_chunks: int, chunk_len: int, state_len: int,
                         warmup_len: int, chain_steps: list[tuple[int, int]],
                         n_refine: int = 2, state_vocab_size: int = 2) -> dict:
    """
    Chain steps threaded in sequence, each with its own local round-0(+refine)
    block. Chain step i's own round-0 STATE region doubles as both its own
    recall register and the relay source the next chain step reads via a
    genuine attention permission (see chunk_mask_fb_hop), not a forced copy.
    """
    enc_blocks_c: list[dict] = []
    enc_blocks_m: list[dict] = []
    tags: list[tuple[int, int]] = []
    offset = 0

    for _ in range(n_chunks):
        tags.append((offset, HMN_SRC_OPEN)); offset += 1
        s0 = offset; s1 = s0 + chunk_len; offset = s1
        tags.append((offset, HMN_SRC_CLOSE)); offset += 1
        sl0 = offset; sl1 = sl0 + state_len; offset = sl1

        enc_blocks_c.append(dict(s0=s0, s1=s1, sl0=sl0, sl1=sl1))
        enc_blocks_m.append(dict(s0=s0 - 1, s1=s1 + 1, sl0=sl0, sl1=sl1))

    enc_end = offset
    rec_blocks_c: list[dict] = []
    rec_blocks_m: list[dict] = []

    query_open, query_close = HMN_QUERY_OPEN, HMN_QUERY_CLOSE

    for chain_step_i, span in enumerate(chain_steps):
        span_s, span_e = span
        span_len = (span_e - span_s) * chunk_len
        out_len  = span_len - warmup_len

        def _emit_round(round_idx: int):
            nonlocal offset
            if round_idx == 0:
                sl0 = offset; sl1 = sl0 + state_len; offset = sl1
                tags.append((offset, query_open)); offset += 1
                w0 = offset; w1 = w0 + warmup_len; offset = w1
                tags.append((offset, query_close)); offset += 1
                tags.append((offset, HMN_RESPONSE_OPEN)); offset += 1
                c0 = offset; c1 = c0 + out_len; offset = c1
                tags.append((offset, HMN_RESPONSE_CLOSE)); offset += 1
                c_fields = dict(sl0=sl0, sl1=sl1, w0=w0, w1=w1, c0=c0, c1=c1)
                m_fields = dict(sl0=sl0, sl1=sl1, w0=w0 - 1, w1=w1 + 1, c0=c0 - 1, c1=c1 + 1)
                return c_fields, m_fields
            else:
                sla0 = offset; sla1 = sla0 + state_len; offset = sla1
                tags.append((offset, HMN_RESPONSE_OPEN)); offset += 1
                am0 = offset; am1 = am0 + out_len; offset = am1
                tags.append((offset, HMN_RESPONSE_CLOSE)); offset += 1
                slb0 = offset; slb1 = slb0 + state_len; offset = slb1
                tags.append((offset, query_open)); offset += 1
                w0 = offset; w1 = w0 + warmup_len; offset = w1
                tags.append((offset, query_close)); offset += 1
                tags.append((offset, HMN_RESPONSE_OPEN)); offset += 1
                c0 = offset; c1 = c0 + out_len; offset = c1
                tags.append((offset, HMN_RESPONSE_CLOSE)); offset += 1
                c_fields = dict(sla0=sla0, sla1=sla1, am0=am0, am1=am1, slb0=slb0, slb1=slb1,
                                w0=w0, w1=w1, c0=c0, c1=c1)
                m_fields = dict(sla0=sla0, sla1=sla1, am0=am0 - 1, am1=am1 + 1,
                                slb0=slb0, slb1=slb1, w0=w0 - 1, w1=w1 + 1,
                                c0=c0 - 1, c1=c1 + 1)
                return c_fields, m_fields

        initial_c, initial_m = _emit_round(0)
        rec_blocks_c.append(dict(type='initial', span=span, span_len=span_len,
                                 out_len=out_len, is_clean=(n_refine == 0),
                                 warmup_train_range=(0, 0), warmup_x_dist='fixed', **initial_c))
        rec_blocks_m.append(dict(type='initial', span=span, **initial_m))

        prev_c0_c = initial_c['c0']
        for _ in range(n_refine):
            refine_c, refine_m = _emit_round(1)
            rec_blocks_c.append(dict(type='refine', span=span, span_len=span_len,
                                     out_len=out_len, is_clean=True,
                                     argmax_src_c0=prev_c0_c, **refine_c))
            rec_blocks_m.append(dict(type='refine', span=span, **refine_m))
            prev_c0_c = refine_c['c0']

    L = offset

    pos_content = dict(enc_blocks=enc_blocks_c, rec_blocks=rec_blocks_c,
                       enc_end=enc_end, warmup_len=warmup_len, L=L)
    pos_mask = dict(enc_blocks=enc_blocks_m, rec_blocks=rec_blocks_m,
                    enc_end=enc_end, warmup_len=warmup_len, L=L)

    return dict(pos_content=pos_content, pos_mask=pos_mask, tags=tags, L=L)


def chunk_positions_traj(chunk_len: int, state_len: int, warmup_len: int,
                         operations: list[tuple], n_refine: int = 0,
                         state_vocab_size: int = 2) -> dict:
    """
    Generalizes chunk_positions_hop to arbitrary interleaved encode/query
    operation sequences — every named trajectory pattern (batch, stream,
    interleave-delayed, repeat-query, ...) is just a different `operations`
    list fed to this same function.

    operations: list of ops, each one of:
      ('E', chunk_idx)        — ingest chunk_idx's raw bytes only (<src>...</src>).
                                Must be immediately followed by ('S', None).
      ('S', None)              — emit one STATE region. Claims the immediately
                                preceding unclaimed 'E' (that chunk's own
                                encoding-STATE) if there is one; otherwise a
                                bare relay-only no-op hop (blocked from all raw
                                chunks, no local recall target).
      ('Q', (span_s, span_e))  — query/recall chunks [span_s, span_e). Every
                                chunk in the span must already be encoded
                                (causal requirement, asserted below).

    Relay: same single-hop STATE-to-STATE attention permission as
    chunk_positions_hop (see chunk_mask_fb_traj), grouped by op_idx instead of
    chain-step span since the same span can recur (repeat-query).

    enc_blocks here is a dict keyed by chunk_idx (not emission order), since
    'E'/'S' pairs can be interspersed with 'Q' ops in any order.
    """
    enc_blocks_c: dict[int, dict] = {}
    enc_blocks_m: dict[int, dict] = {}
    tags: list[tuple[int, int]] = []
    offset = 0

    query_open, query_close = HMN_QUERY_OPEN, HMN_QUERY_CLOSE

    rec_blocks_c: list[dict] = []
    rec_blocks_m: list[dict] = []
    op_count = 0  # counts 'Q' ops AND bare 'S' ops — both produce a relay-eligible STATE

    pending_chunk_idx: int | None = None  # unclaimed 'E' waiting for its 'S'

    for op, arg in operations:
        if op == 'E':
            chunk_idx = arg
            assert chunk_idx not in enc_blocks_c, f'chunk {chunk_idx} encoded twice'
            assert pending_chunk_idx is None, \
                f"chunk {pending_chunk_idx}'s 'E' was never followed by 'S' before chunk {chunk_idx}'s 'E'"
            tags.append((offset, HMN_SRC_OPEN)); offset += 1
            s0 = offset; s1 = s0 + chunk_len; offset = s1
            tags.append((offset, HMN_SRC_CLOSE)); offset += 1
            enc_blocks_c[chunk_idx] = dict(s0=s0, s1=s1)  # sl0/sl1 filled in when 'S' claims it, below
            enc_blocks_m[chunk_idx] = dict(s0=s0 - 1, s1=s1 + 1)
            pending_chunk_idx = chunk_idx

        elif op == 'S':
            sl0 = offset; sl1 = sl0 + state_len; offset = sl1
            if pending_chunk_idx is not None:
                # Claims the immediately-preceding unclaimed 'E' — this IS
                # that chunk's own encoding-STATE (encoding isolation role),
                # NOT part of the single-hop relay chain (same as the shared
                # encoding pass always was — see chunk_mask_fb_traj's
                # encoding-isolation handling, unchanged).
                enc_blocks_c[pending_chunk_idx]['sl0'] = sl0
                enc_blocks_c[pending_chunk_idx]['sl1'] = sl1
                enc_blocks_m[pending_chunk_idx]['sl0'] = sl0
                enc_blocks_m[pending_chunk_idx]['sl1'] = sl1
                pending_chunk_idx = None
            else:
                # Bare 'S' — no immediately-preceding unclaimed 'E' — this is
                # a relay-only no-op hop (formerly a separate 'N' op type).
                # Same relay-read permission as a 'Q' op's STATE row (see
                # chunk_mask_fb_traj) but no local recall bottleneck rules
                # (nothing to bound a warmup/response region around).
                op_idx = op_count
                op_count += 1
                rec_blocks_c.append(dict(type='noop', span=None, is_clean=False,
                                         op_idx=op_idx, sl0=sl0, sl1=sl1))
                rec_blocks_m.append(dict(type='noop', span=None, op_idx=op_idx, sl0=sl0, sl1=sl1))

        else:  # 'Q'
            span_s, span_e = arg
            for k in range(span_s, span_e):
                assert k in enc_blocks_c, \
                    f'query span {arg} references chunk {k} which has not been encoded yet — ' \
                    f'causal violation, fix the operations list'
            span_len = (span_e - span_s) * chunk_len
            out_len = span_len - warmup_len
            op_idx = op_count
            op_count += 1

            def _emit_round(round_idx: int):
                nonlocal offset
                if round_idx == 0:
                    sl0 = offset; sl1 = sl0 + state_len; offset = sl1
                    tags.append((offset, query_open)); offset += 1
                    w0 = offset; w1 = w0 + warmup_len; offset = w1
                    tags.append((offset, query_close)); offset += 1
                    tags.append((offset, HMN_RESPONSE_OPEN)); offset += 1
                    c0 = offset; c1 = c0 + out_len; offset = c1
                    tags.append((offset, HMN_RESPONSE_CLOSE)); offset += 1
                    c_fields = dict(sl0=sl0, sl1=sl1, w0=w0, w1=w1, c0=c0, c1=c1)
                    m_fields = dict(sl0=sl0, sl1=sl1, w0=w0 - 1, w1=w1 + 1, c0=c0 - 1, c1=c1 + 1)
                    return c_fields, m_fields
                else:
                    sla0 = offset; sla1 = sla0 + state_len; offset = sla1
                    tags.append((offset, HMN_RESPONSE_OPEN)); offset += 1
                    am0 = offset; am1 = am0 + out_len; offset = am1
                    tags.append((offset, HMN_RESPONSE_CLOSE)); offset += 1
                    slb0 = offset; slb1 = slb0 + state_len; offset = slb1
                    tags.append((offset, query_open)); offset += 1
                    w0 = offset; w1 = w0 + warmup_len; offset = w1
                    tags.append((offset, query_close)); offset += 1
                    tags.append((offset, HMN_RESPONSE_OPEN)); offset += 1
                    c0 = offset; c1 = c0 + out_len; offset = c1
                    tags.append((offset, HMN_RESPONSE_CLOSE)); offset += 1
                    c_fields = dict(sla0=sla0, sla1=sla1, am0=am0, am1=am1, slb0=slb0, slb1=slb1,
                                    w0=w0, w1=w1, c0=c0, c1=c1)
                    m_fields = dict(sla0=sla0, sla1=sla1, am0=am0 - 1, am1=am1 + 1,
                                    slb0=slb0, slb1=slb1, w0=w0 - 1, w1=w1 + 1,
                                    c0=c0 - 1, c1=c1 + 1)
                    return c_fields, m_fields

            initial_c, initial_m = _emit_round(0)
            rec_blocks_c.append(dict(type='initial', span=(span_s, span_e), span_len=span_len,
                                     out_len=out_len, is_clean=(n_refine == 0), op_idx=op_idx,
                                     warmup_train_range=(0, 0), warmup_x_dist='fixed', **initial_c))
            rec_blocks_m.append(dict(type='initial', span=(span_s, span_e), op_idx=op_idx, **initial_m))

            prev_c0_c = initial_c['c0']
            for _ in range(n_refine):
                refine_c, refine_m = _emit_round(1)
                rec_blocks_c.append(dict(type='refine', span=(span_s, span_e), span_len=span_len,
                                         out_len=out_len, is_clean=True, op_idx=op_idx,
                                         argmax_src_c0=prev_c0_c, **refine_c))
                rec_blocks_m.append(dict(type='refine', span=(span_s, span_e), op_idx=op_idx, **refine_m))
                prev_c0_c = refine_c['c0']

    assert pending_chunk_idx is None, \
        f"chunk {pending_chunk_idx}'s 'E' was never followed by 'S' — every 'E' needs an 'S' right after it"

    L = offset
    enc_end = max((b['s1'] for b in enc_blocks_c.values()), default=0)  # informational only, not a hard boundary here

    # Sort by chunk_idx explicitly, not dict insertion order — consumers assume
    # enc_blocks[k] is chunk k's block by list position.
    chunk_idx_order = sorted(enc_blocks_c.keys())
    enc_blocks_c_list = [enc_blocks_c[k] for k in chunk_idx_order]
    enc_blocks_m_list = [enc_blocks_m[k] for k in chunk_idx_order]

    pos_content = dict(enc_blocks=enc_blocks_c_list, rec_blocks=rec_blocks_c,
                       enc_end=enc_end, warmup_len=warmup_len, L=L)
    pos_mask = dict(enc_blocks=enc_blocks_m_list, rec_blocks=rec_blocks_m,
                    enc_end=enc_end, warmup_len=warmup_len, L=L)

    return dict(pos_content=pos_content, pos_mask=pos_mask, tags=tags, L=L)


def chunk_positions_stitch(chunk_len: int, n_chunks: int, state_len: int,
                           warmup_len: int, src_stride: int,
                           state_vocab_size: int = 2) -> dict:
    """
    Byte-precise continuous-stitch query chain, decoupled from chunk_len:
    encode n_chunks*chunk_len bytes as usual (E/S per chunk), then a chain
    of `initial`-type rec_blocks whose warmup/response windows advance by
    exactly `src_stride` bytes each — query i's warmup is source bytes
    [i*src_stride, i*src_stride+warmup_len), response is
    [i*src_stride+warmup_len, i*src_stride+warmup_len+src_stride). By
    construction, query i+1's warmup ((i+1)*src_stride, ...) is EXACTLY the
    tail `warmup_len` bytes of query i's response — only query 0's warmup
    is genuinely new (unseen) source content; every later query's warmup
    is literally the previous query's own response, byte for byte. Trains/
    tests continuous, self-fed generation (only the first anchor is real
    ground truth) rather than chunk_positions_traj's fresh-ground-truth-
    per-window recall.

    Reuses the same generic rec_block shape (type/op_idx/sl0.../c0/c1) as
    chunk_positions_traj, so chunk_mask_fb_traj (keys off op_idx, not
    span) and _forward_segmented/_iter_forward_segments work unchanged.
    `src0` (absolute byte offset into the source) replaces `span` — a
    chunk-index tuple doesn't apply here since windows aren't chunk-aligned.

    `src_stride` need not evenly divide `(src_len - warmup_len)` — the
    LAST query's response is simply clipped shorter than `src_stride` so
    the chain covers the source EXACTLY, no gap and no overlap. Every
    query's `src0` still advances by the regular `src_stride` (so the
    continuity invariant — query i+1's warmup is exactly query i's own
    response — holds all the way through, including into the shortened
    last query), only its `out_len` differs. Requiring exact tiling would
    otherwise force `src_stride` to be a divisor of `(src_len -
    warmup_len)`, which can be awkwardly restrictive (e.g. at
    src_len=1024, warmup_len=8: `1016 = 8*127`, and 127 is prime, so 8 and
    127 are the ONLY two exact divisors — no usable middle ground between
    "126 hops" and "7 hops").
    """
    src_len = n_chunks * chunk_len
    n_queries = -(-(src_len - warmup_len) // src_stride)  # ceiling division
    assert n_queries >= 1

    enc_blocks_c: list[dict] = []
    enc_blocks_m: list[dict] = []
    tags: list[tuple[int, int]] = []
    offset = 0
    for _ in range(n_chunks):
        tags.append((offset, HMN_SRC_OPEN)); offset += 1
        s0 = offset; s1 = s0 + chunk_len; offset = s1
        tags.append((offset, HMN_SRC_CLOSE)); offset += 1
        sl0 = offset; sl1 = sl0 + state_len; offset = sl1
        enc_blocks_c.append(dict(s0=s0, s1=s1, sl0=sl0, sl1=sl1))
        enc_blocks_m.append(dict(s0=s0 - 1, s1=s1 + 1, sl0=sl0, sl1=sl1))
    enc_end = offset

    query_open, query_close = HMN_QUERY_OPEN, HMN_QUERY_CLOSE
    rec_blocks_c: list[dict] = []
    rec_blocks_m: list[dict] = []
    for i in range(n_queries):
        src0 = i * src_stride
        out_len = min(src_stride, src_len - warmup_len - src0)  # last query clipped to land exactly on src_len
        assert out_len > 0
        sl0 = offset; sl1 = sl0 + state_len; offset = sl1
        tags.append((offset, query_open)); offset += 1
        w0 = offset; w1 = w0 + warmup_len; offset = w1
        tags.append((offset, query_close)); offset += 1
        tags.append((offset, HMN_RESPONSE_OPEN)); offset += 1
        c0 = offset; c1 = c0 + out_len; offset = c1
        tags.append((offset, HMN_RESPONSE_CLOSE)); offset += 1
        rec_blocks_c.append(dict(type='initial', src0=src0, out_len=out_len,
                                 is_clean=True, op_idx=i,
                                 sl0=sl0, sl1=sl1, w0=w0, w1=w1, c0=c0, c1=c1))
        rec_blocks_m.append(dict(type='initial', op_idx=i,
                                 sl0=sl0, sl1=sl1, w0=w0 - 1, w1=w1 + 1, c0=c0 - 1, c1=c1 + 1))

    L = offset
    pos_content = dict(enc_blocks=enc_blocks_c, rec_blocks=rec_blocks_c,
                       enc_end=enc_end, warmup_len=warmup_len, L=L)
    pos_mask = dict(enc_blocks=enc_blocks_m, rec_blocks=rec_blocks_m,
                    enc_end=enc_end, warmup_len=warmup_len, L=L)
    return dict(pos_content=pos_content, pos_mask=pos_mask, tags=tags, L=L)


def chunk_mask_fb_traj(pos: dict, hops: int = -1) -> np.ndarray:
    """
    Mask for chunk_positions_traj layouts — same relay exception as
    chunk_mask_fb_hop, but grouped by `op_idx` instead of chain-step span
    since the same span can recur (repeat-query). `hops` semantics identical
    to chunk_mask_fb_hop's, with "the first" meaning op_idx==0.
    """
    if hops == 0:
        raise ValueError("hops=0 is invalid — use hops=-1 for unbounded "
                         "(routing-style, full access to every prior op's "
                         "STATE and the encoding pass) or hops>=1 for a "
                         "bounded N-op recurrent window.")
    L = pos['L']
    r = np.arange(L); c = np.arange(L)
    causal  = c[None, :] <= r[:, None]
    blocked = np.zeros((L, L), dtype=bool)

    enc_blocks = pos['enc_blocks']
    rec_blocks = pos['rec_blocks']

    is_any_chunk = np.zeros(L, dtype=bool)
    for b in enc_blocks:
        is_any_chunk |= (c >= b['s0']) & (c < b['s1'])

    is_any_enc_state = np.zeros(L, dtype=bool)
    for b in enc_blocks:
        is_any_enc_state |= (c >= b['sl0']) & (c < b['sl1'])

    is_any_rec_output = np.zeros(L, dtype=bool)
    for rb2 in rec_blocks:
        if 'c0' in rb2:  # 'noop' blocks have no output region
            is_any_rec_output |= (c >= rb2['c0']) & (c < rb2['c1'])

    last_rb_of_op: dict[int, int] = {}
    for i_rb, rb in enumerate(rec_blocks):
        last_rb_of_op[rb['op_idx']] = i_rb  # last write wins -> last rec_block per relay-producing op

    def _relay_source(prev_rb: dict) -> tuple[int, int]:
        if prev_rb['type'] == 'initial' or prev_rb['type'] == 'noop':
            return prev_rb['sl0'], prev_rb['sl1']
        return prev_rb['slb0'], prev_rb['slb1']

    def _relay_ranges(op_idx: int) -> list[tuple[int, int]]:
        back_range = range(1, op_idx + 1) if hops == -1 else range(1, hops + 1)
        ranges = []
        for back in back_range:
            src_op = op_idx - back
            if src_op < 0:
                break
            ranges.append(_relay_source(rec_blocks[last_rb_of_op[src_op]]))
        return ranges

    def _prior_blocked_union(i_rb: int) -> np.ndarray:
        prior_all = np.zeros(L, dtype=bool)
        for prev_rb in rec_blocks[:i_rb]:
            if prev_rb['type'] == 'noop':
                prior_all |= (c >= prev_rb['sl0']) & (c < prev_rb['sl1'])
            elif prev_rb['type'] == 'initial':
                prior_all |= (c >= prev_rb['sl0']) & (c < prev_rb['sl1'])
                prior_all |= (c >= prev_rb['w0'])  & (c < prev_rb['w1'])
                prior_all |= (c >= prev_rb['c0'])  & (c < prev_rb['c1'])
            else:  # 'refine'
                prior_all |= (c >= prev_rb['sla0']) & (c < prev_rb['sla1'])
                prior_all |= (c >= prev_rb['am0'])  & (c < prev_rb['am1'])
                prior_all |= (c >= prev_rb['slb0']) & (c < prev_rb['slb1'])
                prior_all |= (c >= prev_rb['w0'])   & (c < prev_rb['w1'])
                prior_all |= (c >= prev_rb['c0'])   & (c < prev_rb['c1'])
        return prior_all

    for k, b in enumerate(enc_blocks):
        sl_row = (r >= b['sl0']) & (r < b['sl1'])
        for j, bj in enumerate(enc_blocks):
            if j == k: continue
            blocked |= sl_row[:, None] & ((c >= bj['s0']) & (c < bj['s1']))[None, :]

    for i_rb, rb in enumerate(rec_blocks):
        if rb['type'] == 'noop':
            sl_row = (r >= rb['sl0']) & (r < rb['sl1'])
            blocked |= sl_row[:, None] & is_any_chunk[None, :]
            if hops >= 1 and rb['op_idx'] > 0:
                blocked |= sl_row[:, None] & is_any_enc_state[None, :]

            prior_all = _prior_blocked_union(i_rb)
            for lo, hi in _relay_ranges(rb['op_idx']):
                prior_all = prior_all & ~((c >= lo) & (c < hi))
            blocked |= sl_row[:, None] & prior_all[None, :]
            # a no-op has no warmup/response fields to bound

        elif rb['type'] == 'initial':
            sl_row = (r >= rb['sl0']) & (r < rb['sl1'])
            blocked |= sl_row[:, None] & is_any_chunk[None, :]
            if hops >= 1 and rb['op_idx'] > 0:
                blocked |= sl_row[:, None] & is_any_enc_state[None, :]

            prior_all = _prior_blocked_union(i_rb)
            for lo, hi in _relay_ranges(rb['op_idx']):
                prior_all = prior_all & ~((c >= lo) & (c < hi))
            blocked |= sl_row[:, None] & prior_all[None, :]

            if rb['w0'] < rb['w1']:
                wm_row = (r >= rb['w0']) & (r < rb['w1'])
                own = (c >= rb['sl0']) & (c < rb['sl1']) | (c >= rb['w0']) & (c < rb['w1'])
                blocked |= wm_row[:, None] & ~own[None, :]
            out_row = (r >= rb['c0']) & (r < rb['c1'])
            own = ((c >= rb['sl0']) & (c < rb['sl1']) |
                   (c >= rb['w0'])  & (c < rb['w1'])  |
                   (c >= rb['c0'])  & (c < rb['c1']))
            blocked |= out_row[:, None] & ~own[None, :]

        else:  # type == 'refine'
            sla_row = (r >= rb['sla0']) & (r < rb['sla1'])
            am_row  = (r >= rb['am0'])  & (r < rb['am1'])
            slb_row = (r >= rb['slb0']) & (r < rb['slb1'])
            blocked |= (sla_row | am_row | slb_row)[:, None] & (is_any_chunk | is_any_rec_output)[None, :]

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
# Named trajectory patterns (operations-list constructors) for
# chunk_positions_traj. batch/stream/interleave_delayed are train-mix
# candidates; repeat_query/long_hop_recovery are test-only generalization
# probes — training on them would defeat their purpose.
# ---------------------------------------------------------------------------

def traj_batch(n_chunks: int, window_chunks: int = 2) -> list[tuple]:
    spans = ' '.join(f'Q({i},{i + window_chunks})' for i in range(n_chunks - window_chunks + 1))
    ops, _ = parse_traj_dsl(f'E{n_chunks} {spans}')
    return ops


def traj_stream(n_chunks: int, window_chunks: int = 2) -> list[tuple]:
    dsl_parts = [f'E{window_chunks}']
    for i in range(n_chunks - window_chunks + 1):
        if i > 0:
            dsl_parts.append('E')
        dsl_parts.append(f'Q({i},{i + window_chunks})')
    ops, _ = parse_traj_dsl(' '.join(dsl_parts))
    return ops


def traj_interleave_delayed(n_chunks: int, window_chunks: int = 2) -> list[tuple]:
    spans = [(i, i + window_chunks) for i in range(n_chunks - window_chunks + 1)]
    q_str = ' '.join(f'Q({s},{e})' for s, e in reversed(spans))  # query last span first
    ops, _ = parse_traj_dsl(f'E{n_chunks} {q_str}')
    return ops


def traj_suffix(n_chunks: int, window_chunks: int = 2) -> list[tuple]:
    """Encode everything, then a SINGLE query spanning the last
    `window_chunks` chunks: `Q(n_chunks-window_chunks, n_chunks)` — warmup
    (a few real ground-truth bytes) anchors at the START of that span,
    response covers everything after warmup through the true end of the
    source. No relay chain at all (`op_idx=0`, always exempt from the
    `hops`-bounded relay restriction — see chunk_mask_fb_traj), so this
    doesn't need `hops`/`forward_granularity`/`segment_checkpoint` the way
    the multi-query stitch chain did; `window_chunks` here plays the role
    of "how far from the end the warmup anchor sits" — smaller
    `window_chunks` = warmup closer to the end = less left to generate."""
    assert window_chunks >= 2, 'window_chunks must be >=2 so there is a non-trivial response to generate'
    ops, _ = parse_traj_dsl(f'E{n_chunks} Q({n_chunks - window_chunks},{n_chunks})')
    return ops


def traj_repeat_query(n_chunks: int, window_chunks: int = 2) -> list[tuple]:
    """TEST-ONLY."""
    spans = [(i, i + window_chunks) for i in range(n_chunks - window_chunks + 1)]
    q_str = ' '.join(f'Q({s},{e})' for s, e in spans)
    first_s, first_e = spans[0]
    ops, _ = parse_traj_dsl(f'E{n_chunks} {q_str} Q({first_s},{first_e})')
    return ops


def traj_long_hop_recovery(n_chunks: int, window_chunks: int = 2) -> list[tuple]:
    """TEST-ONLY — like traj_repeat_query but meant to be run with a larger
    n_chunks than training used, to stress the relay over more hops."""
    return traj_repeat_query(n_chunks, window_chunks)


# ---------------------------------------------------------------------------
# Trajectory DSL — compact string notation for operations lists.
#
# Grammar: E (ingest next chunk, must be followed by S) | E<n> (n ingest+
# compress pairs) | S (emit STATE, claims preceding unclaimed E or else a
# bare relay hop) | S<n> (n bare S ops) | Q(s,e) (query span [s,e)) |
# R<n> (n refine rounds — GLOBAL, applies uniformly to every Q in the
# string, not per-query; bare R means R1; at most one R token per string)
#
# Examples: batch "E4 Q(0,2) Q(1,3) Q(2,4)"; stream "E2 Q(0,2) E S Q(1,3) E S
# Q(2,4)"; decay_curve(4 hops) "E2 Q(0,2) S4 Q(0,2)"; one refine round after
# every query "E4 Q(0,2) Q(1,3) Q(2,4) R1"
#
# Returns (ops, n_refine) — n_refine=0 if no R token appears. Every internal
# caller below (traj_batch/stream/interleave_delayed/suffix/repeat_query/
# decay_curve) never emits an R token, so they just discard the n_refine
# part; only a config's own explicit `dsl=` string (see the weave_mix
# dispatch in train()) can set n_refine>0 today.
# ---------------------------------------------------------------------------

def parse_traj_dsl(s: str) -> tuple[list[tuple], int]:
    ops: list[tuple] = []
    next_chunk_idx = 0
    n_refine = 0
    seen_r = False
    for tok in s.split():
        if tok.startswith('Q('):
            inner = tok[2:-1]
            s_str, e_str = inner.split(',')
            ops.append(('Q', (int(s_str), int(e_str))))
        elif tok.startswith('E'):
            n = int(tok[1:]) if len(tok) > 1 else 1
            for _ in range(n):
                ops.append(('E', next_chunk_idx))
                ops.append(('S', None))
                next_chunk_idx += 1
        elif tok.startswith('R'):
            assert not seen_r, f'at most one R token allowed per DSL string, got a second in {s!r}'
            seen_r = True
            n_refine = int(tok[1:]) if len(tok) > 1 else 1
        elif tok.startswith('S'):
            n = int(tok[1:]) if len(tok) > 1 else 1
            for _ in range(n):
                ops.append(('S', None))
        else:
            raise ValueError(f'unrecognized trajectory DSL token: {tok!r}')
    return ops, n_refine


def traj_decay_curve(n_noop_hops: int, window_chunks: int = 2) -> list[tuple]:
    """TEST-ONLY (or train-mix at low hop counts). Query once, take
    n_noop_hops content-free relay hops, then repeat the same query —
    isolates pure relay decay rate from per-hop recall accuracy.

    CONFOUND: produces a much shorter total sequence than checkpoints trained
    with larger n_chunks — zero-shot eval against such a checkpoint can score
    near-0% purely from length extrapolation, not decay. Only trust results
    from a checkpoint actually trained on decay_curve-shaped trajectories.
    """
    ops, _ = parse_traj_dsl(f'E{window_chunks} Q(0,{window_chunks}) S{n_noop_hops} Q(0,{window_chunks})')
    return ops


# =============================================================================
# Attention mask construction
# =============================================================================

def chunk_mask_fb(pos: dict) -> np.ndarray:
    """
    Feedback-argmax refine layout mask. Nochain blackout: each round-0 STATE
    row is blocked from ALL tokens in prior rec_blocks (STATE, warmup, argmax,
    output), not just prior STATEs — blocking only STATEs lets the model
    chain through prior output tokens instead of encoding independently.
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

    # Refine turns must reach earlier turns' output only via their explicit
    # argmax copy, never by attending straight to c0:c1 — those tokens are
    # ground truth during training but greedy-decoded at eval, so a direct
    # path lets training cheat via leaked ground truth.
    is_any_rec_output = np.zeros(L, dtype=bool)
    for rb2 in rec_blocks:
        is_any_rec_output |= (c >= rb2['c0']) & (c < rb2['c1'])

    # Encoding isolation: encoding STATE_k blocked from chunk_j (j≠k)
    for k, b in enumerate(enc_blocks):
        sl_row = (r >= b['sl0']) & (r < b['sl1'])
        for j, bj in enumerate(enc_blocks):
            if j == k: continue
            blocked |= sl_row[:, None] & ((c >= bj['s0']) & (c < bj['s1']))[None, :]

    for i_rb, rb in enumerate(rec_blocks):
        if rb['type'] == 'initial':
            sl_row = (r >= rb['sl0']) & (r < rb['sl1'])
            blocked |= sl_row[:, None] & is_any_chunk[None, :]
            # nochain blackout: round-0 STATE blocked from ALL tokens in prior
            # rec_blocks, not just their STATE (see chunk_mask_fb docstring)
            prior_all = np.zeros(L, dtype=bool)
            for prev_rb in rec_blocks[:i_rb]:
                if prev_rb['type'] == 'initial':
                    prior_all |= (c >= prev_rb['sl0']) & (c < prev_rb['sl1'])
                    prior_all |= (c >= prev_rb['w0'])  & (c < prev_rb['w1'])
                    prior_all |= (c >= prev_rb['c0'])  & (c < prev_rb['c1'])
                else:
                    prior_all |= (c >= prev_rb['sla0']) & (c < prev_rb['sla1'])
                    prior_all |= (c >= prev_rb['am0'])  & (c < prev_rb['am1'])
                    prior_all |= (c >= prev_rb['slb0']) & (c < prev_rb['slb1'])
                    prior_all |= (c >= prev_rb['w0'])   & (c < prev_rb['w1'])
                    prior_all |= (c >= prev_rb['c0'])   & (c < prev_rb['c1'])
            blocked |= sl_row[:, None] & prior_all[None, :]
            if rb['w0'] < rb['w1']:
                wm_row = (r >= rb['w0']) & (r < rb['w1'])
                own = (c >= rb['sl0']) & (c < rb['sl1']) | (c >= rb['w0']) & (c < rb['w1'])
                blocked |= wm_row[:, None] & ~own[None, :]
            out_row = (r >= rb['c0']) & (r < rb['c1'])
            own = ((c >= rb['sl0']) & (c < rb['sl1']) |
                   (c >= rb['w0'])  & (c < rb['w1'])  |
                   (c >= rb['c0'])  & (c < rb['c1']))
            blocked |= out_row[:, None] & ~own[None, :]

        else:  # type == 'refine'
            sla_row = (r >= rb['sla0']) & (r < rb['sla1'])
            am_row  = (r >= rb['am0'])  & (r < rb['am1'])
            slb_row = (r >= rb['slb0']) & (r < rb['slb1'])
            blocked |= (sla_row | am_row | slb_row)[:, None] & (is_any_chunk | is_any_rec_output)[None, :]

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


def chunk_mask_fb_hop(pos: dict, hops: int = -1) -> np.ndarray:
    """
    Mask for chunk_positions_hop layouts — identical to chunk_mask_fb except
    one exception to the nochain blackout: a round-0 STATE row may also
    attend to the relay window's chain steps' last-round STATE (the relay
    exception).

    hops: 0 is invalid (raises). -1 (default) = unbounded — every chain step
    sees the union of ALL earlier chain steps' STATE plus permanent encoding-
    pass access (routing-style). N>=1 = bounded recurrent window — union of
    only the last N chain steps' STATE, AND every chain step past the first
    is additionally blocked from the encoding pass directly, making the
    relay window its only channel (genuine h_t=f(h_{t-1..t-N}, x_t)).
    """
    if hops == 0:
        raise ValueError("hops=0 is invalid — use hops=-1 for unbounded "
                         "(routing-style, full access to every prior chain "
                         "step's STATE and the encoding pass) or hops>=1 for "
                         "a bounded N-hop recurrent window.")
    L = pos['L']
    r = np.arange(L); c = np.arange(L)
    causal  = c[None, :] <= r[:, None]
    blocked = np.zeros((L, L), dtype=bool)

    enc_blocks = pos['enc_blocks']
    rec_blocks = pos['rec_blocks']

    is_any_chunk = np.zeros(L, dtype=bool)
    for b in enc_blocks:
        is_any_chunk |= (c >= b['s0']) & (c < b['s1'])

    is_any_enc_state = np.zeros(L, dtype=bool)
    for b in enc_blocks:
        is_any_enc_state |= (c >= b['sl0']) & (c < b['sl1'])

    is_any_rec_output = np.zeros(L, dtype=bool)
    for rb2 in rec_blocks:
        is_any_rec_output |= (c >= rb2['c0']) & (c < rb2['c1'])

    # Group rec_block indices by chain step so the relay exception can find
    # the immediately preceding chain step's last rec_block.
    chain_step_of_rb: list[int] = []
    span_to_idx: dict[tuple, int] = {}
    for rb in rec_blocks:
        if rb['span'] not in span_to_idx:
            span_to_idx[rb['span']] = len(span_to_idx)
        chain_step_of_rb.append(span_to_idx[rb['span']])
    last_rb_of_chain_step: dict[int, int] = {}
    for i_rb, ci in enumerate(chain_step_of_rb):
        last_rb_of_chain_step[ci] = i_rb  # last write wins -> last index per chain step

    for k, b in enumerate(enc_blocks):
        sl_row = (r >= b['sl0']) & (r < b['sl1'])
        for j, bj in enumerate(enc_blocks):
            if j == k: continue
            blocked |= sl_row[:, None] & ((c >= bj['s0']) & (c < bj['s1']))[None, :]

    for i_rb, rb in enumerate(rec_blocks):
        if rb['type'] == 'initial':
            sl_row = (r >= rb['sl0']) & (r < rb['sl1'])
            blocked |= sl_row[:, None] & is_any_chunk[None, :]

            this_chain_step = chain_step_of_rb[i_rb]
            if hops >= 1 and this_chain_step > 0:
                blocked |= sl_row[:, None] & is_any_enc_state[None, :]
            # relay exception: union of the last `hops` chain steps' own last-round STATE
            relay_ranges: list[tuple[int, int]] = []
            back_range = range(1, this_chain_step + 1) if hops == -1 else range(1, hops + 1)
            for back in back_range:
                src_chain_step = this_chain_step - back
                if src_chain_step < 0:
                    break
                src_last_i_rb = last_rb_of_chain_step[src_chain_step]
                src_last_rb = rec_blocks[src_last_i_rb]
                if src_last_rb['type'] == 'initial':
                    relay_ranges.append((src_last_rb['sl0'], src_last_rb['sl1']))
                else:
                    relay_ranges.append((src_last_rb['slb0'], src_last_rb['slb1']))

            prior_all = np.zeros(L, dtype=bool)
            for prev_rb in rec_blocks[:i_rb]:
                if prev_rb['type'] == 'initial':
                    prior_all |= (c >= prev_rb['sl0']) & (c < prev_rb['sl1'])
                    prior_all |= (c >= prev_rb['w0'])  & (c < prev_rb['w1'])
                    prior_all |= (c >= prev_rb['c0'])  & (c < prev_rb['c1'])
                else:
                    prior_all |= (c >= prev_rb['sla0']) & (c < prev_rb['sla1'])
                    prior_all |= (c >= prev_rb['am0'])  & (c < prev_rb['am1'])
                    prior_all |= (c >= prev_rb['slb0']) & (c < prev_rb['slb1'])
                    prior_all |= (c >= prev_rb['w0'])   & (c < prev_rb['w1'])
                    prior_all |= (c >= prev_rb['c0'])   & (c < prev_rb['c1'])
            for lo, hi in relay_ranges:
                relay_cols = (c >= lo) & (c < hi)
                prior_all = prior_all & ~relay_cols
            blocked |= sl_row[:, None] & prior_all[None, :]

            # relay visibility is NOT extended to warmup/output rows — only the STATE row itself
            if rb['w0'] < rb['w1']:
                wm_row = (r >= rb['w0']) & (r < rb['w1'])
                own = (c >= rb['sl0']) & (c < rb['sl1']) | (c >= rb['w0']) & (c < rb['w1'])
                blocked |= wm_row[:, None] & ~own[None, :]
            out_row = (r >= rb['c0']) & (r < rb['c1'])
            own = ((c >= rb['sl0']) & (c < rb['sl1']) |
                   (c >= rb['w0'])  & (c < rb['w1'])  |
                   (c >= rb['c0'])  & (c < rb['c1']))
            blocked |= out_row[:, None] & ~own[None, :]

        else:  # type == 'refine'
            sla_row = (r >= rb['sla0']) & (r < rb['sla1'])
            am_row  = (r >= rb['am0'])  & (r < rb['am1'])
            slb_row = (r >= rb['slb0']) & (r < rb['slb1'])
            blocked |= (sla_row | am_row | slb_row)[:, None] & (is_any_chunk | is_any_rec_output)[None, :]

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


# =============================================================================
# Batch filling / AR-decode
# =============================================================================

def _fill_argmax_fb(tok_np: np.ndarray, logits: torch.Tensor,
                    pos: dict) -> np.ndarray:
    """Replace teacher-forced argmax tokens with actual model predictions
    from pass 1. Call between pass 1 (no_grad) and pass 2 (grad)."""
    tok = tok_np.copy()
    for rb in pos['rec_blocks']:
        if rb['type'] == 'refine':
            src_c0  = rb['argmax_src_c0']
            out_len = rb['out_len']
            am = logits[:, src_c0-1:src_c0-1+out_len].argmax(-1).cpu().numpy()
            tok[:, rb['am0']:rb['am1']] = am
    return tok


def _cat_kv(kv_a: list, kv_b: list) -> list:
    """Concatenate two layer-wise KV caches along the sequence dim (dim=2)."""
    return [(torch.cat([ka, kb], dim=2), torch.cat([va, vb], dim=2))
            for (ka, va), (kb, vb) in zip(kv_a, kv_b)]


def _iter_forward_segments(pos_content: dict) -> list[dict]:
    """Ordered, contiguous (seg_start, seg_end, kind, block) list spanning the
    whole packed sequence: one entry per encoding block (`<src>` through the
    end of that chunk's own STATE) and one per rec_block (STATE through
    `</response>`). Positions are contiguous by construction (chunk_positions_*
    builds every block from one monotonically increasing offset), so
    seg_start of entry i+1 always equals seg_end of entry i."""
    segs = []
    for b in pos_content['enc_blocks']:
        segs.append(dict(seg_start=b['s0'] - 1, seg_end=b['sl1'], kind='enc', block=b))
    for rb in pos_content['rec_blocks']:
        end = rb['c1'] + 1 if 'c1' in rb else rb['sl1']  # 'noop' blocks have no c0/c1
        start = rb['sl0'] if rb['type'] != 'refine' else rb['sla0']
        segs.append(dict(seg_start=start, seg_end=end, kind='rec', block=rb))
    segs.sort(key=lambda s: s['seg_start'])
    return segs


def _forward_segmented(model: nn.Module, tok_t: torch.Tensor, mask_np: np.ndarray,
                       pos_content: dict, device: torch.device, ls_max: float,
                       granularity: float | int, segment_checkpoint: bool = False) -> torch.Tensor:
    """Alternative to one dense `model(tok_t, mask_t)` call: walk the packed
    sequence in GROUPS of consecutive STATE-bounded segments
    (`_iter_forward_segments`), one forward pass per group, carrying a KV
    cache between groups (same `past_kv`/`return_kv`/`offset` primitives
    `ar_decode_iq_global_rw_tagged` already uses for eval-time decode, here
    run WITH gradients enabled instead of under `torch.no_grad()` — safe
    because `_cat_kv` is a plain `torch.cat`, so `loss.backward()` still
    reaches every earlier group's forward pass). `granularity` is the
    memory/compute knob, either form: an int >=1 is an EXACT segment count
    per group (`1` = the extreme case, one pass per STATE segment, smallest
    possible per-pass attention matrix, most passes); a float in (0, 1] is a
    FRACTION of the total segment count per group, so it scales with
    sequence length instead of needing per-config retuning (`1.0` groups
    everything into one pass — mathematically the same as not using this
    path at all, see the `forward_granularity=None` default in train() which
    skips it entirely instead; smaller fractions approach the per-STATE
    extreme as they shrink toward `1/len(segments)`). Every downstream
    region only ever attends through its own STATE bottleneck (verified
    against the mask-construction rules in chunk_mask_fb_traj/
    chunk_mask_fb_hop), so slicing the SAME precomputed full mask per group
    reproduces exactly the attention a single joint pass would compute for
    those rows — this changes peak memory (never materializes the full
    (L,L) mask/scores at once), not model behavior. Only supports
    'initial'-type rec_blocks today (no refine/argmax-feedback — see the
    segmented-forward plan's Stage 2).

    `segment_checkpoint` (time-axis gradient checkpointing, separate from
    and independent of HMNModel's own `grad_checkpoint` model-DEPTH
    checkpointing — this one checkpoints across the SEGMENT/time axis
    instead, recomputing each group's own forward pass during backward
    rather than retaining its activations): when True, each group's
    `model(...)` call is wrapped in `torch.utils.checkpoint.checkpoint`
    instead of called directly. This is the fix for the OOM measured when
    training `hmn_stitch_src1024.py` at full batch size — without it,
    every group's internal activations (Q/K/V, attention scores, RoPE
    intermediates — everything except the K/V explicitly returned for the
    relay cache) are retained for the WHOLE run's backward pass, so a long
    STATE-segment chain (~100+ groups) accumulates a correspondingly large
    graph even though each individual group's own work is small. With
    `segment_checkpoint=True`, only the returned K/V tensors persist
    (needed for the relay itself) — everything else is recomputed on
    demand during backward, at the cost of ~2x forward compute for
    whichever groups actually need a backward pass through them. Standard
    nested-checkpoint semantics apply: recomputing group i's forward may
    itself need group i-1's own (checkpointed) output recomputed first if
    that hasn't been done yet — PyTorch's autograd handles this
    automatically, same as any other chained/nested checkpoint use."""
    segs = _iter_forward_segments(pos_content)
    if isinstance(granularity, float):
        assert 0 < granularity <= 1.0, 'fractional granularity must be in (0, 1]'
        group_size = max(1, round(granularity * len(segs)))
    else:
        assert granularity >= 1
        group_size = int(granularity)
    groups = [segs[i:i + group_size] for i in range(0, len(segs), group_size)]

    def _seg_fwd(tok_slice, seg_mask_t, kv_cache_arg, offset_val):
        return model(tok_slice, seg_mask_t, past_kv=kv_cache_arg, return_kv=True, offset=offset_val)

    kv_cache = None
    L_cached = 0
    nlls = []
    for group in groups:
        s0, s1 = group[0]['seg_start'], group[-1]['seg_end']
        assert s0 == L_cached, f'segment gap: expected start {L_cached}, got {s0}'
        seg_mask_np = mask_np[s0:s1, :s1]
        seg_mask_t = torch.tensor(seg_mask_np, dtype=torch.float32, device=device)
        tok_slice = tok_t[:, s0:s1]
        if segment_checkpoint:
            logits_grp, seg_kv = _ckpt(_seg_fwd, tok_slice, seg_mask_t, kv_cache, L_cached,
                                       use_reentrant=False)
        else:
            logits_grp, seg_kv = _seg_fwd(tok_slice, seg_mask_t, kv_cache, L_cached)
        kv_cache = seg_kv if kv_cache is None else _cat_kv(kv_cache, seg_kv)
        L_cached = s1

        for seg in group:
            if seg['kind'] != 'rec':
                continue
            rb = seg['block']
            if rb['type'] != 'initial':
                raise NotImplementedError(
                    "_forward_segmented only supports 'initial' rec_blocks today "
                    "(no refine/argmax-feedback support yet)")
            if rb['is_clean']:
                lp  = F.log_softmax(logits_grp[:, rb['c0'] - 1 - s0:rb['c1'] - 1 - s0], dim=-1)
                tgt = tok_t[:, rb['c0']:rb['c1']]
                nll_per = _positional_ls_nll(lp, tgt, ls_max)
                nlls.append(nll_per.mean())
    return torch.stack(nlls).mean()


def make_batch_tagged(rng: np.random.Generator, B: int, n_chunks: int, chunk_len: int,
                      state_len: int, state_vocab_size: int, pos_content: dict,
                      tags: list[tuple[int, int]],
                      data_kind: str = 'random', data_target_bits: float | None = None) -> np.ndarray:
    """data_kind: 'random' (default, uniform bytes) or a structured-data
    generator name (kvmem/structured_data.py) — each batch item gets a fresh
    call so the model can't bake a fixed rule into static weights."""
    sids = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
    sids_feedback_state = np.array(
        _cyclic_state_ids(state_len, state_vocab_size, family=HMN_FEEDBACK_STATE_FAMILY),
        dtype=np.int64)
    wl = pos_content['warmup_len']
    L = pos_content['L']
    tok = np.zeros((B, L), dtype=np.int64)
    if data_kind == 'random':
        segs = rng.integers(0, 256, size=(B, n_chunks, chunk_len), dtype=np.int64)
    else:
        segs = np.stack([generate_structured_chunks(rng, data_kind, n_chunks, chunk_len,
                                                     target_bits=data_target_bits)
                        for _ in range(B)], axis=0)

    for k, b in enumerate(pos_content['enc_blocks']):
        tok[:, b['s0']:b['s1']] = segs[:, k, :]
        tok[:, b['sl0']:b['sl1']] = sids

    rw_xs: np.ndarray | None = None
    for rb in pos_content['rec_blocks']:
        if rb['type'] == 'noop':
            tok[:, rb['sl0']:rb['sl1']] = sids
            continue

        span_s, span_e = rb['span']
        gt = np.concatenate([segs[:, i, :] for i in range(span_s, span_e)], axis=1)

        if rb['type'] == 'initial':
            tok[:, rb['sl0']:rb['sl1']] = sids
            x_min, x_max = rb['warmup_train_range']
            _xdist = rb.get('warmup_x_dist', 'uniform')
            if _xdist == 'arcsine':
                rw_xs = np.clip(np.round(x_min + (x_max - x_min) * rng.beta(0.5, 0.5, size=B)).astype(int), x_min, x_max)
            elif _xdist == 'early_bias':
                rw_xs = np.clip(np.round(x_min + (x_max - x_min) * rng.beta(0.5, 2.0, size=B)).astype(int), x_min, x_max)
            else:
                rw_xs = np.array([int(rng.integers(x_min, x_max + 1)) for _ in range(B)])
            for b_idx in range(B):
                X = rw_xs[b_idx]
                tok[b_idx, rb['w0']:rb['w1']] = gt[b_idx, X:X + wl]
                tok[b_idx, rb['c0']:rb['c1']] = gt[b_idx, X + wl:X + wl + rb['out_len']]
        else:  # 'refine'
            tok[:, rb['sla0']:rb['sla1']] = sids
            tok[:, rb['slb0']:rb['slb1']] = sids_feedback_state
            assert rw_xs is not None
            for b_idx in range(B):
                X = rw_xs[b_idx]
                tok[b_idx, rb['am0']:rb['am1']] = gt[b_idx, X + wl:X + wl + rb['out_len']]
                tok[b_idx, rb['w0']:rb['w1']]   = gt[b_idx, X:X + wl]
                tok[b_idx, rb['c0']:rb['c1']]   = gt[b_idx, X + wl:X + wl + rb['out_len']]

    tag_pos = np.array([p for p, _ in tags], dtype=np.int64)
    tag_ids = np.array([i for _, i in tags], dtype=np.int64)
    tok[:, tag_pos] = tag_ids[None, :]

    return tok


def make_batch_stitch(rng: np.random.Generator, B: int, n_chunks: int, chunk_len: int,
                      state_len: int, state_vocab_size: int, pos_content: dict,
                      tags: list[tuple[int, int]],
                      data_kind: str = 'random', data_target_bits: float | None = None) -> np.ndarray:
    """Batch filler for chunk_positions_stitch layouts — same shape as
    make_batch_tagged but sources warmup/response ground truth by absolute
    byte offset (`rb['src0']`) into the flattened source, not by chunk-index
    span, since stitch windows aren't chunk-aligned. No random warmup-offset
    augmentation (chunk_positions_stitch has no warmup_train_range/rw_xs
    concept — every window's position is fixed by the stitch geometry
    itself, not randomized per batch item)."""
    sids = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
    L = pos_content['L']
    tok = np.zeros((B, L), dtype=np.int64)
    if data_kind == 'random':
        segs = rng.integers(0, 256, size=(B, n_chunks, chunk_len), dtype=np.int64)
    else:
        segs = np.stack([generate_structured_chunks(rng, data_kind, n_chunks, chunk_len,
                                                     target_bits=data_target_bits)
                        for _ in range(B)], axis=0)
    src = segs.reshape(B, n_chunks * chunk_len)  # flattened, byte-addressable source

    for k, b in enumerate(pos_content['enc_blocks']):
        tok[:, b['s0']:b['s1']] = segs[:, k, :]
        tok[:, b['sl0']:b['sl1']] = sids

    wl = pos_content['warmup_len']
    for rb in pos_content['rec_blocks']:
        assert rb['type'] == 'initial', 'chunk_positions_stitch only ever produces initial rec_blocks'
        src0 = rb['src0']
        tok[:, rb['sl0']:rb['sl1']] = sids
        tok[:, rb['w0']:rb['w1']] = src[:, src0:src0 + wl]
        tok[:, rb['c0']:rb['c1']] = src[:, src0 + wl:src0 + wl + rb['out_len']]

    tag_pos = np.array([p for p, _ in tags], dtype=np.int64)
    tag_ids = np.array([i for _, i in tags], dtype=np.int64)
    tok[:, tag_pos] = tag_ids[None, :]

    return tok


@torch.no_grad()
def ar_decode_iq_global_rw_tagged(model, chunks_arr, state_len: int, state_vocab_size: int,
                                  mask_np: np.ndarray, pos_content: dict,
                                  tags: list[tuple[int, int]], device,
                                  warmup_offset: int = 0) -> dict:
    """KV-cached greedy AR decode for the iq_global_rw_tagged layout."""
    if isinstance(chunks_arr, np.ndarray) and chunks_arr.ndim == 2:
        chunks_list = [chunks_arr[k] for k in range(chunks_arr.shape[0])]
    else:
        chunks_list = list(chunks_arr)

    wl = pos_content['warmup_len']
    sids = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
    sids_feedback_state = np.array(
        _cyclic_state_ids(state_len, state_vocab_size, family=HMN_FEEDBACK_STATE_FAMILY),
        dtype=np.int64)
    L = pos_content['L']
    full_mask = torch.tensor(mask_np, dtype=torch.float32, device=device)

    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos_content['enc_blocks']):
        tok[b['s0']:b['s1']]   = chunks_list[k]
        tok[b['sl0']:b['sl1']] = sids

    tag_pos = np.array([p for p, _ in tags], dtype=np.int64)
    tag_ids = np.array([i for _, i in tags], dtype=np.int64)
    tok[tag_pos] = tag_ids

    enc_end  = pos_content['enc_end']
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
        if rb['out_len'] > 0:
            last_pos  = rb['c1'] - 1
            last_t    = torch.tensor([tok[last_pos]], dtype=torch.long, device=device)
            last_mask = full_mask[last_pos:last_pos+1, :L_cached + 1]
            _, last_kv = model(last_t, last_mask, past_kv=kv_cache,
                               return_kv=True, offset=L_cached)
            kv_cache  = _cat_kv(kv_cache, last_kv)
            L_cached += 1

    turn_match_pcts: list[float] = []
    for rb in pos_content['rec_blocks']:
        span_s, span_e = rb['span']
        gt_span = np.concatenate(chunks_list[span_s:span_e])

        if rb['type'] == 'initial':
            tok[rb['sl0']:rb['sl1']] = sids
            if wl > 0:
                tok[rb['w0']:rb['w1']] = gt_span[warmup_offset:warmup_offset + wl].astype(np.int64)
            # seg_start must equal L_cached exactly — sweeps in every tag
            # token between the last cached position and c0 automatically.
            _decode_segment(L_cached, rb)
        else:  # 'refine'
            tok[rb['sla0']:rb['sla1']] = sids
            src_c0 = rb['argmax_src_c0']
            tok[rb['am0']:rb['am1']] = tok[src_c0:src_c0 + rb['out_len']]
            tok[rb['slb0']:rb['slb1']] = sids_feedback_state
            if wl > 0:
                tok[rb['w0']:rb['w1']] = gt_span[warmup_offset:warmup_offset + wl].astype(np.int64)
            _decode_segment(L_cached, rb)

        rb_target = gt_span[warmup_offset + wl: warmup_offset + wl + rb['out_len']]
        rb_gen    = tok[rb['c0']:rb['c1']]
        rb_match  = 100.0 * float(np.sum(rb_gen[:len(rb_target)] == rb_target)) / max(len(rb_target), 1)
        turn_match_pcts.append(rb_match)

    final_rb = pos_content['rec_blocks'][-1]
    span_s, span_e = final_rb['span']
    gt_final  = np.concatenate(chunks_list[span_s:span_e])
    out_len   = final_rb['out_len']
    final_gen = tok[final_rb['c0']:final_rb['c1']]
    target    = gt_final[warmup_offset + wl:warmup_offset + wl + out_len]
    eff_out   = len(target)

    gen_v, target_v = final_gen[:eff_out], target
    match_pct = 100.0 * float(np.sum(gen_v == target_v)) / max(len(target_v), 1)

    tok_tf = torch.tensor(tok, dtype=torch.long, device=device)
    if eff_out > 0:
        tok_tf[final_rb['c0']:final_rb['c0'] + eff_out] = torch.tensor(
            target.astype(np.int64), dtype=torch.long, device=device)
    logits_tf  = model(tok_tf, full_mask)
    tgt_tensor = torch.tensor(target.astype(np.int64), dtype=torch.long, device=device)
    lp_full    = F.log_softmax(logits_tf[final_rb['c0']-1:final_rb['c0']-1+eff_out], dim=-1)
    nll_vals   = -lp_full.gather(1, tgt_tensor.unsqueeze(1)).squeeze(1).cpu().numpy()
    nll = float(nll_vals.mean()) if len(nll_vals) > 0 else float('nan')
    bpb = nll / math.log(2)

    return dict(bpb=bpb, nll=nll, match_pct=match_pct,
               decoded_bytes=final_gen.tolist(), n_valid=len(target_v),
               turn_match_pcts=turn_match_pcts)


@torch.no_grad()
def ar_decode_srs_stitched_tagged_nokv(model, chunks_arr, state_len: int, state_vocab_size: int,
                                       mask_np: np.ndarray, pos_content: dict,
                                       tags: list[tuple[int, int]], device) -> dict:
    """
    Full-recompute (no KV cache) AR decode for the stitched SRS layout — only
    the first window's warmup comes from ground truth, later windows chain
    from the model's own decoded output. No KV cache since dual_attn blocks
    have two attention sublayers per layer, breaking the single-KV-pair
    format the rest of this project's decode functions assume. Fine at this
    scale (L~742); not a production inference path.
    """
    if isinstance(chunks_arr, np.ndarray) and chunks_arr.ndim == 2:
        chunks_list = [chunks_arr[k] for k in range(chunks_arr.shape[0])]
    else:
        chunks_list = list(chunks_arr)

    chunk_len = len(chunks_list[0])
    n_chunks  = len(chunks_list)
    src_len   = n_chunks * chunk_len
    wl        = pos_content['warmup_len']
    sids      = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
    sids_feedback_state = np.array(
        _cyclic_state_ids(state_len, state_vocab_size, family=HMN_FEEDBACK_STATE_FAMILY),
        dtype=np.int64)
    L         = pos_content['L']
    full_mask = torch.tensor(mask_np, dtype=torch.float32, device=device)
    gt_full   = np.concatenate(chunks_list)

    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos_content['enc_blocks']):
        tok[b['s0']:b['s1']]   = chunks_list[k]
        tok[b['sl0']:b['sl1']] = sids

    tag_pos = np.array([p for p, _ in tags], dtype=np.int64)
    tag_ids = np.array([i for _, i in tags], dtype=np.int64)
    tok[tag_pos] = tag_ids

    def _fwd_logits_at(pos: int) -> torch.Tensor:
        """Full forward over tok[:pos+1], return logits at position pos-1
        (i.e. the prediction FOR position pos, causal convention)."""
        t = torch.tensor(tok[:pos], dtype=torch.long, device=device)
        m = full_mask[:pos, :pos]
        logits = model(t, m)
        return logits[-1]

    def _decode_segment(rb):
        for j in range(rb['out_len']):
            pos = rb['c0'] + j
            logits = _fwd_logits_at(pos)
            tok[pos] = int(logits.argmax())

    decoded_bytes = np.full(src_len, -1, dtype=np.int64)
    turn_match_pcts: list[float] = []

    for rb in pos_content['rec_blocks']:
        span_s, span_e = rb['span']
        win_byte_lo = span_s * chunk_len
        gt_span = np.concatenate(chunks_list[span_s:span_e])

        warmup_src = (gt_full[win_byte_lo:win_byte_lo + wl] if win_byte_lo == 0
                     else decoded_bytes[win_byte_lo:win_byte_lo + wl])

        if rb['type'] == 'initial':
            tok[rb['sl0']:rb['sl1']] = sids
            if wl > 0:
                tok[rb['w0']:rb['w1']] = warmup_src
            _decode_segment(rb)
        else:  # 'refine'
            tok[rb['sla0']:rb['sla1']] = sids
            src_c0 = rb['argmax_src_c0']
            tok[rb['am0']:rb['am1']] = tok[src_c0:src_c0 + rb['out_len']]
            tok[rb['slb0']:rb['slb1']] = sids_feedback_state
            if wl > 0:
                tok[rb['w0']:rb['w1']] = warmup_src
            _decode_segment(rb)

        out_len = rb['out_len']
        decoded_bytes[win_byte_lo + wl:win_byte_lo + wl + out_len] = tok[rb['c0']:rb['c1']]
        if wl > 0:
            decoded_bytes[win_byte_lo:win_byte_lo + wl] = tok[rb['w0']:rb['w1']]

        rb_target = gt_span[wl:wl + out_len]
        rb_gen    = tok[rb['c0']:rb['c1']]
        rb_match  = 100.0 * float(np.sum(rb_gen[:len(rb_target)] == rb_target)) / max(len(rb_target), 1)
        turn_match_pcts.append(rb_match)

    gen_v    = decoded_bytes[wl:]
    target_v = gt_full[wl:]
    match_pct = 100.0 * float(np.sum(gen_v == target_v)) / max(len(target_v), 1)

    # Teacher-forced BPB: one extra full forward pass with GT filled into each
    # window's last block's output.
    window_blocks: list[list[dict]] = []
    cur_span = None
    for rb in pos_content['rec_blocks']:
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
        span_s, _ = rb['span']
        win_byte_lo = span_s * chunk_len
        out_len = rb['out_len']
        tgt = torch.tensor(tok_tf[rb['c0']:rb['c1']], dtype=torch.long, device=device)
        lp  = lp_full[rb['c0']-1:rb['c1']-1]
        nll = -lp.gather(1, tgt.unsqueeze(1)).squeeze(1).cpu().numpy()
        nll_buf[win_byte_lo + wl:win_byte_lo + wl + out_len] = nll

    valid = ~np.isnan(nll_buf[wl:])
    bpb = float(np.mean(nll_buf[wl:][valid]) / math.log(2)) if valid.any() else float('nan')

    return dict(bpb=bpb, match_pct=match_pct, decoded_bytes=decoded_bytes.tolist(),
               turn_match_pcts=turn_match_pcts)


@torch.no_grad()
def ar_decode_traj_nokv(model, chunks_arr, state_len: int, state_vocab_size: int,
                        mask_np: np.ndarray, pos_content: dict,
                        tags: list[tuple[int, int]], device) -> dict:
    """
    AR decode for chunk_positions_traj layouts — same mechanics as
    ar_decode_srs_stitched_tagged_nokv (full-recompute, no KV cache; only the
    first query's warmup comes from ground truth) but also handles 'noop'
    rec_blocks (span=None, no decode step, exists only as relay context) and
    keys warmup chaining by span rather than a global byte buffer, since the
    same span can recur non-contiguously (repeat_query/interleave_delayed).
    No BPB — the "last block per contiguous same-span run" grouping that
    would require assumes no span recurs, which doesn't hold here.
    """
    if isinstance(chunks_arr, np.ndarray) and chunks_arr.ndim == 2:
        chunks_list = [chunks_arr[k] for k in range(chunks_arr.shape[0])]
    else:
        chunks_list = list(chunks_arr)

    chunk_len = len(chunks_list[0])
    wl        = pos_content['warmup_len']
    sids      = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
    sids_feedback_state = np.array(
        _cyclic_state_ids(state_len, state_vocab_size, family=HMN_FEEDBACK_STATE_FAMILY),
        dtype=np.int64)
    L         = pos_content['L']
    full_mask = torch.tensor(mask_np, dtype=torch.float32, device=device)
    gt_full   = np.concatenate(chunks_list)

    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos_content['enc_blocks']):
        tok[b['s0']:b['s1']]   = chunks_list[k]
        tok[b['sl0']:b['sl1']] = sids

    tag_pos = np.array([p for p, _ in tags], dtype=np.int64)
    tag_ids = np.array([i for _, i in tags], dtype=np.int64)
    tok[tag_pos] = tag_ids

    def _fwd_logits_at(pos: int) -> torch.Tensor:
        t = torch.tensor(tok[:pos], dtype=torch.long, device=device)
        m = full_mask[:pos, :pos]
        logits = model(t, m)
        return logits[-1]

    def _decode_segment(rb):
        for j in range(rb['out_len']):
            pos = rb['c0'] + j
            logits = _fwd_logits_at(pos)
            tok[pos] = int(logits.argmax())

    decoded_by_span: dict[tuple, np.ndarray] = {}  # last-decoded bytes per span (for warmup chaining)
    turn_match_pcts: list[float] = []

    for rb in pos_content['rec_blocks']:
        if rb['type'] == 'noop':
            tok[rb['sl0']:rb['sl1']] = sids  # placeholder only — real content comes from causal attention
            continue

        span_s, span_e = rb['span']
        gt_span = np.concatenate(chunks_list[span_s:span_e])

        # First-ever occurrence of ANY span starting at byte 0 uses ground
        # truth (nothing decoded yet); every other occurrence (including a
        # REPEATED query of the same span) chains from whatever was most
        # recently decoded for that exact span.
        if span_s == 0 and rb['span'] not in decoded_by_span:
            warmup_src = gt_span[:wl]
        else:
            warmup_src = decoded_by_span.get(rb['span'], gt_span)[:wl]

        if rb['type'] == 'initial':
            tok[rb['sl0']:rb['sl1']] = sids
            if wl > 0:
                tok[rb['w0']:rb['w1']] = warmup_src
            _decode_segment(rb)
        else:  # 'refine'
            tok[rb['sla0']:rb['sla1']] = sids
            src_c0 = rb['argmax_src_c0']
            tok[rb['am0']:rb['am1']] = tok[src_c0:src_c0 + rb['out_len']]
            tok[rb['slb0']:rb['slb1']] = sids_feedback_state
            if wl > 0:
                tok[rb['w0']:rb['w1']] = warmup_src
            _decode_segment(rb)

        out_len = rb['out_len']
        decoded_by_span[rb['span']] = np.concatenate([
            warmup_src if wl > 0 else np.array([], dtype=np.int64),
            tok[rb['c0']:rb['c1']],
        ])

        rb_target = gt_span[wl:wl + out_len]
        rb_gen    = tok[rb['c0']:rb['c1']]
        rb_match  = 100.0 * float(np.sum(rb_gen[:len(rb_target)] == rb_target)) / max(len(rb_target), 1)
        turn_match_pcts.append(rb_match)

    return dict(match_pct=(sum(turn_match_pcts) / len(turn_match_pcts) if turn_match_pcts else float('nan')),
               turn_match_pcts=turn_match_pcts)


@torch.no_grad()
def ar_decode_stitch(model, chunks_arr, state_len: int, state_vocab_size: int,
                        mask_np: np.ndarray, pos_content: dict,
                        tags: list[tuple[int, int]], device) -> dict:
    """
    KV-cached AR decode for chunk_positions_stitch layouts — true continuous
    decode: only query 0's warmup is ground truth (the single real anchor);
    every later query's warmup is the model's OWN just-decoded response from
    the immediately preceding query (rec_blocks are ordered by op_idx with
    no span concept, so "immediately preceding rec_block" is unambiguous,
    unlike ar_decode_traj_nokv's span-matching).

    KV-cached (same past_kv/return_kv/offset primitives as
    ar_decode_iq_global_rw_tagged, adapted here), NOT full-recompute —
    required at this layout's scale: a naive "_nokv" version (recompute the
    whole growing prefix from scratch for every single generated byte, one
    O(pos^2) dense pass per byte) hits real MPS OOM well before L~5000
    (measured directly: OOM'd at L~4900 attempting full recompute). Caching
    keeps each step's forward pass small (existing cache + 1 new token),
    not the whole prefix, and avoids ever rebuilding a large (L,L) mask.
    """
    if isinstance(chunks_arr, np.ndarray) and chunks_arr.ndim == 2:
        chunks_list = [chunks_arr[k] for k in range(chunks_arr.shape[0])]
    else:
        chunks_list = list(chunks_arr)

    wl        = pos_content['warmup_len']
    sids      = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
    L         = pos_content['L']
    full_mask = torch.tensor(mask_np, dtype=torch.float32, device=device)
    gt_full   = np.concatenate(chunks_list)

    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos_content['enc_blocks']):
        tok[b['s0']:b['s1']]   = chunks_list[k]
        tok[b['sl0']:b['sl1']] = sids

    tag_pos = np.array([p for p, _ in tags], dtype=np.int64)
    tag_ids = np.array([i for _, i in tags], dtype=np.int64)
    tok[tag_pos] = tag_ids

    enc_end  = pos_content['enc_end']
    enc_t    = torch.tensor(tok[:enc_end], dtype=torch.long, device=device)
    enc_mask = full_mask[:enc_end, :enc_end]
    _, kv_cache = model(enc_t, enc_mask, return_kv=True)
    L_cached = enc_end

    def _decode_segment(seg_start: int, rb: dict):
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
        if rb['out_len'] > 0:
            last_pos  = rb['c1'] - 1
            last_t    = torch.tensor([tok[last_pos]], dtype=torch.long, device=device)
            last_mask = full_mask[last_pos:last_pos+1, :L_cached + 1]
            _, last_kv = model(last_t, last_mask, past_kv=kv_cache,
                               return_kv=True, offset=L_cached)
            kv_cache  = _cat_kv(kv_cache, last_kv)
            L_cached += 1

        # Cache the `</response>` closing tag at position c1 too (its token
        # id is already known — filled by the tok[tag_pos]=tag_ids line at
        # the top of this function, before any decoding starts) — without
        # this, L_cached stops one short of the next rec_block's own sl0
        # (which sits right after this tag), and the next segment's
        # assert seg_start == L_cached fails.
        tag_pos_ = rb['c1']
        tag_t    = torch.tensor([tok[tag_pos_]], dtype=torch.long, device=device)
        tag_mask = full_mask[tag_pos_:tag_pos_+1, :L_cached + 1]
        _, tag_kv = model(tag_t, tag_mask, past_kv=kv_cache, return_kv=True, offset=L_cached)
        kv_cache  = _cat_kv(kv_cache, tag_kv)
        L_cached += 1

    turn_match_pcts: list[float] = []
    prev_response: np.ndarray | None = None

    for rb in pos_content['rec_blocks']:
        src0 = rb['src0']
        warmup_src = gt_full[:wl] if prev_response is None else prev_response[-wl:]

        seg_start = rb['sl0']
        assert seg_start == L_cached, f'segment gap: expected start {L_cached}, got {seg_start}'
        tok[rb['sl0']:rb['sl1']] = sids
        if wl > 0:
            tok[rb['w0']:rb['w1']] = warmup_src

        _decode_segment(seg_start, rb)

        rb_gen    = tok[rb['c0']:rb['c1']]
        rb_target = gt_full[src0 + wl:src0 + wl + rb['out_len']]
        rb_match  = 100.0 * float(np.sum(rb_gen == rb_target)) / max(len(rb_target), 1)
        turn_match_pcts.append(rb_match)
        prev_response = rb_gen

    return dict(match_pct=(sum(turn_match_pcts) / len(turn_match_pcts) if turn_match_pcts else float('nan')),
               turn_match_pcts=turn_match_pcts)


# =============================================================================
# Attention / norm / RoPE primitives
# =============================================================================

def rope_freqs(d_head: int, base: float = 10000.0, device=None) -> torch.Tensor:
    i = torch.arange(0, d_head, 2, dtype=torch.float32, device=device)
    return 1.0 / (base ** (i / d_head))


def yarn_freqs(d_head: int, L_train: int, L_max: int,
               base: float = 10000.0,
               beta_fast: int = 32, beta_slow: int = 1,
               device=None) -> torch.Tensor:
    """YaRN NTK-aware scaled RoPE (arXiv:2309.00071)."""
    s     = L_max / L_train
    i     = torch.arange(0, d_head, 2, dtype=torch.float32, device=device)
    inv_f = 1.0 / (base ** (i / d_head))
    wl    = 2 * math.pi / inv_f
    lo, hi = 2 * math.pi * beta_slow, 2 * math.pi * beta_fast
    ramp  = torch.clamp((wl - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (1 - ramp) * inv_f + ramp * (inv_f / s)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """x: (..., H, L, d_head)  freqs: (d_head//2,)  offset: position base."""
    L, dh  = x.shape[-2], x.shape[-1]
    pos    = torch.arange(offset, offset + L, dtype=torch.float32, device=x.device)
    angles = pos[:, None] * freqs[None, :]
    cos_a, sin_a = angles.cos(), angles.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos_a - x2 * sin_a,
                        x1 * sin_a + x2 * cos_a], dim=-1).reshape(x.shape)


class MHAttention(nn.Module):
    def __init__(self, d: int, n_heads: int,
                 rope: bool = False, freqs: torch.Tensor | None = None,
                 null_kv: bool = False, qk_norm: bool = False,
                 logit_cap: float = 0.0, attn_temp: bool = False):
        """null_kv=True: append a learnable (null_k, null_v) pair to the KV
        sequence before softmax — a "blank slot" to attend to when no real
        token is relevant, soft gating without hard masking. null_k inits to
        zero (Q·null_k=0 initially) but is learned."""
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d // n_heads
        self.rope    = rope
        self.null_kv = null_kv
        self.qk_norm  = qk_norm
        self.logit_cap = logit_cap   # tanh soft-cap value (0 = disabled)
        self.attn_temp = attn_temp   # learned per-head temperature
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d, bias=False)
        if qk_norm:
            self.q_norm_scale = nn.Parameter(torch.ones(self.d_head))
            self.k_norm_scale = nn.Parameter(torch.ones(self.d_head))
        if attn_temp:
            # log-scale per head, init 0 → temperature = 1/sqrt(d_head) at start
            self.log_attn_temp = nn.Parameter(torch.zeros(n_heads))
        if freqs is not None:
            self.register_buffer('freqs', freqs)
        else:
            self.freqs = None

    @staticmethod
    def _rms_normalize(x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        return x * norm * scale

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                past_kv: tuple | None = None,
                return_kv: bool = False,
                offset: int = 0) -> torch.Tensor | tuple:
        batched = x.dim() == 3
        if not batched:
            x = x.unsqueeze(0)
        B, L, d = x.shape
        H, dh = self.n_heads, self.d_head
        Q = self.W_Q(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)
        K = self.W_K(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)
        V = self.W_V(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)
        if self.qk_norm:
            Q = self._rms_normalize(Q, self.q_norm_scale)
            K = self._rms_normalize(K, self.k_norm_scale)
        if self.rope and self.freqs is not None:
            Q = apply_rope(Q, self.freqs, offset=offset)
            K = apply_rope(K, self.freqs, offset=offset)
        K_cur, V_cur = K, V
        if past_kv is not None:
            K_past, V_past = past_kv
            K = torch.cat([K_past, K], dim=2)
            V = torch.cat([V_past, V], dim=2)
        if self.null_kv:
            null = torch.zeros(B, H, 1, dh, device=K.device, dtype=K.dtype)
            K    = torch.cat([K, null], dim=2)
            V    = torch.cat([V, null], dim=2)
            mask = F.pad(mask, (0, 1), value=0.0)
        # Logit soft-cap or learned temperature: bypass SDPA, compute manually.
        # Chunked attention: compute in row-chunks (chunk_attn=0 = full SDPA).
        chunk = getattr(self, 'chunk_attn', 0)
        if self.logit_cap > 0 or self.attn_temp:
            scale = 1.0 / math.sqrt(dh)
            if self.attn_temp:
                # per-head multiplicative scale: exp(log_temp) / sqrt(d_head)
                scale = scale * self.log_attn_temp.exp().view(1, H, 1, 1)
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            if self.logit_cap > 0:
                scores = torch.tanh(scores / self.logit_cap) * self.logit_cap
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
            out = torch.softmax(scores, dim=-1) @ V
        elif chunk > 0 and L > chunk:
            m = mask.unsqueeze(0).unsqueeze(0)           # (1,1,L_q,L_kv)
            parts = []
            for i in range(0, L, chunk):
                parts.append(F.scaled_dot_product_attention(
                    Q[:, :, i:i+chunk, :], K, V,
                    attn_mask=m[:, :, i:i+chunk, :]))
            out = torch.cat(parts, dim=2)
        else:
            out = F.scaled_dot_product_attention(Q, K, V,
                                                 attn_mask=mask.unsqueeze(0).unsqueeze(0))
        out = out.permute(0, 2, 1, 3).reshape(B, L, d)
        out = self.W_O(out)
        if not batched:
            out = out.squeeze(0)
        if return_kv:
            return out, (K_cur, V_cur)
        return out


class RMSNorm(nn.Module):
    """True RMSNorm: no mean-centering, no bias — just a learned per-dim scale."""
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


def _make_norm(d: int, rmsnorm: bool) -> nn.Module:
    return RMSNorm(d) if rmsnorm else nn.LayerNorm(d)


class FFN(nn.Module):
    """gated=True: SwiGLU (silu(W1 x) * W3 x -> W2) instead of plain GELU-MLP.
    Note: at the same d_ff this adds ~50% more params (extra W3) — ablation
    flag, not param-matched."""
    def __init__(self, d: int, d_ff: int, gated: bool = False):
        super().__init__()
        self.gated = gated
        self.W1 = nn.Linear(d, d_ff, bias=False)
        if gated:
            self.W3 = nn.Linear(d, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            h = F.silu(self.W1(x)) * self.W3(x)
        else:
            h = self.W1(x)
            h = 0.5 * h * (1.0 + torch.tanh(0.7978845608028654 * (h + 0.044715 * h ** 3)))
        return self.W2(h)


def _attn_sublayer(attn, norm, x, mask):
    return x + attn(norm(x), mask)


# =============================================================================
# Model — three selectable block types on one unified model class
# =============================================================================

class AttnMlpBlock(nn.Module):
    """block_type='attn_mlp': x = x + attn(norm1(x)); x = x + ffn(norm2(x))"""
    def __init__(self, d: int, n_heads: int, d_ff: int,
                 rope: bool = False, freqs: torch.Tensor | None = None,
                 null_kv: bool = False, qk_norm: bool = False,
                 gated_ffn: bool = False, rmsnorm: bool = False,
                 logit_cap: float = 0.0, attn_temp: bool = False):
        super().__init__()
        self.norm1 = _make_norm(d, rmsnorm)
        self.attn  = MHAttention(d, n_heads, rope=rope, freqs=freqs,
                                 null_kv=null_kv, qk_norm=qk_norm,
                                 logit_cap=logit_cap, attn_temp=attn_temp)
        self.norm2 = _make_norm(d, rmsnorm)
        self.ffn   = FFN(d, d_ff, gated=gated_ffn)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                past_kv: tuple | None = None,
                return_kv: bool = False,
                offset: int = 0) -> torch.Tensor | tuple:
        attn_out = self.attn(self.norm1(x), mask,
                             past_kv=past_kv, return_kv=return_kv, offset=offset)
        if return_kv:
            attn_out, kv = attn_out
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        if return_kv:
            return x, kv
        return x


class DualAttnBlock(nn.Module):
    """block_type='dual_attn': x = x + attn1(norm1(x)); x = x + attn2(norm2(x)),
    no FFN. No KV-cache support (two attn sublayers per block breaks the
    single-KV-pair-per-layer assumption used elsewhere)."""
    def __init__(self, d: int, n_heads: int,
                 rope: bool = False, freqs: torch.Tensor | None = None,
                 null_kv: bool = False, qk_norm: bool = False,
                 rmsnorm: bool = False, logit_cap: float = 0.0,
                 attn_temp: bool = False):
        super().__init__()
        self.norm1 = _make_norm(d, rmsnorm)
        self.attn1 = MHAttention(d, n_heads, rope=rope, freqs=freqs,
                                 null_kv=null_kv, qk_norm=qk_norm,
                                 logit_cap=logit_cap, attn_temp=attn_temp)
        self.norm2 = _make_norm(d, rmsnorm)
        self.attn2 = MHAttention(d, n_heads, rope=rope, freqs=freqs,
                                 null_kv=null_kv, qk_norm=qk_norm,
                                 logit_cap=logit_cap, attn_temp=attn_temp)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                ckpt_attn: bool = False) -> torch.Tensor:
        if ckpt_attn and self.training:
            x = _ckpt(_attn_sublayer, self.attn1, self.norm1, x, mask, use_reentrant=False)
            x = _ckpt(_attn_sublayer, self.attn2, self.norm2, x, mask, use_reentrant=False)
        else:
            x = _attn_sublayer(self.attn1, self.norm1, x, mask)
            x = _attn_sublayer(self.attn2, self.norm2, x, mask)
        return x


class SingleAttnBlock(nn.Module):
    """block_type='single_attn' (default): x = x + attn(norm(x)), no FFN.
    Use n_layers = 2x the equivalent dual_attn config to match total
    attention-op count."""
    def __init__(self, d: int, n_heads: int,
                 rope: bool = False, freqs: torch.Tensor | None = None,
                 null_kv: bool = False, qk_norm: bool = False,
                 rmsnorm: bool = False, logit_cap: float = 0.0,
                 attn_temp: bool = False):
        super().__init__()
        self.norm = _make_norm(d, rmsnorm)
        self.attn = MHAttention(d, n_heads, rope=rope, freqs=freqs,
                                null_kv=null_kv, qk_norm=qk_norm,
                                logit_cap=logit_cap, attn_temp=attn_temp)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                past_kv: tuple | None = None,
                return_kv: bool = False,
                offset: int = 0) -> torch.Tensor | tuple:
        attn_out = self.attn(self.norm(x), mask,
                             past_kv=past_kv, return_kv=return_kv, offset=offset)
        if return_kv:
            attn_out, kv = attn_out
            x = x + attn_out
            return x, kv
        x = x + attn_out
        return x


class HMNModel(nn.Module):
    """Unified model class parameterized by block_type in
    {'attn_mlp', 'dual_attn', 'single_attn'}. KV-cache (past_kv/return_kv/
    offset) is only meaningful for attn_mlp/single_attn (one attn per block);
    dual_attn has two KV pairs per layer and doesn't support it."""
    def __init__(self, V: int, d: int, n_layers: int, n_heads: int,
                 d_ff: int = 0,
                 block_type: str = 'single_attn',
                 rope: bool = False, yarn: bool = False,
                 L_train: int = 512, L_max: int = 4096,
                 grad_checkpoint: bool | str | None = False,
                 null_kv: bool = False,
                 qk_norm: bool = False,
                 gated_ffn: bool = False,
                 rmsnorm: bool = False,
                 embed_scale: bool = False,
                 zero_init_residual: bool = False,
                 depth_scaled_init: bool = False,
                 logit_cap: float = 0.0,
                 attn_temp: bool = False,
                 tied_embed: bool = False,
                 V_out: int = 256):
        assert block_type in ('attn_mlp', 'dual_attn', 'single_attn')
        super().__init__()
        self.block_type = block_type
        n_special            = V - 256             # number of special tokens (tags, slot IDs)
        self.data_embed      = nn.Embedding(256, d)        # data bytes 0-255
        self.special_embed   = nn.Embedding(n_special, d)  # special tokens 256+
        self.n_special       = n_special
        self.norm_out        = _make_norm(d, rmsnorm)
        self.W_out            = nn.Linear(d, V_out, bias=False)  # output: data bytes only
        self.grad_checkpoint  = grad_checkpoint
        self.embed_scale      = math.sqrt(d) if embed_scale else None
        self.tied_embed       = tied_embed
        self.V_out             = V_out

        freqs = None
        if rope:
            d_head = d // n_heads
            freqs  = (yarn_freqs(d_head, L_train=L_train, L_max=L_max)
                      if yarn else rope_freqs(d_head))

        if block_type == 'attn_mlp':
            self.blocks = nn.ModuleList([
                AttnMlpBlock(d, n_heads, d_ff, rope=rope, freqs=freqs, null_kv=null_kv,
                            qk_norm=qk_norm, gated_ffn=gated_ffn, rmsnorm=rmsnorm,
                            logit_cap=logit_cap, attn_temp=attn_temp)
                for _ in range(n_layers)
            ])
        elif block_type == 'dual_attn':
            self.blocks = nn.ModuleList([
                DualAttnBlock(d, n_heads, rope=rope, freqs=freqs, null_kv=null_kv,
                              qk_norm=qk_norm, rmsnorm=rmsnorm,
                              logit_cap=logit_cap, attn_temp=attn_temp)
                for _ in range(n_layers)
            ])
        else:  # single_attn
            self.blocks = nn.ModuleList([
                SingleAttnBlock(d, n_heads, rope=rope, freqs=freqs, null_kv=null_kv,
                                qk_norm=qk_norm, rmsnorm=rmsnorm,
                                logit_cap=logit_cap, attn_temp=attn_temp)
                for _ in range(n_layers)
            ])

        self._init_weights(zero_init_residual=zero_init_residual,
                           depth_scaled_init=depth_scaled_init,
                           n_layers=n_layers)

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Route tokens to data_embed (0-255) or special_embed (256+)."""
        is_sp = tokens >= 256
        data_ids    = tokens.clamp(0, 255)
        special_ids = (tokens - 256).clamp(0, self.n_special - 1)
        d_emb = self.data_embed(data_ids)
        s_emb = self.special_embed(special_ids)
        mask  = is_sp.unsqueeze(-1).to(d_emb.dtype)
        x = s_emb * mask + d_emb * (1.0 - mask)
        if self.embed_scale is not None:
            x = x * self.embed_scale
        return x

    def _init_weights(self, zero_init_residual: bool = False,
                      depth_scaled_init: bool = False,
                      n_layers: int = 0):
        nn.init.normal_(self.data_embed.weight, std=0.02)
        nn.init.normal_(self.special_embed.weight, std=0.05)
        nn.init.normal_(self.W_out.weight, std=0.02)
        for name, p in self.named_parameters():
            if 'embed' in name or 'W_out' in name:
                continue
            if p.dim() == 2:
                nn.init.normal_(p, std=math.sqrt(2.0 / p.shape[-1]))
        if zero_init_residual:
            for block in self.blocks:
                if self.block_type == 'attn_mlp':
                    nn.init.zeros_(block.attn.W_O.weight)
                    nn.init.zeros_(block.ffn.W2.weight)
                elif self.block_type == 'dual_attn':
                    nn.init.zeros_(block.attn1.W_O.weight)
                    nn.init.zeros_(block.attn2.W_O.weight)
                else:
                    nn.init.zeros_(block.attn.W_O.weight)
        if depth_scaled_init and n_layers > 0:
            # GPT-2 style: scale residual projections by 1/sqrt(2*n_layers)
            # so the residual stream variance stays O(1) at init regardless of depth.
            std = 0.02 / math.sqrt(2.0 * n_layers)
            for block in self.blocks:
                if self.block_type == 'attn_mlp':
                    nn.init.normal_(block.attn.W_O.weight, std=std)
                    nn.init.normal_(block.ffn.W2.weight, std=std)
                elif self.block_type == 'dual_attn':
                    nn.init.normal_(block.attn1.W_O.weight, std=std)
                    nn.init.normal_(block.attn2.W_O.weight, std=std)
                else:
                    nn.init.normal_(block.attn.W_O.weight, std=std)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor,
                past_kv: list | None = None,
                return_kv: bool = False,
                offset: int = 0,
                return_features: bool = False) -> torch.Tensor | tuple:
        """
        tokens          : (B, L) or (L,) int64
        mask            : (L_q, L_kv) — L_kv = L_past + L when past_kv given
        past_kv         : list[n_layers] of (K_past, V_past) — cached prefix KV.
                          Only supported for block_type in ('attn_mlp', 'single_attn').
        return_kv       : return (logits, kv_list) instead of just logits
        return_features : return (logits, x) where x is the pre-head residual stream
                          (B, L, d); disables grad_checkpoint to preserve full graph.
        offset          : RoPE position offset (= L_past for suffix pass)

        grad_checkpoint: for attn_mlp/single_attn, True checkpoints each block
        during backward. For dual_attn, None | 'block' | 'attn' (whole-block
        vs per-attn-sublayer checkpointing).
        """
        batched = tokens.dim() == 2
        if not batched:
            tokens = tokens.unsqueeze(0)
        x      = self._embed(tokens)

        if self.block_type == 'dual_attn':
            for block in self.blocks:
                if self.grad_checkpoint == 'block' and self.training:
                    x = _ckpt(block, x, mask, use_reentrant=False)
                elif self.grad_checkpoint == 'attn':
                    x = block(x, mask, ckpt_attn=True)
                else:
                    x = block(x, mask)
            h_out  = self.norm_out(x)
            logits = self.W_out(h_out)
            if not batched:
                logits = logits.squeeze(0)
            if return_features:
                return logits, x if batched else x.squeeze(0)
            if return_kv:
                raise NotImplementedError('dual_attn blocks do not support KV caching')
            return logits

        kv_out = []
        L_past = past_kv[0][0].shape[2] if past_kv is not None else 0
        _offset = offset if offset else L_past

        for i, block in enumerate(self.blocks):
            pkv = past_kv[i] if past_kv is not None else None
            use_ckpt = (self.grad_checkpoint and self.training
                        and pkv is None and not return_kv and not return_features)
            if use_ckpt:
                x = _ckpt(block, x, mask, use_reentrant=False)
            else:
                result = block(x, mask, past_kv=pkv,
                               return_kv=return_kv, offset=_offset)
                if return_kv:
                    x, kv_i = result
                    kv_out.append(kv_i)
                else:
                    x = result

        h_out  = self.norm_out(x)
        if self.tied_embed:
            logits = F.linear(h_out, self.data_embed.weight[:self.V_out])
        else:
            logits = self.W_out(h_out)
        if not batched:
            logits = logits.squeeze(0)
            x = x.squeeze(0)
        if return_features:
            return logits, x
        if return_kv:
            return logits, kv_out
        return logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(hp: dict, device=None) -> HMNModel:
    """Factory for HMNModel. block_type defaults to 'single_attn'."""
    V_in = hp.get('V', HMN_TAG_VOCAB_SIZE)
    model = HMNModel(
        V=V_in, d=hp['d'], n_layers=hp['n_layers'],
        n_heads=hp['n_heads'], d_ff=hp.get('d_ff', 0),
        block_type=hp.get('block_type', 'single_attn'),
        rope=hp.get('rope', False),
        yarn=hp.get('yarn', False),
        L_train=hp.get('L_train', hp.get('seg_len', 512)),
        L_max=hp.get('L_max', hp.get('seg_len', 512) * 8),
        grad_checkpoint=hp.get('grad_checkpoint', False),
        null_kv=hp.get('null_kv', False),
        qk_norm=hp.get('qk_norm', False),
        gated_ffn=hp.get('gated_ffn', False),
        rmsnorm=hp.get('rmsnorm', False),
        embed_scale=hp.get('embed_scale', False),
        zero_init_residual=hp.get('zero_init_residual', False),
        depth_scaled_init=hp.get('depth_scaled_init', False),
        logit_cap=hp.get('logit_cap', 0.0),
        attn_temp=hp.get('attn_temp', False),
        tied_embed=hp.get('tied_embed', False),
        V_out=hp.get('V_out', 256),
    )
    chunk_attn = hp.get('chunk_attn', 0)
    if chunk_attn > 0:
        for block in model.blocks:
            if model.block_type == 'dual_attn':
                block.attn1.chunk_attn = chunk_attn
                block.attn2.chunk_attn = chunk_attn
            else:
                block.attn.chunk_attn = chunk_attn
    if device is not None:
        model = model.to(device)
    return model


# =============================================================================
# Misc training-loop utilities
# =============================================================================

def load_chunks_padded(path: str, n_chunks: int,
                       chunk_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Load file, split by newline, distribute into n_chunks groups, pad to
    chunk_len. First (n_lines % n_chunks) groups get one extra line, ensuring
    no empty groups when n_lines >= n_chunks."""
    raw     = open(path, 'rb').read()
    lines   = [l for l in raw.split(b'\n') if l]
    n_lines = len(lines)

    base  = n_lines // n_chunks
    extra = n_lines % n_chunks      # first `extra` groups get one extra line
    groups: list[bytes] = []
    start = 0
    for gi in range(n_chunks):
        count = base + (1 if gi < extra else 0)
        groups.append(b''.join(lines[start:start + count]))
        start += count

    chunks     = np.zeros((n_chunks, chunk_len), dtype=np.int64)
    valid_mask = np.zeros((n_chunks, chunk_len), dtype=bool)
    for k, g in enumerate(groups):
        if g:
            b       = np.frombuffer(g[:chunk_len], dtype=np.uint8).astype(np.int64)
            n_real  = min(len(b), chunk_len)
            chunks[k, :n_real]     = b[:n_real]
            valid_mask[k, :n_real] = True

    return chunks, valid_mask

class _StatusWriter:
    """Truncate-and-rewrite file for tqdm — stays 1-2 lines, tail -f works."""
    def __init__(self, path: str):
        self._f = open(path, 'w', buffering=1)

    def write(self, s: str):
        self._f.seek(0)
        self._f.truncate()
        self._f.write(s)
        self._f.flush()

    def flush(self): pass

    def close(self): self._f.close()


def _positional_ls_nll(lp: torch.Tensor, tgt: torch.Tensor, ls_max: float) -> torch.Tensor:
    """
    NLL with positional label smoothing: ε=0 at position 0, ε=ls_max at position N-1.
    lp:  (B, out_len, V)  log-probs
    tgt: (B, out_len)     target token IDs
    Returns (B, out_len) per-token NLL.
    """
    out_len = lp.shape[1]
    nll_hard = -lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)          # (B, out_len)
    if ls_max <= 0.0:
        return nll_hard
    eps = torch.linspace(0.0, ls_max, out_len, device=lp.device)    # (out_len,)
    nll_soft = -lp.mean(dim=-1)                                       # (B, out_len)
    return (1.0 - eps) * nll_hard + eps * nll_soft


def load_config(path: str) -> dict:
    """Load hp dict from a Python config file (must define module-level `hp`)."""
    spec   = importlib.util.spec_from_file_location('_kvmem_cfg', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'hp'):
        raise ValueError(f'{path!r} must define a module-level `hp` dict')
    return dict(module.hp)


def make_test_sequences(seg_len: int) -> dict[str, list[int]]:
    """
    Deterministic held-out test sequences of length seg_len.
    All bytes in [DATA_LO=0x20, 0xFF], never protocol bytes.
    """
    V = 256 - DATA_LO
    seqs = {}
    seqs['up_counter']   = [DATA_LO + (i % V) for i in range(seg_len)]
    seqs['down_counter'] = [DATA_LO + (V - 1 - i % V) for i in range(seg_len)]
    base_odd = 1 if V % 2 == 0 else 0
    seqs['odd']          = [DATA_LO + (base_odd + 2*i) % V for i in range(seg_len)]
    seqs['even']         = [DATA_LO + (2*i) % V for i in range(seg_len)]
    seqs['linear']       = [DATA_LO + (4*i) % V for i in range(seg_len)]
    period = max(4, min(seg_len // 2, V // 4))
    step   = V // period
    seqs['sawtooth']     = [DATA_LO + (i % period) * step for i in range(seg_len)]
    half = seg_len // 2
    first_half  = [DATA_LO + (2*i) % V for i in range(half)]
    second_half = list(reversed(first_half))
    extra = [DATA_LO + (2*half) % V] if seg_len % 2 == 1 else []
    seqs['palindrome']   = first_half + extra + second_half
    geo = [DATA_LO]
    for _ in range(seg_len - 1):
        nxt = int(geo[-1] * 1.1)
        geo.append(DATA_LO if nxt > 255 else nxt)
    seqs['geometric'] = geo
    return seqs


# =============================================================================
# Training loop
# =============================================================================

def train(hp: dict, log_base: str = 'logs', device_str: str = 'cpu'):
    device = torch.device(device_str)
    rng    = np.random.default_rng(hp.get('seed', 42))
    torch.manual_seed(hp.get('seed', 42))

    name     = hp.get('name', 'chat_tags')
    log_dir  = os.path.join(log_base, name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_file    = open(os.path.join(log_dir, 'train.log'),    'a', buffering=1)
    jsonl_file  = open(os.path.join(log_dir, 'train.jsonl'), 'a', buffering=1)
    status_file = _StatusWriter(os.path.join(log_dir, 'train_status.log'))

    def _log(msg): print(msg); print(msg, file=log_file)
    def _jlog(d):  jsonl_file.write(json.dumps(d) + '\n')

    hp_model = dict(V=hp.get('V', HMN_TAG_VOCAB_SIZE),
                    d=hp['d'], n_layers=hp['n_layers'],
                    n_heads=hp['n_heads'], d_ff=hp.get('d_ff', 0),
                    block_type=hp.get('block_type', 'single_attn'),
                    rope=hp.get('rope', True), yarn=hp.get('yarn', True),
                    null_kv=hp.get('null_kv', True), compile=hp.get('compile', False),
                    rmsnorm=hp.get('rmsnorm', False),
                    grad_checkpoint=hp.get('grad_checkpoint', False),
                    chunk_attn=hp.get('chunk_attn', 0))
    model    = build_model(hp_model, device)
    n_params = sum(p.numel() for p in model.parameters())
    _log(f'Model: {n_params:,} params  device={device}  V={hp_model["V"]}  '
         f'block_type={hp_model["block_type"]}  (kvmem/hmn.py consolidated draft)')

    if hp.get('_pretrained_ckpt'):
        ckpt = torch.load(hp['_pretrained_ckpt'], map_location=device)
        src_sd = ckpt['model']
        dst_sd = model.state_dict()
        grown = []
        for k, dst_t in dst_sd.items():
            if k not in src_sd:
                continue
            src_t = src_sd[k]
            if src_t.shape == dst_t.shape:
                dst_sd[k] = src_t
            elif src_t.dim() >= 1 and src_t.shape[1:] == dst_t.shape[1:] and src_t.shape[0] < dst_t.shape[0]:
                # vocab grew (new tag IDs appended) — copy the overlapping prefix,
                # leave the new rows at their fresh random init.
                dst_sd[k][:src_t.shape[0]] = src_t
                grown.append(f'{k}: {tuple(src_t.shape)}->{tuple(dst_t.shape)}')
            else:
                raise RuntimeError(f'Unhandled shape mismatch for {k}: {src_t.shape} vs {dst_t.shape}')
        model.load_state_dict(dst_sd)
        _log(f'Loaded: {hp["_pretrained_ckpt"]}' + (f'  (grown: {grown})' if grown else ''))

    lr_max  = hp.get('lr_max', 3e-4)
    wd      = hp.get('wd', 0.001)
    warmup_steps = hp.get('warmup_steps', 500)
    use_actual_am = hp.get('use_actual_argmax', True)
    wrong_token_weight = hp.get('wrong_token_weight', 0.0)  # alpha: extra NLL weight on wrong-argmax positions
    data_kind = hp.get('data_kind', 'random')  # 'random' (default) | 'chaotic'/'fractal'/'ca' (structured-data track)
    data_target_bits = hp.get('data_target_bits', None)
    repeat_batch = hp.get('repeat_batch', 1)  # gradient steps taken on the same sampled batch before resampling
    assert repeat_batch >= 1
    # memory/compute knob: None (default) = one dense pass, zero overhead, current behavior.
    # int N = N STATE-segments per forward pass; float f in (0,1] = fraction of segments per
    # pass (scales with sequence length). Wired into the weave_mix/stitch_mix branches.
    forward_granularity = hp.get('forward_granularity', None)
    # time-axis (segment) gradient checkpointing — separate from and independent of
    # HMNModel's own model-depth grad_checkpoint. Only meaningful when forward_granularity
    # is set (nothing to checkpoint across segments otherwise).
    segment_checkpoint = hp.get('segment_checkpoint', False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=wd, betas=(0.9, 0.999))

    curriculum = hp.get('curriculum', [])
    assert curriculum
    log_every  = hp.get('log_every', 500)

    global_step = 0
    t_start = time.time()

    for stage_i, stage in enumerate(curriculum):
        if 'weave_mix' in stage:
            # Samples from a weighted mix of named weave patterns each step.
            # Only train-mix-safe patterns are accepted (asserted below) —
            # test-only generalization probes would defeat their own purpose
            # if trained on. n_refine fixed at 0 (no refine support for weave yet).
            n_chunks   = stage['n_chunks']
            chunk_len  = stage['chunk_len']
            state_len  = hp.get('state_len', 8)
            state_vocab_size = hp.get('state_vocab_size', 2)
            warmup_len = hp.get('warmup_len', 8)
            window_chunks = stage.get('window_chunks', 2)
            B          = stage.get('B', 8)
            n_steps    = stage.get('n_steps', 60000)
            stage_eval_every = stage.get('eval_every', 5000)
            ls_max     = hp.get('ls_max', 0.0)

            _WEAVE_TRAIN_PATTERNS = dict(batch=traj_batch, stream=traj_stream,
                                         interleave_delayed=traj_interleave_delayed,
                                         suffix=traj_suffix)
            hops = stage.get('hops', -1)  # default -1 = unbounded (routing-style); hops=0 is invalid

            weave_mix_cfg = stage['weave_mix']  # list of {weight, pattern[, n_chunks, window_chunks]} OR {weight, dsl}
            trajectories = []
            for wcfg in weave_mix_cfg:
                if 'dsl' in wcfg:
                    # Explicit DSL string — bypasses the named-pattern lookup entirely, e.g.
                    # dsl='E4 Q(0,2) Q(1,3) Q(2,4)' (see parse_traj_dsl's grammar comment).
                    # n_chunks is derived from the DSL itself (count of 'E' ops), not passed
                    # separately — the string is already the single source of truth for shape.
                    pname = wcfg['dsl']  # used only for logging/bookkeeping below
                    ops, w_n_refine = parse_traj_dsl(wcfg['dsl'])
                    w_n_chunks = sum(1 for op, _ in ops if op == 'E')
                else:
                    pname = wcfg['pattern']
                    assert pname in _WEAVE_TRAIN_PATTERNS, (
                        f"weave_mix pattern {pname!r} is not a train-mix candidate — only "
                        f"{list(_WEAVE_TRAIN_PATTERNS)} are safe to train on; repeat_query/"
                        f"long_hop_recovery/decay_curve are deliberately test-only "
                        f"generalization probes (see docs/HISTORY.md §4c)")
                    w_n_chunks = wcfg.get('n_chunks', n_chunks)
                    w_window_chunks = wcfg.get('window_chunks', window_chunks)
                    w_n_refine = wcfg.get('n_refine', 0)
                    ops = _WEAVE_TRAIN_PATTERNS[pname](w_n_chunks, w_window_chunks)
                built = chunk_positions_traj(chunk_len, state_len, warmup_len, ops,
                                             n_refine=w_n_refine, state_vocab_size=state_vocab_size)
                pos_content, pos_mask, tags, L = (built['pos_content'], built['pos_mask'],
                                                  built['tags'], built['L'])
                mask_np = chunk_mask_fb_traj(pos_mask, hops=hops)
                mask_t  = torch.tensor(mask_np, dtype=torch.float32, device=device)
                trajectories.append(dict(weight=wcfg['weight'], pattern=pname, n_chunks=w_n_chunks,
                                         pos_content=pos_content, mask_np=mask_np, mask_t=mask_t,
                                         tags=tags, L=L))
            traj_weights = np.array([t['weight'] for t in trajectories], dtype=np.float64)
            traj_weights = traj_weights / traj_weights.sum()

            _log(f'\n[stage {stage_i}] weave_mix='
                 f'{[(t["pattern"], t["n_chunks"], round(w, 2)) for t, w in zip(trajectories, traj_weights)]}  '
                 f'chunk_len={chunk_len} state={state_len} wl={warmup_len} '
                 f'hops={hops}  B={B}  steps={n_steps}')

            lr_min      = hp.get('lr_min', 0.0)
            cosine_T0   = hp.get('cosine_T0', 20000)
            cosine_Tmul = hp.get('cosine_T_mult', 1)
            lr_schedule = hp.get('lr_schedule', 'constant')

            def _lr(s):
                if s <= warmup_steps:
                    return lr_max * s / max(warmup_steps, 1)
                if lr_schedule != 'cosine_restarts':
                    return lr_max
                t = s - warmup_steps
                T_i = cosine_T0
                while t >= T_i:
                    t -= T_i
                    T_i = int(T_i * cosine_Tmul)
                return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t / max(T_i, 1)))

            stage_best_val = -1.0
            _cached_batch = None  # (traj, tok_np, tok_t) — reused for `repeat_batch` consecutive steps
            pbar = tqdm(range(1, n_steps + 1), desc=f'stage{stage_i}', dynamic_ncols=True, file=status_file)
            for local_step in pbar:
                global_step += 1
                lr = _lr(local_step)
                for pg in opt.param_groups: pg['lr'] = lr

                model.train(); opt.zero_grad()

                if _cached_batch is None or (local_step - 1) % repeat_batch == 0:
                    traj = trajectories[rng.choice(len(trajectories), p=traj_weights)]
                    t_pos_content, t_mask_t, t_tags = traj['pos_content'], traj['mask_t'], traj['tags']
                    tok_np = make_batch_tagged(rng, B, traj['n_chunks'], chunk_len, state_len, state_vocab_size,
                                               t_pos_content, t_tags, data_kind=data_kind,
                                               data_target_bits=data_target_bits)
                    tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)
                    _cached_batch = (traj, t_pos_content, t_mask_t, t_tags, tok_t)
                else:
                    traj, t_pos_content, t_mask_t, t_tags, tok_t = _cached_batch

                if forward_granularity is not None:
                    loss = _forward_segmented(model, tok_t, traj['mask_np'], t_pos_content,
                                              device, ls_max, forward_granularity,
                                              segment_checkpoint=segment_checkpoint)
                else:
                    logits = model(tok_t, t_mask_t)
                    nlls = []
                    for rb in t_pos_content['rec_blocks']:
                        if not rb['is_clean']:
                            continue
                        lp  = F.log_softmax(logits[:, rb['c0'] - 1:rb['c1'] - 1], dim=-1)
                        tgt = tok_t[:, rb['c0']:rb['c1']]
                        nll_per = _positional_ls_nll(lp, tgt, ls_max)
                        nlls.append(nll_per.mean())
                    loss = torch.stack(nlls).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                loss_f = float(loss.detach())
                pbar.set_postfix(loss=f'{loss_f:.3f}', lr=f'{lr:.1e}', refresh=False)
                if local_step % log_every == 0:
                    _jlog(dict(step=global_step, loss=round(loss_f, 5), lr=lr))
                    print(str(pbar), file=log_file, flush=True)

                if local_step % stage_eval_every == 0 or local_step == n_steps:
                    model.eval()
                    elapsed = time.time() - t_start
                    h, m = divmod(int(elapsed), 3600); m, s = divmod(m, 60)
                    _log(f'\n--- stage={stage_i} step={local_step}/{n_steps}'
                         f'  g={global_step}  loss={loss_f:.4f}  {h:02d}:{m:02d}:{s:02d} ---')

                    val_means = []
                    for traj in trajectories:
                        val_seqs = make_test_sequences(traj['n_chunks'] * chunk_len)
                        val_n_seqs = hp.get('val_n_seqs')
                        if val_n_seqs is not None:
                            val_seqs = dict(list(val_seqs.items())[:val_n_seqs])
                        pcts = []
                        for sname, seq_bytes in val_seqs.items():
                            chunks_list = [seq_bytes[k * chunk_len:(k + 1) * chunk_len]
                                          for k in range(traj['n_chunks'])]
                            r = ar_decode_traj_nokv(model, np.array(chunks_list), state_len,
                                                    state_vocab_size, traj['mask_np'],
                                                    traj['pos_content'], traj['tags'], device)
                            pcts.append(r['match_pct'])
                        m_ = sum(pcts) / len(pcts)
                        val_means.append(m_)
                        _log(f'  val/weave/{traj["pattern"]:<20} match={m_:.1f}%')
                    vmean = sum(val_means) / len(val_means)
                    _log(f'  val/weave/MEAN               match={vmean:.1f}%')

                    torch.save(dict(model=model.state_dict(), hp=hp, step=global_step),
                              os.path.join(ckpt_dir, f'stage{stage_i}_last.pt'))
                    if vmean > stage_best_val:
                        stage_best_val = vmean
                        torch.save(dict(model=model.state_dict(), hp=hp, step=global_step, val_mean=vmean),
                                  os.path.join(ckpt_dir, f'stage{stage_i}_best.pt'))

            torch.save(dict(model=model.state_dict(), hp=hp, step=global_step),
                      os.path.join(ckpt_dir, f'stage{stage_i}_end.pt'))
            _log(f'[stage {stage_i}] done. saved stage{stage_i}_end.pt (best={stage_best_val:.1f}%)')
            continue

        if 'stitch_mix' in stage:
            # True continuous-decode training: chunk_positions_stitch builds a
            # byte-precise query chain where only the very first query's warmup
            # is genuine unseen ground truth — every later query's warmup is
            # EXACTLY the previous query's own response (see that function's
            # docstring). Mixes multiple src_stride entries (partial-stitch-depth
            # supervision) the same way weave_mix mixes pattern/window_chunks.
            n_chunks   = stage['n_chunks']
            chunk_len  = stage['chunk_len']
            state_len  = hp.get('state_len', 8)
            state_vocab_size = hp.get('state_vocab_size', 2)
            warmup_len = hp.get('warmup_len', 8)
            B          = stage.get('B', 8)
            n_steps    = stage.get('n_steps', 60000)
            stage_eval_every = stage.get('eval_every', 5000)
            ls_max     = hp.get('ls_max', 0.0)
            hops = stage.get('hops', -1)  # default -1 = unbounded; hops=0 is invalid

            stitch_mix_cfg = stage['stitch_mix']  # list of {weight, src_stride}
            trajectories = []
            for scfg in stitch_mix_cfg:
                src_stride = scfg['src_stride']
                built = chunk_positions_stitch(chunk_len, n_chunks, state_len, warmup_len,
                                               src_stride, state_vocab_size=state_vocab_size)
                pos_content, pos_mask, tags, L = (built['pos_content'], built['pos_mask'],
                                                  built['tags'], built['L'])
                mask_np = chunk_mask_fb_traj(pos_mask, hops=hops)
                mask_t  = torch.tensor(mask_np, dtype=torch.float32, device=device)
                trajectories.append(dict(weight=scfg['weight'], src_stride=src_stride, n_chunks=n_chunks,
                                         pos_content=pos_content, mask_np=mask_np, mask_t=mask_t,
                                         tags=tags, L=L))
            traj_weights = np.array([t['weight'] for t in trajectories], dtype=np.float64)
            traj_weights = traj_weights / traj_weights.sum()

            _log(f'\n[stage {stage_i}] stitch_mix='
                 f'{[(t["src_stride"], round(w, 2)) for t, w in zip(trajectories, traj_weights)]}  '
                 f'chunk_len={chunk_len} n_chunks={n_chunks} state={state_len} wl={warmup_len} '
                 f'hops={hops}  B={B}  steps={n_steps}')

            lr_min      = hp.get('lr_min', 0.0)
            cosine_T0   = hp.get('cosine_T0', 20000)
            cosine_Tmul = hp.get('cosine_T_mult', 1)
            lr_schedule = hp.get('lr_schedule', 'constant')

            def _lr(s):
                if s <= warmup_steps:
                    return lr_max * s / max(warmup_steps, 1)
                if lr_schedule != 'cosine_restarts':
                    return lr_max
                t = s - warmup_steps
                T_i = cosine_T0
                while t >= T_i:
                    t -= T_i
                    T_i = int(T_i * cosine_Tmul)
                return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t / max(T_i, 1)))

            stage_best_val = -1.0
            _cached_batch = None
            pbar = tqdm(range(1, n_steps + 1), desc=f'stage{stage_i}', dynamic_ncols=True, file=status_file)
            for local_step in pbar:
                global_step += 1
                lr = _lr(local_step)
                for pg in opt.param_groups: pg['lr'] = lr

                model.train(); opt.zero_grad()

                if _cached_batch is None or (local_step - 1) % repeat_batch == 0:
                    traj = trajectories[rng.choice(len(trajectories), p=traj_weights)]
                    t_pos_content, t_mask_t, t_tags = traj['pos_content'], traj['mask_t'], traj['tags']
                    tok_np = make_batch_stitch(rng, B, n_chunks, chunk_len, state_len, state_vocab_size,
                                               t_pos_content, t_tags, data_kind=data_kind,
                                               data_target_bits=data_target_bits)
                    tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)
                    _cached_batch = (traj, t_pos_content, t_mask_t, t_tags, tok_t)
                else:
                    traj, t_pos_content, t_mask_t, t_tags, tok_t = _cached_batch

                if forward_granularity is not None:
                    loss = _forward_segmented(model, tok_t, traj['mask_np'], t_pos_content,
                                              device, ls_max, forward_granularity,
                                              segment_checkpoint=segment_checkpoint)
                else:
                    logits = model(tok_t, t_mask_t)
                    nlls = []
                    for rb in t_pos_content['rec_blocks']:
                        if not rb['is_clean']:
                            continue
                        lp  = F.log_softmax(logits[:, rb['c0'] - 1:rb['c1'] - 1], dim=-1)
                        tgt = tok_t[:, rb['c0']:rb['c1']]
                        nll_per = _positional_ls_nll(lp, tgt, ls_max)
                        nlls.append(nll_per.mean())
                    loss = torch.stack(nlls).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                loss_f = float(loss.detach())
                pbar.set_postfix(loss=f'{loss_f:.3f}', lr=f'{lr:.1e}', refresh=False)
                if local_step % log_every == 0:
                    _jlog(dict(step=global_step, loss=round(loss_f, 5), lr=lr))
                    print(str(pbar), file=log_file, flush=True)

                if local_step % stage_eval_every == 0 or local_step == n_steps:
                    model.eval()
                    elapsed = time.time() - t_start
                    h, m = divmod(int(elapsed), 3600); m, s = divmod(m, 60)
                    _log(f'\n--- stage={stage_i} step={local_step}/{n_steps}'
                         f'  g={global_step}  loss={loss_f:.4f}  {h:02d}:{m:02d}:{s:02d} ---')

                    val_means = []
                    for traj in trajectories:
                        val_seqs = make_test_sequences(n_chunks * chunk_len)
                        val_n_seqs = hp.get('val_n_seqs')
                        if val_n_seqs is not None:
                            val_seqs = dict(list(val_seqs.items())[:val_n_seqs])
                        pcts = []
                        for sname, seq_bytes in val_seqs.items():
                            chunks_list = [seq_bytes[k * chunk_len:(k + 1) * chunk_len]
                                          for k in range(n_chunks)]
                            r = ar_decode_stitch(model, np.array(chunks_list), state_len,
                                                    state_vocab_size, traj['mask_np'],
                                                      traj['pos_content'], traj['tags'], device)
                            pcts.append(r['match_pct'])
                        m_ = sum(pcts) / len(pcts)
                        val_means.append(m_)
                        _log(f'  val/stitch/src_stride={traj["src_stride"]:<4} match={m_:.1f}%')
                    vmean = sum(val_means) / len(val_means)
                    _log(f'  val/stitch/MEAN               match={vmean:.1f}%')

                    torch.save(dict(model=model.state_dict(), hp=hp, step=global_step),
                              os.path.join(ckpt_dir, f'stage{stage_i}_last.pt'))
                    if vmean > stage_best_val:
                        stage_best_val = vmean
                        torch.save(dict(model=model.state_dict(), hp=hp, step=global_step, val_mean=vmean),
                                  os.path.join(ckpt_dir, f'stage{stage_i}_best.pt'))

            torch.save(dict(model=model.state_dict(), hp=hp, step=global_step),
                      os.path.join(ckpt_dir, f'stage{stage_i}_end.pt'))
            _log(f'[stage {stage_i}] done. saved stage{stage_i}_end.pt (best={stage_best_val:.1f}%)')
            continue

        if 'chain_steps' in stage:
            # One packed pos/mask built from a fixed `chain_steps` list (one
            # rec_block per chain step, each with n_refine refine rounds),
            # unlike the traj_mix branch below which samples many small
            # per-trajectory sequences by weight each step. Always uses
            # chunk_positions_hop/chunk_mask_fb_hop.
            n_chunks   = stage['n_chunks']
            chunk_len  = stage['chunk_len']
            state_len  = hp.get('state_len', 8)
            state_vocab_size = hp.get('state_vocab_size', 2)
            warmup_len = hp.get('warmup_len', 8)
            n_refine   = stage.get('n_refine', 2)
            B          = stage.get('B', 8)
            n_steps    = stage.get('n_steps', 60000)
            stage_eval_every = stage.get('eval_every', 5000)
            ls_max     = hp.get('ls_max', 0.0)
            chain_steps = stage['chain_steps']
            hops  = stage.get('hops', -1)  # default -1 = unbounded (routing-style); hops=0 is invalid

            built = chunk_positions_hop(n_chunks, chunk_len, state_len, warmup_len,
                                         chain_steps, n_refine=n_refine,
                                         state_vocab_size=state_vocab_size)
            pos_content, pos_mask, tags, L = (built['pos_content'], built['pos_mask'],
                                              built['tags'], built['L'])
            mask_np = chunk_mask_fb_hop(pos_mask, hops=hops)
            mask_t  = torch.tensor(mask_np, dtype=torch.float32, device=device)

            _log(f'\n[stage {stage_i}] n_chunks={n_chunks} chunk_len={chunk_len} state={state_len} '
                 f'hops={hops} '
                 f'wl={warmup_len} chain_steps={chain_steps} n_refine={n_refine}  '
                 f'B={B}  steps={n_steps}  L={L}')

            lr_min      = hp.get('lr_min', 0.0)
            cosine_T0   = hp.get('cosine_T0', 20000)
            cosine_Tmul = hp.get('cosine_T_mult', 1)
            lr_schedule = hp.get('lr_schedule', 'constant')

            def _lr(s):
                if s <= warmup_steps:
                    return lr_max * s / max(warmup_steps, 1)
                if lr_schedule != 'cosine_restarts':
                    return lr_max
                t = s - warmup_steps
                T_i = cosine_T0
                while t >= T_i:
                    t -= T_i
                    T_i = int(T_i * cosine_Tmul)
                return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t / max(T_i, 1)))

            val_seg_len = n_chunks * chunk_len
            val_seqs    = make_test_sequences(val_seg_len)
            val_n_seqs  = hp.get('val_n_seqs')
            if val_n_seqs is not None:
                val_seqs = dict(list(val_seqs.items())[:val_n_seqs])

            eval_file   = hp.get('eval_file', None)
            test_chunks = None
            if eval_file:
                try:
                    test_chunks, _ = load_chunks_padded(eval_file, n_chunks, chunk_len)
                except Exception as e:
                    _log(f'  [test eval disabled: {e}]')

            stage_best_val = -1.0
            _cached_base_np = None  # raw sampled batch (pre-argmax-feedback), reused for `repeat_batch` steps
            pbar = tqdm(range(1, n_steps + 1), desc=f'stage{stage_i}', dynamic_ncols=True, file=status_file)
            for local_step in pbar:
                global_step += 1
                lr = _lr(local_step)
                for pg in opt.param_groups: pg['lr'] = lr

                model.train(); opt.zero_grad()

                if _cached_base_np is None or (local_step - 1) % repeat_batch == 0:
                    _cached_base_np = make_batch_tagged(rng, B, n_chunks, chunk_len, state_len, state_vocab_size,
                                               pos_content, tags, data_kind=data_kind,
                                               data_target_bits=data_target_bits)
                tok_np = _cached_base_np
                tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)

                wrong_masks: dict[int, np.ndarray] = {}

                if use_actual_am:
                    with torch.no_grad():
                        logits_1 = model(tok_t, mask_t)
                    if wrong_token_weight > 0:
                        for i, rb in enumerate(pos_content['rec_blocks']):
                            if rb['type'] != 'refine':
                                continue
                            src_c0 = rb['argmax_src_c0']
                            wrong_masks[i] = tok_np[:, src_c0:src_c0 + rb['out_len']].copy()
                    tok_np = _fill_argmax_fb(tok_np, logits_1, pos_content)
                    if wrong_token_weight > 0:
                        for i, rb in enumerate(pos_content['rec_blocks']):
                            if i not in wrong_masks:
                                continue
                            wrong_masks[i] = (tok_np[:, rb['am0']:rb['am1']] != wrong_masks[i])
                    tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)

                logits = model(tok_t, mask_t)
                logits_by_rb = {i: logits for i in range(len(pos_content['rec_blocks']))}

                nlls = []
                for i, rb in enumerate(pos_content['rec_blocks']):
                    if not rb['is_clean']:
                        continue
                    logits_i = logits_by_rb[i]
                    lp  = F.log_softmax(logits_i[:, rb['c0']-1:rb['c1']-1], dim=-1)
                    tgt = tok_t[:, rb['c0']:rb['c1']]
                    nll_per = _positional_ls_nll(lp, tgt, ls_max)
                    if i in wrong_masks:
                        w = 1.0 + wrong_token_weight * wrong_masks[i].astype(np.float32)
                        nll_per = nll_per * torch.tensor(w, device=device, dtype=torch.float32)
                    nlls.append(nll_per.mean())
                loss = torch.stack(nlls).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                loss_f = float(loss.detach())
                pbar.set_postfix(loss=f'{loss_f:.3f}', lr=f'{lr:.1e}', refresh=False)
                if local_step % log_every == 0:
                    _jlog(dict(step=global_step, loss=round(loss_f, 5), lr=lr))
                    print(str(pbar), file=log_file, flush=True)

                if local_step % stage_eval_every == 0 or local_step == n_steps:
                    model.eval()
                    elapsed = time.time() - t_start
                    h, m = divmod(int(elapsed), 3600); m, s = divmod(m, 60)
                    _log(f'\n--- stage={stage_i} step={local_step}/{n_steps}'
                         f'  g={global_step}  loss={loss_f:.4f}  {h:02d}:{m:02d}:{s:02d} ---')

                    span_last_idx: dict[tuple, int] = {}
                    for i, rb in enumerate(pos_content['rec_blocks']):
                        span_last_idx[rb['span']] = i

                    def _eval_on(seqs_iter, tag_prefix):
                        span_means = {span: [] for span in chain_steps}
                        stitched_means = []
                        for sname, chunks_arr in seqs_iter:
                            r = ar_decode_srs_stitched_tagged_nokv(model, chunks_arr, state_len, state_vocab_size,
                                                                   mask_np, pos_content, tags, device)
                            stitched_means.append(r['match_pct'])
                            for span in chain_steps:
                                idx = span_last_idx[span]
                                span_means[span].append(r['turn_match_pcts'][idx])
                            _log(f'  {tag_prefix}/{sname:<15} per-span={[round(r["turn_match_pcts"][span_last_idx[s]],1) for s in chain_steps]}  stitched={r["match_pct"]:.1f}%')
                        means = []
                        for span in chain_steps:
                            m_ = sum(span_means[span]) / len(span_means[span])
                            means.append(m_)
                            _log(f'  {tag_prefix}/span{span}/MEAN               match={m_:.1f}%')
                        overall = sum(means) / len(means)
                        _log(f'  {tag_prefix}/MEAN               match={overall:.1f}%')
                        stitched_overall = sum(stitched_means) / len(stitched_means)
                        _log(f'  {tag_prefix}/STITCHED_MEAN               match={stitched_overall:.1f}%')
                        return stitched_overall

                    val_iter = ((sname, np.array([seq[k*chunk_len:(k+1)*chunk_len] for k in range(n_chunks)], np.int64))
                               for sname, seq in val_seqs.items())
                    vmean = _eval_on(val_iter, 'val/srs')

                    if test_chunks is not None:
                        test_iter = iter([('test', test_chunks)])
                        _eval_on(test_iter, 'test/srs')

                    _jlog(dict(step=global_step, eval_mean=round(vmean, 2)))

                    torch.save(dict(model=model.state_dict(), hp=hp, step=global_step),
                              os.path.join(ckpt_dir, f'stage{stage_i}_last.pt'))
                    if vmean > stage_best_val:
                        stage_best_val = vmean
                        torch.save(dict(model=model.state_dict(), hp=hp, step=global_step, val_mean=vmean),
                                  os.path.join(ckpt_dir, f'stage{stage_i}_best.pt'))

            torch.save(dict(model=model.state_dict(), hp=hp, step=global_step),
                      os.path.join(ckpt_dir, f'stage{stage_i}_end.pt'))
            _log(f'[stage {stage_i}] done. saved stage{stage_i}_end.pt (best={stage_best_val:.1f}%)')
            continue

        n_chunks   = stage['n_chunks']
        chunk_len  = stage['chunk_len']
        state_len  = hp.get('state_len', 8)
        state_vocab_size = hp.get('state_vocab_size', 2)
        warmup_len = hp.get('warmup_len', 8)
        window_chunks = stage.get('window_chunks', 2)
        B          = stage.get('B', 8)
        n_steps    = stage.get('n_steps', 50000)
        stage_eval_every = stage.get('eval_every', 5000)
        ls_max     = hp.get('ls_max', 0.0)

        # traj_mix: list of dicts each {weight, n_refine, warmup_x_fixed, warmup_x_dist}.
        traj_mix_cfg = stage.get('traj_mix')
        if traj_mix_cfg is None:
            traj_mix_cfg = [dict(weight=1.0, n_refine=stage.get('n_refine', 0))]

        trajectories = []
        for tcfg in traj_mix_cfg:
            t_n_refine = tcfg.get('n_refine', 0)
            built = chunk_positions_iq_global_rw_tagged(
                n_chunks, chunk_len, state_len, warmup_len,
                window_chunks=window_chunks,
                warmup_x_fixed=tcfg.get('warmup_x_fixed'),
                warmup_x_dist=tcfg.get('warmup_x_dist', 'uniform'),
                n_refine=t_n_refine)
            pos_content, pos_mask, tags, L = (built['pos_content'], built['pos_mask'],
                                              built['tags'], built['L'])
            mask_np = chunk_mask_fb(pos_mask)
            mask_t  = torch.tensor(mask_np, dtype=torch.float32, device=device)
            trajectories.append(dict(weight=tcfg['weight'], n_refine=t_n_refine,
                                     pos_content=pos_content, mask_np=mask_np, mask_t=mask_t,
                                     tags=tags, L=L, has_ir=t_n_refine > 0,
                                     warmup_x_fixed=tcfg.get('warmup_x_fixed')))
        traj_weights = np.array([t['weight'] for t in trajectories], dtype=np.float64)
        traj_weights = traj_weights / traj_weights.sum()

        # Build one eval trajectory per canonical warmup offset X — a
        # trajectory built for a different X's warmup_x_fixed would use the
        # wrong tag — falling back to the single highest-n_refine trajectory
        # when none match exactly.
        default_eval_traj = max(trajectories, key=lambda t: t['n_refine'])
        eval_traj_by_x: dict[int, dict] = {}
        for t in trajectories:
            x = t.get('warmup_x_fixed')
            if x is None:
                continue
            if x not in eval_traj_by_x or t['n_refine'] > eval_traj_by_x[x]['n_refine']:
                eval_traj_by_x[x] = t

        _log(f'\n[stage {stage_i}] n_chunks={n_chunks} chunk_len={chunk_len} '
             f'state={state_len} wl={warmup_len} B={B}  steps={n_steps}  '
             f'traj_mix={[(round(w,2), t["n_refine"], t.get("warmup_x_fixed")) for t, w in zip(trajectories, traj_weights)]}  '
             f'L(eval)={default_eval_traj["L"]}  eval_traj_by_x={sorted(eval_traj_by_x.keys())}')

        lr_min      = hp.get('lr_min', 0.0)
        cosine_T0   = hp.get('cosine_T0', 20000)
        cosine_Tmul = hp.get('cosine_T_mult', 1)
        lr_schedule = hp.get('lr_schedule', 'constant')

        def _lr(s):
            if s <= warmup_steps:
                return lr_max * s / max(warmup_steps, 1)
            if lr_schedule != 'cosine_restarts':
                return lr_max
            t = s - warmup_steps
            T_i = cosine_T0
            while t >= T_i:
                t -= T_i
                T_i = int(T_i * cosine_Tmul)
            return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t / max(T_i, 1)))

        val_seg_len = n_chunks * chunk_len
        val_seqs    = make_test_sequences(val_seg_len)
        val_n_seqs  = hp.get('val_n_seqs')
        if val_n_seqs is not None:
            val_seqs = dict(list(val_seqs.items())[:val_n_seqs])

        stage_best_val = -1.0
        _cached_base_np = None  # (traj, raw sampled batch pre-argmax-feedback) — reused for `repeat_batch` steps
        pbar = tqdm(range(1, n_steps + 1), desc=f'stage{stage_i}', dynamic_ncols=True, file=status_file)
        for local_step in pbar:
            global_step += 1
            lr = _lr(local_step)
            for pg in opt.param_groups: pg['lr'] = lr

            model.train(); opt.zero_grad()

            if _cached_base_np is None or (local_step - 1) % repeat_batch == 0:
                traj = trajectories[rng.choice(len(trajectories), p=traj_weights)]
                t_pos_content, t_mask_t, t_tags, t_has_ir = (traj['pos_content'], traj['mask_t'],
                                                              traj['tags'], traj['has_ir'])
                tok_np = make_batch_tagged(rng, B, n_chunks, chunk_len, state_len, state_vocab_size,
                                           t_pos_content, t_tags, data_kind=data_kind,
                                           data_target_bits=data_target_bits)
                _cached_base_np = (traj, t_pos_content, t_mask_t, t_tags, t_has_ir, tok_np)
            else:
                traj, t_pos_content, t_mask_t, t_tags, t_has_ir, tok_np = _cached_base_np
            tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)

            # wrong_token_weight: upweight NLL at positions where the fed-back
            # argmax was wrong vs. ground truth, rather than diffusing gradient
            # equally over already-correct positions.
            wrong_masks: dict[int, np.ndarray] = {}
            if use_actual_am and t_has_ir:
                with torch.no_grad():
                    logits_1 = model(tok_t, t_mask_t)
                if wrong_token_weight > 0:
                    for i, rb in enumerate(t_pos_content['rec_blocks']):
                        if rb['type'] != 'refine':
                            continue
                        src_c0 = rb['argmax_src_c0']
                        wrong_masks[i] = tok_np[:, src_c0:src_c0 + rb['out_len']].copy()  # GT, pre-fill
                tok_np = _fill_argmax_fb(tok_np, logits_1, t_pos_content)
                if wrong_token_weight > 0:
                    for i, rb in enumerate(t_pos_content['rec_blocks']):
                        if i not in wrong_masks:
                            continue
                        wrong_masks[i] = (tok_np[:, rb['am0']:rb['am1']] != wrong_masks[i])
                tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)

            logits = model(tok_t, t_mask_t)
            nlls = []
            for i, rb in enumerate(t_pos_content['rec_blocks']):
                if not rb['is_clean']:
                    continue
                lp  = F.log_softmax(logits[:, rb['c0']-1:rb['c1']-1], dim=-1)
                tgt = tok_t[:, rb['c0']:rb['c1']]
                nll_per = _positional_ls_nll(lp, tgt, ls_max)
                if i in wrong_masks:
                    w = 1.0 + wrong_token_weight * wrong_masks[i].astype(np.float32)
                    nll_per = nll_per * torch.tensor(w, device=device, dtype=torch.float32)
                nlls.append(nll_per.mean())
            loss = torch.stack(nlls).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_f = float(loss.detach())
            pbar.set_postfix(loss=f'{loss_f:.3f}', lr=f'{lr:.1e}',
                             refine=traj['n_refine'], refresh=False)
            if local_step % log_every == 0:
                _jlog(dict(step=global_step, loss=round(loss_f, 5), lr=lr, n_refine=traj['n_refine']))
                print(str(pbar), file=log_file, flush=True)

            if local_step % stage_eval_every == 0 or local_step == n_steps:
                model.eval()
                elapsed = time.time() - t_start
                h, m = divmod(int(elapsed), 3600); m, s = divmod(m, 60)
                _log(f'\n--- stage={stage_i} step={local_step}/{n_steps}'
                     f'  g={global_step}  loss={loss_f:.4f}  {h:02d}:{m:02d}:{s:02d} ---')

                rw_rb = default_eval_traj['pos_content']['rec_blocks'][0]
                rw_valid_offsets = rw_rb['warmup_valid_offsets']
                window_means = []
                for X in rw_valid_offsets:
                    x_traj = eval_traj_by_x.get(X, default_eval_traj)
                    e_pos_content, e_mask_np, e_tags = (x_traj['pos_content'], x_traj['mask_np'],
                                                        x_traj['tags'])
                    ws = X // chunk_len
                    we = ws + window_chunks
                    seq_results = []
                    for sname, seq in val_seqs.items():
                        chunks_arr = np.array(
                            [seq[k*chunk_len:(k+1)*chunk_len] for k in range(n_chunks)], np.int64)
                        r = ar_decode_iq_global_rw_tagged(model, chunks_arr, state_len, state_vocab_size,
                                                          e_mask_np, e_pos_content, e_tags, device,
                                                          warmup_offset=X)
                        seq_results.append(r['match_pct'])
                        tpcts = r.get('turn_match_pcts', [])
                        n_turns = len(tpcts)
                        if n_turns > 1:
                            turn_names = ['initial'] + [f'refine{i}' for i in range(1, n_turns)]
                            turns_str = '  '.join(f'{tn}={p:.1f}%' for tn, p in zip(turn_names, tpcts))
                            _log(f'  val/win({ws},{we})_nc{n_chunks}/{sname:<15} BPB={r["bpb"]:.3f}  match={r["match_pct"]:.1f}%  [{turns_str}]')
                        else:
                            _log(f'  val/win({ws},{we})_nc{n_chunks}/{sname:<15} BPB={r["bpb"]:.3f}  match={r["match_pct"]:.1f}%')
                    win_mean = sum(seq_results) / len(seq_results)
                    window_means.append(win_mean)
                    _log(f'  val/win({ws},{we})_nc{n_chunks}/MEAN               match={win_mean:.1f}%')
                vmean = sum(window_means) / len(window_means)
                _log(f'  val/iq_global_rw_tagged/MEAN               match={vmean:.1f}%')
                _jlog(dict(step=global_step, eval_mean=round(vmean, 2)))

                torch.save(dict(model=model.state_dict(), hp=hp, step=global_step),
                          os.path.join(ckpt_dir, f'stage{stage_i}_last.pt'))
                if vmean > stage_best_val:
                    stage_best_val = vmean
                    torch.save(dict(model=model.state_dict(), hp=hp, step=global_step, val_mean=vmean),
                              os.path.join(ckpt_dir, f'stage{stage_i}_best.pt'))

        torch.save(dict(model=model.state_dict(), hp=hp, step=global_step),
                  os.path.join(ckpt_dir, f'stage{stage_i}_end.pt'))
        _log(f'[stage {stage_i}] done. saved stage{stage_i}_end.pt (best={stage_best_val:.1f}%)')


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
    train(hp, log_base=args.log_dir, device_str=args.device)


if __name__ == '__main__':
    main()
