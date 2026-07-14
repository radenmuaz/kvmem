"""
kvmem/hmn.py (DRAFT) — consolidated single-file rewrite of the HashMemNet (HMN)
chat-tags / dual-attn training stack.

This is a drafting pass per the approved plan:
  /Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md

It ports, into one self-contained file (no imports from the `kvmem` or
`experiments` packages):
  - chat-tags vocab constants               (experiments/chat_tags/vocab.py)
  - tag-aware position/mask-field builder    (experiments/chat_tags/positions.py)
  - the (L,L) attention-bias mask builder    (kvmem/train_hmn_chunk.py: chunk_mask_fb)
  - batch filling / AR-decode                (experiments/chat_tags/batch.py)
  - attention/norm/rope primitives           (kvmem/model.py)
  - a NEW unified model class supporting three block types (attn_mlp, dual_attn,
    single_attn) — attn_mlp mirrors kvmem/model.py's TransformerBlock, dual_attn
    mirrors experiments/attn_dual/model.py's DualAttnBlock, single_attn is the
    new default (one attn + one norm per block, no MLP, double depth vs dual_attn)
  - the traj_mix training loop                (experiments/chat_tags/train.py)

Masking/position math is ported verbatim — these encode subtle correctness
properties (Rule 3b nochain masking, tag-row leak prevention) that must not
change during this port.
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


# =============================================================================
# Vocab / tags
# ported from kvmem/data.py (base offset) + experiments/chat_tags/vocab.py
# =============================================================================

# HMN_VOCAB_SIZE = 268 — base kvmem vocab size (256 data bytes + 12 special
# tokens: HMN_MEM_START/END, HMN_STATE_0-3, HMN_DEL_START/END, HMN_DEL_SLOT_0-3).
# Hardcoded here (kvmem/data.py:2203) so this file has no dependency on the
# kvmem package.
HMN_VOCAB_SIZE = 268

# kvmem/data.py:2190-2195 — needed by _cyclic_state_ids below.
HMN_MEM_START  = 256
HMN_MEM_END    = 257
HMN_STATE_0    = 258
HMN_STATE_1    = 259
HMN_STATE_2    = 260
HMN_STATE_3    = 261

# kvmem/utils.py:5 — DATA_LO, needed by make_test_sequences below.
DATA_LO = 0x20   # legacy: data restricted to [0x20, 0xFF]

# Shared, generic tag vocabulary — reused identically at every chain step /
# round. No per-step or per-round variants: turn identity comes from position
# + accumulated content only, never from a turn-numbered vocab entry (see
# design-experiment-which-use-atomic-kay.md). <mem>/</mem> is dropped entirely
# (STATE-family regions are always filled with the fixed HMN_STATE_0..3
# placeholder tokens, which are already unambiguous region markers on their
# own — a wrapper tag would add zero information).
HMN_SRC_OPEN       = HMN_VOCAB_SIZE + 0   # 268  <src>
HMN_SRC_CLOSE      = HMN_VOCAB_SIZE + 1   # 269  </src>
HMN_QUERY_OPEN     = HMN_VOCAB_SIZE + 2   # 270  <query>       generic, reused at every chain step
HMN_QUERY_CLOSE    = HMN_VOCAB_SIZE + 3   # 271  </query>
HMN_RESPONSE_OPEN  = HMN_VOCAB_SIZE + 4   # 272  <response>
HMN_RESPONSE_CLOSE = HMN_VOCAB_SIZE + 5   # 273  </response>

# HMN_VOCAB_SIZE (268) + 2 tokens per surviving open/close pair
# (HMN_SRC, HMN_QUERY, HMN_RESPONSE = 3 pairs = 6 tokens) = 274.
HMN_TAG_VOCAB_SIZE = HMN_VOCAB_SIZE + 6    # 274


def _cyclic_state_ids(state_len: int, state_vocab_size: int = 2) -> list[int]:
    # ported from kvmem/train_hmn_chunk.py:63-64 (formerly _slot_ids)
    # Cyclic fill: when state_vocab_size < state_len, the alphabet of
    # HMN_STATE_0..state_vocab_size-1 repeats periodically to fill state_len
    # positions (e.g. state_len=8, state_vocab_size=2 ->
    # [STATE_0,STATE_1,STATE_0,STATE_1,STATE_0,STATE_1,STATE_0,STATE_1]).
    return [HMN_STATE_0 + (i % state_vocab_size) for i in range(state_len)]


# =============================================================================
# Position/mask-field builder (tag-aware iq_global_rw trajectory)
# ported from experiments/chat_tags/positions.py: chunk_positions_iq_global_rw_tagged
# =============================================================================

def chunk_positions_iq_global_rw_tagged(n_chunks: int, chunk_len: int, state_len: int,
                                        warmup_len: int, window_chunks: int = 2,
                                        warmup_x_fixed: int | None = None,
                                        warmup_x_dist: str = 'uniform',
                                        n_refine: int = 0) -> dict:
    """
    Returns dict(pos_content=..., pos_mask=..., tags=[(position, token_id), ...], L=...).

    Sequence (n_refine=0):
      per chunk k: <src> chunk_k </src> STATE
      round 0 (IQ): STATE <query> warmup </query> <response> out </response>
    Sequence (n_refine>0) additionally appends, per refine round:
      round k>0 (IR): STATE_A <response> argmax </response> STATE_B
                      <query> warmup </query> <response> out </response>

    Uses the same shared, generic <query>/<response> tag pair at every round —
    no per-window tag dispatch (see design-experiment-which-use-atomic-kay.md).
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

    # Single shared <query>/<response> tag pair, reused at every round — no
    # per-window tag lookup (this trajectory never had per-window tags beyond
    # the retired WINDOW_QUERY_TAGS scheme; kept generic per the plan).
    query_open, query_close = HMN_QUERY_OPEN, HMN_QUERY_CLOSE

    def _emit_round(round_idx: int):
        """round_idx == 0: STATE + <query>/<response>, no argmax segment.
        round_idx > 0: STATE_A + argmax + STATE_B + <query>/<response>
        (today's IR block). STATE-family regions are bare (no wrapper tag) —
        content-dict and mask-dict field boundaries are identical for them."""
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

    iq_c, iq_m = _emit_round(0)
    rw_extra = dict(warmup_train_range=train_range, warmup_x_dist=_dist,
                    warmup_valid_offsets=eval_offsets, window_chunks=window_chunks)
    rec_blocks_c = [dict(type='iq', span=(0, n_chunks), span_len=src_len,
                         out_len=out_len, is_clean=(n_refine == 0), **iq_c, **rw_extra)]
    rec_blocks_m = [dict(type='iq', **iq_m)]

    prev_c0_c = iq_c['c0']
    for _ in range(n_refine):
        ir_c, ir_m = _emit_round(1)
        rec_blocks_c.append(dict(type='ir', span=(0, n_chunks), span_len=src_len,
                                 out_len=out_len, is_clean=True,
                                 argmax_src_c0=prev_c0_c, **ir_c, **rw_extra))
        rec_blocks_m.append(dict(type='ir', **ir_m))
        prev_c0_c = ir_c['c0']

    L = offset

    pos_content = dict(enc_blocks=enc_blocks_c, rec_blocks=rec_blocks_c,
                       enc_end=enc_end, warmup_len=warmup_len, L=L)
    pos_mask = dict(enc_blocks=enc_blocks_m, rec_blocks=rec_blocks_m,
                    enc_end=enc_end, warmup_len=warmup_len, L=L)

    return dict(pos_content=pos_content, pos_mask=pos_mask, tags=tags, L=L)


def chunk_positions_chained(n_chunks: int, chunk_len: int, state_len: int,
                            warmup_len: int, chain_steps: list[tuple[int, int]],
                            n_refine: int = 2, state_vocab_size: int = 2) -> dict:
    """
    Chained multi-chain-step schedule: each span in `chain_steps` (e.g.
    [(0,2),(1,3),(2,4)]) gets its OWN local round-0 (IQ) turn + n_refine
    chained argmax-IR rounds, reusing the shared, generic <query>/<response>
    tag pair at every chain step (no per-chain-step tag dispatch — turn
    identity comes from position, not a turn-numbered vocab entry).

    Chain step i > 0 additionally gets a STATE_QUEUE_in region (width
    state_len, M=1) immediately before its round-0 STATE region, populated at
    train/eval time via h_inject from chain step i-1's last round's own STATE
    slice (see train()'s chain-step dispatch). Reuses the same HMN_STATE_0..3
    placeholder tokens as any other STATE region — position/mask disambiguate
    the regions, and h_inject overwrites the embedding before any block runs,
    so the placeholder identity is irrelevant.

    Structurally this is chunk_positions_fb_localrefine (kvmem/train_hmn_chunk.py)
    with tags added — same "one shared encoding pass, then each span threaded in
    sequence with its own local round-0+IR" shape, same reliance on
    chunk_mask_fb's Rule 3b (round-0 STATE blocked from ALL tokens in prior
    rec_blocks) for the nochain/no-cross-chain-step-leak property, reused
    unmodified. The *only* channel for cross-chain-step information is the
    injected STATE_QUEUE_in feature vector, never an attention path.

    STATE_QUEUE is a single-hop relay, not an accumulating buffer — M=1 means
    chain step i's STATE_QUEUE_in comes ONLY from chain step i-1's own last
    round's STATE, never from i-2 or earlier directly. There is no separate
    "older states" store to mask or discard: raw content from chain steps
    older than i-1 is already fully blocked by Rule 3b regardless of
    STATE_QUEUE (that invariant predates this mechanism and applies to every
    chain step, chained or not). For information from chain step i-2 to reach
    chain step i, chain step i-1 must have implicitly folded it into its own
    single state_len-wide STATE when producing its own output — there is no
    guarantee this happens; it's exactly what the deferred "recover an
    earlier chain step's span from the last chain step's round-0 recall"
    validation probe is designed to test. Per-chain-step recall accuracy
    alone does NOT test this — each chain step can solve its own span from
    its own encoding-block STATEs without STATE_QUEUE carrying anything
    useful across the relay at all.
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

    # Single shared <query>/<response> tag pair, reused identically at every
    # chain step (no per-chain-step lookup — see module docstring / plan).
    query_open, query_close = HMN_QUERY_OPEN, HMN_QUERY_CLOSE

    for chain_step_i, span in enumerate(chain_steps):
        span_s, span_e = span
        span_len = (span_e - span_s) * chunk_len
        out_len  = span_len - warmup_len
        has_queue_in = chain_step_i > 0

        def _emit_round(round_idx: int, has_queue_in: bool):
            """round_idx == 0: [STATE_QUEUE_in if has_queue_in] + STATE +
            <query>/<response>, no argmax segment.
            round_idx > 0: STATE_A + argmax + STATE_B + <query>/<response>
            (today's IR block). STATE-family regions (STATE, STATE_A,
            STATE_B, STATE_QUEUE_in) are bare (no wrapper tag) — content-dict
            and mask-dict field boundaries are identical for them, no more
            +/-1 tag-absorption widening."""
            nonlocal offset
            if round_idx == 0:
                q0 = q1 = None
                if has_queue_in:
                    q0 = offset; q1 = q0 + state_len; offset = q1
                sl0 = offset; sl1 = sl0 + state_len; offset = sl1
                tags.append((offset, query_open)); offset += 1
                w0 = offset; w1 = w0 + warmup_len; offset = w1
                tags.append((offset, query_close)); offset += 1
                tags.append((offset, HMN_RESPONSE_OPEN)); offset += 1
                c0 = offset; c1 = c0 + out_len; offset = c1
                tags.append((offset, HMN_RESPONSE_CLOSE)); offset += 1
                c_fields = dict(sl0=sl0, sl1=sl1, w0=w0, w1=w1, c0=c0, c1=c1)
                m_fields = dict(sl0=sl0, sl1=sl1, w0=w0 - 1, w1=w1 + 1, c0=c0 - 1, c1=c1 + 1)
                if has_queue_in:
                    c_fields.update(queue0=q0, queue1=q1)
                    m_fields.update(queue0=q0, queue1=q1)
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

        iq_c, iq_m = _emit_round(0, has_queue_in)
        # Chained chain steps have no random warmup offset (warmup is always
        # the chain step's own start byte, X=0 within the span) —
        # make_batch_tagged/ar_decode_iq_global_rw_tagged (reused unmodified
        # from chat_tags) expect this field on every round-0 block, so give
        # it a degenerate fixed (0,0) range rather than touching that shared
        # code.
        rec_blocks_c.append(dict(type='iq', span=span, span_len=span_len,
                                 out_len=out_len, is_clean=(n_refine == 0),
                                 warmup_train_range=(0, 0), warmup_x_dist='fixed', **iq_c))
        rec_blocks_m.append(dict(type='iq', **iq_m))

        prev_c0_c = iq_c['c0']
        for _ in range(n_refine):
            ir_c, ir_m = _emit_round(1, False)
            rec_blocks_c.append(dict(type='ir', span=span, span_len=span_len,
                                     out_len=out_len, is_clean=True,
                                     argmax_src_c0=prev_c0_c, **ir_c))
            rec_blocks_m.append(dict(type='ir', **ir_m))
            prev_c0_c = ir_c['c0']

    L = offset

    pos_content = dict(enc_blocks=enc_blocks_c, rec_blocks=rec_blocks_c,
                       enc_end=enc_end, warmup_len=warmup_len, L=L)
    pos_mask = dict(enc_blocks=enc_blocks_m, rec_blocks=rec_blocks_m,
                    enc_end=enc_end, warmup_len=warmup_len, L=L)

    return dict(pos_content=pos_content, pos_mask=pos_mask, tags=tags, L=L)


# =============================================================================
# Attention mask construction (feedback-argmax IR layout)
# ported from kvmem/train_hmn_chunk.py: chunk_mask_fb (lines 583-688)
# =============================================================================

def chunk_mask_fb(pos: dict) -> np.ndarray:
    """
    Mask for feedback-argmax IR layout. Same rules as chunk_mask for encoding
    blocks and round-0 (IQ) turns. Additional rules for IR rounds:

    5. STATE_A rows: blocked from all chunks (like all recall STATE fields).
    6. argmax rows: blocked from all chunks.
    7. STATE_B rows: blocked from all chunks; sees STATE_A + argmax causally.
    8. IR warmup/out rows: blocked from everything except own STATE_B + own warmup/out.
       (Same strong bottleneck as round-0 out rows, but STATE_B is the gate — not STATE_A or argmax.)

    Rule 3b (always on): Each round-0 STATE row (and its STATE_QUEUE_in, if
    any) is blocked from ALL tokens in prior rec_blocks (STATE, warmup, argmax,
    AND output of earlier chain steps' round-0 and IR turns). Forces every
    chain step to encode independently from enc-block STATEs (+ its own
    injected STATE_QUEUE_in) only. Without this, the model chains through
    prior OUTPUT tokens — chain step 1 reads chain step 0's recalled bytes in
    the 50% overlap region. Blocking only prior STATEs is insufficient
    (v4 lesson).
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

    # Union of every rec_block's own output region (c0:c1) — IR turns must
    # reach earlier turns' output ONLY via their explicit argmax copy
    # (am0:am1), never by attending straight to the raw c0:c1 tokens sitting
    # in context. Those tokens are ground truth during training (teacher-
    # forced) but the model's own greedy decode at eval time — a direct
    # attention path there lets training "cheat" via leaked ground truth,
    # which collapses at AR-decode eval once that region is no longer GT.
    is_any_rec_output = np.zeros(L, dtype=bool)
    for rb2 in rec_blocks:
        is_any_rec_output |= (c >= rb2['c0']) & (c < rb2['c1'])

    # Rule 2: encoding STATE_k blocked from chunk_j (j≠k)
    for k, b in enumerate(enc_blocks):
        sl_row = (r >= b['sl0']) & (r < b['sl1'])
        for j, bj in enumerate(enc_blocks):
            if j == k: continue
            blocked |= sl_row[:, None] & ((c >= bj['s0']) & (c < bj['s1']))[None, :]

    for i_rb, rb in enumerate(rec_blocks):
        if rb['type'] == 'iq':
            has_queue_in = 'queue0' in rb
            # Rule 3 (round-0 STATE): blocked from all chunks
            sl_row = (r >= rb['sl0']) & (r < rb['sl1'])
            blocked |= sl_row[:, None] & is_any_chunk[None, :]
            # Rule 3b: round-0 STATE blocked from ALL tokens in prior rec_blocks
            # (STATE + warmup + argmax + output). Blocking only STATEs is insufficient —
            # the model chains through prior OUTPUT tokens (chain step 1 reads chain step
            # 0's recalled bytes in the 50% overlap). Full blackout forces every chain
            # step to encode from enc-block STATEs (+ its own STATE_QUEUE_in, the only
            # sanctioned cross-chain-step channel) only.
            prior_all = np.zeros(L, dtype=bool)
            for prev_rb in rec_blocks[:i_rb]:
                if prev_rb['type'] == 'iq':
                    prior_all |= (c >= prev_rb['sl0']) & (c < prev_rb['sl1'])
                    prior_all |= (c >= prev_rb['w0'])  & (c < prev_rb['w1'])
                    prior_all |= (c >= prev_rb['c0'])  & (c < prev_rb['c1'])
                    if 'queue0' in prev_rb:
                        prior_all |= (c >= prev_rb['queue0']) & (c < prev_rb['queue1'])
                else:
                    prior_all |= (c >= prev_rb['sla0']) & (c < prev_rb['sla1'])
                    prior_all |= (c >= prev_rb['am0'])  & (c < prev_rb['am1'])
                    prior_all |= (c >= prev_rb['slb0']) & (c < prev_rb['slb1'])
                    prior_all |= (c >= prev_rb['w0'])   & (c < prev_rb['w1'])
                    prior_all |= (c >= prev_rb['c0'])   & (c < prev_rb['c1'])
            blocked |= sl_row[:, None] & prior_all[None, :]
            if has_queue_in:
                # STATE_QUEUE_in row: same nochain treatment as round-0 STATE
                # itself — blocked from all chunks and all prior rec_blocks'
                # tokens. Its only content comes from h_inject, not attention.
                q_row = (r >= rb['queue0']) & (r < rb['queue1'])
                blocked |= q_row[:, None] & is_any_chunk[None, :]
                blocked |= q_row[:, None] & prior_all[None, :]
            # Rule 4a: round-0 warmup rows — own STATE_QUEUE_in (if any) + own STATE + own warmup only
            if rb['w0'] < rb['w1']:
                wm_row = (r >= rb['w0']) & (r < rb['w1'])
                own = (c >= rb['sl0']) & (c < rb['sl1']) | (c >= rb['w0']) & (c < rb['w1'])
                if has_queue_in:
                    own = own | (c >= rb['queue0']) & (c < rb['queue1'])
                blocked |= wm_row[:, None] & ~own[None, :]
            # Rule 4b: round-0 out rows — own STATE_QUEUE_in (if any) + own STATE + own warmup + own output
            out_row = (r >= rb['c0']) & (r < rb['c1'])
            own = ((c >= rb['sl0']) & (c < rb['sl1']) |
                   (c >= rb['w0'])  & (c < rb['w1'])  |
                   (c >= rb['c0'])  & (c < rb['c1']))
            if has_queue_in:
                own = own | (c >= rb['queue0']) & (c < rb['queue1'])
            blocked |= out_row[:, None] & ~own[None, :]

        else:  # type == 'ir'
            # Rules 5,6,7: STATE_A, argmax, STATE_B — blocked from encoding chunks
            # AND from every rec_block's own raw output region (own am0:am1
            # copy is the only sanctioned path back to an earlier turn's
            # output — see is_any_rec_output comment above).
            sla_row = (r >= rb['sla0']) & (r < rb['sla1'])
            am_row  = (r >= rb['am0'])  & (r < rb['am1'])
            slb_row = (r >= rb['slb0']) & (r < rb['slb1'])
            blocked |= (sla_row | am_row | slb_row)[:, None] & (is_any_chunk | is_any_rec_output)[None, :]

            # Rule 8: IR warmup/out rows — only own STATE_B + own warmup + own output
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
# ported from experiments/chat_tags/batch.py (make_batch_tagged,
# ar_decode_iq_global_rw_tagged) and kvmem/train_hmn_chunk.py
# (_fill_argmax_fb, _cat_kv — dependencies of the above, reused unmodified)
# =============================================================================

def _fill_argmax_fb(tok_np: np.ndarray, logits: torch.Tensor,
                    pos: dict) -> np.ndarray:
    # ported from kvmem/train_hmn_chunk.py:833-853
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


def _cat_kv(kv_a: list, kv_b: list) -> list:
    # ported from kvmem/train_hmn_chunk.py:971-974
    """Concatenate two layer-wise KV caches along the sequence dim (dim=2)."""
    return [(torch.cat([ka, kb], dim=2), torch.cat([va, vb], dim=2))
            for (ka, va), (kb, vb) in zip(kv_a, kv_b)]


def make_batch_tagged(rng: np.random.Generator, B: int, n_chunks: int, chunk_len: int,
                      state_len: int, state_vocab_size: int, pos_content: dict,
                      tags: list[tuple[int, int]]) -> np.ndarray:
    sids = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
    wl = pos_content['warmup_len']
    L = pos_content['L']
    tok = np.zeros((B, L), dtype=np.int64)
    segs = rng.integers(0, 256, size=(B, n_chunks, chunk_len), dtype=np.int64)

    for k, b in enumerate(pos_content['enc_blocks']):
        tok[:, b['s0']:b['s1']] = segs[:, k, :]
        tok[:, b['sl0']:b['sl1']] = sids

    rw_xs: np.ndarray | None = None
    for rb in pos_content['rec_blocks']:
        span_s, span_e = rb['span']
        gt = np.concatenate([segs[:, i, :] for i in range(span_s, span_e)], axis=1)

        if rb['type'] == 'iq':
            tok[:, rb['sl0']:rb['sl1']] = sids
            if 'queue0' in rb:
                # STATE_QUEUE_in placeholder fill — content is overridden by
                # h_inject in train()'s chained dispatch; the token identity
                # here is only there to give the region a well-defined shape
                # before injection (see chunk_positions_chained docstring).
                tok[:, rb['queue0']:rb['queue1']] = sids
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
        else:  # 'ir'
            tok[:, rb['sla0']:rb['sla1']] = sids
            tok[:, rb['slb0']:rb['slb1']] = sids
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


@torch.no_grad()
def ar_decode_iq_global_rw_tagged(model, chunks_arr, state_len: int, state_vocab_size: int,
                                  mask_np: np.ndarray, pos_content: dict,
                                  tags: list[tuple[int, int]], device,
                                  warmup_offset: int = 0) -> dict:
    """
    KV-cached greedy AR decode for the tagged iq_global_rw layout. Adapted from
    ar_decode_chunk_fb_kv (kvmem/train_hmn_chunk.py:1218) — same segment-by-
    segment caching strategy, but content is written via pos_content while the
    (tag-widened) attention mask comes from pos_mask via chunk_mask_fb, passed
    in pre-built as mask_np.
    """
    if isinstance(chunks_arr, np.ndarray) and chunks_arr.ndim == 2:
        chunks_list = [chunks_arr[k] for k in range(chunks_arr.shape[0])]
    else:
        chunks_list = list(chunks_arr)

    wl = pos_content['warmup_len']
    sids = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
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

        if rb['type'] == 'iq':
            tok[rb['sl0']:rb['sl1']] = sids
            if wl > 0:
                tok[rb['w0']:rb['w1']] = gt_span[warmup_offset:warmup_offset + wl].astype(np.int64)
            # seg_start must equal L_cached exactly (KV-cache invariant, see
            # CLAUDE.md "KV decode off-by-one") — this sweeps in every tag
            # token between the last cached position and c0 automatically,
            # regardless of how many tags precede this block's content.
            _decode_segment(L_cached, rb)
        else:  # 'ir'
            tok[rb['sla0']:rb['sla1']] = sids
            src_c0 = rb['argmax_src_c0']
            tok[rb['am0']:rb['am1']] = tok[src_c0:src_c0 + rb['out_len']]
            tok[rb['slb0']:rb['slb1']] = sids
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
    Full-recompute (no KV cache) AR decode for DualAttnModel on the tagged
    stitched SRS layout. Mirrors experiments/srs_tagged/stitch_decode.py's
    chaining logic exactly (only window A's warmup from GT, later windows chained
    from the model's own decoded output) but re-runs a full forward pass for
    every generated byte instead of using a KV cache — DualAttnBlock has two
    attention sublayers per layer, which doesn't fit the single-KV-pair-per-layer
    cache format the rest of this project's decode functions assume. Fine at this
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

        if rb['type'] == 'iq':
            tok[rb['sl0']:rb['sl1']] = sids
            if wl > 0:
                tok[rb['w0']:rb['w1']] = warmup_src
            _decode_segment(rb)
        else:  # 'ir'
            tok[rb['sla0']:rb['sla1']] = sids
            src_c0 = rb['argmax_src_c0']
            tok[rb['am0']:rb['am1']] = tok[src_c0:src_c0 + rb['out_len']]
            tok[rb['slb0']:rb['slb1']] = sids
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


# =============================================================================
# Attention / norm / RoPE primitives
# ported from kvmem/model.py verbatim
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
        """
        null_kv=True: append a learnable (null_k, null_v) pair to the KV sequence
        before softmax. Gives each query a "blank slot" to attend to when no real
        token is relevant — soft gating without hard masking.

        null_k is initialised to zero so Q·null_k = 0 initially (score=0 before
        scaling), but it is learned and can diverge. null_v is also learnable —
        the model decides what to emit when attending to nothing.

        qk_norm=True: RMS-normalize Q and K along d_head (with a learned
        per-dim scale) before RoPE/dot-product — stabilizes attention logit
        scale early in training, ablation flag.
        """
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
        # null_kv uses fixed zero K and V — no learned parameters.
        # Q·null_k = 0 always, giving a fixed-score "abstain" option in softmax.
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
    # ported from experiments/attn_dual/model.py:44-45
    return x + attn(norm(x), mask)


# =============================================================================
# Model — three selectable block types on one unified model class
# NEW code per the plan (block_type hp), following:
#   attn_mlp    mirrors kvmem/model.py's TransformerBlock
#   dual_attn   mirrors experiments/attn_dual/model.py's DualAttnBlock/DualAttnModel
#   single_attn NEW — one attn + one norm per block, no MLP (the new default)
# Embedding/init/output-head logic follows KVMemModel / DualAttnModel (both are
# equivalent in that regard) generalized to the 3-way block_type switch.
# =============================================================================

class AttnMlpBlock(nn.Module):
    """block_type='attn_mlp' — mirrors kvmem/model.py's TransformerBlock:
    x = x + attn(norm1(x)); x = x + ffn(norm2(x))"""
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
    """block_type='dual_attn' — mirrors experiments/attn_dual/model.py's
    DualAttnBlock: x = x + attn1(norm1(x)); x = x + attn2(norm2(x)), no FFN.
    No KV-cache support (two attn sublayers per block breaks the single-
    KV-pair-per-layer assumption used elsewhere) — matches the original."""
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
    """block_type='single_attn' — NEW default. One attn + one norm per block,
    no FFN: x = x + attn(norm(x)). Use n_layers = 2x the equivalent dual_attn
    config's n_layers to match total attention-op count (the "flatten
    dual-attn, double depth" design)."""
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
    {'attn_mlp', 'dual_attn', 'single_attn'}. Embedding/init/output-head logic
    follows kvmem/model.py's KVMemModel and experiments/attn_dual/model.py's
    DualAttnModel (the two are equivalent in this regard).

    KV-cache (past_kv/return_kv/offset) is only meaningful for attn_mlp and
    single_attn blocks (one attn per block). dual_attn blocks do not support
    it (two KV pairs per layer) — forward() only passes past_kv/return_kv/
    offset through when block_type != 'dual_attn', matching DualAttnModel's
    original no-KV-cache forward signature.
    """
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
                return_features: bool = False,
                h_inject: dict | None = None) -> torch.Tensor | tuple:
        """
        tokens          : (B, L) or (L,) int64
        mask            : (L_q, L_kv) — L_kv = L_past + L when past_kv given
        past_kv         : list[n_layers] of (K_past, V_past) — cached prefix KV.
                          Only supported for block_type in ('attn_mlp', 'single_attn').
        return_kv       : return (logits, kv_list) instead of just logits
        return_features : return (logits, x) where x is the pre-head residual stream
                          (B, L, d); disables grad_checkpoint to preserve full graph.
        offset          : RoPE position offset (= L_past for suffix pass)
        h_inject        : dict mapping (sl0, sl1) → (B, state_len, d) tensor.
                          Overrides the embedding at x[:, sl0:sl1, :] after _embed()
                          but before transformer blocks.

        grad_checkpoint: for attn_mlp/single_attn, True checkpoints each block
        during backward (depth-only), same semantics as KVMemModel. For
        dual_attn, may be None | 'block' | 'attn' matching DualAttnModel's two
        granularities (whole-block vs per-attn-sublayer checkpointing).
        """
        batched = tokens.dim() == 2
        if not batched:
            tokens = tokens.unsqueeze(0)
        x      = self._embed(tokens)
        if h_inject is not None:
            for (sl0, sl1), h_val in h_inject.items():
                x = x.clone()  # avoid in-place autograd error
                x[:, sl0:sl1, :] = h_val

        if self.block_type == 'dual_attn':
            # No KV-cache support — full pass only, matches DualAttnModel.
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

        # attn_mlp / single_attn: KV-cache-capable path, mirrors KVMemModel.
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
    """Factory for HMNModel. hp keys mirror kvmem/model.py's build_model plus
    the new 'block_type' selector ('attn_mlp' | 'dual_attn' | 'single_attn',
    default 'single_attn' — the new default per the plan)."""
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
    # Chunked attention: set chunk_attn on all attention layers.
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
# ported from kvmem/train_hmn_mono.py (_positional_ls_nll, load_config),
# kvmem/utils.py (make_test_sequences), kvmem/train_hmn_chunk.py
# (_StatusWriter, load_chunks_padded)
# =============================================================================

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
# Training loop with traj_mix
# ported from experiments/chat_tags/train.py: train()
# Adapted call site: build_model/hp_model now passes block_type through (the
# CALLER supplies block_type and an already-doubled n_layers via config when
# using single_attn — this function just forwards whatever hp has).
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
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=wd, betas=(0.9, 0.999))

    curriculum = hp.get('curriculum', [])
    assert curriculum
    log_every  = hp.get('log_every', 500)

    global_step = 0
    t_start = time.time()

    for stage_i, stage in enumerate(curriculum):
        if 'chain_steps' in stage:
            # NEW path: true chained multi-chain-step training (ports
            # experiments/attn_dual/train.py's single-trajectory loop verbatim).
            # Structurally different from the traj_mix branch below: ONE packed
            # pos/mask built from a fixed `chain_steps` list (one rec_block per
            # chain step, each with its own n_refine IR rounds) instead of many
            # small per-trajectory sequences sampled by weight each step.
            #
            # stage['chain']=True switches to the STATE_QUEUE-chained sequential
            # dispatch (one forward pass per chain step, h_inject-linked) —
            # see the `if is_chained:` branch below. stage['chain'] absent/False
            # keeps today's fast single-packed-sequence path (2 forward passes
            # total regardless of chain-step count), valid only when no chain
            # step reads a STATE_QUEUE_in (i.e. no cross-chain-step dependency).
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
            is_chained = bool(stage.get('chain', False))

            built = chunk_positions_chained(n_chunks, chunk_len, state_len, warmup_len,
                                            chain_steps, n_refine=n_refine,
                                            state_vocab_size=state_vocab_size)
            pos_content, pos_mask, tags, L = (built['pos_content'], built['pos_mask'],
                                              built['tags'], built['L'])
            mask_np = chunk_mask_fb(pos_mask)
            mask_t  = torch.tensor(mask_np, dtype=torch.float32, device=device)

            _log(f'\n[stage {stage_i}] n_chunks={n_chunks} chunk_len={chunk_len} state={state_len} '
                 f'wl={warmup_len} chain_steps={chain_steps} n_refine={n_refine} chain={is_chained} '
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

            # Per-chain-step rec_block index groups, in schedule order — used
            # by both the fast packed path and the chained h_inject path to
            # slice out "this chain step's own rec_blocks" and (for chained)
            # "this chain step's last round's own STATE field" for
            # STATE_QUEUE_out extraction.
            chain_step_rb_idxs: list[list[int]] = [[] for _ in chain_steps]
            span_to_chain_idx = {span: i for i, span in enumerate(chain_steps)}
            for i, rb in enumerate(pos_content['rec_blocks']):
                chain_step_rb_idxs[span_to_chain_idx[rb['span']]].append(i)

            stage_best_val = -1.0
            pbar = tqdm(range(1, n_steps + 1), desc=f'stage{stage_i}', dynamic_ncols=True, file=status_file)
            for local_step in pbar:
                global_step += 1
                lr = _lr(local_step)
                for pg in opt.param_groups: pg['lr'] = lr

                model.train(); opt.zero_grad()

                tok_np = make_batch_tagged(rng, B, n_chunks, chunk_len, state_len, state_vocab_size,
                                           pos_content, tags)
                tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)

                wrong_masks: dict[int, np.ndarray] = {}

                if not is_chained:
                    # Fast path: no STATE_QUEUE_in anywhere in this schedule (or
                    # chain=False was set deliberately) — one shared mask, up to
                    # 2 forward passes total for the whole packed sequence,
                    # exactly as before the STATE_QUEUE mechanism existed.
                    if use_actual_am:
                        with torch.no_grad():
                            logits_1 = model(tok_t, mask_t)
                        if wrong_token_weight > 0:
                            for i, rb in enumerate(pos_content['rec_blocks']):
                                if rb['type'] != 'ir':
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
                else:
                    # Chained path: chain step i's input genuinely depends on
                    # chain step i-1's computed hidden state (via h_inject), not
                    # just its decoded byte output — so this needs one
                    # sequential forward pass per chain step. Each chain step's
                    # own IR rounds (if any) still use the existing 2-pass
                    # argmax-feedback trick, applied to the FULL packed sequence
                    # each time (mask_t already encodes Rule 3b nochain across
                    # chain steps, so re-running the full sequence per chain
                    # step is safe/idempotent — only the current chain step's
                    # argmax positions are filled per its own pass). The logits
                    # used for THIS chain step's loss are always the ones from
                    # ITS OWN final pass (recorded per rec_block index in
                    # logits_by_rb), not some stitched full-sequence tensor.
                    h_inject: dict[tuple, torch.Tensor] = {}
                    logits_by_rb: dict[int, torch.Tensor] = {}
                    for ci, rb_idxs in enumerate(chain_step_rb_idxs):
                        if use_actual_am and any(pos_content['rec_blocks'][i]['type'] == 'ir' for i in rb_idxs):
                            with torch.no_grad():
                                logits_1, _ = model(tok_t, mask_t, h_inject=h_inject or None, return_features=True)
                            for i in rb_idxs:
                                rb = pos_content['rec_blocks'][i]
                                if rb['type'] != 'ir':
                                    continue
                                src_c0 = rb['argmax_src_c0']
                                if wrong_token_weight > 0:
                                    wrong_masks[i] = tok_np[:, src_c0:src_c0 + rb['out_len']].copy()
                                am = logits_1[:, src_c0-1:src_c0-1+rb['out_len']].argmax(-1).cpu().numpy()
                                tok_np[:, rb['am0']:rb['am1']] = am
                                if wrong_token_weight > 0:
                                    wrong_masks[i] = (tok_np[:, rb['am0']:rb['am1']] != wrong_masks[i])
                            tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)

                        logits_ci, x_ci = model(tok_t, mask_t, h_inject=h_inject or None, return_features=True)
                        for i in rb_idxs:
                            logits_by_rb[i] = logits_ci

                        # STATE_QUEUE_out = this chain step's last round's own
                        # STATE field (round 0's sl0/sl1 if n_refine==0, else the
                        # final IR round's slb0/slb1) — feed into the NEXT chain
                        # step's STATE_QUEUE_in, if any.
                        last_rb = pos_content['rec_blocks'][rb_idxs[-1]]
                        out_sl0, out_sl1 = (last_rb['sl0'], last_rb['sl1']) if last_rb['type'] == 'iq' \
                            else (last_rb['slb0'], last_rb['slb1'])
                        if ci + 1 < len(chain_step_rb_idxs):
                            next_rb0 = pos_content['rec_blocks'][chain_step_rb_idxs[ci + 1][0]]
                            if 'queue0' in next_rb0:
                                h_inject = {(next_rb0['queue0'], next_rb0['queue1']):
                                            x_ci[:, out_sl0:out_sl1, :].detach()}
                            else:
                                h_inject = {}

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
        # Falls back to a single traj built from stage['n_refine'] (Phase A style).
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

        # Eval trajectory selection: highest n_refine (matches project convention
        # — "Eval uses first IR trajectory" — full IQ+IR chain is the most
        # informative report). With window-specific query tags, a trajectory's
        # tags encode which window it was built for (via warmup_x_fixed), so
        # evaluating window X with a trajectory built for a DIFFERENT window's
        # warmup_x_fixed would silently use the wrong <query_a/b/c> tag. Build
        # one eval trajectory PER canonical warmup offset (falling back to the
        # single highest-n_refine trajectory if none match a given X exactly,
        # e.g. n_refine=0-only stages or uniform-only traj_mix).
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
        pbar = tqdm(range(1, n_steps + 1), desc=f'stage{stage_i}', dynamic_ncols=True, file=status_file)
        for local_step in pbar:
            global_step += 1
            lr = _lr(local_step)
            for pg in opt.param_groups: pg['lr'] = lr

            model.train(); opt.zero_grad()

            traj = trajectories[rng.choice(len(trajectories), p=traj_weights)]
            t_pos_content, t_mask_t, t_tags, t_has_ir = (traj['pos_content'], traj['mask_t'],
                                                          traj['tags'], traj['has_ir'])

            tok_np = make_batch_tagged(rng, B, n_chunks, chunk_len, state_len, state_vocab_size,
                                       t_pos_content, t_tags)
            tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)

            # wrong_token_weight ablation: capture, per IR block, whether the fed-back
            # argmax at each position was wrong (vs the ground truth that was there
            # pre-fill) — used to upweight NLL specifically at positions that need
            # active correction, rather than diffusing gradient equally over positions
            # the model already had right. See docs/FEEDBACK_RESULTS.md § IR-refinement
            # loss redesign, ablation 1.
            wrong_masks: dict[int, np.ndarray] = {}
            if use_actual_am and t_has_ir:
                with torch.no_grad():
                    logits_1 = model(tok_t, t_mask_t)
                if wrong_token_weight > 0:
                    for i, rb in enumerate(t_pos_content['rec_blocks']):
                        if rb['type'] != 'ir':
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
                            turn_names = ['IQ'] + [f'IR{i}' for i in range(1, n_turns)]
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
