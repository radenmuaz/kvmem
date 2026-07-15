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
properties (the nochain blackout, tag-row leak prevention) that must not
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

from kvmem.structured_data import generate_structured_chunks


# =============================================================================
# Vocab / tags
#
# LAYOUT (vocab reorder — supersedes the original kvmem/data.py-ported
# ordering, which put STATE ids BEFORE the chat tags with 6 dead legacy
# padding slots sandwiched between them; that made STATE growth beyond 10
# awkward — see git history / docs/HISTORY.md for the pre-reorder
# scheme). Chat tags now sit immediately after the 256 data bytes — fixed,
# small (3 pairs, 6 ids), never expected to grow. STATE placeholders occupy
# the TAIL of the vocab — the only region ever expected to grow
# (state_vocab_size) — so growth is ALWAYS a pure tail-append, the only mode
# train()'s pretrained-checkpoint loader supports. No two-tier alphabet,
# no collision risk to guard against, ever again.
# =============================================================================

DATA_LO = 0x20   # legacy: data restricted to [0x20, 0xFF]

# Shared, generic tag vocabulary — reused identically at every chain step /
# round. No per-step or per-round variants: turn identity comes from position
# + accumulated content only, never from a turn-numbered vocab entry (see
# design-experiment-which-use-atomic-kay.md). <mem>/</mem> is dropped entirely
# (STATE-family regions are always filled with the fixed HMN_STATE_0..N-1
# placeholder tokens, which are already unambiguous region markers on their
# own — a wrapper tag would add zero information).
HMN_SRC_OPEN       = 256   # <src>
HMN_SRC_CLOSE      = 257   # </src>
HMN_QUERY_OPEN     = 258   # <query>       generic, reused at every chain step
HMN_QUERY_CLOSE    = 259   # </query>
HMN_RESPONSE_OPEN  = 260   # <response>
HMN_RESPONSE_CLOSE = 261   # </response>

# First STATE placeholder id. Everything from here to the end of the vocab
# (hp['V']) is STATE alphabet — pure tail region, no upper neighbor to
# collide with, so state_vocab_size is bounded only by hp['V'] itself.
HMN_STATE_0 = 262

# Default vocab size: 256 bytes + 6 chat tags + 12 reserved STATE ids
# (state_vocab_size <= 12 is free — one more than the pre-reorder scheme's
# 10, and V=274 stays numerically identical to the pre-reorder default for
# an apples-to-apples parameter-count comparison).
HMN_TAG_VOCAB_SIZE = 274


def _cyclic_state_ids(state_len: int, state_vocab_size: int = 2, family: int = 0) -> list[int]:
    # ported from kvmem/train_hmn_chunk.py:63-64 (formerly _slot_ids)
    # Cyclic fill: alphabet HMN_STATE_0..HMN_STATE_0+state_vocab_size-1
    # repeats periodically to fill state_len positions (e.g. state_len=8,
    # state_vocab_size=2 -> [STATE_0,STATE_1,STATE_0,STATE_1,STATE_0,
    # STATE_1,STATE_0,STATE_1]). STATE sits at the TAIL of the vocab (see
    # module docstring above) so there's no collision to guard against —
    # growing state_vocab_size just requires hp['V'] >= HMN_STATE_0 +
    # state_vocab_size, which the pretrained-checkpoint tail-append growth
    # logic in train() already handles.
    #
    # `family`: which same-size block of the tail to draw from — family 0 is
    # the regular alphabet (encoding state, round-0 state, a refine round's
    # own first `state` register); family 1 is the dedicated `feedback_state`
    # alphabet (see HMN_FEEDBACK_STATE_FAMILY below and _emit_round's refine
    # layout) — a distinct, role-based placeholder family so feedback_state
    # carries a content-level signal ("about to seed generation") instead of
    # relying purely on position. Reused identically at every refine round/
    # chain step, same pattern as <query>/<response> — not a per-round-index
    # tag. hp['V'] must cover HMN_STATE_0 + (family+1)*state_vocab_size;
    # never trained with state_vocab_size large enough for this to bump V
    # past the 274 default (2 families * state_vocab_size=2 = 4 <= 12 free
    # tail ids).
    assert state_vocab_size >= 1
    base = HMN_STATE_0 + family * state_vocab_size
    return [base + (i % state_vocab_size) for i in range(state_len)]


# Family index passed to _cyclic_state_ids for a refine round's
# feedback_state alphabet — see _cyclic_state_ids's docstring for why this
# is a separate family rather than a wrapper tag or a position-indexed token.
HMN_FEEDBACK_STATE_FAMILY = 1


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
      round 0 (initial): STATE <query> warmup </query> <response> out </response>
    Sequence (n_refine>0) additionally appends, per refine round:
      round k>0 (refine): state <response> argmax </response> feedback_state
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
        round_idx > 0: state + argmax + feedback_state + <query>/<response>
        (today's refine block). STATE-family regions are bare (no wrapper tag) —
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
    Stage `hop` — alternative to chunk_positions_chained's STATE_QUEUE/
    h_inject relay. Same overall shape (shared encoding pass, then each
    chain step threaded in sequence with its own local round-0(+refine)) but NO
    separate STATE_QUEUE_in region — chain step i's own round-0 STATE region
    serves double duty as both "this chain step's recall register" and "the
    thing the NEXT chain step reads." The relay channel is a genuine
    attention permission (see chunk_mask_fb_hop), not a forced vector copy.

    Why: h_inject's relay is a hand-engineered "copy chain step i-1's final
    STATE into chain step i's input" operation, cut off from gradient flow
    via .detach() (truncated BPTT — the model never gets direct gradient
    signal that STATE must be useful to a FUTURE chain step, only to its own
    recall loss, see chunk_positions_chained's docstring). `hop` instead
    grants chain step i's own round-0 STATE row a narrow, SINGLE-HOP
    attention exception (see chunk_mask_fb_hop's relay exception) letting it read
    chain step i-1's STATE columns directly and let the model LEARN what to
    preserve end-to-end, no detach, no forced copy. This also means `hop`
    training reuses the ordinary non-chained (fast, up-to-2-forward-passes)
    loop — no sequential per-chain-step h_inject orchestration needed, since
    everything is resolved by mask permissions within one packed sequence.

    Same single-hop constraint as STATE_QUEUE (M=1 in that design's terms):
    chain step i's STATE row sees ONLY chain step i-1's STATE columns, never
    i-2 or earlier directly — preserved here for a clean, apples-to-apples
    comparison against `relay` (chunk_positions_chained) where the LEARNING
    MECHANISM, not the information-theoretic constraint, is the isolated
    variable. For information from chain step i-2 to reach chain step i,
    chain step i-1 must still implicitly fold it into its own bottlenecked
    STATE — exactly the same open question chunk_positions_chained's
    docstring raises, now testable with full gradient flow instead of a
    truncated one.
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
            """Same as chunk_positions_chained's _emit_round, minus the
            has_queue_in branch — `hop` never allocates a separate queue
            region."""
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
    Stage `weave` — generalizes chunk_positions_hop to arbitrary INTERLEAVED
    encode/query operation sequences, not just "encode everything, then
    query everything" (batch) or "encode everything, then chain queries in a
    fixed order" (relay/hop). Every named trajectory pattern (batch, stream,
    interleave-delayed, repeat-query, and the chain-memory recovery probe
    generalized to more hops) is just a different `operations` list fed to
    this SAME function — no separate per-pattern position builder needed.

    operations: list of ops, each one of:
      ('E', chunk_idx)     — ingest chunk_idx's RAW BYTES ONLY: emit
                             <src>chunk_bytes</src>. Does NOT emit a STATE by
                             itself — chunk_idx must not have been encoded
                             already, and must be immediately followed by an
                             ('S', None) op (asserted below) to actually
                             compress it. Splitting "ingest" from "compress"
                             into two explicit ops (rather than one bundled
                             "encode" op) makes the compression step a first-
                             class, visible thing in the operations list —
                             see the module-level DSL discussion for why.
      ('S', None)           — emit ONE state_len-wide STATE region. Its ROLE
                             is determined entirely by what IMMEDIATELY
                             precedes it in the operations list:
                               - if an unclaimed ('E', k) op sits directly
                                 before it: this IS chunk k's own encoding-
                                 STATE (encoding isolation — sees chunk k's
                                 own raw bytes, blocked from every OTHER
                                 chunk's raw bytes; NOT part of the
                                 single-hop relay chain, same as the shared
                                 encoding pass always was).
                               - otherwise (no immediately-preceding unclaimed
                                 'E'): this is a bare relay hop — what used to
                                 be a separate 'N' (no-op) op type. Blocked
                                 from ALL raw chunks (chunk blackout) and from
                                 every prior op except the immediately
                                 preceding 'Q'-or-bare-'S' op's own STATE (the
                                 relay exception, single-hop). No local recall target,
                                 is_clean=False, contributes nothing to loss
                                 directly — gradient reaches it only through
                                 whichever LATER op ends up depending on it.
                                 Isolates "pure relay decay rate" from
                                 "recall accuracy at each hop," which
                                 repeated 'Q' ops conflate (a failure there
                                 could mean the relay lost information OR
                                 that hop's own local recall task failed for
                                 unrelated reasons).
      ('Q', (span_s, span_e)) — query/recall chunks [span_s, span_e): emit
                             STATE + <query>warmup</query><response>output
                             </response> (+ n_refine refine rounds). Every chunk
                             in [span_s, span_e) MUST already have been
                             ingested-and-compressed (E immediately followed
                             by S) earlier in the list — a hard causal
                             requirement (attention is causal; a query
                             literally cannot read content that doesn't
                             exist yet), asserted below, not just documented.

    Relay mechanism: same single-hop STATE-to-STATE attention permission as
    chunk_positions_hop (see chunk_mask_fb_traj) — the i-th 'Q'-or-bare-'S'
    op's own STATE row can read the (i-1)-th 'Q'-or-bare-'S' op's own STATE
    columns directly, nothing else cross-op. The SAME span can appear in
    multiple 'Q' ops (repeat-query pattern) — each occurrence is its own
    independent rec_block at its own sequence position, referencing the SAME
    underlying enc_blocks by chunk_idx (compression is never redone).

    Unlike chunk_positions_chained/chunk_positions_hop, enc_blocks here is a
    dict keyed by chunk_idx (not a list in emission order) since 'E'/'S'
    pairs can occur in any order interspersed with 'Q' ops — later 'Q' ops
    need to look up "where is chunk k's STATE" regardless of when it was
    emitted.
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

    # Sort by chunk_idx explicitly (NOT dict insertion order) — consumers
    # like ar_decode_srs_stitched_tagged_nokv assume enc_blocks[k] is chunk
    # k's block by LIST POSITION. Every traj_* pattern here happens to encode
    # chunks in numeric order already, so insertion order would coincidentally
    # match — but that's not guaranteed by the dict structure itself, so sort
    # explicitly rather than rely on it.
    chunk_idx_order = sorted(enc_blocks_c.keys())
    enc_blocks_c_list = [enc_blocks_c[k] for k in chunk_idx_order]
    enc_blocks_m_list = [enc_blocks_m[k] for k in chunk_idx_order]

    pos_content = dict(enc_blocks=enc_blocks_c_list, rec_blocks=rec_blocks_c,
                       enc_end=enc_end, warmup_len=warmup_len, L=L)
    pos_mask = dict(enc_blocks=enc_blocks_m_list, rec_blocks=rec_blocks_m,
                    enc_end=enc_end, warmup_len=warmup_len, L=L)

    return dict(pos_content=pos_content, pos_mask=pos_mask, tags=tags, L=L)


def chunk_mask_fb_traj(pos: dict) -> np.ndarray:
    """
    Mask for chunk_positions_traj layouts — same relay exception as
    chunk_mask_fb_hop (a round-0 STATE row may attend to the immediately
    preceding relay-producing op's own last-round STATE, single-hop only)
    but grouped by `op_idx` (the i-th 'Q'-or-'N' operation encountered)
    instead of by chain-step span, since chunk_positions_traj allows the
    SAME span to recur (repeat-query), allows 'E' ops interspersed anywhere,
    and now allows 'N' (no-op) ops that also participate in the relay chain
    — `span` alone can't identify "which occurrence" the way it could in
    chunk_positions_hop's strictly-one-Q-per-chain-step schedules.

    Encoding isolation (STATE_k isolated from chunk_j, j≠k), chunk blackout
    (query/no-op STATE blocked from ALL raw chunks), and refine feedback
    isolation (refine rounds) are all unchanged from chunk_mask_fb/
    chunk_mask_fb_hop. Only the relay exception changes how "the
    immediately preceding thing" is identified, and a 'noop' block gets the
    SAME chunk-blackout/relay-exception treatment as a 'initial' block's STATE
    row minus the warmup/output bottlenecks (no warmup/response fields to
    bound).
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

    is_any_rec_output = np.zeros(L, dtype=bool)
    for rb2 in rec_blocks:
        if 'c0' in rb2:  # 'noop' blocks have no output region
            is_any_rec_output |= (c >= rb2['c0']) & (c < rb2['c1'])

    last_rb_of_op: dict[int, int] = {}
    for i_rb, rb in enumerate(rec_blocks):
        last_rb_of_op[rb['op_idx']] = i_rb  # last write wins -> last rec_block per relay-producing op

    def _relay_source(prev_rb: dict) -> tuple[int, int]:
        """The (lo, hi) STATE range a later op's relay-read can attend to,
        for whichever op type prev_rb is."""
        if prev_rb['type'] == 'initial' or prev_rb['type'] == 'noop':
            return prev_rb['sl0'], prev_rb['sl1']
        return prev_rb['slb0'], prev_rb['slb1']

    def _prior_blocked_union(i_rb: int) -> np.ndarray:
        """Union of everything a round-0 STATE/no-op row is blocked from —
        every prior rec_block's STATE/warmup/argmax/output (noop blocks only
        contribute their STATE, having no other fields)."""
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

            prior_all = _prior_blocked_union(i_rb)
            if rb['op_idx'] > 0:
                lo, hi = _relay_source(rec_blocks[last_rb_of_op[rb['op_idx'] - 1]])
                prior_all = prior_all & ~((c >= lo) & (c < hi))
            blocked |= sl_row[:, None] & prior_all[None, :]
            # No Rules 4a/4b — a no-op has no warmup/response fields to bound.

        elif rb['type'] == 'initial':
            sl_row = (r >= rb['sl0']) & (r < rb['sl1'])
            blocked |= sl_row[:, None] & is_any_chunk[None, :]

            prior_all = _prior_blocked_union(i_rb)
            if rb['op_idx'] > 0:
                lo, hi = _relay_source(rec_blocks[last_rb_of_op[rb['op_idx'] - 1]])
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
# Named trajectory patterns (operations-list constructors) — see
# docs/HISTORY.md's trajectory-taxonomy discussion. Each takes n_chunks
# and returns an `operations` list for chunk_positions_traj. batch/stream/
# interleave_delayed are the TRAIN-mix candidates; repeat_query and
# long_hop_recovery are TEST-ONLY (never trained on) — the whole point is
# they diagnose generalization beyond whatever rhythm was trained on, so
# training on them would defeat their purpose.
# ---------------------------------------------------------------------------

def traj_batch(n_chunks: int, window_chunks: int = 2) -> list[tuple]:
    """Encode everything, then query everything in order — matches relay/hop exactly."""
    spans = ' '.join(f'Q({i},{i + window_chunks})' for i in range(n_chunks - window_chunks + 1))
    return parse_traj_dsl(f'E{n_chunks} {spans}')


def traj_stream(n_chunks: int, window_chunks: int = 2) -> list[tuple]:
    """Encode just enough for each query, then query immediately — each
    query happens as soon as its dependencies are satisfied, not after
    every chunk has been seen."""
    dsl_parts = [f'E{window_chunks}']  # enough for the first query
    for i in range(n_chunks - window_chunks + 1):
        if i > 0:
            dsl_parts.append('E')  # one more chunk before each subsequent query
        dsl_parts.append(f'Q({i},{i + window_chunks})')
    return parse_traj_dsl(' '.join(dsl_parts))


def traj_interleave_delayed(n_chunks: int, window_chunks: int = 2) -> list[tuple]:
    """Encode everything first (like batch), but query in a SHUFFLED
    (non-monotonic) order — isolates "does an intervening unrelated query
    corrupt recall of a not-yet-queried earlier span" from the
    encode/query-interleaving variable stream already tests."""
    spans = [(i, i + window_chunks) for i in range(n_chunks - window_chunks + 1)]
    # Reverse order: query the LAST span first, earliest span last — the
    # earliest span's query now has to survive every other query's relay
    # update happening first.
    q_str = ' '.join(f'Q({s},{e})' for s, e in reversed(spans))
    return parse_traj_dsl(f'E{n_chunks} {q_str}')


def traj_repeat_query(n_chunks: int, window_chunks: int = 2) -> list[tuple]:
    """TEST-ONLY. Encode everything, query in order, then query the FIRST
    span again at the very end — tests whether that content is still
    recoverable after the relay state kept moving forward through every
    other query in between."""
    spans = [(i, i + window_chunks) for i in range(n_chunks - window_chunks + 1)]
    q_str = ' '.join(f'Q({s},{e})' for s, e in spans)
    first_s, first_e = spans[0]
    return parse_traj_dsl(f'E{n_chunks} {q_str} Q({first_s},{first_e})')


def traj_long_hop_recovery(n_chunks: int, window_chunks: int = 2) -> list[tuple]:
    """TEST-ONLY. Encode everything, chain-query every span in order (so the
    single-hop relay is unbroken all the way through), then probe the FIRST
    span AGAIN as the final operation — same shape as traj_repeat_query, but
    intended to be run with a LARGER n_chunks than training ever used (e.g.
    n_chunks=8, 7 chain steps) to stress the relay over many more hops than
    it was ever trained on. This IS the deferred "chain-memory recovery
    probe" — generalized to arbitrary hop count via the same operations list
    mechanism, not a separate implementation."""
    return traj_repeat_query(n_chunks, window_chunks)


# ---------------------------------------------------------------------------
# Trajectory DSL — compact string notation for operations lists, matching
# this project's established curriculum-as-DSL convention (see
# archive_v1/kvmem/curriculum_dsl.py's `nN/rK/Xk` stage tokens for the
# precedent this follows, adapted to weave's E/S/Q operation alphabet).
#
# Grammar (whitespace-separated tokens):
#   E          ingest the next not-yet-ingested chunk's raw bytes (no STATE
#              yet — MUST be immediately followed by 'S', asserted by
#              chunk_positions_traj, not just documented)
#   E<n>       n consecutive (ingest, compress) PAIRS — shorthand that still
#              expands to explicit alternating E S E S ... (n of each), never
#              a bundled "encode" primitive — see chunk_positions_traj's
#              docstring for why E and S are kept as two separate ops
#   S          emit one STATE region — claims the immediately-preceding
#              unclaimed 'E' if there is one (that chunk's own encoding-
#              STATE), otherwise a bare relay-only no-op hop
#   S<n>       n bare 'S' ops in a row (only meaningful with no preceding
#              unclaimed 'E' — e.g. after a 'Q', for decay-curve patterns)
#   Q(s,e)     query span [s,e)
#
# Examples (equivalent to the Python constructors above):
#   batch:               "E4 Q(0,2) Q(1,3) Q(2,4)"
#   stream:               "E2 Q(0,2) E S Q(1,3) E S Q(2,4)"
#   interleave_delayed:    "E4 Q(2,4) Q(1,3) Q(0,2)"
#   repeat_query:          "E4 Q(0,2) Q(1,3) Q(2,4) Q(0,2)"
#   decay_curve (4 hops): "E2 Q(0,2) S4 Q(0,2)"
# ---------------------------------------------------------------------------

def parse_traj_dsl(s: str) -> list[tuple]:
    """Parse a trajectory DSL string into an operations list for
    chunk_positions_traj. See the grammar comment above this function.
    'E<n>' expands to n explicit (E, S) pairs, never a bundled op — the
    resulting operations list always has 'S' immediately after every 'E',
    matching chunk_positions_traj's hard requirement."""
    ops: list[tuple] = []
    next_chunk_idx = 0
    for tok in s.split():
        if tok.startswith('Q('):
            inner = tok[2:-1]  # strip 'Q(' and trailing ')'
            s_str, e_str = inner.split(',')
            ops.append(('Q', (int(s_str), int(e_str))))
        elif tok.startswith('E'):
            n = int(tok[1:]) if len(tok) > 1 else 1
            for _ in range(n):
                ops.append(('E', next_chunk_idx))
                ops.append(('S', None))
                next_chunk_idx += 1
        elif tok.startswith('S'):
            n = int(tok[1:]) if len(tok) > 1 else 1
            for _ in range(n):
                ops.append(('S', None))
        else:
            raise ValueError(f'unrecognized trajectory DSL token: {tok!r}')
    return ops


def traj_decay_curve(n_noop_hops: int, window_chunks: int = 2) -> list[tuple]:
    """TEST-ONLY (or a train-mix candidate at low hop counts — see
    docs/HISTORY.md). Query once, take n_noop_hops content-free relay
    hops, then repeat the SAME query — isolates pure relay decay rate from
    recall-accuracy-at-each-hop the way repeat_query/long_hop_recovery
    cannot (their intermediate hops are all real queries with their own
    local recall task, confounding "did the relay lose information" with
    "did that hop's own recall fail for unrelated reasons"). Cheap to
    stretch arbitrarily far since no-ops carry no extra warmup/response
    tokens and contribute no extra loss terms — sweep n_noop_hops to build
    an actual decay curve (match% vs. hop count) for the checkpoint being
    tested. Built on parse_traj_dsl as a demonstration of the DSL's purpose:
    every named trajectory constructor above could be expressed this way.

    Only encodes exactly `window_chunks` chunks (just enough for the probe
    span) — no unused chunks, since the point is isolating pure decay, not
    exercising a full multi-chunk schedule (unlike batch/stream/etc., this
    doesn't take a separate n_chunks parameter).

    CONFOUND WARNING for zero-shot eval against an existing checkpoint
    (measured, not theoretical — see eval_weave smoke test against `solo`'s
    checkpoint): this produces a MUCH SHORTER total sequence (L=156 at
    window_chunks=2/state_len=8/chunk_len=16) than a checkpoint trained with
    a larger n_chunks (e.g. `solo`'s L=236 at n_chunks=4). A checkpoint that
    scores 100% on batch/repeat_query (same n_chunks as training) can score
    0% here purely from never having seen a sequence this short/differently
    laid out — that's a LENGTH-EXTRAPOLATION failure, not evidence about
    decay. Only trust decay_curve results from a checkpoint that was
    actually TRAINED on decay_curve-shaped trajectories (or close to this
    exact length), not zero-shot against solo/relay/hop.
    """
    return parse_traj_dsl(f'E{window_chunks} Q(0,{window_chunks}) S{n_noop_hops} Q(0,{window_chunks})')


# =============================================================================
# Attention mask construction (feedback-argmax refine layout)
# ported from kvmem/train_hmn_chunk.py: chunk_mask_fb (lines 583-688)
# =============================================================================

def chunk_mask_fb(pos: dict) -> np.ndarray:
    """
    Mask for feedback-argmax refine layout. Same rules as chunk_mask for encoding
    blocks and round-0 (initial) turns. Additional rules for refine rounds (together,
    the refine feedback isolation rule):

    - state rows: blocked from all chunks (like all recall STATE fields).
    - argmax rows: blocked from all chunks.
    - feedback_state rows: blocked from all chunks; sees state + argmax causally.
    - refine output bottleneck: warmup/out rows blocked from everything except own feedback_state + own warmup/out.
       (Same strong bottleneck as the round-0 output bottleneck, but feedback_state is the gate — not state or argmax.)

    Nochain blackout (always on): Each round-0 STATE row is blocked from ALL
    tokens in prior rec_blocks (STATE, warmup, argmax, AND output of earlier
    chain steps' round-0 and refine turns). Forces every chain step to encode
    independently from enc-block STATEs only. Without this, the model chains
    through prior OUTPUT tokens — chain step 1 reads chain step 0's recalled
    bytes in the 50% overlap region. Blocking only prior STATEs is
    insufficient (v4 lesson).
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

    # Union of every rec_block's own output region (c0:c1) — refine turns must
    # reach earlier turns' output ONLY via their explicit argmax copy
    # (am0:am1), never by attending straight to the raw c0:c1 tokens sitting
    # in context. Those tokens are ground truth during training (teacher-
    # forced) but the model's own greedy decode at eval time — a direct
    # attention path there lets training "cheat" via leaked ground truth,
    # which collapses at AR-decode eval once that region is no longer GT.
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
            # Chunk blackout (round-0 STATE): blocked from all chunks
            sl_row = (r >= rb['sl0']) & (r < rb['sl1'])
            blocked |= sl_row[:, None] & is_any_chunk[None, :]
            # Nochain blackout: round-0 STATE blocked from ALL tokens in
            # prior rec_blocks (STATE + warmup + argmax + output). Blocking
            # only STATEs is insufficient — the model chains through prior
            # OUTPUT tokens (chain step 1 reads chain step 0's recalled
            # bytes in the 50% overlap). Full blackout forces every chain
            # step to encode from enc-block STATEs only.
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
            # Warmup bottleneck: round-0 warmup rows — own STATE + own warmup only
            if rb['w0'] < rb['w1']:
                wm_row = (r >= rb['w0']) & (r < rb['w1'])
                own = (c >= rb['sl0']) & (c < rb['sl1']) | (c >= rb['w0']) & (c < rb['w1'])
                blocked |= wm_row[:, None] & ~own[None, :]
            # Output bottleneck: round-0 out rows — own STATE + own warmup + own output
            out_row = (r >= rb['c0']) & (r < rb['c1'])
            own = ((c >= rb['sl0']) & (c < rb['sl1']) |
                   (c >= rb['w0'])  & (c < rb['w1'])  |
                   (c >= rb['c0'])  & (c < rb['c1']))
            blocked |= out_row[:, None] & ~own[None, :]

        else:  # type == 'refine'
            # refine feedback isolation: state, argmax, feedback_state — blocked from
            # encoding chunks AND from every rec_block's own raw output
            # region (own am0:am1 copy is the only sanctioned path back to
            # an earlier turn's output — see is_any_rec_output comment above).
            sla_row = (r >= rb['sla0']) & (r < rb['sla1'])
            am_row  = (r >= rb['am0'])  & (r < rb['am1'])
            slb_row = (r >= rb['slb0']) & (r < rb['slb1'])
            blocked |= (sla_row | am_row | slb_row)[:, None] & (is_any_chunk | is_any_rec_output)[None, :]

            # refine output bottleneck: warmup/out rows — only own feedback_state + own warmup + own output
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


def chunk_mask_fb_hop(pos: dict, hops: int = 0) -> np.ndarray:
    """
    Mask for Stage `hop` layouts (chunk_positions_hop) — identical to
    chunk_mask_fb in every rule EXCEPT one deliberate, narrow exception to
    the nochain blackout: a round-0 STATE row is still blocked from every
    prior chain step's warmup/argmax/output and from every prior chain
    step's STATE EXCEPT the last-round STATE of the `hops` chain
    steps immediately preceding it (each one's sl0:sl1 if that chain step
    had no refine rounds, else its slb0:slb1 — the same "last round's own
    STATE" definition the original STATE_QUEUE_out used). This is the
    sanctioned relay channel (the relay exception), now a genuine attention
    permission instead of a forced h_inject copy — see
    chunk_positions_hop's docstring for the full rationale.

    hops: how many chain steps back the exception reaches.
      - 0 (default): NO relay exception at all — a round-0 STATE row is
        blocked from every prior chain step's content with no carve-out,
        the same nochain-blackout-only behavior chunk_mask_fb itself
        already has (functionally equivalent to `solo`'s no-relay case for
        any multi-chain-step schedule). This is a deliberate default —
        `hop`'s actual relay behavior is opt-in via hops>=1, not
        assumed; any stage that wants the relay MUST set hops
        explicitly (see kvmem/configs/hmn_recall_queue.py, which sets
        hops=1).
      - 1: the originally-designed, verified single-hop relay (chain step
        i's STATE sees only chain step i-1's STATE, never i-2 or earlier
        directly) — everything `hop`'s prior results (CLAUDE.md) were
        measured against.
      - N>1: chain step i's STATE sees the union of the last N chain
        steps' own STATE columns (i-1, i-2, ..., i-N) — untested territory,
        a direct generalization of the M parameter the original
        STATE_QUEUE design named but never built past M=1.

    Everything else (encoding isolation, refine feedback isolation for refine
    rounds, the is_any_rec_output leak-prevention union) is copied verbatim
    from chunk_mask_fb — this function is NOT a general-purpose replacement,
    only chunk_positions_hop's own pos dicts (which never have a 'queue0'
    field) should be passed to it.
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

    is_any_rec_output = np.zeros(L, dtype=bool)
    for rb2 in rec_blocks:
        is_any_rec_output |= (c >= rb2['c0']) & (c < rb2['c1'])

    # Group rec_block indices by chain step (consecutive runs sharing the
    # same 'span') so the relay exception can find "the immediately
    # preceding chain step's last rec_block" for the single-hop
    # STATE-to-STATE exception.
    chain_step_of_rb: list[int] = []
    span_to_idx: dict[tuple, int] = {}
    for rb in rec_blocks:
        if rb['span'] not in span_to_idx:
            span_to_idx[rb['span']] = len(span_to_idx)
        chain_step_of_rb.append(span_to_idx[rb['span']])
    last_rb_of_chain_step: dict[int, int] = {}
    for i_rb, ci in enumerate(chain_step_of_rb):
        last_rb_of_chain_step[ci] = i_rb  # overwritten each time -> ends up as the LAST index per chain step

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

            this_chain_step = chain_step_of_rb[i_rb]
            # Relay exception: union of the last `hops` chain steps'
            # own last-round STATE ranges (hops=0 -> empty union, no
            # exception at all; hops=1 -> just chain step i-1, the
            # originally-designed and verified behavior).
            relay_ranges: list[tuple[int, int]] = []
            for back in range(1, hops + 1):
                src_chain_step = this_chain_step - back
                if src_chain_step < 0:
                    break
                src_last_i_rb = last_rb_of_chain_step[src_chain_step]
                src_last_rb = rec_blocks[src_last_i_rb]
                if src_last_rb['type'] == 'initial':
                    relay_ranges.append((src_last_rb['sl0'], src_last_rb['sl1']))
                else:
                    relay_ranges.append((src_last_rb['slb0'], src_last_rb['slb1']))

            # round-0 STATE blocked from ALL tokens in prior rec_blocks
            # EXCEPT the union of relay_ranges above (the carve-out from
            # the nochain blackout — see module docstring).
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

            # Warmup bottleneck: round-0 warmup rows — own STATE + own warmup only
            # (relay visibility is NOT extended to warmup/output rows — only
            # the STATE row itself gets the single-hop exception, same
            # bottleneck-forcing intent as STATE_QUEUE_in's original design).
            if rb['w0'] < rb['w1']:
                wm_row = (r >= rb['w0']) & (r < rb['w1'])
                own = (c >= rb['sl0']) & (c < rb['sl1']) | (c >= rb['w0']) & (c < rb['w1'])
                blocked |= wm_row[:, None] & ~own[None, :]
            # Output bottleneck: round-0 out rows — own STATE + own warmup + own output
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
    Returns updated tok_np with argmax filled at refine turn am0:am1 positions.
    """
    tok = tok_np.copy()
    for rb in pos['rec_blocks']:
        if rb['type'] == 'refine':
            # logits at positions src_c0-1 .. src_c0-1+out_len predict the
            # source block's own output — works whether the source is the
            # same-span initial block (chunk_positions_fb) or a byte-sliced /
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
                      tags: list[tuple[int, int]],
                      data_kind: str = 'random', data_target_bits: float | None = None) -> np.ndarray:
    """
    data_kind: 'random' (default, unchanged behavior — uniform random bytes,
    the source distribution every prior architecture in this project trained
    and validated on) or 'chaotic'/'fractal'/'ca' (kvmem/structured_data.py —
    the structured-data track, see docs/HISTORY.md §8). Each batch item
    gets a FRESH call to generate_structured_chunks (fresh rule/seed per
    example, same principle as random bytes already being resampled per
    batch — required so the model can't bake a fixed rule into static
    weights instead of encoding it into STATE, see structured_data.py's
    module docstring).
    """
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
            # weave's bare-'S' relay hop: no span, no warmup/response, only
            # its own STATE region needs filling (placeholder ids only —
            # content is resolved by attention via the relay exception, not
            # by anything written here, same as any other STATE region).
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
            # seg_start must equal L_cached exactly (KV-cache invariant, see
            # CLAUDE.md "KV decode off-by-one") — this sweeps in every tag
            # token between the last cached position and c0 automatically,
            # regardless of how many tags precede this block's content.
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
    AR decode for chunk_positions_traj layouts (Stage `weave`) — same
    mechanics as ar_decode_srs_stitched_tagged_nokv (full-recompute, no KV
    cache; only the FIRST query's warmup comes from ground truth, every
    later query's warmup is chained from the model's own decoded bytes) but
    generalized to handle `'noop'` rec_blocks, which
    ar_decode_srs_stitched_tagged_nokv cannot: it unconditionally unpacks
    `rb['span']` and branches only on 'initial'/'refine', and a `'noop'` block has
    `span=None` and none of `sla0`/`am0`/etc. — would raise on both counts.

    A no-op block needs no decode step at all: its STATE region is filled
    with the same fixed placeholder tokens (`sids`) as any other STATE
    region, and its actual VALUE is whatever the causal forward pass
    computes there — there is no c0:c1 output to autoregressively generate,
    no target to match, no `turn_match_pcts` entry. It exists purely as
    context for whatever LATER op's single-hop relay reads it.

    Simplifications vs. ar_decode_srs_stitched_tagged_nokv: (1) no BPB —
    that function's teacher-forced BPB pass groups rec_blocks by "last
    block per contiguous same-span run," which assumes each span appears
    at most once in a monotonic sequence; repeat_query/interleave_delayed
    violate that (the same span can recur non-contiguously), so BPB is
    dropped here rather than computed incorrectly — match_pct (mean of
    turn_match_pcts) is the metric this diagnostic actually needs. (2)
    warmup chaining keyed by span (`decoded_by_span`), not a single global
    byte buffer — a repeated span's second occurrence chains its warmup
    from that SAME span's own most recent decode, not from whatever
    happens to sit at that byte offset globally (meaningful for weave's
    arbitrary/non-overlapping spans, unlike relay/hop's fixed 50%-overlap
    schedule where a global buffer and per-span chaining coincide).
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
        during backward (depth-only), same semantics as KVMemModel. For
        dual_attn, may be None | 'block' | 'attn' matching DualAttnModel's two
        granularities (whole-block vs per-attn-sublayer checkpointing).
        """
        batched = tokens.dim() == 2
        if not batched:
            tokens = tokens.unsqueeze(0)
        x      = self._embed(tokens)

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
    data_kind = hp.get('data_kind', 'random')  # 'random' (default) | 'chaotic'/'fractal'/'ca' (structured-data track)
    data_target_bits = hp.get('data_target_bits', None)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=wd, betas=(0.9, 0.999))

    curriculum = hp.get('curriculum', [])
    assert curriculum
    log_every  = hp.get('log_every', 500)

    global_step = 0
    t_start = time.time()

    for stage_i, stage in enumerate(curriculum):
        if 'weave_mix' in stage:
            # Stage `weave` training dispatch — samples from a weighted mix
            # of named weave patterns each training step, analogous to the
            # traj_mix branch below but built on chunk_positions_traj/
            # chunk_mask_fb_traj instead of chunk_positions_iq_global_rw_tagged/
            # chunk_mask_fb. Only the three patterns flagged as train-mix
            # candidates in docs/HISTORY.md §4c are accepted here —
            # repeat_query/long_hop_recovery/decay_curve are deliberately
            # test-only generalization probes (training on them would
            # defeat their purpose) and are rejected with an assertion, not
            # silently allowed. n_refine is fixed at 0 — no argmax-feedback
            # refine support for weave patterns yet, matching every existing
            # weave usage (kvmem/eval_weave.py never uses n_refine>0 either).
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
                                         interleave_delayed=traj_interleave_delayed)

            weave_mix_cfg = stage['weave_mix']  # list of {weight, pattern[, n_chunks, window_chunks]}
            trajectories = []
            for wcfg in weave_mix_cfg:
                pname = wcfg['pattern']
                assert pname in _WEAVE_TRAIN_PATTERNS, (
                    f"weave_mix pattern {pname!r} is not a train-mix candidate — only "
                    f"{list(_WEAVE_TRAIN_PATTERNS)} are safe to train on; repeat_query/"
                    f"long_hop_recovery/decay_curve are deliberately test-only "
                    f"generalization probes (see docs/HISTORY.md §4c)")
                w_n_chunks = wcfg.get('n_chunks', n_chunks)
                w_window_chunks = wcfg.get('window_chunks', window_chunks)
                ops = _WEAVE_TRAIN_PATTERNS[pname](w_n_chunks, w_window_chunks)
                built = chunk_positions_traj(chunk_len, state_len, warmup_len, ops,
                                             n_refine=0, state_vocab_size=state_vocab_size)
                pos_content, pos_mask, tags, L = (built['pos_content'], built['pos_mask'],
                                                  built['tags'], built['L'])
                mask_np = chunk_mask_fb_traj(pos_mask)
                mask_t  = torch.tensor(mask_np, dtype=torch.float32, device=device)
                trajectories.append(dict(weight=wcfg['weight'], pattern=pname, n_chunks=w_n_chunks,
                                         pos_content=pos_content, mask_np=mask_np, mask_t=mask_t,
                                         tags=tags, L=L))
            traj_weights = np.array([t['weight'] for t in trajectories], dtype=np.float64)
            traj_weights = traj_weights / traj_weights.sum()

            _log(f'\n[stage {stage_i}] weave_mix='
                 f'{[(t["pattern"], t["n_chunks"], round(w, 2)) for t, w in zip(trajectories, traj_weights)]}  '
                 f'chunk_len={chunk_len} state={state_len} wl={warmup_len}  B={B}  steps={n_steps}')

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
            pbar = tqdm(range(1, n_steps + 1), desc=f'stage{stage_i}', dynamic_ncols=True, file=status_file)
            for local_step in pbar:
                global_step += 1
                lr = _lr(local_step)
                for pg in opt.param_groups: pg['lr'] = lr

                model.train(); opt.zero_grad()

                traj = trajectories[rng.choice(len(trajectories), p=traj_weights)]
                t_pos_content, t_mask_t, t_tags = traj['pos_content'], traj['mask_t'], traj['tags']

                tok_np = make_batch_tagged(rng, B, traj['n_chunks'], chunk_len, state_len, state_vocab_size,
                                           t_pos_content, t_tags, data_kind=data_kind,
                                           data_target_bits=data_target_bits)
                tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)

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

        if 'chain_steps' in stage:
            # Chained multi-chain-step training (ports experiments/attn_dual/
            # train.py's single-trajectory loop verbatim). Structurally
            # different from the traj_mix branch below: ONE packed pos/mask
            # built from a fixed `chain_steps` list (one rec_block per chain
            # step, each with its own n_refine refine rounds) instead of many
            # small per-trajectory sequences sampled by weight each step.
            #
            # Always uses chunk_positions_hop/chunk_mask_fb_hop — the
            # single-hop STATE-to-STATE attention permission (the relay
            # exception) is a strict superset of the single-chain-step case (no prior chain
            # step to read from, so the exception never triggers — byte-
            # identical layout to a plain non-relay schedule) AND the
            # multi-chain-step relay case, resolved entirely within one
            # packed-sequence forward pass (no sequential per-chain-step
            # orchestration needed). Superseded the earlier h_inject-based
            # STATE_QUEUE/`chain=True` relay (chunk_positions_chained,
            # sequential per-chain-step forward passes with a .detach()'d
            # injected feature) once `hop`'s learned-attention-permission
            # mechanism was validated to substantially outperform it — see
            # docs/HISTORY.md §4b and CLAUDE.md's relay-vs-hop comparison.
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
            hops  = stage.get('hops', 0)  # default 0 = no relay exception at all, opt-in required

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
            pbar = tqdm(range(1, n_steps + 1), desc=f'stage{stage_i}', dynamic_ncols=True, file=status_file)
            for local_step in pbar:
                global_step += 1
                lr = _lr(local_step)
                for pg in opt.param_groups: pg['lr'] = lr

                model.train(); opt.zero_grad()

                tok_np = make_batch_tagged(rng, B, n_chunks, chunk_len, state_len, state_vocab_size,
                                           pos_content, tags, data_kind=data_kind,
                                           data_target_bits=data_target_bits)
                tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)

                wrong_masks: dict[int, np.ndarray] = {}

                # One shared mask, up to 2 forward passes total for the whole
                # packed sequence — the hop relay (the relay-exception
                # attention permission) is resolved entirely by mask_t, no sequential
                # per-chain-step orchestration needed.
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
        # — "Eval uses first refine trajectory" — full initial+refine chain is the most
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
                                       t_pos_content, t_tags, data_kind=data_kind,
                                       data_target_bits=data_target_bits)
            tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)

            # wrong_token_weight ablation: capture, per refine block, whether the fed-back
            # argmax at each position was wrong (vs the ground truth that was there
            # pre-fill) — used to upweight NLL specifically at positions that need
            # active correction, rather than diffusing gradient equally over positions
            # the model already had right. See docs/FEEDBACK_RESULTS.md § refine-refinement
            # loss redesign, ablation 1.
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
