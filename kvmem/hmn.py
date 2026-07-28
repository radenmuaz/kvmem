"""
kvmem/hmn.py — HashMemNet (HMN). Single-file consolidated implementation.

CHAT-TAG-FREE by design: no `HMN_SRC_OPEN/CLOSE`, `HMN_QUERY_OPEN/CLOSE`,
`HMN_RESPONSE_OPEN/CLOSE` boundary-marker tokens are ever emitted into the
packed sequence. `chunk_positions_traj` places region content back-to-back
with NO wrapper tokens: an encode ('E') region is exactly `chunk_len` raw
bytes, a query's round-0 layout is warmup directly followed by response (no
QUERY_OPEN/CLOSE, no RESPONSE_OPEN/CLOSE), and a refine round's
argmax-feedback/second-STATE/warmup/response run together the same way.
**E/S/Q/R region boundaries are INFERRED, not marked** — the only
structural signal available to the model is byte-content type (raw data
bytes are 0-255; STATE placeholder IDs are 256+) and relative position.
This replaced an earlier tagged design (preserved as `hmn_v4_backup.py`,
see the dated-snapshot table in CLAUDE.md) after a design review found the
tagged vocab's per-turn STATE role was ambiguous for serial chunk
ingestion (e.g. `E1 S E2 S Q1 S Q2` — the `S` before `Q2` could be read as
either an encode-claim or a query-recall STATE with no way to disambiguate
from token identity alone). Fixed via:

- A factorized **opcode + shared value alphabet** (`HMN_OP_UPDATE`/`_NOOP`/
  `_FEEDBACK`, `HMN_STATE_0`) — every STATE emission is `[opcode_token,
  value_0, ..., value_{state_len-1}]`, `V=271` (256 bytes + 3 opcodes + 12
  reserved shared STATE values).
- A query's own STATE is no longer a pre-filter register before
  warmup/response — it's built AFTER (end-of-turn), claimed by an explicit
  trailing `'S'` in the operations list, and omitted entirely for a
  terminal query (nothing relays from it, so there's nothing to build —
  see docs/HISTORY.md §15 for the full derivation, including why this
  isn't redundant for any op with a successor, and the refine-round
  argmax/feedback-opcode boundary-fix design that places `OP_FEEDBACK`
  before the argmax content rather than after).

ALL FOUR stage-dispatch paths use this same mechanism: `weave_mix`
(`chunk_positions_traj`, native), `chain_steps` (`chunk_positions_hop`, a
thin wrapper delegating to `chunk_positions_traj`), the legacy global-
window `traj_mix` path (`chunk_positions_iq_global_rw_tagged`, its own
port — genuinely different shape, fixed-size sliding window rather than
"through the end of span"), and `stitch_mix` (`chunk_positions_stitch`,
its own port — continuous byte-address stitching, not chunk-aligned).
Each verified individually via direct mask-matrix inspection plus an
end-to-end train+eval smoke test (see docs/HISTORY.md §15's "Status").

NLL loss covers the warmup region too, not just the response/continuation
(`rb['w0']:rb['c1']` instead of `rb['c0']:rb['c1']`) — without chat tags
marking "this is where the query starts," scoring reconstruction of the
warmup bytes themselves (available as ground truth in the input either
way) gives the model a training signal for recognizing "I am now in a
query's warmup region" from content/position alone.

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
# Vocab / tags — THIS FORK HAS NO CHAT TAGS, AND NO PER-ROLE STATE FAMILIES
#
# Layout: 256 data bytes, then 3 opcode tokens, then a SHARED STATE value
# alphabet. No HMN_SRC_OPEN/CLOSE/HMN_QUERY_OPEN/CLOSE/HMN_RESPONSE_OPEN/
# CLOSE (the 6 chat-tag ids hmn.py reserves at 256-261) — chunk_positions_
# traj never emits them. See docs/HISTORY.md §15 for the full design
# rationale behind what follows.
#
# Every STATE emission (encode-claim, query end-of-turn, refine feedback,
# bare relay no-op) is now `[opcode_token, value_0, value_1, ..., value_
# {state_len-1}]` — ONE opcode token marking the ROLE, followed by
# `state_len` value tokens drawn from a SINGLE SHARED alphabet reused
# across every role (the opcode alone carries the role distinction now).
# This replaces hmn.py's per-role token FAMILIES (regular STATE family +
# a separate feedback_state family, each with its own private
# state_vocab_size-sized alphabet) — factorizing role from value is
# cheaper (`3 + state_vocab_size` extra ids instead of `3 * state_vocab_
# size`) AND lets the model learn what "value 0 vs value 1" means ONCE,
# shared across every role, instead of relearning it independently per
# family.
# =============================================================================

DATA_LO = 0x20   # legacy: data restricted to [0x20, 0xFF]

# Opcode tokens — one per STATE emission, marking what role that STATE plays:
#   update   — ordinary STATE update: an encode's claim, or a query's
#              end-of-turn STATE built from its own warmup+response.
#   noop     — pure relay pass-through, nothing new incorporated (rare,
#              opt-in — e.g. decay_curve-style trajectories; no currently-
#              implemented trajectory in this fork emits it yet).
#   feedback — a refine round's post-argmax-feedback STATE, placed BEFORE
#              the argmax content (`opf0`, in chunk_positions_traj/chunk_
#              positions_iq_global_rw_tagged's refine-round layout) rather
#              than after, resolving the boundary ambiguity between
#              feedback content and the prior round's response — see
#              docs/HISTORY.md §15.
HMN_OP_UPDATE   = 256
HMN_OP_NOOP     = 257
HMN_OP_FEEDBACK = 258

# First STATE value id — the shared alphabet, reused regardless of opcode.
HMN_STATE_0 = 259

# 256 bytes + 3 opcodes + 12 reserved STATE value ids (headroom for
# state_vocab_size growth, same reservation convention hmn.py used for its
# own V=274 — actual configs use state_vocab_size=2 today).
HMN_TAG_VOCAB_SIZE = 271


def _cyclic_state_ids(state_len: int, state_vocab_size: int = 2) -> list[int]:
    """Cycles through the SHARED value alphabet — no `family` parameter
    anymore (that's what opcode tokens replace, see module comment above)."""
    assert state_vocab_size >= 1
    return [HMN_STATE_0 + (i % state_vocab_size) for i in range(state_len)]


# =============================================================================
# Position/mask-field builders
# =============================================================================

def chunk_positions_iq_global_rw_tagged(n_chunks: int, chunk_len: int, state_len: int,
                                        warmup_len: int, window_chunks: int = 2,
                                        warmup_x_fixed: int | None = None,
                                        warmup_x_dist: str = 'uniform',
                                        n_refine: int = 0) -> dict:
    """
    Returns dict(pos_content=..., pos_mask=..., tags=[], L=...) — REDESIGNED
    (docs/HISTORY.md §15, opcode/no-chat-tag mechanism). Always exactly ONE
    global query (round-0 + optional refine rounds) spanning the WHOLE
    `n_chunks` source but with a FIXED response length
    (`window_chunks*chunk_len - warmup_len`, a sliding window that can start
    anywhere in `[0, x_max]` via `warmup_x_fixed`/`warmup_x_dist`) — this is
    why it's NOT delegated to `chunk_positions_traj` like `chunk_positions_
    hop` was: that function's out_len is always "through the end of the
    span," not a fixed window size. Always terminal (nothing ever relays
    from this — the caller uses the simpler `chunk_mask_fb`, no hops/relay
    concept applies here at all), so there is no end-of-turn STATE, ever —
    same principle as any terminal query in chunk_positions_traj.

    Sequence: per chunk k: chunk_k bytes, then STATE (opcode=update). Then
    round 0: warmup, response, THEN (if n_refine>0) each refine round:
    [OP_FEEDBACK][argmax feedback][feedback STATE values][re-presented
    warmup][new response] — same layout/boundary-fix rationale as
    chunk_positions_traj's refine rounds.
    """
    enc_blocks_c: list[dict] = []
    enc_blocks_m: list[dict] = []
    tags: list[tuple[int, int]] = []  # stays empty — no chat tags in this fork
    offset = 0

    for _ in range(n_chunks):
        s0 = offset; s1 = s0 + chunk_len; offset = s1
        sl0 = offset; sl1 = sl0 + state_len + 1; offset = sl1  # +1 for the opcode token
        enc_blocks_c.append(dict(s0=s0, s1=s1, sl0=sl0, sl1=sl1))
        enc_blocks_m.append(dict(s0=s0, s1=s1, sl0=sl0, sl1=sl1))

    enc_end = offset
    out_len = window_chunks * chunk_len - warmup_len

    w0 = offset; w1 = w0 + warmup_len; offset = w1
    c0 = offset; c1 = c0 + out_len; offset = c1

    src_len = n_chunks * chunk_len
    x_max = src_len - warmup_len - out_len
    n_windows = n_chunks - window_chunks + 1
    eval_offsets = [i * chunk_len for i in range(n_windows)]
    train_range = (warmup_x_fixed, warmup_x_fixed) if warmup_x_fixed is not None else (0, x_max)
    _dist = 'fixed' if warmup_x_fixed is not None else warmup_x_dist
    rw_extra = dict(warmup_train_range=train_range, warmup_x_dist=_dist,
                    warmup_valid_offsets=eval_offsets, window_chunks=window_chunks)

    rec_blocks_c = [dict(type='initial', span=(0, n_chunks), span_len=src_len,
                         out_len=out_len, is_clean=(n_refine == 0), op_idx=0,
                         w0=w0, w1=w1, c0=c0, c1=c1, sl0=None, sl1=None, **rw_extra)]
    rec_blocks_m = [dict(type='initial', span=(0, n_chunks), op_idx=0,
                         w0=w0, w1=w1, c0=c0, c1=c1, sl0=None, sl1=None)]

    prev_c0 = c0
    for _ in range(n_refine):
        opf0 = offset; offset += 1
        am0 = offset; am1 = am0 + out_len; offset = am1
        sl0 = offset; sl1 = sl0 + state_len; offset = sl1
        rw0 = offset; rw1 = rw0 + warmup_len; offset = rw1
        rc0 = offset; rc1 = rc0 + out_len; offset = rc1
        rec_blocks_c.append(dict(type='refine', span=(0, n_chunks), span_len=src_len,
                                 out_len=out_len, is_clean=True, op_idx=0,
                                 opf0=opf0, am0=am0, am1=am1, sl0=sl0, sl1=sl1,
                                 w0=rw0, w1=rw1, c0=rc0, c1=rc1,
                                 argmax_src_c0=prev_c0, end_sl0=None, end_sl1=None, **rw_extra))
        rec_blocks_m.append(dict(type='refine', span=(0, n_chunks), op_idx=0,
                                 opf0=opf0, am0=am0, am1=am1, sl0=sl0, sl1=sl1,
                                 w0=rw0, w1=rw1, c0=rc0, c1=rc1, end_sl0=None, end_sl1=None))
        prev_c0 = rc0

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
    THIN WRAPPER over `chunk_positions_traj` (docs/HISTORY.md §15) — chain_steps
    is structurally "encode all n_chunks up front, then query each span in
    sequence, each one relaying to the next" — exactly `chunk_positions_traj`'s
    batch-style operations list, with an explicit trailing 'S' after every
    chain step except the last (which needs no end-of-turn STATE — nothing
    relays from it, the double-state-redundancy fix). Reuses the already-
    verified opcode/end-of-turn-STATE mechanism and masking instead of a
    separate, duplicate implementation — see chunk_positions_traj's own
    docstring for the layout/opcode details this now shares.
    """
    ops: list[tuple] = []
    for i in range(n_chunks):
        ops.append(('E', i))
        ops.append(('S', None))
    for i, span in enumerate(chain_steps):
        ops.append(('Q', span))
        if i < len(chain_steps) - 1:  # every step but the last relays forward
            ops.append(('S', None))
    return chunk_positions_traj(chunk_len, state_len, warmup_len, ops,
                                n_refine=n_refine, state_vocab_size=state_vocab_size)


def chunk_positions_traj(chunk_len: int, state_len: int, warmup_len: int,
                         operations: list[tuple], n_refine: int = 0,
                         state_vocab_size: int = 2) -> dict:
    """
    Generalizes chunk_positions_hop to arbitrary interleaved encode/query
    operation sequences — every named trajectory pattern (batch, stream,
    interleave-delayed, repeat-query, ...) is just a different `operations`
    list fed to this same function.

    REDESIGNED (see docs/HISTORY.md §15 for the full derivation) — every
    STATE emission (encode-claim, query end-of-turn, bare no-op relay hop)
    is now `[opcode_token, value_0, ..., value_{state_len-1}]`: ONE opcode
    token (HMN_OP_UPDATE/NOOP/FEEDBACK) marking the ROLE, then `state_len`
    value tokens from a SHARED alphabet. `sl0:sl1` below spans the WHOLE
    block (opcode + values), size `1+state_len` — chosen deliberately so
    every existing mask-rule reference to `sl0:sl1` keeps working
    unchanged, just one token wider.

    A query's own STATE is no longer emitted BEFORE its warmup/response —
    warmup/response now attend DIRECTLY to whatever `hops` makes available
    (no intermediate pre-filter register). A query's end-of-turn STATE
    (built from warmup+response, needed ONLY if a later op will relay from
    it) is emitted AFTER, claimed by an explicit trailing 'S' — exactly the
    same claim mechanism 'S' already uses for a pending 'E'. A terminal
    query (nothing relays from it) simply has no trailing 'S' and gets no
    STATE of its own at all — see the double-state-redundancy derivation
    in docs/HISTORY.md §15.

    operations: list of ops, each one of:
      ('E', chunk_idx)        — ingest chunk_idx's raw bytes only. Must be
                                immediately followed by ('S', None) to claim it.
      ('S', None)              — claims, in priority order: (1) an immediately-
                                preceding unclaimed 'E' (encode-claim STATE,
                                opcode=update), (2) an immediately-preceding
                                just-finished 'Q' with no STATE of its own yet
                                (query's end-of-turn STATE, opcode=update),
                                (3) otherwise a bare relay-only no-op hop
                                (opcode=noop, blocked from all raw chunks, no
                                local recall target).
      ('Q', (span_s, span_e))  — query/recall chunks [span_s, span_e); emits
                                ONLY warmup+response, no STATE (see above).
                                Every chunk in the span must already be
                                encoded (causal requirement, asserted below).

    n_refine>0: each refine round is `[OP_FEEDBACK][argmax feedback, out_len
    raw bytes][feedback STATE values][re-presented warmup][new response]` —
    see docs/HISTORY.md §15's refine-trajectory + boundary-fix derivation
    for why OP_FEEDBACK precedes the argmax content specifically (argmax
    feedback and the previous round's response are both raw byte-range
    tokens with no other structural cue marking the boundary between them).
    A refine round's `sl0/sl1` are ALWAYS its feedback STATE values (not
    optional) — a separate `end_sl0/end_sl1` holds the OPTIONAL end-of-turn
    STATE, claimable by a trailing 'S' only on the last round emitted for
    this op (same terminal-op-omission rule as round-0-only queries).

    Relay: same single-hop STATE-to-STATE attention permission as
    chunk_positions_hop (see chunk_mask_fb_traj), grouped by op_idx instead of
    chain-step span since the same span can recur (repeat-query).

    enc_blocks here is a dict keyed by chunk_idx (not emission order), since
    'E'/'S' pairs can be interspersed with 'Q' ops in any order.
    """
    enc_blocks_c: dict[int, dict] = {}
    enc_blocks_m: dict[int, dict] = {}
    tags: list[tuple[int, int]] = []  # stays empty in this fork — no chat tags ever emitted
    offset = 0

    rec_blocks_c: list[dict] = []
    rec_blocks_m: list[dict] = []
    op_count = 0  # counts 'Q' ops AND bare 'S' ops — both produce a relay-eligible STATE

    pending_chunk_idx: int | None = None   # unclaimed 'E' waiting for its 'S'
    pending_query_i: int | None = None     # index into rec_blocks_c of a just-finished
                                            # 'Q' still eligible to be claimed by a trailing 'S'

    def _emit_state_block(opcode: int):
        """[opcode_token, value_0, ..., value_{state_len-1}] — returns (sl0, sl1)
        spanning the WHOLE block (opcode included) so every mask rule below
        can keep treating `sl0:sl1` as one contiguous STATE region."""
        nonlocal offset
        sl0 = offset
        offset += 1  # opcode token position, filled by make_batch_tagged/decode with `opcode`
        offset += state_len  # value token positions, cyclic shared alphabet
        sl1 = offset
        return sl0, sl1

    for op, arg in operations:
        if op == 'E':
            chunk_idx = arg
            assert chunk_idx not in enc_blocks_c, f'chunk {chunk_idx} encoded twice'
            assert pending_chunk_idx is None, \
                f"chunk {pending_chunk_idx}'s 'E' was never followed by 'S' before chunk {chunk_idx}'s 'E'"
            # No HMN_SRC_OPEN/CLOSE wrapper — the chunk_len raw bytes sit directly at
            # `offset`, back-to-back with whatever preceded them. Boundary inferred.
            s0 = offset; s1 = s0 + chunk_len; offset = s1
            enc_blocks_c[chunk_idx] = dict(s0=s0, s1=s1)  # sl0/sl1 filled in when 'S' claims it, below
            enc_blocks_m[chunk_idx] = dict(s0=s0, s1=s1)  # no ±1 padding — no flanking tag to include
            pending_chunk_idx = chunk_idx

        elif op == 'S':
            if pending_chunk_idx is not None:
                # Claims the immediately-preceding unclaimed 'E' — this IS
                # that chunk's own encoding-STATE (encoding isolation role),
                # NOT part of the single-hop relay chain (same as the shared
                # encoding pass always was — see chunk_mask_fb_traj's
                # encoding-isolation handling, unchanged).
                sl0, sl1 = _emit_state_block(HMN_OP_UPDATE)
                enc_blocks_c[pending_chunk_idx]['sl0'] = sl0
                enc_blocks_c[pending_chunk_idx]['sl1'] = sl1
                enc_blocks_m[pending_chunk_idx]['sl0'] = sl0
                enc_blocks_m[pending_chunk_idx]['sl1'] = sl1
                pending_chunk_idx = None
            elif pending_query_i is not None:
                # Claims the just-finished 'Q' (or its last refine round) — its
                # END-OF-TURN STATE, built from whatever the relay/hops window
                # made available to its warmup+response PLUS its own warmup+
                # response content (see docs/HISTORY.md §15). Needed only when a
                # LATER op will relay from it — a terminal query has no trailing
                # 'S' and simply never gets one (the double-state-redundancy
                # fix). 'initial' type stores this in sl0/sl1 (never otherwise
                # used, since round-0-only queries have no feedback values);
                # 'refine' type stores it in end_sl0/end_sl1 (sl0/sl1 on a
                # refine block always holds that round's feedback STATE values).
                sl0, sl1 = _emit_state_block(HMN_OP_UPDATE)
                claimed_type = rec_blocks_c[pending_query_i]['type']
                key0, key1 = ('sl0', 'sl1') if claimed_type == 'initial' else ('end_sl0', 'end_sl1')
                rec_blocks_c[pending_query_i][key0] = sl0
                rec_blocks_c[pending_query_i][key1] = sl1
                rec_blocks_m[pending_query_i][key0] = sl0
                rec_blocks_m[pending_query_i][key1] = sl1
                pending_query_i = None
            else:
                # Bare 'S' — no immediately-preceding unclaimed 'E' or 'Q' —
                # this is a relay-only no-op hop (formerly a separate 'N' op
                # type). Same relay-read permission as a claimed STATE (see
                # chunk_mask_fb_traj) but no local recall bottleneck rules
                # (nothing to bound a warmup/response region around).
                sl0, sl1 = _emit_state_block(HMN_OP_NOOP)
                op_idx = op_count
                op_count += 1
                rec_blocks_c.append(dict(type='noop', span=None, is_clean=False,
                                         op_idx=op_idx, sl0=sl0, sl1=sl1))
                rec_blocks_m.append(dict(type='noop', span=None, op_idx=op_idx, sl0=sl0, sl1=sl1))

        else:  # 'Q'
            # arg is (span_s, span_e) OR (span_s, span_e, warmup_start) — warmup_start
            # is a BYTE offset within the span where the warmup/query excerpt begins
            # (default 0 = the span's own first warmup_len bytes, today's behavior).
            # Lets a query be "here's a ground-truth excerpt from somewhere in the
            # middle, find it and continue" rather than always "here's the start."
            if len(arg) == 3:
                span_s, span_e, warmup_start = arg
            else:
                span_s, span_e = arg
                warmup_start = 0
            for k in range(span_s, span_e):
                assert k in enc_blocks_c, \
                    f'query span {arg} references chunk {k} which has not been encoded yet — ' \
                    f'causal violation, fix the operations list'
            span_len = (span_e - span_s) * chunk_len
            assert 0 <= warmup_start <= span_len - warmup_len, \
                f'warmup_start={warmup_start} leaves no room for warmup_len={warmup_len} within span_len={span_len}'
            out_len = span_len - warmup_start - warmup_len
            op_idx = op_count
            op_count += 1

            # No pre-filter STATE — warmup/response sit directly at `offset`,
            # attending straight to whatever chunk_mask_fb_traj's hops-permitted
            # sources allow (see module docstring / docs/HISTORY.md §15).
            w0 = offset; w1 = w0 + warmup_len; offset = w1
            c0 = offset; c1 = c0 + out_len; offset = c1

            rw_extra = dict(warmup_start=warmup_start,
                            # (warmup_start, warmup_start) — a degenerate single-value range —
                            # reuses make_batch_tagged's existing rw_xs mechanism to
                            # deterministically place the warmup/query excerpt at warmup_start
                            # rather than always byte 0, with no new batch-filling code needed.
                            # warmup_x_dist stays 'fixed' (rng.integers(x,x+1) always returns x).
                            warmup_train_range=(warmup_start, warmup_start), warmup_x_dist='fixed')
            rec_blocks_c.append(dict(type='initial', span=(span_s, span_e), span_len=span_len,
                                     out_len=out_len, is_clean=(n_refine == 0), op_idx=op_idx,
                                     w0=w0, w1=w1, c0=c0, c1=c1,
                                     sl0=None, sl1=None,  # filled in IF a trailing 'S' claims this
                                                          # Q AND n_refine==0 (no refine rounds follow)
                                     **rw_extra))
            rec_blocks_m.append(dict(type='initial', span=(span_s, span_e), op_idx=op_idx,
                                     w0=w0, w1=w1, c0=c0, c1=c1, sl0=None, sl1=None))
            pending_query_i = len(rec_blocks_c) - 1

            # Refine rounds (docs/HISTORY.md §15's refine-trajectory + boundary-fix
            # derivation): each round is [OP_FEEDBACK][argmax feedback, out_len raw
            # bytes][feedback STATE values][re-presented warmup][new response]. OP_F
            # precedes the argmax content specifically because argmax feedback and
            # the previous round's response are BOTH raw byte-range tokens with
            # nothing else marking the boundary between them (unlike E->STATE or
            # STATE->warmup, which are already vocab-distinguishable) — see the
            # "boundary problem" discussion in docs/HISTORY.md §15. The feedback
            # STATE values themselves need no separate marker (raw-byte->STATE-ID
            # is already vocab-distinguishable). `sl0/sl1` here are ALWAYS the
            # feedback values (not optional, unlike 'initial' type) — a genuinely
            # separate `end_sl0/end_sl1` holds the optional end-of-turn STATE,
            # claimable by a trailing 'S' only on the LAST round emitted.
            prev_c0 = c0
            for _ in range(n_refine):
                opf0 = offset; offset += 1
                am0 = offset; am1 = am0 + out_len; offset = am1
                sl0 = offset; sl1 = sl0 + state_len; offset = sl1
                rw0 = offset; rw1 = rw0 + warmup_len; offset = rw1
                rc0 = offset; rc1 = rc0 + out_len; offset = rc1
                rec_blocks_c.append(dict(type='refine', span=(span_s, span_e), span_len=span_len,
                                         out_len=out_len, is_clean=True, op_idx=op_idx,
                                         opf0=opf0, am0=am0, am1=am1, sl0=sl0, sl1=sl1,
                                         w0=rw0, w1=rw1, c0=rc0, c1=rc1,
                                         argmax_src_c0=prev_c0,
                                         end_sl0=None, end_sl1=None,  # filled in IF a trailing
                                                                      # 'S' claims THIS round
                                         **rw_extra))
                rec_blocks_m.append(dict(type='refine', span=(span_s, span_e), op_idx=op_idx,
                                         opf0=opf0, am0=am0, am1=am1, sl0=sl0, sl1=sl1,
                                         w0=rw0, w1=rw1, c0=rc0, c1=rc1,
                                         end_sl0=None, end_sl1=None))
                prev_c0 = rc0
                pending_query_i = len(rec_blocks_c) - 1

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


def _dual_positions(pos_content: dict, L: int) -> tuple[np.ndarray, np.ndarray]:
    """DEPRECATED (2026-07-28) — `dual_rope` was one of three attempted
    fixes for the `batch`/`interleave_delayed` positional shortcut
    (`kvmem/probe_positional_shortcut.py`); abandoned mid-design in favor
    of `rope_state_scale` (itself now also deprecated, see
    `_scaled_state_positions`'s docstring) — see CLAUDE.md's "Positional
    shortcut" entry and docs/HISTORY.md §12-13. Not wired into any active
    or archived-for-reuse config. Kept correct (see the origin-tracking/
    field-name/None-guard fix below) rather than deleted only because
    deletion of a whole RoPE-clock mechanism wasn't asked for — do not
    build new work on top of this without re-deriving it for the current
    opcode/end-of-turn-STATE design first.

    Builds (pos_state, pos_local) position-ID arrays for apply_rope_dual from
    a chunk_positions_traj-built pos_content. pos_state increments by 1 exactly
    at each STATE region's START position and stays frozen until the next
    STATE region begins — so every position between one STATE emission and
    the next (regardless of how many queries occur there, or in what order)
    sees the SAME macro value. pos_local resets to 0 at the start of every
    encode block (chunk content, continuing through its own STATE) and every
    query "turn" (its own STATE-if-any + warmup + response, continuing
    through refine rounds) — everything else (standalone tag tokens between
    blocks) defaults to pos_local=0.

    Only chunk_positions_traj's block shape (enc_blocks/rec_blocks fields) is
    supported — chunk_positions_hop/_iq_global_rw_tagged/_stitch are NOT
    wired to this yet (dual_rope is only used by the weave_mix branch)."""
    state_starts: list[int] = []
    local_spans: list[tuple[int, int, int]] = []  # (origin, start, end)

    for cb in pos_content['enc_blocks']:
        state_starts.append(cb['sl0'])
        local_spans.append((cb['s0'], cb['s0'], cb['s1']))
        local_spans.append((cb['s0'], cb['sl0'], cb['sl1']))

    # NOTE: a query's own recall-STATE row (rec_blocks' end-of-turn sl0/
    # end_sl0) does NOT advance state_starts — only ENCODING STATE does.
    # This is the whole point: every query following the same encoding pass
    # must see the identical frozen macro value, regardless of query order,
    # or the shortcut measured by kvmem/probe_positional_shortcut.py just
    # reappears one level up (the first bug found here: an earlier version
    # of this function DID increment on rec_blocks' own state row too, and
    # re-created query-order-dependent macro values — caught by direct
    # numerical check before this was trusted).
    #
    # Every rec_block's `origin` is that OP's own `w0` (its turn's actual
    # first token) — NOT its own `sl0`/`end_sl0`, which under the current
    # end-of-turn-STATE design sits AFTER warmup/response, not before (this
    # fn predates that redesign; using sl0 as origin, or leaving it
    # unguarded against a terminal query's sl0=None, both silently broke —
    # see docs/HISTORY.md §15). A 'refine' round shares its op's round-0
    # origin (pos_local continues across refine rounds within one turn,
    # per this function's own docstring), tracked via op_origin below.
    op_origin: dict[int, int] = {}
    for rb in pos_content['rec_blocks']:
        if rb['type'] == 'initial':
            op_origin[rb['op_idx']] = rb['w0']

    for rb in pos_content['rec_blocks']:
        if rb['type'] == 'noop':
            local_spans.append((rb['sl0'], rb['sl0'], rb['sl1']))
        elif rb['type'] == 'initial':
            origin = op_origin[rb['op_idx']]
            if rb['sl0'] is not None:
                local_spans.append((origin, rb['sl0'], rb['sl1']))
            local_spans.append((origin, rb['w0'], rb['w1']))
            local_spans.append((origin, rb['c0'], rb['c1']))
        else:  # 'refine'
            origin = op_origin[rb['op_idx']]
            local_spans.append((origin, rb['opf0'], rb['opf0'] + 1))
            local_spans.append((origin, rb['am0'], rb['am1']))
            local_spans.append((origin, rb['sl0'], rb['sl1']))  # feedback STATE values
            local_spans.append((origin, rb['w0'], rb['w1']))
            local_spans.append((origin, rb['c0'], rb['c1']))
            if rb['end_sl0'] is not None:
                local_spans.append((origin, rb['end_sl0'], rb['end_sl1']))

    pos_state = np.zeros(L, dtype=np.int64)
    starts_sorted = sorted(state_starts)
    macro_val = 0
    si = 0
    for i in range(L):
        while si < len(starts_sorted) and i >= starts_sorted[si]:
            macro_val += 1
            si += 1
        pos_state[i] = macro_val

    pos_local = np.zeros(L, dtype=np.int64)
    for origin, start, end in local_spans:
        idx = np.arange(start, end)
        pos_local[start:end] = idx - origin

    return pos_state, pos_local


def _scaled_state_positions(pos_content: dict, L: int, state_scale: float) -> np.ndarray:
    """DEPRECATED (2026-07-28) — `rope_state_scale`, along with `dual_rope`,
    was one of three attempted fixes for the `batch`/`interleave_delayed`
    positional shortcut (`kvmem/probe_positional_shortcut.py`); all three
    (dual-clock RoPE, this, and `relpos`) either failed outright or were
    abandoned — see CLAUDE.md's "Positional shortcut" entry and
    docs/HISTORY.md §12-13. Not wired into any active or archived-for-reuse
    config. Kept correct (see the field-name/None-guard fix below) rather
    than deleted only because deletion of a whole RoPE-clock mechanism
    wasn't asked for — do not build new work on top of this without
    re-deriving it for the current opcode/end-of-turn-STATE design first.

    Single-clock position array (used with plain apply_rope, NOT
    apply_rope_dual) — every non-STATE token keeps its real, ordinary
    absolute index (identical to plain RoPE, zero special-casing). Every
    STATE-region token gets `(i - s0) + s0 / state_scale`: its position
    WITHIN the region (i - s0) stays a native, unscaled integer (0, 1, 2,
    ..., state_len-1 — full disambiguation power for the region's own
    cyclic-token-ID slots, `_cyclic_state_ids`, which only has
    `state_vocab_size` distinct IDs repeating through `state_len` slots and
    needs SOME undamaged positional signal to tell them apart), while only
    the region's overall BASELINE (s0, where it sits in the whole sequence)
    gets compressed toward negligible.

    BUG this replaced (caught by direct comparison against the original
    hmn_single_recall_c64 baseline's logs, same step count, same task): an
    earlier version divided the ENTIRE real index by state_scale, which
    also crushed the WITHIN-region spacing (native spacing of ~state_len
    collapsed to ~state_len/state_scale) — not just the intended
    cross-region/cross-query distance. That version wasn't just failing to
    fix the query-order shortcut, it was breaking ordinary single-chunk
    recall outright (best val=3.0% vs the baseline's 100% at matched step
    counts, loss stuck ~100x higher than baseline's near-zero). This
    version preserves within-region spacing exactly while still killing
    cross-region distance, verified before trusting (see the offline check
    run before wiring this in)."""
    pos = np.arange(L, dtype=np.float64)
    state_regions: list[tuple[int, int]] = []
    for cb in pos_content['enc_blocks']:
        state_regions.append((cb['sl0'], cb['sl1']))
    for rb in pos_content['rec_blocks']:
        if rb['type'] in ('noop', 'initial'):
            if rb['sl0'] is not None:  # 'initial' is None for a terminal query — no end-of-turn STATE
                state_regions.append((rb['sl0'], rb['sl1']))
        else:  # 'refine' — sl0/sl1 holds the feedback STATE values (always present);
               # end_sl0/end_sl1 holds the optional end-of-turn STATE (last round only)
            state_regions.append((rb['sl0'], rb['sl1']))
            if rb['end_sl0'] is not None:
                state_regions.append((rb['end_sl0'], rb['end_sl1']))
    for s0, s1 in state_regions:
        pos[s0:s1] = (pos[s0:s1] - s0) + s0 / state_scale
    return pos


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

    REDESIGNED (docs/HISTORY.md §15, opcode/no-chat-tag/end-of-turn-STATE
    mechanism) — reuses the same generic rec_block shape (type/op_idx/
    sl0.../c0/c1) `chunk_positions_traj` uses, so `chunk_mask_fb_traj` (keys
    off op_idx, not span) works unchanged. `src0` (absolute byte offset
    into the source) replaces `span` — a chunk-index tuple doesn't apply
    here since windows aren't chunk-aligned. Every query EXCEPT THE LAST
    gets an end-of-turn STATE (opcode=update, claimed implicitly here since
    every non-final query always needs one — the stitch chain's whole
    continuity mechanism depends on the relay, unlike chunk_positions_traj's
    general case where it's conditional on an explicit trailing 'S'); the
    final query is terminal, no end-of-turn STATE (same double-state-
    redundancy rule as everywhere else).

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
    tags: list[tuple[int, int]] = []  # stays empty — no chat tags in this fork
    offset = 0
    for _ in range(n_chunks):
        s0 = offset; s1 = s0 + chunk_len; offset = s1
        sl0 = offset; sl1 = sl0 + state_len + 1; offset = sl1  # +1 for the opcode token
        enc_blocks_c.append(dict(s0=s0, s1=s1, sl0=sl0, sl1=sl1))
        enc_blocks_m.append(dict(s0=s0, s1=s1, sl0=sl0, sl1=sl1))
    enc_end = offset

    rec_blocks_c: list[dict] = []
    rec_blocks_m: list[dict] = []
    for i in range(n_queries):
        src0 = i * src_stride
        out_len = min(src_stride, src_len - warmup_len - src0)  # last query clipped to land exactly on src_len
        assert out_len > 0
        w0 = offset; w1 = w0 + warmup_len; offset = w1
        c0 = offset; c1 = c0 + out_len; offset = c1
        is_last = (i == n_queries - 1)
        if is_last:
            sl0 = sl1 = None
        else:
            sl0 = offset; offset += 1; offset += state_len; sl1 = offset
        rec_blocks_c.append(dict(type='initial', src0=src0, out_len=out_len,
                                 is_clean=True, op_idx=i,
                                 sl0=sl0, sl1=sl1, w0=w0, w1=w1, c0=c0, c1=c1))
        rec_blocks_m.append(dict(type='initial', op_idx=i,
                                 sl0=sl0, sl1=sl1, w0=w0, w1=w1, c0=c0, c1=c1))

    L = offset
    pos_content = dict(enc_blocks=enc_blocks_c, rec_blocks=rec_blocks_c,
                       enc_end=enc_end, warmup_len=warmup_len, L=L)
    pos_mask = dict(enc_blocks=enc_blocks_m, rec_blocks=rec_blocks_m,
                    enc_end=enc_end, warmup_len=warmup_len, L=L)
    return dict(pos_content=pos_content, pos_mask=pos_mask, tags=tags, L=L)


def chunk_mask_fb_traj(pos: dict, hops: int = -1) -> np.ndarray:
    """
    Mask for chunk_positions_traj layouts (REDESIGNED — see docs/HISTORY.md
    §15). No pre-filter STATE row anymore: warmup/response attend DIRECTLY
    to whatever `hops` makes available, via one positive allowlist ("own"
    set) instead of the old three-part chunk-blackout + nochain-blackout +
    bottleneck logic — a row is allowed to see its own local content
    (warmup/response, or the sources an end-of-turn STATE was built from)
    PLUS `allowed_state` (below), and blocked from everything else by
    construction (raw chunks and other ops' content are simply never IN
    the allowlist, no separate blocking rule needed for them). `hops`
    semantics unchanged: op_idx==0 is always exempt (entry point, same role
    as an RNN's h_0=f(x_0)); hops=-1 gives every op permanent access to all
    encoding-pass STATE PLUS the union of every earlier op's own STATE;
    hops>=1 blocks encoding-pass access for op_idx>0, leaving the bounded
    N-op relay window as the ONLY channel.
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

    is_any_enc_state = np.zeros(L, dtype=bool)
    for b in enc_blocks:
        is_any_enc_state |= (c >= b['sl0']) & (c < b['sl1'])

    last_rb_of_op: dict[int, int] = {}
    op_first_w0: dict[int, int] = {}  # each op's FIRST (round-0) w0 — see 'refine' handling below
    for i_rb, rb in enumerate(rec_blocks):
        last_rb_of_op[rb['op_idx']] = i_rb  # last write wins -> last rec_block per relay-producing op
        if rb['type'] == 'initial':
            op_first_w0[rb['op_idx']] = rb['w0']

    def _relay_source(prev_rb: dict) -> tuple[int, int]:
        # 'noop' type always has a real STATE in sl0/sl1 (never optional — a
        # bare relay hop always produces one). 'initial' type stores its
        # (optional) end-of-turn STATE in sl0/sl1 too (never otherwise used
        # by that type). 'refine' type's sl0/sl1 always holds that round's
        # feedback STATE values, so its end-of-turn STATE (if any) lives in
        # the separate end_sl0/end_sl1 fields instead.
        key0, key1 = ('sl0', 'sl1') if prev_rb['type'] in ('initial', 'noop') else ('end_sl0', 'end_sl1')
        assert prev_rb[key0] is not None, (
            f"op_idx={prev_rb['op_idx']} has no end-of-turn STATE (terminal — no trailing "
            f"'S' claimed it) but a later op is trying to relay from it. Fix the operations "
            f"list: add an 'S' right after this op if anything downstream needs to relay from it.")
        return prev_rb[key0], prev_rb[key1]

    def _relay_ranges(op_idx: int) -> list[tuple[int, int]]:
        back_range = range(1, op_idx + 1) if hops == -1 else range(1, hops + 1)
        ranges = []
        for back in back_range:
            src_op = op_idx - back
            if src_op < 0:
                break
            ranges.append(_relay_source(rec_blocks[last_rb_of_op[src_op]]))
        return ranges

    def _allowed_state(op_idx: int) -> np.ndarray:
        """The STATE positions this op's warmup/response/end-of-turn-STATE
        are allowed to attend to directly — permanent encoding-pass access
        if op_idx==0 (entry point) or hops==-1 (unbounded/routing), PLUS
        the relay window for op_idx>0 under any hops setting."""
        allowed = np.zeros(L, dtype=bool)
        if op_idx == 0 or hops == -1:
            allowed |= is_any_enc_state
        if op_idx > 0:
            for lo, hi in _relay_ranges(op_idx):
                allowed |= (c >= lo) & (c < hi)
        return allowed

    # Encoding isolation: an encode chunk's claim-STATE blocked from every
    # OTHER chunk's raw bytes directly (unchanged from before the redesign).
    for k, b in enumerate(enc_blocks):
        sl_row = (r >= b['sl0']) & (r < b['sl1'])
        for j, bj in enumerate(enc_blocks):
            if j == k: continue
            blocked |= sl_row[:, None] & ((c >= bj['s0']) & (c < bj['s1']))[None, :]

    for rb in rec_blocks:
        allowed_state = _allowed_state(rb['op_idx'])

        if rb['type'] == 'noop':
            # No local content — purely a relay pass-through, "own" is just
            # whatever the relay/permanent-access rule permits.
            sl_row = (r >= rb['sl0']) & (r < rb['sl1'])
            blocked |= sl_row[:, None] & ~allowed_state[None, :]
            continue

        if rb['type'] != 'initial':
            continue  # 'refine' handled entirely in its own loop below

        # type == 'initial': warmup/response attend directly to allowed_state +
        # their own local content — no intermediate pre-filter STATE row anymore.
        if rb['w0'] < rb['w1']:
            wm_row = (r >= rb['w0']) & (r < rb['w1'])
            own = allowed_state | (c >= rb['w0']) & (c < rb['w1'])
            blocked |= wm_row[:, None] & ~own[None, :]
        out_row = (r >= rb['c0']) & (r < rb['c1'])
        own = (allowed_state |
               (c >= rb['w0']) & (c < rb['w1']) |
               (c >= rb['c0']) & (c < rb['c1']))
        blocked |= out_row[:, None] & ~own[None, :]

        if rb['sl0'] is not None:
            # This op has an end-of-turn STATE (non-terminal — a later op will
            # relay from it) — built from allowed_state + its OWN warmup+response.
            sl_row = (r >= rb['sl0']) & (r < rb['sl1'])
            own_end = (allowed_state |
                      (c >= rb['w0']) & (c < rb['w1']) |
                      (c >= rb['c0']) & (c < rb['c1']))
            blocked |= sl_row[:, None] & ~own_end[None, :]

    # 'refine' type: no nochain-blackout needed WITHIN one op's own rounds
    # (they're all internal elaboration of the SAME op, not a boundary
    # between different ops — see docs/HISTORY.md §15) — every row here is
    # simply allowed to see allowed_state PLUS everything this op has done
    # so far (from its round-0 w0 onward; causal already caps at the row's
    # own position, so this can never leak into a DIFFERENT op's content,
    # since no other op's positions fall inside [op_start, r] for r within
    # this op's own span).
    for rb in rec_blocks:
        if rb['type'] != 'refine':
            continue
        allowed_state = _allowed_state(rb['op_idx'])
        own = allowed_state | (c >= op_first_w0[rb['op_idx']])

        # OP_FEEDBACK + argmax content treated as one contiguous block for
        # masking purposes (see the boundary-fix rationale in the module's
        # refine docstring above).
        am_row = (r >= rb['opf0']) & (r < rb['am1'])
        blocked |= am_row[:, None] & ~own[None, :]

        sl_row = (r >= rb['sl0']) & (r < rb['sl1'])  # feedback STATE values
        blocked |= sl_row[:, None] & ~own[None, :]

        if rb['w0'] < rb['w1']:
            wm_row = (r >= rb['w0']) & (r < rb['w1'])
            blocked |= wm_row[:, None] & ~own[None, :]
        out_row = (r >= rb['c0']) & (r < rb['c1'])
        blocked |= out_row[:, None] & ~own[None, :]

        if rb['end_sl0'] is not None:  # optional end-of-turn STATE (last round only)
            end_row = (r >= rb['end_sl0']) & (r < rb['end_sl1'])
            blocked |= end_row[:, None] & ~own[None, :]

    visible = causal & ~blocked
    return np.where(visible, 0.0, -1e9).astype(np.float32)


# ---------------------------------------------------------------------------
# Named trajectory patterns (operations-list constructors) for
# chunk_positions_traj. batch/stream/interleave_delayed are train-mix
# candidates; repeat_query/long_hop_recovery are test-only generalization
# probes — training on them would defeat their purpose.
# ---------------------------------------------------------------------------

def traj_batch(n_chunks: int, window_chunks: int = 2) -> list[tuple]:
    # 'S' after every Q but the last — each query but the final one needs an
    # end-of-turn STATE for the NEXT query to relay from (docs/HISTORY.md
    # §15's double-state-redundancy rule: terminal query only, not every
    # query, gets to skip its own STATE). Missing this means chunk_mask_fb_
    # traj's relay lookup raises for op_idx>0 the moment there's more than
    # one query — caught by a smoke test of this exact function before
    # trusting the redesign with multi-query configs.
    q_strs = [f'Q({i},{i + window_chunks})' for i in range(n_chunks - window_chunks + 1)]
    ops, _, _, _, _ = parse_traj_dsl(f'E{n_chunks} ' + ' S '.join(q_strs))
    return ops


def traj_stream(n_chunks: int, window_chunks: int = 2) -> list[tuple]:
    dsl_parts = [f'E{window_chunks}']
    n_q = n_chunks - window_chunks + 1
    for i in range(n_q):
        if i > 0:
            dsl_parts.append('E')
        dsl_parts.append(f'Q({i},{i + window_chunks})')
        if i < n_q - 1:  # every query but the last needs an end-of-turn STATE
            dsl_parts.append('S')
    ops, _, _, _, _ = parse_traj_dsl(' '.join(dsl_parts))
    return ops


def traj_interleave_delayed(n_chunks: int, window_chunks: int = 2) -> list[tuple]:
    spans = [(i, i + window_chunks) for i in range(n_chunks - window_chunks + 1)]
    q_strs = [f'Q({s},{e})' for s, e in reversed(spans)]  # query last span first
    ops, _, _, _, _ = parse_traj_dsl(f'E{n_chunks} ' + ' S '.join(q_strs))
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
    ops, _, _, _, _ = parse_traj_dsl(f'E{n_chunks} Q({n_chunks - window_chunks},{n_chunks})')
    return ops


def traj_locate_and_continue(chunk_len: int, query_start: int) -> list[tuple]:
    """Single unchunked span (n_chunks=1): encode the whole source (exactly
    `chunk_len` bytes, embedded directly via the E(len) DSL token — no
    external chunk_len= config override needed), then a single query whose
    warmup/query excerpt starts at BYTE OFFSET `query_start` within the
    source (not always 0) — the model has to LOCATE this ground-truth
    excerpt (which could sit anywhere) rather than always finding it at a
    fixed, predictable position, then continue generating from right after
    it through the true end of the source. Validity (enough room left for
    warmup_len + a non-trivial response) is enforced by
    chunk_positions_traj's own Q-handling assert — this function's caller
    is responsible for choosing a `query_start` that respects
    `0 <= query_start <= chunk_len - min_recall_len - warmup_len`, see
    kvmem/configs/hmn_single_recall_c64_locate.py for a concrete grid.
    `warmup_len` is still supplied externally (weave_mix's per-entry
    `warmup_len` override) — only chunk_len moved into the DSL string
    itself, since that's the piece this function's whole design is about
    varying freely across a mix without an external key per entry."""
    ops, _, _, _, _ = parse_traj_dsl(f'E({chunk_len}) Q(0,1,{query_start})')
    return ops


def traj_repeat_query(n_chunks: int, window_chunks: int = 2) -> list[tuple]:
    """TEST-ONLY."""
    spans = [(i, i + window_chunks) for i in range(n_chunks - window_chunks + 1)]
    q_strs = [f'Q({s},{e})' for s, e in spans]
    first_s, first_e = spans[0]
    q_strs.append(f'Q({first_s},{first_e})')  # repeated final query — genuinely terminal, no trailing S
    ops, _, _, _, _ = parse_traj_dsl(f'E{n_chunks} ' + ' S '.join(q_strs))
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
# bare relay hop) | S<n> (n bare S ops) | Q(s,e) (query span [s,e), warmup
# excerpt taken from BYTE OFFSET 0 within the span, today's default) |
# Q(s,e,w) (same, but the warmup/query excerpt starts at BYTE OFFSET w
# within the span instead of 0 — response covers everything from w+warmup_len
# through the true end of the span, i.e. out_len is span-length-dependent,
# not a fixed constant; used for "find this ground-truth excerpt wherever it
# sits and continue" tasks, see traj_locate_and_continue) | Q(s,e,w,wl) (same,
# but also sets warmup_len=wl for this trajectory — GLOBAL to the whole
# string, like E(len)/R<n>/B<n>, NOT genuinely per-query: every downstream
# consumer (chunk_positions_traj's out_len math, make_batch_tagged,
# ar_decode_traj_nokv) reads one pos_content['warmup_len'] for the whole
# trajectory, so if more than one Q in a string sets this 4th arg, all must
# agree; falls back to the old external wcfg['warmup_len']/stage-level
# warmup_len if no Q sets it, so existing configs using the dict key are
# unaffected — Q(...,wl) is the preferred/newer form, same precedence
# relationship E(len) has over the old chunk_len= key) | R<n> (n refine
# rounds — GLOBAL, applies uniformly to every Q in the string, not
# per-query; bare R means R1; at most one R token per string) | B<n>
# (repeat_batch for THIS trajectory only — take n gradient steps on the
# same sampled batch before resampling; bare B means B1; at most one B token
# per string; default is 1 if no B token appears — not all trajectory shapes
# are equally easy, so a harder shape in a weave_mix can ask for more
# repeated steps per batch than an easier one mixed alongside it)
#
# Examples: batch "E4 Q(0,2) Q(1,3) Q(2,4)"; stream "E2 Q(0,2) E S Q(1,3) E S
# Q(2,4)"; decay_curve(4 hops) "E2 Q(0,2) S4 Q(0,2)"; one refine round after
# every query "E4 Q(0,2) Q(1,3) Q(2,4) R1"; harder shape gets more repeated
# steps per batch "E8 Q(0,8) B4"; find-and-continue with warmup starting 20
# bytes into a single 64-byte chunk "E1 Q(0,1,20)"; warmup_len=4 embedded
# directly "E(16) Q(0,1,8,4)"
#
# Returns (ops, n_refine, repeat_batch, dsl_chunk_len, dsl_warmup_len) —
# n_refine=0/repeat_batch=1/dsl_chunk_len=None/dsl_warmup_len=None if no
# R/B/E(len)/Q(...,wl) appears. Every internal caller below (traj_batch/
# stream/interleave_delayed/suffix/repeat_query/decay_curve) never emits
# any of R/B/Q(...,wl), so they just discard those extra fields; only a
# config's own explicit `dsl=` string (see the weave_mix dispatch in
# train()) sets n_refine>0 / repeat_batch>1 / dsl_warmup_len today.
# ---------------------------------------------------------------------------

def parse_traj_dsl(s: str) -> tuple[list[tuple], int, int, int | None, int | None]:
    ops: list[tuple] = []
    next_chunk_idx = 0
    n_refine = 0
    repeat_batch = 1
    dsl_chunk_len: int | None = None
    dsl_warmup_len: int | None = None
    seen_r = False
    seen_b = False
    seen_elen = False
    for tok in s.split():
        if tok.startswith('Q('):
            inner = tok[2:-1]
            parts = inner.split(',')
            if len(parts) == 4:
                # Q(s,e,w,wl) — 4th arg sets warmup_len, GLOBAL to the string (like
                # E(len)/R<n>/B<n>) rather than genuinely per-query — every consumer
                # downstream (chunk_positions_traj's out_len math, make_batch_tagged,
                # ar_decode_traj_nokv) reads a single pos_content['warmup_len'] for
                # the whole trajectory, so a per-query value would need a much larger
                # refactor that nothing here actually needs yet (every existing
                # multi-Q trajectory — batch/stream/interleave_delayed — already
                # shares one warmup_len across all its queries). If more than one Q
                # in a string sets this, all must agree.
                s_str, e_str, w_str, wl_str = parts
                ops.append(('Q', (int(s_str), int(e_str), int(w_str))))
                wl_here = int(wl_str)
                assert dsl_warmup_len is None or dsl_warmup_len == wl_here, (
                    f'conflicting warmup_len values from multiple Q(...,wl) in {s!r}: '
                    f'{dsl_warmup_len} vs {wl_here}')
                dsl_warmup_len = wl_here
            elif len(parts) == 3:
                s_str, e_str, w_str = parts
                ops.append(('Q', (int(s_str), int(e_str), int(w_str))))
            else:
                s_str, e_str = parts
                ops.append(('Q', (int(s_str), int(e_str))))
        elif tok.startswith('E('):
            # E(len) — GLOBAL to the string (like R<n>/B<n>), not per-E-token:
            # sets the chunk_len every 'E' op in this string uses, embedded in
            # the DSL itself instead of an external chunk_len= config override.
            # Emits exactly one (E,S) pair, same as bare E. At most one E(len)
            # token per string — chunk_positions_traj only supports a single
            # uniform chunk_len across a whole ops list, so this token sets
            # that one value rather than pretending per-chunk lengths exist.
            assert not seen_elen, f'at most one E(len) token allowed per DSL string, got a second in {s!r}'
            seen_elen = True
            dsl_chunk_len = int(tok[2:-1])
            ops.append(('E', next_chunk_idx))
            ops.append(('S', None))
            next_chunk_idx += 1
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
        elif tok.startswith('B'):
            assert not seen_b, f'at most one B token allowed per DSL string, got a second in {s!r}'
            seen_b = True
            repeat_batch = int(tok[1:]) if len(tok) > 1 else 1
        elif tok.startswith('S'):
            n = int(tok[1:]) if len(tok) > 1 else 1
            for _ in range(n):
                ops.append(('S', None))
        else:
            raise ValueError(f'unrecognized trajectory DSL token: {tok!r}')
    return ops, n_refine, repeat_batch, dsl_chunk_len, dsl_warmup_len


def traj_decay_curve(n_noop_hops: int, window_chunks: int = 2) -> list[tuple]:
    """TEST-ONLY (or train-mix at low hop counts). Query once, take
    n_noop_hops content-free relay hops, then repeat the same query —
    isolates pure relay decay rate from per-hop recall accuracy.

    CONFOUND: produces a much shorter total sequence than checkpoints trained
    with larger n_chunks — zero-shot eval against such a checkpoint can score
    near-0% purely from length extrapolation, not decay. Only trust results
    from a checkpoint actually trained on decay_curve-shaped trajectories.
    """
    ops, _, _, _, _ = parse_traj_dsl(f'E{window_chunks} Q(0,{window_chunks}) S{n_noop_hops} Q(0,{window_chunks})')
    return ops


# =============================================================================
# Attention mask construction
# =============================================================================

def chunk_mask_fb(pos: dict) -> np.ndarray:
    """
    THIN WRAPPER — `chunk_positions_iq_global_rw_tagged` always produces a
    single query (round-0 + optional refine rounds) all sharing `op_idx=0`,
    which `chunk_mask_fb_traj` treats as the ALWAYS-exempt entry point
    (permanent access to every encoding-pass STATE, regardless of `hops`) —
    exactly this function's "global" access semantics, with no relay concept
    needed at all since there is never an op_idx>0 here. `hops` value passed
    is irrelevant (op_idx=0 is exempt from it either way) — kept as -1 for
    clarity.
    """
    return chunk_mask_fb_traj(pos, hops=-1)


def chunk_mask_fb_hop(pos: dict, hops: int = -1) -> np.ndarray:
    """
    THIN WRAPPER — `chunk_positions_hop` now delegates to `chunk_positions_traj`
    (docs/HISTORY.md §15), which numbers rec_blocks by `op_idx` (assigned
    sequentially, one per chain step in order) rather than a separate
    chain-step grouping — identical relay semantics, so `chunk_mask_fb_traj`
    (already verified against multiple hand-derived trajectories) applies
    directly with no adaptation needed.
    """
    return chunk_mask_fb_traj(pos, hops=hops)


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
    seg_start of entry i+1 always equals seg_end of entry i.

    BROKEN under the current end-of-turn-STATE design (2026-07-28 review) —
    disabled below, NOT YET RE-DERIVED. Do not re-enable this without
    redoing the segment-tiling verification the project's own masking-
    change rule requires (CLAUDE.md's "Always verify masking changes
    against the actual attention-mask matrix"). This function's
    `start = rb['sl0']` assumed STATE sits BEFORE warmup/response (the old
    pre-filter-register layout, where sl0 < w0 < c0); under the current
    design a rec_block's STATE (if any) is built AFTER response
    (`sl0 > c1`), so this is wrong for every 'initial' rec_block, not just
    'refine' ones: a terminal query has sl0=None (crashes outright —
    confirmed via direct reproduction), a non-terminal query has
    sl0 > c1 (silently produces seg_start > seg_end instead of crashing).
    A from-scratch attempt at fixing just the terminal case (`start =
    rb['w0'] - 1`, mirroring the enc-block convention) was tried and
    discarded here too — it doesn't actually satisfy this function's own
    contiguity invariant (`seg_start of entry i+1 == seg_end of entry i`,
    checked by `_forward_segmented`'s `assert s0 == L_cached`) without
    further verification this pass didn't have budget for. Raising
    unconditionally is the honest state until someone does that
    verification — see docs/HISTORY.md §15 and CLAUDE.md's
    `forward_granularity` entries for context."""
    raise NotImplementedError(
        "_iter_forward_segments/_forward_segmented (forward_granularity) has not been "
        "re-derived for the current end-of-turn-STATE design (see this function's own "
        "docstring) — every rec_block's segment boundaries are wrong under it, not just "
        "refine/non-terminal ones. Not used by any current or kept-for-future config; "
        "re-derive and re-verify (direct segment/mask inspection, per CLAUDE.md's masking-"
        "change rule) before using forward_granularity again.")


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
    call so the model can't bake a fixed rule into static weights.

    REDESIGNED for the opcode/end-of-turn-STATE mechanism (docs/HISTORY.md
    §15) — every STATE block (`sl0:sl1`, `1+state_len` wide) is filled as
    `[opcode_token, value_0, ..., value_{state_len-1}]`. A query's 'initial'
    rec_block has NO end-of-turn STATE (`sl0 is None`) unless a trailing
    'S' claimed it — in that terminal case there's nothing to fill beyond
    warmup/response.

    'refine' type rec_blocks: fills `am0:am1` with the ROUND'S OWN GROUND
    TRUTH bytes (same content the previous round's response targets) as a
    placeholder — this is NOT the `use_actual_argmax` mechanism (feeding
    the model's own predictions back in), which requires a forward pass
    BETWEEN rounds and doesn't exist in the weave_mix training loop today
    (only the chain_steps/iq_global loops have that injection logic, not
    yet ported — see docs/HISTORY.md §15). Fine for exercising the
    data-construction/masking shapes; NOT a faithful refine-training signal
    until argmax injection is wired into the weave_mix loop too."""
    sids = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
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
        tok[:, b['sl0']] = HMN_OP_UPDATE
        tok[:, b['sl0'] + 1:b['sl1']] = sids

    for rb in pos_content['rec_blocks']:
        if rb['type'] == 'noop':
            tok[:, rb['sl0']] = HMN_OP_NOOP
            tok[:, rb['sl0'] + 1:rb['sl1']] = sids
            continue

        span_s, span_e = rb['span']
        gt = np.concatenate([segs[:, i, :] for i in range(span_s, span_e)], axis=1)

        if rb['type'] == 'initial':
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

            if rb['sl0'] is not None:  # non-terminal: end-of-turn STATE, claimed by a trailing 'S'
                tok[:, rb['sl0']] = HMN_OP_UPDATE
                tok[:, rb['sl0'] + 1:rb['sl1']] = sids

        else:  # 'refine' — see this function's docstring re: placeholder (not real) argmax feedback
            x_min, x_max = rb['warmup_train_range']
            rw_xs = np.array([int(rng.integers(x_min, x_max + 1)) for _ in range(B)])
            tok[:, rb['opf0']] = HMN_OP_FEEDBACK
            for b_idx in range(B):
                X = rw_xs[b_idx]
                tok[b_idx, rb['am0']:rb['am1']] = gt[b_idx, X + wl:X + wl + rb['out_len']]  # placeholder
                tok[b_idx, rb['w0']:rb['w1']]   = gt[b_idx, X:X + wl]
                tok[b_idx, rb['c0']:rb['c1']]   = gt[b_idx, X + wl:X + wl + rb['out_len']]
            # sl0:sl1 is the feedback VALUES ONLY (state_len wide, no opcode slot of
            # its own — OP_FEEDBACK already sits at opf0, before am; see
            # chunk_positions_traj's refine layout).
            tok[:, rb['sl0']:rb['sl1']] = sids
            if rb['end_sl0'] is not None:  # last round, non-terminal: end-of-turn STATE
                tok[:, rb['end_sl0']] = HMN_OP_UPDATE
                tok[:, rb['end_sl0'] + 1:rb['end_sl1']] = sids

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
        tok[:, b['sl0']]        = HMN_OP_UPDATE
        tok[:, b['sl0'] + 1:b['sl1']] = sids

    wl = pos_content['warmup_len']
    for rb in pos_content['rec_blocks']:
        assert rb['type'] == 'initial', 'chunk_positions_stitch only ever produces initial rec_blocks'
        src0 = rb['src0']
        tok[:, rb['w0']:rb['w1']] = src[:, src0:src0 + wl]
        tok[:, rb['c0']:rb['c1']] = src[:, src0 + wl:src0 + wl + rb['out_len']]
        if rb['sl0'] is not None:  # every query but the last (see chunk_positions_stitch)
            tok[:, rb['sl0']]        = HMN_OP_UPDATE
            tok[:, rb['sl0'] + 1:rb['sl1']] = sids

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
    L = pos_content['L']
    full_mask = torch.tensor(mask_np, dtype=torch.float32, device=device)

    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos_content['enc_blocks']):
        tok[b['s0']:b['s1']] = chunks_list[k]
        tok[b['sl0']]        = HMN_OP_UPDATE
        tok[b['sl0'] + 1:b['sl1']] = sids

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

        if rb['type'] == 'refine':
            # Real argmax feedback: _decode_segment for the PREVIOUS round already
            # ran the model and wrote its own greedy predictions into
            # tok[argmax_src_c0:...] (see ar_decode_traj_nokv for the same pattern).
            tok[rb['opf0']] = HMN_OP_FEEDBACK
            src_c0 = rb['argmax_src_c0']
            tok[rb['am0']:rb['am1']] = tok[src_c0:src_c0 + rb['out_len']]
            tok[rb['sl0']:rb['sl1']] = sids  # feedback VALUES only (opcode already at opf0)

        if wl > 0:
            tok[rb['w0']:rb['w1']] = gt_span[warmup_offset:warmup_offset + wl].astype(np.int64)
        # seg_start must equal L_cached exactly — sweeps in every new-content
        # token between the last cached position and c0 automatically.
        _decode_segment(L_cached, rb)

        end_key0 = 'sl0' if rb['type'] == 'initial' else 'end_sl0'
        end_key1 = 'sl1' if rb['type'] == 'initial' else 'end_sl1'
        if rb[end_key0] is not None:  # non-terminal (never happens for this function today,
                                       # every query here is terminal — kept for consistency)
            tok[rb[end_key0]]                = HMN_OP_UPDATE
            tok[rb[end_key0] + 1:rb[end_key1]] = sids

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
    L         = pos_content['L']
    full_mask = torch.tensor(mask_np, dtype=torch.float32, device=device)
    gt_full   = np.concatenate(chunks_list)

    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos_content['enc_blocks']):
        tok[b['s0']:b['s1']] = chunks_list[k]
        tok[b['sl0']]        = HMN_OP_UPDATE
        tok[b['sl0'] + 1:b['sl1']] = sids

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

        if rb['type'] == 'refine':
            # Real argmax feedback: _decode_segment for the PREVIOUS round already
            # ran the model and wrote its own greedy predictions into
            # tok[argmax_src_c0:...], so this is genuinely "feed the model its own
            # prior guess" — no extra plumbing needed (see ar_decode_traj_nokv).
            tok[rb['opf0']] = HMN_OP_FEEDBACK
            src_c0 = rb['argmax_src_c0']
            tok[rb['am0']:rb['am1']] = tok[src_c0:src_c0 + rb['out_len']]
            tok[rb['sl0']:rb['sl1']] = sids  # feedback VALUES only (opcode already at opf0)

        if wl > 0:
            tok[rb['w0']:rb['w1']] = warmup_src
        _decode_segment(rb)

        end_key0 = 'sl0' if rb['type'] == 'initial' else 'end_sl0'
        end_key1 = 'sl1' if rb['type'] == 'initial' else 'end_sl1'
        if rb[end_key0] is not None:  # non-terminal: end-of-turn STATE, built AFTER warmup+response
            tok[rb[end_key0]]                = HMN_OP_UPDATE
            tok[rb[end_key0] + 1:rb[end_key1]] = sids

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
                        tags: list[tuple[int, int]], device,
                        dual_rope: bool = False,
                        rope_state_scale: float | None = None) -> dict:
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
    L         = pos_content['L']
    full_mask = torch.tensor(mask_np, dtype=torch.float32, device=device)
    gt_full   = np.concatenate(chunks_list)

    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos_content['enc_blocks']):
        tok[b['s0']:b['s1']] = chunks_list[k]
        tok[b['sl0']]        = HMN_OP_UPDATE
        tok[b['sl0'] + 1:b['sl1']] = sids

    tag_pos = np.array([p for p, _ in tags], dtype=np.int64)
    tag_ids = np.array([i for _, i in tags], dtype=np.int64)
    tok[tag_pos] = tag_ids

    pos_state_full = pos_local_full = scaled_pos_full = None
    if dual_rope:
        ps, pl = _dual_positions(pos_content, L)
        pos_state_full = torch.tensor(ps, dtype=torch.long, device=device)
        pos_local_full = torch.tensor(pl, dtype=torch.long, device=device)
    if rope_state_scale:
        sp = _scaled_state_positions(pos_content, L, rope_state_scale)
        scaled_pos_full = torch.tensor(sp, dtype=torch.float32, device=device)

    def _fwd_logits_at(pos: int) -> torch.Tensor:
        t = torch.tensor(tok[:pos], dtype=torch.long, device=device)
        m = full_mask[:pos, :pos]
        if dual_rope:
            logits = model(t, m, pos_state=pos_state_full[:pos], pos_local=pos_local_full[:pos])
        elif rope_state_scale:
            logits = model(t, m, offset=scaled_pos_full[:pos])
        else:
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
            tok[rb['sl0']]            = HMN_OP_NOOP  # placeholder only — real content comes from causal attention
            tok[rb['sl0'] + 1:rb['sl1']] = sids
            continue

        span_s, span_e = rb['span']
        gt_span = np.concatenate(chunks_list[span_s:span_e])
        ws = rb.get('warmup_start', 0)  # byte offset within the span (0 = today's default)

        # First-ever occurrence of ANY span starting at byte 0 uses ground
        # truth (nothing decoded yet); every other occurrence (including a
        # REPEATED query of the same span) chains from whatever was most
        # recently decoded for that exact span. (warmup chaining across
        # repeated queries only ever uses ws=0 in practice — a nonzero
        # warmup_start is for the single-shot "find this excerpt" case.)
        if span_s == 0 and rb['span'] not in decoded_by_span:
            warmup_src = gt_span[ws:ws + wl]
        else:
            warmup_src = decoded_by_span.get(rb['span'], gt_span)[ws:ws + wl]

        if rb['type'] == 'refine':
            # REAL argmax feedback here (unlike make_batch_tagged's training-time
            # placeholder) — `_decode_segment` for the PREVIOUS round already ran
            # the model and wrote its own greedy predictions into
            # tok[argmax_src_c0:argmax_src_c0+out_len], so this is genuinely
            # "feed the model its own prior guess," no extra plumbing needed.
            tok[rb['opf0']] = HMN_OP_FEEDBACK
            src_c0 = rb['argmax_src_c0']
            tok[rb['am0']:rb['am1']] = tok[src_c0:src_c0 + rb['out_len']]
            # sl0:sl1 is the feedback VALUES ONLY (state_len wide) — OP_FEEDBACK
            # already sits at opf0, before am (see chunk_positions_traj).
            tok[rb['sl0']:rb['sl1']] = sids

        # No pre-filter STATE to fill before decoding — warmup/response attend
        # directly to whatever hops permits (see docs/HISTORY.md §15).
        if wl > 0:
            tok[rb['w0']:rb['w1']] = warmup_src
        _decode_segment(rb)

        end_key0 = 'sl0' if rb['type'] == 'initial' else 'end_sl0'
        end_key1 = 'sl1' if rb['type'] == 'initial' else 'end_sl1'
        if rb[end_key0] is not None:  # non-terminal: end-of-turn STATE, built AFTER warmup+response
            tok[rb[end_key0]]                = HMN_OP_UPDATE
            tok[rb[end_key0] + 1:rb[end_key1]] = sids

        out_len = rb['out_len']
        decoded_by_span[rb['span']] = np.concatenate([
            warmup_src if wl > 0 else np.array([], dtype=np.int64),
            tok[rb['c0']:rb['c1']],
        ])

        rb_target = gt_span[ws + wl:ws + wl + out_len]
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
        tok[b['s0']:b['s1']] = chunks_list[k]
        tok[b['sl0']]        = HMN_OP_UPDATE
        tok[b['sl0'] + 1:b['sl1']] = sids

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

    def _sweep_known(seg_start: int, seg_end: int):
        """Sweep an already-known (not autoregressively generated) token
        range into the KV cache — used for a non-terminal query's own
        end-of-turn STATE block (opcode+values, filled directly, not
        decoded), replacing the old design's "cache the closing tag" step
        (there's no more closing tag in this fork — see
        chunk_positions_stitch's docstring for the new layout)."""
        nonlocal kv_cache, L_cached
        seg_len = seg_end - seg_start
        seg_t    = torch.tensor(tok[seg_start:seg_end], dtype=torch.long, device=device)
        seg_mask = full_mask[seg_start:seg_end, :L_cached + seg_len]
        _, seg_kv = model(seg_t, seg_mask, past_kv=kv_cache, return_kv=True, offset=L_cached)
        kv_cache  = _cat_kv(kv_cache, seg_kv)
        L_cached += seg_len

    turn_match_pcts: list[float] = []
    prev_response: np.ndarray | None = None

    for rb in pos_content['rec_blocks']:
        src0 = rb['src0']
        warmup_src = gt_full[:wl] if prev_response is None else prev_response[-wl:]

        seg_start = L_cached
        if wl > 0:
            tok[rb['w0']:rb['w1']] = warmup_src

        _decode_segment(seg_start, rb)

        if rb['sl0'] is not None:  # every query but the last (chunk_positions_stitch)
            tok[rb['sl0']]        = HMN_OP_UPDATE
            tok[rb['sl0'] + 1:rb['sl1']] = sids
            _sweep_known(rb['sl0'], rb['sl1'])

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


def apply_rope(x: torch.Tensor, freqs: torch.Tensor, offset: int | torch.Tensor = 0) -> torch.Tensor:
    """x: (..., H, L, d_head)  freqs: (d_head//2,)  offset: position base (int,
    legacy) OR a precomputed (L,) position tensor — lets a caller hand in
    arbitrary per-token position values instead of a plain arange (used by
    apply_rope_dual below)."""
    L, dh  = x.shape[-2], x.shape[-1]
    if isinstance(offset, torch.Tensor):
        pos = offset.to(dtype=torch.float32, device=x.device)
    else:
        pos = torch.arange(offset, offset + L, dtype=torch.float32, device=x.device)
    angles = pos[:, None] * freqs[None, :]
    cos_a, sin_a = angles.cos(), angles.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos_a - x2 * sin_a,
                        x1 * sin_a + x2 * cos_a], dim=-1).reshape(x.shape)


def apply_rope_dual(x: torch.Tensor, freqs: torch.Tensor,
                    pos_state: torch.Tensor, pos_local: torch.Tensor) -> torch.Tensor:
    """Dual-clock RoPE: splits d_head channels into two equal halves — the
    first half rotated by `pos_state` (a per-token macro clock that only
    advances at STATE-emission events, frozen everywhere else — see
    `_dual_positions`), the second half by `pos_local` (resets to 0 at the
    start of every encode/query block). freqs has d_head//2 entries total,
    split evenly (d_head//4 each) between the two halves.

    Motivation: a plain single absolute-position RoPE lets two queries
    following the SAME encoding pass be distinguished purely by their raw
    token-distance to each STATE (whichever query comes second is always
    farther from the first-encoded STATE) — measured directly via
    kvmem/probe_positional_shortcut.py to be an exploited shortcut (91.1%
    match to the WRONG chunk's content when given the right chunk's warmup
    bytes, i.e. the model ignored content entirely and used slot position).
    Freezing pos_state during the query phase removes that shortcut: every
    query following the same encode block sees an IDENTICAL relative
    macro-distance to every STATE regardless of query order, so the only
    remaining basis for attending correctly is the query's own content."""
    dh = x.shape[-1]
    half = dh // 2
    assert half % 2 == 0, 'd_head//2 must itself be even to split freqs evenly'
    nf = freqs.shape[0]
    x_state, x_local = x[..., :half], x[..., half:]
    freqs_state, freqs_local = freqs[:nf // 2], freqs[nf // 2:]
    out_state = apply_rope(x_state, freqs_state, offset=pos_state)
    out_local = apply_rope(x_local, freqs_local, offset=pos_local)
    return torch.cat([out_state, out_local], dim=-1)


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
                offset: int = 0,
                pos_state: torch.Tensor | None = None,
                pos_local: torch.Tensor | None = None) -> torch.Tensor | tuple:
        """pos_state/pos_local: optional (L,) dual-clock position tensors (see
        _dual_positions/apply_rope_dual) — when both given, used INSTEAD of
        the plain `offset` scalar for RoPE. Only meaningful when self.rope."""
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
            if pos_state is not None and pos_local is not None:
                Q = apply_rope_dual(Q, self.freqs, pos_state, pos_local)
                K = apply_rope_dual(K, self.freqs, pos_state, pos_local)
            else:
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
                offset: int = 0,
                pos_state: torch.Tensor | None = None,
                pos_local: torch.Tensor | None = None) -> torch.Tensor | tuple:
        attn_out = self.attn(self.norm(x), mask,
                             past_kv=past_kv, return_kv=return_kv, offset=offset,
                             pos_state=pos_state, pos_local=pos_local)
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
                return_features: bool = False,
                pos_state: torch.Tensor | None = None,
                pos_local: torch.Tensor | None = None) -> torch.Tensor | tuple:
        """
        tokens          : (B, L) or (L,) int64
        mask            : (L_q, L_kv) — L_kv = L_past + L when past_kv given
        past_kv         : list[n_layers] of (K_past, V_past) — cached prefix KV.
                          Only supported for block_type in ('attn_mlp', 'single_attn').
        return_kv       : return (logits, kv_list) instead of just logits
        return_features : return (logits, x) where x is the pre-head residual stream
                          (B, L, d); disables grad_checkpoint to preserve full graph.
        offset          : RoPE position offset (= L_past for suffix pass)
        pos_state/pos_local : optional (L,) dual-clock position tensors (see
                          _dual_positions/apply_rope_dual), covering ONLY this
                          call's L tokens (not the cached prefix) — when given,
                          used INSTEAD of `offset` for RoPE. Not supported with
                          grad_checkpoint (same limitation `offset` already has
                          in that path) or block_type='dual_attn'.

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
        # `offset` may be a full (L,) position tensor (see _scaled_state_positions) —
        # `if offset` on a multi-element tensor raises, so branch on type explicitly.
        _offset = offset if (isinstance(offset, torch.Tensor) or offset) else L_past

        for i, block in enumerate(self.blocks):
            pkv = past_kv[i] if past_kv is not None else None
            use_ckpt = (self.grad_checkpoint and self.training
                        and pkv is None and not return_kv and not return_features)
            if use_ckpt:
                x = _ckpt(block, x, mask, use_reentrant=False)
            else:
                result = block(x, mask, past_kv=pkv,
                               return_kv=return_kv, offset=_offset,
                               pos_state=pos_state, pos_local=pos_local)
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
            early_stop_mean = stage.get('early_stop_mean', None)  # e.g. 80.0 — if val MEAN
                                                                   # reaches this at any eval,
                                                                   # move to the next curriculum
                                                                   # stage immediately instead of
                                                                   # burning the rest of n_steps.
                                                                   # n_steps remains the hard cap
                                                                   # if the threshold is never hit.
            ls_max     = hp.get('ls_max', 0.0)

            _WEAVE_TRAIN_PATTERNS = dict(batch=traj_batch, stream=traj_stream,
                                         interleave_delayed=traj_interleave_delayed,
                                         suffix=traj_suffix)
            hops = stage.get('hops', -1)  # default -1 = unbounded (routing-style); hops=0 is invalid
            dual_rope = hp.get('dual_rope', False)  # see apply_rope_dual/_dual_positions —
                                                     # frozen macro clock during query phase,
                                                     # kills the query-order positional shortcut
                                                     # measured by kvmem/probe_positional_shortcut.py
            rope_state_scale = hp.get('rope_state_scale', None)  # see _scaled_state_positions —
                                                     # single-clock alternative to dual_rope: STATE
                                                     # tokens' real position divided by this factor
                                                     # (e.g. 1e6), everything else normal. Simpler,
                                                     # no per-block reset bookkeeping; supersedes
                                                     # dual_rope as the recommended mechanism (see
                                                     # docs/HISTORY.md §12) — mutually exclusive with it
            assert not (dual_rope and rope_state_scale), 'dual_rope and rope_state_scale are alternatives, not both'

            # Adaptive weave_mix reweighting (merged from the former kvmem/hmn_adaptive_trainer.py
            # fork) — every eval step, re-derive each trajectory's sampling weight from how much
            # it's currently struggling, so training effort automatically shifts toward whichever
            # shape(s) are lagging instead of staying fixed at whatever the config authored up
            # front. Off by default (adaptive=False) — every existing config is unaffected.
            adaptive        = hp.get('adaptive', False)
            adapt_signal    = hp.get('adapt_signal', 'val_match')  # 'val_match' | 'train_loss'
            assert adapt_signal in ('val_match', 'train_loss')
            adapt_temp      = hp.get('adapt_temp', 1.0)  # softmax temperature over normalized
                                                          # difficulty — lower = more aggressive
            adapt_ema_alpha = hp.get('adapt_ema_alpha', 0.5)  # train_loss EMA smoothing (separate
                                                               # from traj['last_loss'], which stays
                                                               # a raw single-batch value for display)
                                                               # — 0.5 (not the more conservative 0.9)
                                                               # so the adapt signal reacts faster to
                                                               # recent change rather than lagging behind it
            adapt_floor     = hp.get('adapt_floor', 0.05)  # min relative weight share even for an
                                                            # already-solved trajectory
            _eval_count     = 0  # gates val_match-driven adaptation until the 2nd eval — the first
                                 # reading is off the least-trained model, too noisy to act on

            weave_mix_cfg = stage['weave_mix']  # list of {weight, pattern[, n_chunks, window_chunks]} OR {weight, dsl}
            trajectories = []
            for wcfg in weave_mix_cfg:
                if 'dsl' in wcfg:
                    # Explicit DSL string — bypasses the named-pattern lookup entirely, e.g.
                    # dsl='E4 Q(0,2) Q(1,3) Q(2,4)' (see parse_traj_dsl's grammar comment).
                    # n_chunks is derived from the DSL itself (count of 'E' ops), not passed
                    # separately — the string is already the single source of truth for shape.
                    pname = wcfg['dsl']  # used only for logging/bookkeeping below
                    ops, w_n_refine, w_repeat_batch, w_dsl_chunk_len, w_dsl_warmup_len = parse_traj_dsl(wcfg['dsl'])
                    w_n_chunks = sum(1 for op, _ in ops if op == 'E')
                    # DSL-embedded E(len)/W<n> win if present; fall back to the old
                    # external wcfg['chunk_len']/['warmup_len'] override, then the stage
                    # default — E(len)/W<n> are the preferred/newer form (see
                    # parse_traj_dsl's grammar comment), the wcfg keys stay supported
                    # for existing configs.
                    w_chunk_len = w_dsl_chunk_len if w_dsl_chunk_len is not None else wcfg.get('chunk_len', chunk_len)
                    w_warmup_len = w_dsl_warmup_len if w_dsl_warmup_len is not None else wcfg.get('warmup_len', warmup_len)
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
                    w_repeat_batch = wcfg.get('repeat_batch', 1)
                    w_chunk_len = wcfg.get('chunk_len', chunk_len)
                    w_warmup_len = wcfg.get('warmup_len', warmup_len)
                    ops = _WEAVE_TRAIN_PATTERNS[pname](w_n_chunks, w_window_chunks)
                built = chunk_positions_traj(w_chunk_len, state_len, w_warmup_len, ops,
                                             n_refine=w_n_refine, state_vocab_size=state_vocab_size)
                pos_content, pos_mask, tags, L = (built['pos_content'], built['pos_mask'],
                                                  built['tags'], built['L'])
                mask_np = chunk_mask_fb_traj(pos_mask, hops=hops)
                mask_t  = torch.tensor(mask_np, dtype=torch.float32, device=device)
                pos_state_t = pos_local_t = scaled_pos_t = None
                if dual_rope:
                    ps, pl = _dual_positions(pos_content, L)
                    pos_state_t = torch.tensor(ps, dtype=torch.long, device=device)
                    pos_local_t = torch.tensor(pl, dtype=torch.long, device=device)
                if rope_state_scale:
                    sp = _scaled_state_positions(pos_content, L, rope_state_scale)
                    scaled_pos_t = torch.tensor(sp, dtype=torch.float32, device=device)
                trajectories.append(dict(weight=wcfg['weight'], pattern=pname, n_chunks=w_n_chunks,
                                         chunk_len=w_chunk_len,
                                         pos_content=pos_content, mask_np=mask_np, mask_t=mask_t,
                                         tags=tags, L=L, repeat_batch=w_repeat_batch,
                                         pos_state_t=pos_state_t, pos_local_t=pos_local_t,
                                         scaled_pos_t=scaled_pos_t,
                                         base_weight=wcfg['weight'], ema_loss=None, last_match=None))
            traj_weights = np.array([t['weight'] for t in trajectories], dtype=np.float64)
            traj_weights = traj_weights / traj_weights.sum()

            def _log_weave_mix(tag=''):
                # One entry per line (not a single-line tuple-list repr) — with dense
                # mixes now running 25-51 entries (hmn_locate_nope_curriculum_dense.py),
                # the old one-liner wrapped into an unreadable wall of text.
                name_w = max(len(t['pattern']) for t in trajectories)
                lines = [f'    {w:5.2f}  {t["pattern"]:<{name_w}}  repeat_batch={t["repeat_batch"]}'
                         for t, w in zip(trajectories, traj_weights)]
                _log(f'\n[stage {stage_i}] weave_mix{tag}  '
                     f'chunk_len={chunk_len} state={state_len} wl={warmup_len} '
                     f'hops={hops}  B={B}  steps={n_steps}')
                _log('\n'.join(lines))
            _log_weave_mix()

            def _temp_softmax_rescale(diffs):
                """softmax(diffs/adapt_temp), rescaled so a perfectly uniform difficulty
                maps every trajectory back to d=1.0 — preserves the convention the
                floor-blending formula below assumes. Lower adapt_temp => peakier
                softmax => more aggressive reallocation."""
                n = len(diffs)
                scores = diffs / adapt_temp
                scores = scores - scores.max()  # shift for numerical stability; softmax is shift-invariant
                exp_s = np.exp(scores)
                p = exp_s / exp_s.sum()
                return p * n

            def _adapt_reweight():
                """Recompute traj_weights (sampling probability only — repeat_batch
                stays fixed at whatever the config set) from the chosen signal.
                difficulty is normalized to the mix's own mean so a trajectory
                exactly at the mix's average difficulty gets its base_weight back
                unchanged; harder-than-average trajectories get scaled up, easier
                ones scaled down (never below adapt_floor's relative share)."""
                if adapt_signal == 'val_match':
                    diffs = np.array([max(100.0 - (t['last_match'] if t['last_match'] is not None else 50.0), 0.0)
                                      for t in trajectories])
                else:
                    known = [t['ema_loss'] for t in trajectories if t['ema_loss'] is not None]
                    fallback = (sum(known) / len(known)) if known else 1.0
                    diffs = np.array([t['ema_loss'] if t['ema_loss'] is not None else fallback
                                      for t in trajectories])
                diffs = diffs / max(diffs.mean(), 1e-8)  # 1.0 = mix-average difficulty
                diffs = _temp_softmax_rescale(diffs)
                for t, d in zip(trajectories, diffs):
                    t['weight'] = t['base_weight'] * (adapt_floor + (1 - adapt_floor) * d)
                new_weights = np.array([t['weight'] for t in trajectories], dtype=np.float64)
                return new_weights / new_weights.sum()

            lr_min      = hp.get('lr_min', 0.0)
            cosine_T0   = hp.get('cosine_T0', 20000)
            cosine_Tmul = hp.get('cosine_T_mult', 1)
            lr_schedule = hp.get('lr_schedule', 'constant')

            def _lr(s):
                if s <= warmup_steps:
                    return lr_max * s / max(warmup_steps, 1)
                if adaptive:
                    # Fixed after linear warmup, NOT cosine, when adaptive reweighting
                    # is on — a decaying schedule assumes the training signal gets
                    # easier to fit over time; here the signal itself keeps shifting
                    # as reweighting moves effort around, so annealing lr toward ~0
                    # would blunt the model's ability to respond to a newly
                    # up-weighted hard trajectory. cosine_T0/lr_schedule/lr_min are
                    # ignored in this case.
                    return lr_max
                if lr_schedule != 'cosine_restarts':
                    return lr_max
                t = s - warmup_steps
                T_i = cosine_T0
                while t >= T_i:
                    t -= T_i
                    T_i = int(T_i * cosine_Tmul)
                return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t / max(T_i, 1)))

            stage_best_val = -1.0
            _cached_batch = None  # (traj, pos_content, mask_t, tags, tok_t)
            _cached_repeat_left = 0  # steps remaining on the cached batch — per-trajectory, see traj['repeat_batch']
            pbar = tqdm(range(1, n_steps + 1), desc=f'stage{stage_i}', dynamic_ncols=True, file=status_file)
            for local_step in pbar:
                global_step += 1
                lr = _lr(local_step)
                for pg in opt.param_groups: pg['lr'] = lr

                model.train(); opt.zero_grad()

                if _cached_batch is None or _cached_repeat_left <= 0:
                    traj = trajectories[rng.choice(len(trajectories), p=traj_weights)]
                    t_pos_content, t_mask_t, t_tags = traj['pos_content'], traj['mask_t'], traj['tags']
                    tok_np = make_batch_tagged(rng, B, traj['n_chunks'], traj['chunk_len'], state_len, state_vocab_size,
                                               t_pos_content, t_tags, data_kind=data_kind,
                                               data_target_bits=data_target_bits)
                    tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)
                    _cached_batch = (traj, t_pos_content, t_mask_t, t_tags, tok_t)
                    _cached_repeat_left = traj['repeat_batch']
                else:
                    traj, t_pos_content, t_mask_t, t_tags, tok_t = _cached_batch
                _cached_repeat_left -= 1

                if forward_granularity is not None:
                    assert not dual_rope and not rope_state_scale, \
                        'dual_rope/rope_state_scale + forward_granularity not yet supported together'
                    loss = _forward_segmented(model, tok_t, traj['mask_np'], t_pos_content,
                                              device, ls_max, forward_granularity,
                                              segment_checkpoint=segment_checkpoint)
                else:
                    if dual_rope:
                        logits = model(tok_t, t_mask_t, pos_state=traj['pos_state_t'], pos_local=traj['pos_local_t'])
                    elif rope_state_scale:
                        logits = model(tok_t, t_mask_t, offset=traj['scaled_pos_t'])
                    else:
                        logits = model(tok_t, t_mask_t)
                    nlls = []
                    for rb in t_pos_content['rec_blocks']:
                        if not rb['is_clean']:
                            continue
                        # w0:c1 instead of c0:c1 — NLL now covers the warmup region too,
                        # not just the response/continuation (this fork's 2nd change, see
                        # module docstring). w0 always precedes c0 directly (no gap, no
                        # tag in between in this fork's tag-free layout), so this is a
                        # straight causal-shift extension of the same log_softmax slicing.
                        lp  = F.log_softmax(logits[:, rb['w0'] - 1:rb['c1'] - 1], dim=-1)
                        tgt = tok_t[:, rb['w0']:rb['c1']]
                        nll_per = _positional_ls_nll(lp, tgt, ls_max)
                        nlls.append(nll_per.mean())
                    loss = torch.stack(nlls).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                loss_f = float(loss.detach())
                # Per-trajectory loss — RAW, from whatever single batch was last
                # sampled for that entry (no smoothing) — lets you watch each DSL
                # entry's own loss directly (e.g. spot a rehearsal entry regressing/
                # being forgotten as later stages/entries dominate sampling), not
                # just the aggregate loss which only ever reflects whichever entry
                # was sampled this exact step.
                traj['last_loss'] = loss_f
                traj['ema_loss'] = loss_f if traj['ema_loss'] is None else (
                    adapt_ema_alpha * traj['ema_loss'] + (1 - adapt_ema_alpha) * loss_f)
                _traj_loss_str = '[' + ','.join(f'{t["last_loss"]:.2f}' if t.get('last_loss') is not None else 'NA'
                                                for t in trajectories) + ']'
                pbar.set_postfix(loss=f'{loss_f:.3f}', lr=f'{lr:.1e}', traj_loss=_traj_loss_str, refresh=False)
                if local_step % log_every == 0:
                    _jlog(dict(step=global_step, loss=round(loss_f, 5), lr=lr,
                               traj_loss=[round(t['last_loss'], 4) if t.get('last_loss') is not None else None
                                         for t in trajectories]))
                    print(str(pbar), file=log_file, flush=True)

                if local_step % stage_eval_every == 0 or local_step == n_steps:
                    model.eval()
                    elapsed = time.time() - t_start
                    h, m = divmod(int(elapsed), 3600); m, s = divmod(m, 60)
                    _log(f'\n--- stage={stage_i} step={local_step}/{n_steps}'
                         f'  g={global_step}  loss={loss_f:.4f}  {h:02d}:{m:02d}:{s:02d} ---')

                    val_means = []
                    for traj in trajectories:
                        val_seqs = make_test_sequences(traj['n_chunks'] * traj['chunk_len'])
                        val_n_seqs = hp.get('val_n_seqs')
                        if val_n_seqs is not None:
                            val_seqs = dict(list(val_seqs.items())[:val_n_seqs])
                        pcts = []
                        for sname, seq_bytes in val_seqs.items():
                            chunks_list = [seq_bytes[k * traj['chunk_len']:(k + 1) * traj['chunk_len']]
                                          for k in range(traj['n_chunks'])]
                            r = ar_decode_traj_nokv(model, np.array(chunks_list), state_len,
                                                    state_vocab_size, traj['mask_np'],
                                                    traj['pos_content'], traj['tags'], device,
                                                    dual_rope=dual_rope, rope_state_scale=rope_state_scale)
                            pcts.append(r['match_pct'])
                        m_ = sum(pcts) / len(pcts)
                        traj['last_match'] = m_
                        val_means.append(m_)
                        _ema_disp = f'{traj["ema_loss"]:.3f}' if traj['ema_loss'] is not None else 'NA'
                        _log(f'  val/weave/{traj["pattern"]:<20} match={m_:.1f}%  ema_loss={_ema_disp}')
                    vmean = sum(val_means) / len(val_means)
                    _log(f'  val/weave/MEAN               match={vmean:.1f}%')
                    _jlog(dict(step=global_step, stage=stage_i, eval_mean=round(vmean, 2),
                               traj_match=[round(t['last_match'], 2) for t in trajectories]))
                    _eval_count += 1

                    if adaptive:
                        # val_match's very first reading is the noisiest possible signal
                        # (least-trained model) — skip adapting on it, wait for the 2nd
                        # eval. train_loss doesn't have this problem (its EMA has already
                        # been accumulating every step since step 1).
                        if adapt_signal == 'train_loss' or _eval_count >= 2:
                            traj_weights = _adapt_reweight()
                            _log_weave_mix(' (adapted)')
                        else:
                            _log(f'  [stage {stage_i}] adaptive=True but adapt_signal=val_match '
                                 f'skips adapting until the 2nd eval (this is eval #{_eval_count})')

                    torch.save(dict(model=model.state_dict(), hp=hp, step=global_step),
                              os.path.join(ckpt_dir, f'stage{stage_i}_last.pt'))
                    if vmean > stage_best_val:
                        stage_best_val = vmean
                        torch.save(dict(model=model.state_dict(), hp=hp, step=global_step, val_mean=vmean),
                                  os.path.join(ckpt_dir, f'stage{stage_i}_best.pt'))

                    if early_stop_mean is not None and vmean >= early_stop_mean:
                        _log(f'  [stage {stage_i}] EARLY STOP: val MEAN {vmean:.1f}% >= '
                             f'early_stop_mean={early_stop_mean} at step {local_step}/{n_steps} '
                             f'— moving to next stage now instead of burning the remaining steps.')
                        break

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
