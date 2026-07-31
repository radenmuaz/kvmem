"""
JAX/Flax NNX port of `kvmem/hmn.py` — SINGLE FILE, self-contained (no import
of `kvmem.hmn`, no `torch` dependency at all). Motivated by the TPU-port
investigation in CLAUDE.md's "TPU port" section: an unresolved,
data-dependent `loss=NaN` on real XLA/TPU hardware (bug 5) that never
reproduces under any CPU emulation, survived ruling out padding,
`grad_checkpoint`, bf16-vs-fp32, `rope`/`state_vocab_size`, and batch size
individually (each ablation confounded by also shifting the RNG stream —
see that section for the full, honest account). `torch_xla` is one
PyTorch-on-XLA bridge among several; JAX is XLA's native, first-party
frontend — trading `torch_xla`'s bridge layer (a plausible fault surface
for exactly this kind of hardware-only numerical bug) for JAX's own,
independently-implemented XLA lowering is a real, different data point,
not just a rewrite for its own sake.

**Scope, deliberately narrow**: only `block_type='single_attn'` (the
project's own default going forward) with `rope`+`yarn`, `null_kv`,
`rmsnorm` — the exact feature set `hmn_notags_w25_rope.py` uses, no more.
`attn_mlp`/`dual_attn`, `qk_norm`, `logit_cap`, `attn_temp`, `embed_scale`,
`tied_embed`, `gated_ffn`, refine rounds/argmax feedback, structured
(non-random) data, adaptive reweighting, and length bucketing/padding are
NOT ported. **KV-caching, gradient checkpointing (`remat`), decode-eval
(match%), and checkpoint save/load ARE ported** (added 2026-07-30, for
feature parity with `kvmem.hmn`'s own `train()` within this file's scope):
`HMNModel.__call__`'s `past_kv`/`return_kv`/`offset` signature mirrors
`kvmem.hmn.HMNModel.forward`'s exactly; `ar_decode_traj_nokv`/`ar_decode_
traj_kv` are the full-recompute and KV-cached decode counterparts (the
`_kv` one is new — `kvmem.hmn`'s own KV-cached decoders target other
position layouts, not `chunk_positions_traj`); `save_checkpoint`/`load_
checkpoint` mirror the `stage{i}_last/best/end.pt` pattern (pickle instead
of `torch.save`/orbax — no new dependency, and a PyTorch checkpoint and a
JAX one were never going to be interchangeable regardless of format, since
the two models don't share a state_dict layout).

**What's copied verbatim vs. reimplemented**: `chunk_positions_traj`,
`chunk_mask_fb_traj`, `parse_traj_dsl`, and `make_batch_tagged` (minus its
`data_kind != 'random'` structured-data branch) are copied byte-for-byte
from `kvmem/hmn.py` — they're pure NumPy/Python, no `torch` involved, so
copying (not importing) makes this file genuinely standalone while staying
numerically identical to the PyTorch pipeline's own batch/mask
construction. `load_config` is likewise a plain copy. The model
(`MHAttention`/`RMSNorm`/`SingleAttnBlock`/`HMNModel`/`build_model`) and the
optimization loop (`train_jax`) are the actual new code.

**Same signature, translated to JAX idiom**: `build_model(hp, rngs) ->
HMNModel` mirrors `kvmem.hmn.build_model(hp, device) -> HMNModel`; the
model's `__call__(tokens, mask)` takes the same `(B, L)` int token array and
`(L, L)` additive-bias float mask, returning the same `(B, L, V_out)`
logits shape. `rngs` (an `nnx.Rngs`) replaces PyTorch's implicit global RNG.

`train_jax(hp, ...)` only handles the non-refine, single-`Q`-per-entry case
every `_w25*`/Run-A-style config uses (`is_clean` always True, one `(w0,c1)`
NLL slice per trajectory). No label smoothing (plain NLL). Its periodic eval
(`eval_every`, `val_n_seqs`) DOES report real byte-exact match% now (via
`ar_decode_traj_kv`), directly comparable to `kvmem.hmn.train()`'s own
`val/weave/*` numbers.

Run (never two jobs at once — this is a local/CPU sanity run, not a real
training schedule; `jax-mps`, this machine's Metal backend, has its own
unrelated bug, `TypeError: aval_to_ir_type() missing 1 required positional
argument` inside `jax_plugins/mps/ops.py`'s RNG lowering, marked
experimental upstream — force CPU):
    JAX_PLATFORMS=cpu python3 -m kvmem.hmn_jax
    JAX_PLATFORMS=cpu python3 -m kvmem.hmn_jax --config kvmem/configs/hmn_notags_w25_rope_jax.py
"""
import argparse
import importlib.util
import inspect
import json
import math
import os
import time

from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

# Persistent XLA compilation cache — compiled executables for a given (shape,
# architecture) survive across process restarts (e.g. a killed/relaunched
# training run hitting the same bucket shapes again), instead of recompiling
# from scratch every time. Directory is overridable via JAX_CACHE_DIR (e.g.
# to point at a persistent disk on a TPU VM); defaults to /tmp so it's a free
# win even with zero configuration. Safe to enable unconditionally — a cache
# miss just falls back to a normal compile.
jax.config.update('jax_compilation_cache_dir', os.environ.get('JAX_CACHE_DIR', '/tmp/jax_cache'))
jax.config.update('jax_persistent_cache_min_compile_time_secs', 1)
jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)

# =============================================================================
# Vocab constants — copied verbatim from kvmem.hmn (kvmem/hmn.py:161-185)
# =============================================================================

HMN_OP_UPDATE   = 256
HMN_OP_NOOP     = 257
HMN_OP_FEEDBACK = 258
HMN_STATE_0     = 259
HMN_TAG_VOCAB_SIZE = 271


def _cyclic_state_ids(state_len: int, state_vocab_size: int = 2) -> list[int]:
    assert state_vocab_size >= 1
    return [HMN_STATE_0 + (i % state_vocab_size) for i in range(state_len)]


# =============================================================================
# Trajectory DSL + position/mask builders — copied verbatim from kvmem.hmn
# (chunk_positions_traj, chunk_mask_fb_traj, parse_traj_dsl — pure NumPy/
# Python, no torch involved in any of these three). See kvmem/hmn.py's own
# copies for the full design-rationale docstrings (the redesign history,
# docs/HISTORY.md §15-17 references, etc.) — trimmed here to save space,
# not because the rationale doesn't apply; this IS that same code.
# =============================================================================

def chunk_positions_traj(chunk_len: int, state_len: int, warmup_len: int,
                         operations: list[tuple], n_refine: int = 0,
                         state_vocab_size: int = 2) -> dict:
    enc_blocks_c: dict[int, dict] = {}
    enc_blocks_m: dict[int, dict] = {}
    tags: list[tuple[int, int]] = []
    offset = 0

    rec_blocks_c: list[dict] = []
    rec_blocks_m: list[dict] = []
    op_count = 0

    pending_chunk_idx: int | None = None
    pending_query_i: int | None = None

    def _emit_state_block(opcode: int):
        nonlocal offset
        sl0 = offset
        offset += 1
        offset += state_len
        sl1 = offset
        return sl0, sl1

    for op, arg in operations:
        if op == 'E':
            chunk_idx = arg
            assert chunk_idx not in enc_blocks_c, f'chunk {chunk_idx} encoded twice'
            assert pending_chunk_idx is None, \
                f"chunk {pending_chunk_idx}'s 'E' was never followed by 'S' before chunk {chunk_idx}'s 'E'"
            s0 = offset; s1 = s0 + chunk_len; offset = s1
            enc_blocks_c[chunk_idx] = dict(s0=s0, s1=s1)
            enc_blocks_m[chunk_idx] = dict(s0=s0, s1=s1)
            pending_chunk_idx = chunk_idx

        elif op == 'S':
            if pending_chunk_idx is not None:
                sl0, sl1 = _emit_state_block(HMN_OP_UPDATE)
                enc_blocks_c[pending_chunk_idx]['sl0'] = sl0
                enc_blocks_c[pending_chunk_idx]['sl1'] = sl1
                enc_blocks_m[pending_chunk_idx]['sl0'] = sl0
                enc_blocks_m[pending_chunk_idx]['sl1'] = sl1
                pending_chunk_idx = None
            elif pending_query_i is not None:
                assert rec_blocks_c[pending_query_i]['sl0'] is None, \
                    'pending_query_i should only ever point at a round whose sl0 is still unset'
                sl0, sl1 = _emit_state_block(HMN_OP_UPDATE)
                rec_blocks_c[pending_query_i]['sl0'] = sl0
                rec_blocks_c[pending_query_i]['sl1'] = sl1
                rec_blocks_m[pending_query_i]['sl0'] = sl0
                rec_blocks_m[pending_query_i]['sl1'] = sl1
                pending_query_i = None
            else:
                sl0, sl1 = _emit_state_block(HMN_OP_NOOP)
                op_idx = op_count
                op_count += 1
                rec_blocks_c.append(dict(type='noop', span=None, is_clean=False,
                                         op_idx=op_idx, sl0=sl0, sl1=sl1))
                rec_blocks_m.append(dict(type='noop', span=None, op_idx=op_idx, sl0=sl0, sl1=sl1))

        else:  # 'Q'
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

            w0 = offset; w1 = w0 + warmup_len; offset = w1
            c0 = offset; c1 = c0 + out_len; offset = c1

            rw_extra = dict(warmup_start=warmup_start,
                            warmup_train_range=(warmup_start, warmup_start), warmup_x_dist='fixed')
            if n_refine > 0:
                sl0, sl1 = _emit_state_block(HMN_OP_UPDATE)
            else:
                sl0 = sl1 = None
            rec_blocks_c.append(dict(type='initial', span=(span_s, span_e), span_len=span_len,
                                     out_len=out_len, is_clean=(n_refine == 0), op_idx=op_idx,
                                     w0=w0, w1=w1, c0=c0, c1=c1,
                                     sl0=sl0, sl1=sl1,
                                     **rw_extra))
            rec_blocks_m.append(dict(type='initial', span=(span_s, span_e), op_idx=op_idx,
                                     w0=w0, w1=w1, c0=c0, c1=c1, sl0=sl0, sl1=sl1))
            pending_query_i = len(rec_blocks_c) - 1

            gt_c0 = c0
            prev_c0 = c0
            for round_i in range(n_refine):
                wa0 = offset; wa1 = wa0 + warmup_len; offset = wa1
                opf0 = offset; offset += 1
                am0 = offset; am1 = am0 + out_len; offset = am1
                fsl0 = offset; fsl1 = fsl0 + state_len; offset = fsl1
                rw0 = offset; rw1 = rw0 + warmup_len; offset = rw1
                rc0 = offset; rc1 = rc0 + out_len; offset = rc1
                is_last_round = (round_i == n_refine - 1)
                if not is_last_round:
                    sl0, sl1 = _emit_state_block(HMN_OP_UPDATE)
                else:
                    sl0 = sl1 = None
                rec_blocks_c.append(dict(type='refine', span=(span_s, span_e), span_len=span_len,
                                         out_len=out_len, is_clean=True, op_idx=op_idx,
                                         wa0=wa0, wa1=wa1, opf0=opf0, am0=am0, am1=am1,
                                         fsl0=fsl0, fsl1=fsl1,
                                         w0=rw0, w1=rw1, c0=rc0, c1=rc1,
                                         sl0=sl0, sl1=sl1,
                                         argmax_src_c0=prev_c0, gt_c0=gt_c0,
                                         **rw_extra))
                rec_blocks_m.append(dict(type='refine', span=(span_s, span_e), op_idx=op_idx,
                                         wa0=wa0, wa1=wa1, opf0=opf0, am0=am0, am1=am1,
                                         fsl0=fsl0, fsl1=fsl1,
                                         w0=rw0, w1=rw1, c0=rc0, c1=rc1,
                                         sl0=sl0, sl1=sl1))
                prev_c0 = rc0
                pending_query_i = len(rec_blocks_c) - 1

    assert pending_chunk_idx is None, \
        f"chunk {pending_chunk_idx}'s 'E' was never followed by 'S' — every 'E' needs an 'S' right after it"

    L = offset
    enc_end = max((b['s1'] for b in enc_blocks_c.values()), default=0)

    chunk_idx_order = sorted(enc_blocks_c.keys())
    enc_blocks_c_list = [enc_blocks_c[k] for k in chunk_idx_order]
    enc_blocks_m_list = [enc_blocks_m[k] for k in chunk_idx_order]

    pos_content = dict(enc_blocks=enc_blocks_c_list, rec_blocks=rec_blocks_c,
                       enc_end=enc_end, warmup_len=warmup_len, L=L)
    pos_mask = dict(enc_blocks=enc_blocks_m_list, rec_blocks=rec_blocks_m,
                    enc_end=enc_end, warmup_len=warmup_len, L=L)

    return dict(pos_content=pos_content, pos_mask=pos_mask, tags=tags, L=L)


def chunk_mask_fb_traj(pos: dict, hops: int = -1, enc_hops: int = -1,
                       enc_active_backs: dict | None = None) -> np.ndarray:
    """`enc_hops`/`enc_active_backs` — opt-in (default -1 = fully unchanged
    behavior, byte-identical to before this parameter existed) windowing of
    the ENCODING-CHUNK sequence, orthogonal to `hops`/`_relay_ranges` (which
    governs relay between QUERY OPS and is a no-op for any single-Q
    trajectory, since op_idx is always 0 there and `op_idx==0` is
    unconditionally exempt).

    Motivation: a single-query design (e.g. `hmn_stitch_src1024_anchor`'s
    suffix recall) gives its one query permanent, unbounded attention to
    EVERY encoded chunk's STATE (`_allowed_state(0)` always includes
    `is_any_enc_state`) — `hops` has nothing to bound in that design. This
    generalizes the same "bounded N-back window, back=1 never dropped"
    relay concept that already governs op-to-op relay (see `hops` in
    CLAUDE.md) to the CHUNK sequence itself: with `enc_hops=N`, chunk k's
    own STATE computation may attend to at most the previous N chunks'
    STATE (not just its own raw bytes — "encoding isolation" for RAW bytes
    is untouched, only cross-chunk STATE-to-STATE visibility is windowed),
    and the query is windowed the same way against the LAST N chunks
    (treating the query as a virtual chunk immediately following the last
    real one). `enc_active_backs` (dict: chunk index -> set of active back
    distances, or the string key `'query'` for the query itself) lets a
    caller further restrict an already-windowed back-range — used both for
    per-step stochastic hop DROPOUT during training (back=1 always present,
    back 2..N independently kept/dropped) and for combinatorial per-hop-size
    eval (decode once per subset of active backs). Absent, every back
    distance 1..enc_hops is active (the deterministic, no-dropout window).
    """
    if hops == 0:
        raise ValueError("hops=0 is invalid — use hops=-1 for unbounded "
                         "(routing-style, full access to every prior op's "
                         "STATE and the encoding pass) or hops>=1 for a "
                         "bounded N-op recurrent window.")
    if enc_hops != -1:
        assert enc_hops >= 1, "enc_hops must be -1 (unbounded, legacy) or >=1"
    L = pos['L']
    r = np.arange(L); c = np.arange(L)
    causal  = c[None, :] <= r[:, None]
    blocked = np.zeros((L, L), dtype=bool)

    enc_blocks = pos['enc_blocks']
    rec_blocks = pos['rec_blocks']

    is_any_enc_state = np.zeros(L, dtype=bool)
    for b in enc_blocks:
        is_any_enc_state |= (c >= b['sl0']) & (c < b['sl1'])

    def _enc_window_cols(backs_key) -> np.ndarray:
        """Columns of the up-to-`enc_hops` preceding chunks' STATE, counted
        back from chunk index `backs_key` (an int) or from one-past-the-last
        chunk when `backs_key == 'query'`. Only called when enc_hops != -1."""
        n_enc = len(enc_blocks)
        ref_k = n_enc if backs_key == 'query' else backs_key
        allowed_backs = (enc_active_backs.get(backs_key, set(range(1, enc_hops + 1)))
                         if enc_active_backs else set(range(1, enc_hops + 1)))
        cols = np.zeros(L, dtype=bool)
        for back in range(1, enc_hops + 1):
            src_k = ref_k - back
            if src_k < 0:
                break
            if back not in allowed_backs:
                continue
            bsrc = enc_blocks[src_k]
            cols |= (c >= bsrc['sl0']) & (c < bsrc['sl1'])
        return cols

    if enc_hops != -1:
        n_enc = len(enc_blocks)
        for k in range(1, n_enc):  # k=0 is the entry point, no predecessor, exempt
            b_k = enc_blocks[k]
            sl_row = (r >= b_k['sl0']) & (r < b_k['sl1'])
            own_state = (c >= b_k['sl0']) & (c < b_k['sl1'])
            allowed_cols = _enc_window_cols(k) | own_state
            blocked |= sl_row[:, None] & (is_any_enc_state & ~allowed_cols)[None, :]

    last_rb_of_op: dict[int, int] = {}
    for i_rb, rb in enumerate(rec_blocks):
        last_rb_of_op[rb['op_idx']] = i_rb

    def _relay_source(prev_rb: dict) -> tuple[int, int]:
        assert prev_rb['sl0'] is not None, (
            f"op_idx={prev_rb['op_idx']} has no post-response commit STATE (terminal — no "
            f"trailing 'S' claimed it) but a later op is trying to relay from it. Fix the "
            f"operations list: add an 'S' right after this op if anything downstream needs "
            f"to relay from it.")
        return prev_rb['sl0'], prev_rb['sl1']

    prev_state_for_round: dict[int, tuple[int, int]] = {}
    _last_round_state: dict[int, tuple[int, int]] = {}
    for i_rb, rb in enumerate(rec_blocks):
        if rb['type'] == 'refine':
            assert rb['op_idx'] in _last_round_state, \
                f'refine round at rec_block {i_rb} has no preceding round in the same op — construction bug'
            prev_state_for_round[i_rb] = _last_round_state[rb['op_idx']]
        if rb['type'] in ('initial', 'refine') and rb['sl0'] is not None:
            _last_round_state[rb['op_idx']] = (rb['sl0'], rb['sl1'])

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
        allowed = np.zeros(L, dtype=bool)
        if op_idx == 0 or hops == -1:
            if enc_hops != -1 and op_idx == 0:
                # Only the query (op_idx==0) gets the chunk-sequence window —
                # `enc_hops` windows the CHUNK sequence, which has no defined
                # correspondence to op_idx>0 (a different index space). Any
                # op_idx>0 with hops==-1 keeps the original unrestricted
                # is_any_enc_state; enc_hops does not currently generalize to
                # that case (not needed by any single-Q trajectory, which is
                # all this file's train_jax supports).
                allowed |= _enc_window_cols('query')
            else:
                allowed |= is_any_enc_state
        if op_idx > 0:
            for lo, hi in _relay_ranges(op_idx):
                allowed |= (c >= lo) & (c < hi)
        return allowed

    for k, b in enumerate(enc_blocks):
        sl_row = (r >= b['sl0']) & (r < b['sl1'])
        for j, bj in enumerate(enc_blocks):
            if j == k: continue
            blocked |= sl_row[:, None] & ((c >= bj['s0']) & (c < bj['s1']))[None, :]

    for rb in rec_blocks:
        allowed_state = _allowed_state(rb['op_idx'])

        if rb['type'] == 'noop':
            sl_row = (r >= rb['sl0']) & (r < rb['sl1'])
            blocked |= sl_row[:, None] & ~allowed_state[None, :]
            continue

        if rb['type'] != 'initial':
            continue

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
            sl_row = (r >= rb['sl0']) & (r < rb['sl1'])
            own_end = (allowed_state |
                      (c >= rb['w0']) & (c < rb['w1']) |
                      (c >= rb['c0']) & (c < rb['c1']))
            blocked |= sl_row[:, None] & ~own_end[None, :]

    for i_rb, rb in enumerate(rec_blocks):
        if rb['type'] != 'refine':
            continue
        allowed_state = _allowed_state(rb['op_idx'])
        prev_lo, prev_hi = prev_state_for_round[i_rb]
        own = allowed_state | ((c >= prev_lo) & (c < prev_hi)) | (c >= rb['wa0'])

        wa_row = (r >= rb['wa0']) & (r < rb['wa1'])
        blocked |= wa_row[:, None] & ~own[None, :]

        am_row = (r >= rb['opf0']) & (r < rb['am1'])
        blocked |= am_row[:, None] & ~own[None, :]

        fsl_row = (r >= rb['fsl0']) & (r < rb['fsl1'])
        blocked |= fsl_row[:, None] & ~own[None, :]

        if rb['w0'] < rb['w1']:
            wm_row = (r >= rb['w0']) & (r < rb['w1'])
            blocked |= wm_row[:, None] & ~own[None, :]
        out_row = (r >= rb['c0']) & (r < rb['c1'])
        blocked |= out_row[:, None] & ~own[None, :]

        if rb['sl0'] is not None:
            sl_row = (r >= rb['sl0']) & (r < rb['sl1'])
            blocked |= sl_row[:, None] & ~own[None, :]

    visible = causal & ~blocked
    return np.where(visible, 0.0, -1e9).astype(np.float32)


def _sample_enc_active_backs(rng: np.random.Generator, enc_hops: int, drop_prob: float,
                             n_enc: int) -> dict:
    """Per-training-step stochastic hop dropout ("layer drop in the time
    axis") for `chunk_mask_fb_traj`'s `enc_active_backs` — independently for
    each chunk k=1..n_enc-1 and for the query, back=1 (the immediately
    preceding chunk) is always kept; each back 2..enc_hops is independently
    kept with probability (1-drop_prob), dropped otherwise. `drop_prob=0`
    reproduces the full deterministic window (equivalent to
    `enc_active_backs=None`)."""
    keys = list(range(1, n_enc)) + ['query']
    result = {}
    for key in keys:
        backs = {1}
        for back in range(2, enc_hops + 1):
            if rng.random() >= drop_prob:
                backs.add(back)
        result[key] = backs
    return result


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


def make_batch_tagged(rng: np.random.Generator, B: int, n_chunks: int, chunk_len: int,
                      state_len: int, state_vocab_size: int, pos_content: dict,
                      tags: list[tuple[int, int]],
                      data_kind: str = 'random', data_target_bits: float | None = None) -> np.ndarray:
    """Copied from kvmem.hmn.make_batch_tagged. `data_kind='random'` (default)
    is the original path this file always had; the structured-data branch
    (`kvmem/structured_data.py`'s `generate_structured_chunks` — pure NumPy,
    no torch, safe to import directly per this file's own "avoid a torch
    import chain" constraint, not a blanket no-cross-file-import rule) is
    ADDED here (2026-07-31) for the structured-vs-random convergence-speed
    ablation — each batch item gets a FRESH generator call (required, not
    optional: a fixed rule baked across all examples would let the model
    memorize it into static weights instead of encoding anything into
    STATE, the same failure mode this project's architecture already
    avoids elsewhere — see kvmem.hmn's own make_batch_tagged docstring)."""
    sids = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
    wl = pos_content['warmup_len']
    L = pos_content['L']
    tok = np.zeros((B, L), dtype=np.int64)
    if data_kind == 'random':
        segs = rng.integers(0, 256, size=(B, n_chunks, chunk_len), dtype=np.int64)
    else:
        from kvmem.structured_data import generate_structured_chunks
        segs = np.stack([generate_structured_chunks(rng, data_kind, n_chunks, chunk_len,
                                                     target_bits=data_target_bits)
                         for _ in range(B)])

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
            idx_w = rw_xs[:, None] + np.arange(wl)
            idx_c = rw_xs[:, None] + wl + np.arange(rb['out_len'])
            tok[:, rb['w0']:rb['w1']] = np.take_along_axis(gt, idx_w, axis=1)
            tok[:, rb['c0']:rb['c1']] = np.take_along_axis(gt, idx_c, axis=1)

            if rb['sl0'] is not None:
                tok[:, rb['sl0']] = HMN_OP_UPDATE
                tok[:, rb['sl0'] + 1:rb['sl1']] = sids

        else:  # 'refine' — see kvmem.hmn.make_batch_tagged's own docstring for the
               # placeholder-argmax caveat; not exercised by any DSL this file targets.
            x_min, x_max = rb['warmup_train_range']
            rw_xs = np.array([int(rng.integers(x_min, x_max + 1)) for _ in range(B)])
            tok[:, rb['opf0']] = HMN_OP_FEEDBACK
            idx_w = rw_xs[:, None] + np.arange(wl)
            idx_c = rw_xs[:, None] + wl + np.arange(rb['out_len'])
            gathered_w = np.take_along_axis(gt, idx_w, axis=1)
            gathered_c = np.take_along_axis(gt, idx_c, axis=1)
            tok[:, rb['wa0']:rb['wa1']] = gathered_w
            tok[:, rb['am0']:rb['am1']] = gathered_c
            tok[:, rb['w0']:rb['w1']]   = gathered_w
            tok[:, rb['c0']:rb['c1']]   = gathered_c
            tok[:, rb['fsl0']:rb['fsl1']] = sids
            if rb['sl0'] is not None:
                tok[:, rb['sl0']] = HMN_OP_UPDATE
                tok[:, rb['sl0'] + 1:rb['sl1']] = sids

    if tags:
        tag_pos = np.array([p for p, _ in tags], dtype=np.int64)
        tag_ids = np.array([i for _, i in tags], dtype=np.int64)
        tok[:, tag_pos] = tag_ids[None, :]

    return tok


def load_config(path: str) -> dict:
    """Copied verbatim from kvmem.hmn.load_config."""
    spec   = importlib.util.spec_from_file_location('_kvmem_jax_cfg', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'hp'):
        raise ValueError(f'{path!r} must define a module-level `hp` dict')
    return dict(module.hp)


# =============================================================================
# RoPE
# =============================================================================

def rope_freqs(d_head: int, base: float = 10000.0) -> jnp.ndarray:
    i = jnp.arange(0, d_head, 2, dtype=jnp.float32)
    return 1.0 / (base ** (i / d_head))


def yarn_freqs(d_head: int, L_train: int, L_max: int,
               base: float = 10000.0,
               beta_fast: int = 32, beta_slow: int = 1) -> jnp.ndarray:
    """YaRN NTK-aware scaled RoPE (arXiv:2309.00071) — matches kvmem.hmn.yarn_freqs."""
    s = L_max / L_train
    i = jnp.arange(0, d_head, 2, dtype=jnp.float32)
    inv_f = 1.0 / (base ** (i / d_head))
    wl = 2 * math.pi / inv_f
    lo, hi = 2 * math.pi * beta_slow, 2 * math.pi * beta_fast
    ramp = jnp.clip((wl - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (1 - ramp) * inv_f + ramp * (inv_f / s)


def apply_rope(x: jnp.ndarray, freqs: jnp.ndarray, offset: int = 0) -> jnp.ndarray:
    """x: (..., H, L, d_head)  freqs: (d_head//2,) — matches kvmem.hmn.apply_rope's
    interleaved (even/odd channel pair) rotation convention exactly."""
    L = x.shape[-2]
    # `jnp.arange(L) + offset` instead of `jnp.arange(offset, offset+L)` —
    # mathematically identical, but lets `offset` be a TRACED (non-static)
    # scalar under jit (only `L`, always static from `x`'s shape, needs to be
    # concrete for `arange`) — required by the fixed-KV-buffer decode path
    # (`ar_decode_traj_kv_jit_*`), where `offset` changes every generated
    # token and marking it static would force a recompile per token, defeating
    # the entire point of a fixed-shape buffer. Existing static-offset callers
    # are unaffected (same values either way).
    pos = jnp.arange(L, dtype=jnp.float32) + offset
    angles = pos[:, None] * freqs[None, :]
    cos_a, sin_a = jnp.cos(angles), jnp.sin(angles)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot1 = x1 * cos_a - x2 * sin_a
    rot2 = x1 * sin_a + x2 * cos_a
    out = jnp.stack([rot1, rot2], axis=-1)
    return out.reshape(x.shape)


# =============================================================================
# RMSNorm — matches kvmem.hmn.RMSNorm (no mean-centering, no bias)
# =============================================================================

class RMSNorm(nnx.Module):
    def __init__(self, d: int, *, eps: float = 1e-6, rngs: nnx.Rngs):
        self.weight = nnx.Param(jnp.ones((d,)))
        self.eps = eps

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        norm = jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x * norm * self.weight[...]


# =============================================================================
# Attention — MHAttention restricted to rope+null_kv (qk_norm/logit_cap/
# attn_temp/chunk_attn/KV-cache all omitted, see module docstring)
# =============================================================================

class MHAttention(nnx.Module):
    def __init__(self, d: int, n_heads: int, *,
                 rope: bool = False, freqs: jnp.ndarray | None = None,
                 null_kv: bool = False, rngs: nnx.Rngs):
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.rope = rope
        self.freqs = freqs
        self.null_kv = null_kv
        init = nnx.initializers.normal(stddev=math.sqrt(2.0 / d))
        self.W_Q = nnx.Linear(d, d, use_bias=False, kernel_init=init, rngs=rngs)
        self.W_K = nnx.Linear(d, d, use_bias=False, kernel_init=init, rngs=rngs)
        self.W_V = nnx.Linear(d, d, use_bias=False, kernel_init=init, rngs=rngs)
        self.W_O = nnx.Linear(d, d, use_bias=False, kernel_init=init, rngs=rngs)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, *,
                 past_kv: tuple[jnp.ndarray, jnp.ndarray] | None = None,
                 return_kv: bool = False,
                 offset: int = 0,
                 write_pos: int | None = None):
        """x: (B, L, d)  mask: (Lq, Lkv) additive bias (0.0=attend, -1e9=blocked) —
        same convention as kvmem.hmn (see its "Mask convention" key principle).

        KV-cache support (mirrors kvmem.hmn.MHAttention.forward): `past_kv` is
        an optional (K_past, V_past) of shape (B,H,L_past,d_head) to prepend
        before this call's own (unrotated-by-past-position) K/V; `offset` is
        the RoPE position base for THIS call's L new tokens (= L_past for the
        usual "encode prefix once, then decode one new token at a time"
        pattern); `return_kv=True` returns `(out, (K_cur, V_cur))` where
        K_cur/V_cur are THIS call's own (rotated) K/V only, not the
        concatenated-with-past tensor — caller accumulates the cache across
        calls, exactly as kvmem.hmn's own KV-cache callers do.

        `write_pos` (NEW, opt-in, default None => byte-identical to the above):
        FIXED-SIZE KV BUFFER mode, for jit-compiled autoregressive decode —
        see `ar_decode_traj_kv_jit_minimal`/`_scan` in this file for why this
        exists (growing `past_kv` via concat means a different shape, hence a
        full XLA recompile, on EVERY decode step; a fixed-size buffer written
        via `dynamic_update_slice` keeps shapes constant across all steps, so
        the compiled program is reused as-is). When given, `past_kv` is
        treated as a PRE-ALLOCATED buffer of the FULL decode length (not just
        the real length so far) and the new K/V is written in at `write_pos`
        instead of concatenated — `return_kv=True` then returns the FULL
        updated buffer (not just this call's own slice), since that buffer
        IS the new "past_kv" the caller threads into the next step."""
        B, L, d = x.shape
        H, dh = self.n_heads, self.d_head
        Q = self.W_Q(x).reshape(B, L, H, dh).transpose(0, 2, 1, 3)
        K = self.W_K(x).reshape(B, L, H, dh).transpose(0, 2, 1, 3)
        V = self.W_V(x).reshape(B, L, H, dh).transpose(0, 2, 1, 3)
        if self.rope and self.freqs is not None:
            Q = apply_rope(Q, self.freqs, offset=offset)
            K = apply_rope(K, self.freqs, offset=offset)
        K_cur, V_cur = K, V
        if past_kv is not None:
            K_past, V_past = past_kv
            if write_pos is not None:
                K = jax.lax.dynamic_update_slice_in_dim(K_past, K, write_pos, axis=2)
                V = jax.lax.dynamic_update_slice_in_dim(V_past, V, write_pos, axis=2)
            else:
                K = jnp.concatenate([K_past, K], axis=2)
                V = jnp.concatenate([V_past, V], axis=2)
        # Captured BEFORE null_kv appends its extra column below — the write_pos
        # return path must hand back the buffer at its own fixed width (real bug
        # caught directly: returning post-null-concat K/V here made the "fixed"
        # buffer grow by one column every call, defeating the whole point and
        # breaking the static-shape guarantee jit needs).
        K_buf_out, V_buf_out = K, V
        if self.null_kv:
            # Matches kvmem.hmn.MHAttention.forward EXACTLY: despite that class's own
            # docstring calling this "learnable," the actual code (torch.zeros(...)
            # constructed fresh every forward call, never wrapped in nn.Parameter)
            # never receives gradients — permanently zero, not trainable. Porting the
            # real behavior, not the (now-corrected, see kvmem/hmn.py) stale docstring.
            null = jnp.zeros((B, H, 1, dh), dtype=K.dtype)
            K = jnp.concatenate([K, null], axis=2)
            V = jnp.concatenate([V, null], axis=2)
            mask = jnp.pad(mask, ((0, 0), (0, 1)), constant_values=0.0)
        scale = 1.0 / math.sqrt(dh)
        scores = jnp.einsum('bhqd,bhkd->bhqk', Q, K) * scale
        scores = scores + mask[None, None, :, :]
        probs = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum('bhqk,bhkd->bhqd', probs, V)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, d)
        out = self.W_O(out)
        if return_kv:
            if write_pos is not None:
                return out, (K_buf_out, V_buf_out)  # full updated buffer, fixed width
            return out, (K_cur, V_cur)
        return out


class SingleAttnBlock(nnx.Module):
    """block_type='single_attn': x = x + attn(norm(x)) — matches kvmem.hmn's
    SingleAttnBlock exactly (the project's default architecture)."""
    def __init__(self, d: int, n_heads: int, *,
                 rope: bool = False, freqs: jnp.ndarray | None = None,
                 null_kv: bool = False, rmsnorm: bool = False, rngs: nnx.Rngs):
        assert rmsnorm, 'LayerNorm variant not ported — every config this file targets uses rmsnorm=True'
        self.norm = RMSNorm(d, rngs=rngs)
        self.attn = MHAttention(d, n_heads, rope=rope, freqs=freqs, null_kv=null_kv, rngs=rngs)

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, *,
                 past_kv: tuple[jnp.ndarray, jnp.ndarray] | None = None,
                 return_kv: bool = False,
                 offset: int = 0,
                 write_pos: int | None = None):
        attn_out = self.attn(self.norm(x), mask, past_kv=past_kv, return_kv=return_kv, offset=offset,
                             write_pos=write_pos)
        if return_kv:
            attn_out, kv = attn_out
            return x + attn_out, kv
        return x + attn_out


def _block_call(block, x, mask, past_kv, return_kv, offset, write_pos=None):
    return block(x, mask, past_kv=past_kv, return_kv=return_kv, offset=offset, write_pos=write_pos)


# `nnx.remat` = JAX's gradient-checkpoint transform (recomputes this block's
# activations during backward instead of storing them) — the JAX counterpart
# to kvmem.hmn's `grad_checkpoint='block'` (`_ckpt`/`torch.utils.checkpoint`
# there). Built once at import time, not per forward call, matching how
# `nnx.jit`-wrapped step functions are cached per-trajectory in `train_jax`
# rather than rebuilt every step. `static_argnums=(4, 5)` marks `return_kv`/
# `offset` static — caught directly: without this, tracing `_block_call`
# fails with `ConcretizationTypeError` on `jnp.arange(offset, ...)` inside
# `apply_rope` (needs a concrete Python int, not a traced value) and would
# fail identically on `if return_kv:` if that branch were reached first.
_block_call_remat = nnx.remat(_block_call, static_argnums=(4, 5))


def _group_call(blocks: list, x: jnp.ndarray, mask: jnp.ndarray, offset: int) -> jnp.ndarray:
    """Runs a GROUP of consecutive blocks in sequence, no KV-cache/return_kv
    — used only from `HMNModel.__call__`'s `use_ckpt` path, which is already
    gated to `past_kv is None and not return_kv` (checkpointing is only ever
    active during a fresh training forward, never during eval/decode), so
    this helper doesn't need to carry either through the group."""
    for block in blocks:
        x = block(x, mask, offset=offset)
    return x


# static_argnums=(3,): `offset` is a Python int (always 0 for the depth-axis
# checkpointing path — every group shares the SAME sequence positions,
# unlike the time-axis segmented forward's per-group offset), same
# static-vs-traced requirement as `_block_call_remat`. `blocks` (arg 0) is a
# plain Python list of `SingleAttnBlock` nnx.Module objects — a valid nnx
# graph/pytree, same category as `_block_call_remat`'s own single-`block`
# arg 0, just a list instead of one module.
_group_call_remat = nnx.remat(_group_call, static_argnums=(3,))


def _grad_checkpoint_groups(grad_checkpoint, n_layers: int) -> list[tuple[int, int]] | None:
    """Parses `grad_checkpoint` into a list of (start,end) index ranges over
    the `n_layers` blocks — depth-axis counterpart to `forward_granularity`'s
    own int/float duality (see `_make_train_step_segmented`'s docstring for
    the time-axis version this mirrors). `False`/`None`/`0` -> no
    checkpointing (returns `None`). `True`/`'block'` -> one group per layer
    (group_size=1 — the project's original per-layer remat, kept as the
    default so every existing config's behavior is UNCHANGED: verified
    bit-exact against the pre-granularity per-block loop, see CLAUDE.md).
    An int >=1 -> exact group size (layers per checkpoint unit). A float in
    (0,1] -> fraction of `n_layers` per group, so it scales with model depth
    instead of needing per-config retuning — `1.0` groups the WHOLE stack
    into one checkpoint unit (still saves memory relative to no
    checkpointing at all, since only that one group's OWN input is
    retained rather than every intermediate layer's activations — just with
    the fewest, largest remat call-sites, i.e. least per-call overhead but
    the largest single recompute+peak-memory footprint; smaller fractions
    trade the other way, more call-sites, smaller peaks, approaching the
    per-layer extreme as they shrink toward `1/n_layers`)."""
    if not grad_checkpoint:
        return None
    if grad_checkpoint is True or grad_checkpoint == 'block':
        group_size = 1
    elif isinstance(grad_checkpoint, float):
        assert 0 < grad_checkpoint <= 1.0, 'fractional grad_checkpoint granularity must be in (0, 1]'
        group_size = max(1, round(grad_checkpoint * n_layers))
    else:
        assert isinstance(grad_checkpoint, int) and grad_checkpoint >= 1, \
            f'grad_checkpoint must be False/True/"block"/an int>=1/a float in (0,1], got {grad_checkpoint!r}'
        group_size = int(grad_checkpoint)
    return [(i, min(i + group_size, n_layers)) for i in range(0, n_layers, group_size)]


# =============================================================================
# HMNModel — restricted to block_type='single_attn' (see module docstring)
# =============================================================================

class HMNModel(nnx.Module):
    def __init__(self, V: int, d: int, n_layers: int, n_heads: int, *,
                 rope: bool = False, yarn: bool = False,
                 L_train: int = 512, L_max: int = 4096,
                 null_kv: bool = False, rmsnorm: bool = False,
                 grad_checkpoint: bool = False,
                 V_out: int = 256, rngs: nnx.Rngs):
        self.n_special = V - 256
        data_init = nnx.initializers.normal(stddev=0.02)
        special_init = nnx.initializers.normal(stddev=0.05)
        out_init = nnx.initializers.normal(stddev=0.02)
        self.data_embed = nnx.Embed(256, d, embedding_init=data_init, rngs=rngs)
        self.special_embed = nnx.Embed(self.n_special, d, embedding_init=special_init, rngs=rngs)
        self.norm_out = RMSNorm(d, rngs=rngs)
        self.W_out = nnx.Linear(d, V_out, use_bias=False, kernel_init=out_init, rngs=rngs)
        self.V_out = V_out

        freqs = None
        if rope:
            d_head = d // n_heads
            freqs = yarn_freqs(d_head, L_train=L_train, L_max=L_max) if yarn else rope_freqs(d_head)

        blocks = [
            SingleAttnBlock(d, n_heads, rope=rope, freqs=freqs, null_kv=null_kv, rmsnorm=rmsnorm, rngs=rngs)
            for _ in range(n_layers)
        ]
        # nnx.List (strict data/static pytree typing for a plain list attribute) was
        # added in a flax version newer than what's installable on some TPU VMs stuck
        # on Python 3.10 (flax>=0.11 requires Python 3.11+; verified directly on tpu2,
        # newest available there is 0.10.7). A bare list of submodules works fine on
        # those older nnx versions (no strict check yet) — degrade gracefully rather
        # than hard-requiring the newer flax API.
        self.blocks = nnx.List(blocks) if hasattr(nnx, 'List') else blocks
        self.grad_checkpoint = grad_checkpoint
        # Static (plain Python, not traced) — computed once here rather than
        # per forward call. None when grad_checkpoint is falsy.
        self._ckpt_groups = _grad_checkpoint_groups(grad_checkpoint, n_layers)

    def _embed(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """Route tokens to data_embed (0-255) or special_embed (256+) — matches
        kvmem.hmn.HMNModel._embed's masked-blend routing exactly."""
        is_sp = tokens >= 256
        data_ids = jnp.clip(tokens, 0, 255)
        special_ids = jnp.clip(tokens - 256, 0, self.n_special - 1)
        d_emb = self.data_embed(data_ids)
        s_emb = self.special_embed(special_ids)
        blend = is_sp[..., None].astype(d_emb.dtype)
        return s_emb * blend + d_emb * (1.0 - blend)

    def __call__(self, tokens: jnp.ndarray, mask: jnp.ndarray, *,
                 past_kv: list[tuple[jnp.ndarray, jnp.ndarray]] | None = None,
                 return_kv: bool = False,
                 offset: int = 0,
                 write_pos: int | None = None):
        """tokens: (B, L) int32/int64  mask: (Lq, Lkv) additive bias -> logits (B, L, V_out).

        Mirrors kvmem.hmn.HMNModel.forward's KV-cache signature exactly:
        `past_kv`: list[n_layers] of (K_past, V_past), one per block, or None.
        `return_kv=True`: returns `(logits, kv_out)` instead of just logits,
        `kv_out` a list[n_layers] of this call's own (K,V) — same shape/
        semantics the PyTorch decode callers (`ar_decode_iq_global_rw_tagged`
        etc.) already rely on, so a caller written against the PyTorch model
        works against this one with no changes beyond the import.

        `write_pos`: see `MHAttention.__call__`'s own docstring — fixed-size
        KV buffer mode for jit-compiled decode, opt-in, default None leaves
        every existing (non-decode-jit) caller byte-identical."""
        x = self._embed(tokens)
        kv_out = []
        # grad_checkpoint: only meaningful (and only cheap) when there's no KV cache and
        # no KV being returned — mirrors kvmem.hmn.HMNModel.forward's own `use_ckpt =
        # (self.grad_checkpoint and self.training and pkv is None and not return_kv and
        # not return_features)` gate. This port has no explicit train/eval mode flag
        # (nnx has no automatic train()/eval() state tracking the way nn.Module does),
        # but that gate only matters for AVOIDING POINTLESS RECOMPUTE OVERHEAD during
        # eval/decode — remat's recompute only ever triggers on a backward pass, so
        # calling a remat-wrapped block outside of `jax.grad`/`nnx.value_and_grad` is
        # already a no-op overhead-wise; the `past_kv is None and not return_kv` gate
        # alone is sufficient here.
        use_ckpt = self.grad_checkpoint and past_kv is None and not return_kv
        if use_ckpt:
            for start, end in self._ckpt_groups:
                x = _group_call_remat(list(self.blocks[start:end]), x, mask, offset)
        else:
            for i, block in enumerate(self.blocks):
                pkv = past_kv[i] if past_kv is not None else None
                result = _block_call(block, x, mask, pkv, return_kv, offset, write_pos)
                if return_kv:
                    x, kv_i = result
                    kv_out.append(kv_i)
                else:
                    x = result
        h_out = self.norm_out(x)
        logits = self.W_out(h_out)
        if return_kv:
            return logits, kv_out
        return logits

    def count_params(self) -> int:
        return sum(p.size for p in jax.tree.leaves(nnx.state(self, nnx.Param)))


def build_model(hp: dict, rngs: nnx.Rngs) -> HMNModel:
    """Factory mirroring kvmem.hmn.build_model(hp, device) -> HMNModel's own
    signature/defaults, restricted to this file's single_attn-only scope."""
    assert hp.get('block_type', 'single_attn') == 'single_attn', \
        'kvmem/hmn_jax.py only ports block_type=single_attn — see module docstring'
    return HMNModel(
        V=hp.get('V', 271), d=hp['d'], n_layers=hp['n_layers'], n_heads=hp['n_heads'],
        rope=hp.get('rope', False), yarn=hp.get('yarn', False),
        L_train=hp.get('L_train', hp.get('seg_len', 512)),
        L_max=hp.get('L_max', hp.get('seg_len', 512) * 8),
        null_kv=hp.get('null_kv', False), rmsnorm=hp.get('rmsnorm', False),
        # kvmem.hmn's grad_checkpoint is bool|str|None ('block'/'attn' select a
        # PyTorch-specific granularity that doesn't apply here — single_attn only
        # has one thing to checkpoint per layer). Passed through UNCHANGED (not
        # bool()-collapsed — a real bug caught directly: bool(2)/bool(0.25)/
        # bool('block') are all True, which silently forced every numeric/string
        # granularity down to the coarsest per-layer grouping regardless of what
        # was actually requested) so HMNModel's own int/float granularity parsing
        # (`_grad_checkpoint_groups`) sees the real value.
        grad_checkpoint=hp.get('grad_checkpoint', False),
        V_out=hp.get('V_out', 256), rngs=rngs,
    )


# =============================================================================
# Training loop — new code (not a port), scoped to non-refine single-Q
# entries only (every hmn_notags_w25*.py config). See module docstring.
# =============================================================================

def _build_trajectory(hp: dict, entry: dict, stage_chunk_len: int) -> dict:
    """One weave_mix entry -> {pos_content, mask (jnp), L, chunk_len, weight}.

    `hp['enc_hops']` (default -1, unchanged legacy behavior) opts into the
    bounded encoding-chain window (see `chunk_mask_fb_traj`'s own
    docstring). When enabled, `mask` is the full deterministic window (no
    dropout — used for eval/decode and as the training default), and
    `pos_mask` is additionally kept on the returned dict so `train_jax` can
    rebuild a fresh, stochastically hop-dropped mask each step when
    `hp['hop_drop_prob'] > 0`."""
    ops, n_refine, _repeat_batch, dsl_chunk_len, dsl_warmup_len = parse_traj_dsl(entry['dsl'])
    assert n_refine == 0, 'kvmem/hmn_jax.py train_jax does not support refine rounds (R token)'
    chunk_len = dsl_chunk_len if dsl_chunk_len is not None else stage_chunk_len
    warmup_len = dsl_warmup_len if dsl_warmup_len is not None else hp['warmup_len']
    built = chunk_positions_traj(chunk_len, hp['state_len'], warmup_len, ops,
                                 n_refine=0, state_vocab_size=hp['state_vocab_size'])
    pos_content = built['pos_content']
    enc_hops = hp.get('enc_hops', -1)
    mask_np = chunk_mask_fb_traj(built['pos_mask'], hops=-1, enc_hops=enc_hops)
    rec_blocks = pos_content['rec_blocks']
    assert len(rec_blocks) == 1 and rec_blocks[0]['type'] == 'initial', \
        'kvmem/hmn_jax.py train_jax only supports one Q per entry (batch/stream/etc. not ported)'
    rb = rec_blocks[0]
    return dict(pos_content=pos_content, pos_mask=built['pos_mask'], tags=built['tags'],
               mask=jnp.asarray(mask_np), enc_hops=enc_hops,
               L=built['L'], chunk_len=chunk_len, weight=entry['weight'],
               w0=rb['w0'], c1=rb['c1'], dsl=entry['dsl'],
               n_chunks=len(pos_content['enc_blocks']))


# ---------------------------------------------------------------------------
# Length bucketing (TPU/XLA support) — copied verbatim from kvmem.hmn's own
# _bucket_ceilings/_assign_bucket/_pad_mask_to/_pad_tok_to/_pow2_floor (pure
# NumPy/Python, no torch involved, matching this file's copy-don't-import
# convention). See kvmem.hmn's own docstring for the full rationale: XLA
# compiles one graph per distinct input shape, so a weave_mix with many
# distinct L values triggers a recompile storm; bucketing groups the
# distinct L values into <= max_buckets ceilings (drawn from the observed
# lengths themselves, not rounded to a power of 2) and pads each trajectory
# up to its assigned ceiling once, at stage setup. Opt-in via
# hp['bucket_lengths'] (default False) — existing configs unaffected.
#
# Tuning `token_budget`/`attn_sq_budget` against real HBM — worked example
# (v6e/Trillium, 31.25 GiB usable per chip, confirmed via `tpu-info`):
#
#   b_cap = min(B, pow2_floor(token_budget / Lb), pow2_floor(attn_sq_budget / Lb**2))
#
# `token_budget` caps the LINEAR-in-L cost (embeddings, FFN, residual-stream
# activations — everything that's O(B*L)); `attn_sq_budget` caps the
# QUADRATIC-in-L cost (the (B,H,L,L) attention-score matrix — the term that
# actually dominates at long L). Two separate budgets because a single one
# calibrated for short L is wastefully small at long L, and vice versa.
#
# ONE real calibration point (2026-07-31, `hmn_tpu_recall1024_flat_rope_jax.py`
# — d=128/n_layers=16/n_heads=8, ~1.12M params, grad_checkpoint='block',
# no_autocast=True/fp32): B=64, Lb=2128 measured 29.51/31.25 GiB HBM via
# `tpu-info` (steady state, post-compile). That's B*Lb=136,192 (token term)
# and B*Lb**2=289,816,576 (attn term) mapping to ~29.5 GiB TOTAL usage — this
# figure includes params/optimizer state/embeddings too, not purely the
# attention matrix, so it is NOT a clean per-unit conversion factor to reuse
# for a different architecture; it's a single-point anchor for THIS one.
# `hmn_tpu_recall1024_flat_rope_jax.py` sets token_budget=200_000 (>136,192)
# and attn_sq_budget=320_000_000 (>289,816,576) — both intentionally ABOVE
# the calibration point, specifically so no bucket at or below Lb=2128 gets
# capped below the already-verified-safe B=64 (the ~1.7 GiB gap to the 31.25
# GiB ceiling is deliberate headroom, not something these budgets are meant
# to fully consume — eval/checkpoint-save briefly need extra memory too).
# To push B higher than what's already verified (not done in this session —
# would need its own real HBM check before trusting it): raise attn_sq_budget
# toward `B_target * Lb**2`, e.g. B=96 at Lb=2128 needs attn_sq_budget >=
# 96*2128**2 ≈ 434.7M — then RE-VERIFY via `tpu-info`, don't just trust the
# arithmetic, since the ~29.5 GiB figure already includes unknown fixed
# overhead this formula doesn't separate out.
# ---------------------------------------------------------------------------

def _bucket_ceilings(lengths: list[int], weights: list[float], max_buckets: int) -> list[int]:
    """Partitions the sorted distinct `lengths` into <= `max_buckets` contiguous
    groups, minimizing sum over groups of `ceiling^2 * group_weight` (squared,
    since attention cost scales with L^2) — a weighted k-segment DP. Returns
    the sorted list of chosen ceilings (each an actual value from `lengths`)."""
    from collections import defaultdict
    import itertools
    agg: dict[int, float] = defaultdict(float)
    for L, w in zip(lengths, weights):
        agg[L] += w
    Ls = sorted(agg)
    n = len(Ls)
    if n <= max_buckets:
        return Ls
    W = [agg[L] for L in Ls]
    prefix = [0.0] + list(itertools.accumulate(W))

    def _cost(i: int, j: int) -> float:
        return float(Ls[j]) ** 2 * (prefix[j + 1] - prefix[i])

    INF = float('inf')
    dp = [[INF] * (n + 1) for _ in range(max_buckets + 1)]
    choice: list[list[int | None]] = [[None] * (n + 1) for _ in range(max_buckets + 1)]
    dp[0][0] = 0.0
    for k in range(1, max_buckets + 1):
        for i in range(1, n + 1):
            for j in range(i):
                c = dp[k - 1][j] + _cost(j, i - 1)
                if c < dp[k][i]:
                    dp[k][i] = c
                    choice[k][i] = j
    best_k = min(range(1, max_buckets + 1), key=lambda k: dp[k][n])
    ceilings = []
    i, k = n, best_k
    while i > 0:
        j = choice[k][i]
        assert j is not None
        ceilings.append(Ls[i - 1])
        i, k = j, k - 1
    ceilings.reverse()
    return ceilings


def _assign_bucket(L: int, ceilings: list[int]) -> int:
    """Smallest ceiling >= L (`ceilings` must be sorted and cover every L used)."""
    import bisect
    idx = bisect.bisect_left(ceilings, L)
    assert idx < len(ceilings), f'L={L} exceeds every bucket ceiling {ceilings}'
    return ceilings[idx]


def _pad_mask_to(mask_np: np.ndarray, Lb: int) -> np.ndarray:
    """Pads an [L,L] additive attention-bias mask to [Lb,Lb]. Every new column/
    row is fully blocked (-1e9); safe against NaN softmax rows because
    `null_kv=True` is mandatory in this project."""
    L = mask_np.shape[0]
    assert mask_np.shape == (L, L) and Lb >= L
    if Lb == L:
        return mask_np
    out = np.full((Lb, Lb), -1e9, dtype=mask_np.dtype)
    out[:L, :L] = mask_np
    return out


def _pad_tok_to(tok_np: np.ndarray, Lb: int) -> np.ndarray:
    """Pads a [B,L] token batch to [B,Lb] with zeros. Safe because the loss
    (see `_make_train_step_bucket`) is masked to the real [w0,c1) region."""
    B, L = tok_np.shape
    assert Lb >= L
    if Lb == L:
        return tok_np
    out = np.zeros((B, Lb), dtype=tok_np.dtype)
    out[:, :L] = tok_np
    return out


def _pow2_floor(x: float) -> int:
    x = max(1, int(x))
    return 1 << (x.bit_length() - 1)


def _make_schedule(hp: dict, total_steps: int):
    lr_max = hp.get('lr_max', 1e-4)
    warmup_steps = min(max(1, hp.get('warmup_steps', 500)), total_steps)
    warmup = optax.linear_schedule(0.0, lr_max, warmup_steps)
    const = optax.constant_schedule(lr_max)
    return optax.join_schedules([warmup, const], boundaries=[warmup_steps])


# =============================================================================
# Eval — ported from kvmem.hmn, restricted to the single non-refine 'initial'
# rec_block case train_jax/_build_trajectory itself targets (no noop/refine
# handling — batch/stream/interleave_delayed/repeat_query patterns and
# refine rounds are out of scope, same as the rest of this file).
# =============================================================================

def make_test_sequences(seg_len: int) -> dict[str, list[int]]:
    """Ported from kvmem.hmn.make_test_sequences (2026-07-31 — this file's
    own copy previously had only `up_counter`/`down_counter`/`const_mid`,
    a divergence from the torch original, not an intentional trim; also
    fixed `down_counter`'s formula to match torch's exact operator
    precedence: `V - 1 - i % V` means `V - 1 - (i % V)`, not `(V-1-i) % V`
    — this file's prior version had the latter, a real discrepancy from
    the torch original, now corrected to match exactly).

    `const_mid` (a single repeated byte) intentionally NOT ported/kept
    removed — trivially continuable with zero genuine content-addressed
    recall ("keep emitting the same byte"), which was inflating val MEAN
    with a free ~33% (1-of-3 patterns solved for free) that looked like
    progress but wasn't — see CLAUDE.md's curriculum-staged entry:
    qualitative eyeball check showed const_mid=100% while up_counter/
    down_counter sat near 0%, and the blended MEAN masked that entirely.
    Every OTHER torch pattern requires genuine position-tracking/content-
    addressing to continue correctly, so they don't have that problem.

    Deterministic held-out test sequences of length seg_len, all bytes in
    [DATA_LO=0x20, 0xFF], never protocol bytes."""
    DATA_LO = 0x20
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


def ar_decode_traj_nokv(model, chunks_arr, state_len: int, state_vocab_size: int,
                        mask_np: np.ndarray, pos_content: dict,
                        tags: list[tuple[int, int]]) -> dict:
    """JAX port of kvmem.hmn.ar_decode_traj_nokv (full-recompute, no KV
    cache — matches that function's own eval usage inside `train()`, so this
    is what `train_jax`'s own periodic eval uses too, for direct parity).
    Restricted to a single non-refine 'initial' rec_block (see module
    docstring) — the noop/refine machinery in the PyTorch original is not
    reachable via `_build_trajectory`'s own assertion, so it's not ported.

    Deliberately NOT jitted: `out_len` token-at-a-time, each call's sequence
    length GROWS by one every iteration, so jit would need to either retrace
    every single token (worse than no jit) or pad to a fixed max length. The
    PyTorch `_nokv` sibling has this exact same tradeoff (full recompute,
    no cache) — see its own docstring."""
    if isinstance(chunks_arr, np.ndarray) and chunks_arr.ndim == 2:
        chunks_list = [chunks_arr[k] for k in range(chunks_arr.shape[0])]
    else:
        chunks_list = list(chunks_arr)

    wl = pos_content['warmup_len']
    sids = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
    L = pos_content['L']

    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos_content['enc_blocks']):
        tok[b['s0']:b['s1']] = chunks_list[k]
        tok[b['sl0']] = HMN_OP_UPDATE
        tok[b['sl0'] + 1:b['sl1']] = sids

    if tags:
        tag_pos = np.array([p for p, _ in tags], dtype=np.int64)
        tag_ids = np.array([i for _, i in tags], dtype=np.int64)
        tok[tag_pos] = tag_ids

    rec_blocks = pos_content['rec_blocks']
    assert len(rec_blocks) == 1 and rec_blocks[0]['type'] == 'initial', \
        'kvmem/hmn_jax.py ar_decode_traj_nokv only supports one non-refine Q per entry'
    rb = rec_blocks[0]
    span_s, span_e = rb['span']
    gt_span = np.concatenate(chunks_list[span_s:span_e])
    ws = rb.get('warmup_start', 0)
    warmup_src = gt_span[ws:ws + wl]
    if wl > 0:
        tok[rb['w0']:rb['w1']] = warmup_src

    for j in range(rb['out_len']):
        pos = rb['c0'] + j
        t = jnp.asarray(tok[:pos], dtype=jnp.int32)[None, :]
        m = jnp.asarray(mask_np[:pos, :pos])
        logits = model(t, m)
        tok[pos] = int(jnp.argmax(logits[0, -1]))

    out_len = rb['out_len']
    rb_target = gt_span[ws + wl:ws + wl + out_len]
    rb_gen = tok[rb['c0']:rb['c1']]
    match_pct = 100.0 * float(np.sum(rb_gen[:len(rb_target)] == rb_target)) / max(len(rb_target), 1)
    return dict(match_pct=match_pct)


def ar_decode_traj_kv(model, chunks_arr, state_len: int, state_vocab_size: int,
                      mask_np: np.ndarray, pos_content: dict,
                      tags: list[tuple[int, int]]) -> dict:
    """KV-cached counterpart to `ar_decode_traj_nokv` — same restricted
    single-'initial'-rec_block scope, same match% result (mathematically
    identical decode, since both are greedy argmax over the same causal
    mask), but encodes the fixed prefix ONCE via `return_kv=True` and then
    grows the cache one token at a time via `past_kv`/`offset`, instead of
    recomputing the whole growing prefix from scratch every generated byte.
    Not part of `kvmem.hmn`'s own `chunk_positions_traj`-layout eval (that
    codebase's KV-cached decoders — `ar_decode_iq_global_rw_tagged`,
    `ar_decode_stitch` — target OTHER position layouts); this is new, JAX-
    only, built directly against this file's own `HMNModel.__call__`
    KV-cache signature (mirrors the PyTorch one exactly, see that method's
    own docstring)."""
    if isinstance(chunks_arr, np.ndarray) and chunks_arr.ndim == 2:
        chunks_list = [chunks_arr[k] for k in range(chunks_arr.shape[0])]
    else:
        chunks_list = list(chunks_arr)

    wl = pos_content['warmup_len']
    sids = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
    L = pos_content['L']

    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos_content['enc_blocks']):
        tok[b['s0']:b['s1']] = chunks_list[k]
        tok[b['sl0']] = HMN_OP_UPDATE
        tok[b['sl0'] + 1:b['sl1']] = sids

    if tags:
        tag_pos = np.array([p for p, _ in tags], dtype=np.int64)
        tag_ids = np.array([i for _, i in tags], dtype=np.int64)
        tok[tag_pos] = tag_ids

    rec_blocks = pos_content['rec_blocks']
    assert len(rec_blocks) == 1 and rec_blocks[0]['type'] == 'initial', \
        'kvmem/hmn_jax.py ar_decode_traj_kv only supports one non-refine Q per entry'
    rb = rec_blocks[0]
    span_s, span_e = rb['span']
    gt_span = np.concatenate(chunks_list[span_s:span_e])
    ws = rb.get('warmup_start', 0)
    warmup_src = gt_span[ws:ws + wl]
    if wl > 0:
        tok[rb['w0']:rb['w1']] = warmup_src

    # Encode the fixed prefix (everything up to and including the warmup, i.e.
    # up to rb['c0']) in ONE dense forward pass, capturing the KV cache.
    prefix_end = rb['c0']
    prefix_tok = jnp.asarray(tok[:prefix_end], dtype=jnp.int32)[None, :]
    prefix_mask = jnp.asarray(mask_np[:prefix_end, :prefix_end])
    _, kv_cache = model(prefix_tok, prefix_mask, return_kv=True)
    L_cached = prefix_end

    for j in range(rb['out_len']):
        pos = prefix_end + j
        # Single new token, attending to everything cached so far (+1 for
        # itself once appended) — matches kvmem.hmn's own KV-cache decode
        # mask slicing convention (`full_mask[pos:pos+1, :pos+1]`).
        t = jnp.asarray(tok[pos:pos + 1], dtype=jnp.int32)[None, :]
        m = jnp.asarray(mask_np[pos:pos + 1, :pos + 1])
        logits, kv_new = model(t, m, past_kv=kv_cache, return_kv=True, offset=L_cached)
        tok[pos] = int(jnp.argmax(logits[0, -1]))
        kv_cache = [(jnp.concatenate([k_old, k_new], axis=2), jnp.concatenate([v_old, v_new], axis=2))
                   for (k_old, v_old), (k_new, v_new) in zip(kv_cache, kv_new)]
        L_cached += 1

    out_len = rb['out_len']
    rb_target = gt_span[ws + wl:ws + wl + out_len]
    rb_gen = tok[rb['c0']:rb['c1']]
    match_pct = 100.0 * float(np.sum(rb_gen[:len(rb_target)] == rb_target)) / max(len(rb_target), 1)
    return dict(match_pct=match_pct)


# `ar_decode_traj_kv_jit`'s decode-loop cache: keyed by `id(model)` and the
# trajectory's own shape (prefix_end, out_len) — a `fori_loop`-jitted decode
# body is compiled once per shape and reused across every later call with
# that same shape, exactly like `train_jax`'s own bucket/trajectory step_fn
# caching. Module-level (not a train_jax-local closure) so eval callers
# outside train_jax can benefit too.
_decode_jit_cache: dict[tuple[int, int, int], object] = {}


def _make_decode_jit_body(prefix_end: int, out_len: int):
    """Builds the `fori_loop`-jitted decode body for one fixed (prefix_end,
    out_len) shape — see CLAUDE.md's decode-jit benchmark entry for the full
    rationale/measurements (2026-07-31): a fixed-size KV buffer written via
    `dynamic_update_slice` (not `concatenate`) keeps every step's shapes
    identical, so ONE compile serves the WHOLE `out_len`-token decode instead
    of either (a) eager per-op dispatch (baseline) or (b) a fresh XLA
    recompile every single token (naively jitting the old growing-`past_kv`
    call) — measured ~118x (forward-pass-only jit) and ~358x (this
    `fori_loop` version, entire loop as one compiled program) faster than
    eager on a small local benchmark, byte-identical match% in both cases.
    `L_final = prefix_end + out_len` is the buffer width; `step_masks` (the
    per-step causal row, built by the caller from the trajectory's own
    `mask_np` — identical values `ar_decode_traj_kv`'s own per-step slicing
    would use, just precomputed as one dense array) is passed as a regular
    (non-static) argument so it varies by CONTENT (call site) without
    forcing a recompile, since its SHAPE is fixed by `out_len`."""
    L_final = prefix_end + out_len

    @nnx.jit
    def decode_jit(model, tok_buf0, kv_buf0, step_masks):
        def body(j, carry):
            tok_buf, kv_buf = carry
            pos = prefix_end + j
            cur = jax.lax.dynamic_slice_in_dim(tok_buf, pos, 1, axis=0)[None, :]
            m = jax.lax.dynamic_slice_in_dim(step_masks, j, 1, axis=0)
            logits, kv_new = model(cur, m, past_kv=kv_buf, return_kv=True, offset=pos, write_pos=pos)
            next_tok = jnp.argmax(logits[0, -1]).astype(tok_buf.dtype)
            tok_buf = jax.lax.dynamic_update_slice_in_dim(tok_buf, next_tok[None], pos, axis=0)
            return (tok_buf, kv_new)
        tok_buf, _ = jax.lax.fori_loop(0, out_len, body, (tok_buf0, kv_buf0))
        return tok_buf
    return decode_jit


def ar_decode_traj_kv_jit(model, chunks_arr, state_len: int, state_vocab_size: int,
                          mask_np: np.ndarray, pos_content: dict,
                          tags: list[tuple[int, int]]) -> dict:
    """Jit-compiled counterpart to `ar_decode_traj_kv` — same restricted
    single-'initial'-rec_block scope, same signature/return, mathematically
    identical greedy-argmax result (verified locally, byte-identical match%
    across every tested case) — but the WHOLE `out_len`-token decode is one
    compiled `fori_loop` program instead of `out_len` separate eager Python
    calls. Compiles once per distinct (prefix_end, out_len) shape (cached in
    `_decode_jit_cache`, module-level) and reuses that compiled program for
    every later call with the same shape, including across different
    trajectories that happen to share a shape and across repeated eval
    passes within one `train_jax` run."""
    if isinstance(chunks_arr, np.ndarray) and chunks_arr.ndim == 2:
        chunks_list = [chunks_arr[k] for k in range(chunks_arr.shape[0])]
    else:
        chunks_list = list(chunks_arr)

    wl = pos_content['warmup_len']
    sids = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
    L = pos_content['L']

    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos_content['enc_blocks']):
        tok[b['s0']:b['s1']] = chunks_list[k]
        tok[b['sl0']] = HMN_OP_UPDATE
        tok[b['sl0'] + 1:b['sl1']] = sids

    if tags:
        tag_pos = np.array([p for p, _ in tags], dtype=np.int64)
        tag_ids = np.array([i for _, i in tags], dtype=np.int64)
        tok[tag_pos] = tag_ids

    rec_blocks = pos_content['rec_blocks']
    assert len(rec_blocks) == 1 and rec_blocks[0]['type'] == 'initial', \
        'kvmem/hmn_jax.py ar_decode_traj_kv_jit only supports one non-refine Q per entry'
    rb = rec_blocks[0]
    span_s, span_e = rb['span']
    gt_span = np.concatenate(chunks_list[span_s:span_e])
    ws = rb.get('warmup_start', 0)
    warmup_src = gt_span[ws:ws + wl]
    if wl > 0:
        tok[rb['w0']:rb['w1']] = warmup_src

    prefix_end = rb['c0']
    out_len = rb['out_len']
    L_final = prefix_end + out_len

    prefix_tok = jnp.asarray(tok[:prefix_end], dtype=jnp.int32)[None, :]
    prefix_mask = jnp.asarray(mask_np[:prefix_end, :prefix_end])
    _, kv_prefix = model(prefix_tok, prefix_mask, return_kv=True)
    kv_buf0 = list((jnp.pad(k, ((0, 0), (0, 0), (0, out_len), (0, 0))),
                   jnp.pad(v, ((0, 0), (0, 0), (0, out_len), (0, 0))))
                  for k, v in kv_prefix)
    tok_buf0 = jnp.asarray(tok, dtype=jnp.int32)

    step_masks = jnp.asarray(np.stack([
        np.pad(mask_np[prefix_end + j, :prefix_end + j + 1],
              (0, L_final - (prefix_end + j + 1)), constant_values=-1e9)
        for j in range(out_len)
    ]))

    cache_key = (id(model), prefix_end, out_len)
    decode_jit = _decode_jit_cache.get(cache_key)
    if decode_jit is None:
        decode_jit = _make_decode_jit_body(prefix_end, out_len)
        _decode_jit_cache[cache_key] = decode_jit

    tok_out = np.asarray(decode_jit(model, tok_buf0, kv_buf0, step_masks))

    rb_target = gt_span[ws + wl:ws + wl + out_len]
    rb_gen = tok_out[rb['c0']:rb['c1']]
    match_pct = 100.0 * float(np.sum(rb_gen[:len(rb_target)] == rb_target)) / max(len(rb_target), 1)
    return dict(match_pct=match_pct)


# =============================================================================
# Checkpointing — ported from kvmem.hmn's `train()` (`stage{i}_last/best/
# end.pt` via `torch.save(dict(model=state_dict, hp=hp, step=...))`). Uses
# plain `pickle` + numpy arrays rather than `torch.save`/orbax — no new
# dependency, and the on-disk format is this port's own (a PyTorch
# checkpoint and a JAX one are never interchangeable regardless of format,
# since the two models don't share a state_dict layout).
# =============================================================================

def save_checkpoint(path: str, model, hp: dict, step: int, val_mean: float | None = None):
    import pickle
    state = nnx.state(model, nnx.Param)
    flat_state = jax.tree.map(np.asarray, state)
    payload = dict(model=flat_state, hp=hp, step=step)
    if val_mean is not None:
        payload['val_mean'] = val_mean
    with open(path, 'wb') as f:
        pickle.dump(payload, f)


def load_checkpoint(path: str, model) -> dict:
    """Loads a checkpoint saved by `save_checkpoint` into `model` IN PLACE
    (mirrors `model.load_state_dict(...)` — `nnx.update` mutates the passed
    module's Variables directly) and returns the full payload dict (`hp`,
    `step`[, `val_mean`]) for the caller to inspect."""
    import pickle
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    jax_state = jax.tree.map(jnp.asarray, payload['model'])
    nnx.update(model, jax_state)
    return payload


# ---------------------------------------------------------------------------
# Segmented forward (`forward_granularity`/`segment_checkpoint`) — JAX port
# AND re-derivation, not a straight port. kvmem.hmn's own `_iter_forward_
# segments` is currently unconditionally `NotImplementedError`-guarded (see
# its docstring): the old segment-boundary logic assumed STATE sits BEFORE
# warmup/response, which broke under the current end-of-turn-STATE design
# for ANY rec_block with a trailing STATE commit (`sl0 is not None` — the
# non-terminal/relay case). This file's own scope (`_build_trajectory`'s own
# assertion: exactly one 'initial', non-refine, TERMINAL rec_block) never
# has that case — a terminal query's `sl0` is always `None` (nothing relays
# from it, so no post-response STATE commit is ever built) — which is
# EXACTLY the case kvmem.hmn's bug does NOT cover (verified directly below:
# `rb['sl0'] is None`, `rb['w0']` sits immediately after the last encode
# block's `sl1`, `rb['c1'] == L`). So the re-derivation here is narrower
# than a general fix would be, but is independently correct for what this
# file supports — segmenting is applied ONLY to the encode portion [0, w0);
# the query itself always gets its own dedicated final forward pass (see
# `_make_train_step_segmented`), never split further.
# ---------------------------------------------------------------------------

def _iter_forward_segments_jax(pos_content: dict) -> list[tuple[int, int]]:
    """Returns the ordered, contiguous (seg_start, seg_end) list spanning
    ONLY the encode portion of the packed sequence (one entry per encoding
    block: `<chunk bytes>` through the end of that chunk's own STATE) —
    verified contiguous by construction (`chunk_positions_traj` builds every
    block from one monotonically increasing `offset`) and verified directly
    (by running `_build_trajectory` against `hmn_tpu_recall1024_flat_rope.py`)
    that `enc_blocks[-1]['sl1'] == rec_blocks[0]['w0']` and
    `rec_blocks[0]['c1'] == L` — i.e. the encode portion and the query
    portion partition the WHOLE sequence with no gap either side."""
    enc_blocks = pos_content['enc_blocks']
    rec_blocks = pos_content['rec_blocks']
    assert len(rec_blocks) == 1 and rec_blocks[0]['type'] == 'initial' and rec_blocks[0]['sl0'] is None, (
        '_iter_forward_segments_jax only supports one terminal non-refine Q '
        '(matches _build_trajectory\'s own assertion) — kvmem.hmn\'s broken '
        'general case (a rec_block with its own trailing STATE commit) is '
        'out of scope here, not silently mishandled')
    segs = []
    prev_end = 0
    for b in enc_blocks:
        assert b['s0'] == prev_end, f"segment gap: expected {prev_end}, got {b['s0']}"
        segs.append((b['s0'], b['sl1']))
        prev_end = b['sl1']
    rb = rec_blocks[0]
    assert rb['w0'] == prev_end, f"segment gap before query: expected {prev_end}, got {rb['w0']}"
    return segs


def _make_train_step_segmented(w0: int, end: int, enc_segs: list[tuple[int, int]],
                               granularity: float | int, segment_checkpoint: bool,
                               update_takes_model: bool):
    """Segmented counterpart to `_make_train_step`/`_make_train_step_bucket`
    — walks the encode portion (`enc_segs`, from `_iter_forward_segments_jax`)
    in GROUPS instead of one dense `model(tokens, mask)` call, carrying a KV
    cache between groups (`past_kv`/`return_kv`/`offset`, the same primitives
    `ar_decode_traj_kv` already uses for eval decode — this is that same
    "process a NEW slice given an existing cache" operation, just with
    `new_len` possibly >1 instead of always 1). `granularity`: an int >=1 is
    an exact segment count per group; a float in (0,1] is a FRACTION of
    `len(enc_segs)` per group (scales with sequence length instead of
    needing per-config retuning). The query segment [w0,end) always gets its
    OWN dedicated final forward pass — never merged into an encode group —
    so this file's scope (see `_iter_forward_segments_jax`) never needs the
    general per-rec_block accumulation kvmem.hmn's own `_forward_segmented`
    does.

    Loss reconstruction detail (the one non-obvious part): the loss slice
    needs logits at GLOBAL index w0-1 (predicts token w0, the first byte of
    the [w0,end) target region) THROUGH end-2. Index w0-1 is the LAST local
    position of the FINAL encode group's own output (it's the last token of
    the encode portion) — everything else (w0..end-2) comes from the query
    group's own output (local indices 0..end-w0-2, i.e. all but its own
    last position). These two slices are concatenated before the softmax/
    NLL, reproducing exactly what a single dense `model(tokens[:,:end],
    mask[:end,:end])` call's `logits[:, w0-1:end-1]` would have given —
    verified numerically against the dense path in this session (see
    `verify_segmented_matches_dense` / CLAUDE.md's segmented-forward entry).

    `segment_checkpoint`: wraps EVERY group's own `model(...)` call
    (encode groups AND the final query call, uniformly) in `nnx.remat` —
    the JAX counterpart to kvmem.hmn's `segment_checkpoint` (`torch.utils.
    checkpoint` there). Recomputes each group's forward during backward
    instead of retaining its activations — the actual memory win this
    feature exists for, same tradeoff (~2x forward compute for whichever
    groups need a backward pass) as kvmem.hmn's own docstring describes."""
    if isinstance(granularity, float):
        assert 0 < granularity <= 1.0, 'fractional granularity must be in (0, 1]'
        group_size = max(1, round(granularity * len(enc_segs)))
    else:
        assert granularity >= 1
        group_size = int(granularity)
    groups = [enc_segs[i:i + group_size] for i in range(0, len(enc_segs), group_size)]

    def _group_fwd(model, tok_slice, seg_mask, kv_cache, offset_val):
        return model(tok_slice, seg_mask, past_kv=kv_cache, return_kv=True, offset=offset_val)

    # static_argnums=(4,): `offset_val` is a Python int closed over per call
    # site (from `groups`/`w0`, structural — never traced data), same
    # requirement `_block_call_remat` already has for its own `offset` arg.
    fwd = nnx.remat(_group_fwd, static_argnums=(4,)) if segment_checkpoint else _group_fwd

    @nnx.jit
    def step(model, optimizer, tokens, mask, loss_mask):
        def loss_fn(model):
            kv_cache = None
            last_enc_logit = None
            for group in groups:
                s0, s1 = group[0][0], group[-1][1]
                tok_slice = tokens[:, s0:s1]
                seg_mask = mask[s0:s1, :s1]
                logits_grp, kv_seg = fwd(model, tok_slice, seg_mask, kv_cache, s0)
                if kv_cache is None:
                    kv_cache = kv_seg
                else:
                    kv_cache = tuple((jnp.concatenate([ka, kb], axis=2), jnp.concatenate([va, vb], axis=2))
                                     for (ka, va), (kb, vb) in zip(kv_cache, kv_seg))
                last_enc_logit = logits_grp[:, -1:, :]

            q_tok = tokens[:, w0:end]
            q_mask = mask[w0:end, :end]
            logits_q, _ = fwd(model, q_tok, q_mask, kv_cache, w0)

            lp_all = jnp.concatenate([last_enc_logit, logits_q[:, :-1, :]], axis=1)
            lp = jax.nn.log_softmax(lp_all, axis=-1)
            tgt = tokens[:, w0:end]
            nll = -jnp.take_along_axis(lp, tgt[..., None], axis=-1).squeeze(-1)
            m = loss_mask[w0:end]
            denom = jnp.sum(m) * tokens.shape[0]
            return jnp.sum(nll * m[None, :]) / jnp.maximum(denom, 1.0)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        if update_takes_model:
            optimizer.update(model, grads)
        else:
            optimizer.update(grads)
        return loss
    return step


class _StatusWriter:
    """Truncate-and-rewrite file for tqdm — stays 1-2 lines, tail -f works.
    Copied from kvmem.hmn's own _StatusWriter (kept file-local, matching this
    file's no-cross-import design)."""
    def __init__(self, path: str):
        self._f = open(path, 'w', buffering=1)

    def write(self, s: str):
        self._f.seek(0)
        self._f.truncate()
        self._f.write(s)
        self._f.flush()

    def flush(self): pass

    def close(self): self._f.close()


def train_jax(hp: dict, log_base: str = 'logs'):
    """Training loop with feature parity to kvmem.hmn's own `train()` for the
    scope this file targets (single non-refine Q per weave_mix entry — see
    module docstring for what's still NOT ported: batch/stream/interleave_
    delayed/repeat_query patterns, refine rounds, label smoothing, adaptive
    reweighting). Per stage: weighted-sample a trajectory, build a fresh
    random batch, one NLL-loss gradient step (jit-compiled per trajectory);
    at `eval_every` boundaries, run `ar_decode_traj_kv` (KV-cached — faster
    than the PyTorch original's own `_nokv` eval, mathematically identical
    result) over `hp['val_n_seqs']` deterministic test sequences per entry,
    log per-entry + MEAN + by-chunk_len match%, and save `stage{i}_last.pt`/
    `stage{i}_best.pt`; `stage{i}_end.pt` at stage end. `early_stop_mean`
    (per-stage key, like the PyTorch version) breaks out of the stage early
    once val MEAN reaches it."""
    rng = np.random.default_rng(hp.get('seed', 42))
    rngs = nnx.Rngs(hp.get('seed', 42))
    model = build_model(hp, rngs)
    n_params = model.count_params()
    _pretrained_ckpt = hp.get('pretrained_ckpt')
    if _pretrained_ckpt:
        _pretrained_payload = load_checkpoint(_pretrained_ckpt, model)
        print(f'[warm-start] loaded {_pretrained_ckpt} (step={_pretrained_payload["step"]}, '
              f'val_mean={_pretrained_payload.get("val_mean")}) before stage 0 — architecture '
              f'(d/n_layers/n_heads/V) must match exactly, nnx.update has no shape-mismatch '
              f'tolerance the way kvmem.hmn\'s torch loader does')

    name = hp.get('name', 'hmn_jax')
    log_dir = os.path.join(log_base, name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_file    = open(os.path.join(log_dir, 'train.log'),    'a', buffering=1)
    jsonl_file  = open(os.path.join(log_dir, 'train.jsonl'), 'a', buffering=1)
    status_file = _StatusWriter(os.path.join(log_dir, 'train_status.log'))

    def _log(msg):
        print(msg)
        print(msg, file=log_file)

    def _jlog(d):
        jsonl_file.write(json.dumps(d) + '\n')

    _log(f'JAX/Flax NNX model: {n_params:,} params  backend={jax.default_backend()}')
    val_n_seqs = hp.get('val_n_seqs')

    def _make_train_step(w0: int, c1: int, update_takes_model: bool):
        """One `nnx.jit`-compiled step per trajectory, `w0`/`c1` closed over as
        Python constants (not traced) so the loss slice uses plain Python
        indexing rather than needing `jax.lax.dynamic_slice`. Built once per
        trajectory at setup time and cached on it (`traj['step_fn']`) — every
        later sampled step for that trajectory reuses the same compiled
        executable instead of retracing. `update_takes_model` is likewise a
        Python bool baked in at trace time (see the flax-version comment
        below) — a plain `if` on it inside the jitted function is a trace-time
        branch, not part of the traced computation, so this is safe."""
        @nnx.jit
        def step(model, optimizer, tokens, mask):
            def loss_fn(model):
                logits = model(tokens, mask)
                lp = jax.nn.log_softmax(logits[:, w0 - 1:c1 - 1], axis=-1)
                tgt = tokens[:, w0:c1]
                nll = -jnp.take_along_axis(lp, tgt[..., None], axis=-1).squeeze(-1)
                return jnp.mean(nll)
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            if update_takes_model:
                optimizer.update(model, grads)
            else:
                optimizer.update(grads)
            return loss
        return step

    def _make_train_step_bucket(w0: int, Lb: int, update_takes_model: bool):
        """Bucketed counterpart to `_make_train_step` — ONE jit-compiled step
        function SHARED across every trajectory assigned to bucket ceiling
        `Lb` (as opposed to one per trajectory). `w0`/`Lb` are still closed
        over as Python constants (static shape, one compile per distinct Lb),
        but different trajectories sharing this bucket have different real
        `c1` (out_len varies with anchor) — so unlike `_make_train_step`, the
        loss can't be a fixed-size Python slice ending at a per-trajectory
        `c1`. Instead it's computed over the full static range [w0, Lb) and
        weighted by a per-trajectory `loss_mask` (traced array input, 1.0 for
        w0<=pos<c1 else 0.0) passed in at call time — this is what lets one
        compiled program serve every trajectory in the bucket without
        retracing. Requires all bucketed trajectories to share the same w0
        (true for this file's single-query suffix-recall shape, where w0 is
        fixed by n_chunks/chunk_len/state_len alone — asserted at setup)."""
        @nnx.jit
        def step(model, optimizer, tokens, mask, loss_mask):
            def loss_fn(model):
                logits = model(tokens, mask)
                lp = jax.nn.log_softmax(logits[:, w0 - 1:Lb - 1], axis=-1)
                tgt = tokens[:, w0:Lb]
                nll = -jnp.take_along_axis(lp, tgt[..., None], axis=-1).squeeze(-1)
                m = loss_mask[w0:Lb]
                denom = jnp.sum(m) * tokens.shape[0]
                return jnp.sum(nll * m[None, :]) / jnp.maximum(denom, 1.0)
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            if update_takes_model:
                optimizer.update(model, grads)
            else:
                optimizer.update(grads)
            return loss
        return step

    global_step = 0
    for stage_i, stage in enumerate(hp['curriculum']):
        # Warm-start each stage (after the first) from the PREVIOUS stage's own
        # BEST checkpoint, not whatever the live model happens to hold — a real gap
        # found (2026-07-31): the model object persists across stages unconditionally,
        # so a stage that collapsed late (loss still fine, but val match cratered —
        # see the grad-clipping comment above) would silently hand that collapsed
        # state to the NEXT stage, compounding rather than recovering. Opt-out via
        # `hp['warm_start_from_best']=False` (default True) to reproduce the exact
        # prior always-continue behavior.
        if stage_i > 0 and hp.get('warm_start_from_best', True):
            prev_best_path = os.path.join(ckpt_dir, f'stage{stage_i - 1}_best.pt')
            if os.path.exists(prev_best_path):
                prev_payload = load_checkpoint(prev_best_path, model)
                _log(f'\n[stage {stage_i}] warm-started from stage {stage_i - 1}\'s own best '
                    f'checkpoint (step={prev_payload["step"]}, val_mean='
                    f'{prev_payload.get("val_mean")}) instead of continuing from wherever '
                    f'stage {stage_i - 1} training ended')
            else:
                _log(f'\n[stage {stage_i}] warm_start_from_best=True but no '
                    f'stage{stage_i - 1}_best.pt found — continuing from live model state')

        trajectories = [_build_trajectory(hp, e, stage['chunk_len']) for e in stage['weave_mix']]
        weights = np.array([t['weight'] for t in trajectories], dtype=np.float64)
        weights /= weights.sum()

        # Adaptive weave_mix reweighting — JAX port of kvmem.hmn's own `train()`
        # mechanism (merged there from the former kvmem/hmn_adaptive_trainer.py),
        # same formula, same config keys. Off by default (`hp['adaptive']=False`)
        # — every existing config unaffected. Without this, a mixed-difficulty
        # weave_mix (e.g. short-and-easy + long-and-hard entries together) samples
        # every entry at its STATIC config weight forever — no mechanism ever
        # shifts sampling effort toward entries the model is still struggling
        # with, or away from ones it's already solved. That's a real gap: mixing
        # entries of varying difficulty is not the same thing as a curriculum
        # unless something adapts to per-entry performance.
        adaptive = hp.get('adaptive', False)
        adapt_signal = hp.get('adapt_signal', 'val_match')
        assert adapt_signal in ('val_match', 'train_loss')
        adapt_temp = hp.get('adapt_temp', 1.0)
        adapt_floor = hp.get('adapt_floor', 0.05)
        adapt_ema_alpha = hp.get('adapt_ema_alpha', 0.5)
        for t in trajectories:
            t['base_weight'] = t['weight']
            t['ema_loss'] = None
            t['last_match'] = None
        _eval_count = 0

        def _temp_softmax_rescale(diffs):
            """softmax(diffs/adapt_temp), rescaled so a perfectly uniform difficulty
            maps every trajectory back to d=1.0 — direct port of kvmem.hmn's own
            helper (pure NumPy, unchanged)."""
            n = len(diffs)
            scores = diffs / adapt_temp
            scores = scores - scores.max()
            exp_s = np.exp(scores)
            p = exp_s / exp_s.sum()
            return p * n

        def _adapt_reweight():
            """Recompute sampling weights from the chosen difficulty signal —
            direct port of kvmem.hmn's own `_adapt_reweight` (identical formula):
            harder-than-average trajectories get scaled up, easier ones scaled
            down (never below `adapt_floor`'s relative share)."""
            if adapt_signal == 'val_match':
                diffs = np.array([max(100.0 - (t['last_match'] if t['last_match'] is not None else 50.0), 0.0)
                                  for t in trajectories])
            else:
                known = [t['ema_loss'] for t in trajectories if t['ema_loss'] is not None]
                fallback = (sum(known) / len(known)) if known else 1.0
                diffs = np.array([t['ema_loss'] if t['ema_loss'] is not None else fallback
                                  for t in trajectories])
            diffs = diffs / max(diffs.mean(), 1e-8)
            diffs = _temp_softmax_rescale(diffs)
            for t, d in zip(trajectories, diffs):
                t['weight'] = t['base_weight'] * (adapt_floor + (1 - adapt_floor) * d)
            new_weights = np.array([t['weight'] for t in trajectories], dtype=np.float64)
            return new_weights / new_weights.sum()

        B = stage['B']
        n_steps = stage['n_steps']
        log_every = hp.get('log_every', 100)
        eval_every = stage.get('eval_every', n_steps)

        lr_schedule = _make_schedule(hp, n_steps)
        # Gradient clipping — a real gap this port had (2026-07-31): kvmem.hmn's own
        # train() clips every step (`torch.nn.utils.clip_grad_norm_(model.parameters(),
        # 1.0)`, right after backward()); this file had NO clipping anywhere, plain
        # optax.adamw. A single large/outlier gradient step going unclipped is a
        # plausible root cause of the sudden multi-thousand-step-in collapses observed
        # (hmn_tpu_recall1024_jax_curriculum_staged.py's stage 0: peaked at 34.7% then
        # collapsed to ~1% over a few evals with no warning in the logged loss).
        # `hp['grad_clip_norm']` (default 1.0, matching kvmem.hmn's own hardcoded value)
        # -- set to None/0 to disable and reproduce the exact prior (unclipped) behavior.
        grad_clip_norm = hp.get('grad_clip_norm', 1.0)
        if grad_clip_norm:
            tx = optax.chain(optax.clip_by_global_norm(grad_clip_norm),
                             optax.adamw(lr_schedule, weight_decay=hp.get('wd', 0.0)))
        else:
            tx = optax.adamw(lr_schedule, weight_decay=hp.get('wd', 0.0))
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        # nnx.Optimizer.update's signature changed across flax versions: newer
        # (verified 0.12.8) takes (model, grads) positionally; older (verified
        # 0.10.7, the newest installable on a TPU VM still on Python 3.10 — see
        # kvmem/setup_tpu_jax.sh's own comment) takes just (grads), with the model
        # reference already stored at __init__ time. Both signatures also accept
        # **kwargs, so a raw parameter-COUNT check is not a valid discriminator
        # (caught directly: it returned True on tpu2's older API too, since
        # (grads, **kwargs) is already 2 params) — check for the 'model' NAME
        # specifically. Detected once per stage (cheap) rather than per-step.
        _update_takes_model = 'model' in inspect.signature(optimizer.update).parameters

        bucket_lengths = hp.get('bucket_lengths', False)
        forward_granularity = hp.get('forward_granularity')
        segment_checkpoint = hp.get('segment_checkpoint', False)
        hop_drop_prob = stage.get('hop_drop_prob', hp.get('hop_drop_prob', 0.0))
        enc_hops = hp.get('enc_hops', -1)
        if hop_drop_prob > 0:
            assert enc_hops != -1, "hop_drop_prob>0 requires hp['enc_hops']>=1 (see " \
                "chunk_mask_fb_traj's own docstring — dropout has nothing to drop when the " \
                "encoding-chain window is unbounded)"
            assert not bucket_lengths and forward_granularity is None, (
                "hop_drop_prob>0 is not currently compatible with bucket_lengths or "
                "forward_granularity — bucketing pads to a shared Lb per group (the per-"
                "trajectory dropout mask would need re-padding every step) and segmented "
                "forward walks the encode pass as independent KV-cached groups no finer than "
                "enc_hops chunks at a time (a group boundary narrower than the hop window "
                "would silently truncate cross-chunk relay); neither integration has been "
                "built. Use the plain (non-bucketed, non-segmented) path for hop-dropout runs.")
        if forward_granularity is not None:
            assert not bucket_lengths, (
                'forward_granularity + bucket_lengths not supported together yet — '
                'each trajectory uses its own exact (unpadded) L in this path')
            for t in trajectories:
                enc_segs = _iter_forward_segments_jax(t['pos_content'])
                t['loss_mask'] = jnp.ones(t['L'], dtype=jnp.float32)
                t['step_fn'] = _make_train_step_segmented(
                    t['w0'], t['c1'], enc_segs, forward_granularity, segment_checkpoint,
                    _update_takes_model)
        elif bucket_lengths:
            max_buckets = hp.get('max_shape_buckets', 8)
            token_budget = hp.get('token_budget', 131072)
            attn_sq_budget = hp.get('attn_sq_budget', 125_000_000)
            ceilings = _bucket_ceilings([t['L'] for t in trajectories],
                                       [t['weight'] for t in trajectories], max_buckets)
            w0_ref = trajectories[0]['w0']
            for t in trajectories:
                assert t['w0'] == w0_ref, (
                    f'bucket_lengths requires every entry in a stage to share w0 '
                    f'(got {t["w0"]} vs {w0_ref} for {t["dsl"]!r}) — see '
                    f'_make_train_step_bucket docstring')
                Lb = _assign_bucket(t['L'], ceilings)
                t['Lb'] = Lb
                t['mask_bucket'] = jnp.asarray(_pad_mask_to(np.asarray(t['mask']), Lb))
                loss_mask = np.zeros(Lb, dtype=np.float32)
                loss_mask[t['w0']:t['c1']] = 1.0
                t['loss_mask'] = jnp.asarray(loss_mask)
                b_cap = B
                b_cap = min(b_cap, _pow2_floor(token_budget / Lb))
                b_cap = min(b_cap, _pow2_floor(attn_sq_budget / (Lb * Lb)))
                t['B_bucket'] = max(1, b_cap)

            bucket_step_fns = {Lb: _make_train_step_bucket(w0_ref, Lb, _update_takes_model)
                               for Lb in sorted(set(t['Lb'] for t in trajectories))}
            for t in trajectories:
                t['step_fn'] = bucket_step_fns[t['Lb']]

            by_bucket: dict[int, list] = {}
            for t in trajectories:
                by_bucket.setdefault(t['Lb'], []).append(t)
            _log(f'\n[stage {stage_i}] bucket table (max_buckets={max_buckets}):')
            for Lb in sorted(by_bucket):
                ts = by_bucket[Lb]
                mean_L = sum(t['L'] for t in ts) / len(ts)
                waste = 100.0 * (1.0 - mean_L / Lb)
                _log(f'  Lb={Lb:<6} n_entries={len(ts):<3} B={ts[0]["B_bucket"]:<4} '
                    f'mean_real_L={mean_L:.0f}  waste={waste:.1f}%')
        else:
            for traj in trajectories:
                traj['step_fn'] = _make_train_step(traj['w0'], traj['c1'], _update_takes_model)

        _log(f'\n[stage {stage_i}] chunk_len={stage["chunk_len"]} n_entries={len(trajectories)} '
             f'B={B} steps={n_steps} bucket_lengths={bucket_lengths} '
             f'forward_granularity={forward_granularity} segment_checkpoint={segment_checkpoint}')

        early_stop_mean = stage.get('early_stop_mean')
        stage_best_val = -1.0
        t_start = time.time()
        # Tracks which underlying compiled `step_fn`s have already been called once —
        # keyed by `id(step_fn)`, not per-trajectory, since `step_fn` is SHARED across
        # every trajectory in a bucket (`bucket_lengths=True`) or a shared Lb group.
        # JAX compiles lazily on first call, so timing that first call directly
        # measures compile+execute — `float(loss)` forces the (otherwise async)
        # dispatch to actually finish before the timer stops, or the measurement
        # would just capture dispatch overhead, not the real compile wait.
        _compiled_step_fn_ids: set[int] = set()
        _total_compile_s = 0.0
        _n_distinct_shapes = len(set(id(t['step_fn']) for t in trajectories))
        pbar = tqdm(range(1, n_steps + 1), desc=f'stage{stage_i}', dynamic_ncols=True, file=status_file)
        for local_step in pbar:
            global_step += 1
            traj = trajectories[rng.choice(len(trajectories), p=weights)]
            B_eff = traj['B_bucket'] if bucket_lengths else B
            tok_np = make_batch_tagged(rng, B_eff, traj['n_chunks'], traj['chunk_len'], hp['state_len'],
                                       hp['state_vocab_size'], traj['pos_content'], traj['tags'],
                                       data_kind=hp.get('data_kind', 'random'),
                                       data_target_bits=hp.get('data_target_bits'))
            if bucket_lengths and traj['Lb'] != traj['L']:
                tok_np = _pad_tok_to(tok_np, traj['Lb'])
            tokens = jnp.asarray(tok_np, dtype=jnp.int32)

            step_fn = traj['step_fn']
            is_first_call = id(step_fn) not in _compiled_step_fn_ids
            if is_first_call:
                _compile_t0 = time.time()

            if forward_granularity is not None:
                loss = step_fn(model, optimizer, tokens, traj['mask'], traj['loss_mask'])
            elif bucket_lengths:
                loss = step_fn(model, optimizer, tokens, traj['mask_bucket'], traj['loss_mask'])
            elif hop_drop_prob > 0:
                # Stochastic "hop dropout" (LayerDrop-style, applied along the chunk/time
                # axis rather than depth): rebuilt fresh every step from the trajectory's
                # own `pos_mask` (host-side numpy, cheap relative to the device forward/
                # backward pass) — back=1 (the immediately preceding chunk) is never
                # dropped, back 2..enc_hops each independently dropped w.p. hop_drop_prob.
                # eval/decode always use `traj['mask']` (the static full-window mask, no
                # dropout) — this is train-time-only regularization, same convention as
                # any other dropout mechanism.
                active_backs = _sample_enc_active_backs(rng, traj['enc_hops'], hop_drop_prob,
                                                        traj['n_chunks'])
                mask_step = jnp.asarray(chunk_mask_fb_traj(
                    traj['pos_mask'], hops=-1, enc_hops=traj['enc_hops'],
                    enc_active_backs=active_backs))
                loss = step_fn(model, optimizer, tokens, mask_step)
            else:
                loss = step_fn(model, optimizer, tokens, traj['mask'])

            if is_first_call:
                float(loss)  # force the async dispatch to actually finish before stopping the timer
                compile_s = time.time() - _compile_t0
                _total_compile_s += compile_s
                _compiled_step_fn_ids.add(id(step_fn))
                shape_desc = f'Lb={traj["Lb"]}' if bucket_lengths else f'L={traj["L"]}'
                _log(f'  [compile] step_fn for {shape_desc:<12} B={B_eff:<4} '
                    f'first-call={compile_s:.1f}s  total_compile={_total_compile_s:.1f}s '
                    f'({len(_compiled_step_fn_ids)}/{_n_distinct_shapes} shapes compiled so far)')
                if len(_compiled_step_fn_ids) == _n_distinct_shapes:
                    _log(f'  [compile] ALL {_n_distinct_shapes} shapes compiled — '
                        f'total_compile_time={_total_compile_s:.1f}s')

            # Per-trajectory loss bookkeeping — tracked every step (not just for
            # adapt_signal='train_loss') so `weights`/`traj_loss` can be logged
            # together at every log_every boundary for later plotting (weight vs.
            # loss trajectory per entry over training). `float(loss)` here is the
            # same host-sync already needed by the log_every block below in the
            # common case; the extra cost on non-log_every steps is the tradeoff
            # for having a per-trajectory loss curve at all, not just the single
            # sampled trajectory's own.
            _loss_f_step = float(loss)
            traj['last_loss'] = _loss_f_step
            traj['ema_loss'] = _loss_f_step if traj['ema_loss'] is None else (
                traj['ema_loss'] * adapt_ema_alpha + _loss_f_step * (1 - adapt_ema_alpha))

            if local_step % log_every == 0 or local_step == n_steps:
                loss_f = _loss_f_step
                lr = float(lr_schedule(global_step))
                elapsed = time.time() - t_start
                pbar.set_postfix(loss=f'{loss_f:.3f}', lr=f'{lr:.1e}', entry=traj['dsl'], refresh=False)
                _traj_loss = [round(t['last_loss'], 4) if t.get('last_loss') is not None else None
                             for t in trajectories]
                _weights_list = [round(float(w), 5) for w in weights]
                _jlog(dict(step=global_step, stage=stage_i, loss=round(loss_f, 5), lr=lr, entry=traj['dsl'],
                          weights=_weights_list, traj_loss=_traj_loss))
                print(str(pbar), file=log_file, flush=True)
                _traj_loss_str = '[' + ','.join(f'{v:.2f}' if v is not None else 'NA' for v in _traj_loss) + ']'
                _weights_str = '[' + ','.join(f'{w:.2f}' for w in _weights_list) + ']'
                _log(f'stage{stage_i} step={local_step}/{n_steps} g={global_step} '
                    f'loss={loss_f:.4f} {elapsed:.1f}s  weights={_weights_str}  traj_loss={_traj_loss_str}')
                if not np.isfinite(loss_f):
                    _log(f'  !! non-finite loss on entry {traj["dsl"]!r} (L={traj["L"]})')

            if local_step % eval_every == 0 or local_step == n_steps:
                elapsed = time.time() - t_start
                h, m = divmod(int(elapsed), 3600); m, s = divmod(m, 60)
                _log(f'\n--- stage={stage_i} step={local_step}/{n_steps}  g={global_step} '
                    f'loss={float(loss):.4f}  {h:02d}:{m:02d}:{s:02d} ---')

                _eval_t0 = time.time()
                val_means = []
                for t in trajectories:
                    val_seqs = make_test_sequences(t['n_chunks'] * t['chunk_len'])
                    if val_n_seqs is not None:
                        val_seqs = dict(list(val_seqs.items())[:val_n_seqs])
                    pcts = []
                    _traj_decode_t0 = time.time()
                    for seq_bytes in val_seqs.values():
                        chunks_list = [seq_bytes[k * t['chunk_len']:(k + 1) * t['chunk_len']]
                                      for k in range(t['n_chunks'])]
                        r = ar_decode_traj_kv_jit(model, np.array(chunks_list), hp['state_len'],
                                                  hp['state_vocab_size'], np.asarray(t['mask']),
                                                  t['pos_content'], t['tags'])
                        pcts.append(r['match_pct'])
                    t['last_decode_s'] = time.time() - _traj_decode_t0
                    m_ = sum(pcts) / len(pcts)
                    t['last_match'] = m_
                    val_means.append(m_)
                    _log(f'  val/weave/{t["dsl"]:<30} match={m_:.1f}%  decode={t["last_decode_s"]:.1f}s '
                        f'(out_len={t["pos_content"]["rec_blocks"][0]["out_len"]})')
                vmean = sum(val_means) / len(val_means)
                _eval_total_s = time.time() - _eval_t0
                _log(f'  val/weave/MEAN                  match={vmean:.1f}%')
                _log(f'  val/weave/decode_time_total      {_eval_total_s:.1f}s  '
                    f'({len(trajectories)} entries, {(val_n_seqs or 1)} seqs each)')
                _jlog(dict(step=global_step, stage=stage_i, eval_mean=round(vmean, 2),
                          eval_decode_total_s=round(_eval_total_s, 2),
                          traj_decode_s=[round(t.get('last_decode_s', 0.0), 3) for t in trajectories]))

                by_cl: dict[int, list[float]] = {}
                for t in trajectories:
                    by_cl.setdefault(t['chunk_len'], []).append(t['last_match'])
                for cl in sorted(by_cl):
                    cl_mean = sum(by_cl[cl]) / len(by_cl[cl])
                    _log(f'  val/weave/by_chunk_len/{cl:<4}     match={cl_mean:.1f}%  (n={len(by_cl[cl])})')

                if hp.get('eval_combinatorial_hops', False) and any(t['enc_hops'] != -1 for t in trajectories):
                    # "Combinatorial try each hop size" — for every subset S of the back
                    # distances {2..enc_hops} (always unioned with {1}, since back=1 is
                    # never dropped in training either), decode every trajectory with a
                    # mask that applies THAT SAME subset uniformly (every chunk and the
                    # query alike) instead of the full deterministic window, and report
                    # the aggregate MEAN for that subset. This is a direct read of which
                    # relay distances the model actually depends on vs. tolerates losing
                    # (a hop-dropout-trained model should degrade gracefully as S shrinks;
                    # one trained with hop_drop_prob=0 is the sharpest test of whether the
                    # bounded window is load-bearing at all beyond back=1).
                    import itertools
                    H = next(t['enc_hops'] for t in trajectories if t['enc_hops'] != -1)
                    for r_size in range(0, H):  # subset sizes of {2..H}: 0 backs .. H-1 backs
                        for combo in itertools.combinations(range(2, H + 1), r_size):
                            S = frozenset({1, *combo})
                            combo_means = []
                            for t in trajectories:
                                if t['enc_hops'] == -1:
                                    continue
                                active_backs = {k: S for k in range(1, t['n_chunks'])}
                                active_backs['query'] = S
                                mask_combo = np.asarray(chunk_mask_fb_traj(
                                    t['pos_mask'], hops=-1, enc_hops=t['enc_hops'],
                                    enc_active_backs=active_backs))
                                val_seqs = make_test_sequences(t['n_chunks'] * t['chunk_len'])
                                if val_n_seqs is not None:
                                    val_seqs = dict(list(val_seqs.items())[:val_n_seqs])
                                pcts = []
                                for seq_bytes in val_seqs.values():
                                    chunks_list = [seq_bytes[k * t['chunk_len']:(k + 1) * t['chunk_len']]
                                                  for k in range(t['n_chunks'])]
                                    r = ar_decode_traj_kv_jit(model, np.array(chunks_list), hp['state_len'],
                                                              hp['state_vocab_size'], mask_combo,
                                                              t['pos_content'], t['tags'])
                                    pcts.append(r['match_pct'])
                                combo_means.append(sum(pcts) / len(pcts))
                            combo_mean = sum(combo_means) / len(combo_means)
                            S_str = '{' + ','.join(str(b) for b in sorted(S)) + '}'
                            _log(f'  val/weave/hopcombo/S={S_str:<12} MEAN={combo_mean:.1f}%  '
                                f'(n_entries={len(combo_means)})')
                            _jlog(dict(step=global_step, stage=stage_i, hopcombo=sorted(S),
                                      hopcombo_mean=round(combo_mean, 2)))

                _eval_count += 1
                if adaptive:
                    # val_match's very first reading is the noisiest possible signal
                    # (least-trained model) — skip adapting on it, matches kvmem.hmn's
                    # own train_loss doesn't have this problem (EMA already accumulating
                    # every step since step 1).
                    if adapt_signal == 'train_loss' or _eval_count >= 2:
                        weights = _adapt_reweight()
                        _log(f'  [stage {stage_i}] adaptive reweight applied (signal={adapt_signal}): '
                            + ', '.join(f'{t["dsl"]}={w:.2f}' for t, w in zip(trajectories, weights)))
                    else:
                        _log(f'  [stage {stage_i}] adaptive=True but adapt_signal=val_match skips '
                            f'adapting until the 2nd eval (this is eval #{_eval_count})')

                save_checkpoint(os.path.join(ckpt_dir, f'stage{stage_i}_last.pt'), model, hp, global_step)
                if vmean > stage_best_val:
                    stage_best_val = vmean
                    save_checkpoint(os.path.join(ckpt_dir, f'stage{stage_i}_best.pt'), model, hp,
                                    global_step, val_mean=vmean)

                if early_stop_mean is not None and vmean >= early_stop_mean:
                    _log(f'  [stage {stage_i}] EARLY STOP: val MEAN {vmean:.1f}% >= '
                        f'early_stop_mean={early_stop_mean} at step {local_step}/{n_steps}')
                    break

        save_checkpoint(os.path.join(ckpt_dir, f'stage{stage_i}_end.pt'), model, hp, global_step)
        _log(f'[stage {stage_i}] done. saved stage{stage_i}_end.pt (best={stage_best_val:.1f}%)')


def _sanity_check():
    """Builds the model at hmn_notags_w25_rope.py's exact architecture
    (d=64, n_layers=8, n_heads=4, rope=True, yarn defaults, null_kv=True,
    rmsnorm=True, state_len=8/state_vocab_size=2 -> V=271), constructs a
    REAL batch via THIS FILE's own (copied, not imported) data pipeline for
    that config's actual stage-0 entries, runs one forward pass, and checks
    the output shape and finiteness — a real functional smoke test, not
    just "does it run." No training, no weight-parity claim against the
    PyTorch model (different init scheme/RNG entirely) — this only verifies
    the JAX/Flax NNX architecture itself is correct and numerically sane on
    real data shapes from the real DSL/masking pipeline."""
    hp = load_config('kvmem/configs/hmn_notags_w25_rope.py')
    entry = hp['curriculum'][0]['weave_mix'][0]
    traj = _build_trajectory(hp, entry, hp['curriculum'][0]['chunk_len'])
    B = 4

    print(f'entry: {entry["dsl"]!r}  L={traj["L"]}  chunk_len={traj["chunk_len"]}  B={B}')

    rngs = nnx.Rngs(0)
    model = build_model(hp, rngs)
    n_params = model.count_params()
    print(f'JAX/Flax NNX model: {n_params:,} params')

    tok_np = make_batch_tagged(np.random.default_rng(0), B, traj['n_chunks'], traj['chunk_len'],
                               hp['state_len'], hp['state_vocab_size'], traj['pos_content'], traj['tags'])
    tokens = jnp.asarray(tok_np, dtype=jnp.int32)
    logits = model(tokens, traj['mask'])

    assert logits.shape == (B, traj['L'], hp.get('V_out', 256)), f'unexpected logits shape {logits.shape}'
    finite = bool(jnp.all(jnp.isfinite(logits)))
    print(f'logits shape: {logits.shape}  finite: {finite}  '
         f'mean={float(jnp.mean(logits)):.4f}  std={float(jnp.std(logits)):.4f}')
    assert finite, 'NaN/Inf in forward pass output — architecture port has a bug'
    print('PASS — forward pass runs, output shape correct, all logits finite.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    args = parser.parse_args()

    print(f'JAX backend: {jax.default_backend()}  devices: {jax.devices()}')
    if args.config is None:
        _sanity_check()
    else:
        hp = load_config(args.config)
        train_jax(hp)
