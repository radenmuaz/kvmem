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
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

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


def chunk_mask_fb_traj(pos: dict, hops: int = -1) -> np.ndarray:
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
                      tags: list[tuple[int, int]]) -> np.ndarray:
    """Copied from kvmem.hmn.make_batch_tagged, `data_kind='random'` path only
    (the structured-data branch, `kvmem/structured_data.py`, is not ported)."""
    sids = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
    wl = pos_content['warmup_len']
    L = pos_content['L']
    tok = np.zeros((B, L), dtype=np.int64)
    segs = rng.integers(0, 256, size=(B, n_chunks, chunk_len), dtype=np.int64)

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
    pos = jnp.arange(offset, offset + L, dtype=jnp.float32)
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
                 offset: int = 0):
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
        calls, exactly as kvmem.hmn's own KV-cache callers do."""
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
            K = jnp.concatenate([K_past, K], axis=2)
            V = jnp.concatenate([V_past, V], axis=2)
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
                 offset: int = 0):
        attn_out = self.attn(self.norm(x), mask, past_kv=past_kv, return_kv=return_kv, offset=offset)
        if return_kv:
            attn_out, kv = attn_out
            return x + attn_out, kv
        return x + attn_out


def _block_call(block, x, mask, past_kv, return_kv, offset):
    return block(x, mask, past_kv=past_kv, return_kv=return_kv, offset=offset)


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
                 offset: int = 0):
        """tokens: (B, L) int32/int64  mask: (Lq, Lkv) additive bias -> logits (B, L, V_out).

        Mirrors kvmem.hmn.HMNModel.forward's KV-cache signature exactly:
        `past_kv`: list[n_layers] of (K_past, V_past), one per block, or None.
        `return_kv=True`: returns `(logits, kv_out)` instead of just logits,
        `kv_out` a list[n_layers] of this call's own (K,V) — same shape/
        semantics the PyTorch decode callers (`ar_decode_iq_global_rw_tagged`
        etc.) already rely on, so a caller written against the PyTorch model
        works against this one with no changes beyond the import."""
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
        fn = _block_call_remat if use_ckpt else _block_call
        for i, block in enumerate(self.blocks):
            pkv = past_kv[i] if past_kv is not None else None
            result = fn(block, x, mask, pkv, return_kv, offset)
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
        # has one thing to checkpoint per layer). Any truthy value enables remat.
        grad_checkpoint=bool(hp.get('grad_checkpoint', False)),
        V_out=hp.get('V_out', 256), rngs=rngs,
    )


# =============================================================================
# Training loop — new code (not a port), scoped to non-refine single-Q
# entries only (every hmn_notags_w25*.py config). See module docstring.
# =============================================================================

def _build_trajectory(hp: dict, entry: dict, stage_chunk_len: int) -> dict:
    """One weave_mix entry -> {pos_content, mask (jnp), L, chunk_len, weight}."""
    ops, n_refine, _repeat_batch, dsl_chunk_len, dsl_warmup_len = parse_traj_dsl(entry['dsl'])
    assert n_refine == 0, 'kvmem/hmn_jax.py train_jax does not support refine rounds (R token)'
    chunk_len = dsl_chunk_len if dsl_chunk_len is not None else stage_chunk_len
    warmup_len = dsl_warmup_len if dsl_warmup_len is not None else hp['warmup_len']
    built = chunk_positions_traj(chunk_len, hp['state_len'], warmup_len, ops,
                                 n_refine=0, state_vocab_size=hp['state_vocab_size'])
    pos_content = built['pos_content']
    mask_np = chunk_mask_fb_traj(built['pos_mask'], hops=-1)
    rec_blocks = pos_content['rec_blocks']
    assert len(rec_blocks) == 1 and rec_blocks[0]['type'] == 'initial', \
        'kvmem/hmn_jax.py train_jax only supports one Q per entry (batch/stream/etc. not ported)'
    rb = rec_blocks[0]
    return dict(pos_content=pos_content, tags=built['tags'], mask=jnp.asarray(mask_np),
               L=built['L'], chunk_len=chunk_len, weight=entry['weight'],
               w0=rb['w0'], c1=rb['c1'], dsl=entry['dsl'],
               n_chunks=len(pos_content['enc_blocks']))


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
    """Copied verbatim from kvmem.hmn.make_test_sequences. Deterministic
    held-out test sequences of length seg_len, all bytes in [DATA_LO=0x20,
    0xFF], never protocol bytes."""
    DATA_LO = 0x20
    V = 256 - DATA_LO
    seqs = {}
    seqs['up_counter']   = [DATA_LO + (i % V) for i in range(seg_len)]
    seqs['down_counter'] = [DATA_LO + ((V - 1 - i) % V) for i in range(seg_len)]
    seqs['const_mid']    = [DATA_LO + V // 2] * seg_len
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
    import os
    rng = np.random.default_rng(hp.get('seed', 42))
    rngs = nnx.Rngs(hp.get('seed', 42))
    model = build_model(hp, rngs)
    n_params = model.count_params()

    name = hp.get('name', 'hmn_jax')
    log_dir = os.path.join(log_base, name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'train_jax.log')
    log_file = open(log_path, 'a', buffering=1)

    def _log(msg):
        print(msg)
        print(msg, file=log_file)

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

    global_step = 0
    for stage_i, stage in enumerate(hp['curriculum']):
        trajectories = [_build_trajectory(hp, e, stage['chunk_len']) for e in stage['weave_mix']]
        weights = np.array([t['weight'] for t in trajectories], dtype=np.float64)
        weights /= weights.sum()
        B = stage['B']
        n_steps = stage['n_steps']
        log_every = hp.get('log_every', 100)
        eval_every = stage.get('eval_every', n_steps)

        tx = optax.adamw(_make_schedule(hp, n_steps), weight_decay=hp.get('wd', 0.0))
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
        for traj in trajectories:
            traj['step_fn'] = _make_train_step(traj['w0'], traj['c1'], _update_takes_model)

        _log(f'\n[stage {stage_i}] chunk_len={stage["chunk_len"]} n_entries={len(trajectories)} '
             f'B={B} steps={n_steps}')

        early_stop_mean = stage.get('early_stop_mean')
        stage_best_val = -1.0
        t_start = time.time()
        for local_step in range(1, n_steps + 1):
            global_step += 1
            traj = trajectories[rng.choice(len(trajectories), p=weights)]
            tok_np = make_batch_tagged(rng, B, traj['n_chunks'], traj['chunk_len'], hp['state_len'],
                                       hp['state_vocab_size'], traj['pos_content'], traj['tags'])
            tokens = jnp.asarray(tok_np, dtype=jnp.int32)

            loss = traj['step_fn'](model, optimizer, tokens, traj['mask'])

            if local_step % log_every == 0 or local_step == n_steps:
                elapsed = time.time() - t_start
                _log(f'stage{stage_i} step={local_step}/{n_steps} g={global_step} '
                    f'loss={float(loss):.4f} {elapsed:.1f}s')
                if not np.isfinite(float(loss)):
                    _log(f'  !! non-finite loss on entry {traj["dsl"]!r} (L={traj["L"]})')

            if local_step % eval_every == 0 or local_step == n_steps:
                elapsed = time.time() - t_start
                h, m = divmod(int(elapsed), 3600); m, s = divmod(m, 60)
                _log(f'\n--- stage={stage_i} step={local_step}/{n_steps}  g={global_step} '
                    f'loss={float(loss):.4f}  {h:02d}:{m:02d}:{s:02d} ---')

                val_means = []
                for t in trajectories:
                    val_seqs = make_test_sequences(t['n_chunks'] * t['chunk_len'])
                    if val_n_seqs is not None:
                        val_seqs = dict(list(val_seqs.items())[:val_n_seqs])
                    pcts = []
                    for seq_bytes in val_seqs.values():
                        chunks_list = [seq_bytes[k * t['chunk_len']:(k + 1) * t['chunk_len']]
                                      for k in range(t['n_chunks'])]
                        r = ar_decode_traj_kv(model, np.array(chunks_list), hp['state_len'],
                                              hp['state_vocab_size'], np.asarray(t['mask']),
                                              t['pos_content'], t['tags'])
                        pcts.append(r['match_pct'])
                    m_ = sum(pcts) / len(pcts)
                    t['last_match'] = m_
                    val_means.append(m_)
                    _log(f'  val/weave/{t["dsl"]:<30} match={m_:.1f}%')
                vmean = sum(val_means) / len(val_means)
                _log(f'  val/weave/MEAN                  match={vmean:.1f}%')

                by_cl: dict[int, list[float]] = {}
                for t in trajectories:
                    by_cl.setdefault(t['chunk_len'], []).append(t['last_match'])
                for cl in sorted(by_cl):
                    cl_mean = sum(by_cl[cl]) / len(by_cl[cl])
                    _log(f'  val/weave/by_chunk_len/{cl:<4}     match={cl_mean:.1f}%  (n={len(by_cl[cl])})')

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
