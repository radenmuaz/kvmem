"""
Tag-aware position builder for the iq_global_rw trajectory.

Mirrors kvmem.train_hmn_chunk.chunk_positions_iq_global_rw region-by-region, but
wraps every region (source chunk, SLOT/mem, warmup/query, output/response) with
an explicit open/close boundary token.

Produces TWO parallel views of the same physical sequence (both share L and
warmup_len):
  - pos_content: field ranges denote exactly the same content as the untagged
    layout (e.g. sl1-sl0 == slot_len always). Fed to the batch filler / decode.
  - pos_mask: every field range widened by exactly 1 on each side to absorb its
    immediately-adjacent tag token. Fed unmodified to
    kvmem.train_hmn_chunk.chunk_mask_fb — required for correctness: a tag row
    not folded into its content's row-blocking group would get full causal
    (unrestricted) visibility and become a leak path around the SLOT
    bottleneck, even though the tag's own value is constant.

See /Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md for the
full design rationale.
"""
from __future__ import annotations

from experiments.chat_tags.vocab import (
    HMN_SRC_OPEN, HMN_SRC_CLOSE,
    HMN_MEM_OPEN, HMN_MEM_CLOSE,
    HMN_QUERY_OPEN, HMN_QUERY_CLOSE,
    HMN_RESPONSE_OPEN, HMN_RESPONSE_CLOSE,
)


def chunk_positions_iq_global_rw_tagged(n_chunks: int, chunk_len: int, slot_len: int,
                                        warmup_len: int, window_chunks: int = 2,
                                        warmup_x_fixed: int | None = None,
                                        warmup_x_dist: str = 'uniform',
                                        n_refine: int = 0) -> dict:
    """
    Returns dict(pos_content=..., pos_mask=..., tags=[(position, token_id), ...], L=...).

    Sequence (n_refine=0):
      per chunk k: <src> chunk_k </src> <mem> SLOT </mem>
      IQ:          <mem> SLOT </mem> <query> warmup </query> <response> out </response>
    Sequence (n_refine>0) additionally appends, per refine step:
      IR:  <mem> SLOT_A </mem> <response> argmax </response> <mem> SLOT_B </mem>
           <query> warmup </query> <response> out </response>
    """
    enc_blocks_c: list[dict] = []
    enc_blocks_m: list[dict] = []
    tags: list[tuple[int, int]] = []
    offset = 0

    for _ in range(n_chunks):
        tags.append((offset, HMN_SRC_OPEN)); offset += 1
        s0 = offset; s1 = s0 + chunk_len; offset = s1
        tags.append((offset, HMN_SRC_CLOSE)); offset += 1
        tags.append((offset, HMN_MEM_OPEN)); offset += 1
        sl0 = offset; sl1 = sl0 + slot_len; offset = sl1
        tags.append((offset, HMN_MEM_CLOSE)); offset += 1

        enc_blocks_c.append(dict(s0=s0, s1=s1, sl0=sl0, sl1=sl1))
        enc_blocks_m.append(dict(s0=s0 - 1, s1=s1 + 1, sl0=sl0 - 1, sl1=sl1 + 1))

    enc_end = offset

    out_len = window_chunks * chunk_len - warmup_len

    def _emit_iq_block():
        nonlocal offset
        tags.append((offset, HMN_MEM_OPEN)); offset += 1
        sl0 = offset; sl1 = sl0 + slot_len; offset = sl1
        tags.append((offset, HMN_MEM_CLOSE)); offset += 1
        tags.append((offset, HMN_QUERY_OPEN)); offset += 1
        w0 = offset; w1 = w0 + warmup_len; offset = w1
        tags.append((offset, HMN_QUERY_CLOSE)); offset += 1
        tags.append((offset, HMN_RESPONSE_OPEN)); offset += 1
        c0 = offset; c1 = c0 + out_len; offset = c1
        tags.append((offset, HMN_RESPONSE_CLOSE)); offset += 1
        return (dict(sl0=sl0, sl1=sl1, w0=w0, w1=w1, c0=c0, c1=c1),
                dict(sl0=sl0 - 1, sl1=sl1 + 1, w0=w0 - 1, w1=w1 + 1, c0=c0 - 1, c1=c1 + 1))

    def _emit_ir_block():
        nonlocal offset
        tags.append((offset, HMN_MEM_OPEN)); offset += 1
        sla0 = offset; sla1 = sla0 + slot_len; offset = sla1
        tags.append((offset, HMN_MEM_CLOSE)); offset += 1
        tags.append((offset, HMN_RESPONSE_OPEN)); offset += 1
        am0 = offset; am1 = am0 + out_len; offset = am1
        tags.append((offset, HMN_RESPONSE_CLOSE)); offset += 1
        tags.append((offset, HMN_MEM_OPEN)); offset += 1
        slb0 = offset; slb1 = slb0 + slot_len; offset = slb1
        tags.append((offset, HMN_MEM_CLOSE)); offset += 1
        tags.append((offset, HMN_QUERY_OPEN)); offset += 1
        w0 = offset; w1 = w0 + warmup_len; offset = w1
        tags.append((offset, HMN_QUERY_CLOSE)); offset += 1
        tags.append((offset, HMN_RESPONSE_OPEN)); offset += 1
        c0 = offset; c1 = c0 + out_len; offset = c1
        tags.append((offset, HMN_RESPONSE_CLOSE)); offset += 1
        c_fields = dict(sla0=sla0, sla1=sla1, am0=am0, am1=am1, slb0=slb0, slb1=slb1,
                        w0=w0, w1=w1, c0=c0, c1=c1)
        m_fields = dict(sla0=sla0 - 1, sla1=sla1 + 1, am0=am0 - 1, am1=am1 + 1,
                        slb0=slb0 - 1, slb1=slb1 + 1, w0=w0 - 1, w1=w1 + 1,
                        c0=c0 - 1, c1=c1 + 1)
        return c_fields, m_fields

    src_len = n_chunks * chunk_len
    x_max = src_len - warmup_len - out_len
    n_windows = n_chunks - window_chunks + 1
    eval_offsets = [i * chunk_len for i in range(n_windows)]
    train_range = (warmup_x_fixed, warmup_x_fixed) if warmup_x_fixed is not None else (0, x_max)
    _dist = 'fixed' if warmup_x_fixed is not None else warmup_x_dist

    iq_c, iq_m = _emit_iq_block()
    rw_extra = dict(warmup_train_range=train_range, warmup_x_dist=_dist,
                    warmup_valid_offsets=eval_offsets, window_chunks=window_chunks)
    rec_blocks_c = [dict(type='iq', span=(0, n_chunks), span_len=src_len,
                         out_len=out_len, is_clean=(n_refine == 0), **iq_c, **rw_extra)]
    rec_blocks_m = [dict(type='iq', **iq_m)]

    prev_c0_c = iq_c['c0']
    for _ in range(n_refine):
        ir_c, ir_m = _emit_ir_block()
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
