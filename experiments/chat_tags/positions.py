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
    WINDOW_QUERY_TAGS,
    HMN_QUERY_A_OPEN, HMN_QUERY_A_CLOSE,
    HMN_QUERY_B_OPEN, HMN_QUERY_B_CLOSE,
    HMN_QUERY_C_OPEN, HMN_QUERY_C_CLOSE,
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

    # Window-specific query tags: use <query_a/b/c> for the three canonical
    # windows (warmup_x_fixed in {0,16,32}) so the model gets an explicit
    # window-identity signal instead of one shared <query> anchor. Uniform-X
    # training (warmup_x_fixed=None) keeps the generic tag since arbitrary X
    # doesn't map to one named window.
    query_open, query_close = WINDOW_QUERY_TAGS.get(warmup_x_fixed, (HMN_QUERY_OPEN, HMN_QUERY_CLOSE))

    def _emit_iq_block():
        nonlocal offset
        tags.append((offset, HMN_MEM_OPEN)); offset += 1
        sl0 = offset; sl1 = sl0 + slot_len; offset = sl1
        tags.append((offset, HMN_MEM_CLOSE)); offset += 1
        tags.append((offset, query_open)); offset += 1
        w0 = offset; w1 = w0 + warmup_len; offset = w1
        tags.append((offset, query_close)); offset += 1
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
        tags.append((offset, query_open)); offset += 1
        w0 = offset; w1 = w0 + warmup_len; offset = w1
        tags.append((offset, query_close)); offset += 1
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


# Query tags assigned by POSITION IN THE SCHEDULE (not by byte offset — SRS spans
# aren't keyed by a fixed X the way iq_global_rw's 3 canonical windows are).
# srs_schedule_depth2(n) always yields exactly 3 spans, so this covers the first
# SRS experiment with zero new vocab. A 7-span full srs_schedule(4) would need
# 4 more tags — not yet defined, deferred to a follow-up once depth-2 is validated.
_SRS_SPAN_TAGS = [
    (HMN_QUERY_A_OPEN, HMN_QUERY_A_CLOSE),
    (HMN_QUERY_B_OPEN, HMN_QUERY_B_CLOSE),
    (HMN_QUERY_C_OPEN, HMN_QUERY_C_CLOSE),
]


def chunk_positions_srs_tagged(n_chunks: int, chunk_len: int, slot_len: int,
                               warmup_len: int, schedule: list[tuple[int, int]],
                               n_refine: int = 2) -> dict:
    """
    True SRS: each span in `schedule` (e.g. srs_schedule_depth2(n_chunks) ->
    [(0,half),(half,n),(0,n)]) gets its OWN local IQ turn + n_refine chained
    argmax-IR turns, reusing the exact tag-wrapping pattern from
    chunk_positions_iq_global_rw_tagged's _emit_iq_block/_emit_ir_block, but with
    the query tag assigned by the span's POSITION in the schedule (via
    _SRS_SPAN_TAGS) rather than by a fixed byte offset X.

    Structurally this is chunk_positions_fb_localrefine (kvmem/train_hmn_chunk.py)
    with tags added — same "one shared encoding pass, then each span threaded in
    sequence with its own local IQ+IR" shape, same reliance on chunk_mask_fb's
    Rule 3b (IQ SLOT blocked from ALL tokens in prior rec_blocks) for the
    nochain/no-cross-span-leak property, reused unmodified.

    Requires len(schedule) <= len(_SRS_SPAN_TAGS) (3, for now).
    """
    if len(schedule) > len(_SRS_SPAN_TAGS):
        raise ValueError(f'{len(schedule)} spans requested but only '
                         f'{len(_SRS_SPAN_TAGS)} span tags defined — extend '
                         f'_SRS_SPAN_TAGS (and vocab.py) for larger schedules')

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
    rec_blocks_c: list[dict] = []
    rec_blocks_m: list[dict] = []

    for span_i, span in enumerate(schedule):
        span_s, span_e = span
        span_len = (span_e - span_s) * chunk_len
        out_len  = span_len - warmup_len
        query_open, query_close = _SRS_SPAN_TAGS[span_i]

        def _emit_iq_block():
            nonlocal offset
            tags.append((offset, HMN_MEM_OPEN)); offset += 1
            sl0 = offset; sl1 = sl0 + slot_len; offset = sl1
            tags.append((offset, HMN_MEM_CLOSE)); offset += 1
            tags.append((offset, query_open)); offset += 1
            w0 = offset; w1 = w0 + warmup_len; offset = w1
            tags.append((offset, query_close)); offset += 1
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
            tags.append((offset, query_open)); offset += 1
            w0 = offset; w1 = w0 + warmup_len; offset = w1
            tags.append((offset, query_close)); offset += 1
            tags.append((offset, HMN_RESPONSE_OPEN)); offset += 1
            c0 = offset; c1 = c0 + out_len; offset = c1
            tags.append((offset, HMN_RESPONSE_CLOSE)); offset += 1
            c_fields = dict(sla0=sla0, sla1=sla1, am0=am0, am1=am1, slb0=slb0, slb1=slb1,
                            w0=w0, w1=w1, c0=c0, c1=c1)
            m_fields = dict(sla0=sla0 - 1, sla1=sla1 + 1, am0=am0 - 1, am1=am1 + 1,
                            slb0=slb0 - 1, slb1=slb1 + 1, w0=w0 - 1, w1=w1 + 1,
                            c0=c0 - 1, c1=c1 + 1)
            return c_fields, m_fields

        iq_c, iq_m = _emit_iq_block()
        # SRS spans have no random warmup offset (warmup is always the span's own
        # start byte, X=0 within the span) — make_batch_tagged/ar_decode_iq_global_rw_tagged
        # (reused unmodified from chat_tags) expect this field on every IQ block, so
        # give it a degenerate fixed (0,0) range rather than touching that shared code.
        rec_blocks_c.append(dict(type='iq', span=span, span_len=span_len,
                                 out_len=out_len, is_clean=(n_refine == 0),
                                 warmup_train_range=(0, 0), warmup_x_dist='fixed', **iq_c))
        rec_blocks_m.append(dict(type='iq', **iq_m))

        prev_c0_c = iq_c['c0']
        for _ in range(n_refine):
            ir_c, ir_m = _emit_ir_block()
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
