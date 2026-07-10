"""
Stitched AR-decode for the tagged SRS layout, when `schedule` is a set of
OVERLAPPING windows (stride < window size) rather than disjoint
halves+full-span.

Why this exists: the atomic full-span block in srs_depth2_nc4_slot8 (schedule
[(0,2),(2,4),(0,4)]) asks one IQ+IR unit to decode the ENTIRE 64B source in a
single shot (56-byte output) — a mechanism that has never been validated at
this project (see docs/SRS_RECIPE.md "Stitching vs atomic full-span"). Eval
confirmed it as the clear bottleneck: span(0,4) match% was far below both
half-spans on every test sequence (e.g. val: 100/65/3.6, test: 79/4/9).

The proven alternative (`ir_local` track, `ar_decode_chunk_fb_stitch_kv` in
kvmem/train_hmn_chunk.py) never asks for a long single-shot decode: it chains
several small, already-validated 32B windows, seeding each later window's
warmup from the PREVIOUS window's own just-decoded output (valid because
warmup_len=8 always fits inside the 50%-overlap region between adjacent
windows). Full-sequence coverage emerges from the chain, not from one big
block.

This module is that same stitching mechanism, adapted for the tag-wrapped SRS
layout (mirrors experiments/chat_tags/batch.py's ar_decode_iq_global_rw_tagged
for tag-handling, and kvmem.train_hmn_chunk.ar_decode_chunk_fb_stitch_kv for
the stitching/chaining logic). Use with an overlapping-window schedule, e.g.
for n_chunks=4, chunk_len=16: windows=[(0,2),(1,3),(2,4)] (32B window, 16B
stride, chunk-aligned) built via chunk_positions_srs_tagged(..., schedule=windows).

Only the VERY FIRST window's warmup is seeded from ground truth (the source's
own first warmup_len bytes) — every later window's warmup comes from the
model's own previously decoded output for that byte range.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from kvmem.train_hmn_chunk import _slot_ids, _cat_kv


@torch.no_grad()
def ar_decode_srs_stitched_tagged(model, chunks_arr, slot_len: int, slot_count: int,
                                  mask_np: np.ndarray, pos_content: dict,
                                  tags: list[tuple[int, int]], device) -> dict:
    if isinstance(chunks_arr, np.ndarray) and chunks_arr.ndim == 2:
        chunks_list = [chunks_arr[k] for k in range(chunks_arr.shape[0])]
    else:
        chunks_list = list(chunks_arr)

    chunk_len = len(chunks_list[0])
    n_chunks  = len(chunks_list)
    src_len   = n_chunks * chunk_len
    wl        = pos_content['warmup_len']
    sids      = np.array(_slot_ids(slot_len, slot_count), dtype=np.int64)
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

    # Global stitched-byte buffer, -1 = not yet decoded. Later windows
    # overwrite earlier ones in the overlap (matches generation order).
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
            # seg_start must equal L_cached exactly (KV-cache invariant) —
            # sweeps in any tag tokens between the last cached position and
            # c0 automatically.
            _decode_segment(L_cached, rb)
        else:  # 'ir'
            tok[rb['sla0']:rb['sla1']] = sids
            src_c0 = rb['argmax_src_c0']
            tok[rb['am0']:rb['am1']] = tok[src_c0:src_c0 + rb['out_len']]
            tok[rb['slb0']:rb['slb1']] = sids
            if wl > 0:
                tok[rb['w0']:rb['w1']] = warmup_src
            _decode_segment(L_cached, rb)

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

    # Teacher-forced BPB: one extra forward pass, each window's LAST block's
    # output set to ground truth, read off per-byte NLL stitched the same way.
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
