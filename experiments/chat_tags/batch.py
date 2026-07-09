"""
Tag-aware batch filler and AR-decode for the iq_global_rw_tagged trajectory.

_chunk_make_batch_fb (kvmem/train_hmn_chunk.py:745) cannot be reused unmodified:
it does `tok[:, rb['sl0']:rb['sl1']] = sids`, a shape-exact broadcast that would
break against tag-widened ranges. So the content fill here is a fresh adaptation
of that same logic against pos_content (content-only, same widths as the
untagged layout) — then one extra vectorized pass writes the constant tag IDs.

_fill_argmax_fb (kvmem/train_hmn_chunk.py:833) only touches am0:am1 (content
range, untouched by tags) — reused unmodified via import.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from kvmem.train_hmn_chunk import _slot_ids, _fill_argmax_fb, _cat_kv  # noqa: F401  (re-exported for train.py)


def make_batch_tagged(rng: np.random.Generator, B: int, n_chunks: int, chunk_len: int,
                      slot_len: int, slot_count: int, pos_content: dict,
                      tags: list[tuple[int, int]]) -> np.ndarray:
    sids = np.array(_slot_ids(slot_len, slot_count), dtype=np.int64)
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
def ar_decode_iq_global_rw_tagged(model, chunks_arr, slot_len: int, slot_count: int,
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
    sids = np.array(_slot_ids(slot_len, slot_count), dtype=np.int64)
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
