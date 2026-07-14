"""
Slot-position extraction and a simple (non-KV-cached, full-recompute) AR decode
for DenseSlotKVModel.

No incremental KV caching for this first prototype — the growing cross-layer SLOT
history has no obvious efficient incremental scheme yet (unlike kvmem's existing
segment-by-segment cache, which assumes single-layer per-position KV). Each
generation step does a full forward pass over the known-so-far tokens; correctness
over efficiency for this ablation. Val sets here are tiny (a few sequences, 24-byte
outputs) so this is not a practical bottleneck.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from experiments.chat_tags.batch import _slot_ids


def slot_positions_from_pos(pos_content: dict) -> np.ndarray:
    """Union of every SLOT/SLOT_A/SLOT_B content range across enc_blocks + rec_blocks."""
    positions: set[int] = set()
    for b in pos_content['enc_blocks']:
        positions.update(range(b['sl0'], b['sl1']))
    for rb in pos_content['rec_blocks']:
        if rb['type'] == 'iq':
            positions.update(range(rb['sl0'], rb['sl1']))
        else:
            positions.update(range(rb['sla0'], rb['sla1']))
            positions.update(range(rb['slb0'], rb['slb1']))
    return np.array(sorted(positions), dtype=np.int64)


@torch.no_grad()
def ar_decode_densekv(model, chunks_arr, slot_len: int, slot_count: int,
                      mask_t: torch.Tensor, pos_content: dict,
                      tags: list[tuple[int, int]], slot_positions_t: torch.Tensor,
                      device, warmup_offset: int = 0) -> dict:
    if isinstance(chunks_arr, np.ndarray) and chunks_arr.ndim == 2:
        chunks_list = [chunks_arr[k] for k in range(chunks_arr.shape[0])]
    else:
        chunks_list = list(chunks_arr)

    wl   = pos_content['warmup_len']
    sids = np.array(_slot_ids(slot_len, slot_count), dtype=np.int64)
    L    = pos_content['L']

    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos_content['enc_blocks']):
        tok[b['s0']:b['s1']]   = chunks_list[k]
        tok[b['sl0']:b['sl1']] = sids
    for p, tid in tags:
        tok[p] = tid

    turn_match_pcts: list[float] = []
    for rb in pos_content['rec_blocks']:
        span_s, span_e = rb['span']
        gt_span = np.concatenate(chunks_list[span_s:span_e])

        if rb['type'] == 'iq':
            tok[rb['sl0']:rb['sl1']] = sids
            if wl > 0:
                tok[rb['w0']:rb['w1']] = gt_span[warmup_offset:warmup_offset + wl].astype(np.int64)
        else:
            tok[rb['sla0']:rb['sla1']] = sids
            src_c0 = rb['argmax_src_c0']
            tok[rb['am0']:rb['am1']] = tok[src_c0:src_c0 + rb['out_len']]
            tok[rb['slb0']:rb['slb1']] = sids
            if wl > 0:
                tok[rb['w0']:rb['w1']] = gt_span[warmup_offset:warmup_offset + wl].astype(np.int64)

        for j in range(rb['out_len']):
            tok_t = torch.tensor(tok, dtype=torch.long, device=device)
            logits = model(tok_t, mask_t, slot_positions_t)
            pos = rb['c0'] + j
            tok[pos] = int(logits[pos - 1].argmax())

        rw_target = gt_span[warmup_offset + wl: warmup_offset + wl + rb['out_len']]
        rb_gen = tok[rb['c0']:rb['c1']]
        rb_match = 100.0 * float(np.sum(rb_gen[:len(rw_target)] == rw_target)) / max(len(rw_target), 1)
        turn_match_pcts.append(rb_match)

    final_rb = pos_content['rec_blocks'][-1]
    span_s, span_e = final_rb['span']
    gt_final = np.concatenate(chunks_list[span_s:span_e])
    out_len  = final_rb['out_len']
    final_gen = tok[final_rb['c0']:final_rb['c1']]
    target = gt_final[warmup_offset + wl: warmup_offset + wl + out_len]
    eff_out = len(target)

    match_pct = 100.0 * float(np.sum(final_gen[:eff_out] == target)) / max(eff_out, 1)

    tok_tf = torch.tensor(tok, dtype=torch.long, device=device)
    if eff_out > 0:
        tok_tf[final_rb['c0']:final_rb['c0'] + eff_out] = torch.tensor(
            target.astype(np.int64), dtype=torch.long, device=device)
    logits_tf = model(tok_tf, mask_t, slot_positions_t)
    tgt_tensor = torch.tensor(target.astype(np.int64), dtype=torch.long, device=device)
    lp = F.log_softmax(logits_tf[final_rb['c0']-1:final_rb['c0']-1+eff_out], dim=-1)
    nll_vals = -lp.gather(1, tgt_tensor.unsqueeze(1)).squeeze(1).cpu().numpy()
    nll = float(nll_vals.mean()) if len(nll_vals) > 0 else float('nan')
    bpb = nll / math.log(2)

    return dict(bpb=bpb, nll=nll, match_pct=match_pct,
               decoded_bytes=final_gen.tolist(), turn_match_pcts=turn_match_pcts)
