"""
kvmem/probe_stitch_content_addressing.py — content-vs-position swap test for
the suffix-recall stitch design (`traj_suffix`/`_grid_stitch`, e.g.
`hmn_stitch_src1024_anchor.py`). `kvmem/probe_positional_shortcut.py`'s own
swap mechanism doesn't apply here: it needs TWO queries sharing a slot
(swap chunk1's warmup into "the slot that normally recalls chunk0"), but a
suffix-recall trajectory only ever has ONE query per packed sequence
(`op_idx=0`, no relay, no second slot to confuse it with).

Adapted mechanism: build the packed sequence at a FIXED structural anchor
(e.g. anchor=44 within a window) — that fixes `w0:w1`/`c0:c1`'s POSITIONS
in the sequence — but fill the warmup bytes at those positions with the
TRUE content from a DIFFERENT anchor (e.g. anchor=0) of the SAME encoded
source, instead of anchor=44's own true content. Compare the greedy-
decoded response against two candidate targets:
  - content-addressed target: the TRUE continuation that actually follows
    the swapped-in bytes at THEIR real source location — high match here
    means the model read the content it was given, not the slot it's in.
  - position-addressed target: the TRUE continuation that would normally
    follow AT THIS anchor's own default position — high match here means
    the model ignored the swapped content and generated whatever this
    structural slot "usually" produces (positional shortcut).

Usage:
    python3 -m kvmem.probe_stitch_content_addressing <ckpt> --device cpu --n-trials 10
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from kvmem.hmn import (HMN_OP_UPDATE, _cyclic_state_ids, build_model,
                       chunk_positions_traj, chunk_mask_fb_traj)


def _load(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    hp = ckpt['hp']
    hp_model = dict(V=hp['V'], d=hp['d'], n_layers=hp['n_layers'], n_heads=hp['n_heads'],
                    block_type=hp.get('block_type', 'single_attn'),
                    rope=hp.get('rope', True), yarn=hp.get('yarn', True),
                    null_kv=hp.get('null_kv', True), rmsnorm=hp.get('rmsnorm', False), chunk_attn=0)
    model = build_model(hp_model, device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model, hp


@torch.no_grad()
def run_trial(model, hp, device, rng, chunk_len, n_chunks, window_chunks,
             warmup_len, anchor_true, anchor_swap):
    state_len = hp.get('state_len', 8)
    state_vocab_size = hp.get('state_vocab_size', 2)
    span_s = n_chunks - window_chunks

    ops = [('E', i) for i in range(n_chunks)]
    ops2 = []
    for i in ops:
        ops2.append(i); ops2.append(('S', None))
    ops2.append(('Q', (span_s, n_chunks, anchor_true)))
    built = chunk_positions_traj(chunk_len, state_len, warmup_len, ops2, n_refine=0,
                                 state_vocab_size=state_vocab_size)
    pos_content, pos_mask = built['pos_content'], built['pos_mask']
    mask_np = chunk_mask_fb_traj(pos_mask, hops=-1)
    L = pos_content['L']
    full_mask = torch.tensor(mask_np, dtype=torch.float32, device=device)

    src = rng.integers(0, 256, size=n_chunks * chunk_len, dtype=np.int64)
    sids = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos_content['enc_blocks']):
        tok[b['s0']:b['s1']] = src[k * chunk_len:(k + 1) * chunk_len]
        tok[b['sl0']] = HMN_OP_UPDATE
        tok[b['sl0'] + 1:b['sl1']] = sids

    rb = pos_content['rec_blocks'][0]
    span_len = window_chunks * chunk_len
    span_off = span_s * chunk_len

    def true_continuation(anchor):
        start = span_off + anchor + warmup_len
        out_len = span_len - anchor - warmup_len
        return src[start:start + out_len], out_len

    # --- baseline: fill warmup with THIS anchor's own true content (sanity check) ---
    tok[rb['w0']:rb['w1']] = src[span_off + anchor_true:span_off + anchor_true + warmup_len]

    def fwd_at(pos):
        t = torch.tensor(tok[:pos], dtype=torch.long, device=device)
        m = full_mask[:pos, :pos]
        return model(t, m)[-1]

    for j in range(rb['out_len']):
        pos = rb['c0'] + j
        tok[pos] = int(fwd_at(pos).argmax())
    baseline_gen = tok[rb['c0']:rb['c1']].copy()
    true_cont_true, _ = true_continuation(anchor_true)
    baseline_match = 100.0 * np.mean(baseline_gen == true_cont_true[:rb['out_len']])

    # --- swap test: fill warmup with a DIFFERENT anchor's true content, same slot ---
    tok2 = tok.copy()
    tok2[rb['w0']:rb['w1']] = src[span_off + anchor_swap:span_off + anchor_swap + warmup_len]

    def fwd_at2(pos):
        t = torch.tensor(tok2[:pos], dtype=torch.long, device=device)
        m = full_mask[:pos, :pos]
        return model(t, m)[-1]

    for j in range(rb['out_len']):
        pos = rb['c0'] + j
        tok2[pos] = int(fwd_at2(pos).argmax())
    swap_gen = tok2[rb['c0']:rb['c1']].copy()

    content_target, content_len = true_continuation(anchor_swap)
    n = min(rb['out_len'], content_len)
    swap_match_content = 100.0 * np.mean(swap_gen[:n] == content_target[:n])
    swap_match_position = 100.0 * np.mean(swap_gen == true_cont_true[:rb['out_len']])

    return dict(baseline_match=baseline_match,
               swap_match_content=swap_match_content,
               swap_match_position=swap_match_position)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt')
    p.add_argument('--device', default='cpu')
    p.add_argument('--n-trials', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--chunk-len', type=int, default=64)
    p.add_argument('--n-chunks', type=int, default=4)
    p.add_argument('--window-chunks', type=int, default=2)
    p.add_argument('--warmup-len', type=int, default=32)
    p.add_argument('--anchor-true', type=int, default=44, help='the structural slot being tested')
    p.add_argument('--anchor-swap', type=int, default=0, help='whose true content gets swapped in')
    args = p.parse_args()

    device = torch.device(args.device)
    model, hp = _load(args.ckpt, device)
    rng = np.random.default_rng(args.seed)

    results = [run_trial(model, hp, device, rng, args.chunk_len, args.n_chunks, args.window_chunks,
                         args.warmup_len, args.anchor_true, args.anchor_swap)
              for _ in range(args.n_trials)]
    b = np.mean([r['baseline_match'] for r in results])
    sc = np.mean([r['swap_match_content'] for r in results])
    sp = np.mean([r['swap_match_position'] for r in results])
    print(f'checkpoint: {args.ckpt}')
    print(f'slot anchor={args.anchor_true}, swapped-in content from anchor={args.anchor_swap}, n_trials={args.n_trials}')
    print(f'baseline (own true content) match:            {b:.1f}%  (sanity check)')
    print(f'swap test vs CONTENT target (anchor={args.anchor_swap}\'s real continuation):  {sc:.1f}%  <- HIGH means content-addressed')
    print(f'swap test vs POSITION target (anchor={args.anchor_true}\'s usual continuation): {sp:.1f}%  <- HIGH means position-addressed')
    if sc > sp + 10:
        print('=> CONTENT-ADDRESSED: model followed the swapped content, not the structural slot.')
    elif sp > sc + 10:
        print('=> POSITION-ADDRESSED: model ignored the swapped content, defaulted to slot behavior.')
    else:
        print('=> INCONCLUSIVE: neither clearly dominates (likely because baseline recall itself is weak).')


if __name__ == '__main__':
    main()
