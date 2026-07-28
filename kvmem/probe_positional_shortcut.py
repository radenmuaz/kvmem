"""
kvmem/probe_positional_shortcut.py — tests whether a `batch`/`interleave_delayed`-
style trajectory (E2 Q(0,1) Q(1,2)) is recalling STATE content-addressed (via the
query slot's own warmup bytes) or position-addressed (via which query slot it sits
in, ignoring warmup content) — motivated by traj1 (batch)/traj3 (interleave_delayed)
sharing a byte-identical E2 encode prefix and diverging only in query order, a
plausible source of the persistent weakness both showed in hmn_weave_c64/
hmn_weave_c64_adaptive (batch/interleave stuck near-random while stream converged).

Procedure: build the batch trajectory (query slot 1 normally recalls chunk index 0,
the FARTHER-from-query STATE; query slot 2 normally recalls chunk index 1, the
NEARER STATE). Encode two independent random chunks as usual (STATE0 from chunk0,
STATE1 from chunk1). Then feed query SLOT 1 chunk1's real warmup bytes instead of
chunk0's (content says "recall chunk 1", position says "this is the slot that
normally recalls chunk 0"). Compare the greedy-decoded continuation against BOTH
chunk0's true continuation and chunk1's true continuation:
  - high match vs chunk1, low vs chunk0  => content-addressed (correctly follows
    the swapped warmup, ignores slot position)
  - high match vs chunk0, low vs chunk1  => position-addressed (falls back to
    whatever this slot normally recalls, ignoring the warmup content it was
    actually given)

Usage:
    python3 -m kvmem.probe_positional_shortcut <ckpt> --device cpu --n-trials 5
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from kvmem.hmn import (_cyclic_state_ids, _dual_positions, _scaled_state_positions,
                       build_model, chunk_positions_traj, chunk_mask_fb_traj, traj_batch)


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
def run_trial(model, hp, device, rng):
    chunk_len = 64
    state_len = hp.get('state_len', 8)
    state_vocab_size = hp.get('state_vocab_size', 2)
    warmup_len = hp['warmup_len']
    # Match whatever positional mechanism + mask permission (`hops`) the checkpoint
    # was actually trained under — running it under plain sequential positions (the
    # old default) or the wrong hops window would test a positional scheme the model
    # never saw, producing meaningless results regardless of what's actually learned.
    dual_rope = hp.get('dual_rope', False)
    rope_state_scale = hp.get('rope_state_scale', None)
    hops = hp.get('curriculum', [{}])[0].get('hops', -1)

    ops = traj_batch(2, 1)  # 'E2 Q(0,1) Q(1,2)'
    built = chunk_positions_traj(chunk_len, state_len, warmup_len, ops, n_refine=0,
                                 state_vocab_size=state_vocab_size)
    pos_content, pos_mask, tags = built['pos_content'], built['pos_mask'], built['tags']
    mask_np = chunk_mask_fb_traj(pos_mask, hops=hops)
    L = pos_content['L']
    full_mask = torch.tensor(mask_np, dtype=torch.float32, device=device)

    pos_state_full = pos_local_full = scaled_pos_full = None
    if dual_rope:
        ps, pl = _dual_positions(pos_content, L)
        pos_state_full = torch.tensor(ps, dtype=torch.long, device=device)
        pos_local_full = torch.tensor(pl, dtype=torch.long, device=device)
    if rope_state_scale:
        sp = _scaled_state_positions(pos_content, L, rope_state_scale)
        scaled_pos_full = torch.tensor(sp, dtype=torch.float32, device=device)

    chunk0 = rng.integers(0, 256, size=chunk_len, dtype=np.int64)
    chunk1 = rng.integers(0, 256, size=chunk_len, dtype=np.int64)
    chunks_list = [chunk0, chunk1]

    sids = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos_content['enc_blocks']):
        tok[b['s0']:b['s1']] = chunks_list[k]
        tok[b['sl0']:b['sl1']] = sids
    tag_pos = np.array([p for p, _ in tags], dtype=np.int64)
    tag_ids = np.array([i for _, i in tags], dtype=np.int64)
    tok[tag_pos] = tag_ids

    def fwd_at(pos):
        t = torch.tensor(tok[:pos], dtype=torch.long, device=device)
        m = full_mask[:pos, :pos]
        if dual_rope:
            return model(t, m, pos_state=pos_state_full[:pos], pos_local=pos_local_full[:pos])[-1]
        elif rope_state_scale:
            return model(t, m, offset=scaled_pos_full[:pos])[-1]
        return model(t, m)[-1]

    rec_blocks = [rb for rb in pos_content['rec_blocks'] if rb['type'] != 'noop']
    rb0, rb1 = rec_blocks[0], rec_blocks[1]  # rb0: span(0,1) slot1 recalls chunk0 normally
    assert rb0['span'] == (0, 1) and rb1['span'] == (1, 2)
    wl = warmup_len

    # --- baseline: slot 1 given its OWN correct warmup (chunk0), sanity check ---
    tok[rb0['sl0']:rb0['sl1']] = sids
    tok[rb0['w0']:rb0['w1']] = chunk0[:wl]
    for j in range(rb0['out_len']):
        pos = rb0['c0'] + j
        tok[pos] = int(fwd_at(pos).argmax())
    baseline_gen = tok[rb0['c0']:rb0['c1']].copy()
    baseline_match_vs_chunk0 = 100.0 * np.mean(baseline_gen == chunk0[wl:wl + rb0['out_len']])

    # --- swap test: slot 1 (normally recalls chunk0) given chunk1's warmup instead ---
    tok2 = tok.copy()
    tok2[rb0['sl0']:rb0['sl1']] = sids
    tok2[rb0['w0']:rb0['w1']] = chunk1[:wl]  # SWAPPED — real chunk1 bytes, in the "recall chunk0" slot

    def fwd_at2(pos):
        t = torch.tensor(tok2[:pos], dtype=torch.long, device=device)
        m = full_mask[:pos, :pos]
        if dual_rope:
            return model(t, m, pos_state=pos_state_full[:pos], pos_local=pos_local_full[:pos])[-1]
        elif rope_state_scale:
            return model(t, m, offset=scaled_pos_full[:pos])[-1]
        return model(t, m)[-1]

    for j in range(rb0['out_len']):
        pos = rb0['c0'] + j
        tok2[pos] = int(fwd_at2(pos).argmax())
    swap_gen = tok2[rb0['c0']:rb0['c1']].copy()
    swap_match_vs_chunk1 = 100.0 * np.mean(swap_gen == chunk1[wl:wl + rb0['out_len']])
    swap_match_vs_chunk0 = 100.0 * np.mean(swap_gen == chunk0[wl:wl + rb0['out_len']])

    return dict(baseline_match_vs_chunk0=baseline_match_vs_chunk0,
               swap_match_vs_chunk1=swap_match_vs_chunk1,
               swap_match_vs_chunk0=swap_match_vs_chunk0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt')
    p.add_argument('--device', default='cpu')
    p.add_argument('--n-trials', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    model, hp = _load(args.ckpt, device)
    rng = np.random.default_rng(args.seed)

    results = [run_trial(model, hp, device, rng) for _ in range(args.n_trials)]
    b = np.mean([r['baseline_match_vs_chunk0'] for r in results])
    s1 = np.mean([r['swap_match_vs_chunk1'] for r in results])
    s0 = np.mean([r['swap_match_vs_chunk0'] for r in results])
    print(f'checkpoint: {args.ckpt}')
    print(f'n_trials={args.n_trials}')
    print(f'baseline (own correct warmup) match vs chunk0:      {b:.1f}%  (sanity check — should be nontrivial)')
    print(f'swap test (slot1 given chunk1 warmup) vs chunk1:    {s1:.1f}%  <- HIGH means content-addressed')
    print(f'swap test (slot1 given chunk1 warmup) vs chunk0:    {s0:.1f}%  <- HIGH means position-addressed')
    if s1 > s0 + 10:
        print('=> CONTENT-ADDRESSED: model followed the swapped warmup content, not slot position.')
    elif s0 > s1 + 10:
        print('=> POSITION-ADDRESSED: model ignored the swapped warmup, defaulted to slot-position behavior.')
    else:
        print('=> INCONCLUSIVE: neither clearly dominates.')


if __name__ == '__main__':
    main()
