"""
kvmem/probe_stitch_mechanistic_addressing.py — mechanistic (attention-mass +
gradient-saliency) counterpart to `probe_stitch_content_addressing.py`'s
purely behavioral swap test, adapted the same way `probe_mechanistic_
addressing.py` complements `probe_positional_shortcut.py` for the
two-query `batch`/`interleave_delayed` shapes — see that file's own
docstring for the general method. This version targets the suffix-recall
stitch design (`hmn_stitch_src1024_anchor.py`), which has only ONE query
per packed sequence and `hops=-1` (routing-style: the query's STATE
attends the union of ALL encoding-pass STATEs directly, one per chunk).

Behavioral swap test alone (probe_stitch_content_addressing.py) already
showed content-match >> position-match when a different anchor's true
bytes are substituted into a fixed structural warmup slot. This probe asks
whether that behavioral result is actually CAUSED by attention shifting
onto the swapped-in content's own source chunk's STATE (and gradient
sensitivity concentrating there), rather than some other route.

Swap is made maximally unambiguous here: the warmup slot's structural
"home" is inside the query's own window (chunks [span_s, n_chunks) at
chunk_len=64, e.g. anchor=44 sits inside chunk index 2 for window (2,4)).
The swapped-in content instead comes from chunk index 0 — GLOBALLY
outside the query's own window, encoded by a completely separate STATE
block. If recall is content-addressed, response-row attention/gradient
should concentrate on chunk 0's STATE (the swap source), not chunk 2's
STATE (the structurally "usual" chunk for this anchor).

Usage:
    python3 -m kvmem.probe_stitch_mechanistic_addressing <ckpt> --device cpu \\
        --chunk-len 64 --n-chunks 4 --window-chunks 2 --warmup-len 32 \\
        --anchor-true 44 --swap-chunk 0 --n-trials 5
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from kvmem.hmn import (HMN_OP_UPDATE, MHAttention, _cyclic_state_ids,
                       build_model, chunk_positions_traj, chunk_mask_fb_traj)


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


def _find_attn_modules(model):
    return [m for m in model.modules() if isinstance(m, MHAttention)]


def run_trial(model, hp, device, rng, chunk_len, n_chunks, window_chunks,
             warmup_len, anchor_true, swap_chunk):
    state_len = hp.get('state_len', 8)
    state_vocab_size = hp.get('state_vocab_size', 2)
    span_s = n_chunks - window_chunks
    true_chunk_idx = span_s + anchor_true // chunk_len
    assert swap_chunk != true_chunk_idx, 'swap source must be a genuinely different chunk'

    ops = []
    for i in range(n_chunks):
        ops.append(('E', i)); ops.append(('S', None))
    ops.append(('Q', (span_s, n_chunks, anchor_true)))
    built = chunk_positions_traj(chunk_len, state_len, warmup_len, ops, n_refine=0,
                                 state_vocab_size=state_vocab_size)
    pos_content, pos_mask = built['pos_content'], built['pos_mask']
    mask_np = chunk_mask_fb_traj(pos_mask, hops=-1)
    L = pos_content['L']

    src = rng.integers(0, 256, size=n_chunks * chunk_len, dtype=np.int64)
    sids = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
    tok = np.zeros(L, dtype=np.int64)
    state_ranges = []  # per chunk index
    for k, b in enumerate(pos_content['enc_blocks']):
        tok[b['s0']:b['s1']] = src[k * chunk_len:(k + 1) * chunk_len]
        tok[b['sl0']] = HMN_OP_UPDATE
        tok[b['sl0'] + 1:b['sl1']] = sids
        state_ranges.append((b['sl0'], b['sl1']))

    rb = pos_content['rec_blocks'][0]
    swap_src0 = swap_chunk * chunk_len
    tok[rb['w0']:rb['w1']] = src[swap_src0:swap_src0 + warmup_len]
    # Teacher-force the response with the TRUE continuation of the swapped-in
    # bytes' own source location (the content-addressed hypothesis's target).
    cont0 = swap_src0 + warmup_len
    tok[rb['c0']:rb['c1']] = src[cont0:cont0 + rb['out_len']]

    true_lo, true_hi = state_ranges[true_chunk_idx]
    swap_lo, swap_hi = state_ranges[swap_chunk]

    attn_modules = _find_attn_modules(model)
    for m in attn_modules:
        m.capture_attn = True

    tok_t = torch.tensor(tok, dtype=torch.long, device=device).unsqueeze(0)
    mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device)

    captured = {}
    def pre_hook(module, args):
        x = args[0]
        x.requires_grad_(True)
        x.retain_grad()
        captured['x0'] = x
        return None
    h = model.blocks[0].register_forward_pre_hook(pre_hook)

    logits = model(tok_t, mask_t)
    h.remove()

    c0, c1 = rb['c0'], rb['c1']
    lp = F.log_softmax(logits[0, c0 - 1:c1 - 1], dim=-1)
    tgt = tok_t[0, c0:c1]
    nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()
    model.zero_grad(set_to_none=True)
    nll.backward()

    per_layer_attn = []
    for li, m in enumerate(attn_modules):
        probs = m.last_attn_probs  # (1, H, Lq, Lkv)
        resp_rows = probs[0, :, c0:c1, :]
        mass_true = resp_rows[:, :, true_lo:true_hi].sum(-1).mean().item()
        mass_swap = resp_rows[:, :, swap_lo:swap_hi].sum(-1).mean().item()
        per_layer_attn.append((li, mass_true, mass_swap))
        m.capture_attn = False
        m.last_attn_probs = None

    x0 = captured['x0']
    grad = x0.grad[0]
    val = x0.detach()[0]
    grad_l2 = grad.pow(2).sum(-1).sqrt()
    inputxgrad = (grad * val).sum(-1)

    def region_stat(stat, lo, hi):
        return stat[lo:hi].mean().item()

    saliency = dict(
        grad_l2_true=region_stat(grad_l2, true_lo, true_hi),
        grad_l2_swap=region_stat(grad_l2, swap_lo, swap_hi),
        inputxgrad_true=region_stat(inputxgrad, true_lo, true_hi),
        inputxgrad_swap=region_stat(inputxgrad, swap_lo, swap_hi),
    )

    pred = logits[0, c0 - 1:c1 - 1].argmax(-1)
    behavior_match = 100.0 * (pred == tgt).float().mean().item()

    return dict(per_layer_attn=per_layer_attn, saliency=saliency,
               behavior_match=behavior_match)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt')
    p.add_argument('--device', default='cpu')
    p.add_argument('--n-trials', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--chunk-len', type=int, default=64)
    p.add_argument('--n-chunks', type=int, default=4)
    p.add_argument('--window-chunks', type=int, default=2)
    p.add_argument('--warmup-len', type=int, default=32)
    p.add_argument('--anchor-true', type=int, default=44)
    p.add_argument('--swap-chunk', type=int, default=0,
                   help='chunk index (0-based, GLOBAL) to pull swapped-in warmup content from')
    args = p.parse_args()

    device = torch.device(args.device)
    model, hp = _load(args.ckpt, device)
    rng = np.random.default_rng(args.seed)

    results = [run_trial(model, hp, device, rng, args.chunk_len, args.n_chunks, args.window_chunks,
                         args.warmup_len, args.anchor_true, args.swap_chunk)
              for _ in range(args.n_trials)]

    n_layers = len(results[0]['per_layer_attn'])
    print(f'checkpoint: {args.ckpt}')
    print(f'chunk_len={args.chunk_len} n_chunks={args.n_chunks} window_chunks={args.window_chunks} '
         f'warmup_len={args.warmup_len} anchor_true={args.anchor_true} swap_chunk={args.swap_chunk} '
         f'n_trials={args.n_trials}')
    behavior = np.mean([r['behavior_match'] for r in results])
    print(f'\nteacher-forced behavioral match vs swap-source true continuation (sanity check): {behavior:.1f}%\n')

    print('--- attention mass: response rows -> TRUE-slot STATE (structurally "usual" chunk) '
         'vs SWAP-source STATE (chunk the given bytes actually came from) ---')
    print(f'{"layer":>6} {"mass->TRUE":>12} {"mass->SWAP":>12}  note')
    for li in range(n_layers):
        t = np.mean([r['per_layer_attn'][li][1] for r in results])
        s = np.mean([r['per_layer_attn'][li][2] for r in results])
        note = 'SWAP dominant' if s > t * 1.5 else ('TRUE dominant' if t > s * 1.5 else 'mixed/neither')
        print(f'{li:>6} {t:>11.4f}  {s:>11.4f}  {note}')

    print('\n--- gradient saliency at embedded input (NLL wrt teacher-forced swap-source continuation) ---')
    gt = np.mean([r['saliency']['grad_l2_true'] for r in results])
    gs = np.mean([r['saliency']['grad_l2_swap'] for r in results])
    it = np.mean([r['saliency']['inputxgrad_true'] for r in results])
    is_ = np.mean([r['saliency']['inputxgrad_swap'] for r in results])
    print(f'  grad L2 norm      TRUE={gt:.4f}  SWAP={gs:.4f}')
    print(f'  input x gradient  TRUE={it:+.4f}  SWAP={is_:+.4f}')

    print()
    attn_favors_swap = np.mean([np.mean([r['per_layer_attn'][li][2] for r in results]) -
                                np.mean([r['per_layer_attn'][li][1] for r in results])
                                for li in range(n_layers)]) > 0
    grad_favors_swap = gs > gt
    if attn_favors_swap and grad_favors_swap:
        print('=> MECHANISTICALLY CONFIRMED: both attention and gradient evidence point to the SWAP-source '
             "STATE actually driving the response — content-addressing is causal, not coincidental.")
    elif not attn_favors_swap and not grad_favors_swap:
        print('=> MECHANISTICALLY CONTRADICTS content-addressing: both attention and gradient evidence '
             'point to the structurally TRUE/usual STATE despite any behavioral match — investigate further.')
    else:
        print('=> MIXED: attention and gradient evidence disagree — inconclusive, worth a larger '
             '--n-trials or inspecting per-layer detail above.')


if __name__ == '__main__':
    main()
