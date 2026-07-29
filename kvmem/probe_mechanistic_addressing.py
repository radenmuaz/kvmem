"""
kvmem/probe_mechanistic_addressing.py — complements kvmem/probe_positional_
shortcut.py's purely BEHAVIORAL swap test (does the generated continuation
match content vs. slot-position) with a MECHANISTIC check: does the model's
belief in the correct next token actually trace back, inside the network,
to the swapped-in STATE (STATE1, from chunk1) rather than the positionally-
"normal" STATE (STATE0, from chunk0)? A behavioral match alone can't rule
out the model getting the right answer via some other route (e.g. partial
memorization, or an unrelated pathway that happens to correlate with STATE1
in this particular construction) — this probe checks the CAUSAL/internal
story two independent ways:

1. ATTENTION MASS — with the swapped-warmup input, at every layer, how much
   of the response rows' attention lands on STATE0's columns vs STATE1's
   columns (averaged over heads and response positions). If the model is
   genuinely reading from STATE1, attention mass should concentrate there,
   not on STATE0 (the slot's "usual" address).
2. GRADIENT SALIENCY — teacher-forced with chunk1's TRUE continuation (the
   content-addressed hypothesis) as the target, backprop the NLL loss to
   the embedded input (before layer 0) and compare the gradient signal
   (L2 norm, and input x gradient) landing on STATE0's embedded positions
   vs STATE1's. High sensitivity to STATE1 and low to STATE0 means STATE1
   is actually load-bearing for the correct-token belief, not just
   correlated with a correct-looking output.

Requires kvmem/hmn.py's `MHAttention.capture_attn` opt-in diagnostic flag
(added alongside this script) — forces the already-existing manual softmax
attention branch (used for logit_cap/attn_temp) instead of fused SDPA, and
stashes `self.last_attn_probs` (B,H,Lq,Lkv) after the call. Off by default
everywhere (`getattr(self, 'capture_attn', False)`), zero effect on
training/eval/decode unless explicitly enabled here.

Usage:
    python3 -m kvmem.probe_mechanistic_addressing <ckpt> --device cpu \\
        --chunk-len 32 --warmup-len 8 --anchor 0 --n-trials 5
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


def _build_swap_sequence(hp, device, rng, chunk_len, warmup_len, anchor):
    """Same construction as probe_positional_shortcut.run_trial's swap test,
    but returns the raw pieces (tok2, masks, block boundaries, chunk arrays)
    instead of decoding — this script needs the token array BEFORE any
    generation, to fill it with chunk1's TRUE continuation for teacher-forced
    gradient computation."""
    state_len = hp.get('state_len', 8)
    state_vocab_size = hp.get('state_vocab_size', 2)
    hops = hp.get('curriculum', [{}])[0].get('hops', -1)

    ops = [('E', 0), ('S', None), ('E', 1), ('S', None),
           ('Q', (0, 1, anchor)), ('S', None), ('Q', (1, 2, anchor))]
    built = chunk_positions_traj(chunk_len, state_len, warmup_len, ops, n_refine=0,
                                 state_vocab_size=state_vocab_size)
    pos_content, pos_mask = built['pos_content'], built['pos_mask']
    mask_np = chunk_mask_fb_traj(pos_mask, hops=hops)
    L = pos_content['L']

    chunk0 = rng.integers(0, 256, size=chunk_len, dtype=np.int64)
    chunk1 = rng.integers(0, 256, size=chunk_len, dtype=np.int64)
    chunks_list = [chunk0, chunk1]

    sids = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)
    tok = np.zeros(L, dtype=np.int64)
    enc_state_ranges = []  # [(sl0, sl1), ...] per encode block, k=0 -> STATE0, k=1 -> STATE1
    for k, b in enumerate(pos_content['enc_blocks']):
        tok[b['s0']:b['s1']] = chunks_list[k]
        tok[b['sl0']] = HMN_OP_UPDATE
        tok[b['sl0'] + 1:b['sl1']] = sids
        enc_state_ranges.append((b['sl0'], b['sl1']))

    rec_blocks = [rb for rb in pos_content['rec_blocks'] if rb['type'] != 'noop']
    rb0, rb1 = rec_blocks[0], rec_blocks[1]
    assert rb0['span'] == (0, 1) and rb1['span'] == (1, 2)
    wl = warmup_len

    tok[rb0['sl0']] = HMN_OP_UPDATE
    tok[rb0['sl0'] + 1:rb0['sl1']] = sids
    tok[rb0['w0']:rb0['w1']] = chunk1[:wl]  # SWAPPED — chunk1's real warmup, in "slot0"'s position
    # Teacher-force the response with chunk1's TRUE continuation (the content-
    # addressed hypothesis) — this is what lets the gradient probe ask "how
    # sensitive is the model's belief in THIS specific correct answer to each
    # STATE region," independent of whatever the model would actually decode.
    tok[rb0['c0']:rb0['c1']] = chunk1[wl:wl + rb0['out_len']]

    return dict(tok=tok, mask_np=mask_np, L=L, rb0=rb0,
               state0_range=enc_state_ranges[0], state1_range=enc_state_ranges[1],
               chunk0=chunk0, chunk1=chunk1)


def _find_attn_modules(model):
    return [m for m in model.modules() if isinstance(m, MHAttention)]


def run_trial(model, hp, device, rng, chunk_len, warmup_len, anchor):
    built = _build_swap_sequence(hp, device, rng, chunk_len, warmup_len, anchor)
    tok, mask_np, L = built['tok'], built['mask_np'], built['L']
    rb0 = built['rb0']
    s0_lo, s0_hi = built['state0_range']
    s1_lo, s1_hi = built['state1_range']

    attn_modules = _find_attn_modules(model)
    for m in attn_modules:
        m.capture_attn = True

    tok_t = torch.tensor(tok, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)
    mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device)

    captured = {}
    def pre_hook(module, args):
        x = args[0]
        x.requires_grad_(True)
        x.retain_grad()
        captured['x0'] = x
        return None
    h = model.blocks[0].register_forward_pre_hook(pre_hook)

    logits = model(tok_t, mask_t)  # (1, L, V_out)
    h.remove()

    # NLL of the teacher-forced chunk1 continuation — the standard next-token
    # shift: logits[pos-1] predicts token at `pos`.
    c0, c1 = rb0['c0'], rb0['c1']
    lp = F.log_softmax(logits[0, c0 - 1:c1 - 1], dim=-1)
    tgt = tok_t[0, c0:c1]
    nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()
    model.zero_grad(set_to_none=True)
    nll.backward()

    # --- 1. Attention mass, per layer, response rows -> STATE0 vs STATE1 columns ---
    per_layer_attn = []
    for li, m in enumerate(attn_modules):
        probs = m.last_attn_probs  # (1, H, Lq, Lkv) — Lkv includes the +1 null-kv column if null_kv
        resp_rows = probs[0, :, c0:c1, :]  # (H, out_len, Lkv)
        mass_s0 = resp_rows[:, :, s0_lo:s0_hi].sum(-1).mean().item()
        mass_s1 = resp_rows[:, :, s1_lo:s1_hi].sum(-1).mean().item()
        per_layer_attn.append((li, mass_s0, mass_s1))
        m.capture_attn = False
        m.last_attn_probs = None

    # --- 2. Gradient saliency at the embedded input (before layer 0) ---
    x0 = captured['x0']
    grad = x0.grad[0]  # (L, d)
    val = x0.detach()[0]
    grad_l2 = grad.pow(2).sum(-1).sqrt()          # (L,) per-position gradient L2 norm
    inputxgrad = (grad * val).sum(-1)             # (L,) signed input x gradient saliency

    def region_stat(stat, lo, hi):
        return stat[lo:hi].mean().item()

    saliency = dict(
        grad_l2_state0=region_stat(grad_l2, s0_lo, s0_hi),
        grad_l2_state1=region_stat(grad_l2, s1_lo, s1_hi),
        inputxgrad_state0=region_stat(inputxgrad, s0_lo, s0_hi),
        inputxgrad_state1=region_stat(inputxgrad, s1_lo, s1_hi),
    )

    # Behavioral cross-check (argmax of the teacher-forced logits at each response
    # position — NOT a generation loop, just "would greedy have picked the right
    # byte here given everything up to it," consistent with the gradient's target).
    pred = logits[0, c0 - 1:c1 - 1].argmax(-1)
    behavior_match_vs_chunk1 = 100.0 * (pred == tgt).float().mean().item()

    return dict(per_layer_attn=per_layer_attn, saliency=saliency,
               behavior_match_vs_chunk1=behavior_match_vs_chunk1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpt')
    p.add_argument('--device', default='cpu')
    p.add_argument('--n-trials', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--chunk-len', type=int, default=32)
    p.add_argument('--warmup-len', type=int, required=True,
                   help='must be a warmup_len actually trained at --chunk-len — see '
                        'probe_positional_shortcut.py\'s own --warmup-len note')
    p.add_argument('--anchor', type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    model, hp = _load(args.ckpt, device)
    rng = np.random.default_rng(args.seed)

    results = [run_trial(model, hp, device, rng, args.chunk_len, args.warmup_len, args.anchor)
              for _ in range(args.n_trials)]

    n_layers = len(results[0]['per_layer_attn'])
    print(f'checkpoint: {args.ckpt}')
    print(f'chunk_len={args.chunk_len} warmup_len={args.warmup_len} anchor={args.anchor} n_trials={args.n_trials}')
    behavior = np.mean([r['behavior_match_vs_chunk1'] for r in results])
    print(f'\nteacher-forced behavioral match vs chunk1 (sanity — is the target even plausible): {behavior:.1f}%\n')

    print('--- attention mass: response rows -> STATE0 (chunk0, "usual" slot) vs STATE1 (chunk1, swapped-in) ---')
    print(f'{"layer":>6} {"mass->STATE0":>14} {"mass->STATE1":>14}  note')
    for li in range(n_layers):
        s0 = np.mean([r['per_layer_attn'][li][1] for r in results])
        s1 = np.mean([r['per_layer_attn'][li][2] for r in results])
        note = 'STATE1 dominant' if s1 > s0 * 1.5 else ('STATE0 dominant' if s0 > s1 * 1.5 else 'mixed/neither')
        print(f'{li:>6} {s0:>13.4f}  {s1:>13.4f}  {note}')

    print('\n--- gradient saliency at embedded input (NLL wrt teacher-forced chunk1 continuation) ---')
    gl0 = np.mean([r['saliency']['grad_l2_state0'] for r in results])
    gl1 = np.mean([r['saliency']['grad_l2_state1'] for r in results])
    ig0 = np.mean([r['saliency']['inputxgrad_state0'] for r in results])
    ig1 = np.mean([r['saliency']['inputxgrad_state1'] for r in results])
    print(f'  grad L2 norm      STATE0={gl0:.4f}  STATE1={gl1:.4f}')
    print(f'  input x gradient  STATE0={ig0:+.4f}  STATE1={ig1:+.4f}')

    print()
    attn_favors_1 = np.mean([np.mean([r['per_layer_attn'][li][2] for r in results]) -
                            np.mean([r['per_layer_attn'][li][1] for r in results])
                            for li in range(n_layers)]) > 0
    grad_favors_1 = gl1 > gl0
    if attn_favors_1 and grad_favors_1:
        print('=> MECHANISTICALLY CONFIRMED: both attention and gradient evidence point to STATE1 '
              '(the swapped-in content) actually driving the response — not just a coincidental match.')
    elif not attn_favors_1 and not grad_favors_1:
        print('=> MECHANISTICALLY CONTRADICTS content-addressing: both attention and gradient evidence '
              'point to STATE0 (the "usual" slot) despite any behavioral match — investigate further, '
              'a correct-looking output may not mean what it appears to.')
    else:
        print('=> MIXED: attention and gradient evidence disagree on which STATE dominates — '
              'inconclusive, worth a larger --n-trials or inspecting per-layer detail above.')


if __name__ == '__main__':
    main()
