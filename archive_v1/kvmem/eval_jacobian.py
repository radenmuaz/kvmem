"""
kvmem/eval_jacobian.py — Jacobian / Lipschitz diagnostics for HMN checkpoints.

Measures the local Lipschitz constant of the MEM hidden-state → output-logit
mapping at each refinement turn.  Activated only with --jacobian flag; default
eval-only mode behaves identically to train_hmn_mono.py.

Usage:
    python -m kvmem.eval_jacobian \\
        --config configs/hmn_mono_p2.py \\
        --eval-only logs/hmn_mono_p2/checkpoints/stage0_end.pt \\
        --device mps \\
        --jacobian

Lipschitz estimate: Monte Carlo lower bound via random unit-vector perturbations
of the hidden state at MEM slot positions.  Uses the model's h_inject API.

  L_lower(k) = max over n_samples of ||Δlogits|| / ||Δh_slot||

Interpretation:
  L < 1  →  contractive at this turn — iteration converges
  L ≈ 1  →  neutral — neither contracting nor expanding
  L > 1  →  expansive — perturbations grow; OOD turns will diverge
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kvmem.model import build_model
from kvmem.data import hmn_ir_positions, hmn_mask_ir
from kvmem.utils import make_test_sequences, cer
from kvmem.train_hmn_mono import (
    _hmn_fill_1tok,
    ar_decode_hmn_ir,
    run_hmn_eval,
    load_config,
)


# ---------------------------------------------------------------------------
# Lipschitz estimator
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_lipschitz_at_turn(
    model, x_S: list[int], slot_len: int, warmup: list[int],
    out_len: int, device, k: int, src_period: int = 1,
    n_samples: int = 16, eps: float = 0.05,
) -> float:
    """Lower-bound estimate of Lipschitz constant of h_slot → logits mapping.

    Perturbs the hidden state at MEM slot positions with n_samples random unit
    vectors (scaled by eps), measures ||Δlogits|| / ||Δh_slot||, returns max.

    Uses model.forward(..., h_inject=..., return_features=True) to:
      1. Get baseline hidden states at slot positions
      2. Re-run with perturbed hidden states
      3. Compare output logit change vs input perturbation magnitude
    """
    src_len = len(x_S)
    wl      = len(warmup)
    pos     = hmn_ir_positions(src_len, slot_len, wl, out_len, k)
    mask_np = hmn_mask_ir(src_len, slot_len, wl, out_len, k)
    mask_t  = torch.tensor(mask_np, dtype=torch.float32, device=device)

    tokens = np.zeros(pos['L'], dtype=np.int64)
    _hmn_fill_1tok(tokens, pos)
    for t, sb in enumerate(pos['src_blocks']):
        if src_period == 1 or t % src_period == 0:
            tokens[sb['s0']:sb['s1']] = x_S
    if wl > 0:
        tokens[pos['w0']:pos['w1']] = warmup
    tok_t = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    # Baseline: get hidden states at last MEM slot block
    # last src block index = k (0-indexed)
    last_mem = pos['mem_blocks'][-1]   # {'ms': ..., 'sl0': ..., 'sl1': ..., 'me': ...}
    sl0 = last_mem['sl0']
    sl1 = last_mem['sl1']
    c0  = pos['c0']
    c1  = pos['c1']

    _, h_base = model(tok_t, mask_t, return_features=True)
    # h_base: [1, seq_len, d]
    h_slot = h_base[:, sl0:sl1, :].detach()   # [1, slot_len, d]

    logits_base = model(tok_t, mask_t, h_inject={(sl0, sl1): h_slot})
    logits_base = logits_base[:, c0-1:c1-1].detach()   # [1, out_len, V]

    ratios = []
    for _ in range(n_samples):
        noise      = torch.randn_like(h_slot)
        noise      = noise / (noise.norm() + 1e-8)           # unit vector
        h_perturb  = h_slot + eps * noise
        logits_p   = model(tok_t, mask_t, h_inject={(sl0, sl1): h_perturb})
        logits_p   = logits_p[:, c0-1:c1-1]
        delta_out  = (logits_p - logits_base).norm().item()
        delta_in   = eps
        ratios.append(delta_out / delta_in)

    return max(ratios)


def run_jacobian_eval(
    model, stage_cfg, device,
    src_period_override=None,
    n_samples: int = 16,
    eps: float = 0.05,
    eval_turns=None,
):
    """Run HMN eval + Lipschitz estimate at each turn. Prints match% and L."""
    seg_len    = stage_cfg.get('src_len',    stage_cfg.get('seg_len', 32))
    slot_len   = stage_cfg.get('slot_len',   4)
    warmup_len = stage_cfg.get('warmup_len', 8)
    out_len    = stage_cfg.get('out_len',    24)
    eval_turns = eval_turns or stage_cfg.get('hmn_eval_turns', [0, 1, 2, 3, 4])
    eval_offset = stage_cfg.get('eval_offset', 0.25)
    _jm        = next((jm for jm in stage_cfg.get('joint_mix', [])
                       if jm.get('traj') == 'hmn_ir'), {})
    src_period = (src_period_override if src_period_override is not None
                  else _jm.get('src_period', 1))

    test_seqs = make_test_sequences(seg_len)
    f_start   = min(int(seg_len * eval_offset), seg_len - warmup_len - out_len)
    y_start   = f_start + warmup_len
    y_end     = min(y_start + out_len, seg_len)

    print(f'  {"k":>4}  {"match%":>8}  {"L_lower":>9}  {"contractive":>12}')
    print(f'  {"----":>4}  {"-------":>8}  {"--------":>9}  {"-----------":>12}')

    for hk in range(max(eval_turns) + 1):
        if hk not in eval_turns:
            continue
        hk_cer = []
        lip_vals = []
        for sname, x_S in test_seqs.items():
            wm  = x_S[max(0, y_start - warmup_len):y_start]
            if len(wm) < warmup_len:
                wm = [x_S[0]] * (warmup_len - len(wm)) + list(wm)
            tgt = x_S[y_start:y_end]
            with torch.no_grad():
                g = ar_decode_hmn_ir(model, x_S, slot_len, wm, len(tgt),
                                     device, k=hk, src_period=src_period)
            hk_cer.append(cer(g, tgt))
            lip = estimate_lipschitz_at_turn(
                model, x_S, slot_len, wm, len(tgt), device,
                k=hk, src_period=src_period,
                n_samples=n_samples, eps=eps,
            )
            lip_vals.append(lip)

        match_pct = round(100 * (1 - sum(hk_cer) / len(hk_cer)), 1)
        lip_mean  = sum(lip_vals) / len(lip_vals)
        contractive = '✓ (<1)' if lip_mean < 1.0 else '✗ (≥1)'
        print(f'  {hk:>4}  {match_pct:>7.1f}%  {lip_mean:>9.4f}  {contractive:>12}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='HMN Jacobian / Lipschitz diagnostics')
    p.add_argument('--config',     required=True)
    p.add_argument('--eval-only',  required=True, metavar='CKPT')
    p.add_argument('--device',     default='cpu')
    p.add_argument('--jacobian',   action='store_true',
                   help='Measure Lipschitz constant at each refinement turn (default: off)')
    p.add_argument('--jac-samples', type=int, default=16,
                   help='Random directions per turn for Lipschitz estimate (default: 16)')
    p.add_argument('--jac-eps',    type=float, default=0.05,
                   help='Perturbation magnitude (default: 0.05)')
    p.add_argument('--bottleneck', action='store_true',
                   help='Also run eval with src_period=∞ (only t=0 sees src)')
    args = p.parse_args()

    device = torch.device(args.device)
    hp     = load_config(args.config)

    ckpt   = torch.load(args.eval_only, map_location=device)
    sd     = ckpt['model']
    V_in   = sd['data_embed.weight'].shape[0] + sd['special_embed.weight'].shape[0]
    d      = sd['data_embed.weight'].shape[1]
    n_lay  = sum(1 for k in sd if k.endswith('.norm1.weight'))
    _cur0  = hp.get('curriculum', [{}])[0]
    ll     = _cur0.get('latent_len', hp.get('latent_len', 0))
    ckpt_hp = {**hp, **ckpt.get('hp', {}),
               'V': V_in, 'd': d, 'n_layers': n_lay,
               'd_ff': sd['blocks.0.ffn.W1.weight'].shape[0],
               'latent_len': ll}
    model = build_model(ckpt_hp, device)
    model.load_state_dict(sd)
    model.eval()

    print(f'Checkpoint: {args.eval_only}  '
          f'(stage={ckpt.get("stage","?")}  step={ckpt.get("step","?")})')

    _jm0  = next((jm for jm in _cur0.get('joint_mix', [])
                  if jm.get('traj') == 'hmn_ir'), {})
    _train_period = _jm0.get('src_period', 1)
    _vn   = hp.get('verbose_eval_n', 4)

    if args.jacobian:
        # Jacobian mode: print match% + Lipschitz per turn
        print(f'\n--- Jacobian eval [src_period={_train_period}, in-distribution] ---')
        run_jacobian_eval(model, _cur0, device,
                          src_period_override=None,
                          n_samples=args.jac_samples, eps=args.jac_eps)
        if args.bottleneck and _train_period != 1:
            print(f'\n--- Jacobian eval [src_period=∞, bottleneck] ---')
            run_jacobian_eval(model, _cur0, device,
                              src_period_override=9999,
                              n_samples=args.jac_samples, eps=args.jac_eps)
    else:
        # Standard eval mode (same as train_hmn_mono --eval-only)
        print(f'\n--- HMN eval [src_period={_train_period}, in-distribution] ---')
        run_hmn_eval(model, _cur0, device, verbose=True, verbose_n=_vn,
                     src_period_override=None)
        if _train_period != 1:
            print(f'\n--- HMN eval [src_period=∞, bottleneck] ---')
            run_hmn_eval(model, _cur0, device, verbose=False,
                         src_period_override=9999)


if __name__ == '__main__':
    main()
