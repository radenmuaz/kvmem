"""
attn_viz.py — attention map heatmaps for KVMemModel checkpoints.

Captures attention weights by temporarily replacing F.scaled_dot_product_attention
with a version that saves the softmax weights before returning. No model or train
files are modified.

Usage:
  python3 attn_viz.py --ckpt logs/hmn_chunk_global_iq_rw_nc4_slot8_wina_s0/checkpoints/stage0_best.pt
  python3 attn_viz.py --ckpt logs/hmn_chunk_local_64_v5/checkpoints/stage0_end.pt
  python3 attn_viz.py --ckpt <path> --seq-idx 0 --warmup-x 0 --out attn_maps.png

Output:
  <ckpt_dir>/attn_<step>_x<warmup_x>_seq<idx>.png
  One figure per sequence: (n_layers) rows, (n_heads+1) columns.
  Last column = head-averaged map. Segment boundaries annotated as colored bars.
"""

import argparse
import math
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from kvmem.model import build_model
from kvmem.train_hmn_chunk import (
    chunk_positions_iq_global_rw,
    chunk_positions_fb_localrefine,
    chunk_mask_fb,
    _slot_ids,
)
from kvmem.utils import make_test_sequences


# ---------------------------------------------------------------------------
# Attention capture — patch F.scaled_dot_product_attention in a context
# ---------------------------------------------------------------------------

class AttentionCapture:
    """
    Context manager that replaces torch.nn.functional.scaled_dot_product_attention
    with a version that also stores attention weight tensors.

    Usage:
        cap = AttentionCapture()
        with cap:
            logits = model(tokens, mask)
        # cap.weights: list[(B, H, L_q, L_kv)] — one entry per layer per call

    null_kv=True appends a null K/V pair, making L_kv = L+1. We trim the last
    column so all maps are (L_q, L) — the null token's weight goes to waste but
    is not semantically meaningful for visualisation.
    """

    def __init__(self):
        self.weights: list[torch.Tensor] = []
        self._orig = None

    def __enter__(self):
        self.weights.clear()
        capture = self

        def _patched(Q, K, V, attn_mask=None, dropout_p=0.0,
                     is_causal=False, scale=None, **kwargs):
            dh = Q.shape[-1]
            s = (scale if scale is not None else 1.0 / math.sqrt(dh))
            scores = torch.matmul(Q, K.transpose(-2, -1)) * s
            if attn_mask is not None:
                scores = scores + attn_mask
            w = torch.softmax(scores.float(), dim=-1).to(Q.dtype)
            capture.weights.append(w.detach().cpu())
            return w @ V

        self._orig = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = _patched
        return self

    def __exit__(self, *_):
        F.scaled_dot_product_attention = self._orig


# ---------------------------------------------------------------------------
# Segment annotation helpers
# ---------------------------------------------------------------------------

# Colour palette for each segment type
_SEG_COLORS = {
    'src':   '#4CAF50',   # green — source bytes in enc_blocks
    'enc_sl':'#2196F3',   # blue  — enc_block SLOT tokens
    'iq_sl': '#FF9800',   # orange — IQ SLOT
    'iq_wm': '#FFEB3B',   # yellow — IQ warmup
    'iq_out':'#F44336',   # red   — IQ output
    'ira':   '#9C27B0',   # purple — IR SLOT_A
    'am':    '#E91E63',   # pink  — IR argmax
    'irb':   '#673AB7',   # indigo — IR SLOT_B
    'ir_wm': '#FFEB3B',   # yellow — IR warmup (same as IQ warmup)
    'ir_out':'#F44336',   # red   — IR output (same as IQ output)
}


def _build_segments(pos: dict) -> list[tuple[int, int, str, str]]:
    """Return [(start, end, label, color), ...] for every position segment."""
    segs = []
    for i, b in enumerate(pos['enc_blocks']):
        segs.append((b['s0'],  b['s1'],  f'src{i}',   _SEG_COLORS['src']))
        segs.append((b['sl0'], b['sl1'], f'esl{i}',   _SEG_COLORS['enc_sl']))

    iq_idx = ir_idx = 0
    for rb in pos['rec_blocks']:
        if rb['type'] == 'iq':
            segs.append((rb['sl0'], rb['sl1'], f'IQ{iq_idx}_sl', _SEG_COLORS['iq_sl']))
            if rb['w0'] < rb['w1']:
                segs.append((rb['w0'],  rb['w1'],  f'IQ{iq_idx}_wm', _SEG_COLORS['iq_wm']))
            segs.append((rb['c0'],  rb['c1'],  f'IQ{iq_idx}_out', _SEG_COLORS['iq_out']))
            iq_idx += 1
        else:
            segs.append((rb['sla0'], rb['sla1'], f'IR{ir_idx}_A',   _SEG_COLORS['ira']))
            segs.append((rb['am0'],  rb['am1'],  f'IR{ir_idx}_am',  _SEG_COLORS['am']))
            segs.append((rb['slb0'], rb['slb1'], f'IR{ir_idx}_B',   _SEG_COLORS['irb']))
            if rb['w0'] < rb['w1']:
                segs.append((rb['w0'],  rb['w1'],  f'IR{ir_idx}_wm', _SEG_COLORS['ir_wm']))
            segs.append((rb['c0'],  rb['c1'],  f'IR{ir_idx}_out', _SEG_COLORS['ir_out']))
            ir_idx += 1
    return segs


def _draw_seg_bar(ax, segs: list, L: int, axis: str, bar_frac: float = 0.03):
    """Draw a thin colored bar along `axis` ('x' or 'y') marking segment boundaries."""
    lim = ax.get_xlim() if axis == 'x' else ax.get_ylim()
    span = lim[1] - lim[0]
    bar_w = span * bar_frac
    for (s, e, lbl, col) in segs:
        if axis == 'x':
            rect = mpatches.Rectangle(
                (s - 0.5, lim[1] - bar_w), e - s, bar_w,
                linewidth=0, facecolor=col, alpha=0.85, clip_on=False,
                transform=ax.transData)
        else:  # y axis — top = position 0 in image coords
            rect = mpatches.Rectangle(
                (lim[0], s - 0.5), bar_w, e - s,
                linewidth=0, facecolor=col, alpha=0.85, clip_on=False,
                transform=ax.transData)
        ax.add_patch(rect)


def _draw_block_lines(ax, segs: list):
    """Draw thin gray vertical + horizontal lines at every segment boundary."""
    boundaries = sorted({s for (s, e, *_) in segs} | {e for (s, e, *_) in segs})
    for b in boundaries:
        ax.axvline(b - 0.5, color='gray', lw=0.3, alpha=0.4)
        ax.axhline(b - 0.5, color='gray', lw=0.3, alpha=0.4)


def _legend_patches():
    return [mpatches.Patch(color=v, label=k) for k, v in _SEG_COLORS.items()]


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------

def visualise(ckpt_path: str, seq_idx: int = 0, warmup_x: int = 0,
              device_str: str = 'cpu', out_path: str | None = None,
              avg_heads: bool = False):
    device = torch.device(device_str)
    ckpt = torch.load(ckpt_path, map_location=device)
    hp   = ckpt['hp']
    hp_model = ckpt.get('hp_model')

    if hp_model is None:
        # Reconstruct from hp
        hp_model = dict(
            V=hp.get('V', 268), d=hp['d'], n_layers=hp['n_layers'],
            n_heads=hp['n_heads'], d_ff=hp['d_ff'],
            rope=hp.get('rope', True), yarn=hp.get('yarn', True),
            null_kv=hp.get('null_kv', True), compile=False,
            chunk_attn=hp.get('chunk_attn', 0),
        )

    # Disable chunk_attn for viz — we need full SDPA calls (not chunked) to
    # capture one weight tensor per layer instead of multiple micro-chunks.
    hp_model_viz = dict(hp_model)
    hp_model_viz['chunk_attn'] = 0

    model = build_model(hp_model_viz, device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    n_layers = hp['n_layers']
    n_heads  = hp['n_heads']
    slot_len  = hp.get('slot_len', 4)
    slot_count = hp.get('slot_count', 2)
    warmup_len = hp.get('warmup_len', 8)
    stage = ckpt.get('stage', 0)
    step  = ckpt.get('step', 0)

    # ── Build trajectory (auto-detect from traj_mix) ─────────────────────────
    curriculum = hp.get('curriculum', [])
    assert curriculum, 'hp has no curriculum'
    stage_cfg = curriculum[0]
    n_chunks  = stage_cfg['n_chunks']
    chunk_len = stage_cfg['chunk_len']

    traj_mix  = stage_cfg.get('traj_mix', [])
    traj_type = traj_mix[0]['type'] if traj_mix else 'iq_global_rw'

    if traj_type in ('iq_global_rw', 'iq_global_rw_ir'):
        window_chunks = traj_mix[0].get('window_chunks', 2)
        n_refine = traj_mix[0].get('n_refine', 0)
        pos = chunk_positions_iq_global_rw(
            n_chunks, chunk_len, slot_len, warmup_len, window_chunks,
            warmup_x_fixed=warmup_x, n_refine=n_refine)
    elif traj_type == 'ir_local':
        windows  = traj_mix[0]['windows']
        n_refine = traj_mix[0].get('n_refine', 2)
        pos = chunk_positions_fb_localrefine(
            n_chunks, chunk_len, slot_len, warmup_len, windows, n_refine=n_refine)
        warmup_x = 0
    else:
        print(f'[attn_viz] unsupported traj_type={traj_type!r}, falling back to iq_global_rw')
        pos = chunk_positions_iq_global_rw(
            n_chunks, chunk_len, slot_len, warmup_len, 2,
            warmup_x_fixed=warmup_x, n_refine=0)

    mask_np = chunk_mask_fb(pos)
    L = pos['L']
    mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device)
    sids   = np.array(_slot_ids(slot_len, slot_count), dtype=np.int64)

    # ── Pick test sequence ────────────────────────────────────────────────────
    val_seqs = make_test_sequences(n_chunks * chunk_len)
    seq_name = list(val_seqs.keys())[seq_idx % len(val_seqs)]
    seq_bytes = list(val_seqs.values())[seq_idx % len(val_seqs)]
    chunks_list = [seq_bytes[k * chunk_len:(k + 1) * chunk_len] for k in range(n_chunks)]

    # Build token sequence
    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos['enc_blocks']):
        tok[b['s0']:b['s1']]   = chunks_list[k]
        tok[b['sl0']:b['sl1']] = sids

    wl = warmup_len
    for rb in pos['rec_blocks']:
        span_s, span_e = rb['span']
        gt_span = np.concatenate([np.array(chunks_list[i]) for i in range(span_s, span_e)])
        if rb['type'] == 'iq':
            tok[rb['sl0']:rb['sl1']] = sids
            X = warmup_x if rb.get('warmup_train_range') else 0
            if wl > 0:
                tok[rb['w0']:rb['w1']] = np.array(gt_span[X:X + wl], dtype=np.int64)
            tok[rb['c0']:rb['c1']] = np.array(gt_span[X + wl:X + wl + rb['out_len']], dtype=np.int64)
        else:
            tok[rb['sla0']:rb['sla1']] = sids
            src_c0 = rb['argmax_src_c0']
            tok[rb['am0']:rb['am1']] = tok[src_c0:src_c0 + rb['out_len']]
            tok[rb['slb0']:rb['slb1']] = sids
            X = warmup_x if rb.get('warmup_train_range') else 0
            if wl > 0:
                tok[rb['w0']:rb['w1']] = np.array(gt_span[X:X + wl], dtype=np.int64)
            tok[rb['c0']:rb['c1']] = np.array(gt_span[X + wl:X + wl + rb['out_len']], dtype=np.int64)

    tok_t = torch.tensor(tok, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)

    # ── Forward pass with attention capture ──────────────────────────────────
    cap = AttentionCapture()
    with cap, torch.no_grad():
        # chunk_attn disabled above, so one SDPA call per layer
        _ = model(tok_t, mask_t)

    n_calls = len(cap.weights)
    if n_calls != n_layers:
        print(f'[attn_viz] WARNING: expected {n_layers} attention calls, got {n_calls}. '
              f'(null_kv or chunk_attn may have caused extra calls.)')

    # Use the first n_layers captures; if null_kv=True the last column is null token
    attn_maps = []
    for w in cap.weights[:n_layers]:
        # w: (B, H, L_q, L_kv) — L_kv may be L+1 if null_kv
        w = w[0]           # (H, L_q, L_kv)
        w = w[:, :, :L]    # trim null token column if present
        attn_maps.append(w.numpy())  # (H, L, L)

    # ── Build segments for annotation ────────────────────────────────────────
    segs = _build_segments(pos)

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_cols = (1 if avg_heads else n_heads) + 1   # +1 for head-avg column
    n_rows = n_layers
    fig_w  = max(12, n_cols * 2.5)
    fig_h  = max(8,  n_rows * 2.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)
    fig.suptitle(
        f'{Path(ckpt_path).parent.parent.name}  step={step}  '
        f'seq={seq_name}  x={warmup_x}  L={L}',
        fontsize=10)

    vmax_global = max(m.mean(0).max() for m in attn_maps)  # head-avg max for shared scale

    for layer_i, W in enumerate(attn_maps):
        # W: (H, L, L)
        W_avg = W.mean(0)  # (L, L)

        col_range = range(n_heads) if not avg_heads else []
        for head_j in col_range:
            ax = axes[layer_i][head_j]
            im = ax.imshow(W[head_j], aspect='auto', origin='upper',
                           cmap='hot', vmin=0, vmax=W[head_j].max())
            _draw_block_lines(ax, segs)
            _draw_seg_bar(ax, segs, L, 'x')
            _draw_seg_bar(ax, segs, L, 'y')
            ax.set_title(f'L{layer_i} H{head_j}', fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])

        # Head-averaged column
        ax_avg = axes[layer_i][-1]
        im_avg = ax_avg.imshow(W_avg, aspect='auto', origin='upper',
                               cmap='hot', vmin=0, vmax=vmax_global)
        _draw_block_lines(ax_avg, segs)
        _draw_seg_bar(ax_avg, segs, L, 'x')
        _draw_seg_bar(ax_avg, segs, L, 'y')
        ax_avg.set_title(f'L{layer_i} avg', fontsize=7)
        ax_avg.set_xticks([]); ax_avg.set_yticks([])

    # Legend
    fig.legend(handles=_legend_patches(), loc='lower right',
               fontsize=7, ncol=2, framealpha=0.8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if out_path is None:
        ckpt_dir  = Path(ckpt_path).parent
        ckpt_stem = Path(ckpt_path).stem
        out_path  = str(ckpt_dir / f'attn_{ckpt_stem}_x{warmup_x}_seq{seq_idx}.png')

    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[attn_viz] saved → {out_path}')
    return out_path


# ---------------------------------------------------------------------------
# Expected-pattern analysis — text report of where each critical row attends
# ---------------------------------------------------------------------------

def analyse_patterns(ckpt_path: str, warmup_x: int = 0,
                     device_str: str = 'cpu'):
    """
    Text analysis: for each rec_block, report which source region each
    row-type (SLOT, warmup, output) directs the majority of its attention.
    Flags violations of the expected pattern.
    """
    device = torch.device(device_str)
    ckpt = torch.load(ckpt_path, map_location=device)
    hp   = ckpt['hp']
    hp_model = ckpt.get('hp_model') or dict(
        V=hp.get('V', 268), d=hp['d'], n_layers=hp['n_layers'],
        n_heads=hp['n_heads'], d_ff=hp['d_ff'],
        rope=hp.get('rope', True), yarn=hp.get('yarn', True),
        null_kv=hp.get('null_kv', True), compile=False, chunk_attn=0,
    )
    hp_model_viz = dict(hp_model)
    hp_model_viz['chunk_attn'] = 0

    model = build_model(hp_model_viz, device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    slot_len   = hp.get('slot_len', 4)
    slot_count = hp.get('slot_count', 2)
    warmup_len = hp.get('warmup_len', 8)
    stage_cfg  = hp['curriculum'][0]
    n_chunks   = stage_cfg['n_chunks']
    chunk_len  = stage_cfg['chunk_len']
    traj_mix   = stage_cfg.get('traj_mix', [])
    traj_type  = traj_mix[0]['type'] if traj_mix else 'iq_global_rw'

    if traj_type in ('iq_global_rw', 'iq_global_rw_ir'):
        wc = traj_mix[0].get('window_chunks', 2)
        nr = traj_mix[0].get('n_refine', 0)
        pos = chunk_positions_iq_global_rw(
            n_chunks, chunk_len, slot_len, warmup_len, wc,
            warmup_x_fixed=warmup_x, n_refine=nr)
    else:
        windows = traj_mix[0].get('windows', [(0, 2)])
        nr = traj_mix[0].get('n_refine', 2)
        pos = chunk_positions_fb_localrefine(
            n_chunks, chunk_len, slot_len, warmup_len, windows, n_refine=nr)
        warmup_x = 0

    mask_np = chunk_mask_fb(pos)
    L  = pos['L']
    mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device)
    sids   = np.array(_slot_ids(slot_len, slot_count), dtype=np.int64)

    # Use all val sequences, average attention across them
    val_seqs   = make_test_sequences(n_chunks * chunk_len)
    all_weights: list[list[np.ndarray]] = [[] for _ in range(hp['n_layers'])]

    for seq_bytes in list(val_seqs.values()):
        chunks_list = [seq_bytes[k * chunk_len:(k + 1) * chunk_len] for k in range(n_chunks)]
        tok = np.zeros(L, dtype=np.int64)
        for k, b in enumerate(pos['enc_blocks']):
            tok[b['s0']:b['s1']]   = chunks_list[k]
            tok[b['sl0']:b['sl1']] = sids
        wl = warmup_len
        for rb in pos['rec_blocks']:
            span_s, span_e = rb['span']
            gt = np.concatenate([np.array(chunks_list[i]) for i in range(span_s, span_e)])
            X  = warmup_x if rb.get('warmup_train_range') else 0
            if rb['type'] == 'iq':
                tok[rb['sl0']:rb['sl1']] = sids
                if wl > 0: tok[rb['w0']:rb['w1']] = np.array(gt[X:X+wl], dtype=np.int64)
                tok[rb['c0']:rb['c1']] = np.array(gt[X+wl:X+wl+rb['out_len']], dtype=np.int64)
            else:
                tok[rb['sla0']:rb['sla1']] = sids
                tok[rb['am0']:rb['am1']]   = tok[rb['argmax_src_c0']:rb['argmax_src_c0']+rb['out_len']]
                tok[rb['slb0']:rb['slb1']] = sids
                if wl > 0: tok[rb['w0']:rb['w1']] = np.array(gt[X:X+wl], dtype=np.int64)
                tok[rb['c0']:rb['c1']] = np.array(gt[X+wl:X+wl+rb['out_len']], dtype=np.int64)

        tok_t = torch.tensor(tok, dtype=torch.long, device=device).unsqueeze(0)
        cap = AttentionCapture()
        with cap, torch.no_grad():
            _ = model(tok_t, mask_t)
        for li, w in enumerate(cap.weights[:hp['n_layers']]):
            all_weights[li].append(w[0].mean(0).numpy())  # head-avg (L, L_kv)

    # Average across sequences per layer
    avg_maps = [np.stack(ws).mean(0)[:, :L] for ws in all_weights]  # list of (L, L)

    # ── Define named regions ──────────────────────────────────────────────────
    regions: dict[str, np.ndarray] = {}
    def _mark(name, lo, hi):
        m = np.zeros(L, dtype=bool)
        m[lo:hi] = True
        regions[name] = m

    for k, b in enumerate(pos['enc_blocks']):
        _mark(f'enc_src{k}',  b['s0'],  b['s1'])
        _mark(f'enc_slot{k}', b['sl0'], b['sl1'])

    enc_src_mask  = np.zeros(L, dtype=bool)
    enc_slot_mask = np.zeros(L, dtype=bool)
    for b in pos['enc_blocks']:
        enc_src_mask[b['s0']:b['s1']]   = True
        enc_slot_mask[b['sl0']:b['sl1']] = True

    # ── Text report ───────────────────────────────────────────────────────────
    print(f'\n{"="*80}')
    print(f'Attention pattern analysis: {Path(ckpt_path).name}  step={ckpt.get("step",0)}')
    print(f'  n_chunks={n_chunks}  chunk_len={chunk_len}  slot_len={slot_len}  '
          f'warmup_x={warmup_x}  L={L}')
    print(f'  Averaged over {len(val_seqs)} val sequences, {hp["n_layers"]} layers, head-avg')
    print(f'{"="*80}')

    # Build the full set of named column masks for detailed breakdown
    def _named_col_masks(rb: dict) -> list[tuple[str, np.ndarray, str]]:
        """Return [(name, mask, expected)] for all meaningful column regions."""
        named = []
        named.append(('enc_src',   enc_src_mask,  'low'))   # should be blocked
        named.append(('enc_slot',  enc_slot_mask, ''))       # compressed memory
        if rb['type'] == 'iq':
            iq_sl = np.zeros(L, dtype=bool); iq_sl[rb['sl0']:rb['sl1']] = True
            iq_wm = np.zeros(L, dtype=bool); iq_wm[rb['w0']:rb['w1']]   = True
            iq_out= np.zeros(L, dtype=bool); iq_out[rb['c0']:rb['c1']]  = True
            named += [('IQ_slot', iq_sl, ''), ('IQ_warm', iq_wm, ''), ('IQ_out', iq_out, '')]
        else:
            sla = np.zeros(L, dtype=bool); sla[rb['sla0']:rb['sla1']] = True
            am  = np.zeros(L, dtype=bool); am[rb['am0']:rb['am1']]    = True
            slb = np.zeros(L, dtype=bool); slb[rb['slb0']:rb['slb1']] = True
            ir_wm = np.zeros(L, dtype=bool); ir_wm[rb['w0']:rb['w1']] = True
            ir_out= np.zeros(L, dtype=bool); ir_out[rb['c0']:rb['c1']]= True
            named += [('IR_slA', sla,''), ('IR_am', am,''), ('IR_slB', slb,''),
                      ('IR_warm', ir_wm,''), ('IR_out', ir_out,'')]
        return named

    for rb_i, rb in enumerate(pos['rec_blocks']):
        span_s, span_e = rb['span']
        print(f'\n  rec_block[{rb_i}] type={rb["type"]}  span=({span_s},{span_e})')
        named_cols = _named_col_masks(rb)

        def _report_rows(row_lo, row_hi, row_name, expect_low=(), expect_high=()):
            if row_lo >= row_hi:
                return
            # Accumulate per-region attention mass across layers
            region_pcts: dict[str, list[float]] = {n: [] for n, _, _ in named_cols}
            for W in avg_maps:
                row_mean = W[row_lo:row_hi, :].mean(0)
                row_mean = row_mean / (row_mean.sum() + 1e-9)
                for name, col_mask, _ in named_cols:
                    region_pcts[name].append(float(row_mean[col_mask].sum()))

            parts = []
            flags = []
            for name, _, _ in named_cols:
                pct = 100 * np.mean(region_pcts[name])
                if pct < 0.5:
                    continue   # skip near-zero regions to keep output tight
                flag = ''
                if name in expect_low  and pct > 5:  flag = '!!'
                if name in expect_high and pct < 10: flag = '??'
                parts.append(f'{name}={pct:.0f}%{flag}')

            # residual (null token + any unlisted region)
            total = 100 * sum(np.mean(region_pcts[n]) for n, _, _ in named_cols)
            residual = 100 - total
            if abs(residual) > 1:
                parts.append(f'null/other={residual:.0f}%')

            print(f'    {row_name:20s} {row_lo:3d}:{row_hi:3d}  '
                  + '  '.join(parts))

        if rb['type'] == 'iq':
            _report_rows(rb['sl0'], rb['sl1'], 'IQ_SLOT',
                         expect_low=('enc_src',), expect_high=('enc_slot',))
            if rb['w0'] < rb['w1']:
                _report_rows(rb['w0'], rb['w1'], 'IQ_warmup',
                             expect_low=('enc_src', 'enc_slot'))
            _report_rows(rb['c0'], rb['c1'], 'IQ_output',
                         expect_low=('enc_src', 'enc_slot'))
        else:
            _report_rows(rb['sla0'], rb['sla1'], 'IR_SLOT_A',
                         expect_low=('enc_src',), expect_high=('enc_slot',))
            _report_rows(rb['am0'],  rb['am1'],  'IR_argmax',
                         expect_low=('enc_src',))
            _report_rows(rb['slb0'], rb['slb1'], 'IR_SLOT_B',
                         expect_low=('enc_src',))
            if rb['w0'] < rb['w1']:
                _report_rows(rb['w0'], rb['w1'], 'IR_warmup',
                             expect_low=('enc_src', 'enc_slot'))
            _report_rows(rb['c0'],   rb['c1'],   'IR_output',
                         expect_low=('enc_src', 'enc_slot'))

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Attention map visualiser for KVMemModel')
    p.add_argument('--ckpt',      required=True,       help='Path to checkpoint .pt')
    p.add_argument('--seq-idx',   type=int, default=0, help='Val sequence index (0-7)')
    p.add_argument('--warmup-x',  type=int, default=0, help='Warmup byte offset (for iq_global_rw)')
    p.add_argument('--device',    default='cpu')
    p.add_argument('--out',       default=None,        help='Output PNG path (auto if omitted)')
    p.add_argument('--avg-heads', action='store_true', help='Show head-avg only (not per-head)')
    p.add_argument('--analyse',   action='store_true', help='Print text pattern analysis instead of plotting')
    p.add_argument('--all-x',     action='store_true', help='Run for all chunk-aligned warmup offsets')
    args = p.parse_args()

    if args.analyse:
        xs = [0, 16, 32] if args.all_x else [args.warmup_x]
        for x in xs:
            analyse_patterns(args.ckpt, warmup_x=x, device_str=args.device)
    else:
        xs = [0, 16, 32] if args.all_x else [args.warmup_x]
        for x in xs:
            visualise(args.ckpt, seq_idx=args.seq_idx, warmup_x=x,
                      device_str=args.device, out_path=args.out,
                      avg_heads=args.avg_heads)


if __name__ == '__main__':
    main()
