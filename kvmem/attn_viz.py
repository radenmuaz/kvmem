"""
attn_viz.py — attention map heatmaps for HMNModel checkpoints (kvmem/hmn.py).

Rewired from the original archive_v1/root_scripts/attn_viz.py (built for the
pre-rewrite kvmem.model/kvmem.train_hmn_chunk stack) to the current
consolidated kvmem/hmn.py: STATE terminology (not SLOT), chain_steps (not
traj_mix/windows), shared chat tags written into the token sequence, and the
new STATE_QUEUE_in region. Only supports chunk_positions_chained layouts
(round-0-then-IR-per-chain-step) — the old iq_global_rw/ir_local trajectory
branching is gone since the new hp format has no traj_mix.

Captures attention weights by temporarily replacing F.scaled_dot_product_attention
with a version that saves the softmax weights before returning. No model or train
files are modified.

Usage:
  python3 -m kvmem.attn_viz --ckpt kvmem/logs/hmn_stage0_round0_single/checkpoints/stage0_best.pt
  python3 -m kvmem.attn_viz --ckpt <path> --seq-idx 0 --out attn_maps.png
  python3 -m kvmem.attn_viz --ckpt <path> --analyse

Output:
  <ckpt_dir>/attn_<step>_seq<idx>.png
  One figure per sequence: (n_layers) rows, (n_heads+1) columns.
  Last column = head-averaged map. Segment boundaries annotated as colored bars.
"""

import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from kvmem.hmn import (
    build_model,
    chunk_positions_chained,
    chunk_mask_fb,
    _cyclic_state_ids,
    make_test_sequences,
)


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
        # (single_attn/attn_mlp: n_layers calls; dual_attn: 2*n_layers calls,
        # one per attn1/attn2 sublayer, in block order.)

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

_SEG_COLORS = {
    'src':    '#4CAF50',   # green  — source bytes in enc_blocks
    'enc_st': '#2196F3',   # blue   — enc_block STATE tokens
    'queue':  '#00BCD4',   # cyan   — STATE_QUEUE_in (chain memory carry-in)
    'r0_st':  '#FF9800',   # orange — round-0 STATE
    'r0_wm':  '#FFEB3B',   # yellow — round-0 warmup
    'r0_out': '#F44336',   # red    — round-0 output
    'sta':    '#9C27B0',   # purple — IR STATE_A
    'am':     '#E91E63',   # pink   — IR argmax
    'stb':    '#673AB7',   # indigo — IR STATE_B
    'ir_wm':  '#FFEB3B',   # yellow — IR warmup (same as round-0 warmup)
    'ir_out': '#F44336',   # red    — IR output (same as round-0 output)
}


def _build_segments(pos_content: dict) -> list[tuple[int, int, str, str]]:
    """Return [(start, end, label, color), ...] for every position segment."""
    segs = []
    for i, b in enumerate(pos_content['enc_blocks']):
        segs.append((b['s0'],  b['s1'],  f'src{i}',  _SEG_COLORS['src']))
        segs.append((b['sl0'], b['sl1'], f'est{i}',  _SEG_COLORS['enc_st']))

    r0_idx = ir_idx = 0
    for rb in pos_content['rec_blocks']:
        if rb['type'] == 'iq':
            if 'queue0' in rb:
                segs.append((rb['queue0'], rb['queue1'], f'R{r0_idx}_q', _SEG_COLORS['queue']))
            segs.append((rb['sl0'], rb['sl1'], f'R{r0_idx}_st', _SEG_COLORS['r0_st']))
            if rb['w0'] < rb['w1']:
                segs.append((rb['w0'], rb['w1'], f'R{r0_idx}_wm', _SEG_COLORS['r0_wm']))
            segs.append((rb['c0'], rb['c1'], f'R{r0_idx}_out', _SEG_COLORS['r0_out']))
            r0_idx += 1
        else:
            segs.append((rb['sla0'], rb['sla1'], f'IR{ir_idx}_A',  _SEG_COLORS['sta']))
            segs.append((rb['am0'],  rb['am1'],  f'IR{ir_idx}_am', _SEG_COLORS['am']))
            segs.append((rb['slb0'], rb['slb1'], f'IR{ir_idx}_B',  _SEG_COLORS['stb']))
            if rb['w0'] < rb['w1']:
                segs.append((rb['w0'], rb['w1'], f'IR{ir_idx}_wm', _SEG_COLORS['ir_wm']))
            segs.append((rb['c0'], rb['c1'], f'IR{ir_idx}_out', _SEG_COLORS['ir_out']))
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
# Shared setup — reconstruct model + trajectory from a checkpoint's hp
# ---------------------------------------------------------------------------

def _load_model_and_pos(ckpt_path: str, device: torch.device):
    """Returns (model, pos_content, mask_np, tags, hp, sids)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    hp = ckpt['hp']

    hp_model = dict(
        V=hp['V'], d=hp['d'], n_layers=hp['n_layers'], n_heads=hp['n_heads'],
        block_type=hp.get('block_type', 'single_attn'),
        rope=hp.get('rope', True), yarn=hp.get('yarn', True),
        null_kv=hp.get('null_kv', True), rmsnorm=hp.get('rmsnorm', False),
        # chunk_attn disabled for viz — need full SDPA calls (not chunked) to
        # capture one weight tensor per layer instead of multiple micro-chunks.
        chunk_attn=0,
    )
    model = build_model(hp_model, device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    state_len = hp.get('state_len', 8)
    state_vocab_size = hp.get('state_vocab_size', 2)
    warmup_len = hp['warmup_len']
    stage_cfg = hp['curriculum'][0]
    n_chunks = stage_cfg['n_chunks']
    chunk_len = stage_cfg['chunk_len']
    chain_steps = stage_cfg['chain_steps']
    n_refine = stage_cfg.get('n_refine', 0)

    built = chunk_positions_chained(n_chunks, chunk_len, state_len, warmup_len,
                                    chain_steps, n_refine=n_refine,
                                    state_vocab_size=state_vocab_size)
    pos_content, pos_mask, tags = built['pos_content'], built['pos_mask'], built['tags']
    mask_np = chunk_mask_fb(pos_mask)
    sids = np.array(_cyclic_state_ids(state_len, state_vocab_size), dtype=np.int64)

    return model, pos_content, mask_np, tags, hp, sids


def _fill_tokens(pos_content: dict, tags: list, chunks_list: list, sids: np.ndarray,
                 warmup_len: int) -> np.ndarray:
    """Build a ground-truth-filled (teacher-forced) token array for one sequence."""
    L = pos_content['L']
    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos_content['enc_blocks']):
        tok[b['s0']:b['s1']] = chunks_list[k]
        tok[b['sl0']:b['sl1']] = sids

    wl = warmup_len
    for rb in pos_content['rec_blocks']:
        span_s, span_e = rb['span']
        gt_span = np.concatenate([np.array(chunks_list[i]) for i in range(span_s, span_e)])
        if rb['type'] == 'iq':
            if 'queue0' in rb:
                tok[rb['queue0']:rb['queue1']] = sids  # placeholder — h_inject overrides at runtime, not needed for a static viz pass
            tok[rb['sl0']:rb['sl1']] = sids
            if wl > 0:
                tok[rb['w0']:rb['w1']] = np.array(gt_span[:wl], dtype=np.int64)
            tok[rb['c0']:rb['c1']] = np.array(gt_span[wl:wl + rb['out_len']], dtype=np.int64)
        else:
            tok[rb['sla0']:rb['sla1']] = sids
            src_c0 = rb['argmax_src_c0']
            tok[rb['am0']:rb['am1']] = tok[src_c0:src_c0 + rb['out_len']]
            tok[rb['slb0']:rb['slb1']] = sids
            if wl > 0:
                tok[rb['w0']:rb['w1']] = np.array(gt_span[:wl], dtype=np.int64)
            tok[rb['c0']:rb['c1']] = np.array(gt_span[wl:wl + rb['out_len']], dtype=np.int64)

    # Write the shared chat tags (<src>/<query>/<response> — same tokens at
    # every chain step, no per-position variants).
    tag_pos = np.array([p for p, _ in tags], dtype=np.int64)
    tag_ids = np.array([i for _, i in tags], dtype=np.int64)
    tok[tag_pos] = tag_ids
    return tok


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------

def visualise(ckpt_path: str, seq_idx: int = 0, device_str: str = 'cpu',
             out_path: str | None = None, avg_heads: bool = False):
    device = torch.device(device_str)
    model, pos_content, mask_np, tags, hp, sids = _load_model_and_pos(ckpt_path, device)
    L = pos_content['L']
    mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device)
    n_layers = hp['n_layers']
    n_heads = hp['n_heads']
    warmup_len = hp['warmup_len']
    stage_cfg = hp['curriculum'][0]
    n_chunks, chunk_len = stage_cfg['n_chunks'], stage_cfg['chunk_len']
    step = torch.load(ckpt_path, map_location=device).get('step', 0)

    val_seqs = make_test_sequences(n_chunks * chunk_len)
    seq_name = list(val_seqs.keys())[seq_idx % len(val_seqs)]
    seq_bytes = list(val_seqs.values())[seq_idx % len(val_seqs)]
    chunks_list = [seq_bytes[k * chunk_len:(k + 1) * chunk_len] for k in range(n_chunks)]

    tok = _fill_tokens(pos_content, tags, chunks_list, sids, warmup_len)
    tok_t = torch.tensor(tok, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)

    # NOTE: this is a single static forward pass with plain placeholder tokens
    # in any STATE_QUEUE_in region (no h_inject) — for chained checkpoints
    # this only shows what round-0 attends to BEFORE chain memory is wired
    # in, not the true chained-training-time attention pattern. Good enough
    # for inspecting the round-0/IR mechanism itself; not yet a chain-memory
    # visualization (that would need to replay train()'s h_inject sequence).
    cap = AttentionCapture()
    with cap, torch.no_grad():
        _ = model(tok_t, mask_t)

    n_calls = len(cap.weights)
    if n_calls not in (n_layers, 2 * n_layers):
        print(f'[attn_viz] WARNING: expected {n_layers} (or {2*n_layers} for dual_attn) '
              f'attention calls, got {n_calls}.')

    attn_maps = []
    for w in cap.weights[:n_layers]:
        w = w[0]           # (H, L_q, L_kv)
        w = w[:, :, :L]    # trim null token column if present
        attn_maps.append(w.numpy())  # (H, L, L)

    segs = _build_segments(pos_content)

    n_cols = (1 if avg_heads else n_heads) + 1
    n_rows = len(attn_maps)
    fig_w = max(12, n_cols * 2.5)
    fig_h = max(8, n_rows * 2.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle(
        f'{Path(ckpt_path).parent.parent.name}  step={step}  seq={seq_name}  L={L}',
        fontsize=10)

    vmax_global = max(m.mean(0).max() for m in attn_maps)

    for layer_i, W in enumerate(attn_maps):
        W_avg = W.mean(0)
        col_range = range(n_heads) if not avg_heads else []
        for head_j in col_range:
            ax = axes[layer_i][head_j]
            ax.imshow(W[head_j], aspect='auto', origin='upper',
                     cmap='hot', vmin=0, vmax=W[head_j].max())
            _draw_block_lines(ax, segs)
            _draw_seg_bar(ax, segs, L, 'x')
            _draw_seg_bar(ax, segs, L, 'y')
            ax.set_title(f'L{layer_i} H{head_j}', fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])

        ax_avg = axes[layer_i][-1]
        ax_avg.imshow(W_avg, aspect='auto', origin='upper',
                     cmap='hot', vmin=0, vmax=vmax_global)
        _draw_block_lines(ax_avg, segs)
        _draw_seg_bar(ax_avg, segs, L, 'x')
        _draw_seg_bar(ax_avg, segs, L, 'y')
        ax_avg.set_title(f'L{layer_i} avg', fontsize=7)
        ax_avg.set_xticks([]); ax_avg.set_yticks([])

    fig.legend(handles=_legend_patches(), loc='lower right',
              fontsize=7, ncol=2, framealpha=0.8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if out_path is None:
        ckpt_dir = Path(ckpt_path).parent
        ckpt_stem = Path(ckpt_path).stem
        out_path = str(ckpt_dir / f'attn_{ckpt_stem}_seq{seq_idx}.png')

    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[attn_viz] saved -> {out_path}')
    return out_path


# ---------------------------------------------------------------------------
# Expected-pattern analysis — text report of where each critical row attends
# ---------------------------------------------------------------------------

def analyse_patterns(ckpt_path: str, device_str: str = 'cpu'):
    """
    Text analysis: for each rec_block, report which source region each
    row-type (STATE, warmup, output) directs the majority of its attention.
    Flags violations of the expected pattern.
    """
    device = torch.device(device_str)
    model, pos_content, mask_np, tags, hp, sids = _load_model_and_pos(ckpt_path, device)
    L = pos_content['L']
    mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device)
    warmup_len = hp['warmup_len']
    stage_cfg = hp['curriculum'][0]
    n_chunks, chunk_len = stage_cfg['n_chunks'], stage_cfg['chunk_len']

    val_seqs = make_test_sequences(n_chunks * chunk_len)
    all_weights: list[list[np.ndarray]] = [[] for _ in range(hp['n_layers'])]

    for seq_bytes in list(val_seqs.values()):
        chunks_list = [seq_bytes[k * chunk_len:(k + 1) * chunk_len] for k in range(n_chunks)]
        tok = _fill_tokens(pos_content, tags, chunks_list, sids, warmup_len)
        tok_t = torch.tensor(tok, dtype=torch.long, device=device).unsqueeze(0)
        cap = AttentionCapture()
        with cap, torch.no_grad():
            _ = model(tok_t, mask_t)
        for li, w in enumerate(cap.weights[:hp['n_layers']]):
            all_weights[li].append(w[0].mean(0).numpy())  # head-avg (L, L_kv)

    avg_maps = [np.stack(ws).mean(0)[:, :L] for ws in all_weights]

    enc_src_mask = np.zeros(L, dtype=bool)
    enc_state_mask = np.zeros(L, dtype=bool)
    for b in pos_content['enc_blocks']:
        enc_src_mask[b['s0']:b['s1']] = True
        enc_state_mask[b['sl0']:b['sl1']] = True

    print(f'\n{"="*80}')
    print(f'Attention pattern analysis: {Path(ckpt_path).name}')
    print(f'  n_chunks={n_chunks}  chunk_len={chunk_len}  state_len={hp.get("state_len",8)}  L={L}')
    print(f'  Averaged over {len(val_seqs)} val sequences, {hp["n_layers"]} layers, head-avg')
    print(f'{"="*80}')

    def _named_col_masks(rb: dict) -> list[tuple[str, np.ndarray, str]]:
        named = [('enc_src', enc_src_mask, 'low'), ('enc_state', enc_state_mask, '')]
        if 'queue0' in rb:
            q = np.zeros(L, dtype=bool); q[rb['queue0']:rb['queue1']] = True
            named.append(('queue_in', q, ''))
        if rb['type'] == 'iq':
            st = np.zeros(L, dtype=bool); st[rb['sl0']:rb['sl1']] = True
            wm = np.zeros(L, dtype=bool); wm[rb['w0']:rb['w1']] = True
            out = np.zeros(L, dtype=bool); out[rb['c0']:rb['c1']] = True
            named += [('R0_state', st, ''), ('R0_warm', wm, ''), ('R0_out', out, '')]
        else:
            sla = np.zeros(L, dtype=bool); sla[rb['sla0']:rb['sla1']] = True
            am = np.zeros(L, dtype=bool); am[rb['am0']:rb['am1']] = True
            slb = np.zeros(L, dtype=bool); slb[rb['slb0']:rb['slb1']] = True
            ir_wm = np.zeros(L, dtype=bool); ir_wm[rb['w0']:rb['w1']] = True
            ir_out = np.zeros(L, dtype=bool); ir_out[rb['c0']:rb['c1']] = True
            named += [('IR_stA', sla, ''), ('IR_am', am, ''), ('IR_stB', slb, ''),
                     ('IR_warm', ir_wm, ''), ('IR_out', ir_out, '')]
        return named

    for rb_i, rb in enumerate(pos_content['rec_blocks']):
        span_s, span_e = rb['span']
        print(f'\n  rec_block[{rb_i}] type={rb["type"]}  span=({span_s},{span_e})')
        named_cols = _named_col_masks(rb)

        def _report_rows(row_lo, row_hi, row_name, expect_low=(), expect_high=()):
            if row_lo >= row_hi:
                return
            region_pcts: dict[str, list[float]] = {n: [] for n, _, _ in named_cols}
            for W in avg_maps:
                row_mean = W[row_lo:row_hi, :].mean(0)
                row_mean = row_mean / (row_mean.sum() + 1e-9)
                for name, col_mask, _ in named_cols:
                    region_pcts[name].append(float(row_mean[col_mask].sum()))

            parts = []
            for name, _, _ in named_cols:
                pct = 100 * np.mean(region_pcts[name])
                if pct < 0.5:
                    continue
                flag = ''
                if name in expect_low and pct > 5: flag = '!!'
                if name in expect_high and pct < 10: flag = '??'
                parts.append(f'{name}={pct:.0f}%{flag}')

            total = 100 * sum(np.mean(region_pcts[n]) for n, _, _ in named_cols)
            residual = 100 - total
            if abs(residual) > 1:
                parts.append(f'null/other={residual:.0f}%')

            print(f'    {row_name:20s} {row_lo:3d}:{row_hi:3d}  ' + '  '.join(parts))

        if rb['type'] == 'iq':
            _report_rows(rb['sl0'], rb['sl1'], 'R0_STATE',
                        expect_low=('enc_src',), expect_high=('enc_state',))
            if rb['w0'] < rb['w1']:
                _report_rows(rb['w0'], rb['w1'], 'R0_warmup',
                            expect_low=('enc_src', 'enc_state'))
            _report_rows(rb['c0'], rb['c1'], 'R0_output',
                        expect_low=('enc_src', 'enc_state'))
        else:
            _report_rows(rb['sla0'], rb['sla1'], 'IR_STATE_A',
                        expect_low=('enc_src',), expect_high=('enc_state',))
            _report_rows(rb['am0'], rb['am1'], 'IR_argmax', expect_low=('enc_src',))
            _report_rows(rb['slb0'], rb['slb1'], 'IR_STATE_B', expect_low=('enc_src',))
            if rb['w0'] < rb['w1']:
                _report_rows(rb['w0'], rb['w1'], 'IR_warmup',
                            expect_low=('enc_src', 'enc_state'))
            _report_rows(rb['c0'], rb['c1'], 'IR_output',
                        expect_low=('enc_src', 'enc_state'))

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Attention map visualiser for HMNModel (kvmem/hmn.py)')
    p.add_argument('--ckpt', required=True, help='Path to checkpoint .pt')
    p.add_argument('--seq-idx', type=int, default=0, help='Val sequence index (0-7)')
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', default=None, help='Output PNG path (auto if omitted)')
    p.add_argument('--avg-heads', action='store_true', help='Show head-avg only (not per-head)')
    p.add_argument('--analyse', action='store_true', help='Print text pattern analysis instead of plotting')
    args = p.parse_args()

    if args.analyse:
        analyse_patterns(args.ckpt, device_str=args.device)
    else:
        visualise(args.ckpt, seq_idx=args.seq_idx, device_str=args.device,
                 out_path=args.out, avg_heads=args.avg_heads)


if __name__ == '__main__':
    main()
