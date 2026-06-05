"""
scripts/plot_train.py — Auto-updating matplotlib plot from train.jsonl.

Saves three PNG files alongside the jsonl on every update:
  train_plot.png          — combined (all panels)
  train_plot_train.png    — train metrics: bpb curve + 4 component losses
  train_plot_val.png      — val/eval metrics: loss curves + match% + per-turn bar + query

Redraws immediately when the jsonl file grows (triggered by each log write).

Usage:
  python scripts/plot_train.py logs/online_refine_64/train.jsonl
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── data loading ─────────────────────────────────────────────────────────────

def load_jsonl(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    except FileNotFoundError:
        pass
    return rows


def smooth(ys, w=5):
    if len(ys) < w:
        return list(ys)
    return list(np.convolve(ys, np.ones(w) / w, mode='valid'))


# ── individual panel drawing functions ───────────────────────────────────────

def plot_loss_curves(ax, train_rows, eval_rows):
    ax.cla()
    if train_rows:
        xs = [r['global_step'] for r in train_rows]
        ys = [r['bpb'] for r in train_rows]
        ax.plot(xs, ys, alpha=0.2, color='steelblue', linewidth=0.7)
        w = max(1, len(ys) // 60)
        ax.plot(xs[w-1:], smooth(ys, w), color='steelblue', label='train bpb', linewidth=1.2)
    if eval_rows:
        xs_e = [r['global_step'] for r in eval_rows]
        ax.plot(xs_e, [r['val_bpb'] for r in eval_rows],
                'o-', color='darkorange', markersize=3, linewidth=1, label='val_bpb')
        ref = [(r['global_step'], r['val_ref_bpb'])
               for r in eval_rows if r.get('val_ref_bpb') is not None]
        if ref:
            rx, ry = zip(*ref)
            ax.plot(rx, ry, 's-', color='crimson', markersize=3, linewidth=1, label='val_ref_bpb')
    ax.set_title('Loss (bpb)')
    ax.legend(fontsize=7)
    ax.set_xlabel('step')
    ax.grid(alpha=0.2)


def plot_match_pct(ax, eval_rows):
    ax.cla()
    if eval_rows:
        xs_n  = [r['global_step'] for r in eval_rows if 'n1_r0' in r]
        ys_n  = [r['n1_r0']       for r in eval_rows if 'n1_r0' in r]
        ys_t1 = [r.get('n1_r0_t1', 0) for r in eval_rows if 'n1_r0' in r]
        ys_q  = [r.get('n1_r0_query') for r in eval_rows if 'n1_r0' in r]
        if xs_n:
            ax.plot(xs_n, ys_n,  'o-',  color='seagreen',     markersize=3, linewidth=1, label='final')
            ax.plot(xs_n, ys_t1, 's--', color='mediumpurple', markersize=3, linewidth=1, label='t1')
            ys_q_c = [(v if v is not None else float('nan')) for v in ys_q]
            if any(v is not None for v in ys_q):
                ax.plot(xs_n, ys_q_c, '^:', color='gold', markersize=4, linewidth=1, label='query')
    ax.axhline(95.8, color='gray', linestyle=':', linewidth=0.8, label='baseline 95.8%')
    ax.set_title('Match % over training')
    ax.legend(fontsize=7)
    ax.set_ylim(-2, 105)
    ax.set_xlabel('step')
    ax.grid(alpha=0.2)


def plot_turn_bar(ax, eval_rows):
    ax.cla()
    if eval_rows:
        last   = eval_rows[-1]
        t_keys = sorted([k for k in last if k.startswith('n1_r0_t')],
                        key=lambda k: int(k.split('_t')[1]))
        if t_keys:
            turns  = [int(k.split('_t')[1]) for k in t_keys]
            values = [last[k] for k in t_keys]
            base   = values[0]
            colors = ['seagreen' if v >= base else 'crimson' for v in values]
            ax.bar(turns, values, color=colors, alpha=0.75)
            ax.axhline(100, color='gray', linestyle='--', linewidth=0.8)
            ax.axhline(base, color='purple', linestyle=':', linewidth=0.8, label=f't1={base:.1f}%')
            qm = last.get('n1_r0_query')
            if qm is not None:
                ax.axhline(qm, color='gold', linestyle='-.', linewidth=1.2, label=f'query={qm:.1f}%')
            ax.set_title(f'Per-turn match% @ step {last["global_step"]}  (green≥t1, red<t1)')
            ax.set_xlabel('turn')
            ax.set_ylabel('%')
            ax.set_ylim(0, 107)
            ax.legend(fontsize=7)
    ax.grid(alpha=0.2, axis='y')


def plot_k_dist(ax, online_rows):
    ax.cla()
    if online_rows:
        ks = [r['online_k'] for r in online_rows]
        unique_k = sorted(set(ks))
        counts = [ks.count(k) for k in unique_k]
        ax.bar(unique_k, counts, color='steelblue', alpha=0.7)
        ax.set_title('Sampled k distribution')
        ax.set_xlabel('k (refine turns)')
    ax.grid(alpha=0.2, axis='y')


_LOSS_SPECS = [
    ('online_l_ntp',  'steelblue',    'NTP final'),
    ('online_l_aux',  'darkorange',   'NTP aux turns'),
    ('online_l_mono', 'firebrick',    'mono penalty'),
    ('online_l_h',    'mediumpurple', 'h MSE'),
]


def plot_component_loss(ax, online_rows, key, color, label):
    ax.cla()
    if online_rows:
        xs_o = [r['global_step'] for r in online_rows]
        ys_o = [r.get(key, 0.0) for r in online_rows]
        w    = max(1, len(ys_o) // 40)
        ax.plot(xs_o, ys_o, alpha=0.15, color=color, linewidth=0.7)
        ax.plot(xs_o[w-1:], smooth(ys_o, w), color=color, linewidth=1.2)
    ax.set_title(label, fontsize=8)
    ax.set_xlabel('step', fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.2)


# ── figure builders ───────────────────────────────────────────────────────────

def build_combined(rows, path):
    """8-panel combined figure: 2 rows top + 4 loss panels bottom."""
    train_rows  = [r for r in rows if 'bpb' in r and 'val_bpb' not in r]
    eval_rows   = [r for r in rows if 'val_bpb' in r]
    online_rows = [r for r in train_rows if 'online_l_ntp' in r]

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                            height_ratios=[2.2, 2.2, 1.8], hspace=0.45, wspace=0.35)
    ax_loss  = fig.add_subplot(gs[0, :2])
    ax_match = fig.add_subplot(gs[0, 2:])
    ax_turns = fig.add_subplot(gs[1, :2])
    ax_k     = fig.add_subplot(gs[1, 2:])
    ax_ntp   = fig.add_subplot(gs[2, 0])
    ax_aux   = fig.add_subplot(gs[2, 1])
    ax_mono  = fig.add_subplot(gs[2, 2])
    ax_h     = fig.add_subplot(gs[2, 3])

    plot_loss_curves(ax_loss, train_rows, eval_rows)
    plot_match_pct(ax_match, eval_rows)
    plot_turn_bar(ax_turns, eval_rows)
    plot_k_dist(ax_k, online_rows)
    for ax, (key, color, label) in zip([ax_ntp, ax_aux, ax_mono, ax_h], _LOSS_SPECS):
        plot_component_loss(ax, online_rows, key, color, label)

    last_step = rows[-1].get('global_step', '?')
    fig.suptitle(f'{path}   step={last_step}', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, path.replace('.jsonl', '_plot.png'))


def build_train_metrics(rows, path):
    """Train-only: train bpb + 4 component losses, each a fat panel stacked vertically."""
    train_rows  = [r for r in rows if 'bpb' in r and 'val_bpb' not in r]
    online_rows = [r for r in train_rows if 'online_l_ntp' in r]

    n_rows = 5  # bpb + 4 losses
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, n_rows * 2.2))
    fig.subplots_adjust(hspace=0.5)

    ax_bpb = axes[0]
    ax_bpb.cla()
    if train_rows:
        xs = [r['global_step'] for r in train_rows]
        ys = [r['bpb'] for r in train_rows]
        ax_bpb.plot(xs, ys, alpha=0.2, color='steelblue', linewidth=0.7)
        w = max(1, len(ys) // 60)
        ax_bpb.plot(xs[w-1:], smooth(ys, w), color='steelblue', linewidth=1.2)
    ax_bpb.set_title('Train bpb', fontsize=8)
    ax_bpb.set_xlabel('step', fontsize=7)
    ax_bpb.tick_params(labelsize=7)
    ax_bpb.grid(alpha=0.2)

    for ax, (key, color, label) in zip(axes[1:], _LOSS_SPECS):
        plot_component_loss(ax, online_rows, key, color, label)

    last_step = rows[-1].get('global_step', '?')
    fig.suptitle(f'Train metrics — step={last_step}', fontsize=9)
    _save(fig, path.replace('.jsonl', '_plot_train.png'))


def build_val_metrics(rows, path):
    """Val/eval-only: loss curves + match% + per-turn bar, stacked vertically (fat panels)."""
    train_rows = [r for r in rows if 'bpb' in r and 'val_bpb' not in r]
    eval_rows  = [r for r in rows if 'val_bpb' in r]

    n_rows = 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, n_rows * 2.5))
    fig.subplots_adjust(hspace=0.55)

    plot_loss_curves(axes[0], train_rows, eval_rows)
    plot_match_pct(axes[1], eval_rows)
    plot_turn_bar(axes[2], eval_rows)

    last_step = rows[-1].get('global_step', '?')
    fig.suptitle(f'Val metrics — step={last_step}', fontsize=9)
    _save(fig, path.replace('.jsonl', '_plot_val.png'))


def _save(fig, img_path):
    fig.savefig(img_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── live watcher ──────────────────────────────────────────────────────────────

def refresh(path):
    rows = load_jsonl(path)
    if not rows:
        return
    build_combined(rows, path)
    build_train_metrics(rows, path)
    build_val_metrics(rows, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path', help='Path to train.jsonl')
    args = p.parse_args()
    path = args.path

    last_size = -1
    print(f'Watching {path}')
    print(f'  → {path.replace(".jsonl", "_plot.png")} (combined)')
    print(f'  → {path.replace(".jsonl", "_plot_train.png")} (train metrics)')
    print(f'  → {path.replace(".jsonl", "_plot_val.png")} (val metrics)')
    print('Ctrl-C to stop.')

    try:
        while True:
            try:
                size = os.path.getsize(path)
            except FileNotFoundError:
                size = 0
            if size != last_size:
                last_size = size
                try:
                    refresh(path)
                    print(f'\r[{time.strftime("%H:%M:%S")}] updated', end='', flush=True)
                except Exception as e:
                    print(f'\n[warn] {e}')
            time.sleep(0.5)
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
