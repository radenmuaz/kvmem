"""
kvmem/train.py — PyTorch training loop for KV-memory recall.

Replaces mini_recall.py train_mini. Supports:
  - Full-sequence and random-window (chunk) training
  - Pre-generated curriculum datasets (from gen_dataset.py)
  - MPS / CPU device
  - Greedy AR eval on held-out deterministic test sequences

Usage:
    python -m kvmem.train                                    # defaults
    python -m kvmem.train --seg-len 128 --device mps        # MPS
    python -m kvmem.train --dataset-dir data/curriculum_128_seq --device mps
    python -m kvmem.train --seg-len 576 --N 576 --device mps --rope --yarn
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from kvmem.model import build_model
from kvmem.data import (
    MEM_OPEN, MEM_CLOSE, MEM_OVERHEAD, MEM_OPEN_LEN, MEM_CLOSE_LEN,
    make_slot_ids_tag, make_mask_tag,
)
from kvmem.gen_dataset import _sample_seg


# ---------------------------------------------------------------------------
# Batch builder (on-the-fly, no pre-generated dataset)
# ---------------------------------------------------------------------------

def make_batch(rng: np.random.Generator, B: int,
               seg_len: int, N: int, slot_style: str,
               chunk_len: int, device) -> torch.Tensor:
    """
    Build one training batch as a torch.Tensor on `device`.
    chunk_len=0: full-sequence recall  L = 2*seg_len + N + 7
    chunk_len>0: random-window         L = seg_len + N + 8 + chunk_len
    """
    slot_ids = make_slot_ids_tag(N, slot_style)
    M_start  = seg_len + MEM_OPEN_LEN
    Y_start  = seg_len + MEM_OPEN_LEN + N + MEM_CLOSE_LEN

    if chunk_len == 0:
        L   = seg_len + MEM_OVERHEAD + N + seg_len
        out = np.empty((B, L), dtype=np.int64)
        for i in range(B):
            seg = _sample_seg(rng, seg_len)
            out[i, :seg_len]          = seg
            out[i, seg_len:M_start]   = MEM_OPEN
            out[i, M_start:M_start+N] = slot_ids
            out[i, M_start+N:Y_start] = MEM_CLOSE
            out[i, Y_start:]          = seg
    else:
        L   = seg_len + MEM_OVERHEAD + N + 1 + chunk_len
        out = np.zeros((B, L), dtype=np.int64)
        n_windows = max(1, seg_len - chunk_len)
        for i in range(B):
            seg     = _sample_seg(rng, seg_len)
            y_start = int(rng.integers(0, n_windows + 1))
            y_end   = min(y_start + chunk_len, seg_len)
            warmup  = seg[y_start - 1] if y_start > 0 else seg[0]
            out[i, :seg_len]               = seg
            out[i, seg_len:M_start]        = MEM_OPEN
            out[i, M_start:M_start+N]      = slot_ids
            out[i, M_start+N:Y_start]      = MEM_CLOSE
            out[i, Y_start]                = warmup
            out[i, Y_start+1:Y_start+1+(y_end-y_start)] = seg[y_start:y_end]

    return torch.tensor(out, device=device)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_loss(model, tokens: torch.Tensor, mask: torch.Tensor,
                 Y_start: int, Y_sup_len: int, chunk_len: int) -> torch.Tensor:
    """
    NTP loss on Y positions only.
    chunk_len=0: supervise Y_start..Y_start+Y_sup_len
    chunk_len>0: supervise Y_start+1..Y_start+1+chunk_len (skip warmup)
    """
    B, L = tokens.shape
    # Batched forward: (B, L) → (B, L, V)
    logits = model(tokens, mask)
    lp     = F.log_softmax(logits[:, :-1], dim=-1)   # (B, L-1, V)
    tgts   = tokens[:, 1:]                            # (B, L-1)
    nll    = -lp.gather(2, tgts.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
    sup_start = Y_start + (1 if chunk_len > 0 else 0)
    sup_end   = sup_start + Y_sup_len
    mask_y    = torch.zeros(L - 1, device=tokens.device)
    mask_y[sup_start:sup_end] = 1.0
    return (nll * mask_y).sum(1).mean()


# ---------------------------------------------------------------------------
# Greedy AR decode for eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def ar_decode(model, x_S: list[int], N: int, slot_style: str,
              warmup: list[int], max_new: int, device) -> list[int]:
    """Greedy AR decode from KV memory. Returns warmup + generated."""
    seg_len  = len(x_S)
    slot_ids = make_slot_ids_tag(N, slot_style)
    mem_blk  = MEM_OPEN + slot_ids + MEM_CLOSE
    L_y      = len(warmup) + max_new
    L_full   = seg_len + MEM_OVERHEAD + N + L_y
    mask_t   = torch.tensor(make_mask_tag(seg_len, N, L_y),
                             dtype=torch.float32, device=device)
    generated = list(warmup)
    for _ in range(max_new):
        cur    = x_S + mem_blk + generated
        pad_n  = L_full - len(cur)
        tok    = torch.tensor(cur + [0] * pad_n, dtype=torch.long, device=device)
        logits = model(tok, mask_t)
        nb     = int(logits[len(cur) - 1].argmax())
        generated.append(nb)
    return generated


def cer(pred: list[int], ref: list[int]) -> float:
    m, n = len(ref), len(pred)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j-1] if ref[i-1] == pred[j-1] else 1 + min(prev[j-1], prev[j], dp[j-1])
    return dp[n] / max(m, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(hp: dict, log_base: str = 'logs', device_str: str = 'cpu'):
    device   = torch.device(device_str)
    seg_len  = hp['seg_len']
    N        = hp.get('N', seg_len)
    B        = hp['B']
    lr_max   = hp['lr_max']
    wd       = hp.get('wd', 0.01)
    n_steps  = hp['n_steps']
    eval_every  = hp.get('eval_every', 1000)
    log_every   = hp.get('log_every', 100)
    slot_style  = hp.get('slot_style', 'seq')
    chunk_len   = hp.get('chunk_len', 0)
    warmup_len  = hp.get('warmup_len', 4)   # context bytes before each window
    warmup_steps = hp.get('warmup_steps', 500)
    dataset_dir = hp.get('dataset_dir', None)
    seed        = hp.get('seed', 42)

    # Run dir
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(log_base, f'recall_{ts}')
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'hparams.json'), 'w') as f:
        json.dump(hp, f, indent=2)
    log_f  = open(os.path.join(run_dir, 'train.log'),   'w', buffering=1)
    jlog_f = open(os.path.join(run_dir, 'train.jsonl'), 'w', buffering=1)

    def _log(msg):  tqdm.write(msg); log_f.write(msg + '\n')
    def _jlog(d):   jlog_f.write(json.dumps(d) + '\n')

    # Model + optimizer
    torch.manual_seed(seed)
    model = build_model(hp, device)
    if hp.get('compile', False):
        model = torch.compile(model)
    from kvmem.optim import GrokAdamW
    use_grok = hp.get('grok', True)   # GrokAdamW is default
    if use_grok:
        opt = GrokAdamW(model.parameters(), lr=lr_max, weight_decay=wd,
                        rho=hp.get('grok_rho', 0.9), batch_size=B)
        if hp.get('compile', False):
            opt.step = torch.compile(opt.step)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=wd)

    # Mask
    if chunk_len == 0:
        L_y, L_total = seg_len, seg_len + MEM_OVERHEAD + N + seg_len
        Y_start, Y_sup_len = seg_len + MEM_OPEN_LEN + N + MEM_CLOSE_LEN, seg_len
    else:
        L_y      = warmup_len + chunk_len
        L_total  = seg_len + MEM_OVERHEAD + N + warmup_len + chunk_len
        Y_start  = seg_len + MEM_OPEN_LEN + N + MEM_CLOSE_LEN
        Y_sup_len = chunk_len

    mask_t = torch.tensor(make_mask_tag(seg_len, N, L_y),
                          dtype=torch.float32, device=device)

    # LR schedule — per-stage cosine, resets at each curriculum transition
    stage_start_step = [0]   # mutable ref updated on stage switch

    cycle_steps = hp.get('cycle_steps', 0)   # 0 = one-shot cosine; >0 = cosine restarts

    def lr_schedule(step):
        local = step - stage_start_step[0]
        if local < warmup_steps:
            return lr_max * max(local, 1) / warmup_steps
        local -= warmup_steps
        if cycle_steps > 0:
            # Cosine restarts every cycle_steps
            t = local % cycle_steps
            progress = t / cycle_steps
        else:
            stage_steps = steps_per_stage if (curriculum and steps_per_stage > 0) else n_steps
            progress = min(local / max(stage_steps - warmup_steps, 1), 1.0)
        return max(lr_max * 1e-2, lr_max * 0.5 * (1 + math.cos(math.pi * progress)))

    # Test sequences
    from kvmem.utils import make_test_sequences
    test_seqs = make_test_sequences(seg_len)

    # Curriculum dataset (optional)
    curriculum = None
    curr_idx   = 0
    steps_per_stage = 0
    curr_mask = mask_t
    curr_chunk = chunk_len
    curr_Y_start = Y_start
    curr_Y_sup = Y_sup_len

    if dataset_dir:
        meta   = json.load(open(os.path.join(dataset_dir, 'meta.json')))
        wl     = meta.get('warmup_len', warmup_len)
        d_mode = meta.get('mode', 'curriculum')

        if d_mode == 'multi_size':
            sizes = meta['sizes']
            mix_w = meta.get('mix_weights', None)
            mix_p = np.array(mix_w) / sum(mix_w) if mix_w else None
            curriculum = []
            for ws in sizes:
                data = np.load(os.path.join(dataset_dir, f'win{ws}.npy'))
                is_full = (ws >= seg_len)
                s_chunk_len = 0 if is_full else ws
                s_L_y  = seg_len if is_full else wl + ws
                s_mask = torch.tensor(make_mask_tag(seg_len, N, s_L_y),
                                      dtype=torch.float32, device=device)
                s_Y_start = seg_len + MEM_OPEN_LEN + N + MEM_CLOSE_LEN
                s_Y_sup   = seg_len if is_full else ws
                curriculum.append((data, s_chunk_len, s_mask, s_Y_start, s_Y_sup))
                _log(f'  [multi-size] win={ws}  L={data.shape[1]}  n={len(data)}')
            steps_per_stage = 0   # no stage switching in multi-size
            _log(f'  [multi-size] {len(sizes)} window sizes mixed per step')
            curr_chunk, curr_mask, curr_Y_start, curr_Y_sup = curriculum[-1][1:]
        else:
            # Curriculum: sequential stages
            stages = meta['stages']
            steps_per_stage = n_steps // len(stages)
            curriculum = []
            for s_chunk in stages:
                data = np.load(os.path.join(dataset_dir, f'stage_chunk{s_chunk}.npy'))
                is_full = (s_chunk >= seg_len)
                s_chunk_len = 0 if is_full else s_chunk
                s_L_y  = seg_len if is_full else wl + s_chunk
                s_mask = torch.tensor(make_mask_tag(seg_len, N, s_L_y),
                                      dtype=torch.float32, device=device)
                s_Y_start = seg_len + MEM_OPEN_LEN + N + MEM_CLOSE_LEN
                s_Y_sup   = seg_len if is_full else s_chunk
                curriculum.append((data, s_chunk_len, s_mask, s_Y_start, s_Y_sup))
                _log(f'  [curriculum] stage chunk={s_chunk}  warmup={wl if not is_full else 1}  n={len(data)}  L={data.shape[1]}')
            _log(f'  [curriculum] {len(stages)} stages × ~{steps_per_stage} steps each')
            _, curr_chunk, curr_mask, curr_Y_start, curr_Y_sup = curriculum[0]

    mode_str = f'random-window chunk={chunk_len}' if chunk_len > 0 else 'full-sequence'
    if dataset_dir:
        mode_str = f'curriculum ({mode_str})'

    _log(f'\n=== KV Recall Training (PyTorch) | run_dir={run_dir} ===')
    _log(f'  cmd: {" ".join(sys.argv)}')
    _log(f'  Model: d={hp["d"]}  n_layers={hp["n_layers"]}  '
         f'params={model.count_params():,}  device={device}')
    _log(f'  seg_len={seg_len}  N={N}  slot_style={slot_style}  mode={mode_str}')
    _log(f'  Steps={n_steps}  B={B}  lr={lr_max}  wd={wd}')
    _log(f'  rope={hp.get("rope",False)}  yarn={hp.get("yarn",False)}')

    rng = np.random.default_rng(seed)
    t0  = time.time()

    pbar = tqdm(range(1, n_steps + 1), desc='recall', dynamic_ncols=True)

    if not dataset_dir:
        d_mode = 'online'

    for step in pbar:
        # Stage switch (curriculum only, not multi-size)
        if curriculum is not None and steps_per_stage > 0:
            stage_i = min(step // steps_per_stage, len(curriculum) - 1)
            if stage_i != curr_idx:
                curr_idx = stage_i
                _, curr_chunk, curr_mask, curr_Y_start, curr_Y_sup = curriculum[curr_idx]
                stage_start_step[0] = step
                _log(f'\n  [curriculum] step={step} → stage {curr_idx} '
                     f'chunk={curriculum[curr_idx][1] or "full"}  (LR reset)')

        # Get batch
        if curriculum is not None:
            if d_mode == 'multi_size':
                # Pick a random window size each step
                size_i = int(rng.choice(len(curriculum), p=mix_p) if mix_p is not None
                             else rng.integers(0, len(curriculum)))
                stage_data, curr_chunk, curr_mask, curr_Y_start, curr_Y_sup = curriculum[size_i]
            else:
                stage_data = curriculum[curr_idx][0]
            idx = rng.integers(0, len(stage_data), size=B)
            tokens = torch.tensor(stage_data[idx], dtype=torch.long, device=device)
        else:
            tokens = make_batch(rng, B, seg_len, N, slot_style, chunk_len, device)

        # LR
        lr = lr_schedule(step)
        for pg in opt.param_groups:
            pg['lr'] = lr

        # Train step
        model.train()
        opt.zero_grad()
        loss = compute_loss(model, tokens, curr_mask, curr_Y_start, curr_Y_sup, curr_chunk)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_f = float(loss)
        pbar.set_postfix(loss=f'{loss_f:.4f}', lr=f'{lr:.1e}', refresh=False)

        # Loss log
        if step % log_every == 0:
            elapsed = time.time() - t0
            _log(f'  step={step}/{n_steps}  loss={loss_f:.4f}  lr={lr:.2e}  {elapsed:.0f}s')
            _jlog(dict(step=step, loss=loss_f, lr=lr))

        # Eval
        if step % eval_every == 0 or step == 1:
            model.eval()
            elapsed = time.time() - t0
            _log(f'\n--- step={step}/{n_steps}  loss={loss_f:.4f}  lr={lr:.2e}  {elapsed:.0f}s ---')
            all_cer = []
            for name, x_S in test_seqs.items():
                warmup   = x_S[:1]
                target   = x_S[1:]
                gen      = ar_decode(model, x_S, N, slot_style, warmup, len(target), device)
                gen_tail = gen[1:]
                c = cer(gen_tail, target)
                all_cer.append(c)
                ok = '✓' if c == 0.0 else '✗'
                _log(f'  {ok} {name:15s}  match={100*(1-c):5.1f}%  CER={c:.3f}')
                _log(f'    gen={bytes(gen_tail).hex()}')
                _log(f'    ref={bytes(target).hex()}')
                _jlog(dict(step=step, seq=name, cer=c))

            mean_cer = sum(all_cer) / len(all_cer)
            _log(f'  → full-seq mean CER={mean_cer:.3f}  match={100*(1-mean_cer):.1f}%')
            _jlog(dict(step=step, mean_cer=mean_cer))

            # Windowed eval — always run if dataset has windowed examples
            eval_chunk = hp.get('chunk_len', 0)
            if eval_chunk == 0 and curriculum is not None:
                # infer from dataset: use smallest window size available
                sizes = meta.get('sizes', meta.get('stages', []))
                non_full = [s for s in sizes if s < seg_len]
                eval_chunk = min(non_full) if non_full else 0
            if curriculum is not None and eval_chunk > 0:
                wl = hp.get('warmup_len', 4)
                window_cers = []
                for name, x_S in list(test_seqs.items())[:4]:
                    y_start = seg_len // 4
                    y_end   = y_start + eval_chunk
                    wm_tok  = x_S[max(0, y_start-wl):y_start]
                    if len(wm_tok) < wl:
                        wm_tok = [x_S[0]] * (wl - len(wm_tok)) + wm_tok
                    slot_ids = make_slot_ids_tag(N, slot_style)
                    mem_blk  = MEM_OPEN + slot_ids + MEM_CLOSE
                    L_w = seg_len + MEM_OVERHEAD + N + wl + eval_chunk
                    wm_mask = torch.tensor(make_mask_tag(seg_len, N, wl + eval_chunk),
                                           dtype=torch.float32, device=device)
                    generated = list(wm_tok)
                    with torch.no_grad():
                        for _ in range(eval_chunk):
                            cur = x_S + mem_blk + generated
                            tok = torch.tensor(cur + [0]*(L_w-len(cur)), dtype=torch.long, device=device)
                            logits = model(tok, wm_mask)
                            generated.append(int(logits[len(cur)-1].argmax()))
                    gen_w = generated[wl:]
                    target_w = x_S[y_start:y_end]
                    c_w = cer(gen_w, target_w)
                    window_cers.append(c_w)
                    ok = '✓' if c_w == 0.0 else '✗'
                    _log(f'  {ok} [window] {name:12s} [{y_start}:{y_end}]  match={100*(1-c_w):5.1f}%  CER={c_w:.3f}')
                mean_w = sum(window_cers)/len(window_cers)
                _log(f'  → windowed mean CER={mean_w:.3f}  match={100*(1-mean_w):.1f}%')
                _jlog(dict(step=step, windowed_cer=mean_w))

            if mean_cer == 0.0:
                _log(f'\n★ PERFECT RECALL at step {step}!')
                ckpt = os.path.join(ckpt_dir, f'step{step}.pt')
                torch.save({'model': model.state_dict(), 'hp': hp, 'step': step}, ckpt)
                _log(f'  [ckpt] {ckpt}')
                # Only stop if no curriculum or already at final stage
                is_final = (curriculum is None or curr_idx == len(curriculum) - 1)
                if is_final:
                    break

        if step % (eval_every * 10) == 0:
            ckpt = os.path.join(ckpt_dir, f'step{step}.pt')
            torch.save({'model': model.state_dict(), 'hp': hp, 'step': step}, ckpt)
            _log(f'  [ckpt] {ckpt}')

    _log(f'\nDone. Total: {time.time()-t0:.0f}s')
    log_f.close(); jlog_f.close()
    return model, run_dir


# ---------------------------------------------------------------------------
# Defaults & CLI
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    seg_len=128, N=128, V=256, d=64, n_layers=4, n_heads=4, d_ff=256,
    B=8, lr_max=1e-3, wd=0.01, warmup_steps=200,
    n_steps=10_000, eval_every=1000, log_every=100,
    slot_style='seq', chunk_len=0,
    rope=True, yarn=True, seed=42,
)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seg-len',     type=int,   default=None)
    p.add_argument('--N',           type=int,   default=None)
    p.add_argument('--d',           type=int,   default=None)
    p.add_argument('--n-layers',    type=int,   default=None)
    p.add_argument('--d-ff',        type=int,   default=None)
    p.add_argument('--B',           type=int,   default=None)
    p.add_argument('--lr',          type=float, default=None)
    p.add_argument('--wd',          type=float, default=None)
    p.add_argument('--steps',       type=int,   default=None)
    p.add_argument('--eval-every',  type=int,   default=None)
    p.add_argument('--log-every',   type=int,   default=None)
    p.add_argument('--slot-style',  type=str,   default=None, choices=['zeros','seq'])
    p.add_argument('--chunk-len',   type=int,   default=None)
    p.add_argument('--warmup-len',  type=int,   default=None,
                   help='Context bytes before each window (default 4)')
    p.add_argument('--rope',        action='store_true', default=None)
    p.add_argument('--no-rope',     action='store_false', dest='rope')
    p.add_argument('--yarn',        action='store_true', default=None)
    p.add_argument('--no-yarn',     action='store_false', dest='yarn')
    p.add_argument('--warmup-steps', type=int, default=None,
                   help='LR warmup steps per stage (default 200)')
    p.add_argument('--grok',        action='store_true',
                   help='Use GrokAdamW (SNR-gated, arXiv:2605.01172)')
    p.add_argument('--grok-rho',    type=float, default=None,
                   help='GrokAdamW deviation EMA decay (default 0.9)')
    p.add_argument('--cycle-steps', type=int,   default=None,
                   help='Cosine restart period in steps (0=one-shot, e.g. 5000)')
    p.add_argument('--compile',     action='store_true', help='torch.compile (faster on MPS)')
    p.add_argument('--dataset-dir', type=str,   default=None)
    p.add_argument('--log-dir',     type=str,   default='logs')
    p.add_argument('--device',      type=str,   default='cpu', choices=['cpu','mps','cuda'])
    p.add_argument('--seed',        type=int,   default=None)
    args = p.parse_args()

    hp = dict(DEFAULTS)
    if args.seg_len:    hp['seg_len']    = args.seg_len
    if args.N:          hp['N']          = args.N
    if args.d:          hp['d']          = args.d; hp['d_ff'] = args.d * 4
    if args.d_ff:       hp['d_ff']       = args.d_ff
    if args.n_layers:   hp['n_layers']   = args.n_layers
    if args.B:          hp['B']          = args.B
    if args.lr:         hp['lr_max']     = args.lr
    if args.wd:         hp['wd']         = args.wd
    if args.steps:      hp['n_steps']    = args.steps
    if args.eval_every: hp['eval_every'] = args.eval_every
    if args.log_every:  hp['log_every']  = args.log_every
    if args.slot_style: hp['slot_style'] = args.slot_style
    if args.chunk_len  is not None: hp['chunk_len']  = args.chunk_len
    if args.warmup_len is not None: hp['warmup_len'] = args.warmup_len
    if args.rope is not None: hp['rope'] = args.rope
    if args.yarn is not None: hp['yarn'] = args.yarn
    if args.dataset_dir: hp['dataset_dir'] = args.dataset_dir
    if args.grok:                     hp['grok']         = True
    if args.grok_rho is not None:     hp['grok_rho']     = args.grok_rho
    if args.warmup_steps is not None: hp['warmup_steps'] = args.warmup_steps
    if args.cycle_steps  is not None: hp['cycle_steps']  = args.cycle_steps
    if args.compile:                 hp['compile']     = True
    if args.seed:       hp['seed'] = args.seed

    seg = hp['seg_len']; N = hp['N']
    hp['L_train'] = seg + MEM_OVERHEAD + N + seg
    hp['L_max']   = hp['L_train'] * 4

    train(hp, log_base=args.log_dir, device_str=args.device)
