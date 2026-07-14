"""
kvmem/train_hmn_feedback.py — HashMemNet with argmax feedback refinement.

Architecture: no MEM_START/MEM_END. Only SLOT tokens as structural markers.
slot_count (default 2): number of unique slot token IDs; cycles if slot_len > slot_count.

Sequence layouts
----------------
Turn 0  (IQ — encode src):
    [src: src_len] [SLOT×n] [warmup: wl] [out: ol]
    Mask: warmup/out blocked from src (bottleneck through slots).

Turn t≥1  (IR — argmax feedback):
    [SLOT_A×n] [argmax: ol] [SLOT_B×n] [warmup: wl] [out: ol]
    SLOT_A and SLOT_B have identical token IDs — distinguished by position.
    Mask: warmup/out blocked from SLOT_A and argmax (bottleneck through SLOT_B).

argmax = greedy decode of the previous turn's output positions (detached).
The model must read its own prediction, encode the correction into SLOT_B,
and produce better output.

Training: k+1 separate forward passes per step (no shared hidden state).
Loss: mean NTP over all turns.

Usage:
    python -m kvmem.train_hmn_feedback --config configs/hmn_feedback_32.py --device mps
"""
from __future__ import annotations

import argparse
import importlib.util
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
from kvmem.data import HMN_SLOT_0, HMN_VOCAB_SIZE
from kvmem.utils import make_test_sequences, cer
from kvmem.train_hmn_mono import (
    _stablemax_log_probs,
    _positional_ls_nll,
    load_config,
)


# ---------------------------------------------------------------------------
# Slot token helpers
# ---------------------------------------------------------------------------

def fb_slot_ids(slot_len: int, slot_count: int = 2) -> list[int]:
    """Cycle through slot_count unique IDs for slot_len positions."""
    return [HMN_SLOT_0 + (i % slot_count) for i in range(slot_len)]


# ---------------------------------------------------------------------------
# Sequence positions
# ---------------------------------------------------------------------------

def fb_iq_positions(src_len: int, slot_len: int, warmup_len: int, out_len: int) -> dict:
    """Turn 0: [src][SLOT×n][wm][out]"""
    s0,  s1  = 0,  src_len
    sl0, sl1 = s1, s1 + slot_len
    w0,  w1  = sl1, sl1 + warmup_len
    c0,  c1  = w1,  w1 + out_len
    return dict(s0=s0, s1=s1, sl0=sl0, sl1=sl1, w0=w0, w1=w1, c0=c0, c1=c1, L=c1)


def fb_ir_positions(slot_len: int, out_len: int, warmup_len: int) -> dict:
    """Turn t≥1: [SLOT_A×n][argmax][SLOT_B×n][wm][out]"""
    sla0, sla1 = 0,    slot_len
    am0,  am1  = sla1, sla1 + out_len
    slb0, slb1 = am1,  am1 + slot_len
    w0,   w1   = slb1, slb1 + warmup_len
    c0,   c1   = w1,   w1 + out_len
    return dict(sla0=sla0, sla1=sla1, am0=am0, am1=am1,
                slb0=slb0, slb1=slb1, w0=w0, w1=w1, c0=c0, c1=c1, L=c1)


# ---------------------------------------------------------------------------
# Attention masks
# ---------------------------------------------------------------------------

def fb_iq_mask(src_len: int, slot_len: int, warmup_len: int, out_len: int) -> np.ndarray:
    """Causal + bottleneck: warmup/out cannot attend to src.
    Convention matches hmn_mask_ir: 0.0 = attend, -1e9 = blocked (additive bias).
    """
    pos = fb_iq_positions(src_len, slot_len, warmup_len, out_len)
    L   = pos['L']
    r   = np.arange(L)
    c   = np.arange(L)
    visible = c[None, :] <= r[:, None]                      # causal
    recall_row  = r >= pos['w0']
    in_slot     = (c >= pos['sl0']) & (c < pos['sl1'])
    in_recall   = c >= pos['w0']
    # recall rows can only see slots and recall itself (not src)
    recall_blocked = recall_row[:, None] & ~(in_slot | in_recall)[None, :]
    visible = visible & ~recall_blocked
    return np.where(visible, 0.0, -1e9).astype(np.float32)


def fb_ir_mask(slot_len: int, out_len: int, warmup_len: int) -> np.ndarray:
    """Turn t≥1: warmup/out cannot attend to SLOT_A or argmax (only SLOT_B).
    Convention matches hmn_mask_ir: 0.0 = attend, -1e9 = blocked (additive bias).
    """
    pos = fb_ir_positions(slot_len, out_len, warmup_len)
    L   = pos['L']
    r   = np.arange(L)
    c   = np.arange(L)
    visible = c[None, :] <= r[:, None]                      # causal
    recall_row  = r >= pos['w0']
    in_slb      = (c >= pos['slb0']) & (c < pos['slb1'])
    in_recall   = c >= pos['w0']
    # recall rows can only see slb and recall itself (not sla or argmax)
    recall_blocked = recall_row[:, None] & ~(in_slb | in_recall)[None, :]
    visible = visible & ~recall_blocked
    return np.where(visible, 0.0, -1e9).astype(np.float32)


# ---------------------------------------------------------------------------
# Batch builders
# ---------------------------------------------------------------------------

def _fb_make_iq_batch(rng, B: int, src_len: int, slot_len: int, slot_count: int,
                      warmup_len: int, out_len: int,
                      segs=None, wm_batch=None, tgt_batch=None) -> np.ndarray:
    pos  = fb_iq_positions(src_len, slot_len, warmup_len, out_len)
    sids = np.array(fb_slot_ids(slot_len, slot_count), dtype=np.int64)
    tok  = np.zeros((B, pos['L']), dtype=np.int64)
    if segs is None:
        segs = rng.integers(0, 256, size=(B, src_len), dtype=np.int64)
    if wm_batch is None or tgt_batch is None:
        f = min(int(src_len * 0.25), src_len - warmup_len - out_len)
        y = f + warmup_len
        wm_batch  = segs[:, f:y]
        tgt_batch = segs[:, y:y + out_len]
    tok[:, pos['s0']:pos['s1']]   = segs
    tok[:, pos['sl0']:pos['sl1']] = sids
    tok[:, pos['w0']:pos['w1']]   = wm_batch
    tok[:, pos['c0']:pos['c1']]   = tgt_batch
    return tok


def _fb_make_ir_batch(argmax_np: np.ndarray,
                      slot_len: int, slot_count: int,
                      warmup_len: int, out_len: int,
                      wm_batch: np.ndarray, tgt_batch: np.ndarray) -> np.ndarray:
    """argmax_np: [B, out_len] int64 — previous turn's greedy output."""
    B   = argmax_np.shape[0]
    pos = fb_ir_positions(slot_len, out_len, warmup_len)
    sid = np.array(fb_slot_ids(slot_len, slot_count), dtype=np.int64)
    tok = np.zeros((B, pos['L']), dtype=np.int64)
    tok[:, pos['sla0']:pos['sla1']] = sid
    tok[:, pos['am0']:pos['am1']]   = argmax_np
    tok[:, pos['slb0']:pos['slb1']] = sid
    tok[:, pos['w0']:pos['w1']]     = wm_batch
    tok[:, pos['c0']:pos['c1']]     = tgt_batch
    return tok


# ---------------------------------------------------------------------------
# AR decode (eval)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ar_decode_fb(model, x_S: list[int], slot_len: int, slot_count: int,
                 warmup: list[int], out_len: int, device, k: int = 0) -> list[int]:
    """Greedy AR decode with k feedback refinement turns."""
    src_len = len(x_S)
    wl      = len(warmup)
    sids    = fb_slot_ids(slot_len, slot_count)

    # ── Turn 0: IQ ──────────────────────────────────────────────────────────
    pos0 = fb_iq_positions(src_len, slot_len, wl, out_len)
    m0   = torch.tensor(fb_iq_mask(src_len, slot_len, wl, out_len),
                        dtype=torch.float32, device=device)
    tok0 = np.zeros(pos0['L'], dtype=np.int64)
    tok0[pos0['s0']:pos0['s1']]   = x_S
    tok0[pos0['sl0']:pos0['sl1']] = sids
    tok0[pos0['w0']:pos0['w1']]   = warmup

    t0 = torch.tensor(tok0, dtype=torch.long, device=device)
    gen = []
    for j in range(out_len):
        logits = model(t0, m0)
        nb = int(logits[pos0['c0'] + j - 1].argmax())
        gen.append(nb)
        t0[pos0['c0'] + j] = nb
    if k == 0:
        return gen

    # ── Turns 1..k: feedback ────────────────────────────────────────────────
    pos_ir = fb_ir_positions(slot_len, out_len, wl)
    m_ir   = torch.tensor(fb_ir_mask(slot_len, out_len, wl),
                          dtype=torch.float32, device=device)

    for _ in range(k):
        tok_ir = np.zeros(pos_ir['L'], dtype=np.int64)
        tok_ir[pos_ir['sla0']:pos_ir['sla1']] = sids
        tok_ir[pos_ir['am0']:pos_ir['am1']]   = gen          # previous output
        tok_ir[pos_ir['slb0']:pos_ir['slb1']] = sids
        tok_ir[pos_ir['w0']:pos_ir['w1']]     = warmup

        t_ir = torch.tensor(tok_ir, dtype=torch.long, device=device)
        gen  = []
        for j in range(out_len):
            logits = model(t_ir, m_ir)
            nb = int(logits[pos_ir['c0'] + j - 1].argmax())
            gen.append(nb)
            t_ir[pos_ir['c0'] + j] = nb

    return gen


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fb(hp: dict, log_base: str = 'logs', device_str: str = 'cpu'):
    device = torch.device(device_str)
    rng    = np.random.default_rng(hp.get('seed', 42))
    torch.manual_seed(hp.get('seed', 42))

    name     = hp.get('name', 'hmn_feedback')
    log_dir  = os.path.join(log_base, name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    log_file   = open(os.path.join(log_dir, 'train.log'),   'a', buffering=1)
    jsonl_file = open(os.path.join(log_dir, 'train.jsonl'), 'a', buffering=1)

    def _log(msg):
        print(msg)
        print(msg, file=log_file)

    def _jlog(d):
        jsonl_file.write(json.dumps(d) + '\n')

    # ── Model ────────────────────────────────────────────────────────────────
    hp_model = dict(V=hp.get('V', HMN_VOCAB_SIZE),
                    d=hp['d'], n_layers=hp['n_layers'], n_heads=hp['n_heads'],
                    d_ff=hp['d_ff'],
                    rope=hp.get('rope', True), yarn=hp.get('yarn', True),
                    null_kv=hp.get('null_kv', True),
                    compile=hp.get('compile', False))
    model = build_model(hp_model, device)
    n_params = sum(p.numel() for p in model.parameters())
    _log(f'Model: {n_params:,} params  device={device}')

    if hp.get('_pretrained_ckpt'):
        ckpt = torch.load(hp['_pretrained_ckpt'], map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        _log(f'Loaded pretrained: {hp["_pretrained_ckpt"]}')

    # ── Optimizer ────────────────────────────────────────────────────────────
    lr_max      = hp.get('lr_max', 3e-4)
    wd          = hp.get('wd', 0.001)
    warmup_steps = hp.get('warmup_steps', 500)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=wd)

    # ── Curriculum ───────────────────────────────────────────────────────────
    curriculum = hp.get('curriculum', [])
    assert curriculum, 'hp must have curriculum'

    global_step = 0
    t_start     = time.time()

    use_stablemax = hp.get('stablemax', False)
    log_probs_fn  = (_stablemax_log_probs if use_stablemax
                     else lambda lg: F.log_softmax(lg, dim=-1))

    for stage_i, stage in enumerate(curriculum):
        src_len    = stage.get('src_len',    stage.get('seg_len', 32))
        slot_len   = stage.get('slot_len',   4)
        slot_count = stage.get('slot_count', 2)
        warmup_len = stage.get('warmup_len', 8)
        out_len    = stage.get('out_len',    24)
        B          = stage.get('B',          8)
        n_steps    = stage.get('n_steps',    80000)
        eval_every = hp.get('eval_every',    10000)
        log_every  = hp.get('log_every',     500)
        k_choices  = stage.get('k_choices',  [0, 1, 2, 3, 4])
        ls_max     = stage.get('ls_max',     hp.get('ls_max', 0.0))
        ls_anneal  = stage.get('ls_anneal',  n_steps)
        eval_turns = stage.get('hmn_eval_turns', [0, 1, 2, 3, 4])
        # loss_agg: 'flat' (mean over turns) or 'cum_mean' (mean of cumulative means)
        loss_agg   = stage.get('loss_agg',   hp.get('loss_agg', 'flat'))

        # Precompute masks
        pos_iq = fb_iq_positions(src_len, slot_len, warmup_len, out_len)
        pos_ir = fb_ir_positions(slot_len, out_len, warmup_len)
        mask_iq_t = torch.tensor(fb_iq_mask(src_len, slot_len, warmup_len, out_len),
                                 dtype=torch.float32, device=device)
        mask_ir_t = torch.tensor(fb_ir_mask(slot_len, out_len, warmup_len),
                                 dtype=torch.float32, device=device)

        # Eval helpers
        f_off  = min(int(src_len * 0.25), src_len - warmup_len - out_len)
        y_st   = f_off + warmup_len
        y_en   = y_st + out_len
        test_seqs = make_test_sequences(src_len)

        def _lr_sched(local_step):
            if local_step <= warmup_steps:
                return lr_max * local_step / max(warmup_steps, 1)
            return lr_max

        sids_np = np.array(fb_slot_ids(slot_len, slot_count), dtype=np.int64)

        _log(f'\n[stage {stage_i}] src={src_len} slot={slot_len} slot_count={slot_count}'
             f' wl={warmup_len} out={out_len}  B={B}  steps={n_steps}'
             f'  L_iq={pos_iq["L"]}  L_ir={pos_ir["L"]}')

        pbar = tqdm(range(1, n_steps + 1), desc=f'stage{stage_i}', dynamic_ncols=True)

        for local_step in pbar:
            global_step += 1
            lr = _lr_sched(local_step)
            for pg in opt.param_groups:
                pg['lr'] = lr

            _cur_ls = ls_max * max(0.0, 1.0 - local_step / ls_anneal)
            _jk = int(rng.choice(k_choices))

            model.train()
            opt.zero_grad()

            # Sample batch
            segs = rng.integers(0, 256, size=(B, src_len), dtype=np.int64)
            wm_batch  = segs[:, f_off:y_st]
            tgt_batch = segs[:, y_st:y_en]
            tgt_t     = torch.tensor(tgt_batch, device=device, dtype=torch.long)

            # ── Turn 0 (IQ) ─────────────────────────────────────────────────
            tok0 = _fb_make_iq_batch(rng, B, src_len, slot_len, slot_count,
                                     warmup_len, out_len,
                                     segs=segs, wm_batch=wm_batch, tgt_batch=tgt_batch)
            tk = torch.tensor(tok0, device=device, dtype=torch.long)
            logits = model(tk, mask_iq_t)
            lp = log_probs_fn(logits[:, pos_iq['c0']-1:pos_iq['c1']-1])
            nll_per_turn = [_positional_ls_nll(lp, tgt_t, _cur_ls).mean()]

            argmax_np = logits[:, pos_iq['c0']-1:pos_iq['c1']-1].argmax(-1).detach().cpu().numpy()

            # ── Turns 1..k (feedback) ────────────────────────────────────────
            for _t in range(1, _jk + 1):
                tok_ir = _fb_make_ir_batch(argmax_np, slot_len, slot_count,
                                           warmup_len, out_len, wm_batch, tgt_batch)
                tk = torch.tensor(tok_ir, device=device, dtype=torch.long)
                logits = model(tk, mask_ir_t)
                lp = log_probs_fn(logits[:, pos_ir['c0']-1:pos_ir['c1']-1])
                nll_per_turn.append(_positional_ls_nll(lp, tgt_t, _cur_ls).mean())
                if _t < _jk:
                    argmax_np = logits[:, pos_ir['c0']-1:pos_ir['c1']-1].argmax(-1).detach().cpu().numpy()

            if loss_agg == 'cum_mean' and len(nll_per_turn) > 1:
                cum, running = [], torch.zeros(1, device=device)[0]
                for _t, _n in enumerate(nll_per_turn):
                    running = running + _n
                    cum.append(running / (_t + 1))
                loss_val = torch.stack(cum).mean()
            else:
                loss_val = torch.stack(nll_per_turn).mean()
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_f = float(loss_val.detach())
            pbar.set_postfix(loss=f'{loss_f:.3f}', lr=f'{lr:.1e}',
                             mode=f'fb(k={_jk})', refresh=False)

            if local_step % log_every == 0:
                _jlog(dict(step=global_step, loss=round(loss_f, 5), lr=lr,
                           k=_jk, nll_turns=[round(float(n), 4) for n in nll_per_turn]))

            # ── Eval ─────────────────────────────────────────────────────────
            if local_step % eval_every == 0 or local_step == n_steps:
                model.eval()
                elapsed = time.time() - t_start
                h, m = divmod(int(elapsed), 3600)
                m, s = divmod(m, 60)
                _log(f'\n--- stage={stage_i} step={local_step}/{n_steps}'
                     f'  g={global_step}  loss={loss_f:.4f}  lr={lr:.2e}'
                     f'  {h:02d}:{m:02d}:{s:02d} ---')

                prev_perfect = False
                for hk in range(max(eval_turns) + 1):
                    if hk not in eval_turns:
                        continue
                    hk_cer = []
                    for sname, x_S in test_seqs.items():
                        wm  = x_S[max(0, y_st - warmup_len):y_st]
                        if len(wm) < warmup_len:
                            wm = [x_S[0]] * (warmup_len - len(wm)) + list(wm)
                        tgt = x_S[y_st:y_en]
                        with torch.no_grad():
                            g = ar_decode_fb(model, x_S, slot_len, slot_count,
                                             wm, len(tgt), device, k=hk)
                        hk_cer.append(cer(g, tgt))
                    perfect   = (sum(hk_cer) == 0.0)
                    match_pct = round(100 * (1 - sum(hk_cer) / len(hk_cer)), 1)
                    ok = '✓✓' if (perfect and prev_perfect) else ('✓' if perfect else '✗')
                    _log(f'  {ok} fb k={hk}  match={match_pct:.1f}%')
                    prev_perfect = perfect

        # ── Checkpoint ───────────────────────────────────────────────────────
        ckpt_path = os.path.join(ckpt_dir, f'stage{stage_i}_end.pt')
        torch.save(dict(model=model.state_dict(), hp=hp,
                        stage=stage_i, step=global_step), ckpt_path)
        _log(f'  [ckpt stage {stage_i} end] {ckpt_path}')

    elapsed = time.time() - t_start
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    _log(f'\nDone. Total: {h:02d}:{m:02d}:{s:02d} ({int(elapsed)}s)')
    log_file.close()
    jsonl_file.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     required=True)
    p.add_argument('--device',     default='cpu')
    p.add_argument('--pretrained', default=None)
    p.add_argument('--resume',     default=None)
    p.add_argument('--log-dir',    default='logs')
    args = p.parse_args()

    hp = load_config(args.config)
    if args.pretrained:
        hp['_pretrained_ckpt'] = args.pretrained
    if args.resume:
        hp['_resume_ckpt'] = args.resume

    train_fb(hp, log_base=args.log_dir, device_str=args.device)


if __name__ == '__main__':
    main()
