"""
kvmem/train_role.py — Role-tag recall training.

Sequence format:
  <s> x_S </s> <m> slots </m> <f> warmup </f> <c> output </c>

  <s>source</s>  — the full source sequence encoded into KV slots
  <m>slots</m>   — KV memory (same as before)
  <f>from</f>    — explicit anchor: warmup_len bytes before the window
                   tells model WHERE in the source to start
  <c>continue</c>— model outputs the continuation from the anchor

Key advantage over bare-warmup scheme:
  The model sees explicit role tags, not just bytes. It can learn:
  "search <f>...</f> content in slots, then output the continuation into <c>"
  rather than guessing "is this warmup or source?"

Mask rules:
  - slots attend to x_S (encode source)
  - <f> region attends to slots only (locate via KV, not source directly)
  - <c> region attends to slots + <f>...</f>, CANNOT see x_S
  - Nothing outside <c> attends to <c>

Usage:
    python -m kvmem.train_role --seg-len 32 --N 32 --device mps
    python -m kvmem.train_role --seg-len 128 --N 128 --warmup-len 16 --device mps
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
    SRC_OPEN, SRC_CLOSE, MEM_OPEN, MEM_CLOSE,
    FROM_OPEN, FROM_CLOSE, CONT_OPEN, CONT_CLOSE,
    SRC_OPEN_LEN, SRC_CLOSE_LEN,
    MEM_OPEN_LEN, MEM_CLOSE_LEN,
    FROM_OPEN_LEN, FROM_CLOSE_LEN,
    CONT_OPEN_LEN, CONT_CLOSE_LEN,
    ROLE_OVERHEAD, make_mask_role, make_slot_ids_tag,
)
from kvmem.utils import make_test_sequences, cer
from kvmem.gen_dataset import _sample_seg


# ---------------------------------------------------------------------------
# OCD helpers (Optimal Completion Distillation, arXiv:1810.01398)
# Pure numpy — runs outside autograd graph, before the loss forward pass.
# ---------------------------------------------------------------------------

def _ocd_next_tokens(y_gen: list[int], x_ref: list[int],
                     vocab_size: int = 256) -> np.ndarray:
    """
    Uniform distribution over next tokens that minimise edit distance from
    y_gen to x_ref, given already-generated prefix y_gen.

    Fast path: if y_gen == x_ref[:k], only x_ref[k] is optimal.
    General path: scan all alignment offsets j, pick minimum-cost x_ref[j].
    """
    k = len(y_gen)
    L = len(x_ref)
    dist = np.zeros(vocab_size, dtype=np.float32)
    if L == 0 or k >= L:
        return dist
    if y_gen == x_ref[:k]:
        dist[x_ref[k]] = 1.0
        return dist
    costs = np.empty(L, dtype=np.int32)
    for j in range(L):
        overlap = min(k, j)
        hamm    = sum(y_gen[i] != x_ref[i] for i in range(overlap))
        costs[j] = hamm + abs(k - j)
    min_cost = int(costs.min())
    opts: set[int] = {x_ref[j] for j in range(L) if costs[j] == min_cost}
    p = 1.0 / len(opts)
    for tok in opts:
        dist[tok] = p
    return dist


@torch.no_grad()
def ocd_rollout_role_batch(model, tokens_batch: np.ndarray,
                            pos: dict, refs: list[list[int]],
                            mask_t: torch.Tensor, device
                            ) -> tuple[torch.Tensor, np.ndarray]:
    """
    Batched AR rollout of the <c> region. One forward pass per generation step
    (not one per example) — B examples run in parallel each step.

    tokens_batch : (B, L) int64 — full sequences, <c> region zero-initialised
    pos          : role_positions() dict
    refs         : list of B reference byte lists (x_S[y_start:y_end] per example)
    Returns:
        tok_t       : (B, L) tensor — <c> region filled with AR-generated tokens
        ocd_targets : (B, out_len, 256) float32 — soft OCD targets per step
    """
    out_len = pos['c1'] - pos['c0']
    c0      = pos['c0']
    B       = tokens_batch.shape[0]

    tok_t       = torch.tensor(tokens_batch, dtype=torch.long, device=device)
    ocd_targets = np.zeros((B, out_len, 256), dtype=np.float32)
    y_gens      = [[] for _ in range(B)]

    for k in range(out_len):
        logits = model(tok_t, mask_t)                           # (B, L, V)
        nbs    = logits[:, c0 + k - 1].argmax(-1).cpu().numpy() # (B,)
        for b in range(B):
            ocd_targets[b, k] = _ocd_next_tokens(y_gens[b], refs[b])
            y_gens[b].append(int(nbs[b]))
        tok_t[:, c0 + k] = torch.from_numpy(nbs).to(device)

    return tok_t, ocd_targets


# ---------------------------------------------------------------------------
# Sequence layout helpers
# ---------------------------------------------------------------------------

def role_positions(seg_len: int, N: int, warmup_len: int, out_len: int):
    """Return absolute start positions of each region."""
    s0  = SRC_OPEN_LEN                                # x_S start
    s1  = s0 + seg_len                                # x_S end
    sc1 = s1 + SRC_CLOSE_LEN                          # </s> end
    mo1 = sc1 + MEM_OPEN_LEN                          # <m> end
    sl0 = mo1                                          # slots start
    sl1 = sl0 + N                                      # slots end
    mc1 = sl1 + MEM_CLOSE_LEN                          # </m> end
    fo1 = mc1 + FROM_OPEN_LEN                          # <f> end
    f0  = fo1                                          # warmup start
    f1  = f0 + warmup_len                              # warmup end
    fc1 = f1 + FROM_CLOSE_LEN                          # </f> end
    co1 = fc1 + CONT_OPEN_LEN                          # <c> end
    c0  = co1                                          # output start
    c1  = c0 + out_len                                 # output end
    L   = c1 + CONT_CLOSE_LEN                          # total
    return dict(s0=s0, s1=s1, sl0=sl0, sl1=sl1,
                f0=f0, f1=f1, fc1=fc1, c0=c0, c1=c1, L=L)


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------

def make_role_batch(rng: np.random.Generator, B: int,
                    seg_len: int, N: int, slot_style: str,
                    warmup_len: int, out_len: int,
                    drop_close_prob: float = 0.5) -> np.ndarray:
    """
    drop_close_prob: probability of omitting </c> from each example.
    When </c> is dropped, the <c> region is open-ended — model learns
    it CAN extrapolate beyond the window, not just stop at out_len.
    """
    """
    Build one role-tag batch.

    For each example:
      - sample random source x_S
      - sample random y_start in [0, seg_len - out_len]
      - warmup = x_S[max(0, y_start-warmup_len):y_start]  (padded at start)
      - output = x_S[y_start : y_start+out_len]
    """
    pos      = role_positions(seg_len, N, warmup_len, out_len)
    L        = pos['L']
    slot_ids = make_slot_ids_tag(N, slot_style)
    out      = np.zeros((B, L), dtype=np.int64)
    n_win    = max(1, seg_len - out_len)

    for i in range(B):
        seg     = _sample_seg(rng, seg_len)
        y_start = int(rng.integers(0, n_win + 1))
        y_end   = min(y_start + out_len, seg_len)

        # warmup: last warmup_len bytes before y_start
        w_st  = max(0, y_start - warmup_len)
        wm    = seg[w_st:y_start]
        if len(wm) < warmup_len:
            wm = np.concatenate([np.full(warmup_len - len(wm), seg[0], dtype=np.int32), wm])

        out[i, :SRC_OPEN_LEN]                     = SRC_OPEN
        out[i, pos['s0']:pos['s1']]                = seg
        out[i, pos['s1']:pos['s1']+SRC_CLOSE_LEN]  = SRC_CLOSE
        s1c = pos['s1'] + SRC_CLOSE_LEN
        out[i, s1c:s1c+MEM_OPEN_LEN]               = MEM_OPEN
        out[i, pos['sl0']:pos['sl1']]               = slot_ids
        out[i, pos['sl1']:pos['sl1']+MEM_CLOSE_LEN] = MEM_CLOSE
        sl1c = pos['sl1'] + MEM_CLOSE_LEN
        out[i, sl1c:sl1c+FROM_OPEN_LEN]             = FROM_OPEN
        out[i, pos['f0']:pos['f1']]                 = wm
        out[i, pos['f1']:pos['f1']+FROM_CLOSE_LEN]  = FROM_CLOSE
        f1c = pos['f1'] + FROM_CLOSE_LEN
        out[i, f1c:f1c+CONT_OPEN_LEN]               = CONT_OPEN
        out[i, pos['c0']:pos['c0']+(y_end-y_start)] = seg[y_start:y_end]
        # Randomly drop </c> to teach open-ended continuation
        if rng.random() >= drop_close_prob:
            out[i, pos['c1']:pos['c1']+CONT_CLOSE_LEN] = CONT_CLOSE
        # else: leave as zeros — model sees open <c> with no close tag

    return out


# ---------------------------------------------------------------------------
# AR decode (windowed)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ar_decode_role(model, x_S: list[int], N: int, slot_style: str,
                   warmup: list[int], out_len: int, device) -> list[int]:
    """
    Greedy AR decode using role-tag scheme.
    warmup: warmup_len bytes (the <f> anchor)
    Returns: out_len generated bytes
    """
    seg_len    = len(x_S)
    wl         = len(warmup)
    slot_ids   = make_slot_ids_tag(N, slot_style)
    pos        = role_positions(seg_len, N, wl, out_len)
    L          = pos['L']
    mask_t     = torch.tensor(make_mask_role(seg_len, N, wl, out_len),
                               dtype=torch.float32, device=device)

    tokens = np.zeros(L, dtype=np.int64)
    tokens[:SRC_OPEN_LEN]                    = SRC_OPEN
    tokens[pos['s0']:pos['s1']]              = x_S
    tokens[pos['s1']:pos['s1']+SRC_CLOSE_LEN] = SRC_CLOSE
    s1c = pos['s1'] + SRC_CLOSE_LEN
    tokens[s1c:s1c+MEM_OPEN_LEN]             = MEM_OPEN
    tokens[pos['sl0']:pos['sl1']]            = slot_ids
    tokens[pos['sl1']:pos['sl1']+MEM_CLOSE_LEN] = MEM_CLOSE
    sl1c = pos['sl1'] + MEM_CLOSE_LEN
    tokens[sl1c:sl1c+FROM_OPEN_LEN]          = FROM_OPEN
    tokens[pos['f0']:pos['f1']]              = warmup
    tokens[pos['f1']:pos['f1']+FROM_CLOSE_LEN] = FROM_CLOSE
    f1c = pos['f1'] + FROM_CLOSE_LEN
    tokens[f1c:f1c+CONT_OPEN_LEN]            = CONT_OPEN
    # <c> region starts at pos['c0'], generate token by token

    tok_t   = torch.tensor(tokens, dtype=torch.long, device=device)
    generated = []
    for k in range(out_len):
        pos_k = pos['c0'] + k
        logits = model(tok_t, mask_t)
        nb = int(logits[pos_k - 1].argmax())
        generated.append(nb)
        tok_t[pos_k] = nb

    return generated


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_role(hp: dict, log_base: str = 'logs', device_str: str = 'cpu'):
    device   = torch.device(device_str)
    lr_max   = hp['lr_max']
    wd       = hp['wd']
    eval_every = hp['eval_every']
    log_every  = hp['log_every']
    slot_style      = hp['slot_style']
    drop_close_prob = hp['drop_close_prob']
    warmup_steps = hp['warmup_steps']
    cycle_steps  = hp['cycle_steps']
    seed       = hp['seed']
    use_ocd      = hp['ocd']
    ocd_mode     = hp['ocd_mode']
    ocd_every    = hp['ocd_every']
    ocd_prob     = hp.get('ocd_prob', 1.0 / ocd_every)
    tf_warmup    = hp['tf_warmup']
    grad_clip    = hp['grad_clip']
    dataset_size = hp['dataset_size']

    # Curriculum stages: list of (seg_len, warmup_len, out_len, B, n_steps)
    # If not specified, single stage from hp
    curriculum = hp.get('curriculum', [{
        'seg_len': hp['seg_len'], 'N': hp.get('N', hp['seg_len']),
        'warmup_len': hp.get('warmup_len', 16),
        'out_len': hp.get('out_len', 32),
        'B': hp['B'], 'n_steps': hp['n_steps'],
    }])

    ts   = datetime.now().strftime('%m%d_%H%M')
    # ts   = datetime.now().strftime('%m%d_%H%M')
    name = hp.get('name')
    if name:
        suffix = f'{name}_{ts}' if hp.get('name_date') else name
    else:
        suffix = ts
    run_dir = os.path.join(log_base, f'role_{suffix}')
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'hparams.json'), 'w') as f:
        json.dump(hp, f, indent=2)
    log_f  = open(os.path.join(run_dir, 'train.log'),   'w', buffering=1)
    jlog_f = open(os.path.join(run_dir, 'train.jsonl'), 'w', buffering=1)

    def _log(m): tqdm.write(m); log_f.write(m + '\n')
    def _jlog(d): jlog_f.write(json.dumps(d) + '\n')

    # Build model with max stage dimensions for YaRN L_max
    max_stage = max(curriculum, key=lambda s: s['seg_len'])
    max_seg   = max_stage['seg_len']
    L_max_seq = ROLE_OVERHEAD + max_seg + max_stage.get('N', max_seg) + \
                max_stage.get('warmup_len', 32) + max_stage.get('out_len', 128)
    hp_model  = dict(hp, seg_len=max_seg, N=max_stage.get('N', max_seg),
                     L_train=L_max_seq, L_max=L_max_seq * 4)

    torch.manual_seed(seed)
    model = build_model(hp_model, device)
    if hp.get('compile', False):
        model = torch.compile(model)

    if hp['grok']:
        from kvmem.optim import GrokAdamW
        opt = GrokAdamW(model.parameters(), lr=lr_max, weight_decay=wd,
                        rho=hp.get('grok_rho', 0.9), batch_size=curriculum[0]['B'])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=wd)

    _log(f'\n=== Role-Tag Curriculum | run_dir={run_dir} ===')
    _log(f'  cmd: {" ".join(sys.argv)}')
    params = sum(p.numel() for p in model.parameters())
    _log(f'  Model: d={hp["d"]}  n_layers={hp["n_layers"]}  params={params:,}  device={device}')
    _log(f'  rope={hp.get("rope",False)}  yarn={hp.get("yarn",False)}  drop_close={drop_close_prob}'
         + (f'  OCD mode={ocd_mode} every={ocd_every} tf_warmup={tf_warmup}'
            if use_ocd else '  TF-only'))
    _log(f'  Curriculum: {len(curriculum)} stages')
    for i, st in enumerate(curriculum):
        L_st = role_positions(st['seg_len'], st.get('N', st['seg_len']),
                              st.get('warmup_len',16), st.get('out_len',32))['L']
        _log(f'    stage {i}: seg={st["seg_len"]}  wl={st.get("warmup_len",16)}'
             f'  out={st.get("out_len",32)}  B={st["B"]}  steps={st["n_steps"]}  L={L_st}')

    rng = np.random.default_rng(seed)
    t0  = time.time()
    global_step = 0
    print(curriculum)
    for stage_i, stage in enumerate(curriculum):
        seg_len    = stage['seg_len']
        N          = stage.get('N', seg_len)
        warmup_len = stage.get('warmup_len', 16)
        out_len    = stage.get('out_len', 32)
        B          = stage['B']
        n_steps    = stage['n_steps']
        # Per-stage cosine cycle: stage dict overrides global cycle_steps.
        # Default: decay over the full stage so LR reaches lr_min at end.
        stage_cycle = stage.get('cycle_steps', cycle_steps)

        def lr_schedule(local, _cycle=stage_cycle):
            if warmup_steps > 0 and local < warmup_steps:
                return lr_max * max(local, 1) / warmup_steps
            if _cycle <= 0:
                return lr_max
            t = (local - warmup_steps) % _cycle
            return max(lr_max * 1e-2, lr_max * 0.5 * (1 + math.cos(math.pi * t / _cycle)))

        pos     = role_positions(seg_len, N, warmup_len, out_len)
        L_total = pos['L']
        c0, c1  = pos['c0'], pos['c1']
        mask_t  = torch.tensor(make_mask_role(seg_len, N, warmup_len, out_len),
                                dtype=torch.float32, device=device)
        test_seqs = make_test_sequences(seg_len)
        val_np    = make_role_batch(np.random.default_rng(seed + stage_i + 1),
                                    B, seg_len, N, slot_style, warmup_len, out_len,
                                    drop_close_prob=0.0)
        pool_rng  = np.random.default_rng(seed + stage_i + 1000)
        ds        = dataset_size if dataset_size > 0 else None
        pool      = (np.stack([make_role_batch(pool_rng, B, seg_len, N, slot_style,
                                               warmup_len, out_len, drop_close_prob)
                                for _ in range(ds)])
                     if ds else None)
        _log(f'  dataset: {"infinite stream" if ds is None else f"{ds} batches ({ds*B} examples), {n_steps//ds if ds else 0} epochs"}')

        if hp['grok']:
            for pg in opt.param_groups:
                pg['batch_size'] = B
        _log(f'\n{"="*60}')
        _log(f'  [stage {stage_i}] seg={seg_len}  warmup={warmup_len}  out={out_len}  B={B}  steps={n_steps}')
        _log(f'  Format: <s>x_S({seg_len})</s><m>slots({N})</m><f>wl({warmup_len})</f><c>out({out_len})</c>  L={L_total}')

        pbar = tqdm(range(1, n_steps + 1), desc=f'stage{stage_i}', dynamic_ncols=True)

        for local_step in pbar:
            global_step += 1
            lr = lr_schedule(local_step)
            for pg in opt.param_groups:
                pg['lr'] = lr

            model.train()
            tokens_np = (pool[(local_step - 1) % ds]
                         if pool is not None
                         else make_role_batch(rng, B, seg_len, N, slot_style,
                                              warmup_len, out_len, drop_close_prob))

            if use_ocd and global_step > tf_warmup:
                if ocd_mode == 'every':
                    do_ocd = (global_step % ocd_every == 0)
                else:  # 'prob'
                    do_ocd = (rng.random() < ocd_prob)
            else:
                do_ocd = False

            if do_ocd:
                # OCD: AR rollout <c> region (batched — out_len steps, B examples each)
                # Extract per-example refs from the teacher-forced batch
                refs = [list(tokens_np[b, c0:c1]) for b in range(B)]
                tokens_np_zeroc = tokens_np.copy()
                tokens_np_zeroc[:, c0:c1] = 0   # zero out <c> for rollout
                model.eval()
                tok_ocd, ocd_tgts = ocd_rollout_role_batch(
                    model, tokens_np_zeroc, pos, refs, mask_t, device)
                # ocd_tgts: (B, out_len, 256) — soft targets for <c> positions
                ocd_t = torch.tensor(ocd_tgts, device=device)

                model.train()
                opt.zero_grad()
                logits = model(tok_ocd, mask_t)                          # (B, L, V)
                # logits[:, c0-1 : c1-1] predicts tokens at positions c0..c1-1
                lp     = F.log_softmax(logits[:, c0-1:c1-1], dim=-1)    # (B, out_len, V)
                loss_val = -(ocd_t * lp).sum(-1).mean()
                mode = 'ocd'
            else:
                # Teacher-forcing (fast path)
                tokens = torch.tensor(tokens_np, device=device)
                opt.zero_grad()
                logits  = model(tokens, mask_t)                          # (B, L, V)
                lp      = F.log_softmax(logits[:, :-1], dim=-1)         # (B, L-1, V)
                tgts    = tokens[:, 1:]                                  # (B, L-1)
                nll     = -lp.gather(2, tgts.unsqueeze(-1)).squeeze(-1) # (B, L-1)
                mask_y  = torch.zeros(L_total - 1, device=device)
                mask_y[c0:c1] = 1.0
                loss_val = (nll * mask_y).sum(1).mean() / (c1 - c0)
                mode = 'tf'

            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            loss_f = float(loss_val.detach())
            pbar.set_postfix(loss=f'{loss_f:.3f}', lr=f'{lr:.1e}', mode=mode, refresh=False)
            if local_step % log_every == 0:
                bpb = loss_f / math.log(2)
                _jlog(dict(global_step=global_step, stage=stage_i, loss=loss_f, bpb=bpb, lr=lr, mode=mode))
                log_f.write(str(pbar) + '\n')
                print()

            if local_step % eval_every == 0 or local_step == 1:
                model.eval()
                with torch.no_grad():
                    val_tok = torch.tensor(val_np, device=device)
                    val_lp  = F.log_softmax(model(val_tok, mask_t)[:, :-1], dim=-1)
                    val_nll = -val_lp.gather(2, val_tok[:, 1:].unsqueeze(-1)).squeeze(-1)
                    val_mask = torch.zeros(L_total - 1, device=device)
                    val_mask[c0:c1] = 1.0
                    val_loss = float((val_nll * val_mask).sum(1).mean() / (c1 - c0))
                    val_bpb  = val_loss / math.log(2)
                elapsed = time.time() - t0
                _log(f'\n--- stage={stage_i} step={local_step}/{n_steps}  g={global_step}'
                     f'  loss={loss_f:.4f}  val_loss={val_loss:.4f}  val_bpb={val_bpb:.3f}  lr={lr:.2e}  {elapsed:.0f}s ---')
                all_cer = []
                for name, x_S in test_seqs.items():
                    y_start  = seg_len // 4
                    y_end    = min(y_start + out_len, seg_len)
                    warmup   = x_S[max(0, y_start - warmup_len):y_start]
                    if len(warmup) < warmup_len:
                        warmup = [x_S[0]] * (warmup_len - len(warmup)) + list(warmup)
                    target   = x_S[y_start:y_end]
                    gen      = ar_decode_role(model, x_S, N, slot_style,
                                              warmup, len(target), device)
                    c = cer(gen, target)
                    all_cer.append(c)
                    ok = '✓' if c == 0.0 else '✗'
                    _log(f'  {ok} {name:15s} [{y_start}:{y_end}]  match={100*(1-c):5.1f}%  CER={c:.3f}')
                    _log(f'    gen={bytes(gen).hex()}')
                    _log(f'    ref={bytes(target).hex()}')

                mean_cer = sum(all_cer) / len(all_cer)
                _log(f'  → mean CER={mean_cer:.3f}  match={100*(1-mean_cer):.1f}%')
                _jlog(dict(global_step=global_step, stage=stage_i, loss=loss_f,
                           val_loss=val_loss, val_bpb=val_bpb, mean_cer=mean_cer))

                if mean_cer == 0.0:
                    _log(f'\n★ PERFECT at stage {stage_i} local_step={local_step}!')
                    ckpt = os.path.join(ckpt_dir, f'stage{stage_i}_step{local_step}.pt')
                    torch.save({'model': model.state_dict(), 'hp': hp,
                                'stage': stage_i, 'step': local_step}, ckpt)
                    _log(f'  [ckpt] {ckpt}')
                    break


        # Save checkpoint at end of each stage
        ckpt = os.path.join(ckpt_dir, f'stage{stage_i}_end.pt')
        torch.save({'model': model.state_dict(), 'hp': hp, 'stage': stage_i}, ckpt)
        _log(f'  [ckpt stage {stage_i} end] {ckpt}')

    _log(f'\nDone. Total: {time.time()-t0:.0f}s')
    log_f.close(); jlog_f.close()
    return model, run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CURRICULUM_SURAH = [
    dict(seg_len= 32,  N= 32,  warmup_len= 8,  out_len= 8,   B=16, n_steps=40000),
    dict(seg_len= 64,  N= 64,  warmup_len=16,  out_len=16,   B=16, n_steps=40000),
    dict(seg_len=128,  N=128,  warmup_len=32,  out_len=32,   B= 8, n_steps=40000),
    dict(seg_len=256,  N=256,  warmup_len=32,  out_len=64,   B= 4, n_steps=20000),
    dict(seg_len=576,  N=576,  warmup_len=32,  out_len=128,  B= 4, n_steps=20000),
]

# V2: 100k steps for stages 3-4, per-stage cosine LR decay to stabilize grokking.
# Stages 0-2 are unchanged (peaked fine at 40k); 3-4 were oscillating.
CURRICULUM_SURAH_V2 = [
    dict(seg_len= 32,  N= 32,  warmup_len= 8,  out_len= 8,   B=16, n_steps=20000, cycle_steps=20000),
    dict(seg_len= 64,  N= 64,  warmup_len=16,  out_len=16,   B=16, n_steps=20000, cycle_steps=20000),
    dict(seg_len=128,  N=128,  warmup_len=32,  out_len=32,   B= 8, n_steps=20000, cycle_steps=20000),
    dict(seg_len=256,  N=256,  warmup_len=32,  out_len=64,   B= 4, n_steps=40000, cycle_steps=40000),
    dict(seg_len=576,  N=576,  warmup_len=32,  out_len=128,  B= 4, n_steps=40000, cycle_steps=40000),
]

# CURRICULUM_SURAH_V2 = [
    # dict(seg_len= 32,  N= 32,  warmup_len= 8,  out_len= 8,   B=4, n_steps=20000),
    # dict(seg_len= 64,  N= 64,  warmup_len=16,  out_len=16,   B=4, n_steps=20000),
    # dict(seg_len=128,  N=128,  warmup_len=32,  out_len=32,   B=4, n_steps=40000),
    # dict(seg_len=256,  N=256,  warmup_len=32,  out_len=64,   B=4, n_steps=40000),
    # dict(seg_len=576,  N=576,  warmup_len=32,  out_len=128,  B=4, n_steps=40000),
# ]

DEFAULTS = dict(
    seg_len=32, N=32, V=256, d=64, n_layers=4, n_heads=4, d_ff=256,
    B=64, lr_max=3e-4, wd=0.01, warmup_steps=200, cycle_steps=0,
    n_steps=10000, eval_every=2000, log_every=1000,
    slot_style='seq', warmup_len=8, out_len=8,
    rope=True, yarn=True, grok=True, seed=42,
    drop_close_prob=0.5,
    ocd=False, ocd_mode='every', ocd_every=10, ocd_prob=0.1,
    tf_warmup=0, grad_clip=1.0, dataset_size=5000,
    compile=False,
    curriculum=CURRICULUM_SURAH,
)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seg-len',     type=int)
    p.add_argument('--slot-len',    type=int)
    p.add_argument('--d',           type=int)
    p.add_argument('--n-layers',    type=int)
    p.add_argument('--B',           type=int)
    p.add_argument('--lr',          type=float)
    p.add_argument('--steps',       type=int)
    p.add_argument('--eval-every',  type=int)
    p.add_argument('--log-every',   type=int)
    p.add_argument('--warmup-len',  type=int)
    p.add_argument('--out-len',     type=int)
    p.add_argument('--warmup-steps', type=int)
    p.add_argument('--cycle-steps', type=int)
    p.add_argument('--slot-style',  type=str)
    p.add_argument('--drop-close',  type=float,
                   help='Prob of dropping </c> per example (default 0.5, enables extrapolation)')
    p.add_argument('--ocd',         action='store_true',
                   help='Use Optimal Completion Distillation in <c> region')
    p.add_argument('--ocd-mode',    type=str,   choices=['every', 'prob'],
                   help='OCD schedule: every=deterministic K, prob=stochastic 1/K')
    p.add_argument('--ocd-every',   type=int)
    p.add_argument('--ocd-prob',    type=float)
    p.add_argument('--tf-warmup',   type=int,
                   help='Pure teacher-forcing steps before OCD starts')
    p.add_argument('--grad-clip',   type=float)
    p.add_argument('--dataset-size', type=int,  help='Fixed pool size in batches (0=infinite stream, default=1000)')
    p.add_argument('--no-grok',     action='store_true')
    p.add_argument('--compile',     action='store_true')
    p.add_argument('--name',        type=str,   help='Run folder name (replaces timestamp; use --name-date to append it)')
    p.add_argument('--name-date',   action='store_true', help='Append timestamp after --name')
    p.add_argument('--log-dir',     type=str,   default='logs')
    p.add_argument('--device',      type=str,   default='cpu',
                   choices=['cpu', 'mps', 'cuda'])
    p.add_argument('--seed',        type=int)
    p.add_argument('--curriculum',  type=str,   default='v1',
                   choices=['v1', 'v2', 'none'],
                   help='v1=CURRICULUM_SURAH, v2=100k+cosine, none=single stage from CLI args')
    args = p.parse_args()

    hp = dict(DEFAULTS)
    if args.curriculum == 'v2':
        hp['curriculum'] = CURRICULUM_SURAH_V2
    elif args.curriculum == 'none':
        hp['curriculum'] = None   # resolved below after other args are applied

    # Simple 1:1 overrides
    for src, dst in [
        ('slot_len', 'N'), ('n_layers', 'n_layers'), ('B', 'B'),
        ('steps', 'n_steps'), ('eval_every', 'eval_every'), ('log_every', 'log_every'),
        ('warmup_len', 'warmup_len'), ('out_len', 'out_len'),
        ('warmup_steps', 'warmup_steps'), ('cycle_steps', 'cycle_steps'),
        ('slot_style', 'slot_style'), ('seed', 'seed'),
        ('ocd_mode', 'ocd_mode'), ('ocd_every', 'ocd_every'), ('ocd_prob', 'ocd_prob'),
        ('tf_warmup', 'tf_warmup'), ('grad_clip', 'grad_clip'),
        ('dataset_size', 'dataset_size'),
    ]:
        v = getattr(args, src, None)
        if v is not None:
            hp[dst] = v

    # Compound / renamed args
    if args.seg_len:               hp['seg_len'] = args.seg_len; hp['N'] = args.seg_len
    if args.d:                     hp['d'] = args.d; hp['d_ff'] = args.d * 4
    if args.lr:                    hp['lr_max'] = args.lr
    if args.drop_close is not None: hp['drop_close_prob'] = args.drop_close
    if args.ocd:                   hp['ocd'] = True
    if args.name:                  hp['name'] = args.name
    if args.name_date:             hp['name_date'] = True
    if args.no_grok:               hp['grok'] = False
    if args.compile:               hp['compile'] = True

    seg = hp['seg_len']; N = hp['N']
    hp['L_train'] = ROLE_OVERHEAD + seg + N + hp['warmup_len'] + hp['out_len']
    hp['L_max']   = hp['L_train'] * 4

    if hp.get('curriculum') is None:
        hp['curriculum'] = [dict(
            seg_len=seg, N=N,
            warmup_len=hp['warmup_len'], out_len=hp['out_len'],
            B=hp['B'], n_steps=hp['n_steps'],
            cycle_steps=hp.get('cycle_steps', hp['n_steps']),
        )]

    train_role(hp, log_base=args.log_dir, device_str=args.device)
