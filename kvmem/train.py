"""
kvmem/train.py — Role-tag KV-memory recall training.

Sequence layout (memory-first, n_blocks=1):
  <m> slots </m> <s> x_S </s> <f> warmup </f> <c> output </c>

Memory tokens come FIRST — like an RNN hidden state in token form.
Slots read src NON-CAUSALLY (mask explicitly allows slots → src even though
src comes after slots in the sequence).

Multi-block (n_blocks=N, recall_from=K):
  <m>slots_0</m><s>src_0</s> ... <m>slots_{N-1}</m><s>src_{N-1}</s>
  <f>anchor_from_block_K</f><c>output_from_block_K</c>

  n_blocks=1 is the single-block degenerate case.

Training mode:
  Full-pass TF (default): single SDPA over full sequence — exact gradients.
  --grad-checkpoint: recompute each block in backward (depth-only).
    Saves residual activations at cost of one extra fwd pass per block.
    Does NOT reduce the O(L²) attention matrix within each block.

KV-cache training (kvmem.kvcache module):
  Blockwise two-pass matching inference computation. Exact forward values,
  different float32 backward rounding → different local optima.
  Import blockwise_tf_loss / ocd_rollout_kvcache from kvmem.kvcache.

Config DSL:
  --config path/to/config.py  loads a module-level `hp` dict, merged over DEFAULTS.
  CLI args override loaded config.

Usage:
  python -m kvmem.train --config configs/single_s16.py --device mps
  python -m kvmem.train --config configs/ablate_t1.py  --device mps
  python -m kvmem.train --config configs/single_s16.py --grad-checkpoint --device mps
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
from kvmem.data import (
    HIDDEN_OPEN, HIDDEN_CLOSE, INPUT_OPEN, INPUT_CLOSE,
    QUERY_OPEN, QUERY_CLOSE, OUTPUT_OPEN, OUTPUT_CLOSE,
    INTERMED_OPEN, INTERMED_CLOSE,
    HIDDEN_OPEN_LEN, HIDDEN_CLOSE_LEN,
    INPUT_OPEN_LEN, INPUT_CLOSE_LEN,
    QUERY_OPEN_LEN, QUERY_CLOSE_LEN,
    OUTPUT_OPEN_LEN, OUTPUT_CLOSE_LEN,
    INTERMED_OPEN_LEN, INTERMED_CLOSE_LEN,
    multi_block_positions, make_mask_multi, make_multi_batch,
)
from kvmem.utils import make_test_sequences, cer


# ---------------------------------------------------------------------------
# OCD helper
# ---------------------------------------------------------------------------

def _ocd_next_tokens(y_gen: list[int], x_ref: list[int],
                     vocab_size: int = 256) -> np.ndarray:
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
def ocd_rollout_full(model, tokens_batch: np.ndarray,
                     pos: dict, refs: list[list[int]],
                     mask_t: torch.Tensor, device
                     ) -> tuple[torch.Tensor, np.ndarray]:
    """Batched AR rollout using full forward passes (one per generation step)."""
    out_len = pos['c1'] - pos['c0']
    c0      = pos['c0']
    B       = tokens_batch.shape[0]
    tok_t       = torch.tensor(tokens_batch, dtype=torch.long, device=device)
    ocd_targets = np.zeros((B, out_len, 256), dtype=np.float32)  # output vocab = 256
    y_gens      = [[] for _ in range(B)]
    for k in range(out_len):
        logits = model(tok_t, mask_t)
        nbs    = logits[:, c0 + k - 1].argmax(-1).cpu().numpy()
        for b in range(B):
            ocd_targets[b, k] = _ocd_next_tokens(y_gens[b], refs[b])
            y_gens[b].append(int(nbs[b]))
        tok_t[:, c0 + k] = torch.from_numpy(nbs).to(device)
    return tok_t, ocd_targets


# ---------------------------------------------------------------------------
# AR decode (eval — single-block only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ar_decode_role(model, x_S: list[int], slot_len: int,
                   warmup: list[int], out_len: int, device,
                   n_blocks: int = 1, recall_from: int = 0,
                   intermed_len: int = 0, mem_window: int = 0,
                   distractor_seed: int = 99) -> list[int]:
    """
    Greedy AR decode (eval). Builds a n_blocks sequence where x_S is placed
    in block recall_from; other blocks are filled with random distractors.

    For n_blocks=1: single-block eval (original behaviour).
    For n_blocks>1: correct multi-block eval — distractor blocks test that
    the model routes <q> to the right block from anchor content alone.
    """
    from kvmem.data import make_hidden_slot_ids, make_intermed_slot_ids, _sample_seg
    seg_len    = len(x_S)
    wl         = len(warmup)
    slot_ids   = make_hidden_slot_ids(slot_len)
    intermed_ids = make_intermed_slot_ids(intermed_len, slot_len) if intermed_len > 0 else []
    pos        = multi_block_positions(n_blocks, seg_len, slot_len, wl, out_len, intermed_len)
    L          = pos['L']
    mask_t     = torch.tensor(
        make_mask_multi(n_blocks, seg_len, slot_len, wl, out_len, intermed_len, mem_window),
        dtype=torch.float32, device=device)

    rng    = np.random.default_rng(distractor_seed)
    tokens = np.zeros(L, dtype=np.int64)

    for k, b in enumerate(pos['blocks']):
        src = np.array(x_S) if k == recall_from else _sample_seg(rng, seg_len)
        tokens[b['block_start']:b['s0']]         = INPUT_OPEN
        tokens[b['s0']:b['s1']]                  = src
        tokens[b['s1']:b['s_close_end']]         = INPUT_CLOSE
        if intermed_len > 0:
            tokens[b['p_open']:b['p0']]          = INTERMED_OPEN
            tokens[b['p0']:b['p1']]              = intermed_ids
            tokens[b['p1']:b['p_close_end']]     = INTERMED_CLOSE
        tokens[b['p_close_end']:b['sl0']]        = HIDDEN_OPEN
        tokens[b['sl0']:b['sl1']]                = slot_ids
        tokens[b['sl1']:b['mc1']]                = HIDDEN_CLOSE

    rs = pos['recall_start']
    tokens[rs:rs+QUERY_OPEN_LEN]                  = QUERY_OPEN
    tokens[pos['f0']:pos['f1']]                   = warmup
    tokens[pos['f1']:pos['f1']+QUERY_CLOSE_LEN]   = QUERY_CLOSE
    tokens[pos['fc1']:pos['fc1']+OUTPUT_OPEN_LEN]  = OUTPUT_OPEN

    tok_t     = torch.tensor(tokens, dtype=torch.long, device=device)
    generated = []
    for k in range(out_len):
        logits = model(tok_t, mask_t)
        nb = int(logits[pos['c0'] + k - 1].argmax())
        generated.append(nb)
        tok_t[pos['c0'] + k] = nb
    return generated


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _stablemax_log_probs(logits: torch.Tensor) -> torch.Tensor:
    s = torch.where(logits >= 0, logits + 1.0, 1.0 / (1.0 - logits))
    return (s / s.sum(dim=-1, keepdim=True)).log()


def train_role(hp: dict, log_base: str = 'logs', device_str: str = 'cpu'):
    device      = torch.device(device_str)
    lr_max      = hp['lr_max']
    wd          = hp['wd']
    eval_every  = hp['eval_every']
    log_every   = hp['log_every']
    # slot_style removed — learned embeddings use make_mem_slot_ids()
    drop_close_prob = hp['drop_close_prob']
    warmup_steps = hp['warmup_steps']
    cycle_steps  = hp['cycle_steps']
    seed         = hp['seed']
    use_ocd      = hp['ocd']
    _ocd_prob    = hp['ocd_prob']
    grad_clip    = hp['grad_clip']
    dataset_size = hp['dataset_size']
    # active_slots removed — slot_len IS the bottleneck
    log_probs_fn = _stablemax_log_probs if hp['stablemax'] else \
                   lambda x: F.log_softmax(x, dim=-1)
    eval_offset  = hp['eval_offset']

    def _resolve_ocd_prob(local_step):
        if isinstance(_ocd_prob, (int, float)):
            return float(_ocd_prob)
        prob = 0.0
        for threshold, p in sorted(_ocd_prob, key=lambda x: x[0]):
            if local_step >= threshold:
                prob = float(p)
        return prob

    # If hp contains a 'seq' DSL string, derive sequence params from it
    if 'seq' in hp:
        from kvmem.seq_dsl import parse_seq as _parse_seq
        _spec = _parse_seq(hp['seq'])
        for k, v in _spec.to_hp().items():
            if k not in hp or k == 'seq':
                hp[k] = v

    _cur = hp.get('curriculum')
    curriculum = _cur if _cur else [{
        'seg_len':     hp['seg_len'],
        'slot_len':    hp.get('slot_len', hp['seg_len']),
        'warmup_len':  hp.get('warmup_len', 16),
        'intermed_len':  hp.get('intermed_len', 0),
        'mem_window':    hp.get('mem_window', 0),
        'out_len':     hp.get('out_len', 32),
        'n_blocks':    hp.get('n_blocks', 1),
        'recall_from': hp.get('recall_from', 0),
        'B': hp['B'], 'n_steps': hp['n_steps'],
    }]

    ts     = datetime.now().strftime('%m%d_%H%M')
    name   = hp.get('name')
    suffix = f'{name}_{ts}' if (name and hp.get('name_date')) else (name or ts)
    run_dir  = os.path.join(log_base, f'role_{suffix}')
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'hparams.json'), 'w') as f:
        json.dump(hp, f, indent=2, default=str)
    log_f  = open(os.path.join(run_dir, 'train.log'),   'w', buffering=1)
    jlog_f = open(os.path.join(run_dir, 'train.jsonl'), 'w', buffering=1)
    def _log(m): tqdm.write(m); log_f.write(m + '\n')
    _R5 = {'loss', 'bpb', 'val_loss', 'val_bpb', 'lr'}
    def _jlog(d):
        jlog_f.write(json.dumps(
            {k: round(v, 5) if k in _R5 else v for k, v in d.items()}) + '\n')

    max_stage = max(curriculum, key=lambda s: s['seg_len'])
    max_nb    = max_stage.get('n_blocks', 1)
    max_pos   = multi_block_positions(max_nb, max_stage['seg_len'],
                                      max_stage.get('slot_len', max_stage['seg_len']),
                                      max_stage.get('warmup_len', 32),
                                      max_stage.get('out_len', 128),
                                      max_stage.get('intermed_len', 0))
    L_max_seq = max_pos['L']
    hp_model  = dict(hp, seg_len=max_stage['seg_len'],
                     slot_len=max_stage.get('slot_len', max_stage['seg_len']),
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

    gc_flag = '  grad_checkpoint=ON' if hp.get('grad_checkpoint') else ''
    _log(f'\n=== kvmem memory-first | run_dir={run_dir} ===')
    _log(f'  cmd: {" ".join(sys.argv)}')
    params = sum(p.numel() for p in model.parameters())
    _log(f'  Model: d={hp["d"]}  n_layers={hp["n_layers"]}  params={params:,}'
         f'  device={device}{gc_flag}')
    _log(f'  rope={hp.get("rope",False)}  yarn={hp.get("yarn",False)}'
         f'  drop_close={drop_close_prob}  layout=memory-first'
         + (f'  OCD prob={_ocd_prob}' if use_ocd else '  TF-only'))
    _log(f'  Curriculum: {len(curriculum)} stages')
    for i, st in enumerate(curriculum):
        nb = st.get('n_blocks', 1)
        rf = st.get('recall_from', 0)
        p  = multi_block_positions(nb, st['seg_len'],
                                   st.get('slot_len', st['seg_len']),
                                   st.get('warmup_len', 16), st.get('out_len', 32),
                                   st.get('intermed_len', 0))
        pl = st.get('intermed_len', 0)
        _log(f'    stage {i}: n_blocks={nb} recall_from={rf}'
             f'  seg={st["seg_len"]}  slot={st.get("slot_len",st["seg_len"])}'
             f'  wl={st.get("warmup_len",16)}'
             + (f'  ponder={pl}' if pl else '')
             + f'  out={st.get("out_len",32)}'
             + f'  B={st["B"]}  steps={st["n_steps"]}  L={p["L"]}')

    rng  = np.random.default_rng(seed)
    trng = torch.Generator()
    trng.manual_seed(seed)
    t0          = time.time()
    global_step = 0

    for stage_i, stage in enumerate(curriculum):
        seg_len    = stage['seg_len']
        slot_len   = stage.get('slot_len', seg_len)
        warmup_len = stage.get('warmup_len', 16)
        intermed_len = stage.get('intermed_len', 0)
        mem_window   = stage.get('mem_window', 0)
        out_len    = stage.get('out_len', 32)
        n_blocks   = stage.get('n_blocks', 1)
        recall_from = stage.get('recall_from', 0)
        B          = stage['B']
        n_steps    = stage['n_steps']
        stage_cycle = stage.get('cycle_steps', cycle_steps)
        # -1 = cosine over the full stage; 0 = flat LR
        _eff_cycle  = n_steps if stage_cycle == -1 else stage_cycle

        def lr_schedule(local, _cycle=_eff_cycle):
            if warmup_steps > 0 and local < warmup_steps:
                return lr_max * max(local, 1) / warmup_steps
            if _cycle <= 0:
                return lr_max
            t = (local - warmup_steps) % _cycle
            return max(lr_max * 1e-2, lr_max * 0.5 * (1 + math.cos(math.pi * t / _cycle)))

        pos     = multi_block_positions(n_blocks, seg_len, slot_len, warmup_len, out_len,
                                        intermed_len)
        L_total = pos['L']
        c0, c1  = pos['c0'], pos['c1']
        mask_t  = torch.tensor(
            make_mask_multi(n_blocks, seg_len, slot_len, warmup_len, out_len,
                            intermed_len, mem_window),
            dtype=torch.float32, device=device)

        test_seqs = make_test_sequences(seg_len)
        val_np    = make_multi_batch(
            np.random.default_rng(seed + stage_i + 1),
            B, n_blocks, recall_from, seg_len, slot_len,
            warmup_len, out_len, drop_close_prob=0.0, intermed_len=intermed_len)
        pool_rng  = np.random.default_rng(seed + stage_i + 1000)
        ds   = dataset_size if dataset_size > 0 else None
        pool = (np.stack([make_multi_batch(pool_rng, B, n_blocks, recall_from,
                                           seg_len, slot_len,
                                           warmup_len, out_len, drop_close_prob,
                                           intermed_len)
                          for _ in range(ds)])
                if ds else None)
        _log(f'  dataset: {"infinite" if ds is None else f"{ds} batches ({ds*B} examples)"}')

        if hp['grok']:
            for pg in opt.param_groups:
                pg['batch_size'] = B
        _log(f'\n{"="*60}')
        _log(f'  [stage {stage_i}] n_blocks={n_blocks} recall_from={recall_from}'
             f'  seg={seg_len}  slot={slot_len}  wl={warmup_len}  out={out_len}'
             f'  B={B}  steps={n_steps}  L={L_total}')

        pbar = tqdm(range(1, n_steps + 1), desc=f'stage{stage_i}', dynamic_ncols=True)

        for local_step in pbar:
            global_step += 1
            lr = lr_schedule(local_step)
            for pg in opt.param_groups:
                pg['lr'] = lr

            model.train()
            tokens_np = (pool[(local_step - 1) % ds]
                         if pool is not None
                         else make_multi_batch(rng, B, n_blocks, recall_from,
                                               seg_len, slot_len,
                                               warmup_len, out_len, drop_close_prob,
                                               intermed_len))

            if use_ocd:
                _p     = _resolve_ocd_prob(local_step)
                do_ocd = (_p > 0.0 and torch.rand(1, generator=trng).item() < _p)
            else:
                do_ocd = False

            if do_ocd:
                refs = [list(tokens_np[b, c0:c1]) for b in range(B)]
                tokens_np_zeroc = tokens_np.copy()
                tokens_np_zeroc[:, c0:c1] = 0
                model.eval()
                tok_ocd, ocd_tgts = ocd_rollout_full(
                    model, tokens_np_zeroc, pos, refs, mask_t, device)
                ocd_t = torch.tensor(ocd_tgts, device=device)
                model.train()
                opt.zero_grad()
                logits   = model(tok_ocd, mask_t)
                lp       = log_probs_fn(logits[:, c0-1:c1-1])
                loss_val = -(ocd_t * lp).sum(-1).mean()
                mode = 'ocd'
            else:
                # Full-pass TF — exact gradients
                # Supervise only <c> positions; targets there are always data bytes (0-255).
                # V_out=256 so we gather only on the output region to avoid out-of-range
                # indices from special token IDs (256+) in other sequence positions.
                tokens   = torch.tensor(tokens_np, device=device)
                opt.zero_grad()
                logits   = model(tokens, mask_t)                    # (B, L, 256)
                lp_c     = log_probs_fn(logits[:, c0-1:c1-1])       # (B, out_len, 256)
                tgts_c   = tokens[:, c0:c1]                         # (B, out_len) in [0,255]
                nll_c    = -lp_c.gather(2, tgts_c.unsqueeze(-1)).squeeze(-1)
                loss_val = nll_c.mean()
                mode = 'tf'

            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            loss_f = float(loss_val.detach())
            pbar.set_postfix(loss=f'{loss_f:.3f}', lr=f'{lr:.1e}', mode=mode, refresh=False)
            if local_step % log_every == 0:
                _jlog(dict(global_step=global_step, stage=stage_i,
                           loss=loss_f, bpb=loss_f/math.log(2), lr=lr, mode=mode))
                log_f.write(str(pbar) + '\n')
                print()

            if local_step % eval_every == 0 or local_step == 1:
                model.eval()
                with torch.no_grad():
                    val_tok  = torch.tensor(val_np, device=device)
                    val_log  = model(val_tok, mask_t)               # (B, L, 256)
                    val_lp_c = F.log_softmax(val_log[:, c0-1:c1-1], dim=-1)
                    val_tgt  = val_tok[:, c0:c1]                    # always 0-255
                    val_nll  = -val_lp_c.gather(2, val_tgt.unsqueeze(-1)).squeeze(-1)
                    val_loss = float(val_nll.mean())
                    val_bpb  = val_loss / math.log(2)
                elapsed = time.time() - t0
                _log(f'\n--- stage={stage_i} step={local_step}/{n_steps}  g={global_step}'
                     f'  loss={loss_f:.4f}  val_loss={val_loss:.4f}'
                     f'  val_bpb={val_bpb:.3f}  lr={lr:.2e}  {elapsed:.0f}s ---')

                all_cer = []
                for name, x_S in test_seqs.items():
                    assert out_len <= seg_len - warmup_len
                    f_start = min(int(seg_len * eval_offset), seg_len - warmup_len - out_len)
                    y_start = f_start + warmup_len
                    y_end   = min(y_start + out_len, seg_len)
                    warmup  = x_S[max(0, y_start - warmup_len):y_start]
                    if len(warmup) < warmup_len:
                        warmup = [x_S[0]] * (warmup_len - len(warmup)) + list(warmup)
                    target  = x_S[y_start:y_end]
                    gen     = ar_decode_role(model, x_S, slot_len,
                                             warmup, len(target), device,
                                             n_blocks=n_blocks,
                                             recall_from=recall_from,
                                             intermed_len=intermed_len,
                                             mem_window=mem_window)
                    c = cer(gen, target)
                    all_cer.append(c)
                    ok = '✓' if c == 0.0 else '✗'
                    _log(f'  {ok} {name:15s} <f>[{f_start}:{y_start}]</f>'
                         f' <c>[{y_start}:{y_end}]</c>'
                         f'  match={100*(1-c):5.1f}%  CER={c:.3f}')
                    def _tok_hex(seq):
                        return ''.join(f'{t:02x}' if t < 256 else f'[{t}]' for t in seq)
                    _log(f'    src={_tok_hex(x_S)}')
                    _log(f'    wup={_tok_hex(warmup)}')
                    _log(f'    gen={_tok_hex(gen)}')
                    _log(f'    ref={_tok_hex(target)}')

                mean_cer = sum(all_cer) / len(all_cer)
                _log(f'  → mean CER={mean_cer:.3f}  match={100*(1-mean_cer):.1f}%')
                _jlog(dict(global_step=global_step, stage=stage_i, loss=loss_f,
                           val_loss=val_loss, val_bpb=val_bpb, mean_cer=mean_cer))

                if mean_cer == 0.0:
                    _log(f'\n★ PERFECT at stage {stage_i} step={local_step}!')
                    ckpt = os.path.join(ckpt_dir, f'stage{stage_i}_step{local_step}.pt')
                    torch.save({'model': model.state_dict(), 'hp': hp,
                                'stage': stage_i, 'step': local_step}, ckpt)
                    _log(f'  [ckpt] {ckpt}')
                    break

        ckpt = os.path.join(ckpt_dir, f'stage{stage_i}_end.pt')
        torch.save({'model': model.state_dict(), 'hp': hp, 'stage': stage_i}, ckpt)
        _log(f'  [ckpt stage {stage_i} end] {ckpt}')

        # Cross-config generalisation eval at end of each stage
        # Tests 1-block + 2-block(from=0) + 2-block(from=1) regardless of training config.
        # Measures whether the fast-weight algorithm generalises across n_blocks.
        model.eval()
        _log(f'\n  [generalisation eval]')
        gen_results = {}
        for nb, rf in [(1, 0), (2, 0), (2, 1)]:
            all_c = []
            for sname, x_S in test_seqs.items():
                f_start = min(int(seg_len * eval_offset), seg_len - warmup_len - out_len)
                y_start = f_start + warmup_len
                y_end   = min(y_start + out_len, seg_len)
                wm      = x_S[max(0, y_start - warmup_len):y_start]
                if len(wm) < warmup_len:
                    wm = [x_S[0]] * (warmup_len - len(wm)) + list(wm)
                tgt = x_S[y_start:y_end]
                with torch.no_grad():
                    g = ar_decode_role(model, x_S, slot_len, wm, len(tgt), device,
                                       n_blocks=nb, recall_from=rf,
                                       intermed_len=intermed_len, mem_window=mem_window)
                all_c.append(cer(g, tgt))
            mean_c = sum(all_c) / len(all_c)
            key = f'nb{nb}_rf{rf}'
            gen_results[key] = round(100 * (1 - mean_c), 1)
            _log(f'    n_blocks={nb} recall_from={rf}  match={100*(1-mean_c):.1f}%')
        _jlog(dict(global_step=global_step, stage=stage_i, event='gen_eval', **gen_results))

    _log(f'\nDone. Total: {time.time()-t0:.0f}s')
    log_f.close(); jlog_f.close()
    return model, run_dir


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load hp dict from a Python config file (must define module-level `hp`)."""
    spec   = importlib.util.spec_from_file_location('_kvmem_cfg', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'hp'):
        raise ValueError(f'{path!r} must define a module-level `hp` dict')
    return dict(module.hp)


# ---------------------------------------------------------------------------
# CLI / DEFAULTS
# ---------------------------------------------------------------------------

CURRICULUM_SURAH = [
    dict(seg_len= 32, slot_len=128, warmup_len= 8, out_len= 8,  B=16, n_steps=40000),
    dict(seg_len= 64, slot_len= 64, warmup_len=16, out_len=16,  B=16, n_steps=40000),
    dict(seg_len=128, slot_len= 64, warmup_len=32, out_len=32,  B= 8, n_steps=40000),
    dict(seg_len=256, slot_len= 64, warmup_len=32, out_len=64,  B= 4, n_steps=20000),
    dict(seg_len=576, slot_len= 64, warmup_len=32, out_len=128, B= 4, n_steps=20000),
]

DEFAULTS = dict(
    seg_len=32, slot_len=32, d=64, n_layers=4, n_heads=4, d_ff=256,
    B=64, lr_max=3e-4, wd=0.001, warmup_steps=1000, cycle_steps=0,
    n_steps=10000, eval_every=5000, log_every=1000,
    warmup_len=8, out_len=8,
    rope=True, yarn=True, grok=False, seed=42,
    drop_close_prob=0.5,
    ocd=False, ocd_prob=0.01, tf_warmup=0,
    grad_clip=10.0, dataset_size=5000,
    stablemax=False, eval_offset=0.25,
    grad_checkpoint=False,
    mem_window=0,   # 0=full history; 1=isolated; N=last N h states
    n_blocks=1, recall_from=0,
    compile=False,
    curriculum=CURRICULUM_SURAH,
)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    p.add_argument('--config',          type=str)
    p.add_argument('--seg-len',         type=int)
    p.add_argument('--slot-len',        type=int)
    p.add_argument('--d',               type=int)
    p.add_argument('--n-layers',        type=int)
    p.add_argument('--B',               type=int)
    p.add_argument('--lr',              type=float)
    p.add_argument('--steps',           type=int)
    p.add_argument('--eval-every',      type=int)
    p.add_argument('--log-every',       type=int)
    p.add_argument('--warmup-len',      type=int)
    p.add_argument('--out-len',         type=int)
    p.add_argument('--warmup-steps',    type=int)
    p.add_argument('--cycle-steps',     type=int)
    p.add_argument('--slot-style',      type=str)
    p.add_argument('--drop-close',      type=float)
    p.add_argument('--ocd',             action='store_true')
    p.add_argument('--ocd-prob',        type=str)
    p.add_argument('--grad-clip',       type=float)
    p.add_argument('--dataset-size',    type=int)
    p.add_argument('--eval-offset',     type=float)
    p.add_argument('--stablemax',       action='store_true')
    p.add_argument('--grad-checkpoint', action='store_true',
                   help='Depth-wise gradient checkpointing per block (saves inter-layer activations)')
    p.add_argument('--n-blocks',        type=int)
    p.add_argument('--recall-from',     type=int)
    p.add_argument('--no-grok',         action='store_true')
    p.add_argument('--compile',         action='store_true')
    p.add_argument('--name',            type=str)
    p.add_argument('--name-date',       action='store_true')
    p.add_argument('--log-dir',         type=str, default='logs')
    p.add_argument('--device',          type=str, default='cpu',
                   choices=['cpu', 'mps', 'cuda'])
    p.add_argument('--seed',            type=int)
    p.add_argument('--curriculum',      type=str, default='v1',
                   choices=['v1', 'none'])
    args = p.parse_args()

    hp = dict(DEFAULTS)
    if args.config:
        hp.update(load_config(args.config))
    if args.curriculum == 'none':
        hp['curriculum'] = None

    for src, dst in [
        ('slot_len', 'slot_len'), ('n_layers', 'n_layers'), ('B', 'B'),
        ('steps', 'n_steps'), ('eval_every', 'eval_every'), ('log_every', 'log_every'),
        ('warmup_len', 'warmup_len'), ('out_len', 'out_len'),
        ('warmup_steps', 'warmup_steps'), ('cycle_steps', 'cycle_steps'),
        ('seed', 'seed'),
        ('grad_clip', 'grad_clip'), ('dataset_size', 'dataset_size'),
        ('eval_offset', 'eval_offset'),
        ('n_blocks', 'n_blocks'), ('recall_from', 'recall_from'),
    ]:
        v = getattr(args, src, None)
        if v is not None:
            hp[dst] = v

    if args.seg_len:               hp['seg_len'] = args.seg_len
    if args.seg_len and not args.slot_len: hp['slot_len'] = args.seg_len
    if args.d:                     hp['d'] = args.d; hp['d_ff'] = args.d * 4
    if args.lr:                    hp['lr_max'] = args.lr
    if args.drop_close is not None: hp['drop_close_prob'] = args.drop_close
    if args.ocd:                   hp['ocd'] = True
    if args.ocd_prob is not None:
        try:    hp['ocd_prob'] = json.loads(args.ocd_prob)
        except: hp['ocd_prob'] = float(args.ocd_prob)
    if args.name:                  hp['name'] = args.name
    if args.name_date:             hp['name_date'] = True
    if args.stablemax:             hp['stablemax'] = True
    if args.grad_checkpoint:       hp['grad_checkpoint'] = True
    if args.no_grok:               hp['grok'] = False
    if args.compile:               hp['compile'] = True

    seg      = hp['seg_len']
    slot_len = hp['slot_len']
    n_blocks = hp.get('n_blocks', 1)
    pos_tmp  = multi_block_positions(n_blocks, seg, slot_len,
                                     hp['warmup_len'], hp['out_len'])
    hp['L_train'] = pos_tmp['L']
    hp['L_max']   = hp['L_train'] * 4

    if hp.get('curriculum') is None:
        hp['curriculum'] = [dict(
            seg_len=seg, slot_len=slot_len,
            n_blocks=n_blocks, recall_from=hp.get('recall_from', 0),
            warmup_len=hp['warmup_len'], out_len=hp['out_len'],
            B=hp['B'], n_steps=hp['n_steps'],
        )]

    train_role(hp, log_base=args.log_dir, device_str=args.device)
