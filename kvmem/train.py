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
    LATENT_OPEN, LATENT_CLOSE,
    REFINE_OPEN, REFINE_CLOSE,
    HIDDEN_OPEN_LEN, HIDDEN_CLOSE_LEN,
    INPUT_OPEN_LEN, INPUT_CLOSE_LEN,
    QUERY_OPEN_LEN, QUERY_CLOSE_LEN,
    OUTPUT_OPEN_LEN, OUTPUT_CLOSE_LEN,
    LATENT_OPEN_LEN, LATENT_CLOSE_LEN,
    multi_block_positions, make_mask_multi, make_multi_batch,
    interleaved_positions, make_mask_interleaved, make_interleaved_batch,
    refine_positions, make_mask_refine, make_refine_batch,
    make_mask_old_memory,
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
                   latent_len: int = 0, mem_window: int = 0,
                   distractor_seed: int = 99) -> list[int]:
    """
    Greedy AR decode (eval). Builds a n_blocks sequence where x_S is placed
    in block recall_from; other blocks are filled with random distractors.

    For n_blocks=1: single-block eval (original behaviour).
    For n_blocks>1: correct multi-block eval — distractor blocks test that
    the model routes <q> to the right block from anchor content alone.
    """
    from kvmem.data import make_hidden_slot_ids, make_latent_slot_ids, _sample_seg
    seg_len    = len(x_S)
    wl         = len(warmup)
    slot_ids   = make_hidden_slot_ids(slot_len)
    intermed_ids = make_latent_slot_ids(latent_len, slot_len) if latent_len > 0 else []
    pos        = multi_block_positions(n_blocks, seg_len, slot_len, wl, out_len, latent_len)
    L          = pos['L']
    mask_t     = torch.tensor(
        make_mask_multi(n_blocks, seg_len, slot_len, wl, out_len, latent_len, mem_window),
        dtype=torch.float32, device=device)

    rng    = np.random.default_rng(distractor_seed)
    tokens = np.zeros(L, dtype=np.int64)

    for k, b in enumerate(pos['blocks']):
        src = np.array(x_S) if k == recall_from else _sample_seg(rng, seg_len)
        tokens[b['block_start']:b['s0']]         = INPUT_OPEN
        tokens[b['s0']:b['s1']]                  = src
        tokens[b['s1']:b['s_close_end']]         = INPUT_CLOSE
        if latent_len > 0:
            tokens[b['p_open']:b['p0']]          = LATENT_OPEN
            tokens[b['p0']:b['p1']]              = intermed_ids
            tokens[b['p1']:b['p_close_end']]     = LATENT_CLOSE
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


@torch.no_grad()
def ar_decode_refine(model, x_S: list[int], slot_len: int,
                     warmup: list[int], out_len: int, device,
                     n_attempts: int = 1,
                     latent_len: int = 0,
                     mem_window: int = -1) -> tuple[list[list[int]], list[int]]:
    """
    AR decode for new refine architecture.

    Generates n_attempts attempt outputs (each followed by a correction <z><h> block),
    then stops. Does NOT generate the final turn (inference stopping point).

    Stopping: compare consecutive attempts; stop early if they converge.

    Returns (attempts, last_attempt):
      attempts    : list of n_attempts lists (model's <y> at each step)
      last_attempt: the final generated output (= attempts[-1])
    """
    from kvmem.data import (make_hidden_slot_ids, make_latent_slot_ids,
                             refine_positions, make_mask_refine)
    seg_len    = len(x_S)
    wl         = len(warmup)
    slot_ids   = make_hidden_slot_ids(slot_len)
    ponder_ids = make_latent_slot_ids(latent_len, slot_len) if latent_len > 0 else []
    pos        = refine_positions(n_attempts, 1, seg_len, slot_len, wl, out_len, latent_len)
    L          = pos['L']
    mask_t     = torch.tensor(
        make_mask_refine(n_attempts, 1, seg_len, slot_len, wl, out_len, latent_len, mem_window),
        dtype=torch.float32, device=device)

    tokens = np.zeros(L, dtype=np.int64)
    b = pos['blocks'][0]
    tokens[b['block_start']:b['s0']]   = INPUT_OPEN
    tokens[b['s0']:b['s1']]             = x_S
    tokens[b['s1']:b['s_close_end']]    = INPUT_CLOSE
    if latent_len > 0:
        tokens[b['p_open']:b['p0']]     = LATENT_OPEN
        tokens[b['p0']:b['p1']]         = ponder_ids
        tokens[b['p1']:b['p_close_end']]= LATENT_CLOSE
    tokens[b['p_close_end']:b['sl0']]   = HIDDEN_OPEN
    tokens[b['sl0']:b['sl1']]            = slot_ids
    tokens[b['sl1']:b['mc1']]            = HIDDEN_CLOSE

    # Write <r> warmup
    tokens[pos['r_open']:pos['r0']] = REFINE_OPEN
    tokens[pos['r0']:pos['r1']]     = warmup
    tokens[pos['r1']:pos['rc1']]    = REFINE_CLOSE

    all_attempts = []
    prev_gen     = None

    for k, att in enumerate(pos['attempts']):
        # Write <y> open tag
        tokens[att['c0'] - OUTPUT_OPEN_LEN : att['c0']] = OUTPUT_OPEN
        tok_t = torch.tensor(tokens, dtype=torch.long, device=device)
        gen = []
        for j in range(out_len):
            logits   = model(tok_t, mask_t)
            next_tok = int(logits[att['c0'] + j - 1].argmax())
            gen.append(next_tok)
            tok_t[att['c0'] + j] = next_tok
        # Write output + close tag + correction block into tokens
        tokens[att['c0']:att['c1']]   = gen
        tokens[att['c1']:att['cl1']]  = OUTPUT_CLOSE
        if latent_len > 0:
            tokens[att['cl1']:att['p0']]  = LATENT_OPEN
            tokens[att['p0']:att['p1']]   = ponder_ids
            tokens[att['p1']:att['pc1']]  = LATENT_CLOSE
        tokens[att['pc1']:att['sl0']]     = HIDDEN_OPEN
        tokens[att['sl0']:att['sl1']]     = slot_ids
        tokens[att['sl1']:att['mc1']]     = HIDDEN_CLOSE

        all_attempts.append(gen)
        if prev_gen is not None and gen == prev_gen:
            break  # converged early
        prev_gen = gen

    return all_attempts, all_attempts[-1] if all_attempts else []


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _stablemax_log_probs(logits: torch.Tensor) -> torch.Tensor:
    s = torch.where(logits >= 0, logits + 1.0, 1.0 / (1.0 - logits))
    return (s / s.sum(dim=-1, keepdim=True)).log()


def _positional_ls_nll(lp: torch.Tensor, tgt: torch.Tensor, ls_max: float) -> torch.Tensor:
    """
    NLL with positional label smoothing: ε=0 at position 0, ε=ls_max at position N-1.
    lp:  (B, out_len, V)  log-probs
    tgt: (B, out_len)     target token IDs
    Returns (B, out_len) per-token NLL.
    """
    out_len = lp.shape[1]
    nll_hard = -lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)          # (B, out_len)
    if ls_max <= 0.0:
        return nll_hard
    eps = torch.linspace(0.0, ls_max, out_len, device=lp.device)    # (out_len,)
    nll_soft = -lp.mean(dim=-1)                                       # (B, out_len)
    return (1.0 - eps) * nll_hard + eps * nll_soft


def train_role(hp: dict, log_base: str = 'logs', device_str: str = 'cpu'):
    device      = torch.device(device_str)
    lr_max      = hp['lr_max']
    wd          = hp['wd']
    eval_every  = hp['eval_every']
    log_every   = hp['log_every']
    # slot_style removed — learned embeddings use make_mem_slot_ids()
    warmup_steps = hp['warmup_steps']
    cycle_steps  = hp.get('cycle_steps', 0)
    seed         = hp['seed']
    use_ocd      = hp['ocd']
    _ocd_prob    = hp['ocd_prob']
    grad_clip    = hp['grad_clip']
    dataset_size = hp['dataset_size']
    # active_slots removed — slot_len IS the bottleneck
    log_probs_fn = _stablemax_log_probs if hp['stablemax'] else \
                   lambda x: F.log_softmax(x, dim=-1)
    eval_offset    = hp['eval_offset']
    _ls_max_init   = hp.get('ls_max', 0.0)          # positional label smoothing max ε
    _ls_anneal     = hp.get('ls_anneal_steps', 0)    # steps over which ε decays to 0
    _noise_skew    = hp.get('noise_skew', False)     # skew draft noise toward end of seq

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
    eval_configs = hp.get('eval_configs', None)  # [(n_blocks, recall_from), ...]
    curriculum = _cur if _cur else [{
        'seg_len':     hp['seg_len'],
        'slot_len':    hp.get('slot_len', hp['seg_len']),
        'warmup_len':  hp.get('warmup_len', 16),
        'latent_len':  hp.get('latent_len', 0),
        'mem_window':    hp.get('mem_window', -1),
        'out_len':     hp.get('out_len', 32),
        'n_blocks':    hp.get('n_blocks', 1),
        'recall_from': hp.get('recall_from', 0),
        'B': hp['B'], 'n_steps': hp['n_steps'],
    }]

    # Default eval_configs: unique (n_blocks, recall_from) from curriculum + (1,0) baseline
    if eval_configs is None:
        seen = {(1, 0)}
        for s in curriculum:
            nb  = s.get('n_blocks', 1)
            rfs = s.get('recall_froms', s.get('recall_from', 0))
            for rf in (rfs if isinstance(rfs, list) else [rfs]):
                seen.add((nb, rf))
        eval_configs = sorted(seen)

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

    def _resolve_out_len(st):
        ol = st.get('out_len', 128)
        return st['seg_len'] if ol == -1 else ol

    max_stage = max(curriculum, key=lambda s: s['seg_len'])
    max_nb    = max_stage.get('n_blocks', 1)
    max_pos   = multi_block_positions(max_nb, max_stage['seg_len'],
                                      max_stage.get('slot_len', max_stage['seg_len']),
                                      max_stage.get('warmup_len', 32),
                                      _resolve_out_len(max_stage),
                                      max_stage.get('latent_len', 0))
    L_max_seq = max_pos['L']
    # Joint mode may have refine trajectories that are longer than standard sequences
    for _st in curriculum:
        if _st.get('mode') == 'joint':
            for _jm in _st.get('joint_mix', []):
                _jnb = _jm.get('n_blocks', 1)
                _jla = _st.get('latent_len', 0)
                _jsl = _st.get('slot_len', _st['seg_len'])
                _jsg = _st['seg_len']
                _jwl = _st.get('warmup_len', 16)
                _jol = _resolve_out_len(_st)
                if _jm['traj'] == 'ref':
                    _jp = refine_positions(_jm.get('n_attempts', 5), _jnb, _jsg, _jsl, _jwl, _jol, _jla)
                elif _jm['traj'] == 'int':
                    _jp = interleaved_positions(_jnb, _jsg, _jsl, _jwl, _jol, _jla)
                else:
                    _jp = multi_block_positions(_jnb, _jsg, _jsl, _jwl, _jol, _jla)
                L_max_seq = max(L_max_seq, _jp['L'])
    hp_model  = dict(hp, seg_len=max_stage['seg_len'],
                     slot_len=max_stage.get('slot_len', max_stage['seg_len']),
                     latent_len=max_stage.get('latent_len', hp.get('latent_len', 0)),
                     L_train=L_max_seq, L_max=L_max_seq * 4)

    torch.manual_seed(seed)
    model = build_model(hp_model, device)
    _resume_state = None
    if hp.get('_pretrained_ckpt'):
        _pt = torch.load(hp['_pretrained_ckpt'], map_location=device)
        model.load_state_dict(_pt['model'])
        _log(f'  [pretrained weights from {hp["_pretrained_ckpt"]}]')
    if hp.get('_resume_ckpt'):
        _resume_state = torch.load(hp['_resume_ckpt'], map_location=device)
        model.load_state_dict(_resume_state['model'])
        _log(f'  [resumed from {hp["_resume_ckpt"]}  stage={_resume_state.get("stage","?")}  step={_resume_state.get("step","?")}]')
    if hp.get('compile', False):
        model = torch.compile(model)

    if hp['grok']:
        from kvmem.optim import GrokAdamW
        opt = GrokAdamW(model.parameters(), lr=lr_max, weight_decay=wd,
                        rho=hp.get('grok_rho', 0.9), batch_size=curriculum[0]['B'])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=wd)

    _resume_stage = 0
    _resume_step  = 0
    if _resume_state is not None:
        opt.load_state_dict(_resume_state['opt'])
        _resume_stage = _resume_state.get('stage', 0)
        _resume_step  = _resume_state.get('step', 0)
        if 'rng_np' in _resume_state:
            rng.bit_generator.state = _resume_state['rng_np']
        if 'rng_torch' in _resume_state:
            torch.set_rng_state(_resume_state['rng_torch'])
        global_step = _resume_state.get('global_step', 0)
        _log(f'  [rng + opt state restored from checkpoint]')

    gc_flag = '  grad_checkpoint=ON' if hp.get('grad_checkpoint') else ''
    _log(f'\n=== kvmem memory-first | run_dir={run_dir} ===')
    _log(f'  cmd: {" ".join(sys.argv)}')
    params = sum(p.numel() for p in model.parameters())
    _log(f'  Model: d={hp["d"]}  n_layers={hp["n_layers"]}  params={params:,}'
         f'  device={device}{gc_flag}')
    _log(f'  rope={hp.get("rope",False)}  yarn={hp.get("yarn",False)}'
         f'  layout=memory-first'
         + (f'  OCD prob={_ocd_prob}' if use_ocd else '  TF-only'))
    _log(f'  Curriculum: {len(curriculum)} stages')
    for i, st in enumerate(curriculum):
        nb = st.get('n_blocks', 1)
        rf = st.get('recall_from', 0)
        p  = multi_block_positions(nb, st['seg_len'],
                                   st.get('slot_len', st['seg_len']),
                                   st.get('warmup_len', 16), st.get('out_len', 32),
                                   st.get('latent_len', 0))
        pl = st.get('latent_len', 0)
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
        if stage_i < _resume_stage: continue  # skip completed stages
        seg_len    = stage['seg_len']
        slot_len   = stage.get('slot_len', seg_len)
        warmup_len = stage.get('warmup_len', 16)
        latent_len = stage.get('latent_len', 0)
        mem_window   = stage.get('mem_window', -1)
        out_len    = stage.get('out_len', 32)
        if out_len == -1:
            out_len = seg_len  # -1 = full segment recall
        n_blocks   = stage.get('n_blocks', 1)
        recall_from = stage.get('recall_froms', stage.get('recall_from', 0))
        seq_mode    = stage.get('mode', 'end')   # end|int|mix|ref
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

        if seq_mode in ('int', 'mix'):
            pos_int = interleaved_positions(n_blocks, seg_len, slot_len, warmup_len, out_len, latent_len)
            L_total_int = pos_int['L']
            mask_int_t  = torch.tensor(
                make_mask_interleaved(n_blocks, seg_len, slot_len, warmup_len, out_len, latent_len, mem_window),
                dtype=torch.float32, device=device)

        # Refine mode: k ~ Uniform(0, n_attempts_max) attempts per step + always 1 final clean
        # n_attempts=0: identical to standard recall with <r> tag instead of <q>
        # noise descends linearly: attempt 0 = noise_hi, attempt N-1 = noise_lo, final = 0
        _n_attempts_max = stage.get('n_attempts', stage.get('n_draft_turns', 1))
        _noise_hi       = stage.get('noise_hi', 0.8)
        _noise_lo       = stage.get('noise_lo', 0.05)
        _rand_turns     = stage.get('rand_turns', False)
        _noise_schedule = stage.get('noise_schedule', None)

        def _make_noise_schedule(k):
            """Linear: attempt 0 = noise_hi, attempt k-1 = noise_lo, final <y> always 0."""
            if _noise_schedule is not None and len(_noise_schedule) == k:
                return _noise_schedule
            if k == 0:
                return []
            if k == 1:
                return [_noise_hi]
            return [_noise_hi - (_noise_hi - _noise_lo) * j / (k - 1) for j in range(k)]

        if seq_mode == 'ref':
            # Precompute pos/mask for k=0..n_attempts_max
            _ref_cache = {}
            for k in range(_n_attempts_max + 1):
                _p = refine_positions(k, n_blocks, seg_len, slot_len,
                                      warmup_len, out_len, latent_len)
                _m = torch.tensor(
                    make_mask_refine(k, n_blocks, seg_len, slot_len,
                                     warmup_len, out_len, latent_len, mem_window),
                    dtype=torch.float32, device=device)
                _ref_cache[k] = (_p, _m)
            pos_ref, mask_ref_t = _ref_cache[_n_attempts_max]
            _ref_c0 = pos_ref['query_c0']   # loss on post-refine query (must match 100%)
            _ref_c1 = pos_ref['query_c1']
            _log(f'  refine mode: n_attempts_max={_n_attempts_max}  rand_turns={_rand_turns}'
                 f'  noise_hi={_noise_hi}  noise_lo={_noise_lo}'
                 + f'  L_max={pos_ref["L"]}'
                 + f'  (n=0 same as standard recall with <r> tag)')

        # Joint mode: per-step trajectory sampling from a weighted mixture
        _joint_cache   = []
        _joint_weights = None
        _has_ref_joint = False
        if seq_mode == 'joint':
            _jmix = stage.get('joint_mix', [])
            _jw   = np.array([jm['weight'] for jm in _jmix], dtype=np.float32)
            _joint_weights = _jw / _jw.sum()
            for jm in _jmix:
                jtraj = jm['traj']          # 'end' | 'ref' | 'int'
                jnb   = jm.get('n_blocks', 1)
                jrf   = jm.get('recall_from', 0)
                j_noise_hi = jm.get('noise_hi', _noise_hi)
                j_noise_lo = jm.get('noise_lo', _noise_lo)
                j_n_att    = jm.get('n_attempts', _n_attempts_max)
                if jtraj == 'ref':
                    j_ref_cache = {}
                    for k in range(j_n_att + 1):
                        _p = refine_positions(k, jnb, seg_len, slot_len,
                                              warmup_len, out_len, latent_len)
                        _m = torch.tensor(
                            make_mask_refine(k, jnb, seg_len, slot_len,
                                             warmup_len, out_len, latent_len, mem_window),
                            dtype=torch.float32, device=device)
                        j_ref_cache[k] = (_p, _m)
                    _joint_cache.append(dict(traj='ref', n_blocks=jnb, recall_from=jrf,
                                             ref_cache=j_ref_cache, n_attempts=j_n_att,
                                             noise_lo=j_noise_lo, noise_hi=j_noise_hi))
                    _has_ref_joint = True
                elif jtraj == 'int':
                    jp  = interleaved_positions(jnb, seg_len, slot_len, warmup_len, out_len, latent_len)
                    jmt = torch.tensor(
                        make_mask_interleaved(jnb, seg_len, slot_len, warmup_len, out_len, latent_len, mem_window),
                        dtype=torch.float32, device=device)
                    _joint_cache.append(dict(traj='int', n_blocks=jnb, recall_from=jrf, pos=jp, mask=jmt))
                else:  # 'end'
                    jp  = multi_block_positions(jnb, seg_len, slot_len, warmup_len, out_len, latent_len)
                    jmt = torch.tensor(
                        make_mask_multi(jnb, seg_len, slot_len, warmup_len, out_len, latent_len, mem_window),
                        dtype=torch.float32, device=device)
                    _joint_cache.append(dict(traj='end', n_blocks=jnb, recall_from=jrf, pos=jp, mask=jmt))
            _log(f'  joint mode: {len(_jmix)} trajectory types'
                 + ''.join(f'\n    {jm["traj"]} nb={jm.get("n_blocks",1)} rf={jm.get("recall_from",0)} w={jm["weight"]:.0%}'
                           for jm in _jmix))

        pos     = multi_block_positions(n_blocks, seg_len, slot_len, warmup_len, out_len,
                                        latent_len)
        L_total = pos['L']
        c0, c1  = pos['c0'], pos['c1']
        mask_t  = torch.tensor(
            make_mask_multi(n_blocks, seg_len, slot_len, warmup_len, out_len,
                            latent_len, mem_window),
            dtype=torch.float32, device=device)

        test_seqs = make_test_sequences(seg_len)
        val_np    = make_multi_batch(
            np.random.default_rng(seed + stage_i + 1),
            B, n_blocks, recall_from, seg_len, slot_len,
            warmup_len, out_len, latent_len=latent_len)
        pool_rng  = np.random.default_rng(seed + stage_i + 1000)
        ds   = dataset_size if dataset_size > 0 else None
        pool = (np.stack([make_multi_batch(pool_rng, B, n_blocks, recall_from,
                                           seg_len, slot_len,
                                           warmup_len, out_len,
                                           latent_len)
                          for _ in range(ds)])
                if ds else None)
        _log(f'  dataset: {"infinite" if ds is None else f"{ds} batches ({ds*B} examples)"}')

        # Refine val batch: measure NLL on post-refine query <q><y>
        # Used for both seq_mode='ref' and seq_mode='joint' (if any ref traj in mix)
        val_ref_np  = None
        val_ref_pos = None
        val_ref_mask = None
        _val_ref_n_att = _n_attempts_max  # number of attempts in val refine batch
        if seq_mode == 'ref':
            val_ref_np = make_refine_batch(
                np.random.default_rng(seed + stage_i + 2),
                B, _n_attempts_max, _make_noise_schedule(_n_attempts_max),
                n_blocks, recall_from if isinstance(recall_from, int) else 0,
                seg_len, slot_len, warmup_len, out_len, latent_len,
                noise_skew=_noise_skew)
            val_ref_pos  = refine_positions(_n_attempts_max, n_blocks, seg_len, slot_len, warmup_len, out_len, latent_len)
            val_ref_mask = torch.tensor(
                make_mask_refine(_n_attempts_max, n_blocks, seg_len, slot_len, warmup_len, out_len, latent_len, mem_window),
                dtype=torch.float32, device=device)
        elif seq_mode == 'joint' and _has_ref_joint:
            # Use the first ref trajectory's config for the val refine batch
            _jref = next(jm for jm in _joint_cache if jm['traj'] == 'ref')
            _val_ref_n_att = _jref['n_attempts']
            _jnoise_val = [(_jref['noise_lo'], _jref['noise_hi'])] * _val_ref_n_att
            val_ref_np = make_refine_batch(
                np.random.default_rng(seed + stage_i + 2),
                B, _val_ref_n_att, _jnoise_val,
                _jref['n_blocks'], _jref['recall_from'],
                seg_len, slot_len, warmup_len, out_len, latent_len,
                noise_skew=_noise_skew)
            val_ref_pos, val_ref_mask = _jref['ref_cache'][_val_ref_n_att]

        def _fmt_elapsed(s):
            h, rem = divmod(int(s), 3600)
            m, sec = divmod(rem, 60)
            return f'{h:02d}:{m:02d}:{sec:02d}'

        if hp['grok']:
            for pg in opt.param_groups:
                pg['batch_size'] = B
        _log(f'\n{"="*60}')
        _rf_label = recall_from if isinstance(recall_from, int) else f'mixed{recall_from}'
        _log(f'  [stage {stage_i}] n_blocks={n_blocks} recall_from={_rf_label}'
             f'  seg={seg_len}  slot={slot_len}  wl={warmup_len}  out={out_len}'
             f'  B={B}  steps={n_steps}  L={L_total}')

        pbar = tqdm(range(1, n_steps + 1), desc=f'stage{stage_i}', dynamic_ncols=True)

        for local_step in pbar:
            global_step += 1
            lr = lr_schedule(local_step)
            for pg in opt.param_groups:
                pg['lr'] = lr
            # Positional label smoothing: ε_max anneals linearly to 0 over ls_anneal_steps
            _cur_ls = (_ls_max_init * max(0.0, 1.0 - local_step / _ls_anneal)
                       if _ls_anneal > 0 else _ls_max_init)

            model.train()
            # Batch generation: int mode = interleaved; ref mode = refine; end mode = standard
            if seq_mode in ('int', 'mix'):
                # Single unified mode: k ~ Uniform(1, n_blocks) per step
                # end is a special case (k=1, last block) — not needed separately
                _q_count = int(rng.integers(1, n_blocks + 1))
                tokens_np, _active_c = make_interleaved_batch(
                    rng, B, n_blocks, seg_len, slot_len,
                    warmup_len, out_len, latent_len, _q_count)
                _use_interleaved = True
                _use_refine = False
                _use_joint = False
            elif seq_mode == 'ref':
                _k = int(rng.integers(0, _n_attempts_max + 1)) if _rand_turns else _n_attempts_max
                _pos_k, _mask_k = _ref_cache[_k]
                _ref_c0 = _pos_k['query_c0']
                _ref_c1 = _pos_k['query_c1']
                tokens_np = make_refine_batch(
                    rng, B, _k, _make_noise_schedule(_k),
                    n_blocks, recall_from if isinstance(recall_from, int) else 0,
                    seg_len, slot_len, warmup_len, out_len, latent_len,
                    noise_skew=_noise_skew)
                _use_interleaved = False
                _use_refine = True
                _use_joint = False
            elif seq_mode == 'joint':
                _type_idx = int(rng.choice(len(_joint_cache), p=_joint_weights))
                _jc = _joint_cache[_type_idx]
                if _jc['traj'] == 'ref':
                    _jk = int(rng.integers(0, _jc['n_attempts'] + 1))
                    _jpos_k, _jmask_k = _jc['ref_cache'][_jk]
                    # Flat noise: all turns same U(lo, hi) range
                    _jnoise = [(_jc['noise_lo'], _jc['noise_hi'])] * _jk
                    tokens_np = make_refine_batch(
                        rng, B, _jk, _jnoise,
                        _jc['n_blocks'], _jc['recall_from'],
                        seg_len, slot_len, warmup_len, out_len, latent_len,
                        noise_skew=_noise_skew)
                elif _jc['traj'] == 'int':
                    _jq_count = int(rng.integers(1, _jc['n_blocks'] + 1))
                    tokens_np, _active_c = make_interleaved_batch(
                        rng, B, _jc['n_blocks'], seg_len, slot_len,
                        warmup_len, out_len, latent_len, _jq_count)
                else:  # 'end'
                    tokens_np = make_multi_batch(
                        rng, B, _jc['n_blocks'], _jc['recall_from'],
                        seg_len, slot_len, warmup_len, out_len, latent_len)
                _use_interleaved = False
                _use_refine = False
                _use_joint = True
            else:   # 'end' (default) or 'acc'
                tokens_np = (pool[(local_step - 1) % ds]
                             if pool is not None
                             else make_multi_batch(rng, B, n_blocks, recall_from,
                                                   seg_len, slot_len,
                                                   warmup_len, out_len,
                                                   latent_len))
                _use_interleaved = False
                _use_refine = False
                _use_joint = False

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
                opt.zero_grad()
                if _use_interleaved:
                    # Interleaved: loss on all active <y> regions
                    _L = tokens_np.shape[1]
                    tokens  = torch.tensor(tokens_np, device=device)
                    logits  = model(tokens, mask_int_t)              # (B, _L, 256)
                    loss_parts = []
                    _ranges = _active_c if _active_c else [(c0, c1)]  # fallback to end-mode range
                    for _c0, _c1 in _ranges:
                        if _c1 > _c0 and _c0 > 0:  # valid range
                            lp   = log_probs_fn(logits[:, _c0-1:_c1-1])
                            tgt  = tokens[:, _c0:_c1]
                            nll  = -lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
                            loss_parts.append(nll.mean())
                    loss_val = torch.stack(loss_parts).mean() if loss_parts else torch.tensor(0.0, device=device, requires_grad=True)
                    mode = 'tf-int'
                elif _use_refine:
                    tokens = torch.tensor(tokens_np, device=device)
                    logits = model(tokens, _mask_k)
                    # Post-refine query loss (primary)
                    lp_c     = log_probs_fn(logits[:, _ref_c0-1:_ref_c1-1])
                    tgts_c   = tokens[:, _ref_c0:_ref_c1]
                    nll_c    = _positional_ls_nll(lp_c, tgts_c, _cur_ls)
                    loss_val = nll_c.mean()
                    # Auxiliary loss: each attempt turn vs clean GT
                    # Gives direct gradient through correction path; fixes "ignore draft" sawtooth
                    _aux_w = hp.get('aux_attempt_loss', 0.0)
                    _mono_penalty = hp.get('mono_penalty', 0.0)
                    if _k > 0 and (_aux_w > 0.0 or _mono_penalty > 0.0):
                        gt_clean  = tokens[:, _pos_k['copy_c0']:_pos_k['copy_c1']]
                        turn_nlls = []
                        for _t in _pos_k['attempts']:
                            _att_lp  = log_probs_fn(logits[:, _t['c0']-1:_t['c1']-1])
                            _att_nll = _positional_ls_nll(_att_lp, gt_clean, _cur_ls)
                            turn_nlls.append(_att_nll.mean())
                        if _aux_w > 0.0:
                            loss_val = loss_val + _aux_w * torch.stack(turn_nlls).mean()
                        if _mono_penalty > 0.0:
                            turn_nlls.append(nll_c.mean())
                            mono_loss = sum(F.relu(turn_nlls[i+1] - turn_nlls[i])
                                            for i in range(len(turn_nlls)-1))
                            loss_val = loss_val + _mono_penalty * mono_loss
                    mode = 'tf-ref'
                elif _use_joint:
                    tokens = torch.tensor(tokens_np, device=device)
                    if _jc['traj'] == 'ref':
                        logits = model(tokens, _jmask_k)
                        _jc0, _jc1 = _jpos_k['query_c0'], _jpos_k['query_c1']
                        lp_c   = log_probs_fn(logits[:, _jc0-1:_jc1-1])
                        tgts_c = tokens[:, _jc0:_jc1]
                        nll_c  = _positional_ls_nll(lp_c, tgts_c, _cur_ls)
                        loss_val = nll_c.mean()
                        _aux_w = hp.get('aux_attempt_loss', 0.0)
                        _mono_penalty = hp.get('mono_penalty', 0.0)
                        if _jk > 0 and (_aux_w > 0.0 or _mono_penalty > 0.0):
                            gt_clean  = tokens[:, _jpos_k['copy_c0']:_jpos_k['copy_c1']]
                            turn_nlls = []
                            for _t in _jpos_k['attempts']:
                                _att_lp  = log_probs_fn(logits[:, _t['c0']-1:_t['c1']-1])
                                _att_nll = _positional_ls_nll(_att_lp, gt_clean, _cur_ls)
                                turn_nlls.append(_att_nll.mean())
                            if _aux_w > 0.0:
                                loss_val = loss_val + _aux_w * torch.stack(turn_nlls).mean()
                            if _mono_penalty > 0.0:
                                turn_nlls.append(nll_c.mean())
                                mono_loss = sum(F.relu(turn_nlls[i+1] - turn_nlls[i])
                                                for i in range(len(turn_nlls)-1))
                                loss_val = loss_val + _mono_penalty * mono_loss
                        mode = f'tf-jref(k={_jk})'
                    elif _jc['traj'] == 'int':
                        logits = model(tokens, _jc['mask'])
                        loss_parts = []
                        for _jc0, _jc1 in _active_c:
                            if _jc1 > _jc0 and _jc0 > 0:
                                lp  = log_probs_fn(logits[:, _jc0-1:_jc1-1])
                                tgt = tokens[:, _jc0:_jc1]
                                nll = -lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
                                loss_parts.append(nll.mean())
                        loss_val = torch.stack(loss_parts).mean() if loss_parts else torch.tensor(0.0, device=device, requires_grad=True)
                        mode = 'tf-jint'
                    else:  # end
                        logits = model(tokens, _jc['mask'])
                        _jc0, _jc1 = _jc['pos']['c0'], _jc['pos']['c1']
                        lp_c   = log_probs_fn(logits[:, _jc0-1:_jc1-1])
                        tgts_c = tokens[:, _jc0:_jc1]
                        nll_c  = -lp_c.gather(2, tgts_c.unsqueeze(-1)).squeeze(-1)
                        loss_val = nll_c.mean()
                        mode = 'tf-jend'
                else:
                    # Standard end-mode TF
                    tokens   = torch.tensor(tokens_np, device=device)
                    logits   = model(tokens, mask_t)                    # (B, L, 256)
                    lp_c     = log_probs_fn(logits[:, c0-1:c1-1])
                    tgts_c   = tokens[:, c0:c1]
                    nll_c    = -lp_c.gather(2, tgts_c.unsqueeze(-1)).squeeze(-1)
                    loss_val = nll_c.mean()
                    mode = 'tf'

            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            loss_f = float(loss_val.detach())
            pbar.set_postfix(loss=f'{loss_f:.3f}', lr=f'{lr:.1e}', mode=mode, refresh=False)

            if local_step % log_every == 0:
                elapsed = time.time() - t0
                _jlog(dict(global_step=global_step, stage=stage_i,
                           loss=loss_f, bpb=loss_f/math.log(2), lr=lr, mode=mode,
                           elapsed_s=round(elapsed, 1), elapsed=_fmt_elapsed(elapsed)))
                log_f.write(str(pbar) + '\n')
                print()

            if local_step % eval_every == 0 or local_step == 1:
                model.eval()
                with torch.no_grad():
                    val_tok  = torch.tensor(val_np, device=device)
                    val_log  = model(val_tok, mask_t)
                    val_lp_c = F.log_softmax(val_log[:, c0-1:c1-1], dim=-1)
                    val_tgt  = val_tok[:, c0:c1]
                    val_nll  = -val_lp_c.gather(2, val_tgt.unsqueeze(-1)).squeeze(-1)
                    val_loss = float(val_nll.mean())
                    val_bpb  = val_loss / math.log(2)
                    # Refine val: NLL on post-refine query <q><y> (works for both ref and joint mode)
                    val_ref_bpb = None
                    if val_ref_np is not None and val_ref_pos is not None:
                        vr_tok = torch.tensor(val_ref_np, device=device)
                        vr_log = model(vr_tok, val_ref_mask)
                        _vr_c0 = val_ref_pos['query_c0']
                        _vr_c1 = val_ref_pos['query_c1']
                        vr_lp  = F.log_softmax(vr_log[:, _vr_c0-1:_vr_c1-1], dim=-1)
                        vr_tgt = vr_tok[:, _vr_c0:_vr_c1]
                        vr_nll = -vr_lp.gather(2, vr_tgt.unsqueeze(-1)).squeeze(-1)
                        val_ref_bpb = float(vr_nll.mean()) / math.log(2)
                elapsed = time.time() - t0
                _ref_bpb_str = f'  val_ref_bpb={val_ref_bpb:.3f}' if val_ref_bpb is not None else ''
                _log(f'\n--- stage={stage_i} step={local_step}/{n_steps}  g={global_step}'
                     f'  loss={loss_f:.4f}  val_bpb={val_bpb:.3f}{_ref_bpb_str}  lr={lr:.2e}'
                     f'  {_fmt_elapsed(elapsed)} ---')

                # Multi-config eval: test all eval_configs at every eval step
                def _tok_hex(seq):
                    return ''.join(f'{t:02x}' if t < 256 else f'[{t}]' for t in seq)
                f_start = min(int(seg_len * eval_offset), seg_len - warmup_len - out_len)
                y_start = f_start + warmup_len
                y_end   = min(y_start + out_len, seg_len)

                _verbose_eval = hp.get('verbose_eval', False)
                _verbose_n    = hp.get('verbose_eval_n', 2)

                def _fmt_bytes(seq):
                    return ' '.join(f'{t:02x}' for t in seq)

                cfg_results = {}
                perfect_all = True
                for eval_nb, eval_rf in eval_configs:
                    all_c = []
                    # per-turn CER lists for refine mode: all_drafts[k] = CER list for draft turn k
                    # Eval with n_eval_attempts = max_train + 2 to test extrapolation
                    _n_eval_attempts = _n_attempts_max + 2
                    _use_ref_eval = seq_mode == 'ref' or (seq_mode == 'joint' and _has_ref_joint)
                    all_drafts = [[] for _ in range(_n_eval_attempts)] if _use_ref_eval else []
                    verbose_examples = []   # (sname, wm, tgt, drafts_or_None, g)
                    for sname, x_S in test_seqs.items():
                        wm = x_S[max(0, y_start - warmup_len):y_start]
                        if len(wm) < warmup_len:
                            wm = [x_S[0]] * (warmup_len - len(wm)) + list(wm)
                        tgt = x_S[y_start:y_end]
                        with torch.no_grad():
                            if _use_ref_eval:
                                drafts, g = ar_decode_refine(
                                    model, x_S, slot_len, wm, len(tgt), device,
                                    n_attempts=_n_eval_attempts,
                                    latent_len=latent_len, mem_window=mem_window)
                                for k, d in enumerate(drafts):
                                    if k < len(all_drafts):
                                        all_drafts[k].append(cer(d, tgt))
                            else:
                                drafts = None
                                g = ar_decode_role(model, x_S, slot_len, wm, len(tgt), device,
                                                   n_blocks=eval_nb, recall_from=eval_rf,
                                                   latent_len=latent_len, mem_window=mem_window)
                        all_c.append(cer(g, tgt))
                        if _verbose_eval and len(verbose_examples) < _verbose_n:
                            verbose_examples.append((sname, list(wm), list(tgt), drafts, list(g)))
                    mean_c = sum(all_c) / len(all_c)
                    key = f'n{eval_nb}_r{eval_rf}'
                    cfg_results[key] = round(100 * (1 - mean_c), 1)
                    ok = '✓' if mean_c == 0.0 else '✗'
                    if mean_c != 0.0: perfect_all = False
                    if all_drafts:
                        # Per-turn match% and monotonicity check
                        turn_matches = [round(100*(1 - sum(c)/len(c)), 1) for c in all_drafts]
                        final_match  = round(100*(1 - mean_c), 1)
                        all_matches  = turn_matches + [final_match]
                        monotonic    = all(all_matches[i] <= all_matches[i+1]
                                           for i in range(len(all_matches)-1))
                        mono_flag    = '↑' if monotonic else '⚠'
                        for k, m in enumerate(turn_matches):
                            cfg_results[f'{key}_t{k+1}'] = m
                        step_str = '  '.join(f't{k+1}={m:.1f}%' for k,m in enumerate(turn_matches))
                        delta    = round(final_match - turn_matches[0], 1)
                        _log(f'  {ok} {mono_flag} n={eval_nb} rf={eval_rf}'
                             f'  {step_str}  final={final_match:.1f}%  Δ={delta:+.1f}%')
                    else:
                        _log(f'  {ok} n={eval_nb} rf={eval_rf}  match={100*(1-mean_c):.1f}%')
                    if _verbose_eval:
                        for sname, wm, tgt, drafts, g in verbose_examples:
                            _log(f'    [{sname}] wm={_fmt_bytes(wm)}  gt={_fmt_bytes(tgt)}')
                            if drafts:
                                for k, d in enumerate(drafts):
                                    _match = round(100*(1-cer(d, tgt)), 1)
                                    _log(f'      t{k+1}: {_fmt_bytes(d)}  ({_match}%)')
                            _match = round(100*(1-cer(g, tgt)), 1)
                            _log(f'      fin: {_fmt_bytes(g)}  ({_match}%)')

                _jlog_d = dict(global_step=global_step, stage=stage_i, loss=loss_f,
                               val_loss=val_loss, val_bpb=val_bpb,
                               elapsed_s=round(elapsed, 1), elapsed=_fmt_elapsed(elapsed),
                               **cfg_results)
                if val_ref_bpb is not None:
                    _jlog_d['val_ref_bpb'] = round(val_ref_bpb, 5)
                _jlog(_jlog_d)

                if perfect_all:
                    _log(f'\n★ PERFECT (all eval configs) at stage {stage_i} step={local_step}!')
                    ckpt = os.path.join(ckpt_dir, f'stage{stage_i}_step{local_step}.pt')
                    torch.save({'model': model.state_dict(), 'hp': hp,
                                'stage': stage_i, 'step': local_step}, ckpt)
                    _log(f'  [ckpt] {ckpt}')
                    break

        ckpt = os.path.join(ckpt_dir, f'stage{stage_i}_end.pt')
        torch.save({
            'model':     model.state_dict(),
            'opt':       opt.state_dict(),
            'hp':        hp,
            'stage':     stage_i,
            'step':      local_step,
            'global_step': global_step,
            'rng_np':    rng.bit_generator.state,
            'rng_torch': torch.get_rng_state(),
        }, ckpt)
        _log(f'  [ckpt stage {stage_i} end] {ckpt}')



    _total = time.time() - t0
    h, rem = divmod(int(_total), 3600); m, s = divmod(rem, 60)
    _log(f'\nDone. Total: {h:02d}:{m:02d}:{s:02d} ({_total:.0f}s)')
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
    
    ocd=False, ocd_prob=0.01, tf_warmup=0,
    grad_clip=10.0, dataset_size=5000,
    stablemax=False, eval_offset=0.25,
    grad_checkpoint=False,
    mem_window=-1,  # -1=full history; 1=isolated; N=window
    null_kv=False,  # append fixed zero KV before softmax (abstain option)
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
    p.add_argument('--ocd',             action='store_true')
    p.add_argument('--ocd-prob',        type=str)
    p.add_argument('--grad-clip',       type=float)
    p.add_argument('--dataset-size',    type=int)
    p.add_argument('--eval-offset',     type=float)
    p.add_argument('--stablemax',       action='store_true')
    p.add_argument('--grad-checkpoint', action='store_true',
                   help='Depth-wise gradient checkpointing per block (saves inter-layer activations)')
    p.add_argument('--null-kv',         action='store_true',
                   help='Append fixed zero KV before softmax (abstain option)')
    p.add_argument('--n-blocks',        type=int)
    p.add_argument('--recall-from',     type=int)
    p.add_argument('--no-grok',         action='store_true')
    p.add_argument('--compile',         action='store_true')
    p.add_argument('--name',            type=str)
    p.add_argument('--name-date',       action='store_true')
    p.add_argument('--log-dir',         type=str, default='logs')
    p.add_argument('--device',          type=str, default='cpu',
                   choices=['cpu', 'mps', 'cuda'])
    p.add_argument('--eval-only',       type=str, default=None, metavar='CKPT',
                   help='Load checkpoint, run eval_configs once, print results and exit')
    p.add_argument('--resume',          type=str, default=None, metavar='CKPT',
                   help='Resume training from checkpoint (loads model weights + continues)')
    p.add_argument('--pretrained',      type=str, default=None, metavar='CKPT',
                   help='Load pretrained weights only (no optimizer/rng state — fresh training)')
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
    if args.ocd:                   hp['ocd'] = True
    if args.ocd_prob is not None:
        try:    hp['ocd_prob'] = json.loads(args.ocd_prob)
        except: hp['ocd_prob'] = float(args.ocd_prob)
    if args.name:                  hp['name'] = args.name
    if args.name_date:             hp['name_date'] = True
    if args.stablemax:             hp['stablemax'] = True
    if args.grad_checkpoint:       hp["grad_checkpoint"] = True
    if args.null_kv:               hp["null_kv"] = True
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

    if args.eval_only:
        # Load checkpoint, run eval once, exit
        import torch

        device = torch.device(args.device)
        ckpt = torch.load(args.eval_only, map_location=device)
        # Infer V_in from checkpoint weight shapes (robust to stale hp)
        sd = ckpt['model']
        V_in_ckpt  = sd['data_embed.weight'].shape[0] + sd['special_embed.weight'].shape[0]
        d_ckpt     = sd['data_embed.weight'].shape[1]
        n_lay_ckpt = sum(1 for k in sd if k.endswith('.norm1.weight'))
        _cur0 = hp.get('curriculum', [{}])[0]
        latent_len = _cur0.get('latent_len', hp.get('latent_len', hp.get('latent_len', 0)))
        ckpt_hp = {**hp, **ckpt.get('hp', {}),
                   'V': V_in_ckpt, 'd': d_ckpt, 'n_layers': n_lay_ckpt,
                   'd_ff': sd['blocks.0.ffn.W1.weight'].shape[0],
                   'latent_len': latent_len}
        model = build_model(ckpt_hp, device)
        model.load_state_dict(sd)
        model.eval()
        seg_len    = _cur0.get('seg_len',    hp.get('seg_len', 16))
        slot_len   = _cur0.get('slot_len',   hp.get('slot_len', 1))
        warmup_len = _cur0.get('warmup_len', hp.get('warmup_len', 4))
        out_len    = _cur0.get('out_len',    hp.get('out_len', 8))
        mem_window = _cur0.get('mem_window', hp.get('mem_window', -1))
        eval_offset = hp.get('eval_offset', 0.25)
        eval_cfgs  = hp.get('eval_configs', [(1, 0)])
        test_seqs  = make_test_sequences(seg_len)
        _eval_mode = 'ref' if any(s.get('mode') == 'ref'
                                   for s in hp.get('curriculum', [])) else 'end'
        _n_attempts_eval = _cur0.get('n_attempts', _cur0.get('n_draft_turns', 1)) + 2
        print(f'Checkpoint: {args.eval_only}  (stage={ckpt.get("stage","?")}  step={ckpt.get("step","?")})')
        for nb, rf in eval_cfgs:
            all_c = []
            all_drafts = [[] for _ in range(_n_attempts_eval)] if _eval_mode == 'ref' else []
            f_start = min(int(seg_len * eval_offset), seg_len - warmup_len - out_len)
            y_start = f_start + warmup_len; y_end = min(y_start + out_len, seg_len)
            for sname, x_S in test_seqs.items():
                wm = x_S[max(0, y_start - warmup_len):y_start]
                if len(wm) < warmup_len: wm = [x_S[0]] * (warmup_len - len(wm)) + list(wm)
                tgt = x_S[y_start:y_end]
                if _eval_mode == 'ref':
                    drafts, g = ar_decode_refine(model, x_S, slot_len, wm, y_end - y_start,
                                                 device, n_attempts=_n_attempts_eval,
                                                 latent_len=latent_len, mem_window=mem_window)
                    for k, d in enumerate(drafts):
                        if k < len(all_drafts): all_drafts[k].append(cer(d, tgt))
                else:
                    g = ar_decode_role(model, x_S, slot_len, wm, y_end - y_start, device,
                                       n_blocks=nb, recall_from=rf,
                                       latent_len=latent_len, mem_window=mem_window)
                all_c.append(cer(g, tgt))
            mean_c = sum(all_c) / len(all_c)
            if all_drafts:
                turn_m = [round(100*(1-sum(c)/len(c)),1) for c in all_drafts]
                final_m = round(100*(1-mean_c), 1)
                step_str = '  '.join(f't{k+1}={m}%' for k,m in enumerate(turn_m))
                mono = '↑' if all(turn_m[i]<=turn_m[i+1] if i+1<len(turn_m) else turn_m[-1]<=final_m
                                   for i in range(len(turn_m))) else '⚠'
                print(f'  {mono} n={nb} rf={rf}  {step_str}  final={final_m}%  Δ={final_m-turn_m[0]:+.1f}%')
            else:
                print(f'  n={nb} rf={rf}: match={100*(1-mean_c):.1f}%')
        import sys; sys.exit(0)

    if args.resume:
        hp['_resume_ckpt'] = args.resume
    if args.pretrained:
        hp['_pretrained_ckpt'] = args.pretrained

    train_role(hp, log_base=args.log_dir, device_str=args.device)
