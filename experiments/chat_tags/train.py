"""
experiments/chat_tags/train.py — training entry point for the iq_global_rw_tagged
trajectory (explicit <src>/<mem>/<query>/<response> boundary tokens).

Self-contained: does not fork kvmem/train_hmn_chunk.py's full multi-trajectory
training loop. Reuses via plain import whatever works unmodified against the
tagged pos dicts (chunk_mask_fb, build_model, _positional_ls_nll, eval utils);
everything tag-specific lives in this experiment folder.

See /Users/muaz/.claude/plans/design-experiment-which-use-atomic-kay.md for the
full design.

Usage:
    python -m experiments.chat_tags.train \\
        --config experiments/chat_tags/configs/slot8_tagged_phaseA_iq.py \\
        --device mps
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from kvmem.model import build_model
from kvmem.train_hmn_chunk import chunk_mask_fb, _StatusWriter
from kvmem.train_hmn_mono import _positional_ls_nll, load_config
from kvmem.utils import make_test_sequences

from experiments.chat_tags.vocab import HMN_TAG_VOCAB_SIZE
from experiments.chat_tags.positions import chunk_positions_iq_global_rw_tagged
from experiments.chat_tags.batch import make_batch_tagged, ar_decode_iq_global_rw_tagged, _fill_argmax_fb


def train(hp: dict, log_base: str = 'logs', device_str: str = 'cpu'):
    device = torch.device(device_str)
    rng    = np.random.default_rng(hp.get('seed', 42))
    torch.manual_seed(hp.get('seed', 42))

    name     = hp.get('name', 'chat_tags')
    log_dir  = os.path.join(log_base, name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_file    = open(os.path.join(log_dir, 'train.log'),    'a', buffering=1)
    jsonl_file  = open(os.path.join(log_dir, 'train.jsonl'), 'a', buffering=1)
    status_file = _StatusWriter(os.path.join(log_dir, 'train_status.log'))

    def _log(msg): print(msg); print(msg, file=log_file)
    def _jlog(d):  jsonl_file.write(json.dumps(d) + '\n')

    hp_model = dict(V=hp.get('V', HMN_TAG_VOCAB_SIZE),
                    d=hp['d'], n_layers=hp['n_layers'],
                    n_heads=hp['n_heads'], d_ff=hp['d_ff'],
                    rope=hp.get('rope', True), yarn=hp.get('yarn', True),
                    null_kv=hp.get('null_kv', True), compile=hp.get('compile', False),
                    chunk_attn=hp.get('chunk_attn', 0))
    model    = build_model(hp_model, device)
    n_params = sum(p.numel() for p in model.parameters())
    _log(f'Model: {n_params:,} params  device={device}  V={hp_model["V"]}  (chat_tags experiment)')

    if hp.get('_pretrained_ckpt'):
        ckpt = torch.load(hp['_pretrained_ckpt'], map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        _log(f'Loaded: {hp["_pretrained_ckpt"]}')

    lr_max  = hp.get('lr_max', 3e-4)
    wd      = hp.get('wd', 0.001)
    warmup_steps = hp.get('warmup_steps', 500)
    use_actual_am = hp.get('use_actual_argmax', True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=wd, betas=(0.9, 0.999))

    curriculum = hp.get('curriculum', [])
    assert curriculum
    log_every  = hp.get('log_every', 500)

    global_step = 0
    t_start = time.time()

    for stage_i, stage in enumerate(curriculum):
        n_chunks   = stage['n_chunks']
        chunk_len  = stage['chunk_len']
        slot_len   = hp.get('slot_len', 8)
        slot_count = hp.get('slot_count', 2)
        warmup_len = hp.get('warmup_len', 8)
        window_chunks = stage.get('window_chunks', 2)
        B          = stage.get('B', 8)
        n_steps    = stage.get('n_steps', 50000)
        stage_eval_every = stage.get('eval_every', 5000)
        ls_max     = hp.get('ls_max', 0.0)

        # traj_mix: list of dicts each {weight, n_refine, warmup_x_fixed, warmup_x_dist}.
        # Falls back to a single traj built from stage['n_refine'] (Phase A style).
        traj_mix_cfg = stage.get('traj_mix')
        if traj_mix_cfg is None:
            traj_mix_cfg = [dict(weight=1.0, n_refine=stage.get('n_refine', 0))]

        trajectories = []
        for tcfg in traj_mix_cfg:
            t_n_refine = tcfg.get('n_refine', 0)
            built = chunk_positions_iq_global_rw_tagged(
                n_chunks, chunk_len, slot_len, warmup_len,
                window_chunks=window_chunks,
                warmup_x_fixed=tcfg.get('warmup_x_fixed'),
                warmup_x_dist=tcfg.get('warmup_x_dist', 'uniform'),
                n_refine=t_n_refine)
            pos_content, pos_mask, tags, L = (built['pos_content'], built['pos_mask'],
                                              built['tags'], built['L'])
            mask_np = chunk_mask_fb(pos_mask)
            mask_t  = torch.tensor(mask_np, dtype=torch.float32, device=device)
            trajectories.append(dict(weight=tcfg['weight'], n_refine=t_n_refine,
                                     pos_content=pos_content, mask_np=mask_np, mask_t=mask_t,
                                     tags=tags, L=L, has_ir=t_n_refine > 0,
                                     warmup_x_fixed=tcfg.get('warmup_x_fixed')))
        traj_weights = np.array([t['weight'] for t in trajectories], dtype=np.float64)
        traj_weights = traj_weights / traj_weights.sum()

        # Eval trajectory: the one with the highest n_refine (matches project
        # convention — "Eval uses first IR trajectory" — full IQ+IR chain is
        # the most informative report even though IQ-only/lower-refine steps
        # also happen during training).
        eval_traj = max(trajectories, key=lambda t: t['n_refine'])

        _log(f'\n[stage {stage_i}] n_chunks={n_chunks} chunk_len={chunk_len} '
             f'slot={slot_len} wl={warmup_len} B={B}  steps={n_steps}  '
             f'traj_mix={[(round(w,2), t["n_refine"], t.get("warmup_x_fixed")) for t, w in zip(trajectories, traj_weights)]}  '
             f'L(eval)={eval_traj["L"]}')

        lr_min      = hp.get('lr_min', 0.0)
        cosine_T0   = hp.get('cosine_T0', 20000)
        cosine_Tmul = hp.get('cosine_T_mult', 1)
        lr_schedule = hp.get('lr_schedule', 'constant')

        def _lr(s):
            if s <= warmup_steps:
                return lr_max * s / max(warmup_steps, 1)
            if lr_schedule != 'cosine_restarts':
                return lr_max
            t = s - warmup_steps
            T_i = cosine_T0
            while t >= T_i:
                t -= T_i
                T_i = int(T_i * cosine_Tmul)
            return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t / max(T_i, 1)))

        val_seg_len = n_chunks * chunk_len
        val_seqs    = make_test_sequences(val_seg_len)
        val_n_seqs  = hp.get('val_n_seqs')
        if val_n_seqs is not None:
            val_seqs = dict(list(val_seqs.items())[:val_n_seqs])

        stage_best_val = -1.0
        pbar = tqdm(range(1, n_steps + 1), desc=f'stage{stage_i}', dynamic_ncols=True, file=status_file)
        for local_step in pbar:
            global_step += 1
            lr = _lr(local_step)
            for pg in opt.param_groups: pg['lr'] = lr

            model.train(); opt.zero_grad()

            traj = trajectories[rng.choice(len(trajectories), p=traj_weights)]
            t_pos_content, t_mask_t, t_tags, t_has_ir = (traj['pos_content'], traj['mask_t'],
                                                          traj['tags'], traj['has_ir'])

            tok_np = make_batch_tagged(rng, B, n_chunks, chunk_len, slot_len, slot_count,
                                       t_pos_content, t_tags)
            tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)

            if use_actual_am and t_has_ir:
                with torch.no_grad():
                    logits_1 = model(tok_t, t_mask_t)
                tok_np = _fill_argmax_fb(tok_np, logits_1, t_pos_content)
                tok_t  = torch.tensor(tok_np, device=device, dtype=torch.long)

            logits = model(tok_t, t_mask_t)
            nlls = []
            for rb in t_pos_content['rec_blocks']:
                if not rb['is_clean']:
                    continue
                lp  = F.log_softmax(logits[:, rb['c0']-1:rb['c1']-1], dim=-1)
                tgt = tok_t[:, rb['c0']:rb['c1']]
                nll_per = _positional_ls_nll(lp, tgt, ls_max)
                nlls.append(nll_per.mean())
            loss = torch.stack(nlls).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_f = float(loss.detach())
            pbar.set_postfix(loss=f'{loss_f:.3f}', lr=f'{lr:.1e}',
                             refine=traj['n_refine'], refresh=False)
            if local_step % log_every == 0:
                _jlog(dict(step=global_step, loss=round(loss_f, 5), lr=lr, n_refine=traj['n_refine']))
                print(str(pbar), file=log_file, flush=True)

            if local_step % stage_eval_every == 0 or local_step == n_steps:
                model.eval()
                elapsed = time.time() - t_start
                h, m = divmod(int(elapsed), 3600); m, s = divmod(m, 60)
                _log(f'\n--- stage={stage_i} step={local_step}/{n_steps}'
                     f'  g={global_step}  loss={loss_f:.4f}  {h:02d}:{m:02d}:{s:02d} ---')

                e_pos_content, e_mask_np, e_tags = (eval_traj['pos_content'], eval_traj['mask_np'],
                                                    eval_traj['tags'])
                rw_rb = e_pos_content['rec_blocks'][0]
                rw_valid_offsets = rw_rb['warmup_valid_offsets']
                window_means = []
                for X in rw_valid_offsets:
                    ws = X // chunk_len
                    we = ws + window_chunks
                    seq_results = []
                    for sname, seq in val_seqs.items():
                        chunks_arr = np.array(
                            [seq[k*chunk_len:(k+1)*chunk_len] for k in range(n_chunks)], np.int64)
                        r = ar_decode_iq_global_rw_tagged(model, chunks_arr, slot_len, slot_count,
                                                          e_mask_np, e_pos_content, e_tags, device,
                                                          warmup_offset=X)
                        seq_results.append(r['match_pct'])
                        tpcts = r.get('turn_match_pcts', [])
                        n_turns = len(tpcts)
                        if n_turns > 1:
                            turn_names = ['IQ'] + [f'IR{i}' for i in range(1, n_turns)]
                            turns_str = '  '.join(f'{tn}={p:.1f}%' for tn, p in zip(turn_names, tpcts))
                            _log(f'  val/win({ws},{we})_nc{n_chunks}/{sname:<15} BPB={r["bpb"]:.3f}  match={r["match_pct"]:.1f}%  [{turns_str}]')
                        else:
                            _log(f'  val/win({ws},{we})_nc{n_chunks}/{sname:<15} BPB={r["bpb"]:.3f}  match={r["match_pct"]:.1f}%')
                    win_mean = sum(seq_results) / len(seq_results)
                    window_means.append(win_mean)
                    _log(f'  val/win({ws},{we})_nc{n_chunks}/MEAN               match={win_mean:.1f}%')
                vmean = sum(window_means) / len(window_means)
                _log(f'  val/iq_global_rw_tagged/MEAN               match={vmean:.1f}%')
                _jlog(dict(step=global_step, eval_mean=round(vmean, 2)))

                torch.save(dict(model=model.state_dict(), hp=hp, step=global_step),
                          os.path.join(ckpt_dir, f'stage{stage_i}_last.pt'))
                if vmean > stage_best_val:
                    stage_best_val = vmean
                    torch.save(dict(model=model.state_dict(), hp=hp, step=global_step, val_mean=vmean),
                              os.path.join(ckpt_dir, f'stage{stage_i}_best.pt'))

        torch.save(dict(model=model.state_dict(), hp=hp, step=global_step),
                  os.path.join(ckpt_dir, f'stage{stage_i}_end.pt'))
        _log(f'[stage {stage_i}] done. saved stage{stage_i}_end.pt (best={stage_best_val:.1f}%)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     required=True)
    p.add_argument('--device',     default='cpu')
    p.add_argument('--pretrained', default=None)
    p.add_argument('--log-dir',    default='logs')
    args = p.parse_args()

    hp = load_config(args.config)
    if args.pretrained:
        hp['_pretrained_ckpt'] = args.pretrained
    train(hp, log_base=args.log_dir, device_str=args.device)


if __name__ == '__main__':
    main()
