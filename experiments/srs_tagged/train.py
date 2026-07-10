"""
experiments/srs_tagged/train.py — true SRS (spaced-repetition span scheduling)
with chat-tags' proven fixes: span-specific query tags + wrong-token-weighted
IR loss. See docs/SRS_RECIPE.md § "Resuming true SRS" for the full design.

Unlike experiments/chat_tags/train.py, there is no traj_mix/weighted-trajectory
sampling — every training batch IS the full SRS schedule (one sequence spans
every review span in order), so there's exactly one trajectory per stage.
Reuses experiments/chat_tags/batch.py's make_batch_tagged, _fill_argmax_fb, and
ar_decode_iq_global_rw_tagged completely unchanged — all three are generic over
pos_content['rec_blocks'] and already handle per-span (not just per-window)
structure correctly (see docs/SRS_RECIPE.md for why no changes were needed).

Adds the held-out test-file eval path (load_chunks_padded + eval_file) that the
original (pre-chat-tags) SRS configs used but chat-tags never wired in — kept
here since "reuse similar val and test" was explicit in the design ask.

Usage:
    python -m experiments.srs_tagged.train \\
        --config experiments/srs_tagged/configs/srs_depth2_nc4_slot8.py \\
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
from kvmem.train_hmn_chunk import (
    chunk_mask_fb, _StatusWriter, srs_schedule, srs_schedule_depth2, load_chunks_padded,
)
from kvmem.train_hmn_mono import _positional_ls_nll, load_config
from kvmem.utils import make_test_sequences

from experiments.chat_tags.vocab import HMN_TAG_VOCAB_SIZE_V2
from experiments.chat_tags.positions import chunk_positions_srs_tagged
from experiments.chat_tags.batch import make_batch_tagged, _fill_argmax_fb, ar_decode_iq_global_rw_tagged
from experiments.srs_tagged.stitch_decode import ar_decode_srs_stitched_tagged


def train(hp: dict, log_base: str = 'logs', device_str: str = 'cpu'):
    device = torch.device(device_str)
    rng    = np.random.default_rng(hp.get('seed', 42))
    torch.manual_seed(hp.get('seed', 42))

    name     = hp.get('name', 'srs_tagged')
    log_dir  = os.path.join(log_base, name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_file    = open(os.path.join(log_dir, 'train.log'),    'a', buffering=1)
    jsonl_file  = open(os.path.join(log_dir, 'train.jsonl'), 'a', buffering=1)
    status_file = _StatusWriter(os.path.join(log_dir, 'train_status.log'))

    def _log(msg): print(msg); print(msg, file=log_file)
    def _jlog(d):  jsonl_file.write(json.dumps(d) + '\n')

    hp_model = dict(V=hp.get('V', HMN_TAG_VOCAB_SIZE_V2),
                    d=hp['d'], n_layers=hp['n_layers'],
                    n_heads=hp['n_heads'], d_ff=hp['d_ff'],
                    rope=hp.get('rope', True), yarn=hp.get('yarn', True),
                    null_kv=hp.get('null_kv', True), compile=hp.get('compile', False),
                    chunk_attn=hp.get('chunk_attn', 0))
    model    = build_model(hp_model, device)
    n_params = sum(p.numel() for p in model.parameters())
    _log(f'Model: {n_params:,} params  device={device}  V={hp_model["V"]}  (srs_tagged experiment)')

    if hp.get('_pretrained_ckpt'):
        ckpt = torch.load(hp['_pretrained_ckpt'], map_location=device)
        src_sd = ckpt['model']
        dst_sd = model.state_dict()
        grown = []
        for k, dst_t in dst_sd.items():
            if k not in src_sd:
                continue
            src_t = src_sd[k]
            if src_t.shape == dst_t.shape:
                dst_sd[k] = src_t
            elif src_t.dim() >= 1 and src_t.shape[1:] == dst_t.shape[1:] and src_t.shape[0] < dst_t.shape[0]:
                dst_sd[k][:src_t.shape[0]] = src_t
                grown.append(f'{k}: {tuple(src_t.shape)}->{tuple(dst_t.shape)}')
            else:
                raise RuntimeError(f'Unhandled shape mismatch for {k}: {src_t.shape} vs {dst_t.shape}')
        model.load_state_dict(dst_sd)
        _log(f'Loaded: {hp["_pretrained_ckpt"]}' + (f'  (grown: {grown})' if grown else ''))

    lr_max  = hp.get('lr_max', 3e-4)
    wd      = hp.get('wd', 0.001)
    warmup_steps = hp.get('warmup_steps', 500)
    use_actual_am = hp.get('use_actual_argmax', True)
    wrong_token_weight = hp.get('wrong_token_weight', 0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=wd, betas=(0.9, 0.999))

    curriculum = hp.get('curriculum', [])
    assert curriculum
    log_every  = hp.get('log_every', 500)
    eval_file  = hp.get('eval_file', None)

    global_step = 0
    t_start = time.time()

    for stage_i, stage in enumerate(curriculum):
        n_chunks   = stage['n_chunks']
        chunk_len  = stage['chunk_len']
        slot_len   = hp.get('slot_len', 8)
        slot_count = hp.get('slot_count', 2)
        warmup_len = hp.get('warmup_len', 8)
        n_refine   = stage.get('n_refine', 2)
        depth      = stage.get('depth', 2)
        B          = stage.get('B', 8)
        n_steps    = stage.get('n_steps', 60000)
        stage_eval_every = stage.get('eval_every', 5000)
        ls_max     = hp.get('ls_max', 0.0)
        eval_mode  = stage.get('eval_mode', 'per_span')  # 'per_span' (GT-seeded, atomic) or 'stitch' (chained)

        # 'windows' overrides the depth-based srs_schedule/srs_schedule_depth2
        # generator with an explicit (possibly overlapping) window list — used
        # by the stitched track (see docs/SRS_RECIPE.md "Stitching vs atomic
        # full-span"). Falls back to the original depth-based schedules
        # unchanged when absent.
        if 'windows' in stage:
            schedule = stage['windows']
        else:
            schedule = srs_schedule_depth2(n_chunks) if depth == 2 else srs_schedule(n_chunks)
        built = chunk_positions_srs_tagged(n_chunks, chunk_len, slot_len, warmup_len,
                                           schedule, n_refine=n_refine)
        pos_content, pos_mask, tags, L = (built['pos_content'], built['pos_mask'],
                                          built['tags'], built['L'])
        mask_np = chunk_mask_fb(pos_mask)
        mask_t  = torch.tensor(mask_np, dtype=torch.float32, device=device)

        _log(f'\n[stage {stage_i}] n_chunks={n_chunks} chunk_len={chunk_len} slot={slot_len} '
             f'wl={warmup_len} depth={depth} schedule={schedule} n_refine={n_refine} '
             f'B={B}  steps={n_steps}  L={L}')

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

        test_chunks = test_valid = None
        if eval_file:
            try:
                test_chunks, test_valid = load_chunks_padded(eval_file, n_chunks, chunk_len)
            except Exception as e:
                _log(f'  [test eval disabled: {e}]')

        stage_best_val = -1.0
        pbar = tqdm(range(1, n_steps + 1), desc=f'stage{stage_i}', dynamic_ncols=True, file=status_file)
        for local_step in pbar:
            global_step += 1
            lr = _lr(local_step)
            for pg in opt.param_groups: pg['lr'] = lr

            model.train(); opt.zero_grad()

            tok_np = make_batch_tagged(rng, B, n_chunks, chunk_len, slot_len, slot_count,
                                       pos_content, tags)
            tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)

            wrong_masks: dict[int, np.ndarray] = {}
            if use_actual_am:
                with torch.no_grad():
                    logits_1 = model(tok_t, mask_t)
                if wrong_token_weight > 0:
                    for i, rb in enumerate(pos_content['rec_blocks']):
                        if rb['type'] != 'ir':
                            continue
                        src_c0 = rb['argmax_src_c0']
                        wrong_masks[i] = tok_np[:, src_c0:src_c0 + rb['out_len']].copy()
                tok_np = _fill_argmax_fb(tok_np, logits_1, pos_content)
                if wrong_token_weight > 0:
                    for i, rb in enumerate(pos_content['rec_blocks']):
                        if i not in wrong_masks:
                            continue
                        wrong_masks[i] = (tok_np[:, rb['am0']:rb['am1']] != wrong_masks[i])
                tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)

            logits = model(tok_t, mask_t)
            nlls = []
            for i, rb in enumerate(pos_content['rec_blocks']):
                if not rb['is_clean']:
                    continue
                lp  = F.log_softmax(logits[:, rb['c0']-1:rb['c1']-1], dim=-1)
                tgt = tok_t[:, rb['c0']:rb['c1']]
                nll_per = _positional_ls_nll(lp, tgt, ls_max)
                if i in wrong_masks:
                    w = 1.0 + wrong_token_weight * wrong_masks[i].astype(np.float32)
                    nll_per = nll_per * torch.tensor(w, device=device, dtype=torch.float32)
                nlls.append(nll_per.mean())
            loss = torch.stack(nlls).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_f = float(loss.detach())
            pbar.set_postfix(loss=f'{loss_f:.3f}', lr=f'{lr:.1e}', refresh=False)
            if local_step % log_every == 0:
                _jlog(dict(step=global_step, loss=round(loss_f, 5), lr=lr))
                print(str(pbar), file=log_file, flush=True)

            if local_step % stage_eval_every == 0 or local_step == n_steps:
                model.eval()
                elapsed = time.time() - t_start
                h, m = divmod(int(elapsed), 3600); m, s = divmod(m, 60)
                _log(f'\n--- stage={stage_i} step={local_step}/{n_steps}'
                     f'  g={global_step}  loss={loss_f:.4f}  {h:02d}:{m:02d}:{s:02d} ---')

                # Which rec_block index is each span's OWN final (last) block?
                span_last_idx: dict[tuple, int] = {}
                for i, rb in enumerate(pos_content['rec_blocks']):
                    span_last_idx[rb['span']] = i  # later entries overwrite -> last one wins

                def _eval_on(seqs_iter, tag_prefix):
                    span_means = {span: [] for span in schedule}
                    stitched_means = []
                    for sname, chunks_arr in seqs_iter:
                        if eval_mode == 'stitch':
                            r = ar_decode_srs_stitched_tagged(model, chunks_arr, slot_len, slot_count,
                                                              mask_np, pos_content, tags, device)
                            stitched_means.append(r['match_pct'])
                        else:
                            r = ar_decode_iq_global_rw_tagged(model, chunks_arr, slot_len, slot_count,
                                                              mask_np, pos_content, tags, device,
                                                              warmup_offset=0)
                        for span in schedule:
                            idx = span_last_idx[span]
                            span_means[span].append(r['turn_match_pcts'][idx])
                        tail = f'  stitched={r["match_pct"]:.1f}%' if eval_mode == 'stitch' else ''
                        _log(f'  {tag_prefix}/{sname:<15} per-span={[round(r["turn_match_pcts"][span_last_idx[s]],1) for s in schedule]}{tail}')
                    means = []
                    for span in schedule:
                        m_ = sum(span_means[span]) / len(span_means[span])
                        means.append(m_)
                        _log(f'  {tag_prefix}/span{span}/MEAN               match={m_:.1f}%')
                    overall = sum(means) / len(means)
                    _log(f'  {tag_prefix}/MEAN               match={overall:.1f}%')
                    if eval_mode == 'stitch':
                        stitched_overall = sum(stitched_means) / len(stitched_means)
                        _log(f'  {tag_prefix}/STITCHED_MEAN               match={stitched_overall:.1f}%')
                        return stitched_overall
                    return overall

                val_iter = ((sname, np.array([seq[k*chunk_len:(k+1)*chunk_len] for k in range(n_chunks)], np.int64))
                           for sname, seq in val_seqs.items())
                vmean = _eval_on(val_iter, 'val/srs')

                if test_chunks is not None:
                    test_iter = iter([('test', test_chunks)])
                    _eval_on(test_iter, 'test/srs')

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
    p.add_argument('--log-dir',    default='experiments/srs_tagged/logs')
    args = p.parse_args()

    hp = load_config(args.config)
    if args.pretrained:
        hp['_pretrained_ckpt'] = args.pretrained
    train(hp, log_base=args.log_dir, device_str=args.device)


if __name__ == '__main__':
    main()
