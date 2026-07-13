"""
experiments/attn_dual/train.py — dual-attention-block ablation (no MLP
anywhere, attn+attn per block instead of attn+ffn) vs the proven 3-window
64B stitched SRS baseline (experiments/srs_tagged/configs/srs_stitch_nc4_slot8.py).

Reuses chunk_positions_srs_tagged / make_batch_tagged / _fill_argmax_fb /
chunk_mask_fb completely unchanged from the tagged-stitching track. Only the
model (DualAttnModel, experiments/attn_dual/model.py) and the eval decode
(no-KV-cache full recompute, experiments/attn_dual/decode.py) differ — see
those files' docstrings for why.

No warm-start: MLP removal changes the architecture (different state_dict
keys), so this always trains from scratch — a fair from-scratch vs
from-scratch comparison would need the baseline re-run from scratch too, but
the existing srs_stitch_nc4_slot8 result (100%/100% sustained, warm-started)
is the target ceiling to compare against regardless of starting point, since
the interesting question is "can dual-attn reach/approach that ceiling at
all" not "does warm-starting help it."
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

from kvmem.train_hmn_chunk import chunk_mask_fb, _StatusWriter
from kvmem.train_hmn_mono import _positional_ls_nll, load_config
from kvmem.utils import make_test_sequences

from experiments.chat_tags.positions import chunk_positions_srs_tagged
from experiments.chat_tags.batch import make_batch_tagged, _fill_argmax_fb
from experiments.attn_dual.model import build_dualattn_model
from experiments.attn_dual.decode import ar_decode_srs_stitched_tagged_nokv


def train(hp: dict, log_base: str = 'logs', device_str: str = 'cpu'):
    device = torch.device(device_str)
    rng    = np.random.default_rng(hp.get('seed', 42))
    torch.manual_seed(hp.get('seed', 42))

    name     = hp.get('name', 'attn_dual')
    log_dir  = os.path.join(log_base, name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_file    = open(os.path.join(log_dir, 'train.log'),    'a', buffering=1)
    jsonl_file  = open(os.path.join(log_dir, 'train.jsonl'), 'a', buffering=1)
    status_file = _StatusWriter(os.path.join(log_dir, 'train_status.log'))

    def _log(msg): print(msg); print(msg, file=log_file)
    def _jlog(d):  jsonl_file.write(json.dumps(d) + '\n')

    hp_model = dict(V=hp['V'], d=hp['d'], n_layers=hp['n_layers'],
                    n_heads=hp['n_heads'], rope=hp.get('rope', True),
                    yarn=hp.get('yarn', True), null_kv=hp.get('null_kv', True),
                    rmsnorm=hp.get('rmsnorm', False), chunk_attn=hp.get('chunk_attn', 0))
    model    = build_dualattn_model(hp_model, device)
    n_params = sum(p.numel() for p in model.parameters())
    _log(f'Model: {n_params:,} params  device={device}  V={hp_model["V"]}  (attn_dual ablation, no MLP)')

    if hp.get('_pretrained_ckpt'):
        # Same-architecture staged warm-start: mirrors this project's proven IQ-then-IR
        # curriculum (every prior success in this codebase — hmn_feedback_32_iq->_ir,
        # chat_tags Phase A->B, srs_depth2_nc4_slot8->srs_stitch_nc4_slot8 — warm-starts
        # a harder stage from an easier stage's checkpoint of the SAME architecture,
        # not a cross-architecture weight transplant). Here: dualattn_nc4_slot8_iq.py
        # (IQ-only, n_refine=0) -> dualattn_nc4_slot8_ir.py (this stage, n_refine=2).
        # All keys match exactly (both stages use DualAttnModel), so this is a full
        # state_dict load; the shape-matching loop is kept generic in case a partial
        # load is ever needed for a different lineage.
        # Also handles vocab growth (e.g. nc4->nc8 adding window D-G tags):
        # a growing tensor (e.g. special_embed.weight) gets its overlapping
        # PREFIX copied rather than being skipped outright — same mechanism
        # already proven in experiments/srs_tagged/train.py for the standard
        # architecture's own nc4->nc8 vocab growth.
        ckpt = torch.load(hp['_pretrained_ckpt'], map_location=device)
        src_sd = ckpt['model']
        dst_sd = model.state_dict()
        loaded, grown = [], []
        for k in dst_sd:
            if k not in src_sd:
                continue
            src_t, dst_t = src_sd[k], dst_sd[k]
            if src_t.shape == dst_t.shape:
                dst_sd[k] = src_t; loaded.append(k)
            elif src_t.dim() >= 1 and src_t.shape[1:] == dst_t.shape[1:] and src_t.shape[0] < dst_t.shape[0]:
                dst_sd[k][:src_t.shape[0]] = src_t
                grown.append(f'{k}: {tuple(src_t.shape)}->{tuple(dst_t.shape)}')
            else:
                raise RuntimeError(f'Unhandled shape mismatch for {k}: {src_t.shape} vs {dst_t.shape}')
        model.load_state_dict(dst_sd)
        _log(f'Loaded (staged warm-start): {len(loaded)}/{len(dst_sd)} tensors '
             f'from {hp["_pretrained_ckpt"]}' + (f'  (grown: {grown})' if grown else ''))

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
        B          = stage.get('B', 8)
        n_steps    = stage.get('n_steps', 60000)
        stage_eval_every = stage.get('eval_every', 5000)
        ls_max     = hp.get('ls_max', 0.0)
        windows    = stage['windows']

        built = chunk_positions_srs_tagged(n_chunks, chunk_len, slot_len, warmup_len,
                                           windows, n_refine=n_refine)
        pos_content, pos_mask, tags, L = (built['pos_content'], built['pos_mask'],
                                          built['tags'], built['L'])
        mask_np = chunk_mask_fb(pos_mask)
        mask_t  = torch.tensor(mask_np, dtype=torch.float32, device=device)

        _log(f'\n[stage {stage_i}] n_chunks={n_chunks} chunk_len={chunk_len} slot={slot_len} '
             f'wl={warmup_len} windows={windows} n_refine={n_refine} '
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

        test_chunks = None
        if eval_file:
            from kvmem.train_hmn_chunk import load_chunks_padded
            try:
                test_chunks, _ = load_chunks_padded(eval_file, n_chunks, chunk_len)
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

                span_last_idx: dict[tuple, int] = {}
                for i, rb in enumerate(pos_content['rec_blocks']):
                    span_last_idx[rb['span']] = i

                def _eval_on(seqs_iter, tag_prefix):
                    span_means = {span: [] for span in windows}
                    stitched_means = []
                    for sname, chunks_arr in seqs_iter:
                        r = ar_decode_srs_stitched_tagged_nokv(model, chunks_arr, slot_len, slot_count,
                                                               mask_np, pos_content, tags, device)
                        stitched_means.append(r['match_pct'])
                        for span in windows:
                            idx = span_last_idx[span]
                            span_means[span].append(r['turn_match_pcts'][idx])
                        _log(f'  {tag_prefix}/{sname:<15} per-span={[round(r["turn_match_pcts"][span_last_idx[s]],1) for s in windows]}  stitched={r["match_pct"]:.1f}%')
                    means = []
                    for span in windows:
                        m_ = sum(span_means[span]) / len(span_means[span])
                        means.append(m_)
                        _log(f'  {tag_prefix}/span{span}/MEAN               match={m_:.1f}%')
                    overall = sum(means) / len(means)
                    _log(f'  {tag_prefix}/MEAN               match={overall:.1f}%')
                    stitched_overall = sum(stitched_means) / len(stitched_means)
                    _log(f'  {tag_prefix}/STITCHED_MEAN               match={stitched_overall:.1f}%')
                    return stitched_overall

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
    p.add_argument('--config', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--pretrained', default=None)
    p.add_argument('--log-dir', default='experiments/attn_dual/logs')
    args = p.parse_args()

    hp = load_config(args.config)
    if args.pretrained:
        hp['_pretrained_ckpt'] = args.pretrained
    train(hp, log_base=args.log_dir, device_str=args.device)


if __name__ == '__main__':
    main()
