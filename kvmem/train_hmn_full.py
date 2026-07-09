"""
kvmem/train_hmn_full.py — Full-continuation IQ memorization with variable source length.

Task: given source bytes of random length (nc * chunk_len, nc ~ Uniform[min_nc, max_nc]),
encode into SLOT tokens, then recall from any warmup offset X to end-of-source.

Key differences from train_hmn_chunk.py:
  - Source length varies per batch (nc sampled uniformly each step)
  - Output always continues to end-of-source (variable out_len, padded + CE masked)
  - Cosine LR, no restarts (one smooth decay to lr_min)
  - model2.py (RMSNorm, no bias, depth-scaled init)
  - Eval at each nc in eval_ncs to show position/scale invariance

Run:
    python3 -m kvmem.train_hmn_full --config configs/hmn_full_vlen_s0.py --device mps
"""

from __future__ import annotations
import argparse, copy, importlib.util, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from kvmem.model2 import build_model2
from kvmem.train_hmn_chunk import (
    chunk_positions_iq_global_rw_full,
    chunk_mask_fb,
    _chunk_make_batch_fb,
    ar_decode_chunk_fb_kv,
    _slot_ids,
    _fmt_cmp,
)
from kvmem.train_hmn_mono import _positional_ls_nll
from kvmem.utils import make_test_sequences


# ---------------------------------------------------------------------------
# LR schedule — single cosine decay, no restarts
# ---------------------------------------------------------------------------

def make_lr_fn(lr_max: float, lr_min: float, warmup_steps: int, total_steps: int):
    def _lr(step: int) -> float:
        if step <= warmup_steps:
            return lr_max * step / max(warmup_steps, 1)
        t = step - warmup_steps
        T = max(total_steps - warmup_steps, 1)
        return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t / T))
    return _lr


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_full(hp: dict, log_base: str = 'logs', device_str: str = 'cpu',
               pretrained: str | None = None):
    device = torch.device(device_str)

    # Hyper-params
    name        = hp['name']
    d           = hp['d']
    n_layers    = hp['n_layers']
    n_heads     = hp['n_heads']
    d_ff        = hp['d_ff']
    V           = hp.get('V', 268)
    chunk_len   = hp.get('chunk_len', 16)
    slot_len    = hp.get('slot_len', 8)
    slot_count  = hp.get('slot_count', 2)
    warmup_len  = hp.get('warmup_len', 8)
    min_nc      = hp.get('min_nc', 1)
    max_nc      = hp.get('max_nc', 4)
    B           = hp.get('B', 8)
    n_steps     = hp.get('n_steps', 100000)
    eval_every  = hp.get('eval_every', 20000)
    log_every   = hp.get('log_every', 1000)
    lr_max      = hp.get('lr_max', 3e-4)
    lr_min      = hp.get('lr_min', 1e-6)
    warmup_steps = hp.get('warmup_steps', 2000)
    wd          = hp.get('wd', 0.001)
    ls_max      = hp.get('label_smooth', 0.0)
    seed        = hp.get('seed', 42)
    val_n_seqs  = hp.get('val_n_seqs', None)
    chunk_attn  = hp.get('chunk_attn', 256)

    # Curriculum: list of (step_end, max_nc, eval_ncs).
    # If present, max_nc / eval_ncs from hp are ignored; n_steps taken from last stage.
    nc_curriculum = hp.get('nc_curriculum', None)
    if nc_curriculum is not None:
        global_max_nc = nc_curriculum[-1][1]
        n_steps = nc_curriculum[-1][0]
        def _get_stage(step):
            # Returns (stage_max_nc, eval_ncs_all_stages_so_far) for regression testing.
            seen_eval_ncs = []
            for stage_end, stage_max_nc, stage_eval_ncs in nc_curriculum:
                for enc_nc in stage_eval_ncs:
                    if enc_nc not in seen_eval_ncs:
                        seen_eval_ncs.append(enc_nc)
                if step <= stage_end:
                    return stage_max_nc, sorted(seen_eval_ncs)
            return nc_curriculum[-1][1], sorted(seen_eval_ncs)
    else:
        global_max_nc = max_nc
        eval_ncs = hp.get('eval_ncs', [min_nc, max_nc // 2 if max_nc > 2 else max_nc, max_nc])
        eval_ncs = sorted(set(nc for nc in eval_ncs if min_nc <= nc <= max_nc))
        def _get_stage(_step):
            return max_nc, eval_ncs

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Logging
    log_dir  = os.path.join(log_base, name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_file = open(os.path.join(log_dir, 'train.log'), 'a', buffering=1)
    status_file = sys.stderr

    def _log(msg: str):
        print(msg, flush=True)
        print(msg, file=log_file, flush=True)

    # Model
    hp_model = dict(V=V, d=d, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
                    chunk_attn=chunk_attn)
    model = build_model2(hp_model, device)
    _log(f'Model: {model.count_params():,} params  device={device_str}  chunk_attn={chunk_attn}')

    if pretrained:
        ckpt = torch.load(pretrained, map_location=device)
        sd = ckpt.get('model', ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        _log(f'Loaded pretrained: {pretrained}  missing={len(missing)}  unexpected={len(unexpected)}')

    # SRS schedule (depth-2, single span)
    schedule = [(0, global_max_nc)]

    x0_fraction = hp.get('x0_fraction', 0.0)

    # Pre-build pos/mask for all nc values across all curriculum stages
    nc_cache: dict[int, tuple] = {}
    nc_cache_x0: dict[int, tuple] = {}   # same but warmup_train_range forced to (0, 0)
    for nc in range(min_nc, global_max_nc + 1):
        pos     = chunk_positions_iq_global_rw_full(nc, chunk_len, slot_len, warmup_len)
        mask_np = chunk_mask_fb(pos)
        mask_t  = torch.tensor(mask_np, dtype=torch.float32, device=device)
        nc_cache[nc] = (pos, mask_np, mask_t)
        if x0_fraction > 0:
            pos_x0 = copy.deepcopy(pos)
            for rb in pos_x0['rec_blocks']:
                if 'warmup_train_range' in rb:
                    rb['warmup_train_range'] = (0, 0)
            nc_cache_x0[nc] = (pos_x0, mask_np, mask_t)

    curriculum_str = (f'curriculum={[(e, n) for e, n, _ in nc_curriculum]}'
                      if nc_curriculum else f'max_nc={global_max_nc}')
    _log(f'\n[train] min_nc={min_nc} {curriculum_str} chunk_len={chunk_len} '
         f'slot={slot_len} slot_count={slot_count} wl={warmup_len} depth={n_layers} '
         f'B={B}  steps={n_steps}  x0_fraction={x0_fraction}')
    _log(f'  LR: cosine {lr_max:.1e} -> {lr_min:.1e}  warmup={warmup_steps}')
    for nc in range(min_nc, global_max_nc + 1):
        pos, _, _ = nc_cache[nc]
        _log(f'  nc={nc}  src={nc*chunk_len}B  L={pos["L"]}  '
             f'max_out={pos["rec_blocks"][0]["max_out_len"]}')

    # Optimizer + LR
    opt    = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=wd)
    lr_fn  = make_lr_fn(lr_max, lr_min, warmup_steps, n_steps)

    # Validation sequences (longest nc for general test)
    val_src_len = max_nc * chunk_len
    val_seqs    = make_test_sequences(val_src_len)
    if val_n_seqs is not None:
        val_seqs = dict(list(val_seqs.items())[:val_n_seqs])

    best_val = -1.0
    t_start  = time.time()

    pbar = tqdm(range(1, n_steps + 1), desc='train', dynamic_ncols=True, file=status_file)
    for step in pbar:
        model.train()
        opt.zero_grad()

        lr = lr_fn(step)
        for pg in opt.param_groups:
            pg['lr'] = lr

        # Sample nc for this batch (respects curriculum stage)
        stage_max_nc, _ = _get_stage(step)
        nc = int(rng.integers(min_nc, stage_max_nc + 1))

        # x0 oversample: force warmup offset X=0 with probability x0_fraction
        if x0_fraction > 0 and rng.random() < x0_fraction:
            pos, _, mask_t = nc_cache_x0[nc]
        else:
            pos, _, mask_t = nc_cache[nc]

        tok_np, ce_mask_np = _chunk_make_batch_fb(
            rng, B, nc, chunk_len, slot_len, slot_count, schedule, pos)
        tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)

        logits = model(tok_t, mask_t)

        # Loss: warmup NTP + masked output CE
        nlls = []
        ce_mask_t = (torch.tensor(ce_mask_np, device=device, dtype=torch.float32)
                     if ce_mask_np is not None else None)
        for rb in pos['rec_blocks']:
            if not rb['is_clean']:
                continue
            wl = pos['warmup_len']
            if wl > 0:
                # NTP on warmup: logits[w0-1:w1-1] predict tokens[w0:w1]
                # warmup sees only SLOT (blocked from src) — predicts from memory
                lp_w   = F.log_softmax(logits[:, rb['w0']-1:rb['w1']-1], dim=-1)
                tgt_w  = tok_t[:, rb['w0']:rb['w1']]
                nlls.append(_positional_ls_nll(lp_w, tgt_w, ls_max).mean())
            # Output: masked CE (variable length for out_to_end blocks)
            lp      = F.log_softmax(logits[:, rb['c0']-1:rb['c1']-1], dim=-1)
            tgt     = tok_t[:, rb['c0']:rb['c1']]
            nll_per = _positional_ls_nll(lp, tgt, ls_max)
            if ce_mask_t is not None and rb.get('out_to_end'):
                nlls.append((nll_per * ce_mask_t).sum() / ce_mask_t.sum().clamp(min=1))
            else:
                nlls.append(nll_per.mean())
        loss = torch.stack(nlls).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_f = float(loss.detach())
        pbar.set_postfix(loss=f'{loss_f:.3f}', lr=f'{lr:.1e}', nc=nc, refresh=False)
        if step % log_every == 0:
            print(f'{{"step":{step},"loss":{round(loss_f,5)},"lr":{round(lr,7)},"nc":{nc}}}',
                  file=log_file, flush=True)
            print(str(pbar), file=status_file, flush=True)

        if step % eval_every == 0 or step == n_steps:
            model.eval()
            elapsed = time.time() - t_start
            h, m = divmod(int(elapsed), 3600); m, s = divmod(m, 60)
            _log(f'\n--- step={step}/{n_steps}  loss={loss_f:.4f}  {h:02d}:{m:02d}:{s:02d} ---')

            eval_log_f = open(os.path.join(log_dir, f'eval_{step}.log'), 'w', buffering=1)
            def _elog(msg): print(msg, file=eval_log_f)

            _, stage_eval_ncs = _get_stage(step)
            _log(f'  [eval] stage eval_ncs={stage_eval_ncs}')
            all_means = []
            for enc_nc in stage_eval_ncs:
                src_len  = enc_nc * chunk_len
                epos, emask_np, _ = nc_cache[enc_nc]
                emask = emask_np
                rw_rb = epos['rec_blocks'][0]
                eval_offsets = rw_rb.get('warmup_valid_offsets', [0])
                eval_seqs = make_test_sequences(src_len)
                if val_n_seqs is not None:
                    eval_seqs = dict(list(eval_seqs.items())[:val_n_seqs])

                win_means = []
                for X in eval_offsets:
                    actual_out_len = src_len - (X + warmup_len)
                    if actual_out_len <= 0:
                        continue
                    win_tag = f'nc{enc_nc}_src{src_len}B_x{X}_out{actual_out_len}B'
                    valid_mask = np.zeros(rw_rb['max_out_len'], dtype=bool)
                    valid_mask[:actual_out_len] = True
                    seq_results = []
                    for sname, seq in eval_seqs.items():
                        chunks_arr = np.array(
                            [seq[k*chunk_len:(k+1)*chunk_len] for k in range(enc_nc)], np.int64)
                        r = ar_decode_chunk_fb_kv(model, chunks_arr, slot_len, slot_count,
                                                  schedule, emask, epos, device,
                                                  warmup_offset=X, valid_mask=valid_mask)
                        seq_results.append(r['match_pct'])
                        ref_b = np.array(seq[X:X+warmup_len+actual_out_len], dtype=np.int64)
                        gen_b = np.concatenate([ref_b[:warmup_len],
                                                np.array(r['decoded_bytes'][:actual_out_len],
                                                         dtype=np.int64)])
                        _log(f'  val/{win_tag}/{sname:<15} BPB={r["bpb"]:.3f}  match={r["match_pct"]:.1f}%')
                        _elog(f'\n{win_tag}/{sname}  BPB={r["bpb"]:.3f}  match={r["match_pct"]:.1f}%')
                        _elog(_fmt_cmp(ref_b, gen_b, warmup_len))
                    win_mean = sum(seq_results) / len(seq_results)
                    win_means.append(win_mean)
                    _log(f'  val/{win_tag}/MEAN               match={win_mean:.1f}%')

                nc_mean = sum(win_means) / len(win_means) if win_means else 0.0
                all_means.append(nc_mean)
                _log(f'  val/nc{enc_nc}_src{src_len}B/MEAN      match={nc_mean:.1f}%')

            eval_log_f.close()
            vmean = sum(all_means) / len(all_means) if all_means else 0.0
            _log(f'  val/MEAN (all nc)  match={vmean:.1f}%')

            if vmean > best_val:
                best_val = vmean
                ckpt_path = os.path.join(ckpt_dir, 'best.pt')
                torch.save({'model': model.state_dict(), 'hp': hp, 'step': step}, ckpt_path)
                _log(f'  [new best] step={step} val_mean={vmean:.1f}% -> {ckpt_path}')

            end_path = os.path.join(ckpt_dir, 'end.pt')
            torch.save({'model': model.state_dict(), 'hp': hp, 'step': step}, end_path)
            _log(f'  [ckpt] {end_path}')
            model.train()

    log_file.close()
    _log(f'\nDone. {time.strftime("%H:%M:%S", time.gmtime(time.time() - t_start))}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--pretrained', default=None)
    parser.add_argument('--log-base', default='logs')
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location('cfg', args.config)
    cfg  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    train_full(cfg.hp, log_base=args.log_base, device_str=args.device,
               pretrained=args.pretrained)


if __name__ == '__main__':
    main()
