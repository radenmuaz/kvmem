"""
kvmem/probe_signal_propagation.py — asks two separate questions about whether
`hmn_locate_nope_curriculum_dense.py`'s architecture/task combination (single_attn,
d=64, n_layers=8, n_heads=4, rope=False, null_kv=True, rmsnorm=True; locate-and-
continue task, data_kind='random') is fundamentally limited, or just slow to
converge:

1. `--mode signal` (default): per-layer activation norm, per-layer attention
   entropy, and per-layer gradient norm, compared between a fresh random-init
   model and a checkpoint (default: the current run's stage0_best.pt), plus a
   short (300-step) CPU training run from random init tracking grad norm
   evolution. Checks for vanishing gradients into early layers, exploding
   activations, and whether attention actually sharpens (becomes less uniform)
   with training rather than staying permanently diffuse.

   IMPORTANT when picking a checkpoint/trajectory pair: use a trajectory shape
   the checkpoint was ACTUALLY trained on. The first run of this diagnostic
   evaluated stage0_best.pt (trained only on chunk_len=8) against a chunk_len=16
   batch it had never seen, making the "pretrained" model look worse than
   random init (loss 17.6 vs 5.5) — a distribution-mismatch bug in the harness,
   not a real finding. Fixed by defaulting `--dsl` to one of stage0's own
   trajectories.

   Also fixes a real edge-case bug from that same first run: attention-entropy
   normalization divided by log(n_valid), and rows with exactly one attendable
   position (n_valid=1, common for early causal rows) have log(1)=0 — any
   epsilon clamp on that denominator blows up even a near-zero numerator into
   the thousands (the "attn_entropy=1753215" seen in the first run). Fixed by
   excluding n_valid<2 rows from the normalized average instead of clamping.

2. `--mode ambiguity`: is the training DATA itself (data_kind='random', uniform
   i.i.d. bytes) a limiting factor — i.e., for a given (chunk_len, warmup_len),
   what fraction of random chunks have the TRUE warmup excerpt's exact byte
   string recur elsewhere in the same chunk (making "locate this excerpt"
   genuinely ambiguous from content alone, unsolvable in principle for that
   sample)? Measured empirically, cross-checked against the birthday-bound
   approximation (n_windows*(n_windows-1)/2 / 256^warmup_len) — the two agree
   once accounting for what each is measuring: the birthday bound gives P(ANY
   duplicate exists among all window pairs in the chunk), which is ~3% at
   chunk_len=64/warmup_len=2, while the empirical rate asks the narrower "does
   the SPECIFIC true excerpt collide" question (~0.1%) — consistent, since
   only 2 of ~63 windows are the colliding pair even when one exists.

   Finding: genuine ambiguity is under 0.1% across every (chunk_len,
   warmup_len) combination this project's locate configs actually use, even
   at the shortest warmup_len=2 tested. The dataset is NOT a meaningful
   limiting factor — if anything, uniform random (max-entropy, incompressible)
   data is close to the best-case distribution for this task, since structured/
   repetitive data would introduce MORE genuine duplicate substrings, not
   fewer. The qualitative near-garbage output seen at warmup_len=2 (see
   CLAUDE.md's positional-shortcut/qualitative-decode entries) is better
   explained as a learning-difficulty issue (the exact-match attention
   algorithm is harder to learn from 2 bytes of signal) than a data ceiling.

Usage (CPU only — does not need or touch a GPU/MPS training run):
    python3 -m kvmem.probe_signal_propagation --mode signal \
        --ckpt logs/hmn_locate_nope_curriculum_dense/checkpoints/stage0_best.pt \
        --dsl 'E(8) Q(0,1,2,2) B8'
    python3 -m kvmem.probe_signal_propagation --mode ambiguity
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from kvmem.hmn import (build_model, chunk_positions_traj, chunk_mask_fb_traj,
                       make_batch_tagged, parse_traj_dsl)

STATE_LEN, STATE_VOCAB_SIZE = 8, 2


def _build_batch(dsl: str, B: int, device: torch.device, seed: int = 0):
    ops, n_refine, repeat_batch, chunk_len, warmup_len = parse_traj_dsl(dsl)
    if warmup_len is None:
        warmup_len = 8
    built = chunk_positions_traj(chunk_len, STATE_LEN, warmup_len, ops,
                                  n_refine=n_refine, state_vocab_size=STATE_VOCAB_SIZE)
    pos_content, pos_mask, tags, L = built['pos_content'], built['pos_mask'], built['tags'], built['L']
    mask_np = chunk_mask_fb_traj(pos_mask, hops=-1)
    mask_t = torch.tensor(mask_np, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    n_chunks = sum(1 for op, _ in ops if op == 'E')
    tok_np = make_batch_tagged(rng, B, n_chunks, chunk_len, STATE_LEN, STATE_VOCAB_SIZE,
                                pos_content, tags)
    tok_t = torch.tensor(tok_np, device=device, dtype=torch.long)
    return tok_t, mask_t, pos_content, n_chunks, chunk_len, tags


def _loss_for(model, tok_t, mask_t, pos_content):
    logits = model(tok_t, mask_t)
    nlls = []
    for rb in pos_content['rec_blocks']:
        if not rb['is_clean']:
            continue
        lp = F.log_softmax(logits[:, rb['c0'] - 1:rb['c1'] - 1], dim=-1)
        tgt = tok_t[:, rb['c0']:rb['c1']]
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        nlls.append(nll.mean())
    return torch.stack(nlls).mean()


@torch.no_grad()
def _manual_attn(attn_module, x, mask):
    """Re-derives MHAttention.forward's math WITHOUT SDPA, purely to expose the
    softmax attention-weight tensor for entropy analysis (matches the
    null_kv=True, no-rope, no-qk_norm, no-cache path exactly)."""
    B, L, d = x.shape
    H, dh = attn_module.n_heads, attn_module.d_head
    Q = attn_module.W_Q(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)
    K = attn_module.W_K(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)
    V = attn_module.W_V(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)
    if attn_module.null_kv:
        null = torch.zeros(B, H, 1, dh)
        K = torch.cat([K, null], dim=2)
        mask = F.pad(mask, (0, 1), value=0.0)
    scale = 1.0 / math.sqrt(dh)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale + mask.unsqueeze(0).unsqueeze(0)
    return torch.softmax(scores, dim=-1)


def _analyze(model, tok_t, mask_t, tag):
    model.eval()
    x = model._embed(tok_t)
    print(f'\n=== {tag} ===')
    print(f'{"layer":>6} {"act_norm":>10} {"attn_entropy":>13} {"n_valid_avg":>12}')
    for i, block in enumerate(model.blocks):
        h = block.norm(x)
        probs = _manual_attn(block.attn, h, mask_t)
        n_valid = (mask_t == 0.0).float().sum(-1)  # (L,) valid cols per query row
        ent = -(probs * probs.clamp_min(1e-12).log()).sum(-1)  # (B,H,L)
        row_mean_ent = ent.mean(dim=(0, 1))  # (L,)
        keep = n_valid >= 2  # exclude n_valid<2 rows: log(1)=0 denominator, see module docstring
        ent_norm = row_mean_ent[keep] / n_valid[keep].log()
        act_norm = x.norm(dim=-1).mean().item()
        print(f'{i:>6} {act_norm:>10.3f} {ent_norm.mean().item():>13.3f} {n_valid.mean().item():>12.1f}')
        x = block(x, mask_t)


def _grad_norms(model, tok_t, mask_t, pos_content, tag):
    model.train()
    model.zero_grad()
    loss = _loss_for(model, tok_t, mask_t, pos_content)
    loss.backward()
    print(f'\n=== {tag}: per-layer grad norm (loss={loss.item():.4f}) ===')
    for i, block in enumerate(model.blocks):
        g = sum(p.grad.norm().item() ** 2 for p in block.parameters() if p.grad is not None)
        print(f'  layer {i}: grad_norm={math.sqrt(g):.5f}')


def run_signal(ckpt_path: str, dsl: str, device_str: str, train_steps: int):
    device = torch.device(device_str)
    hp_model = dict(V=274, d=64, n_layers=8, n_heads=4, d_ff=0,
                    block_type='single_attn', rope=False, yarn=False,
                    null_kv=True, rmsnorm=True)
    torch.manual_seed(0)

    tok_t, mask_t, pos_content, n_chunks, chunk_len, tags = _build_batch(dsl, B=8, device=device)
    print(f'dsl={dsl!r}  batch shape: {tuple(tok_t.shape)}  L={pos_content["L"]}')

    model_rand = build_model(hp_model, device)
    _analyze(model_rand, tok_t, mask_t, 'RANDOM INIT: activation/attention')
    _grad_norms(model_rand, tok_t, mask_t, pos_content, 'RANDOM INIT')

    model_pre = build_model(hp_model, device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model_pre.load_state_dict(ckpt['model'])
    _analyze(model_pre, tok_t, mask_t, f'PRETRAINED ({ckpt_path}): activation/attention')
    _grad_norms(model_pre, tok_t, mask_t, pos_content, f'PRETRAINED ({ckpt_path})')

    print(f'\n=== SHORT TRAINING RUN ({train_steps} steps, fresh random init, same trajectory) ===')
    model_train = build_model(hp_model, device)
    opt = torch.optim.AdamW(model_train.parameters(), lr=1e-4, weight_decay=1e-5)
    rng = np.random.default_rng(1)
    log_every = max(train_steps // 10, 1)
    for step in range(1, train_steps + 1):
        tok_np = make_batch_tagged(rng, 8, n_chunks, chunk_len, STATE_LEN, STATE_VOCAB_SIZE,
                                    pos_content, tags)
        tok_t_s = torch.tensor(tok_np, device=device, dtype=torch.long)
        opt.zero_grad()
        loss = _loss_for(model_train, tok_t_s, mask_t, pos_content)
        loss.backward()
        if step % log_every == 0 or step == 1:
            gns = []
            for block in model_train.blocks:
                g = sum(p.grad.norm().item() ** 2 for p in block.parameters() if p.grad is not None)
                gns.append(math.sqrt(g))
            print(f'  step {step:4d}  loss={loss.item():.4f}  '
                  f'per-layer grad_norm=[{",".join(f"{g:.3f}" for g in gns)}]')
        torch.nn.utils.clip_grad_norm_(model_train.parameters(), 1.0)
        opt.step()


def run_ambiguity(n_trials: int):
    rng = np.random.default_rng(0)

    def ambiguity_rate(chunk_len, warmup_len, n_trials):
        hits = 0
        for _ in range(n_trials):
            chunk = rng.integers(0, 256, size=chunk_len, dtype=np.int64)
            max_start = chunk_len - warmup_len
            true_start = rng.integers(0, max_start + 1)
            window = chunk[true_start:true_start + warmup_len].tobytes()
            dup = any(chunk[s:s + warmup_len].tobytes() == window
                     for s in range(max_start + 1) if s != true_start)
            hits += dup
        return hits / n_trials

    print(f'{"chunk_len":>10} {"warmup_len":>11} {"genuine-ambiguity rate":>24}')
    for chunk_len in [8, 16, 32, 64]:
        for warmup_len in [2, 3, 4, 6, 8, 12, 16, 24]:
            if warmup_len > chunk_len - 4:
                continue
            rate = ambiguity_rate(chunk_len, warmup_len, n_trials)
            print(f'{chunk_len:>10} {warmup_len:>11} {rate * 100:>23.2f}%')

    n_windows = 64 - 2 + 1
    pairs = n_windows * (n_windows - 1) / 2
    approx = pairs / (256 ** 2)
    print(f'\nbirthday-bound cross-check (chunk_len=64, warmup_len=2, P(ANY duplicate '
          f'pair exists among all {n_windows} windows)): {approx * 100:.2f}%  '
          f'(vs. the ~0.1% "does the TRUE excerpt specifically collide" rate above — '
          f'consistent: only 2 of {n_windows} windows are the colliding pair even when one exists)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['signal', 'ambiguity'], default='signal')
    p.add_argument('--ckpt', default='logs/hmn_locate_nope_curriculum_dense/checkpoints/stage0_best.pt')
    p.add_argument('--dsl', default='E(8) Q(0,1,2,2) B8')
    p.add_argument('--device', default='cpu')
    p.add_argument('--train-steps', type=int, default=300)
    p.add_argument('--n-trials', type=int, default=5000)
    args = p.parse_args()

    if args.mode == 'signal':
        run_signal(args.ckpt, args.dsl, args.device, args.train_steps)
    else:
        run_ambiguity(args.n_trials)
