"""
train_chunk_standalone.py — Self-contained SRS chunk memorization trainer.

Drop this single file on any machine with PyTorch and run:
    python train_chunk_standalone.py --config q1 --device cuda

No other files from kvmem needed. Surah Al-Fatihah is inlined as test set.

# ====================================================================
# Tanzil Quran Text (Uthmani, Version 1.1)
# Copyright (C) 2007-2026 Tanzil Project
# License: Creative Commons Attribution 3.0
#
# This copy of the Quran text is carefully produced, highly
# verified and continuously monitored by a group of specialists
# at Tanzil Project.
#
# TERMS OF USE:
# - Permission is granted to copy and distribute verbatim copies
#   of this text, but CHANGING IT IS NOT ALLOWED.
# - This Quran text can be used in any website or application,
#   provided that its source (Tanzil Project) is clearly indicated,
#   and a link is made to tanzil.net to enable users to keep
#   track of changes.
# - This copyright notice shall be included in all verbatim copies
#   of the text, and shall be reproduced appropriately in all files
#   derived from or containing substantial portion of this text.
# Please check updates at: http://tanzil.net/updates/
# ====================================================================
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Surah Al-Fatihah (inlined, never used for training — test set only)
# ---------------------------------------------------------------------------
_SURAH_ALFATIHAH: bytes = (
    b'\xd8\xa8\xd9\x90\xd8\xb3\xd9\x92\xd9\x85\xd9\x90 \xd9\xb1\xd9\x84\xd9\x84\xd9\x91\xd9\x8e\xd9\x87\xd9\x90 '
    b'\xd9\xb1\xd9\x84\xd8\xb1\xd9\x91\xd9\x8e\xd8\xad\xd9\x92\xd9\x85\xd9\x8e\xd9\x80\xd9\xb0\xd9\x86\xd9\x90 '
    b'\xd9\xb1\xd9\x84\xd8\xb1\xd9\x91\xd9\x8e\xd8\xad\xd9\x90\xd9\x8a\xd9\x85\xd9\x90\n'
    b'\xd9\xb1\xd9\x84\xd9\x92\xd8\xad\xd9\x8e\xd9\x85\xd9\x92\xd8\xaf\xd9\x8f \xd9\x84\xd9\x90\xd9\x84\xd9\x91\xd9\x8e\xd9\x87\xd9\x90 '
    b'\xd8\xb1\xd9\x8e\xd8\xa8\xd9\x91\xd9\x90 \xd9\xb1\xd9\x84\xd9\x92\xd8\xb9\xd9\x8e\xd9\x80\xd9\xb0\xd9\x84\xd9\x8e\xd9\x85\xd9\x90\xd9\x8a\xd9\x86\xd9\x8e\n'
    b'\xd9\xb1\xd9\x84\xd8\xb1\xd9\x91\xd9\x8e\xd8\xad\xd9\x92\xd9\x85\xd9\x8e\xd9\x80\xd9\xb0\xd9\x86\xd9\x90 '
    b'\xd9\xb1\xd9\x84\xd8\xb1\xd9\x91\xd9\x8e\xd8\xad\xd9\x90\xd9\x8a\xd9\x85\xd9\x90\n'
    b'\xd9\x85\xd9\x8e\xd9\x80\xd9\xb0\xd9\x84\xd9\x90\xd9\x83\xd9\x90 \xd9\x8a\xd9\x8e\xd9\x88\xd9\x92\xd9\x85\xd9\x90 '
    b'\xd9\xb1\xd9\x84\xd8\xaf\xd9\x91\xd9\x90\xd9\x8a\xd9\x86\xd9\x90\n'
    b'\xd8\xa5\xd9\x90\xd9\x8a\xd9\x91\xd9\x8e\xd8\xa7\xd9\x83\xd9\x8e \xd9\x86\xd9\x8e\xd8\xb9\xd9\x92\xd8\xa8\xd9\x8f\xd8\xaf\xd9\x8f '
    b'\xd9\x88\xd9\x8e\xd8\xa5\xd9\x90\xd9\x8a\xd9\x91\xd9\x8e\xd8\xa7\xd9\x83\xd9\x8e \xd9\x86\xd9\x8e\xd8\xb3\xd9\x92\xd8\xaa\xd9\x8e\xd8\xb9\xd9\x90\xd9\x8a\xd9\x86\xd9\x8f\n'
    b'\xd9\xb1\xd9\x87\xd9\x92\xd8\xaf\xd9\x90\xd9\x86\xd9\x8e\xd8\xa7 \xd9\xb1\xd9\x84\xd8\xb5\xd9\x91\xd9\x90\xd8\xb1\xd9\x8e\xd9\xb0\xd8\xb7\xd9\x8e '
    b'\xd9\xb1\xd9\x84\xd9\x92\xd9\x85\xd9\x8f\xd8\xb3\xd9\x92\xd8\xaa\xd9\x8e\xd9\x82\xd9\x90\xd9\x8a\xd9\x85\xd9\x8e\n'
    b'\xd8\xb5\xd9\x90\xd8\xb1\xd9\x8e\xd9\xb0\xd8\xb7\xd9\x8e \xd9\xb1\xd9\x84\xd9\x91\xd9\x8e\xd8\xb0\xd9\x90\xd9\x8a\xd9\x86\xd9\x8e '
    b'\xd8\xa3\xd9\x8e\xd9\x86\xd9\x92\xd8\xb9\xd9\x8e\xd9\x85\xd9\x92\xd8\xaa\xd9\x8e \xd8\xb9\xd9\x8e\xd9\x84\xd9\x8e\xd9\x8a\xd9\x92\xd9\x87\xd9\x90\xd9\x85\xd9\x92 '
    b'\xd8\xba\xd9\x8e\xd9\x8a\xd9\x92\xd8\xb1\xd9\x90 \xd9\xb1\xd9\x84\xd9\x92\xd9\x85\xd9\x8e\xd8\xba\xd9\x92\xd8\xb6\xd9\x8f\xd9\x88\xd8\xa8\xd9\x90 '
    b'\xd8\xb9\xd9\x8e\xd9\x84\xd9\x8e\xd9\x8a\xd9\x92\xd9\x87\xd9\x90\xd9\x85\xd9\x92 '
    b'\xd9\x88\xd9\x8e\xd9\x84\xd9\x8e\xd8\xa7 \xd9\xb1\xd9\x84\xd8\xb6\xd9\x91\xd9\x8e\xd8\xa7\xd9\x93\xd9\x84\xd9\x91\xd9\x90\xd9\x8a\xd9\x86\xd9\x8e'
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HMN_SLOT_0     = 258
HMN_VOCAB_SIZE = 268
DATA_LO        = 0x20

# ---------------------------------------------------------------------------
# RoPE / YaRN
# ---------------------------------------------------------------------------

def rope_freqs(d_head: int, base: float = 10000.0, device=None) -> torch.Tensor:
    i = torch.arange(0, d_head, 2, dtype=torch.float32, device=device)
    return 1.0 / (base ** (i / d_head))


def yarn_freqs(d_head: int, L_train: int, L_max: int, base: float = 10000.0,
               beta_fast: int = 32, beta_slow: int = 1, device=None) -> torch.Tensor:
    s    = L_max / L_train
    i    = torch.arange(0, d_head, 2, dtype=torch.float32, device=device)
    inv_f = 1.0 / (base ** (i / d_head))
    wl   = 2 * math.pi / inv_f
    lo, hi = 2 * math.pi * beta_slow, 2 * math.pi * beta_fast
    ramp = torch.clamp((wl - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (1 - ramp) * inv_f + ramp * (inv_f / s)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor, offset: int = 0) -> torch.Tensor:
    L, dh  = x.shape[-2], x.shape[-1]
    pos    = torch.arange(offset, offset + L, dtype=torch.float32, device=x.device)
    angles = pos[:, None] * freqs[None, :]
    cos_a, sin_a = angles.cos(), angles.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos_a - x2 * sin_a,
                        x1 * sin_a + x2 * cos_a], dim=-1).reshape(x.shape)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MHAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, rope: bool = False,
                 freqs: Optional[torch.Tensor] = None, null_kv: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d // n_heads
        self.rope    = rope
        self.null_kv = null_kv
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d, bias=False)
        if freqs is not None:
            self.register_buffer('freqs', freqs)
        else:
            self.freqs = None

    def forward(self, x, mask, past_kv=None, return_kv=False, offset=0):
        batched = x.dim() == 3
        if not batched:
            x = x.unsqueeze(0)
        B, L, d = x.shape
        H, dh = self.n_heads, self.d_head
        Q = self.W_Q(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)
        K = self.W_K(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)
        V = self.W_V(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)
        if self.rope and self.freqs is not None:
            Q = apply_rope(Q, self.freqs, offset=offset)
            K = apply_rope(K, self.freqs, offset=offset)
        K_cur, V_cur = K, V
        if past_kv is not None:
            K_past, V_past = past_kv
            K = torch.cat([K_past, K], dim=2)
            V = torch.cat([V_past, V], dim=2)
        if self.null_kv:
            null = torch.zeros(B, H, 1, dh, device=K.device, dtype=K.dtype)
            K    = torch.cat([K, null], dim=2)
            V    = torch.cat([V, null], dim=2)
            mask = F.pad(mask, (0, 1), value=0.0)
        out = F.scaled_dot_product_attention(Q, K, V,
                                             attn_mask=mask.unsqueeze(0).unsqueeze(0))
        out = out.permute(0, 2, 1, 3).reshape(B, L, d)
        out = self.W_O(out)
        if not batched:
            out = out.squeeze(0)
        if return_kv:
            return out, (K_cur, V_cur)
        return out


class FFN(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        h = self.W1(x)
        h = 0.5 * h * (1.0 + torch.tanh(0.7978845608028654 * (h + 0.044715 * h ** 3)))
        return self.W2(h)


class TransformerBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, d_ff: int,
                 rope: bool = False, freqs: Optional[torch.Tensor] = None,
                 null_kv: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = MHAttention(d, n_heads, rope=rope, freqs=freqs, null_kv=null_kv)
        self.norm2 = nn.LayerNorm(d)
        self.ffn   = FFN(d, d_ff)

    def forward(self, x, mask, past_kv=None, return_kv=False, offset=0):
        h = self.norm1(x)
        if return_kv:
            h, kv = self.attn(h, mask, past_kv=past_kv, return_kv=True, offset=offset)
        else:
            h = self.attn(h, mask, past_kv=past_kv, offset=offset)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        if return_kv:
            return x, kv
        return x


class KVMemModel(nn.Module):
    def __init__(self, V: int, d: int, n_layers: int, n_heads: int, d_ff: int,
                 rope: bool = False, yarn: bool = False,
                 L_train: int = 512, L_max: int = 4096,
                 null_kv: bool = False, V_out: int = 256):
        super().__init__()
        self.embed   = nn.Embedding(V, d)
        self.norm_out = nn.LayerNorm(d)
        self.W_out   = nn.Linear(d, V_out, bias=False)
        d_head = d // n_heads
        if yarn and rope:
            freqs = yarn_freqs(d_head, L_train, L_max)
        elif rope:
            freqs = rope_freqs(d_head)
        else:
            freqs = None
        self.blocks = nn.ModuleList([
            TransformerBlock(d, n_heads, d_ff, rope=rope, freqs=freqs, null_kv=null_kv)
            for _ in range(n_layers)
        ])

    def _embed(self, tokens):
        return self.embed(tokens)

    def forward(self, tokens, mask, past_kv=None, return_kv=False, offset=0):
        batched = tokens.dim() == 2
        if not batched:
            tokens = tokens.unsqueeze(0)
        x      = self._embed(tokens)
        kv_out = []
        L_past = past_kv[0][0].shape[2] if past_kv is not None else 0
        _off   = offset if offset else L_past
        for i, block in enumerate(self.blocks):
            pkv = past_kv[i] if past_kv is not None else None
            if return_kv:
                x, kv_i = block(x, mask, past_kv=pkv, return_kv=True, offset=_off)
                kv_out.append(kv_i)
            else:
                x = block(x, mask, past_kv=pkv, offset=_off)
        logits = self.W_out(self.norm_out(x))
        if not batched:
            logits = logits.squeeze(0)
        if return_kv:
            return logits, kv_out
        return logits


def build_model(hp: dict, device=None) -> KVMemModel:
    model = KVMemModel(
        V=hp.get('V', HMN_VOCAB_SIZE),
        d=hp['d'], n_layers=hp['n_layers'], n_heads=hp['n_heads'], d_ff=hp['d_ff'],
        rope=hp.get('rope', True), yarn=hp.get('yarn', True),
        L_train=hp.get('L_train', 512), L_max=hp.get('L_max', 8192),
        null_kv=hp.get('null_kv', True), V_out=256,
    )
    if device is not None:
        model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def make_test_sequences(seg_len: int) -> dict:
    V = 256 - DATA_LO
    seqs = {}
    seqs['up_counter']   = [DATA_LO + (i % V) for i in range(seg_len)]
    seqs['down_counter'] = [DATA_LO + (V - 1 - i % V) for i in range(seg_len)]
    base_odd = 1 if V % 2 == 0 else 0
    seqs['odd']          = [DATA_LO + (base_odd + 2*i) % V for i in range(seg_len)]
    seqs['even']         = [DATA_LO + (2*i) % V for i in range(seg_len)]
    seqs['linear']       = [DATA_LO + (4*i) % V for i in range(seg_len)]
    period = max(4, min(seg_len // 2, V // 4))
    step   = V // period
    seqs['sawtooth']     = [DATA_LO + (i % period) * step for i in range(seg_len)]
    half = seg_len // 2
    first_half  = [DATA_LO + (2*i) % V for i in range(half)]
    second_half = list(reversed(first_half))
    extra = [DATA_LO + (2*half) % V] if seg_len % 2 == 1 else []
    seqs['palindrome']   = first_half + extra + second_half
    geo = [DATA_LO]
    for _ in range(seg_len - 1):
        nxt = int(geo[-1] * 1.1)
        geo.append(DATA_LO if nxt > 255 else nxt)
    seqs['geometric'] = geo
    return seqs


def cer(pred: list, ref: list) -> float:
    m, n = len(ref), len(pred)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j-1] if ref[i-1] == pred[j-1] \
                    else 1 + min(prev[j-1], prev[j], dp[j-1])
    return dp[n] / max(m, 1)


def _positional_ls_nll(lp: torch.Tensor, tgt: torch.Tensor, ls_max: float) -> torch.Tensor:
    out_len = lp.shape[1]
    nll_hard = -lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
    if ls_max <= 0.0:
        return nll_hard
    eps = torch.linspace(0.0, ls_max, out_len, device=lp.device)
    nll_soft = -lp.mean(dim=-1)
    return (1.0 - eps) * nll_hard + eps * nll_soft


# ---------------------------------------------------------------------------
# SRS schedule
# ---------------------------------------------------------------------------

def srs_schedule(n_chunks: int) -> list:
    half = n_chunks // 2
    schedule = []
    for i in range(half):              schedule.append((i, i + 1))
    for i in range(0, half, 2):        schedule.append((i, i + 2))
    for i in range(half, n_chunks):    schedule.append((i, i + 1))
    for i in range(half, n_chunks, 2): schedule.append((i, i + 2))
    schedule.append((0, n_chunks))
    return schedule


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def chunk_positions(n_chunks: int, chunk_len: int, slot_len: int,
                    schedule: list, ir_turns: int = 2, warmup_len: int = 0) -> dict:
    enc_block_len = chunk_len + slot_len
    enc_blocks = []
    for k in range(n_chunks):
        s0  = k * enc_block_len
        s1  = s0 + chunk_len
        sl0 = s1
        sl1 = sl0 + slot_len
        enc_blocks.append(dict(s0=s0, s1=s1, sl0=sl0, sl1=sl1))
    enc_end = n_chunks * enc_block_len
    rec_blocks = []
    offset = enc_end
    for span in schedule:
        span_start, span_end = span
        span_len = (span_end - span_start) * chunk_len
        out_len  = span_len - warmup_len
        for turn in range(ir_turns):
            sl0 = offset; sl1 = sl0 + slot_len
            w0  = sl1;    w1  = w0  + warmup_len
            c0  = w1;     c1  = c0  + out_len
            rec_blocks.append(dict(
                sl0=sl0, sl1=sl1, w0=w0, w1=w1, c0=c0, c1=c1,
                span=span, span_len=span_len, out_len=out_len,
                turn=turn, is_clean=(turn == ir_turns - 1),
            ))
            offset = c1
    return dict(enc_blocks=enc_blocks, rec_blocks=rec_blocks,
                enc_end=enc_end, warmup_len=warmup_len, L=offset)


# ---------------------------------------------------------------------------
# Attention mask (strictly causal — source-first layout [chunk][SLOT])
# ---------------------------------------------------------------------------

def chunk_mask(pos: dict) -> np.ndarray:
    L = pos['L']
    r = np.arange(L); c = np.arange(L)
    causal  = c[None, :] <= r[:, None]
    blocked = np.zeros((L, L), dtype=bool)
    enc_blocks = pos['enc_blocks']
    rec_blocks = pos['rec_blocks']

    is_any_chunk = np.zeros(L, dtype=bool)
    for b in enc_blocks:
        is_any_chunk |= (c >= b['s0']) & (c < b['s1'])

    for k, b in enumerate(enc_blocks):
        sl_row = (r >= b['sl0']) & (r < b['sl1'])
        for j, bj in enumerate(enc_blocks):
            if j == k: continue
            blocked |= sl_row[:, None] & ((c >= bj['s0']) & (c < bj['s1']))[None, :]

    for rb in rec_blocks:
        sl_row = (r >= rb['sl0']) & (r < rb['sl1'])
        blocked |= sl_row[:, None] & is_any_chunk[None, :]

    for rb in rec_blocks:
        own_sl  = (c >= rb['sl0']) & (c < rb['sl1'])
        own_wm  = (c >= rb['w0'])  & (c < rb['w1'])
        own_out = (c >= rb['c0'])  & (c < rb['c1'])
        if rb['w0'] < rb['w1']:
            wm_row = (r >= rb['w0']) & (r < rb['w1'])
            blocked |= wm_row[:, None] & ~(own_sl | own_wm)[None, :]
        out_row = (r >= rb['c0']) & (r < rb['c1'])
        blocked |= out_row[:, None] & ~(own_sl | own_wm | own_out)[None, :]

    visible = causal & ~blocked
    return np.where(visible, 0.0, -1e9).astype(np.float32)


# ---------------------------------------------------------------------------
# Slot helpers
# ---------------------------------------------------------------------------

def _slot_ids(slot_len: int, slot_count: int = 2) -> list:
    return [HMN_SLOT_0 + (i % slot_count) for i in range(slot_len)]


# ---------------------------------------------------------------------------
# Batch builder (training — synthetic random bytes only)
# ---------------------------------------------------------------------------

def _chunk_make_batch(rng, B: int, n_chunks: int, chunk_len: int,
                      slot_len: int, slot_count: int, schedule: list,
                      ir_turns: int, noise_p: float, pos: dict) -> np.ndarray:
    sids = np.array(_slot_ids(slot_len, slot_count), dtype=np.int64)
    L    = pos['L']
    tok  = np.zeros((B, L), dtype=np.int64)
    wl   = pos.get('warmup_len', 0)
    segs = rng.integers(0, 256, size=(B, n_chunks, chunk_len), dtype=np.int64)
    for k, b in enumerate(pos['enc_blocks']):
        tok[:, b['s0']:b['s1']]   = segs[:, k, :]
        tok[:, b['sl0']:b['sl1']] = sids
    for rb in pos['rec_blocks']:
        tok[:, rb['sl0']:rb['sl1']] = sids
        span_s, span_e = rb['span']
        gt = np.concatenate([segs[:, i, :] for i in range(span_s, span_e)], axis=1)
        if wl > 0:
            tok[:, rb['w0']:rb['w1']] = gt[:, :wl]
        gt_out = gt[:, wl:]
        if rb['is_clean']:
            tok[:, rb['c0']:rb['c1']] = gt_out
        else:
            noisy = gt_out.copy()
            nm = rng.random((B, rb['out_len'])) < noise_p
            nv = rng.integers(0, 256, size=(B, rb['out_len']), dtype=np.int64)
            noisy[nm] = nv[nm]
            tok[:, rb['c0']:rb['c1']] = noisy
    return tok


# ---------------------------------------------------------------------------
# KV-cache helper
# ---------------------------------------------------------------------------

def _cat_kv(kv_a: list, kv_b: list) -> list:
    return [(torch.cat([ka, kb], dim=2), torch.cat([va, vb], dim=2))
            for (ka, va), (kb, vb) in zip(kv_a, kv_b)]


# ---------------------------------------------------------------------------
# KV-cached AR decode  (~8× faster than naive full-recompute)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ar_decode_chunk_kv(model, chunks_arr, slot_len: int, slot_count: int,
                       schedule: list, mask_np: np.ndarray, pos: dict,
                       device, valid_mask: Optional[np.ndarray] = None) -> dict:
    """
    Greedy AR decode with incremental KV caching.
    Mask is strictly causal (source-first layout), so KV caching is exact.
    """
    if isinstance(chunks_arr, np.ndarray) and chunks_arr.ndim == 2:
        chunks_list = [chunks_arr[k] for k in range(chunks_arr.shape[0])]
    else:
        chunks_list = list(chunks_arr)

    n_chunks  = len(chunks_list)
    wl        = pos.get('warmup_len', 0)
    sids      = np.array(_slot_ids(slot_len, slot_count), dtype=np.int64)
    L         = pos['L']
    full_mask = torch.tensor(mask_np, dtype=torch.float32, device=device)

    tok = np.zeros(L, dtype=np.int64)
    for k, b in enumerate(pos['enc_blocks']):
        tok[b['s0']:b['s1']]   = chunks_list[k]
        tok[b['sl0']:b['sl1']] = sids
    for rb in pos['rec_blocks']:
        tok[rb['sl0']:rb['sl1']] = sids
        if wl > 0:
            span_s, span_e = rb['span']
            gt_span = np.concatenate(chunks_list[span_s:span_e])
            tok[rb['w0']:rb['w1']] = gt_span[:wl].astype(np.int64)

    enc_end = pos['enc_end']
    enc_t   = torch.tensor(tok[:enc_end], dtype=torch.long, device=device)
    _, kv_cache = model(enc_t, full_mask[:enc_end, :enc_end], return_kv=True)
    L_cached = enc_end

    all_gen = {}
    for rb in pos['rec_blocks']:
        seg_start = rb['sl0']; seg_end = rb['c0']; seg_len = seg_end - seg_start
        seg_t     = torch.tensor(tok[seg_start:seg_end], dtype=torch.long, device=device)
        seg_mask  = full_mask[seg_start:seg_end, :L_cached + seg_len]
        seg_logits, seg_kv = model(seg_t, seg_mask, past_kv=kv_cache,
                                   return_kv=True, offset=L_cached)
        kv_cache  = _cat_kv(kv_cache, seg_kv)
        L_cached += seg_len

        gen = [int(seg_logits[-1].argmax())]
        tok[rb['c0']] = gen[0]

        for j in range(1, rb['out_len']):
            prev_pos  = rb['c0'] + j - 1
            prev_t    = torch.tensor([gen[j-1]], dtype=torch.long, device=device)
            prev_mask = full_mask[prev_pos:prev_pos+1, :L_cached + 1]
            prev_logits, prev_kv = model(prev_t, prev_mask, past_kv=kv_cache,
                                         return_kv=True, offset=L_cached)
            kv_cache  = _cat_kv(kv_cache, prev_kv)
            L_cached += 1
            next_tok  = int(prev_logits[-1].argmax())
            gen.append(next_tok)
            tok[rb['c0'] + j] = next_tok

        all_gen[(rb['span'], rb['turn'])] = gen

    full_rb = None
    for rb in reversed(pos['rec_blocks']):
        if rb['is_clean'] and rb['span'] == (0, n_chunks):
            full_rb = rb; break
    assert full_rb is not None

    gen         = np.array(all_gen[(full_rb['span'], full_rb['turn'])], dtype=np.int64)
    target_full = np.concatenate(chunks_list)
    target      = target_full[wl:]
    out_len     = full_rb['out_len']

    if valid_mask is not None:
        vm       = valid_mask.flatten()[wl:wl + out_len]
        gen_v    = gen[:len(vm)][vm]
        target_v = target[:len(vm)][vm]
    else:
        gen_v = gen; target_v = target[:out_len]

    match_pct = 100.0 * float(np.sum(gen_v == target_v)) / max(len(target_v), 1)

    tok_tf = torch.tensor(tok, dtype=torch.long, device=device)
    for rb2 in pos['rec_blocks']:
        if rb2['is_clean']:
            span_s, span_e = rb2['span']
            gt = np.concatenate(chunks_list[span_s:span_e])
            if wl > 0:
                tok_tf[rb2['w0']:rb2['w1']] = torch.tensor(gt[:wl].astype(np.int64), dtype=torch.long, device=device)
            tok_tf[rb2['c0']:rb2['c1']] = torch.tensor(gt[wl:].astype(np.int64), dtype=torch.long, device=device)

    logits_tf  = model(tok_tf, full_mask)
    tgt_tensor = torch.tensor(target[:out_len], dtype=torch.long, device=device)
    lp_full    = F.log_softmax(logits_tf[full_rb['c0']-1:full_rb['c1']-1], dim=-1)
    nll_vals   = -lp_full.gather(1, tgt_tensor.unsqueeze(1)).squeeze(1).cpu().numpy()
    if valid_mask is not None:
        vm_flat  = valid_mask.flatten()[wl:wl + out_len]
        nll_vals = nll_vals[vm_flat[:len(nll_vals)]]
    nll = float(nll_vals.mean())
    bpb = nll / math.log(2)
    return dict(bpb=bpb, nll=nll, match_pct=match_pct,
                decoded_bytes=gen.tolist(), n_valid=len(target_v))


# ---------------------------------------------------------------------------
# Test-set loader (inlined surah or file path)
# ---------------------------------------------------------------------------

def load_chunks_padded_bytes(raw: bytes, n_chunks: int,
                              chunk_len: int) -> tuple:
    lines   = [l for l in raw.split(b'\n') if l]
    n_lines = len(lines)
    base    = n_lines // n_chunks
    extra   = n_lines % n_chunks
    groups  = []
    start   = 0
    for gi in range(n_chunks):
        count = base + (1 if gi < extra else 0)
        groups.append(b''.join(lines[start:start + count]))
        start += count
    chunks     = np.zeros((n_chunks, chunk_len), dtype=np.int64)
    valid_mask = np.zeros((n_chunks, chunk_len), dtype=bool)
    for k, g in enumerate(groups):
        if g:
            b      = np.frombuffer(g[:chunk_len], dtype=np.uint8).astype(np.int64)
            n_real = min(len(b), chunk_len)
            chunks[k, :n_real]     = b[:n_real]
            valid_mask[k, :n_real] = True
    return chunks, valid_mask


def load_chunks_padded(path: Optional[str], n_chunks: int,
                       chunk_len: int) -> tuple:
    raw = open(path, 'rb').read() if path else _SURAH_ALFATIHAH
    return load_chunks_padded_bytes(raw, n_chunks, chunk_len)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_chunk(hp: dict, log_base: str = 'logs', device_str: str = 'cpu'):
    device = torch.device(device_str)
    rng    = np.random.default_rng(hp.get('seed', 42))
    torch.manual_seed(hp.get('seed', 42))

    name     = hp.get('name', 'hmn_chunk')
    log_dir  = os.path.join(log_base, name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    log_file   = open(os.path.join(log_dir, 'train.log'),   'a', buffering=1)
    jsonl_file = open(os.path.join(log_dir, 'train.jsonl'), 'a', buffering=1)

    def _log(msg): print(msg); print(msg, file=log_file)
    def _jlog(d):  jsonl_file.write(json.dumps(d) + '\n')

    hp_model = dict(V=hp.get('V', HMN_VOCAB_SIZE),
                    d=hp['d'], n_layers=hp['n_layers'],
                    n_heads=hp['n_heads'], d_ff=hp['d_ff'],
                    rope=hp.get('rope', True), yarn=hp.get('yarn', True),
                    null_kv=hp.get('null_kv', True), compile=hp.get('compile', False))
    model    = build_model(hp_model, device)
    n_params = sum(p.numel() for p in model.parameters())
    _log(f'Model: {n_params:,} params  device={device}')

    if hp.get('_pretrained_ckpt'):
        ckpt = torch.load(hp['_pretrained_ckpt'], map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        _log(f'Loaded pretrained: {hp["_pretrained_ckpt"]}')

    lr_max       = hp.get('lr_max', 3e-4)
    wd           = hp.get('wd', 0.001)
    warmup_steps = hp.get('warmup_steps', 500)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=wd)

    eval_file  = hp.get('eval_file', None)   # None → use inlined surah
    eval_every = hp.get('eval_every', 5000)
    log_every  = hp.get('log_every', 500)
    curriculum = hp.get('curriculum', [])
    assert curriculum

    global_step = 0
    t_start     = time.time()

    for stage_i, stage in enumerate(curriculum):
        n_chunks   = stage['n_chunks']
        chunk_len  = stage['chunk_len']
        slot_len   = hp.get('slot_len', 2)
        slot_count = hp.get('slot_count', 2)
        ir_turns   = hp.get('ir_turns', 2)
        warmup_len = hp.get('warmup_len', 0)
        noise_p    = hp.get('noise_p', 0.5)
        B          = stage.get('B', 8)
        n_steps    = stage.get('n_steps', 50000)
        use_srs    = stage.get('use_srs', True)
        ls_max     = hp.get('ls_max', 0.0)

        schedule = srs_schedule(n_chunks) if use_srs else [(0, n_chunks)]
        pos      = chunk_positions(n_chunks, chunk_len, slot_len, schedule, ir_turns, warmup_len)
        mask_np  = chunk_mask(pos)
        mask_t   = torch.tensor(mask_np, dtype=torch.float32, device=device)

        _log(f'\n[stage {stage_i}] n_chunks={n_chunks} chunk_len={chunk_len} '
             f'slot={slot_len} wl={warmup_len} ir={ir_turns} srs={use_srs} '
             f'L={pos["L"]}  B={B}  steps={n_steps}')

        def _lr(s): return lr_max * s / max(warmup_steps, 1) if s <= warmup_steps else lr_max

        test_chunks = test_vm = None
        try:
            test_chunks, test_vm = load_chunks_padded(eval_file, n_chunks, chunk_len)
        except Exception as e:
            _log(f'  [test eval disabled: {e}]')

        val_seg_len = n_chunks * chunk_len
        val_seqs    = make_test_sequences(val_seg_len)

        pbar = tqdm(range(1, n_steps + 1), desc=f'stage{stage_i}', dynamic_ncols=True)
        for local_step in pbar:
            global_step += 1
            lr = _lr(local_step)
            for pg in opt.param_groups: pg['lr'] = lr

            model.train(); opt.zero_grad()
            tok_np = _chunk_make_batch(rng, B, n_chunks, chunk_len, slot_len,
                                       slot_count, schedule, ir_turns, noise_p, pos)
            tok_t  = torch.tensor(tok_np, device=device, dtype=torch.long)
            logits = model(tok_t, mask_t)

            nlls = []
            for rb in pos['rec_blocks']:
                if not rb['is_clean']: continue
                lp  = F.log_softmax(logits[:, rb['c0']-1:rb['c1']-1], dim=-1)
                tgt = tok_t[:, rb['c0']:rb['c1']]
                nlls.append(_positional_ls_nll(lp, tgt, ls_max).mean())
            loss = torch.stack(nlls).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_f = float(loss.detach())
            pbar.set_postfix(loss=f'{loss_f:.3f}', lr=f'{lr:.1e}', refresh=False)
            if local_step % log_every == 0:
                _jlog(dict(step=global_step, loss=round(loss_f, 5), lr=lr))

            if local_step % eval_every == 0 or local_step == n_steps:
                model.eval()
                elapsed = time.time() - t_start
                h, m = divmod(int(elapsed), 3600); m, s = divmod(m, 60)
                _log(f'\n--- stage={stage_i} step={local_step}/{n_steps}'
                     f'  g={global_step}  loss={loss_f:.4f}  {h:02d}:{m:02d}:{s:02d} ---')

                val_results = []
                for sname, seq in val_seqs.items():
                    chunks_arr = np.array(
                        [seq[k*chunk_len:(k+1)*chunk_len] for k in range(n_chunks)], np.int64)
                    r = ar_decode_chunk_kv(model, chunks_arr, slot_len, slot_count,
                                           schedule, mask_np, pos, device)
                    val_results.append(r['match_pct'])
                    _log(f'  val/{sname:<15} BPB={r["bpb"]:.3f}  match={r["match_pct"]:.1f}%')
                val_mean = sum(val_results) / len(val_results)
                _log(f'  val/MEAN               match={val_mean:.1f}%')

                if test_chunks is not None:
                    r = ar_decode_chunk_kv(model, test_chunks, slot_len, slot_count,
                                           schedule, mask_np, pos, device, valid_mask=test_vm)
                    _log(f'  test/surah             BPB={r["bpb"]:.3f}'
                         f'  match={r["match_pct"]:.1f}%  valid_bytes={r["n_valid"]}')
                    _jlog(dict(step=global_step, eval=True,
                               val_mean=round(val_mean, 1),
                               test_bpb=round(r['bpb'], 3),
                               test_match=round(r['match_pct'], 1)))

        ckpt_path = os.path.join(ckpt_dir, f'stage{stage_i}_end.pt')
        torch.save(dict(model=model.state_dict(), hp=hp, hp_model=hp_model,
                        stage=stage_i, step=global_step), ckpt_path)
        _log(f'  [ckpt] {ckpt_path}')

    elapsed = time.time() - t_start
    h, m = divmod(int(elapsed), 3600); m, s = divmod(m, 60)
    _log(f'\nDone. {h:02d}:{m:02d}:{s:02d}')
    log_file.close(); jsonl_file.close()


# ---------------------------------------------------------------------------
# Built-in configs  (use --config q1 / q2 / q3 / baseline)
# ---------------------------------------------------------------------------

_CONFIGS = {
    'baseline': dict(
        d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
        lr_max=3e-4, wd=0.001, warmup_steps=500,
        eval_every=5000, log_every=500,
        rope=True, yarn=True, null_kv=True, compile=False,
        name='hmn_chunk_fine', seed=42,
        slot_len=1, slot_count=2, warmup_len=0, ir_turns=2, noise_p=0.5,
        curriculum=[
            dict(n_chunks=8, chunk_len=16, use_srs=False, B=8, n_steps=50000),
            dict(n_chunks=8, chunk_len=16, use_srs=True,  B=8, n_steps=80000),
            dict(n_chunks=8, chunk_len=32, use_srs=True,  B=8, n_steps=80000),
        ],
    ),
    'q1': dict(
        d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
        lr_max=3e-4, wd=0.001, warmup_steps=500,
        eval_every=5000, log_every=500,
        rope=True, yarn=True, null_kv=True, compile=False,
        name='hmn_chunk_fine_wm', seed=42,
        slot_len=1, slot_count=2, warmup_len=8, ir_turns=2, noise_p=0.5,
        curriculum=[
            dict(n_chunks=8, chunk_len=16, use_srs=False, B=8, n_steps=50000),
            dict(n_chunks=8, chunk_len=16, use_srs=True,  B=8, n_steps=80000),
            dict(n_chunks=8, chunk_len=32, use_srs=True,  B=8, n_steps=80000),
        ],
    ),
    'q2': dict(
        d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
        lr_max=3e-4, wd=0.001, warmup_steps=500,
        eval_every=5000, log_every=500,
        rope=True, yarn=True, null_kv=True, compile=False,
        name='hmn_chunk_fine_wm_nosrs0', seed=42,
        slot_len=1, slot_count=2, warmup_len=8, ir_turns=2, noise_p=0.5,
        curriculum=[
            dict(n_chunks=8, chunk_len=16, use_srs=True, B=8, n_steps=130000),
            dict(n_chunks=8, chunk_len=32, use_srs=True, B=8, n_steps=80000),
        ],
    ),
    'q3': dict(
        d=64, n_layers=4, n_heads=4, d_ff=256, V=268,
        lr_max=3e-4, wd=0.001, warmup_steps=500,
        eval_every=5000, log_every=500,
        rope=True, yarn=True, null_kv=True, compile=False,
        name='hmn_chunk_fine_wm_randinit', seed=42,
        slot_len=1, slot_count=2, warmup_len=8, ir_turns=2, noise_p=0.5,
        curriculum=[
            dict(n_chunks=8, chunk_len=16, use_srs=True, B=8, n_steps=130000),
            dict(n_chunks=8, chunk_len=32, use_srs=True, B=8, n_steps=80000),
        ],
    ),
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='SRS chunk memorization — standalone cloud-GPU script')
    p.add_argument('--config',     default='q1',
                   help='Built-in: baseline/q1/q2/q3, or path to .py config file')
    p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--pretrained', default=None,
                   help='Path to pretrained checkpoint (required for q1/q2; omit for q3)')
    p.add_argument('--log-dir',    default='logs')
    args = p.parse_args()

    if args.config in _CONFIGS:
        hp = dict(_CONFIGS[args.config])
    elif os.path.exists(args.config):
        spec   = importlib.util.spec_from_file_location('_cfg', args.config)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        hp = dict(module.hp)
    else:
        raise ValueError(f'Unknown config: {args.config!r}  (choices: {list(_CONFIGS)})')

    if args.pretrained:
        hp['_pretrained_ckpt'] = args.pretrained

    print(f'Config: {args.config}  device: {args.device}  name: {hp["name"]}')
    train_chunk(hp, log_base=args.log_dir, device_str=args.device)


if __name__ == '__main__':
    main()
