"""
DenseNet-style depth-wise cross-layer SLOT-KV concatenation.

Constraint (from design discussion, see docs/SRS_RECIPE.md § direction 6): only SLOT
token positions accumulate KV across layers. Every other position (source bytes,
warmup, output, tags) behaves like a completely regular transformer — single-layer
attention, no cross-layer memory. This keeps the extra cost bounded to slot_len
extra keys per layer (not the whole sequence) and avoids changing the semantics of
anything except the SLOT bottleneck itself, which is the thing under study.

Mechanism: layer i computes its own K,V for the full sequence as normal (RoPE
applied as usual). After each layer, the SLOT-position columns of that layer's own
K,V are extracted and appended to a running history. Layer i+1's attention sees the
normal current-layer K,V for every column, PLUS extra key/value columns holding
layers 1..i's SLOT-position K,V (concatenated, not summed or pooled) — literally the
same `torch.cat` mechanic already used for `past_kv` inference caching elsewhere in
this codebase, just accumulating across depth instead of across time.

Extra columns inherit the same per-row visibility as the "live" slot column they
correspond to (attending to an earlier-layer representation of a position you're
already causally allowed to see is not a new leak — it's a subset of information
already available).

By the final layer, SLOT columns are backed by up to (n_layers-1) prior layers'
worth of extra K,V: for d=64, n_heads=4 (d_head=16), slot_len=8, n_layers=4, the
last layer attends to up to 4x8=32 slot keys against d_head=16 — saturating the
full per-head rank ceiling instead of being capped below it by slot_len alone.

Reused unchanged from kvmem.model: rope_freqs/yarn_freqs/apply_rope, RMSNorm/LayerNorm
choice, FFN. Only attention/block/model forward differ (need to plumb extra_kv/
extra_mask and extract+accumulate SLOT K,V after every layer).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from kvmem.model import rope_freqs, yarn_freqs, apply_rope, RMSNorm, FFN, _make_norm


class DenseSlotKVAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, rope: bool = False,
                 freqs: torch.Tensor | None = None, null_kv: bool = False):
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                extra_kv: tuple | None = None, extra_mask: torch.Tensor | None = None):
        """
        x: (B, L, d). mask: (L, L) additive bias, 0.0=attend/-1e9=blocked.
        extra_kv: optional (K_extra, V_extra), each (B, H, n_extra, dh) — accumulated
                  SLOT KV from prior layers.
        extra_mask: optional (L, n_extra) additive bias for the extra columns.
        Returns (out, (K_own, V_own)) — K_own/V_own are THIS layer's own full-sequence
        K,V (before any extra/null concatenation), for the caller to slice SLOT
        columns from and add to the running history.
        """
        B, L, d = x.shape
        H, dh = self.n_heads, self.d_head
        Q = self.W_Q(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)
        K = self.W_K(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)
        V = self.W_V(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)
        if self.rope and self.freqs is not None:
            Q = apply_rope(Q, self.freqs, offset=0)
            K = apply_rope(K, self.freqs, offset=0)
        K_own, V_own = K, V

        K_full, V_full = K, V
        mask_full = mask
        if extra_kv is not None:
            K_extra, V_extra = extra_kv
            K_full = torch.cat([K_full, K_extra], dim=2)
            V_full = torch.cat([V_full, V_extra], dim=2)
            mask_full = torch.cat([mask_full, extra_mask], dim=1)
        if self.null_kv:
            null = torch.zeros(B, H, 1, dh, device=K.device, dtype=K.dtype)
            K_full = torch.cat([K_full, null], dim=2)
            V_full = torch.cat([V_full, null], dim=2)
            mask_full = F.pad(mask_full, (0, 1), value=0.0)

        out = F.scaled_dot_product_attention(Q, K_full, V_full,
                                             attn_mask=mask_full.unsqueeze(0).unsqueeze(0))
        out = out.permute(0, 2, 1, 3).reshape(B, L, d)
        out = self.W_O(out)
        return out, (K_own, V_own)


class DenseSlotKVBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, d_ff: int, rope: bool = False,
                 freqs: torch.Tensor | None = None, null_kv: bool = False,
                 gated_ffn: bool = False, rmsnorm: bool = False):
        super().__init__()
        self.norm1 = _make_norm(d, rmsnorm)
        self.attn  = DenseSlotKVAttention(d, n_heads, rope=rope, freqs=freqs, null_kv=null_kv)
        self.norm2 = _make_norm(d, rmsnorm)
        self.ffn   = FFN(d, d_ff, gated=gated_ffn)

    def forward(self, x, mask, extra_kv=None, extra_mask=None):
        attn_out, kv_own = self.attn(self.norm1(x), mask, extra_kv=extra_kv, extra_mask=extra_mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, kv_own


class DenseSlotKVModel(nn.Module):
    """Same input/output conventions as kvmem.model.KVMemModel (V input vocab incl.
    tags, V_out=256 data bytes only), but with depth-wise growing SLOT KV.

    forward() takes an additional `slot_positions`: 1D LongTensor of sequence
    indices that are SLOT tokens (encoding + recall SLOT/SLOT_A/SLOT_B ranges,
    tag-inclusive is irrelevant here — pass content-region slot indices).
    """
    def __init__(self, V: int, d: int, n_layers: int, n_heads: int, d_ff: int,
                 rope: bool = True, yarn: bool = True, L_train: int = 512,
                 L_max: int = 4096, null_kv: bool = True, gated_ffn: bool = False,
                 rmsnorm: bool = False, V_out: int = 256):
        super().__init__()
        n_special = V - 256
        self.data_embed    = nn.Embedding(256, d)
        self.special_embed = nn.Embedding(n_special, d)
        self.n_special     = n_special
        self.norm_out      = _make_norm(d, rmsnorm)
        self.W_out         = nn.Linear(d, V_out, bias=False)
        self.V_out         = V_out
        self.n_layers      = n_layers

        freqs = None
        if rope:
            d_head = d // n_heads
            freqs = (yarn_freqs(d_head, L_train=L_train, L_max=L_max) if yarn
                     else rope_freqs(d_head))

        self.blocks = nn.ModuleList([
            DenseSlotKVBlock(d, n_heads, d_ff, rope=rope, freqs=freqs, null_kv=null_kv,
                             gated_ffn=gated_ffn, rmsnorm=rmsnorm)
            for _ in range(n_layers)
        ])
        self._init_weights()

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        is_sp = tokens >= 256
        data_ids    = tokens.clamp(0, 255)
        special_ids = (tokens - 256).clamp(0, self.n_special - 1)
        d_emb = self.data_embed(data_ids)
        s_emb = self.special_embed(special_ids)
        mask  = is_sp.unsqueeze(-1).to(d_emb.dtype)
        return s_emb * mask + d_emb * (1.0 - mask)

    def _init_weights(self):
        nn.init.normal_(self.data_embed.weight, std=0.02)
        nn.init.normal_(self.special_embed.weight, std=0.05)
        nn.init.normal_(self.W_out.weight, std=0.02)
        for name, p in self.named_parameters():
            if 'embed' in name or 'W_out' in name:
                continue
            if p.dim() == 2:
                nn.init.normal_(p, std=math.sqrt(2.0 / p.shape[-1]))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor,
                slot_positions: torch.Tensor) -> torch.Tensor:
        """tokens: (B,L) or (L,). mask: (L,L). slot_positions: 1D LongTensor."""
        batched = tokens.dim() == 2
        if not batched:
            tokens = tokens.unsqueeze(0)
        x = self._embed(tokens)

        live_mask_cols = mask.index_select(1, slot_positions)  # (L, n_slot)
        history_K: list[torch.Tensor] = []
        history_V: list[torch.Tensor] = []
        extra_kv = None
        extra_mask = None

        for i, block in enumerate(self.blocks):
            x, (K_own, V_own) = block(x, mask, extra_kv=extra_kv, extra_mask=extra_mask)
            if i < self.n_layers - 1:
                K_slot = K_own.index_select(2, slot_positions)  # (B,H,n_slot,dh)
                V_slot = V_own.index_select(2, slot_positions)
                history_K.append(K_slot)
                history_V.append(V_slot)
                extra_kv   = (torch.cat(history_K, dim=2), torch.cat(history_V, dim=2))
                extra_mask = live_mask_cols.repeat(1, i + 1)  # (L, n_slot*(i+1))

        h_out  = self.norm_out(x)
        logits = self.W_out(h_out)
        if not batched:
            logits = logits.squeeze(0)
        return logits


def build_densekv_model(hp: dict, device=None) -> DenseSlotKVModel:
    model = DenseSlotKVModel(
        V=hp['V'], d=hp['d'], n_layers=hp['n_layers'], n_heads=hp['n_heads'],
        d_ff=hp['d_ff'], rope=hp.get('rope', True), yarn=hp.get('yarn', True),
        null_kv=hp.get('null_kv', True), gated_ffn=hp.get('gated_ffn', False),
        rmsnorm=hp.get('rmsnorm', False), V_out=256,
    )
    if device is not None:
        model = model.to(device)
    return model
