"""
kvmem/model2.py — Clean transformer for full-continuation memorization.

Architecture choices (hardcoded, not ablation flags):
  - RMSNorm everywhere (no LayerNorm)
  - No bias in any Linear layer
  - GELU FFN (no SwiGLU — keeps param count predictable)
  - RoPE + YaRN positional encoding
  - null_kv learnable abstain token
  - n_heads=2 (dh=d//2) — proven optimal for slot retrieval at d=64

Identical forward API to model.py (tokens, mask, past_kv, return_kv, offset)
so ar_decode_chunk_fb_kv and other eval utilities work unchanged.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ZerO Initialization  (arXiv:2110.12661)
# ---------------------------------------------------------------------------

def _hadamard(p: int) -> torch.Tensor:
    """p×p Walsh-Hadamard matrix (unnormalized), p must be a power of 2."""
    H = torch.tensor([[1.0]])
    while H.shape[0] < p:
        H = torch.cat([torch.cat([H,  H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H


def zero_init(m: int, n: int) -> torch.Tensor:
    """ZerO init for a weight matrix of shape (m=out_features, n=in_features).

    m <= n (square or dim-reducing): partial identity I*(m,n)
    m >  n (dim-increasing):         c * H_p[:m, :n]  where p=2^ceil(log2(m)),
                                      c = 1 / 2^(ceil(log2(m))/2)
    """
    if m <= n:
        W = torch.zeros(m, n)
        W[:m, :m] = torch.eye(m)
        return W
    clog_m = math.ceil(math.log2(m))
    p      = 2 ** clog_m
    H      = _hadamard(p) / (2 ** (clog_m / 2.0))
    return H[:m, :n]


# ---------------------------------------------------------------------------
# RoPE / YaRN
# ---------------------------------------------------------------------------

def rope_freqs(d_head: int, base: float = 10000.0, device=None) -> torch.Tensor:
    i = torch.arange(0, d_head, 2, dtype=torch.float32, device=device)
    return 1.0 / (base ** (i / d_head))


def yarn_freqs(d_head: int, L_train: int, L_max: int,
               base: float = 10000.0,
               beta_fast: int = 32, beta_slow: int = 1,
               device=None) -> torch.Tensor:
    s     = L_max / L_train
    i     = torch.arange(0, d_head, 2, dtype=torch.float32, device=device)
    inv_f = 1.0 / (base ** (i / d_head))
    wl    = 2 * math.pi / inv_f
    lo, hi = 2 * math.pi * beta_slow, 2 * math.pi * beta_fast
    ramp  = torch.clamp((wl - lo) / (hi - lo + 1e-8), 0.0, 1.0)
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
# Norms and FFN
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt() * self.weight


class FFN(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.W1(x)
        h = 0.5 * h * (1.0 + torch.tanh(0.7978845608028654 * (h + 0.044715 * h ** 3)))
        return self.W2(h)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class MHAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, freqs: torch.Tensor | None = None,
                 chunk_attn: int = 0):
        super().__init__()
        self.n_heads   = n_heads
        self.d_head    = d // n_heads
        self.chunk_attn = chunk_attn
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d, bias=False)
        if freqs is not None:
            self.register_buffer('freqs', freqs)
        else:
            self.freqs = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                past_kv: tuple | None = None,
                return_kv: bool = False,
                offset: int = 0) -> torch.Tensor | tuple:
        batched = x.dim() == 3
        if not batched:
            x = x.unsqueeze(0)
        B, L, d = x.shape
        H, dh = self.n_heads, self.d_head

        Q = self.W_Q(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)
        K = self.W_K(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)
        V = self.W_V(x).reshape(B, L, H, dh).permute(0, 2, 1, 3)

        if self.freqs is not None:
            Q = apply_rope(Q, self.freqs, offset=offset)
            K = apply_rope(K, self.freqs, offset=offset)

        K_cur, V_cur = K, V
        if past_kv is not None:
            K_past, V_past = past_kv
            K = torch.cat([K_past, K], dim=2)
            V = torch.cat([V_past, V], dim=2)

        # null_kv: fixed-zero key → Q·null_k = 0 always; learnable null value
        null_k = torch.zeros(B, H, 1, dh, device=K.device, dtype=K.dtype)
        null_v = torch.zeros(B, H, 1, dh, device=K.device, dtype=K.dtype)
        K = torch.cat([K, null_k], dim=2)
        V = torch.cat([V, null_v], dim=2)
        mask = F.pad(mask, (0, 1), value=0.0)

        chunk = self.chunk_attn
        if chunk > 0 and L > chunk:
            m = mask.unsqueeze(0).unsqueeze(0)
            parts = [F.scaled_dot_product_attention(
                         Q[:, :, i:i+chunk, :], K, V,
                         attn_mask=m[:, :, i:i+chunk, :])
                     for i in range(0, L, chunk)]
            out = torch.cat(parts, dim=2)
        else:
            out = F.scaled_dot_product_attention(Q, K, V,
                                                 attn_mask=mask.unsqueeze(0).unsqueeze(0))

        out = out.permute(0, 2, 1, 3).reshape(B, L, d)
        out = self.W_O(out)
        if not batched:
            out = out.squeeze(0)
        if return_kv:
            return out, (K_cur, V_cur)
        return out


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, d_ff: int,
                 freqs: torch.Tensor | None = None, chunk_attn: int = 0):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn  = MHAttention(d, n_heads, freqs=freqs, chunk_attn=chunk_attn)
        self.norm2 = RMSNorm(d)
        self.ffn   = FFN(d, d_ff)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                past_kv: tuple | None = None,
                return_kv: bool = False,
                offset: int = 0) -> torch.Tensor | tuple:
        attn_out = self.attn(self.norm1(x), mask,
                             past_kv=past_kv, return_kv=return_kv, offset=offset)
        if return_kv:
            attn_out, kv = attn_out
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        if return_kv:
            return x, kv
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class KVMemModel2(nn.Module):
    """
    Clean KVMem transformer: RMSNorm, no bias, RoPE+YaRN, null_kv.
    Same forward API as KVMemModel in model.py.
    """
    def __init__(self, V: int, d: int, n_layers: int, n_heads: int, d_ff: int,
                 L_train: int = 512, L_max: int = 4096,
                 chunk_attn: int = 0, V_out: int = 256,
                 use_zero_init: bool = True):
        super().__init__()
        n_special          = V - 256
        self.data_embed    = nn.Embedding(256, d)
        self.special_embed = nn.Embedding(n_special, d)
        self.n_special     = n_special
        self.norm_out      = RMSNorm(d)
        self.W_out         = nn.Linear(d, V_out, bias=False)
        self.V_out         = V_out

        d_head = d // n_heads
        freqs  = yarn_freqs(d_head, L_train=L_train, L_max=L_max)

        self.blocks = nn.ModuleList([
            TransformerBlock(d, n_heads, d_ff, freqs=freqs, chunk_attn=chunk_attn)
            for _ in range(n_layers)
        ])
        self._init_weights(n_layers, use_zero_init=use_zero_init)

    def _init_weights(self, n_layers: int, use_zero_init: bool = True):
        d = self.data_embed.embedding_dim
        if use_zero_init:
            with torch.no_grad():
                # One-hot fill: E[i, i % d] = 1, else 0 (cyclic for vocab > d)
                nn.init.zeros_(self.data_embed.weight)
                for i in range(self.data_embed.num_embeddings):
                    self.data_embed.weight[i, i % d] = 1.0
                nn.init.zeros_(self.special_embed.weight)
                for i in range(self.special_embed.num_embeddings):
                    self.special_embed.weight[i, i % d] = 1.0
            nn.init.normal_(self.W_out.weight, std=0.02)
            for block in self.blocks:
                for proj in (block.attn.W_Q, block.attn.W_K,
                             block.attn.W_V, block.attn.W_O,
                             block.ffn.W1, block.ffn.W2):
                    with torch.no_grad():
                        proj.weight.copy_(zero_init(*proj.weight.shape))
        else:
            # Flat random init: std=0.02 for all projections (no depth scaling)
            nn.init.normal_(self.data_embed.weight, std=0.02)
            nn.init.normal_(self.special_embed.weight, std=0.02)
            nn.init.normal_(self.W_out.weight, std=0.02)
            for block in self.blocks:
                for proj in (block.attn.W_Q, block.attn.W_K,
                             block.attn.W_V, block.attn.W_O,
                             block.ffn.W1, block.ffn.W2):
                    nn.init.normal_(proj.weight, std=0.02)

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        is_sp   = tokens >= 256
        d_ids   = tokens.clamp(0, 255)
        s_ids   = (tokens - 256).clamp(0, self.n_special - 1)
        d_emb   = self.data_embed(d_ids)
        s_emb   = self.special_embed(s_ids)
        mask    = is_sp.unsqueeze(-1).to(d_emb.dtype)
        return s_emb * mask + d_emb * (1.0 - mask)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor,
                past_kv: list | None = None,
                return_kv: bool = False,
                offset: int = 0,
                return_features: bool = False,
                h_inject: dict | None = None) -> torch.Tensor | tuple:
        batched = tokens.dim() == 2
        if not batched:
            tokens = tokens.unsqueeze(0)
        x      = self._embed(tokens)
        kv_out = []
        L_past = past_kv[0][0].shape[2] if past_kv is not None else 0
        _offset = offset if offset else L_past

        for i, block in enumerate(self.blocks):
            pkv = past_kv[i] if past_kv is not None else None
            result = block(x, mask, past_kv=pkv, return_kv=return_kv, offset=_offset)
            if return_kv:
                x, kv_i = result
                kv_out.append(kv_i)
            else:
                x = result

        logits = self.W_out(self.norm_out(x))
        if not batched:
            logits = logits.squeeze(0)
        if return_features:
            return logits, x
        if return_kv:
            return logits, kv_out
        return logits

    def encode_prefix(self, prefix_tokens, prefix_mask):
        return self.forward(prefix_tokens, prefix_mask, return_kv=True)

    def forward_with_prefix_kv(self, suffix_tokens, prefix_kv, suffix_mask):
        L_past = prefix_kv[0][0].shape[2]
        return self.forward(suffix_tokens, suffix_mask, past_kv=prefix_kv, offset=L_past)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model2(hp: dict, device=None) -> KVMemModel2:
    model = KVMemModel2(
        V=hp['V'], d=hp['d'], n_layers=hp['n_layers'],
        n_heads=hp['n_heads'], d_ff=hp['d_ff'],
        L_train=hp.get('L_train', 512),
        L_max=hp.get('L_max', 4096),
        chunk_attn=hp.get('chunk_attn', 0),
        V_out=256,
        use_zero_init=hp.get('use_zero_init', True),
    )
    if device is not None:
        model = model.to(device)
    return model
