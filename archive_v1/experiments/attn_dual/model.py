"""
Dual-attention block ablation: replace each block's attn+MLP sublayers with
attn+attn (no MLP anywhere in the model).

Motivation: MLP layers in a regular LM are the documented mechanism for storing
static factual/associative knowledge in WEIGHTS (Geva et al., "Transformer FFN
Layers Are Key-Value Memories"). This architecture deliberately keeps content
OUT of the weights — it lives in the runtime SLOT KV (see docs/SRS_RECIPE.md's
fast-weight framing) — so MLP's usual job may not be load-bearing here. The
useful computation this task needs (copy/route bytes from encoded SLOT
positions to output positions) is fundamentally an addressing/retrieval
problem, which is attention's job, not a per-token nonlinear transform's job.

Counter-risk: MLPs are the model's only source of per-token NONLINEAR
elementwise recoding — attention (even stacked) is structurally a weighted
AVERAGE of other positions' values, with no native way to apply e.g. "add 1 to
this byte" independent of context. The val suite's up_counter/down_counter/odd
sequences specifically exercise that kind of per-token arithmetic transform,
so this ablation is also a diagnostic for whether that capability is needed.

Reuses kvmem.model's MHAttention/RMSNorm/rope machinery unmodified via import
— only the block/model wiring is new (attn+attn instead of attn+ffn).

Note on KV caching: a dual-attention block has TWO KV pairs per layer, which
breaks the single-KV-pair-per-layer assumption baked into every existing
ar_decode_*_kv function in this project. Rather than extend that machinery,
eval here uses full-recompute AR decode (no KV cache) — same tradeoff
densenet_kv/decode.py made for its own KV-cache-incompatible architecture.
Fine at this scale (L~742, tiny model); not intended as a production path.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint as _ckpt

from kvmem.model import MHAttention, _make_norm, rope_freqs, yarn_freqs


def _attn_sublayer(attn, norm, x, mask):
    return x + attn(norm(x), mask)


class DualAttnBlock(nn.Module):
    def __init__(self, d: int, n_heads: int,
                 rope: bool = False, freqs: torch.Tensor | None = None,
                 null_kv: bool = False, qk_norm: bool = False,
                 rmsnorm: bool = False, logit_cap: float = 0.0,
                 attn_temp: bool = False):
        super().__init__()
        self.norm1 = _make_norm(d, rmsnorm)
        self.attn1 = MHAttention(d, n_heads, rope=rope, freqs=freqs,
                                 null_kv=null_kv, qk_norm=qk_norm,
                                 logit_cap=logit_cap, attn_temp=attn_temp)
        self.norm2 = _make_norm(d, rmsnorm)
        self.attn2 = MHAttention(d, n_heads, rope=rope, freqs=freqs,
                                 null_kv=null_kv, qk_norm=qk_norm,
                                 logit_cap=logit_cap, attn_temp=attn_temp)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                ckpt_attn: bool = False) -> torch.Tensor:
        # ckpt_attn: checkpoint EACH attn sublayer individually (finer-grained
        # than DualAttnModel's own per-block checkpointing option below) — lets
        # "checkpoint each self-attn" be tested as its own granularity, per the
        # question of whether finer or coarser checkpointing is faster on MPS.
        if ckpt_attn and self.training:
            x = _ckpt(_attn_sublayer, self.attn1, self.norm1, x, mask, use_reentrant=False)
            x = _ckpt(_attn_sublayer, self.attn2, self.norm2, x, mask, use_reentrant=False)
        else:
            x = _attn_sublayer(self.attn1, self.norm1, x, mask)
            x = _attn_sublayer(self.attn2, self.norm2, x, mask)
        return x


class DualAttnModel(nn.Module):
    """Mirrors kvmem.model.KVMemModel's embedding/output-head/init logic, but
    with DualAttnBlock instead of TransformerBlock. No KV-cache support
    (forward always does a full pass — no past_kv/return_kv/offset args).

    NOTE: n_layers blocks x 2 attn sublayers each = n_layers*2 total residual-
    attention operations (e.g. n_layers=4 -> 8 attn ops), NOT n_layers.
    Mathematically identical to "n_layers*2 single-attn blocks stacked" — see
    docs/SRS_RECIPE.md "pros/cons of one-attn-per-block vs dual-attn" (decided
    to keep the paired framing for attn1/attn2 mechanistic-role interpretability
    and to avoid breaking existing checkpoints' state_dict keys). If
    depth-scaled init (KVMemModel's `depth_scaled_init`, 1/sqrt(2*n_layers)
    residual scaling) is ever added here, it must use `n_layers*2` as the
    effective depth, not `n_layers` — this class has no such init option yet,
    so it's a latent trap for later, not a current bug.
    """

    def __init__(self, V: int, d: int, n_layers: int, n_heads: int,
                 rope: bool = False, yarn: bool = False,
                 L_train: int = 512, L_max: int = 4096,
                 null_kv: bool = False, V_out: int = 256,
                 rmsnorm: bool = False, grad_checkpoint: str | None = None):
        """
        grad_checkpoint: None (default, off) | 'block' (checkpoint each
        DualAttnBlock as a whole, matching KVMemModel's granularity) | 'attn'
        (checkpoint each attn1/attn2 sublayer individually — finer-grained,
        see DualAttnBlock.forward's ckpt_attn). Speed comparison between the
        two granularities (and off) is the open question this flag exists to
        let us measure empirically rather than guess.
        """
        super().__init__()
        assert grad_checkpoint in (None, 'block', 'attn')
        self.grad_checkpoint = grad_checkpoint
        n_special          = V - 256
        self.data_embed    = nn.Embedding(256, d)
        self.special_embed = nn.Embedding(n_special, d)
        self.n_special     = n_special
        self.norm_out      = _make_norm(d, rmsnorm)
        self.W_out         = nn.Linear(d, V_out, bias=False)
        self.V_out         = V_out

        freqs = None
        if rope:
            d_head = d // n_heads
            freqs  = (yarn_freqs(d_head, L_train=L_train, L_max=L_max)
                      if yarn else rope_freqs(d_head))

        self.blocks = nn.ModuleList([
            DualAttnBlock(d, n_heads, rope=rope, freqs=freqs, null_kv=null_kv,
                          rmsnorm=rmsnorm)
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

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batched = tokens.dim() == 2
        if not batched:
            tokens = tokens.unsqueeze(0)
        x = self._embed(tokens)
        for block in self.blocks:
            if self.grad_checkpoint == 'block' and self.training:
                x = _ckpt(block, x, mask, use_reentrant=False)
            elif self.grad_checkpoint == 'attn':
                x = block(x, mask, ckpt_attn=True)
            else:
                x = block(x, mask)
        h_out  = self.norm_out(x)
        logits = self.W_out(h_out)
        if not batched:
            logits = logits.squeeze(0)
        return logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_dualattn_model(hp: dict, device=None) -> DualAttnModel:
    V_in = hp.get('V', 256)
    model = DualAttnModel(V=V_in, d=hp['d'], n_layers=hp['n_layers'],
                          n_heads=hp['n_heads'], rope=hp.get('rope', True),
                          yarn=hp.get('yarn', True), null_kv=hp.get('null_kv', True),
                          V_out=hp.get('V_out', 256), rmsnorm=hp.get('rmsnorm', False),
                          grad_checkpoint=hp.get('grad_checkpoint', None))
    if device is not None:
        model = model.to(device)
    # Chunked attention (memory only, not a FLOP reduction — see kvmem/model.py's
    # own chunk_attn docstring): computes attention in row-chunks to reduce peak
    # activation memory. DualAttnModel has TWO attention sublayers per block (vs
    # one in the standard architecture), roughly doubling the attention-map memory
    # footprint at the same L — this caused an MPS OOM crash at L=1694 (nc=8) that
    # went undiagnosed for hours since chunk_attn was never wired in here. Must be
    # set on BOTH attn1 and attn2.
    chunk_attn = hp.get('chunk_attn', 0)
    if chunk_attn > 0:
        for block in model.blocks:
            block.attn1.chunk_attn = chunk_attn
            block.attn2.chunk_attn = chunk_attn
    return model
