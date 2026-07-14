"""
kvmem/optim.py — GrokAdamW optimizer (PyTorch).

SNR-Gated AdamW (arXiv:2605.01172).
Gates parameter updates by signal-to-noise ratio of the gradient:
  gate_k = 1{ m_k² > s_k / (B-1) }
where s_k is an EMA of squared gradient deviations (g - m_prev)².

Only updates parameters where gradient signal exceeds noise.
Accelerates grokking / generalization vs standard AdamW.

One extra EMA state per parameter (s), otherwise identical to AdamW.

Usage:
    from kvmem.optim import GrokAdamW
    opt = GrokAdamW(model.parameters(), lr=3e-4, weight_decay=0.01,
                    rho=0.9, batch_size=32)
"""

import math
import torch
from torch.optim import Optimizer


class GrokAdamW(Optimizer):
    """
    SNR-Gated AdamW.

    Args:
        params:      model parameters
        lr:          learning rate
        betas:       (b1, b2) Adam momentum coefficients
        rho:         EMA decay for gradient deviation state s
        eps:         numerical stability
        weight_decay: AdamW weight decay
        batch_size:  training batch size (sets SNR threshold = B-1)
    """

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), rho=0.9,
                 eps=1e-8, weight_decay=0.01, batch_size=32):
        defaults = dict(lr=lr, betas=betas, rho=rho, eps=eps,
                        weight_decay=weight_decay, batch_size=batch_size)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr   = group['lr']
            b1, b2 = group['betas']
            rho  = group['rho']
            eps  = group['eps']
            wd   = group['weight_decay']
            B    = group['batch_size']
            thresh = max(float(B - 1), 1.0)

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m']    = torch.zeros_like(p)
                    state['v']    = torch.zeros_like(p)
                    state['s']    = torch.zeros_like(p)   # squared deviation EMA

                state['step'] += 1
                t = state['step']
                m, v, s = state['m'], state['v'], state['s']

                m_prev = m.clone()

                # Standard Adam moments
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)

                # Squared deviation EMA: variance of gradient around m
                dev = g - m_prev
                s.mul_(rho).addcmul_(dev, dev, value=1 - rho)

                # Bias correction
                bc1 = 1.0 - b1 ** t
                bc2 = 1.0 - b2 ** t
                bcs = 1.0 - rho ** t
                mh = m / bc1
                vh = v / bc2
                sh = s / bcs

                # SNR gate: update only high-SNR parameters
                gate = (mh.pow(2) > sh / thresh).float()

                # AdamW update with SNR gate on momentum term
                denom = vh.sqrt().add_(eps)
                p.addcdiv_(gate * mh, denom, value=-lr)
                p.add_(p, alpha=-lr * wd)   # weight decay

        return loss
