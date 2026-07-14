"""
kvmem/seq_dsl.py — Tiny DSL for sequence format specs.

Parse a compact string describing a multi-block KV-memory sequence into
positions, mask, and batch generator — without manually specifying each
hyperparameter separately.

Grammar
-------
    spec   ::= block* recall
    block  ::= '<m:N>' '<s:N>'
             | '<m:N,active=K>' '<s:N>'   # active_slots on the memory tag
    recall ::= '<f:N>' '<c:N>'
             | '<f:N>' '<c:N,from=K>'     # recall_from on the output tag
    N, K   ::= integer

Shorthand for repeated blocks:
    Kx<m:N><s:M><f:P><c:Q>               # K identical blocks

All <m> tags must share the same N and active value.
All <s> tags must share the same N.
Whitespace between tags is ignored.

Parameters carried by the DSL
------------------------------
    slot_len     from <m:N>
    active_slots from <m:N,active=K>  (default 0 = all slots)
    seg_len      from <s:N>
    warmup_len   from <f:N>
    out_len      from <c:N>
    recall_from  from <c:N,from=K>    (default 0)
    n_blocks     = number of <m><s> pairs × repeat multiplier

Examples
--------
    >>> parse_seq("<m:8><s:16><f:4><c:8>")
    SeqSpec  <m:8><s:16>  <f:4><c:8,from=0>  active_slots=0  n_blocks=1  L=64

    >>> parse_seq("<m:8,active=2><s:16><f:4><c:8>")
    SeqSpec  <m:8,active=2><s:16>  <f:4><c:8,from=0>  active_slots=2  n_blocks=1  L=64

    >>> parse_seq("2x<m:8,active=2><s:16><f:4><c:8,from=1>")
    SeqSpec  <m:8,active=2><s:16> x2  <f:4><c:8,from=1>  active_slots=2  n_blocks=2  L=102

    >>> s = parse_seq("<m:8,active=2><s:16><f:4><c:8>")
    >>> s.info()
    SeqSpec  <m:8,active=2><s:16>  <f:4><c:8,from=0>  active_slots=2  n_blocks=1  L=64
      block 0:  <m> [0,3)  slots [3,11)  </m> [11,15)  <s> [15,18)  src [18,34)  </s> [34,38)
      recall:   <f> [38,41)  warmup [41,45)  </f> [45,49)  <c> [49,52)  output [52,60)  </c> [60,64)
      KV bytes (fp32): 2 × n_layers × active_slots × d_head × 4
               e.g. n_layers=4 d=64 n_heads=4 → 2 slots active → 512 floats = 2 KB

    >>> mask  = s.make_mask()              # uses s.active_slots
    >>> batch = s.make_batch(rng, B=16)
    >>> # Pass to trainer:
    >>> hp = s.to_hp(B=16, n_steps=80000, lr_max=3e-4)
"""

from __future__ import annotations
import re
import numpy as np

from kvmem.data import (
    multi_block_positions, make_mask_multi, make_multi_batch,
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_TAG = re.compile(
    r'<([mscfp])'                     # tag name: m,s,c,f,p
    r':(\d+)'                         # :N  (length)
    r'(?:,active=(\d+))?'             # optional ,active=K  (for <m>)
    r'(?:,from=(\d+))?'               # optional ,from=K   (for <c>)
    r'>',
)
_REPEAT = re.compile(r'^(\d+)x', re.IGNORECASE)


def parse_seq(spec: str) -> 'SeqSpec':
    """
    Parse a sequence spec string into a SeqSpec.

    See module docstring for grammar and examples.
    """
    spec = spec.strip()

    multiplier = 1
    m = _REPEAT.match(spec)
    if m:
        multiplier = int(m.group(1))
        spec       = spec[m.end():]

    tags = _TAG.findall(spec)   # (name, N, active_or_empty, from_or_empty)
    if not tags:
        raise ValueError(f'No tags found in spec: {spec!r}')

    blocks_raw: list[tuple[str, int, int | None]] = []  # (name, n, extra)
    f_len: int | None = None
    p_len: int        = 0
    c_len: int | None = None
    recall_from = 0
    active_slots_vals: list[int] = []

    for name, n_str, active_str, from_str in tags:
        n = int(n_str)
        if name == 'm':
            active = int(active_str) if active_str else 0
            active_slots_vals.append(active)
            blocks_raw.append(('m', n, active))
        elif name == 's':
            blocks_raw.append(('s', n, None))
        elif name == 'f':
            if f_len is not None:
                raise ValueError('Multiple <f> tags')
            f_len = n
        elif name == 'p':
            p_len = n   # ponder scratchpad; 0 = no ponder
        elif name == 'c':
            if c_len is not None:
                raise ValueError('Multiple <c> tags')
            c_len = n
            if from_str:
                recall_from = int(from_str)

    if f_len is None or c_len is None:
        raise ValueError(f'Spec must contain <f:N> and <c:N>: {spec!r}')

    if len(blocks_raw) % 2 != 0:
        raise ValueError(f'Block tags must be <m><s> pairs, got: {[b[:2] for b in blocks_raw]}')

    pairs = [(blocks_raw[i], blocks_raw[i+1]) for i in range(0, len(blocks_raw), 2)]
    for (a, _, _), (b, _, _) in pairs:
        if a != 'm' or b != 's':
            raise ValueError(f'Expected <m><s> pairs, got <{a}><{b}>')

    slot_lens   = [n for name, n, _ in blocks_raw if name == 'm']
    src_lens    = [n for name, n, _ in blocks_raw if name == 's']

    if len(set(slot_lens)) > 1:
        raise ValueError(f'All <m:N> must share the same N, got {slot_lens}')
    if len(set(src_lens)) > 1:
        raise ValueError(f'All <s:N> must share the same N, got {src_lens}')
    if len(set(active_slots_vals)) > 1:
        raise ValueError(f'All <m> active values must match, got {active_slots_vals}')

    n_blocks    = len(pairs) * multiplier
    slot_len    = slot_lens[0]
    seg_len     = src_lens[0]
    active_slots = active_slots_vals[0] if active_slots_vals else 0

    if n_blocks == 0:
        raise ValueError('Spec must contain at least one <m><s> block pair')
    if recall_from >= n_blocks:
        raise ValueError(f'recall_from={recall_from} out of range for n_blocks={n_blocks}')

    return SeqSpec(
        n_blocks=n_blocks, slot_len=slot_len, seg_len=seg_len,
        warmup_len=f_len, out_len=c_len, ponder_len=p_len,
        recall_from=recall_from, active_slots=active_slots,
    )


# ---------------------------------------------------------------------------
# SeqSpec
# ---------------------------------------------------------------------------

class SeqSpec:
    """
    A parsed sequence specification.

    All trainer hyperparameters derived from the DSL string — pass to
    make_mask() / make_batch() / to_hp() without repeating values.

    Attributes
    ----------
    n_blocks     : number of <m><s> ingestion blocks
    slot_len     : slot tokens per block
    seg_len      : source bytes per block
    warmup_len   : <f> anchor length
    out_len      : <c> output length
    recall_from  : which block (0-based) the recall targets
    active_slots : slots visible to <f>/<c> (0 = all)
    L            : total sequence length
    """

    def __init__(self, n_blocks: int, slot_len: int, seg_len: int,
                 warmup_len: int, out_len: int,
                 recall_from: int = 0, active_slots: int = 0,
                 ponder_len: int = 0):
        self.n_blocks     = n_blocks
        self.slot_len     = slot_len
        self.seg_len      = seg_len
        self.warmup_len   = warmup_len
        self.ponder_len   = ponder_len
        self.out_len      = out_len
        self.recall_from  = recall_from
        self.active_slots = active_slots
        self._pos         = None

    @property
    def L(self) -> int:
        return self.positions()['L']

    def positions(self) -> dict:
        """Return multi_block_positions dict (cached)."""
        if self._pos is None:
            self._pos = multi_block_positions(
                self.n_blocks, self.seg_len, self.slot_len,
                self.warmup_len, self.out_len, self.ponder_len)
        return self._pos

    def make_mask(self, active_slots: int | None = None) -> np.ndarray:
        """Build the attention mask. active_slots defaults to self.active_slots."""
        a = self.active_slots if active_slots is None else active_slots
        return make_mask_multi(
            self.n_blocks, self.seg_len, self.slot_len,
            self.warmup_len, self.out_len, a, self.ponder_len)

    def make_batch(self, rng: np.random.Generator, B: int,
                   slot_style: str = 'seq',
                   drop_close_prob: float = 0.5) -> np.ndarray:
        """
        Build one batch of shape (B, L).
        <p> region is filled with zeros (model pondering is unsupervised).
        """
        return make_multi_batch(
            rng, B, self.n_blocks, self.recall_from,
            self.seg_len, self.slot_len, slot_style,
            self.warmup_len, self.out_len, drop_close_prob, self.ponder_len)

    def to_hp(self, **overrides) -> dict:
        """
        Return an hp dict for train_role(). Pass trainer params as kwargs.

        Example:
            hp = parse_seq("2x<m:8,active=2><s:16><f:4><p:4><c:8,from=1>").to_hp(
                B=16, n_steps=80000, dataset_size=20000, name='ablate_t1')
            train_role({**DEFAULTS, **hp}, device_str='mps')
        """
        base = dict(
            seq=self.to_spec_str(),   # carry the spec string for logging
            n_blocks=self.n_blocks,
            recall_from=self.recall_from,
            seg_len=self.seg_len,
            slot_len=self.slot_len,
            active_slots=self.active_slots,
            warmup_len=self.warmup_len,
            ponder_len=self.ponder_len,
            out_len=self.out_len,
            curriculum=None,
        )
        base.update(overrides)
        return base

    def info(self) -> str:
        """Human-readable layout summary with absolute token positions."""
        pos   = self.positions()
        lines = [repr(self)]
        for i, b in enumerate(pos['blocks']):
            lines.append(
                f'  block {i}:  '
                f'<m>[{b["block_start"]},{b["sl0"]})  '
                f'slots[{b["sl0"]},{b["sl1"]})  '
                f'</m>[{b["sl1"]},{b["mc1"]})  '
                f'<s>[{b["mc1"]},{b["s0"]})  '
                f'src[{b["s0"]},{b["s1"]})  '
                f'</s>[{b["s1"]},{b["s_close_end"]})'
            )
        rs = pos['recall_start']
        rec = (f'  recall:   '
               f'<f>[{rs},{pos["f0"]})  '
               f'w[{pos["f0"]},{pos["f1"]})  '
               f'</f>[{pos["f1"]},{pos["fc1"]})')
        if self.ponder_len > 0:
            rec += (f'  <p>[{pos["fc1"]},{pos["p0"]})  '
                    f'ponder[{pos["p0"]},{pos["p1"]})  '
                    f'</p>[{pos["p1"]},{pos["pc1"]})')
        rec += (f'  <c>[{pos["pc1"]},{pos["c0"]})  '
                f'out[{pos["c0"]},{pos["c1"]})  '
                f'</c>[{pos["c1"]},{pos["L"]})')
        lines.append(rec)
        return '\n'.join(lines)

    def to_spec_str(self) -> str:
        """Round-trip to canonical spec string."""
        act = f',active={self.active_slots}' if self.active_slots else ''
        frm = f',from={self.recall_from}'    if self.recall_from  else ''
        pnd = f'<p:{self.ponder_len}>'       if self.ponder_len   else ''
        block = f'<m:{self.slot_len}{act}><s:{self.seg_len}>'
        return f'{block * self.n_blocks}<f:{self.warmup_len}>{pnd}<c:{self.out_len}{frm}>'

    def __repr__(self) -> str:
        act = f',active={self.active_slots}' if self.active_slots else ''
        frm = f',from={self.recall_from}'    if self.recall_from  else ''
        pnd = f'<p:{self.ponder_len}>'       if self.ponder_len   else ''
        block = f'<m:{self.slot_len}{act}><s:{self.seg_len}>'
        return (f'SeqSpec  {(block + " ") * self.n_blocks}'
                f'<f:{self.warmup_len}>{pnd}<c:{self.out_len}{frm}>  '
                f'active_slots={self.active_slots}  '
                f'n_blocks={self.n_blocks}  L={self.L}')
