"""
kvmem/curriculum_dsl.py — DSL for training curriculum specs.

Parses a compact string describing a sequence of training stages.
Produces a list of stage dicts compatible with train_role() curriculum.

Grammar
-------
    curriculum ::= seq_spec '|' stage (',' stage)*
    seq_spec   ::= sequence DSL string (no | characters)
    stage      ::= blocks '/' routes ['/' steps] ['/w' window]
                 | blocks '/' routes '/' steps ['/w' window]

    blocks  ::= 'n' int                         e.g. n1, n2
    routes  ::= 'r' int | 'r[' int (',' int)* ']'   e.g. r0, r1, r[0,1]
    steps   ::= int 'k' | int                   e.g. 40k, 80000
    window  ::= 'w' int                         e.g. w-1, w1, w2

Defaults (if omitted):
    steps   = 40000
    window  = -1  (full history)

Examples
--------
    # Single sequence spec, 5 stages:
    curriculum = parse_curriculum(
        "<x:16><z:7><h:1><q:4><y:8> | n1/r0/40k, n2/r1/40k, n2/r0/40k, n2/r[0,1]/80k, n2/r[0,1]/80k/w1"
    )

    # With shared training params:
    curriculum = parse_curriculum(
        "<x:16><z:7><h:1><q:4><y:8> | n1/r0/40k, n2/r[0,1]/80k",
        B=16, dataset_size=20000
    )

    # Returns list of stage dicts ready for hp['curriculum'].
    # seq_spec params are expanded into each stage dict.
    # Additional kwargs are merged into every stage.
"""

from __future__ import annotations
import re
from kvmem.seq_dsl import parse_seq, SeqSpec


# ---------------------------------------------------------------------------
# Stage parser
# ---------------------------------------------------------------------------

_STEPS_RE  = re.compile(r'^(\d+)k?$', re.IGNORECASE)
_BLOCKS_RE = re.compile(r'^n(\d+)$')
_ROUTES_RE = re.compile(r'^r(\d+|\[\d+(?:,\d+)*\])$')
_WINDOW_RE = re.compile(r'^w(-?\d+)$')


def _parse_steps(s: str) -> int:
    m = _STEPS_RE.match(s)
    if not m:
        raise ValueError(f'Invalid steps token: {s!r}  (expected e.g. 40k or 80000)')
    n = int(m.group(1))
    return n * 1000 if s.lower().endswith('k') else n


def _parse_routes(s: str) -> list[int] | int:
    m = _ROUTES_RE.match(s)
    if not m:
        raise ValueError(f'Invalid routes token: {s!r}  (expected e.g. r0, r1, r[0,1])')
    inner = m.group(1)
    if inner.startswith('['):
        vals = [int(x) for x in inner[1:-1].split(',')]
        return vals if len(vals) > 1 else vals[0]
    return int(inner)


def _parse_stage(token: str, seq: SeqSpec, defaults: dict) -> dict:
    """
    Parse one stage token like 'n2/r[0,1]/80k/w1' into a stage dict.
    """
    parts = [p.strip() for p in token.strip().split('/') if p.strip()]
    n_blocks = None
    recall_froms = None
    n_steps = defaults.get('n_steps', 40000)
    mem_window = defaults.get('mem_window', -1)

    for p in parts:
        if _BLOCKS_RE.match(p):
            n_blocks = int(_BLOCKS_RE.match(p).group(1))
        elif _ROUTES_RE.match(p):
            recall_froms = _parse_routes(p)
        elif _WINDOW_RE.match(p):
            mem_window = int(_WINDOW_RE.match(p).group(1))
        elif _STEPS_RE.match(p):
            n_steps = _parse_steps(p)
        else:
            raise ValueError(f'Unknown stage token part: {p!r} in stage {token!r}')

    if n_blocks is None:
        raise ValueError(f'Missing n_blocks (nN) in stage: {token!r}')
    if recall_froms is None:
        raise ValueError(f'Missing routes (rK or r[K,...]) in stage: {token!r}')

    # Validate recall_froms against n_blocks
    rfs = recall_froms if isinstance(recall_froms, list) else [recall_froms]
    for rf in rfs:
        if rf >= n_blocks:
            raise ValueError(
                f'recall_from={rf} out of range for n_blocks={n_blocks} in stage {token!r}')

    stage = dict(
        # Sequence params from SeqSpec (shared across all stages)
        seg_len=seq.seg_len,
        slot_len=seq.slot_len,
        intermed_len=seq.intermed_len,
        warmup_len=seq.warmup_len,
        out_len=seq.out_len,
        # Stage-specific
        n_blocks=n_blocks,
        recall_froms=recall_froms,
        mem_window=mem_window,
        n_steps=n_steps,
    )
    # Merge extra defaults (B, dataset_size, etc.)
    for k, v in defaults.items():
        if k not in stage:
            stage[k] = v
    return stage


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_curriculum(dsl: str, **stage_defaults) -> tuple[SeqSpec, list[dict]]:
    """
    Parse a curriculum DSL string.

    Returns (seq_spec, curriculum_list) where curriculum_list is ready for
    hp['curriculum'].

    Args:
        dsl           : curriculum DSL string
        stage_defaults: shared params merged into every stage (B, dataset_size, etc.)

    Example:
        spec, cur = parse_curriculum(
            "<x:16><z:7><h:1><q:4><y:8> | n1/r0/40k, n2/r1/40k, n2/r[0,1]/80k",
            B=16, dataset_size=20000
        )
        hp['curriculum'] = cur
    """
    dsl = dsl.strip()

    # Split on ':' to get seq_spec and stages
    if '|' not in dsl:
        raise ValueError(f'Curriculum DSL must contain "|" separating seq spec from stages')

    colon_idx = dsl.index('|')
    seq_str   = dsl[:colon_idx].strip()
    stages_str = dsl[colon_idx+1:].strip()

    seq = parse_seq(seq_str)

    # Split on commas NOT inside brackets (to handle r[0,1])
    stage_tokens = []
    depth, buf = 0, []
    for ch in stages_str:
        if ch == '[': depth += 1
        elif ch == ']': depth -= 1
        if ch == ',' and depth == 0:
            t = ''.join(buf).strip()
            if t: stage_tokens.append(t)
            buf = []
        else:
            buf.append(ch)
    t = ''.join(buf).strip()
    if t: stage_tokens.append(t)
    if not stage_tokens:
        raise ValueError('No stages found after ":"')

    curriculum = [_parse_stage(t, seq, stage_defaults) for t in stage_tokens]
    return seq, curriculum


# ---------------------------------------------------------------------------
# CLI / quick inspection
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    dsl = sys.argv[1] if len(sys.argv) > 1 else (
        "<x:16><z:7><h:1><q:4><y:8> | n1/r0/40k, n2/r1/40k, n2/r0/40k, n2/r[0,1]/80k, n2/r[0,1]/80k/w1"
    )
    spec, cur = parse_curriculum(dsl, B=16, dataset_size=20000)
    print(f'SeqSpec: {spec}')
    print(f'Curriculum ({len(cur)} stages):')
    for i, s in enumerate(cur):
        rf = s['recall_froms']
        w  = s['mem_window']
        print(f'  s{i}: n_blocks={s["n_blocks"]}  recall={rf}  mem_window={w}  steps={s["n_steps"]}')
