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
_MODE_RE   = re.compile(r'^m(end|int|acc|mix)$')


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
    n_steps    = defaults.get('n_steps', 40000)
    mem_window = defaults.get('mem_window', -1)
    mode       = defaults.get('mode', 'end')   # end|int|acc|mix

    for p in parts:
        if _BLOCKS_RE.match(p):
            n_blocks = int(_BLOCKS_RE.match(p).group(1))
        elif _ROUTES_RE.match(p):
            recall_froms = _parse_routes(p)
        elif _WINDOW_RE.match(p):
            mem_window = int(_WINDOW_RE.match(p).group(1))
        elif _MODE_RE.match(p):
            mode = _MODE_RE.match(p).group(1)
        elif _STEPS_RE.match(p):
            n_steps = _parse_steps(p)
        else:
            raise ValueError(f'Unknown stage token part: {p!r} in stage {token!r}')

    if n_blocks is None:
        raise ValueError(f'Missing n_blocks (nN) in stage: {token!r}')
    if recall_froms is None and mode not in ('acc',):
        raise ValueError(f'Missing routes (rK or r[K,...]) in stage: {token!r}  (use macc for ingest-only)')

    # Validate recall_froms against n_blocks (skip for accumulate-only mode)
    rfs = recall_froms if isinstance(recall_froms, list) else [recall_froms] if recall_froms is not None else []
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
        recall_froms=recall_froms if recall_froms is not None else 0,
        mem_window=mem_window,
        mode=mode,
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

def _bracket_split(s: str) -> list[str]:
    """Split on commas not inside brackets."""
    tokens, depth, buf = [], 0, []
    for ch in s:
        if ch == '[': depth += 1
        elif ch == ']': depth -= 1
        if ch == ',' and depth == 0:
            t = ''.join(buf).strip()
            if t: tokens.append(t)
            buf = []
        else:
            buf.append(ch)
    t = ''.join(buf).strip()
    if t: tokens.append(t)
    return tokens


def parse_curriculum(dsl: str, **stage_defaults) -> tuple[SeqSpec, list[dict], list[tuple]]:
    """
    Parse a curriculum DSL string.

    Returns (seq_spec, curriculum_list, eval_configs) where:
      curriculum_list : list of stage dicts for hp['curriculum']
      eval_configs    : list of (n_blocks, recall_from) tuples tested at every eval step

    Grammar extensions:
      @eval:nN/rK,...  explicit eval configs (tested at every eval_every step)
      +nN/rK/Xk        overlap stage — run alongside the previous stage in the same
                        training steps (same batch, mixed recall_froms and n_blocks)

    Examples:
      # Basic:
      "<x:16><z:7><h:1><q:4><y:8> | n2/r[0,1]/160k"

      # With explicit eval:
      "<x:16><z:7><h:1><q:4><y:8> | n2/r[0,1]/160k @eval:n1/r0,n2/r0,n2/r1"

      # Overlapping stages (same steps, different routing mixed per example):
      "<x:16><z:7><h:1><q:4><y:8> | n2/r0/80k +n2/r1, n2/r[0,1]/80k"
      Stage 0 mixes n2/r0 and n2/r1 in recall_froms=[0,1].
      Stage 1 is explicit mixed.

    If @eval is omitted: eval_configs = sorted set of (n_blocks, recall_from)
    from all stages, plus (1, 0) baseline.
    """
    dsl = dsl.strip()
    if '|' not in dsl:
        raise ValueError('Curriculum DSL must contain "|" separating seq spec from stages')

    pipe_idx  = dsl.index('|')
    seq_str   = dsl[:pipe_idx].strip()
    rest      = dsl[pipe_idx+1:].strip()

    # Extract optional @eval: annotation
    eval_str = None
    if '@eval:' in rest:
        eval_idx = rest.index('@eval:')
        eval_str = rest[eval_idx + len('@eval:'):].strip()
        rest     = rest[:eval_idx].strip()

    seq = parse_seq(seq_str)

    # Parse stage tokens, handling '+' overlap prefix
    raw_tokens = _bracket_split(rest)
    stage_tokens = []  # list of lists (each inner list = overlapping group)
    current_group = []
    for tok in raw_tokens:
        tok = tok.strip()
        if tok.startswith('+'):
            current_group.append(tok[1:].strip())   # add to current overlap group
        else:
            if current_group:
                stage_tokens.append(current_group)
            current_group = [tok]
    if current_group:
        stage_tokens.append(current_group)

    if not stage_tokens:
        raise ValueError('No stages found after "|"')

    # Parse each group: overlapping tokens merge into one stage with combined recall_froms
    curriculum = []
    for group in stage_tokens:
        if len(group) == 1:
            curriculum.append(_parse_stage(group[0], seq, stage_defaults))
        else:
            # Merge overlap group: first token sets n_steps/window, all contribute recall_froms
            base = _parse_stage(group[0], seq, stage_defaults)
            all_rfs = []
            for tok in group:
                s = _parse_stage(tok, seq, stage_defaults)
                rfs = s['recall_froms'] if isinstance(s['recall_froms'], list) else [s['recall_froms']]
                all_rfs.extend(rfs)
            base['recall_froms'] = sorted(set(all_rfs))
            curriculum.append(base)

    # Parse eval configs
    if eval_str:
        eval_configs = []
        for tok in _bracket_split(eval_str):
            tok = tok.strip()
            # Format: nN/rK  e.g. n1/r0, n2/r1, n2/r[0,1]
            parts = tok.split('/', 1)
            m_nb  = _BLOCKS_RE.match(parts[0])
            if not m_nb or len(parts) < 2:
                raise ValueError(f'Invalid eval token: {tok!r}  (expected nN/rK)')
            nb   = int(m_nb.group(1))
            rf   = _parse_routes(parts[1])
            rfs  = rf if isinstance(rf, list) else [rf]
            for r in rfs:
                eval_configs.append((nb, r))
    else:
        seen = {(1, 0)}
        for s in curriculum:
            nb  = s['n_blocks']
            rfs = s['recall_froms'] if isinstance(s['recall_froms'], list) else [s['recall_froms']]
            for rf in rfs:
                seen.add((nb, rf))
        eval_configs = sorted(seen)

    return seq, curriculum, eval_configs


# ---------------------------------------------------------------------------
# CLI / quick inspection
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    dsl = sys.argv[1] if len(sys.argv) > 1 else (
        "<x:16><z:7><h:1><q:4><y:8> | n2/r[0,1]/160k @eval:n1/r0,n2/r0,n2/r1"
    )
    spec, cur, evals = parse_curriculum(dsl, B=16, dataset_size=20000)
    print(f'SeqSpec: {spec}')
    print(f'Curriculum ({len(cur)} stages):')
    for i, s in enumerate(cur):
        print(f'  s{i}: n={s["n_blocks"]}  rf={s["recall_froms"]}  mode={s["mode"]}  w={s["mem_window"]}  steps={s["n_steps"]}')
    print(f'Eval configs: {evals}')
