"""
kvmem/eval_weave.py — trajectory-generalization diagnostics for any HMNModel
checkpoint (solo/relay/flow), testing "does recall survive an interleaved or
repeated encode/query trajectory" beyond whatever fixed rhythm the checkpoint
was actually trained on. See docs/HMN_RECIPE.md's trajectory-taxonomy
section and kvmem/hmn.py's traj_batch/traj_stream/traj_interleave_delayed/
traj_repeat_query/traj_long_hop_recovery/traj_decay_curve for the underlying
mechanism (chunk_positions_traj/chunk_mask_fb_traj — generalizes
chunk_positions_flow to arbitrary interleaved 'E'/'S'/'Q' operation
sequences, see kvmem.hmn's trajectory DSL comment block for the grammar).

This is a TEST-ONLY tool — none of the patterns run here should be trained
on directly (that would defeat their purpose as generalization probes). It
works against ANY existing checkpoint zero-shot, since none of solo/relay/
flow were trained on interleaved trajectories at all — a checkpoint that
does well here despite never seeing these specific patterns is showing real
algorithmic generalization, not pattern-matching a trained rhythm.

The headline diagnostic: for repeat_query / long_hop_recovery / decay_curve,
compare a span's FIRST-occurrence match% against its LATER, repeated-
occurrence match% (extracted from ar_decode_traj_nokv's per-rec_block
turn_match_pcts, in trajectory order — the SAME span can legitimately appear
twice in one operations list). A large drop between first and repeated
occurrence is direct evidence the relay loses previously-recoverable
information as new content gets appended ("does a new round delete known
info" — the concern that motivated Stage `flow`'s design in the first
place). decay_curve isolates this cleanly from recall-accuracy-at-each-hop
(its intermediate hops are content-free no-ops, not real queries with their
own local recall task) — sweep --noop-hops to build an actual decay curve.

Usage:
    python3 -m kvmem.eval_weave --ckpt kvmem/logs/hmn_stage1_round0_chained/checkpoints/stage0_best.pt --device mps
    python3 -m kvmem.eval_weave --ckpt <path> --device mps --patterns repeat_query,interleave_delayed
    python3 -m kvmem.eval_weave --ckpt <path> --device mps --patterns long_hop_recovery --n-chunks 8
    python3 -m kvmem.eval_weave --ckpt <path> --device mps --patterns decay_curve --noop-hops 1,2,4,8
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from kvmem.hmn import (
    build_model,
    chunk_positions_traj,
    chunk_mask_fb_traj,
    ar_decode_traj_nokv,
    make_test_sequences,
    traj_batch,
    traj_stream,
    traj_interleave_delayed,
    traj_repeat_query,
    traj_long_hop_recovery,
    traj_decay_curve,
)

# batch/stream/interleave_delayed/repeat_query/long_hop_recovery take
# (n_chunks, window_chunks); decay_curve takes (n_noop_hops, window_chunks) —
# dispatched separately in run_pattern via the `noop_hops` arg.
_PATTERNS = dict(
    batch=traj_batch,
    stream=traj_stream,
    interleave_delayed=traj_interleave_delayed,
    repeat_query=traj_repeat_query,
    long_hop_recovery=traj_long_hop_recovery,
)


def _load(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    hp = ckpt['hp']
    hp_model = dict(
        V=hp['V'], d=hp['d'], n_layers=hp['n_layers'], n_heads=hp['n_heads'],
        block_type=hp.get('block_type', 'single_attn'),
        rope=hp.get('rope', True), yarn=hp.get('yarn', True),
        null_kv=hp.get('null_kv', True), rmsnorm=hp.get('rmsnorm', False),
        chunk_attn=0,
    )
    model = build_model(hp_model, device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model, hp


def run_pattern(model, hp: dict, device: torch.device, pattern_name: str,
                n_chunks: int, chunk_len: int, window_chunks: int = 2,
                noop_hops: int = 0, seq_idx: int = 0) -> dict:
    """Runs one trajectory pattern against one val sequence, returns
    per-rec_block match% in trajectory order plus the headline
    first-vs-repeated comparison when the pattern has a repeated span.
    noop_hops only used for pattern_name='decay_curve'."""
    state_len = hp.get('state_len', 8)
    state_vocab_size = hp.get('state_vocab_size', 2)
    warmup_len = hp['warmup_len']

    if pattern_name == 'decay_curve':
        ops = traj_decay_curve(noop_hops, window_chunks)
        eff_n_chunks = window_chunks  # decay_curve only ever encodes window_chunks chunks
    else:
        traj_fn = _PATTERNS[pattern_name]
        ops = traj_fn(n_chunks, window_chunks)
        eff_n_chunks = n_chunks

    built = chunk_positions_traj(chunk_len, state_len, warmup_len, ops,
                                 n_refine=0, state_vocab_size=state_vocab_size)
    pos_content, pos_mask, tags = built['pos_content'], built['pos_mask'], built['tags']
    mask_np = chunk_mask_fb_traj(pos_mask)

    val_seqs = make_test_sequences(eff_n_chunks * chunk_len)
    seq_bytes = list(val_seqs.values())[seq_idx % len(val_seqs)]
    chunks_list = [seq_bytes[k * chunk_len:(k + 1) * chunk_len] for k in range(eff_n_chunks)]

    r = ar_decode_traj_nokv(model, np.array(chunks_list), state_len,
                            state_vocab_size, mask_np, pos_content, tags, device)

    # ar_decode_traj_nokv's turn_match_pcts only has an entry per NON-'noop'
    # rec_block (no-ops have no recall target) — spans must be filtered the
    # same way, or zip silently truncates/misaligns them against each other.
    spans = [rb['span'] for rb in pos_content['rec_blocks'] if rb['type'] != 'noop']
    per_op = list(zip(spans, r['turn_match_pcts']))

    result = dict(pattern=pattern_name, n_chunks=eff_n_chunks, per_op=per_op,
                 overall_match_pct=r['match_pct'])

    # Headline first-vs-repeated comparison, if the last span duplicates an
    # earlier one (repeat_query / long_hop_recovery / decay_curve's defining
    # structure).
    last_span, last_match = per_op[-1]
    first_occurrence_idx = next(i for i, (s, _) in enumerate(per_op) if s == last_span)
    if first_occurrence_idx != len(per_op) - 1:
        first_match = per_op[first_occurrence_idx][1]
        result['first_vs_repeated'] = dict(
            span=last_span, first_match=first_match, repeated_match=last_match,
            drop=first_match - last_match, hops_between=len(per_op) - 1 - first_occurrence_idx,
        )

    return result


def main():
    p = argparse.ArgumentParser(description='Trajectory-generalization diagnostics for HMNModel checkpoints')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--patterns', default='batch,stream,interleave_delayed,repeat_query')
    p.add_argument('--n-chunks', type=int, default=4, help='use 8 for long_hop_recovery to stress more hops than training used')
    p.add_argument('--chunk-len', type=int, default=16)
    p.add_argument('--n-seqs', type=int, default=3)
    p.add_argument('--noop-hops', default='1,2,4,8', help='comma-separated hop counts for the decay_curve pattern')
    args = p.parse_args()

    device = torch.device(args.device)
    model, hp = _load(args.ckpt, device)
    patterns = args.patterns.split(',')

    for pname in patterns:
        if pname == 'decay_curve':
            for hops in [int(h) for h in args.noop_hops.split(',')]:
                print(f'\n=== pattern=decay_curve  noop_hops={hops} ===')
                drops = []
                for seq_idx in range(args.n_seqs):
                    r = run_pattern(model, hp, device, 'decay_curve', args.n_chunks, args.chunk_len,
                                    noop_hops=hops, seq_idx=seq_idx)
                    ops_str = '  '.join(f'{span}={m:.1f}%' for span, m in r['per_op'])
                    print(f'  seq{seq_idx}: {ops_str}  (overall={r["overall_match_pct"]:.1f}%)')
                    if 'first_vs_repeated' in r:
                        fvr = r['first_vs_repeated']
                        print(f'    -> span {fvr["span"]}: first={fvr["first_match"]:.1f}%  '
                             f'repeated={fvr["repeated_match"]:.1f}%  drop={fvr["drop"]:.1f}pp')
                        drops.append(fvr['drop'])
                if drops:
                    print(f'  MEAN drop @ {hops} hops across {len(drops)} seqs: {sum(drops)/len(drops):.1f}pp')
            continue

        if pname not in _PATTERNS:
            print(f'[eval_weave] unknown pattern {pname!r}, skipping (known: {list(_PATTERNS)}, decay_curve)')
            continue
        print(f'\n=== pattern={pname}  n_chunks={args.n_chunks} ===')
        drops = []
        for seq_idx in range(args.n_seqs):
            r = run_pattern(model, hp, device, pname, args.n_chunks, args.chunk_len, seq_idx=seq_idx)
            ops_str = '  '.join(f'{span}={m:.1f}%' for span, m in r['per_op'])
            print(f'  seq{seq_idx}: {ops_str}  (overall={r["overall_match_pct"]:.1f}%)')
            if 'first_vs_repeated' in r:
                fvr = r['first_vs_repeated']
                print(f'    -> span {fvr["span"]}: first={fvr["first_match"]:.1f}%  '
                     f'repeated={fvr["repeated_match"]:.1f}%  drop={fvr["drop"]:.1f}pp  '
                     f'(after {fvr["hops_between"]} intervening queries)')
                drops.append(fvr['drop'])
        if drops:
            print(f'  MEAN drop across {len(drops)} seqs: {sum(drops)/len(drops):.1f}pp '
                 f'(positive = information lost as the relay moved forward)')


if __name__ == '__main__':
    main()
