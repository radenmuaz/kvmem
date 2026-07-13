"""
chunk_mask_fb_windowed — adds an optional `ir_slot_window` parameter to
kvmem.train_hmn_chunk.chunk_mask_fb's masking rules, controlling how many
PRIOR TURNS' SLOT tokens (within the same window's own IQ->IR1->IR2->...
chain) a later turn's SLOT_A/argmax/SLOT_B rows may attend to.

RNN framing (this is structurally a recurrent update — each turn re-writes
fresh SLOT tokens conditioned on the previous turn's argmax output, the same
"review and strengthen a memory trace" pattern as the macro-level SRS
schedule, just operating turn-by-turn within one forward pass — see
docs/SRS_RECIPE.md "IR loop as micro-SRS recurrence"):

  ir_slot_window = None (default) -> UNBOUNDED, identical to chunk_mask_fb's
                                      existing behavior: every turn's SLOT_A/
                                      argmax/SLOT_B rows can see ALL earlier
                                      turns' SLOT columns in the same window
                                      (no restriction — this is what every
                                      run in this project has used so far).
  ir_slot_window = 1                -> only the CURRENT turn's own SLOT is
                                        visible (all earlier turns' SLOTs in
                                        the same window are blocked) — like
                                        an RNN with only h_t, no history.
                                        Forces the model to rely purely on
                                        the explicit `argmax` copy as its
                                        only feedback channel.
  ir_slot_window = 2                -> current turn + immediately preceding
                                        turn's SLOT visible (h_t, h_{t-1}).
  ir_slot_window = N                -> current + previous (N-1) turns
                                        visible (sliding window).

Does NOT affect Rule 3b (cross-window nochain) — that already fully blocks
all cross-window SLOT attention regardless of this parameter; this only adds
a restriction WITHIN a single window's own turn sequence, which the base
chunk_mask_fb currently leaves completely unrestricted (a structural
byproduct, not a deliberate design choice — see docs/SRS_RECIPE.md's
"is there a parameter for how much a memory slot can attend to past memory
slots" discussion).

Default is unbounded (matches chunk_mask_fb exactly, verified in
verify_unbounded_matches_baseline() below) so existing configs/runs are
unaffected unless `ir_slot_window` is explicitly set.
"""
from __future__ import annotations

import numpy as np


def chunk_mask_fb_windowed(pos: dict, ir_slot_window: int | None = None,
                           rec_spans: list[tuple] | None = None) -> np.ndarray:
    """
    rec_spans: parallel list (same length/order as pos['rec_blocks']) giving
    each block's window span tuple, e.g. pos_content['rec_blocks'][i]['span'].
    pos['rec_blocks'] (the MASK view, tag-widened offsets) doesn't carry
    'span' itself — only pos_content's rec_blocks do (see
    chunk_positions_srs_tagged). Required only when ir_slot_window is not None.
    """
    L = pos['L']
    r = np.arange(L); c = np.arange(L)
    causal  = c[None, :] <= r[:, None]
    blocked = np.zeros((L, L), dtype=bool)

    enc_blocks = pos['enc_blocks']
    rec_blocks = pos['rec_blocks']

    is_any_chunk = np.zeros(L, dtype=bool)
    for b in enc_blocks:
        is_any_chunk |= (c >= b['s0']) & (c < b['s1'])

    is_any_rec_output = np.zeros(L, dtype=bool)
    for rb2 in rec_blocks:
        is_any_rec_output |= (c >= rb2['c0']) & (c < rb2['c1'])

    # Rule 2: encoding SLOT_k blocked from chunk_j (j!=k)
    for k, b in enumerate(enc_blocks):
        sl_row = (r >= b['sl0']) & (r < b['sl1'])
        for j, bj in enumerate(enc_blocks):
            if j == k: continue
            blocked |= sl_row[:, None] & ((c >= bj['s0']) & (c < bj['s1']))[None, :]

    # Group rec_blocks by span (window) in order, to compute each block's
    # turn-index within its own window's chain (0=IQ, 1=IR1, 2=IR2, ...).
    span_turn_idx: dict[int, int] = {}   # rec_block id -> turn index within its window
    if ir_slot_window is not None:
        assert rec_spans is not None and len(rec_spans) == len(rec_blocks), \
            'rec_spans required (parallel to pos["rec_blocks"]) when ir_slot_window is set'
        span_seen: dict[tuple, int] = {}
        for rb, span in zip(rec_blocks, rec_spans):
            t = span_seen.get(span, 0)
            span_turn_idx[id(rb)] = t
            span_seen[span] = t + 1

    for i_rb, rb in enumerate(rec_blocks):
        if rb['type'] == 'iq':
            sl_row = (r >= rb['sl0']) & (r < rb['sl1'])
            blocked |= sl_row[:, None] & is_any_chunk[None, :]
            prior_all = np.zeros(L, dtype=bool)
            for prev_rb in rec_blocks[:i_rb]:
                if prev_rb['type'] == 'iq':
                    prior_all |= (c >= prev_rb['sl0']) & (c < prev_rb['sl1'])
                    prior_all |= (c >= prev_rb['w0'])  & (c < prev_rb['w1'])
                    prior_all |= (c >= prev_rb['c0'])  & (c < prev_rb['c1'])
                else:
                    prior_all |= (c >= prev_rb['sla0']) & (c < prev_rb['sla1'])
                    prior_all |= (c >= prev_rb['am0'])  & (c < prev_rb['am1'])
                    prior_all |= (c >= prev_rb['slb0']) & (c < prev_rb['slb1'])
                    prior_all |= (c >= prev_rb['w0'])   & (c < prev_rb['w1'])
                    prior_all |= (c >= prev_rb['c0'])   & (c < prev_rb['c1'])
            blocked |= sl_row[:, None] & prior_all[None, :]
            if rb['w0'] < rb['w1']:
                wm_row = (r >= rb['w0']) & (r < rb['w1'])
                own = (c >= rb['sl0']) & (c < rb['sl1']) | (c >= rb['w0']) & (c < rb['w1'])
                blocked |= wm_row[:, None] & ~own[None, :]
            out_row = (r >= rb['c0']) & (r < rb['c1'])
            own = ((c >= rb['sl0']) & (c < rb['sl1']) |
                   (c >= rb['w0'])  & (c < rb['w1'])  |
                   (c >= rb['c0'])  & (c < rb['c1']))
            blocked |= out_row[:, None] & ~own[None, :]

        else:  # type == 'ir'
            sla_row = (r >= rb['sla0']) & (r < rb['sla1'])
            am_row  = (r >= rb['am0'])  & (r < rb['am1'])
            slb_row = (r >= rb['slb0']) & (r < rb['slb1'])
            blocked |= (sla_row | am_row | slb_row)[:, None] & (is_any_chunk | is_any_rec_output)[None, :]

            # NEW: ir_slot_window restriction, within-window only.
            if ir_slot_window is not None:
                t = span_turn_idx[id(rb)]
                cutoff = t - (ir_slot_window - 1)   # turns with index < cutoff are blocked
                if cutoff > 0:
                    too_old = np.zeros(L, dtype=bool)
                    my_span = rec_spans[i_rb]
                    for j_rb, prev_rb in enumerate(rec_blocks[:i_rb]):
                        if rec_spans[j_rb] != my_span:
                            continue
                        prev_t = span_turn_idx[id(prev_rb)]
                        if prev_t >= cutoff:
                            continue
                        if prev_rb['type'] == 'iq':
                            too_old |= (c >= prev_rb['sl0']) & (c < prev_rb['sl1'])
                        else:
                            too_old |= (c >= prev_rb['sla0']) & (c < prev_rb['sla1'])
                            too_old |= (c >= prev_rb['slb0']) & (c < prev_rb['slb1'])
                    blocked |= (sla_row | am_row | slb_row)[:, None] & too_old[None, :]

            wm_row  = (r >= rb['w0'])  & (r < rb['w1'])
            out_row = (r >= rb['c0'])  & (r < rb['c1'])
            own_slb = (c >= rb['slb0']) & (c < rb['slb1'])
            own_wm  = (c >= rb['w0'])   & (c < rb['w1'])
            own_out = (c >= rb['c0'])   & (c < rb['c1'])
            allowed = own_slb | own_wm | own_out
            if rb['w0'] < rb['w1']:
                blocked |= wm_row[:, None] & ~(own_slb | own_wm)[None, :]
            blocked |= out_row[:, None] & ~allowed[None, :]

    visible = causal & ~blocked
    return np.where(visible, 0.0, -1e9).astype(np.float32)


def verify_unbounded_matches_baseline():
    """ir_slot_window=None must reproduce kvmem.train_hmn_chunk.chunk_mask_fb
    exactly, bit-for-bit — this is the safety check for the 'default is
    unbounded, no breaking changes' guarantee."""
    from kvmem.train_hmn_chunk import chunk_mask_fb
    from experiments.chat_tags.positions import chunk_positions_srs_tagged

    built = chunk_positions_srs_tagged(4, 16, 8, 8, [(0, 2), (1, 3), (2, 4)], n_refine=2)
    pos_mask = built['pos_mask']
    baseline = chunk_mask_fb(pos_mask)
    windowed = chunk_mask_fb_windowed(pos_mask, ir_slot_window=None)
    assert np.array_equal(baseline, windowed), 'ir_slot_window=None does not match baseline!'

    # Also sanity-check a restricted value actually removes some visibility
    # (mask has strictly MORE -1e9 entries than baseline) without breaking
    # causality (still a subset of the causal upper-triangle).
    rec_spans = [rb['span'] for rb in built['pos_content']['rec_blocks']]
    restricted = chunk_mask_fb_windowed(pos_mask, ir_slot_window=1, rec_spans=rec_spans)
    n_blocked_base = np.sum(baseline < 0)
    n_blocked_restricted = np.sum(restricted < 0)
    assert n_blocked_restricted > n_blocked_base, \
        'ir_slot_window=1 should block strictly more than unbounded'
    causal = (np.arange(built['L'])[None, :] <= np.arange(built['L'])[:, None])
    assert np.all((restricted >= 0) <= causal), 'restricted mask violates causality'
    return True


if __name__ == '__main__':
    verify_unbounded_matches_baseline()
    print('OK: ir_slot_window=None matches chunk_mask_fb exactly')
