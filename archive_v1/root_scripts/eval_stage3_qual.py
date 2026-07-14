"""
Stage 3 qualitative eval: per-window breakdown + stitched full-sequence decode.

Runs on the best/end checkpoint of hmn_chunk_local_64 and prints:
  - Each window independently (ar_decode_chunk_fb_kv, single-window pos/mask)
  - Full stitched sequence (ar_decode_chunk_fb_stitch_kv)
  - Side-by-side ref vs gen with window boundaries and mismatch markers

Usage:
    python3 eval_stage3_qual.py [--ckpt PATH] [--device mps|cpu|cuda]
"""
import argparse, sys
import numpy as np
import torch
from kvmem.model import build_model
from kvmem.train_hmn_chunk import (
    ar_decode_chunk_fb_kv, ar_decode_chunk_fb_stitch_kv,
    chunk_positions_fb_localrefine, chunk_mask_fb,
)
from kvmem.utils import make_test_sequences

# ── Config (matches hmn_chunk_local_64.py) ───────────────────────────────────
SLOT_LEN   = 4
SLOT_COUNT = 2
CHUNK_LEN  = 16
N_CHUNKS   = 4
WARMUP_LEN = 8
WINDOWS    = [(0, 2), (1, 3), (2, 4)]
N_REFINE   = 2
SRC_LEN    = N_CHUNKS * CHUNK_LEN  # 64

HP_MODEL = dict(V=268, d=64, n_layers=4, n_heads=4, d_ff=256,
                rope=True, yarn=True, null_kv=True, compile=False)

WINDOW_BYTE_RANGES = [(ws * CHUNK_LEN, we * CHUNK_LEN) for ws, we in WINDOWS]
# overlap regions between consecutive windows
OVERLAPS = [
    (WINDOW_BYTE_RANGES[i+1][0], WINDOW_BYTE_RANGES[i][1])
    for i in range(len(WINDOWS) - 1)
]

# ── Arg parse ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt',   default='logs/hmn_chunk_local_64/checkpoints/stage0_end.pt')
parser.add_argument('--device', default='mps')
parser.add_argument('--best',   action='store_true', help='use stage0_best.pt instead')
args = parser.parse_args()
if args.best:
    args.ckpt = args.ckpt.replace('stage0_end.pt', 'stage0_best.pt')

device = torch.device(args.device)
print(f'Loading {args.ckpt}')
ckpt  = torch.load(args.ckpt, map_location=device)
model = build_model(HP_MODEL, device)
model.load_state_dict(ckpt['model'])
model.eval()
print(f'Loaded (step={ckpt.get("step","?")}, val={ckpt.get("val_mean","?")}%)')
print()

# ── Build per-window and full-window pos/mask ─────────────────────────────────
schedule = [(0, N_CHUNKS)]

per_window = []
for win in WINDOWS:
    pos  = chunk_positions_fb_localrefine(N_CHUNKS, CHUNK_LEN, SLOT_LEN, WARMUP_LEN,
                                          [win], N_REFINE)
    mask = chunk_mask_fb(pos)
    per_window.append((win, pos, mask))

full_pos  = chunk_positions_fb_localrefine(N_CHUNKS, CHUNK_LEN, SLOT_LEN, WARMUP_LEN,
                                           WINDOWS, N_REFINE)
full_mask = chunk_mask_fb(full_pos)

# ── Sequences ─────────────────────────────────────────────────────────────────
seqs = make_test_sequences(SRC_LEN)


# ── Display helpers ───────────────────────────────────────────────────────────
def _fmt_byte(b: int) -> str:
    """3-char column: show as decimal."""
    return f'{b:3d}'

def _cmp_char(ref: int, gen: int) -> str:
    return '  .' if ref == gen else '  X'

def _print_comparison(ref_bytes, gen_bytes, warmup_len, label='', window_boundaries=None):
    """Print ref / gen / diff rows with optional window boundary markers."""
    n = len(ref_bytes)
    # boundary set: byte positions where a new window starts (0-indexed within output)
    bounds = set(window_boundaries or [])

    def _row(name, vals, mark_fn=None):
        parts = [f'{name:<6}|']
        for i, v in enumerate(vals):
            sep = ' |' if i in bounds else ''
            s = _fmt_byte(v) if mark_fn is None else mark_fn(ref_bytes[i], v)
            parts.append(sep + s)
        print(''.join(parts))

    if label:
        print(f'  [{label}]')
    _row('ref', ref_bytes)
    _row('gen', gen_bytes)
    # diff row
    diffs = [_cmp_char(r, g) for r, g in zip(ref_bytes, gen_bytes)]
    parts = ['diff  |']
    for i, d in enumerate(diffs):
        sep = ' |' if i in bounds else ''
        parts.append(sep + d)
    print(''.join(parts))
    n_wrong = sum(r != g for r, g in zip(ref_bytes, gen_bytes))
    print(f'  match {n-n_wrong}/{n} = {100*(n-n_wrong)/n:.1f}%')


# ── Per-window eval ───────────────────────────────────────────────────────────
print('=' * 72)
print('PER-WINDOW RECALL (each window evaluated independently)')
print('=' * 72)

win_match_table = {}  # win -> {sname: match_pct}
for (win, wpos, wmask) in per_window:
    ws, we = win
    byte_lo, byte_hi = ws * CHUNK_LEN, we * CHUNK_LEN
    print(f'\n--- Window {win}  bytes {byte_lo}-{byte_hi-1} ---')
    win_match_table[win] = {}
    matches = []
    for sname, seq in seqs.items():
        chunks_arr = np.array(seq, dtype=np.int64).reshape(N_CHUNKS, CHUNK_LEN)
        r = ar_decode_chunk_fb_kv(model, chunks_arr, SLOT_LEN, SLOT_COUNT,
                                  schedule, wmask, wpos, device)
        win_match_table[win][sname] = r['match_pct']
        matches.append(r['match_pct'])

        ref_bytes = np.array(seq[byte_lo:byte_hi], dtype=np.int64)
        gen_full  = np.array(r['decoded_bytes'], dtype=np.int64)
        # decoded_bytes is the output region (warmup_len:); prepend warmup from GT
        warmup    = ref_bytes[:WARMUP_LEN]
        gen_out   = gen_full[:byte_hi - byte_lo - WARMUP_LEN]
        gen_bytes = np.concatenate([warmup, gen_out])

        print(f'  {sname:<15} BPB={r["bpb"]:.3f}  match={r["match_pct"]:.1f}%')
        _print_comparison(ref_bytes, gen_bytes, WARMUP_LEN, sname)
        print()
    print(f'  MEAN match={np.mean(matches):.1f}%')

# ── Stitched full-sequence eval ───────────────────────────────────────────────
print()
print('=' * 72)
print('STITCHED FULL-SEQUENCE (prolonged AR, only bytes 0-7 seeded from GT)')
print(f'Window boundaries at output bytes: '
      f'{[b - WARMUP_LEN for ws,we in WINDOWS for b in [ws*CHUNK_LEN] if b > 0]}')
print('Overlaps:', [(lo, hi) for lo, hi in OVERLAPS])
print('=' * 72)

# byte offsets within the output (after warmup) where each window starts
# output region = bytes WARMUP_LEN..SRC_LEN-1, i.e. 56 bytes
# window i starts at byte ws*CHUNK_LEN; relative to output start (WARMUP_LEN):
win_output_starts = [max(0, ws * CHUNK_LEN - WARMUP_LEN) for ws, we in WINDOWS]

stitch_matches = []
for sname, seq in seqs.items():
    chunks_arr = np.array(seq, dtype=np.int64).reshape(N_CHUNKS, CHUNK_LEN)
    r = ar_decode_chunk_fb_stitch_kv(model, chunks_arr, SLOT_LEN, SLOT_COUNT,
                                     full_mask, full_pos, device)
    stitch_matches.append(r['match_pct'])

    ref_full   = np.array(seq, dtype=np.int64)
    # decoded_bytes is the full (src_len,) stitched buffer; -1 = never decoded
    dec_buf    = np.array(r['decoded_bytes'], dtype=np.int64)
    # seeded warmup (bytes 0:wl) from GT, output (bytes wl:) from model
    gen_full   = np.where(np.arange(SRC_LEN) < WARMUP_LEN, ref_full, dec_buf)

    print(f'\n  {sname:<15} BPB={r["bpb"]:.3f}  match={r["match_pct"]:.1f}%')

    # Per-window match breakdown for this sequence
    for win in WINDOWS:
        ws, we = win
        pm = win_match_table[win].get(sname, float('nan'))
        print(f'    window {win} bytes {ws*CHUNK_LEN:2d}-{we*CHUNK_LEN-1:2d}: {pm:.1f}%')

    # Overlap byte ranges with per-byte match
    for (ov_lo, ov_hi) in OVERLAPS:
        ref_ov = ref_full[ov_lo:ov_hi]
        gen_ov = gen_full[ov_lo:ov_hi]
        n_ok   = int(np.sum(ref_ov == gen_ov))
        print(f'    overlap bytes {ov_lo:2d}-{ov_hi-1:2d}: {n_ok}/{ov_hi-ov_lo} = {100*n_ok/(ov_hi-ov_lo):.0f}%')

    # Full side-by-side comparison with window boundary markers
    # output bytes (wl:) broken into groups of 8 for readability
    out_ref = ref_full[WARMUP_LEN:]
    out_gen = gen_full[WARMUP_LEN:]
    # window start offsets within output region
    win_bounds_in_output = sorted(set(
        ws * CHUNK_LEN - WARMUP_LEN for ws, we in WINDOWS if ws * CHUNK_LEN > WARMUP_LEN
    ))
    _print_comparison(out_ref, out_gen, 0,
                      label=f'{sname} output (bytes {WARMUP_LEN}..{SRC_LEN-1})',
                      window_boundaries=win_bounds_in_output)
    print()

print()
print(f'STITCH MEAN match={np.mean(stitch_matches):.1f}%')

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print('=' * 72)
print('SUMMARY')
print('=' * 72)
header = f'{"seq":<15}' + ''.join(f'  win{i}' for i in range(len(WINDOWS))) + '  stitch'
print(header)
for i, (sname, seq) in enumerate(seqs.items()):
    wins = ''.join(f'  {win_match_table[w].get(sname,0):5.1f}' for w in WINDOWS)
    stitch = stitch_matches[i]
    print(f'{sname:<15}{wins}  {stitch:5.1f}')
win_means = [np.mean(list(win_match_table[w].values())) for w in WINDOWS]
stitch_mean = np.mean(stitch_matches)
means_row = ''.join(f'  {m:5.1f}' for m in win_means)
print(f'{"MEAN":<15}{means_row}  {stitch_mean:5.1f}')
