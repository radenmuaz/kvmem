"""
Generic qualitative eval for any hmn_chunk_local_* checkpoint.

Runs per-window independent recall + full stitched prolonged-AR decode.
Prints side-by-side ref/gen with mismatch markers and summary table.

Usage:
    python3 eval_fb_qual.py --ckpt logs/hmn_chunk_local_128/checkpoints/stage0_end.pt
    python3 eval_fb_qual.py --ckpt PATH --n-chunks 8 --windows "0,2 1,3 2,4 3,5 4,6 5,7 6,8"
    python3 eval_fb_qual.py --best  # use stage0_best.pt in auto-detected dir
"""
import argparse, os
import numpy as np
import torch
from kvmem.model import build_model
from kvmem.train_hmn_chunk import (
    ar_decode_chunk_fb_kv, ar_decode_chunk_fb_stitch_kv,
    chunk_positions_fb_localrefine, chunk_mask_fb,
)
from kvmem.utils import make_test_sequences

# ── Arg parse ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt',      default=None)
parser.add_argument('--best',      action='store_true', help='use stage0_best.pt')
parser.add_argument('--device',    default='mps')
parser.add_argument('--n-chunks',  type=int, default=None,
                    help='override n_chunks (auto-detected from checkpoint hp)')
parser.add_argument('--chunk-len', type=int, default=16)
parser.add_argument('--slot-len',  type=int, default=4)
parser.add_argument('--slot-count',type=int, default=2)
parser.add_argument('--warmup-len',type=int, default=8)
parser.add_argument('--n-refine',  type=int, default=2)
parser.add_argument('--windows',   default=None,
                    help='space-separated "start,end" pairs e.g. "0,2 1,3 2,4"')
parser.add_argument('--val-n-seqs',type=int, default=8)
args = parser.parse_args()

# ── Resolve checkpoint ─────────────────────────────────────────────────────────
if args.ckpt is None:
    # scan logs/ for the most recently modified stage0_end or stage0_best
    import glob
    pattern = 'logs/*/checkpoints/stage0_end.pt'
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not candidates:
        raise FileNotFoundError('No stage0_end.pt found in logs/; pass --ckpt explicitly')
    args.ckpt = candidates[0]
    print(f'Auto-detected checkpoint: {args.ckpt}')

if args.best:
    args.ckpt = args.ckpt.replace('stage0_end.pt', 'stage0_best.pt')

device = torch.device(args.device)
print(f'Loading {args.ckpt}')
ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
step = ckpt.get('step', '?')
val  = ckpt.get('val_mean', '?')
print(f'  step={step}  val_mean={val}%')

HP_MODEL = dict(V=268, d=64, n_layers=4, n_heads=4, d_ff=256,
                rope=True, yarn=True, null_kv=True, compile=False)
model = build_model(HP_MODEL, device)
model.load_state_dict(ckpt['model'])
model.eval()
print()

# ── Config (from checkpoint hp or args) ───────────────────────────────────────
hp = ckpt.get('hp', {})
CHUNK_LEN  = args.chunk_len
SLOT_LEN   = args.slot_len
SLOT_COUNT = args.slot_count
WARMUP_LEN = args.warmup_len
N_REFINE   = args.n_refine

# n_chunks: from checkpoint curriculum, else from arg, else error
if args.n_chunks is not None:
    N_CHUNKS = args.n_chunks
elif 'curriculum' in hp:
    N_CHUNKS = hp['curriculum'][0]['n_chunks']
else:
    raise ValueError('Cannot detect n_chunks — pass --n-chunks')

SRC_LEN = N_CHUNKS * CHUNK_LEN

# windows: from arg or auto (stride=16B=1 chunk)
if args.windows:
    WINDOWS = [tuple(int(x) for x in w.split(',')) for w in args.windows.split()]
else:
    # default: stride-1 windows covering all n_chunks
    n_windows = (N_CHUNKS - 2) + 1  # (src_len-32)/16 + 1 when chunk_len=16
    WINDOWS = [(i, i+2) for i in range(n_windows)]

print(f'n_chunks={N_CHUNKS}  src_len={SRC_LEN}  chunk_len={CHUNK_LEN}')
print(f'slot_len={SLOT_LEN}  warmup_len={WARMUP_LEN}  n_refine={N_REFINE}')
print(f'windows={WINDOWS}  ({len(WINDOWS)} windows)')
print()

WINDOW_BYTE_RANGES = [(ws * CHUNK_LEN, we * CHUNK_LEN) for ws, we in WINDOWS]
OVERLAPS = [
    (WINDOW_BYTE_RANGES[i+1][0], WINDOW_BYTE_RANGES[i][1])
    for i in range(len(WINDOWS) - 1)
]

# ── Build per-window and full-window pos/mask ─────────────────────────────────
schedule = [(0, N_CHUNKS)]

NOCHAIN = hp.get('mask_nochain', False)

per_window = []
for win in WINDOWS:
    pos  = chunk_positions_fb_localrefine(N_CHUNKS, CHUNK_LEN, SLOT_LEN, WARMUP_LEN,
                                          [win], N_REFINE)
    mask = chunk_mask_fb(pos, nochain=NOCHAIN)
    per_window.append((win, pos, mask))

full_pos  = chunk_positions_fb_localrefine(N_CHUNKS, CHUNK_LEN, SLOT_LEN, WARMUP_LEN,
                                           WINDOWS, N_REFINE)
full_mask = chunk_mask_fb(full_pos, nochain=NOCHAIN)

# ── Sequences ─────────────────────────────────────────────────────────────────
seqs = make_test_sequences(SRC_LEN)
if args.val_n_seqs < len(seqs):
    seqs = dict(list(seqs.items())[:args.val_n_seqs])

# ── Display helpers ───────────────────────────────────────────────────────────
def _print_comparison(ref_bytes, gen_bytes, label='', window_boundaries=None):
    n = len(ref_bytes)
    bounds = set(window_boundaries or [])

    def _row(name, vals, mark_fn=None):
        parts = [f'{name:<6}|']
        for i, v in enumerate(vals):
            sep = ' |' if i in bounds else ''
            s = f'{v:3d}' if mark_fn is None else mark_fn(ref_bytes[i], v)
            parts.append(sep + s)
        print(''.join(parts))

    if label:
        print(f'  [{label}]')
    _row('ref', ref_bytes)
    _row('gen', gen_bytes)
    parts = ['diff  |']
    for i, (r, g) in enumerate(zip(ref_bytes, gen_bytes)):
        sep = ' |' if i in bounds else ''
        parts.append(sep + ('  .' if r == g else '  X'))
    print(''.join(parts))
    n_wrong = sum(r != g for r, g in zip(ref_bytes, gen_bytes))
    print(f'  match {n-n_wrong}/{n} = {100*(n-n_wrong)/n:.1f}%')

# ── Per-window eval ───────────────────────────────────────────────────────────
print('=' * 72)
print(f'PER-WINDOW RECALL (each window evaluated independently, {len(WINDOWS)} windows)')
print('=' * 72)

win_match_table = {}
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
        gen_out   = np.array(r['decoded_bytes'], dtype=np.int64)[:byte_hi-byte_lo-WARMUP_LEN]
        gen_bytes = np.concatenate([ref_bytes[:WARMUP_LEN], gen_out])

        print(f'  {sname:<15} BPB={r["bpb"]:.3f}  match={r["match_pct"]:.1f}%')
        _print_comparison(ref_bytes, gen_bytes, sname)
        print()
    print(f'  MEAN match={np.mean(matches):.1f}%')

# ── Stitched full-sequence eval ───────────────────────────────────────────────
print()
print('=' * 72)
print('STITCHED FULL-SEQUENCE (prolonged AR, only bytes 0-7 seeded from GT)')
print('=' * 72)

stitch_matches = []
for sname, seq in seqs.items():
    chunks_arr = np.array(seq, dtype=np.int64).reshape(N_CHUNKS, CHUNK_LEN)
    r = ar_decode_chunk_fb_stitch_kv(model, chunks_arr, SLOT_LEN, SLOT_COUNT,
                                     full_mask, full_pos, device)
    stitch_matches.append(r['match_pct'])

    ref_full = np.array(seq, dtype=np.int64)
    dec_buf  = np.array(r['decoded_bytes'], dtype=np.int64)
    gen_full = np.where(np.arange(SRC_LEN) < WARMUP_LEN, ref_full, dec_buf)

    print(f'\n  {sname:<15} BPB={r["bpb"]:.3f}  match={r["match_pct"]:.1f}%')

    for win in WINDOWS:
        ws, we = win
        pm = win_match_table[win].get(sname, float('nan'))
        print(f'    window {win} bytes {ws*CHUNK_LEN:3d}-{we*CHUNK_LEN-1:3d}: {pm:.1f}%')

    for (ov_lo, ov_hi) in OVERLAPS:
        ref_ov = ref_full[ov_lo:ov_hi]
        gen_ov = gen_full[ov_lo:ov_hi]
        n_ok = int(np.sum(ref_ov == gen_ov))
        print(f'    overlap {ov_lo:3d}-{ov_hi-1:3d}: {n_ok}/{ov_hi-ov_lo} = {100*n_ok/(ov_hi-ov_lo):.0f}%')

    out_ref = ref_full[WARMUP_LEN:]
    out_gen = gen_full[WARMUP_LEN:]
    win_bounds = sorted(set(
        ws * CHUNK_LEN - WARMUP_LEN for ws, we in WINDOWS if ws * CHUNK_LEN > WARMUP_LEN
    ))
    _print_comparison(out_ref, out_gen,
                      label=f'{sname} output (bytes {WARMUP_LEN}..{SRC_LEN-1})',
                      window_boundaries=win_bounds)
    print()

print(f'STITCH MEAN match={np.mean(stitch_matches):.1f}%')

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print('=' * 72)
print('SUMMARY')
print('=' * 72)
max_w = max(len(str(w)) for w in WINDOWS)
header = f'{"seq":<15}' + ''.join(f'  w{i}' for i in range(len(WINDOWS))) + '  stitch'
print(header)
for i, (sname, seq) in enumerate(seqs.items()):
    wins = ''.join(f'  {win_match_table[w].get(sname,0):5.1f}' for w in WINDOWS)
    stitch = stitch_matches[i]
    print(f'{sname:<15}{wins}  {stitch:5.1f}')
win_means = [np.mean(list(win_match_table[w].values())) for w in WINDOWS]
means_row = ''.join(f'  {m:5.1f}' for m in win_means)
stitch_mean = np.mean(stitch_matches)
print(f'{"MEAN":<15}{means_row}  {stitch_mean:5.1f}')
print()
print(f'WIN_MEAN={np.mean(win_means):.1f}%  STITCH={stitch_mean:.1f}%  '
      f'COMBINED={(np.mean(win_means)+stitch_mean)/2:.1f}%')
