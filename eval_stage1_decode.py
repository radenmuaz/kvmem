"""
Quick eval comparing KV decode (fixed) vs non-KV decode on stage 1 checkpoint.
Tests whether the KV cache off-by-one was the root cause of 0% match.
"""
import torch
import numpy as np
from kvmem.model import build_model
from kvmem.train_hmn_chunk import (
    ar_decode_chunk_fb_kv, ar_decode_chunk_fb,
    chunk_positions_fb_localrefine, chunk_mask_fb,
)
from kvmem.utils import make_test_sequences

CKPT = 'logs/hmn_chunk_local_32_stage1/checkpoints/stage0_end.pt'
DEVICE = torch.device('mps')

slot_len   = 4
slot_count = 2
chunk_len  = 16
n_chunks   = 2
warmup_len = 8
windows    = [(0, 2)]
n_refine   = 2

hp = dict(V=268, d=64, n_layers=4, n_heads=4, d_ff=256,
          rope=True, yarn=True, null_kv=True, compile=False)

print(f'Loading {CKPT}')
ckpt  = torch.load(CKPT, map_location=DEVICE)
model = build_model(hp, DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()

seqs = make_test_sequences(chunk_len * n_chunks)  # 32-byte seqs
pos  = chunk_positions_fb_localrefine(n_chunks, chunk_len, slot_len, warmup_len, windows, n_refine)
mask = chunk_mask_fb(pos)
schedule = [(0, n_chunks)]

print(f'n_seqs={len(seqs)}  L={pos["L"]}')
print()

kv_matches, nkv_matches = [], []
for name, seq in list(seqs.items())[:8]:
    chunks = np.array(seq, dtype=np.int64).reshape(n_chunks, chunk_len)

    r_kv  = ar_decode_chunk_fb_kv(model, chunks, slot_len, slot_count, schedule, mask, pos, DEVICE)
    r_nkv = ar_decode_chunk_fb   (model, chunks, slot_len, slot_count, schedule, mask, pos, DEVICE)

    kv_matches.append(r_kv['match_pct'])
    nkv_matches.append(r_nkv['match_pct'])

    print(f'{name:30s}  KV={r_kv["match_pct"]:5.1f}%  BPB_kv={r_kv["bpb"]:.3f}   '
          f'noKV={r_nkv["match_pct"]:5.1f}%  BPB_nkv={r_nkv["bpb"]:.3f}')

print()
print(f'MEAN  KV={np.mean(kv_matches):.1f}%   noKV={np.mean(nkv_matches):.1f}%')
