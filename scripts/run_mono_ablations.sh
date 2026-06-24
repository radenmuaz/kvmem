#!/bin/bash
# Runs 3 monotonic ablations in sequence from the same pretrained checkpoint.
# Ablation 1 (hmn_mono) is already running — waits for it, then runs 2 and 3.
#
# Usage: bash scripts/run_mono_ablations.sh > scripts/run_mono_ablations.out 2>&1 &
set -e
CKPT="logs/hmn_32_1tok/checkpoints/stage1_end.pt"

# ── Wait for ablation 1 (hmn_mono, flat_mono only) ──────────────────────────
echo "[seq] waiting for logs/hmn_mono/train.log..."
until [ -f logs/hmn_mono/train.log ]; do sleep 1; done
until grep -q "^Done\." logs/hmn_mono/train.log 2>/dev/null; do sleep 5; done
echo "[seq] done."

echo ""
echo "=== Ablation 1: hmn_mono (flat_mono only, mono_w=1 cum_w=0 cer_b_w=0) ==="
grep -A8 "stage=0 step=80000" logs/hmn_mono/train.log | grep -E "match|val_hmn"

# ── Ablation 2: CER-delta weighted NLL (variant B) ──────────────────────────
echo ""
echo "[seq] launching hmn_mono_cerb (cer_b_w=1, mono_w=0, cum_w=0) ..."
python -m kvmem.train_hmn_mono \
    --config configs/hmn_mono_cerb.py \
    --pretrained "$CKPT" \
    --device mps \
    2>&1 | tee logs/hmn_mono_cerb.log

echo ""
echo "=== Ablation 2: hmn_mono_cerb (delta-CER weighted NLL) ==="
grep -A8 "stage=0 step=80000" logs/hmn_mono_cerb/train.log | grep -E "match|val_hmn"

# ── Ablation 3: cum_mean only ────────────────────────────────────────────────
echo ""
echo "[seq] launching hmn_mono_cumm (cum_w=1, mono_w=0, cer_b_w=0) ..."
python -m kvmem.train_hmn_mono \
    --config configs/hmn_mono_cumm.py \
    --pretrained "$CKPT" \
    --device mps \
    2>&1 | tee logs/hmn_mono_cumm.log

echo ""
echo "=== Ablation 3: hmn_mono_cumm (cum_mean only) ==="
grep -A8 "stage=0 step=80000" logs/hmn_mono_cumm/train.log | grep -E "match|val_hmn"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== FINAL COMPARISON ==="
echo "--- hmn_32 stage3 baseline (structured tokens, teacher h-loss) ---"
grep -A8 "stage=3 step=40000" logs/hmn_32/train.log 2>/dev/null | grep "match" || echo "  (not found)"
echo "--- hmn_mono       (flat_mono,     mono_w=1 cum_w=0 cer_b_w=0) ---"
grep -A8 "stage=0 step=80000" logs/hmn_mono/train.log | grep "match"
echo "--- hmn_mono_cerb  (delta-CER B,   mono_w=0 cum_w=0 cer_b_w=1) ---"
grep -A8 "stage=0 step=80000" logs/hmn_mono_cerb/train.log | grep "match"
echo "--- hmn_mono_cumm  (cum_mean,      mono_w=0 cum_w=1 cer_b_w=0) ---"
grep -A8 "stage=0 step=80000" logs/hmn_mono_cumm/train.log | grep "match"
