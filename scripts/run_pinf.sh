#!/bin/bash
# Waits for hmn_mono_cumm to finish, then runs hmn_mono_pinf (src_period=-1).
# Usage: bash scripts/run_pinf.sh > scripts/run_pinf.out 2>&1 &
set -e
CKPT="logs/hmn_32_1tok/checkpoints/stage1_end.pt"
EVAL_CFG="configs/hmn_eval_k12.py"

echo "[seq] waiting for hmn_mono_cumm to finish..."
until [ -f logs/hmn_mono_cumm/train.log ]; do sleep 1; done
until grep -q "^Done\." logs/hmn_mono_cumm/train.log 2>/dev/null; do sleep 5; done
echo "[seq] hmn_mono_cumm done."

echo ""
echo "[seq] launching hmn_mono_pinf (src_period=-1, only t=0 sees src)..."
python -m kvmem.train_hmn_mono \
    --config configs/hmn_mono_pinf.py \
    --pretrained "$CKPT" \
    --device mps \
    2>&1 | tee logs/hmn_mono_pinf.log

echo ""
echo "=== hmn_mono_pinf final eval ==="
grep -A8 "stage=0 step=80000" logs/hmn_mono_pinf/train.log | grep "hmn k="

echo ""
echo "[seq] k=0..12 extrapolation on hmn_mono_pinf..."
python -m kvmem.train_hmn_mono \
    --config "$EVAL_CFG" \
    --eval-only logs/hmn_mono_pinf/checkpoints/stage0_end.pt \
    --device mps 2>&1 | tee /tmp/eval_k12_pinf.txt
grep "hmn k=" /tmp/eval_k12_pinf.txt
