#!/bin/bash
# Waits for hmn_32_1tok to finish, then runs hmn_kvt, then prints comparison.
set -e
LOG1TOK="logs/hmn_32_1tok/train.log"
CKPT="logs/hmn_32_1tok/checkpoints/stage1_end.pt"

echo "[watcher] waiting for hmn_32_1tok to finish..."
until [ -f "$LOG1TOK" ]; do sleep 1; done
until grep -q "^Done\." "$LOG1TOK" 2>/dev/null; do sleep 5; done
echo "[watcher] hmn_32_1tok done."

echo ""
echo "=== hmn_32_1tok stage 2 (IR) final results ==="
grep -A8 "stage=2 step=80000" "$LOG1TOK" | grep -E "val_hmn|match"

echo ""
echo "[watcher] launching hmn_kvt from $CKPT ..."
python -m kvmem.train_hmn_kvt \
    --config configs/hmn_kvt.py \
    --pretrained "$CKPT" \
    --device mps \
    2>&1 | tee logs/hmn_kvt.log

echo ""
echo "=== COMPARISON ==="
echo "--- hmn_32 stage 3 (structured tokens, trajectory teacher) ---"
grep -E "stage=3 step=40000" logs/hmn_32/train.log | head -1
grep -A6 "stage=3 step=40000" logs/hmn_32/train.log | grep "match"

echo ""
echo "--- hmn_32_1tok stage 2 (1tok, trajectory teacher) ---"
grep -E "stage=2 step=80000" "$LOG1TOK" | head -1
grep -A6 "stage=2 step=80000" "$LOG1TOK" | grep "match"

echo ""
echo "--- hmn_kvt (1tok base, shared KVT target, AdamW 4 steps) ---"
grep -E "step=80000" logs/hmn_kvt/train.log 2>/dev/null | head -1
grep -A6 "step=80000" logs/hmn_kvt/train.log 2>/dev/null | grep "match"
