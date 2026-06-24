#!/bin/bash
# Wait for hmn_32_1tok to finish, then run hmn_mono.
# Run: bash scripts/run_after_1tok.sh > scripts/run_after_1tok.out 2>&1 &
LOG1TOK="logs/hmn_32_1tok/train.log"
CKPT="logs/hmn_32_1tok/checkpoints/stage1_end.pt"

echo "[watcher] waiting for hmn_32_1tok to finish..."
until [ -f "$LOG1TOK" ]; do sleep 1; done
until grep -q "^Done\." "$LOG1TOK" 2>/dev/null; do sleep 5; done
echo "[watcher] hmn_32_1tok done."

echo ""
echo "=== hmn_32_1tok stage 2 (IR, trajectory teacher) final ==="
grep -A6 "stage=2 step=80000" "$LOG1TOK" | grep -E "val_hmn|match"

echo ""
echo "[watcher] launching hmn_mono from $CKPT ..."
python -m kvmem.train_hmn_mono \
    --config configs/hmn_mono.py \
    --pretrained "$CKPT" \
    --device mps \
    2>&1 | tee logs/hmn_mono.log

echo ""
echo "=== COMPARISON (stage 3 equiv: IR refine only) ==="
echo "--- hmn_32 (structured, trajectory teacher) ---"
grep -A6 "stage=3 step=40000" logs/hmn_32/train.log | grep "match"
echo "--- hmn_32_1tok (1tok, trajectory teacher) ---"
grep -A6 "stage=2 step=80000" "$LOG1TOK" | grep "match"
echo "--- hmn_mono (1tok, monotonic NLL only) ---"
grep -A6 "step=80000" logs/hmn_mono/train.log 2>/dev/null | grep "match"
