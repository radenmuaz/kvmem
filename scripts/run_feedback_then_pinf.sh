#!/bin/bash
# Queue: cumm (running) → eval cumm → feedback → eval feedback → pinf → eval pinf
# Usage: bash scripts/run_feedback_then_pinf.sh > scripts/run_feedback_then_pinf.out 2>&1 &
set -e
CKPT="logs/hmn_32_1tok/checkpoints/stage1_end.pt"
EVAL_CFG="configs/hmn_eval_k12.py"

# ── Wait for cumm ────────────────────────────────────────────────────────────
echo "[seq] waiting for hmn_mono_cumm..."
until grep -q "^Done\." logs/hmn_mono_cumm/train.log 2>/dev/null; do sleep 5; done
echo "[seq] hmn_mono_cumm done."

echo ""
echo "=== cumm final eval ==="
grep -A8 "stage=0 step=80000" logs/hmn_mono_cumm/train.log | grep "hmn k="

echo ""
echo "[seq] k=0..12 eval on hmn_mono_cumm..."
python -m kvmem.train_hmn_mono \
    --config "$EVAL_CFG" \
    --eval-only logs/hmn_mono_cumm/checkpoints/stage0_end.pt \
    --device mps 2>&1 | tee /tmp/eval_k12_cumm.txt
grep "hmn k=" /tmp/eval_k12_cumm.txt

# ── Feedback IQ base ─────────────────────────────────────────────────────────
echo ""
echo "[seq] waiting for hmn_feedback_32_iq (IQ pretraining) to finish..."
until grep -q "^Done\." logs/hmn_feedback_32_iq/train.log 2>/dev/null; do sleep 5; done
echo "[seq] hmn_feedback_32_iq done."

echo ""
echo "=== feedback IQ stage final eval ==="
grep -A4 "stage=0 step=50000" logs/hmn_feedback_32_iq/train.log | grep "fb k="

# ── Feedback IR ──────────────────────────────────────────────────────────────
echo ""
echo "[seq] launching hmn_feedback_32_ir (pretrained from IQ ckpt)..."
python -m kvmem.train_hmn_feedback \
    --config configs/hmn_feedback_32_ir.py \
    --pretrained logs/hmn_feedback_32_iq/checkpoints/stage0_end.pt \
    --device mps \
    2>&1 | tee logs/hmn_feedback_32_ir.log

echo ""
echo "=== feedback IR final eval ==="
grep -A8 "stage=0 step=80000" logs/hmn_feedback_32_ir/train.log | grep "fb k="

# ── Feedback IR cum_mean ──────────────────────────────────────────────────────
echo ""
echo "[seq] launching hmn_feedback_32_ir_cumm (cum_mean loss, pretrained from IQ ckpt)..."
python -m kvmem.train_hmn_feedback \
    --config configs/hmn_feedback_32_ir_cumm.py \
    --pretrained logs/hmn_feedback_32_iq/checkpoints/stage0_end.pt \
    --device mps \
    2>&1 | tee logs/hmn_feedback_32_ir_cumm.log

echo ""
echo "=== feedback IR cumm final eval ==="
grep -A8 "stage=0 step=80000" logs/hmn_feedback_32_ir_cumm/train.log | grep "fb k="

# ── pinf ─────────────────────────────────────────────────────────────────────
echo ""
echo "[seq] launching hmn_mono_pinf (src_period=-1)..."
python -m kvmem.train_hmn_mono \
    --config configs/hmn_mono_pinf.py \
    --pretrained "$CKPT" \
    --device mps \
    2>&1 | tee logs/hmn_mono_pinf.log

echo ""
echo "=== pinf final eval ==="
grep -A8 "stage=0 step=80000" logs/hmn_mono_pinf/train.log | grep "hmn k="

echo ""
echo "[seq] k=0..12 eval on hmn_mono_pinf (both modes)..."
python -m kvmem.train_hmn_mono \
    --config "$EVAL_CFG" \
    --eval-only logs/hmn_mono_pinf/checkpoints/stage0_end.pt \
    --device mps 2>&1 | tee /tmp/eval_k12_pinf.txt
grep "hmn k=" /tmp/eval_k12_pinf.txt

echo ""
echo "======================================================================"
echo "FINAL TABLE"
echo "======================================================================"
printf "%-4s  %-12s  %-12s  %-12s  %-12s  %-12s  %-12s\n" \
    "k" "mono(p=1)" "cumm" "tlogit_fx" "feedback" "pinf(p=-1)" "p2"
printf "%-4s  %-12s  %-12s  %-12s  %-12s  %-12s  %-12s\n" \
    "----" "------------" "------------" "------------" "------------" "------------" "------------"
for k in 0 1 2 3 4 6 8 10 12; do
    v1=$(grep "hmn k=$k " /tmp/eval_k12_hmn_mono.txt     2>/dev/null | grep -oE '[0-9]+\.[0-9]+%' | head -1 || echo "?")
    vc=$(grep "hmn k=$k " /tmp/eval_k12_cumm.txt          2>/dev/null | grep -oE '[0-9]+\.[0-9]+%' | head -1 || echo "?")
    vf=$(grep "hmn k=$k " /tmp/eval_k12_tlogit_fixed.txt  2>/dev/null | grep -oE '[0-9]+\.[0-9]+%' | head -1 || echo "?")
    vb=$(grep "fb k=$k \|hmn k=$k " /tmp/eval_k12_feedback.txt 2>/dev/null | grep -oE '[0-9]+\.[0-9]+%' | head -1 || echo "?")
    vp=$(grep "hmn k=$k " /tmp/eval_k12_pinf.txt          2>/dev/null | grep -oE '[0-9]+\.[0-9]+%' | head -1 || echo "?")
    v2=$(grep "hmn k=$k " /tmp/eval_k12_p2.txt            2>/dev/null | grep -oE '[0-9]+\.[0-9]+%' | head -1 || echo "?")
    printf "%-4s  %-12s  %-12s  %-12s  %-12s  %-12s  %-12s\n" "$k" "$v1" "$vc" "$vf" "$vb" "$vp" "$v2"
done
