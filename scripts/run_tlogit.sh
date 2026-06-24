#!/bin/bash
# Chains after hmn_mono_p4: runs tlogit_fixed then tlogit (scheduled alpha).
# Usage: bash scripts/run_tlogit.sh > scripts/run_tlogit.out 2>&1 &
set -e
CKPT="logs/hmn_32_1tok/checkpoints/stage1_end.pt"
EVAL_CFG="configs/hmn_eval_k12.py"

# ── Wait for p4 ──────────────────────────────────────────────────────────────
echo "[seq] waiting for logs/hmn_mono_p4/train.log..."
until [ -f logs/hmn_mono_p4/train.log ]; do sleep 1; done
until grep -q "^Done\." logs/hmn_mono_p4/train.log 2>/dev/null; do sleep 5; done
echo "[seq] done."

# ── Ablation 1: fixed alpha=0.5 ──────────────────────────────────────────────
echo ""
echo "[seq] launching hmn_mono_tlogit_fixed (fixed alpha=0.5, k=1..4)..."
python -m kvmem.train_hmn_mono \
    --config configs/hmn_mono_tlogit_fixed.py \
    --pretrained "$CKPT" \
    --device mps \
    2>&1 | tee logs/hmn_mono_tlogit_fixed.log

echo ""
echo "=== tlogit_fixed final eval ==="
grep -A8 "stage=0 step=80000" logs/hmn_mono_tlogit_fixed/train.log | grep "hmn k="

echo ""
echo "[seq] k=0..12 extrapolation on tlogit_fixed..."
python -m kvmem.train_hmn_mono \
    --config "$EVAL_CFG" \
    --eval-only logs/hmn_mono_tlogit_fixed/checkpoints/stage0_end.pt \
    --device mps 2>&1 | tee /tmp/eval_k12_tlogit_fixed.txt
grep "hmn k=" /tmp/eval_k12_tlogit_fixed.txt

# ── Ablation 2: scheduled alpha, k_max=12 ───────────────────────────────────
echo ""
echo "[seq] launching hmn_mono_tlogit (scheduled alpha 0.1→1.0 at k=12, train k=1..4)..."
python -m kvmem.train_hmn_mono \
    --config configs/hmn_mono_tlogit.py \
    --pretrained "$CKPT" \
    --device mps \
    2>&1 | tee logs/hmn_mono_tlogit.log

echo ""
echo "=== tlogit (scheduled) final eval ==="
grep -A8 "stage=0 step=80000" logs/hmn_mono_tlogit/train.log | grep "hmn k="

echo ""
echo "[seq] k=0..12 extrapolation on tlogit (scheduled)..."
python -m kvmem.train_hmn_mono \
    --config "$EVAL_CFG" \
    --eval-only logs/hmn_mono_tlogit/checkpoints/stage0_end.pt \
    --device mps 2>&1 | tee /tmp/eval_k12_tlogit.txt
grep "hmn k=" /tmp/eval_k12_tlogit.txt

# ── Ablation: cum_mean only ──────────────────────────────────────────────────
echo ""
echo "[seq] launching hmn_mono_cumm (cum_w=1, mono_w=0)..."
python -m kvmem.train_hmn_mono \
    --config configs/hmn_mono_cumm.py \
    --pretrained "$CKPT" \
    --device mps \
    2>&1 | tee logs/hmn_mono_cumm.log

echo ""
echo "=== hmn_mono_cumm final eval ==="
grep -A8 "stage=0 step=80000" logs/hmn_mono_cumm/train.log | grep "hmn k="

echo ""
echo "[seq] k=0..12 extrapolation on hmn_mono_cumm..."
python -m kvmem.train_hmn_mono \
    --config "$EVAL_CFG" \
    --eval-only logs/hmn_mono_cumm/checkpoints/stage0_end.pt \
    --device mps 2>&1 | tee /tmp/eval_k12_cumm.txt
grep "hmn k=" /tmp/eval_k12_cumm.txt

# ── Final comparison table ───────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "EXTRAPOLATION TABLE: match% at k=0..12"
echo "======================================================================"
printf "%-4s  %-14s  %-14s  %-14s  %-14s  %-14s  %-14s\n" \
    "k" "mono(p=1)" "p2" "p4" "tlogit_fixed" "tlogit_sched" "cumm"
printf "%-4s  %-14s  %-14s  %-14s  %-14s  %-14s  %-14s\n" \
    "----" "--------------" "--------------" "--------------" "--------------" "--------------" "--------------"
for k in 0 1 2 3 4 6 8 10 12; do
    v1=$(grep "hmn k=$k " /tmp/eval_k12_hmn_mono.txt     2>/dev/null | grep -oE '[0-9]+\.[0-9]+%' | head -1 || echo "?")
    v2=$(grep "hmn k=$k " /tmp/eval_k12_p2.txt            2>/dev/null | grep -oE '[0-9]+\.[0-9]+%' | head -1 || echo "?")
    v4=$(grep "hmn k=$k " /tmp/eval_k12_p4.txt            2>/dev/null | grep -oE '[0-9]+\.[0-9]+%' | head -1 || echo "?")
    vf=$(grep "hmn k=$k " /tmp/eval_k12_tlogit_fixed.txt  2>/dev/null | grep -oE '[0-9]+\.[0-9]+%' | head -1 || echo "?")
    vs=$(grep "hmn k=$k " /tmp/eval_k12_tlogit.txt        2>/dev/null | grep -oE '[0-9]+\.[0-9]+%' | head -1 || echo "?")
    vc=$(grep "hmn k=$k " /tmp/eval_k12_cumm.txt          2>/dev/null | grep -oE '[0-9]+\.[0-9]+%' | head -1 || echo "?")
    printf "%-4s  %-14s  %-14s  %-14s  %-14s  %-14s  %-14s\n" "$k" "$v1" "$v2" "$v4" "$vf" "$vs" "$vc"
done
