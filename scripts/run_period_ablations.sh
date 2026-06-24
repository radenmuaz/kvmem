#!/bin/bash
# Chains: wait for hmn_mono_p2 → eval k=0..12 → run p4 → eval k=0..12 → comparison table.
# Usage: bash scripts/run_period_ablations.sh > scripts/run_period_ablations.out 2>&1 &
set -e
CKPT="logs/hmn_32_1tok/checkpoints/stage1_end.pt"
EVAL_CFG="configs/hmn_eval_k12.py"

# ── Wait for p2 ──────────────────────────────────────────────────────────────
echo "[seq] waiting for logs/hmn_mono_p2/train.log..."
until [ -f logs/hmn_mono_p2/train.log ]; do sleep 1; done
until grep -q "^Done\." logs/hmn_mono_p2/train.log 2>/dev/null; do sleep 5; done
echo "[seq] done."

echo ""
echo "=== hmn_mono_p2 final eval (step=80000) ==="
grep -A8 "stage=0 step=80000" logs/hmn_mono_p2/train.log | grep -E "hmn k=|match"

# ── Eval p2 k=0..12 ──────────────────────────────────────────────────────────
echo ""
echo "[seq] running k=0..12 eval on hmn_mono_p2..."
python -m kvmem.train_hmn_mono \
    --config "$EVAL_CFG" \
    --eval-only logs/hmn_mono_p2/checkpoints/stage0_end.pt \
    --device mps \
    2>&1 | tee /tmp/eval_k12_p2.txt
echo ""
echo "=== p2 extrapolation (k=0..12) ==="
grep -E "hmn k=" /tmp/eval_k12_p2.txt

# ── Run p4 ───────────────────────────────────────────────────────────────────
echo ""
echo "[seq] launching hmn_mono_p4 (src_period=4)..."
python -m kvmem.train_hmn_mono \
    --config configs/hmn_mono_p4.py \
    --pretrained "$CKPT" \
    --device mps \
    2>&1 | tee logs/hmn_mono_p4.log

echo ""
echo "=== hmn_mono_p4 final eval (step=80000) ==="
grep -A8 "stage=0 step=80000" logs/hmn_mono_p4/train.log | grep -E "hmn k=|match"

# ── Eval p4 k=0..12 ──────────────────────────────────────────────────────────
echo ""
echo "[seq] running k=0..12 eval on hmn_mono_p4..."
python -m kvmem.train_hmn_mono \
    --config "$EVAL_CFG" \
    --eval-only logs/hmn_mono_p4/checkpoints/stage0_end.pt \
    --device mps \
    2>&1 | tee /tmp/eval_k12_p4.txt

# ── Comparison tables ────────────────────────────────────────────────────────
# Extract per-mode match% from eval files.
# Files contain two sections: in-distribution (first) and bottleneck (second, p!=1 only).
# Use awk to split on the section header and grab per-section lines.
_extract() { local f=$1 k=$2 mode=$3
    awk "/src_period=${mode}/{found=1} found && /hmn k=${k} /{print; exit}" "$f" 2>/dev/null \
    | grep -oE '[0-9]+\.[0-9]+%' | head -1 || echo "?"
}

echo ""
echo "=== In-distribution (eval src_period matches training) ==="
printf "%-4s  %-14s  %-14s  %-14s\n" "k" "mono(p=1)" "p2" "p4"
printf "%-4s  %-14s  %-14s  %-14s\n" "----" "--------------" "--------------" "--------------"
for k in 0 1 2 3 4 6 8 10 12; do
    v1=$(grep "hmn k=$k " /tmp/eval_k12_hmn_mono.txt 2>/dev/null | grep -oE '[0-9]+\.[0-9]+%' | head -1 || echo "?")
    v2=$(_extract /tmp/eval_k12_p2.txt $k 2)
    v4=$(_extract /tmp/eval_k12_p4.txt $k 4)
    printf "%-4s  %-14s  %-14s  %-14s\n" "$k" "$v1" "$v2" "$v4"
done

echo ""
echo "=== Bottleneck (src_period=∞: only t=0 sees src) ==="
printf "%-4s  %-14s  %-14s\n" "k" "p2" "p4"
printf "%-4s  %-14s  %-14s\n" "----" "--------------" "--------------"
for k in 0 1 2 3 4 6 8 10 12; do
    v2=$(_extract /tmp/eval_k12_p2.txt $k "∞")
    v4=$(_extract /tmp/eval_k12_p4.txt $k "∞")
    printf "%-4s  %-14s  %-14s\n" "$k" "$v2" "$v4"
done
