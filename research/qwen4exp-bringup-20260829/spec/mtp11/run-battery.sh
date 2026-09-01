#!/usr/bin/env bash
# mtp11 deferred-readback battery — runs ON sbox-eval (box), only when the box is IDLE
# (the YaRN ladder owns the GPUs otherwise; poll nvidia-smi + pgrep qwen4exp_real first).
# Ship admission everywhere: dev1 placement, K=5 ceiling, adapt k_lo=1, pmin 0.3.
set -euo pipefail
cd ~/memra
git fetch origin && git pull --ff-only origin qwen4exp-bringup-20260829
echo "== tip: $(git log -1 --oneline)"   # rebuild-after-checkout law: tip in the receipt
export PATH=$HOME/.cargo/bin:$PATH
cargo build --release -p memra-engine --bin qwen4exp_real_gate --bin qwen4exp_gpu_gate
BIN=./target/release/qwen4exp_real_gate
CKPT=$HOME/data/q48fn-nvfp4
OUT=$HOME/realgate/mtp11
mkdir -p "$OUT"
SHAPES=$HOME/realgate/mtp9/shapes
RAW=$HOME/realgate/dump/prompts.tsv
LONG=$HOME/realgate/mtp10/long-prompts.tsv
SHIP="--mtp --mtp-dev1 --spec-pmin 0.3 --spec-adapt 1"

# ---- phase 1: gates at tip -------------------------------------------------
# verify-bit 24/24 (defer does not touch verify rows; tip regression control).
$BIN "$CKPT" "$OUT" --label m11-vbit $SHIP --spec-k 5 --goldens "$HOME/realgate/dump" \
  --verify-bit-gate 24 2>&1 | tee "$OUT/run-vbit.log"
# spec-gate byte identity, DEFER ARM, raw + thinkon + long (4/4, 4/4, 6/6).
$BIN "$CKPT" "$OUT" --label m11-defer-raw $SHIP --spec-k 5 --spec-defer \
  --prompts "$RAW" --spec-gate 256 2>&1 | tee "$OUT/run-gate-raw.log"
$BIN "$CKPT" "$OUT" --label m11-defer-thinkon $SHIP --spec-k 5 --spec-defer \
  --prompts "$SHAPES/thinkon-prompts.tsv" --spec-gate 256 2>&1 | tee "$OUT/run-gate-thinkon.log"
$BIN "$CKPT" "$OUT" --label m11-defer-long $SHIP --spec-k 5 --spec-defer \
  --prompts "$LONG" --spec-gate 256 2>&1 | tee "$OUT/run-gate-long.log"
# gsync twin on thinkon (the sequential-guard sub-arm's own byte identity).
$BIN "$CKPT" "$OUT" --label m11-gsync-thinkon $SHIP --spec-k 5 --spec-defer-guard-sync \
  --prompts "$SHAPES/thinkon-prompts.tsv" --spec-gate 256 2>&1 | tee "$OUT/run-gate-gsync.log"

# ---- phase 2: per-shape A/B at K=5 (host / defer / defer-gsync interleaved, x3 default + receipted escalation) ----
for shape in thinkon thinkoff efflow; do
  $BIN "$CKPT" "$OUT" --label m11-$shape $SHIP --spec-k 5 \
    --prompts "$SHAPES/$shape-prompts.tsv" --defer-ab 3x256 2>&1 | tee "$OUT/run-ab-$shape.log"
done
$BIN "$CKPT" "$OUT" --label m11-raw $SHIP --spec-k 5 \
  --prompts "$RAW" --defer-ab 3x256 2>&1 | tee "$OUT/run-ab-raw.log"
$BIN "$CKPT" "$OUT" --label m11-long $SHIP --spec-k 5 \
  --prompts "$LONG" --defer-ab 3x256 2>&1 | tee "$OUT/run-ab-long.log"

# ---- phase 3: K ladder around the knee, ONE load per shape ------------------
$BIN "$CKPT" "$OUT" --label m11-ladder-thinkon $SHIP --spec-ladder 1,2,3,4,5,6,8 \
  --prompts "$SHAPES/thinkon-prompts.tsv" --defer-ab 3x256 2>&1 | tee "$OUT/run-ladder-thinkon.log"
$BIN "$CKPT" "$OUT" --label m11-ladder-raw $SHIP --spec-ladder 1,2,3,4,5,6,8 \
  --prompts "$RAW" --defer-ab 3x256 2>&1 | tee "$OUT/run-ladder-raw.log"

# ---- phase 4: sampled probe with the defer arm (serving-law shape) -----------
$BIN "$CKPT" "$OUT" --label m11-sampled-thinkon $SHIP --spec-k 5 --spec-defer \
  --spec-sampled --spec-ab 2x256 --goldens "$HOME/realgate/dump" --prompts "$SHAPES/thinkon-prompts.tsv" \
  2>&1 | tee "$OUT/run-sampled.log"

echo "== mtp11 battery complete: receipts in $OUT"
