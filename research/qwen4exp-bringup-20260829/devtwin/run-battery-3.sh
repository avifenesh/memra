#!/usr/bin/env bash
# devtwin battery v2 (consolidated, post the route-kernel v2 fix) — runs ON sbox-eval.
# v1 postmortem: the v1 route kernel ran every phase on thread 0 over GLOBAL memory
# (~39 us/launch); the plain-decode A/B caught it at +12.5% (ab-routerdev-dt-plain.tsv)
# and the battery was stopped rather than spend GPU-hours measuring a known-slow
# kernel. v2 parallelizes the order-FREE phases (max fold is fmaxf-associative — any
# order is bit-exact; exp; divisions) and keeps the order-SENSITIVE sum + top-k
# sequential over SHARED memory. Correctness receipts re-run here at the v2 tip.
# Ship admission everywhere. Audit ON for correctness phases only.
set -euo pipefail
cd ~/memra
git fetch origin && git pull --ff-only origin qwen4exp-bringup-20260829
echo "== tip: $(git log -1 --oneline)"
export PATH=$HOME/.cargo/bin:$PATH
cargo build --release -p memra-engine --bin qwen4exp_real_gate --bin qwen4exp_gpu_gate
BIN=./target/release/qwen4exp_real_gate
CKPT=$HOME/data/q48fn-nvfp4
OUT=$HOME/realgate/devtwin
mkdir -p "$OUT"
SHAPES=$HOME/realgate/mtp9/shapes
RAW=$HOME/realgate/dump/prompts.tsv
LONG=$HOME/realgate/mtp10/long-prompts.tsv
SHIP="--mtp --mtp-dev1 --spec-pmin 0.3 --spec-adapt 1"
BOTH="MEMRA_Q4E_SEAMS=routerdev,idxcache"
BOTHA="MEMRA_Q4E_SEAMS=routerdev,idxcache MEMRA_Q4E_ROUTER_AUDIT=1"

# ---- phase 0: tiny gate, both seams + audit, box toolchain ------------------------
env $BOTHA ./target/release/qwen4exp_gpu_gate "$OUT/tiny-gate-devtwin-v2-box.tsv" \
  2>&1 | tee "$OUT/run3-tiny.log"

# ---- phase 1: rule gates, BOTH seams + audit (per-commit law, at the v2 tip) -------
env $BOTHA $BIN "$CKPT" "$OUT" --label dt3-vbit $SHIP --spec-k 5 \
  --goldens "$HOME/realgate/dump" --verify-bit-gate 24 2>&1 | tee "$OUT/run3-vbit.log"
env $BOTHA $BIN "$CKPT" "$OUT" --label dt3-raw $SHIP --spec-k 5 \
  --prompts "$RAW" --spec-gate 256 2>&1 | tee "$OUT/run3-gate-raw.log"
env $BOTHA $BIN "$CKPT" "$OUT" --label dt3-thinkon $SHIP --spec-k 5 \
  --prompts "$SHAPES/thinkon-prompts.tsv" --spec-gate 256 2>&1 | tee "$OUT/run3-gate-thinkon.log"
env $BOTHA $BIN "$CKPT" "$OUT" --label dt3-long $SHIP --spec-k 5 \
  --prompts "$LONG" --spec-gate 256 2>&1 | tee "$OUT/run3-gate-long.log"
env MEMRA_Q4E_ROUTER_AUDIT=1 $BIN "$CKPT" "$OUT" --label dt3-seam \
  --goldens "$HOME/realgate/dump" --ab-seam routerdev --seam-gate 24 \
  2>&1 | tee "$OUT/run3-seam.log"

# ---- phase 2: per-change isolation A/Bs --------------------------------------------
# routerdev alone: plain decode + spec thinkon/raw.
$BIN "$CKPT" "$OUT" --label dt3-plain-router --goldens "$HOME/realgate/dump" \
  --ab-seam routerdev --ab-moe 5x128 2>&1 | tee "$OUT/run3-ab-plain-router.log"
$BIN "$CKPT" "$OUT" --label dt3-router-thinkon $SHIP --spec-k 5 \
  --prompts "$SHAPES/thinkon-prompts.tsv" --router-ab 3x256 2>&1 | tee "$OUT/run3-ab-router-thinkon.log"
$BIN "$CKPT" "$OUT" --label dt3-router-raw $SHIP --spec-k 5 \
  --prompts "$RAW" --router-ab 3x256 2>&1 | tee "$OUT/run3-ab-router-raw.log"
# idxcache alone: plain decode + spec thinkon/raw.
$BIN "$CKPT" "$OUT" --label dt3-plain-idxcache --goldens "$HOME/realgate/dump" \
  --ab-seam idxcache --ab-moe 5x128 2>&1 | tee "$OUT/run3-ab-plain-idxcache.log"
$BIN "$CKPT" "$OUT" --label dt3-idxcache-thinkon $SHIP --spec-k 5 --ab-seam idxcache \
  --prompts "$SHAPES/thinkon-prompts.tsv" --router-ab 3x256 2>&1 | tee "$OUT/run3-ab-idxcache-thinkon.log"
$BIN "$CKPT" "$OUT" --label dt3-idxcache-raw $SHIP --spec-k 5 --ab-seam idxcache \
  --prompts "$RAW" --router-ab 3x256 2>&1 | tee "$OUT/run3-ab-idxcache-raw.log"

# ---- phase 3: the COMBINED stack per shape at ship K=5 -----------------------------
for shape in thinkon thinkoff efflow; do
  $BIN "$CKPT" "$OUT" --label dt3-$shape $SHIP --spec-k 5 --ab-seam devtwin \
    --prompts "$SHAPES/$shape-prompts.tsv" --router-ab 3x256 2>&1 | tee "$OUT/run3-ab-devtwin-$shape.log"
done
$BIN "$CKPT" "$OUT" --label dt3-raw $SHIP --spec-k 5 --ab-seam devtwin \
  --prompts "$RAW" --router-ab 3x256 2>&1 | tee "$OUT/run3-ab-devtwin-raw.log"
$BIN "$CKPT" "$OUT" --label dt3-long $SHIP --spec-k 5 --ab-seam devtwin \
  --prompts "$LONG" --router-ab 3x256 2>&1 | tee "$OUT/run3-ab-devtwin-long.log"

# ---- phase 4: K ladders, combined stack --------------------------------------------
$BIN "$CKPT" "$OUT" --label dt3-ladder-thinkon $SHIP --spec-ladder 1,2,3,4,5,6,8 \
  --ab-seam devtwin --prompts "$SHAPES/thinkon-prompts.tsv" --router-ab 3x256 \
  2>&1 | tee "$OUT/run3-ladder-thinkon.log"
$BIN "$CKPT" "$OUT" --label dt3-ladder-raw $SHIP --spec-ladder 1,2,3,4,5,6,8 \
  --ab-seam devtwin --prompts "$RAW" --router-ab 3x256 2>&1 | tee "$OUT/run3-ladder-raw.log"

# ---- phase 5: defer re-measure (PROFILE-8 §4's unlock: the draft step's router and
# indexer dtoh are device-side now; mtp11's banked ladder is the baseline) -----------
env $BOTH $BIN "$CKPT" "$OUT" --label dt3-defer-thinkon $SHIP --spec-k 5 \
  --prompts "$SHAPES/thinkon-prompts.tsv" --defer-ab 3x256 2>&1 | tee "$OUT/run3-defer-thinkon.log"
env $BOTH $BIN "$CKPT" "$OUT" --label dt3-defer-ladder $SHIP \
  --spec-ladder 1,2,3,5 --prompts "$SHAPES/thinkon-prompts.tsv" --defer-ab 3x256 \
  2>&1 | tee "$OUT/run3-defer-ladder.log"

# ---- phase 6: sampled probe, both seams (serving law) ------------------------------
env $BOTH $BIN "$CKPT" "$OUT" --label dt3-sampled $SHIP --spec-k 5 \
  --spec-sampled --spec-ab 2x256 --goldens "$HOME/realgate/dump" \
  --prompts "$SHAPES/thinkon-prompts.tsv" 2>&1 | tee "$OUT/run3-sampled.log"

echo "== devtwin battery v2 complete: receipts in $OUT"
