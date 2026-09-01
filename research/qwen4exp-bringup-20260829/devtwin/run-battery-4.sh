#!/usr/bin/env bash
# devtwin battery v4 — the SPEC-loop and combined-stack rows, at the v3 route kernel.
# Entry state (box receipts already banked, all at this tip):
#   law gates green with routerdev+idxcache armed (vbit 24/24, spec-gate raw/thinkon/long
#   byte identity, seam-gate 24/24 argmax KL 0.00000, audit 187k+ rows zero selection
#   mismatches), and these plain-decode A/Bs:
#     routerdev, graphs ON : 14.91 host -> 16.46 dev  (LOSES)
#     routerdev, graphs OFF: 15.13 host -> 13.97 dev  (WINS 1.083x; beats host+graphs)
#     routerdev + ROUTE_SYNC diag, graphs ON: 14.91 -> 14.26 (WINS 1.046x)
#     idxcache alone, graphs ON: 14.90 -> 14.55 (WINS 1.024x)
#   => the seam's cost with graphs ON is the MISSING per-layer sync (unsynchronized
#   graph-replay storm), not the kernel; the best measured pairing is device router with
#   decode graphs OFF. This battery measures that pairing where the product lives (the
#   spec loop, where t==1 graphs are already off by construction under an armed verify).
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

# ---- phase 0: rule gates at the FINAL tip, both seams + audit ----------------------
# DEVTWIN_SKIP_GATES=1 skips this phase when the gates are ALREADY green at the running
# tip (a resume after a harness fix that changed no engine arithmetic) — the receipt for
# the skipped phase must exist and be cited, never assumed.
if [ "${DEVTWIN_SKIP_GATES:-0}" != "1" ]; then
env $BOTH MEMRA_Q4E_ROUTER_AUDIT=1 ./target/release/qwen4exp_gpu_gate \
  "$OUT/tiny-gate-devtwin-final-box.tsv" 2>&1 | tee "$OUT/run4-tiny.log"
env $BOTH MEMRA_Q4E_ROUTER_AUDIT=1 $BIN "$CKPT" "$OUT" --label dt4-vbit $SHIP --spec-k 5 \
  --goldens "$HOME/realgate/dump" --verify-bit-gate 24 2>&1 | tee "$OUT/run4-vbit.log"
env $BOTH MEMRA_Q4E_ROUTER_AUDIT=1 $BIN "$CKPT" "$OUT" --label dt4-raw $SHIP --spec-k 5 \
  --prompts "$RAW" --spec-gate 256 2>&1 | tee "$OUT/run4-gate-raw.log"
env $BOTH MEMRA_Q4E_ROUTER_AUDIT=1 $BIN "$CKPT" "$OUT" --label dt4-thinkon $SHIP --spec-k 5 \
  --prompts "$SHAPES/thinkon-prompts.tsv" --spec-gate 256 2>&1 | tee "$OUT/run4-gate-thinkon.log"
else
  echo "== phase 0 SKIPPED (DEVTWIN_SKIP_GATES=1); cited receipts: tiny-gate-devtwin-final-box.tsv, verify-bit-gate-dt4-vbit.tsv, spec-gate-k5-dt4-{raw,thinkon}.tsv"
fi

# ---- phase 1: plain decode, COMBINED stack, both graph pairings --------------------
$BIN "$CKPT" "$OUT" --label dt4-plain-graphon --goldens "$HOME/realgate/dump" \
  --ab-seam devtwin --ab-moe 5x128 2>&1 | tee "$OUT/run4-ab-plain-graphon.log"
env MEMRA_Q4E_SEAMS=graph=0 $BIN "$CKPT" "$OUT" --label dt4-plain-nograph \
  --goldens "$HOME/realgate/dump" --ab-seam devtwin --ab-moe 5x128 \
  2>&1 | tee "$OUT/run4-ab-plain-nograph.log"

# ---- phase 2: SPEC per shape at ship K=5, combined stack (the product shape) -------
for shape in thinkon thinkoff efflow; do
  $BIN "$CKPT" "$OUT" --label dt4-$shape $SHIP --spec-k 5 --ab-seam devtwin \
    --prompts "$SHAPES/$shape-prompts.tsv" --router-ab 3x256 2>&1 | tee "$OUT/run4-ab-$shape.log"
done
$BIN "$CKPT" "$OUT" --label dt4-raw $SHIP --spec-k 5 --ab-seam devtwin \
  --prompts "$RAW" --router-ab 3x256 2>&1 | tee "$OUT/run4-ab-raw.log"
$BIN "$CKPT" "$OUT" --label dt4-long $SHIP --spec-k 5 --ab-seam devtwin \
  --prompts "$LONG" --router-ab 3x256 2>&1 | tee "$OUT/run4-ab-long.log"

# ---- phase 3: K ladders, combined stack -------------------------------------------
$BIN "$CKPT" "$OUT" --label dt4-ladder-thinkon $SHIP --spec-ladder 1,2,3,5,8 \
  --ab-seam devtwin --prompts "$SHAPES/thinkon-prompts.tsv" --router-ab 3x256 \
  2>&1 | tee "$OUT/run4-ladder-thinkon.log"
$BIN "$CKPT" "$OUT" --label dt4-ladder-raw $SHIP --spec-ladder 1,2,3,5,8 \
  --ab-seam devtwin --prompts "$RAW" --router-ab 3x256 2>&1 | tee "$OUT/run4-ladder-raw.log"

# ---- phase 4: defer re-measure (PROFILE-8 §4's unlock condition now satisfied) ------
env $BOTH $BIN "$CKPT" "$OUT" --label dt4-defer-thinkon $SHIP --spec-k 5 \
  --prompts "$SHAPES/thinkon-prompts.tsv" --defer-ab 3x256 2>&1 | tee "$OUT/run4-defer-thinkon.log"
env $BOTH $BIN "$CKPT" "$OUT" --label dt4-defer-ladder $SHIP --spec-ladder 1,5 \
  --prompts "$SHAPES/thinkon-prompts.tsv" --defer-ab 3x256 2>&1 | tee "$OUT/run4-defer-ladder.log"

# ---- phase 5: sampled probe, both seams (serving law) -----------------------------
env $BOTH $BIN "$CKPT" "$OUT" --label dt4-sampled $SHIP --spec-k 5 \
  --spec-sampled --spec-ab 2x256 --goldens "$HOME/realgate/dump" \
  --prompts "$SHAPES/thinkon-prompts.tsv" 2>&1 | tee "$OUT/run4-sampled.log"

echo "== devtwin battery v4 complete: receipts in $OUT"
