#!/usr/bin/env bash
# devtwin battery, stage 1 (device MoE router) — runs ON sbox-eval (box), only when the
# box is IDLE. Ship admission everywhere: dev1 placement, K=5 ceiling, adapt k_lo=1,
# pmin 0.3. Audit (MEMRA_Q4E_ROUTER_AUDIT=1) is ON for the correctness phases only — it
# dtohs per route, so it never rides a perf arm.
set -euo pipefail
cd ~/memra
git fetch origin && git pull --ff-only origin qwen4exp-bringup-20260829
echo "== tip: $(git log -1 --oneline)"   # rebuild-after-checkout law: tip in the receipt
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
RD="MEMRA_Q4E_SEAMS=routerdev"
RDA="MEMRA_Q4E_SEAMS=routerdev MEMRA_Q4E_ROUTER_AUDIT=1"

# ---- phase 0: tiny gate on the box toolchain/libm (the oracle's exp-twin question is
# per-host: rig receipt is not a box receipt) --------------------------------------
env $RDA ./target/release/qwen4exp_gpu_gate "$OUT/tiny-gate-routerdev-box.tsv" \
  2>&1 | tee "$OUT/run-tiny.log"

# ---- phase 1: rule gates with the device router armed + the live audit ----------
env $RDA $BIN "$CKPT" "$OUT" --label dt-vbit $SHIP --spec-k 5 \
  --goldens "$HOME/realgate/dump" --verify-bit-gate 24 2>&1 | tee "$OUT/run-vbit.log"
env $RDA $BIN "$CKPT" "$OUT" --label dt-raw $SHIP --spec-k 5 \
  --prompts "$RAW" --spec-gate 256 2>&1 | tee "$OUT/run-gate-raw.log"
env $RDA $BIN "$CKPT" "$OUT" --label dt-thinkon $SHIP --spec-k 5 \
  --prompts "$SHAPES/thinkon-prompts.tsv" --spec-gate 256 2>&1 | tee "$OUT/run-gate-thinkon.log"
env $RDA $BIN "$CKPT" "$OUT" --label dt-long $SHIP --spec-k 5 \
  --prompts "$LONG" --spec-gate 256 2>&1 | tee "$OUT/run-gate-long.log"
# decode-row twin: OFF vs ON logits envelope on the same fed tokens (24 steps).
env MEMRA_Q4E_ROUTER_AUDIT=1 $BIN "$CKPT" "$OUT" --label dt-seam \
  --goldens "$HOME/realgate/dump" --ab-seam routerdev --seam-gate 24 \
  2>&1 | tee "$OUT/run-seam.log"

# ---- phase 2: plain-decode A/B (the graph-driver route boundary), interleaved ----
$BIN "$CKPT" "$OUT" --label dt-plain --goldens "$HOME/realgate/dump" \
  --ab-seam routerdev --ab-moe 5x128 2>&1 | tee "$OUT/run-ab-plain.log"

# ---- phase 3: spec A/B per shape at ship K=5 (x3 + receipted escalation) ---------
for shape in thinkon thinkoff efflow; do
  $BIN "$CKPT" "$OUT" --label dt-$shape $SHIP --spec-k 5 \
    --prompts "$SHAPES/$shape-prompts.tsv" --router-ab 3x256 2>&1 | tee "$OUT/run-ab-$shape.log"
done
$BIN "$CKPT" "$OUT" --label dt-raw $SHIP --spec-k 5 \
  --prompts "$RAW" --router-ab 3x256 2>&1 | tee "$OUT/run-ab-raw.log"
$BIN "$CKPT" "$OUT" --label dt-long $SHIP --spec-k 5 \
  --prompts "$LONG" --router-ab 3x256 2>&1 | tee "$OUT/run-ab-long.log"

# ---- phase 4: K ladder around the knee, ONE load per shape -----------------------
$BIN "$CKPT" "$OUT" --label dt-ladder-thinkon $SHIP --spec-ladder 1,2,3,4,5,6,8 \
  --prompts "$SHAPES/thinkon-prompts.tsv" --router-ab 3x256 2>&1 | tee "$OUT/run-ladder-thinkon.log"
$BIN "$CKPT" "$OUT" --label dt-ladder-raw $SHIP --spec-ladder 1,2,3,4,5,6,8 \
  --prompts "$RAW" --router-ab 3x256 2>&1 | tee "$OUT/run-ladder-raw.log"

# ---- phase 5: sampled probe, device router armed (serving law) -------------------
env $RD $BIN "$CKPT" "$OUT" --label dt-sampled $SHIP --spec-k 5 \
  --spec-sampled --spec-ab 2x256 --goldens "$HOME/realgate/dump" \
  --prompts "$SHAPES/thinkon-prompts.tsv" 2>&1 | tee "$OUT/run-sampled.log"

# phase 6 (defer re-measure from the mtp11 banked baseline) runs AFTER the draft-side
# twins move (devtwin stage 2): the chain-side ceiling only rises then (PROFILE-8 §4).

echo "== devtwin stage-1 battery complete: receipts in $OUT"
