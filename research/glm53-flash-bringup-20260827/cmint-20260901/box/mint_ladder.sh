#!/usr/bin/env bash
# cmint cell 1 - balance-tolerance ladder, CPU only, no GPU, no marker.
# Inputs: the BANKED struct-battery decode-filtered agentic trace (shas verified in WINDOW.md).
set -euo pipefail
T=/root/memra-tpd/tools/build_expert_placement_map.py     # sha256 7d05a8d4d5cc0a5b... (== ep-diet head)
ARGS="--trace traces/agentic-t1-dec.ids --weight-trace traces/agentic-t1-dec.w \
      --ranks 2 --entry-rank 0 --expert-count 288 --strategy coactivation --decode-only"
for tol in 0.00 0.01 0.02 0.05 0.10 0.25; do
  python3 $T $ARGS --balance-tolerance "$tol" --out "maps/coact-tol$tol.json" > "receipts/mint-tol$tol.log" 2>&1 &
done
python3 $T --trace traces/agentic-t1-dec.ids --weight-trace traces/agentic-t1-dec.w \
  --ranks 2 --entry-rank 0 --expert-count 288 --strategy even --decode-only \
  --out maps/even.json > receipts/mint-even.log 2>&1 &
wait
