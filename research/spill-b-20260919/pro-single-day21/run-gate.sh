#!/usr/bin/env bash
# Day 21 twin-gate cell driver, target card: tools/prefix-newest-turn-fits-gate.py on the lane tip binary
# (main fa5185d90 merged, the LRU default) through the collector with the inherited canonical lock
# (--rig pro-single, /tmp/memra-gpu.lock). Bounded lock retries, never kills a holder. Pass/fail, not timed.
# usage: run-gate.sh <cell-name> <collector-timeout-s> [gate args...]
set -uo pipefail
cell=${1:?cell}; tmo=${2:?timeout}; shift 2
R=/root/spill-receipts/b-day21; BIN=$R/bins/tip/memra-server; MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
cd /root/wt-b || exit 1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-before.csv"
for attempt in 0 1 2 3 4 5; do
  out=$R/$cell; [ $attempt -gt 0 ] && out=$R/$cell-retry$attempt
  python3 tools/tier-battery.py --rig pro-single --timeout "$tmo" --out "$out" --external-lock \
    --execute python3 tools/prefix-newest-turn-fits-gate.py --external-lock @COLLECTOR_LOCK_FD@ \
      --model "$MODEL" --bin "$BIN" --out "$out/cell" --gpu-lock /tmp/memra-gpu.lock "$@" > "$out-driver.log" 2>&1
  rc=$?; echo $rc > "$out.exit"
  [ -d "$out" ] && { git rev-parse HEAD > "$out/gate-source.txt"; sha256sum "$BIN" > "$out/binary.sha256"; }
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-after.csv"
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -d "$out/cell" ]; then
    echo "attempt $attempt: lock busy at $(date -u +%T), sleeping 90s" >> "$R/$cell-retries.log"; sleep 90; continue
  fi
  exit $rc
done
exit 3
