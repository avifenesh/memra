#!/usr/bin/env bash
# Day 21 twin-gate cell driver, local RTX 5090: tools/prefix-newest-turn-fits-gate.py on the lane tip's
# release binary (main fa5185d90 merged, the LRU default) through the collector with the inherited
# canonical lock (`--rig rtx5090`, `/tmp/memra-5090.lock`). The collector creates --out and refuses one
# that exists; the gate creates <out>/cell. Bounded lock retries, never kills a holder. Under the owner's
# CPU quota. Pass/fail cell, not timed.
# usage: run-day21-gate.sh <cell-name> <collector-timeout-s> [gate args...]
set -uo pipefail
cell=${1:?cell}; tmo=${2:?timeout}; shift 2
WT=${WT:-$HOME/projects/wt-spill-b}
R=${R:-$WT/research/spill-b-20260919/rtx5090-day21}
BIN=${BIN:-$WT/target/release/memra-server}
MODEL=${MODEL:-/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
run() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@"; }
cd "$WT" || exit 1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-before.csv"
for attempt in 0 1 2 3 4 5; do
  out=$R/$cell
  [ $attempt -gt 0 ] && out=$R/$cell-retry$attempt
  run python3 tools/tier-battery.py --rig rtx5090 --timeout "$tmo" --out "$out" --external-lock \
    --execute python3 tools/prefix-newest-turn-fits-gate.py --external-lock @COLLECTOR_LOCK_FD@ \
      --model "$MODEL" --bin "$BIN" --out "$out/cell" --gpu-lock /tmp/memra-5090.lock "$@" \
      > "$out-driver.log" 2>&1
  rc=$?
  echo $rc > "$out.exit"
  [ -d "$out" ] && { git rev-parse HEAD > "$out/gate-source.txt"; sha256sum "$BIN" > "$out/binary.sha256"; }
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-after.csv"
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -d "$out/cell" ]; then
    echo "attempt $attempt: lock busy at $(date -u +%T), sleeping 90s" >> "$R/$cell-retries.log"
    sleep 90
    continue
  fi
  exit $rc
done
exit 3
