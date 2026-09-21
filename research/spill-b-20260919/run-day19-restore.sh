#!/usr/bin/env bash
# Day 19 restore-identity cell driver, local RTX 5090: tools/prefix-restore-identity-gate.py (the day-17
# five restore points of the 12,350-token day-16 prompt) through the collector with the inherited canonical
# lock, one arm's release binary. Same collector, lock, retry and receipt discipline as run-day18-gate.sh.
# usage: run-day18-restore.sh <arm base|fix> <cell-name> <collector-timeout-s> [gate args...]
set -uo pipefail
arm=${1:?arm}; cell=${2:?cell}; tmo=${3:?timeout}; shift 3
WT=${WT:-$HOME/projects/wt-spill-b}
R=${R:-$WT/research/spill-b-20260919/rtx5090-day19}
BIN=${BIN:-$WT/target/bins/$arm/memra-server}
MODEL=${MODEL:-/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
run() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@"; }
cd "$WT" || exit 1
for attempt in 0 1 2 3 4 5; do
  out=$R/$cell
  [ $attempt -gt 0 ] && out=$R/$cell-retry$attempt
  run python3 tools/tier-battery.py --rig rtx5090 --timeout "$tmo" --out "$out" --external-lock \
    --execute python3 tools/prefix-restore-identity-gate.py --external-lock @COLLECTOR_LOCK_FD@ \
      --model "$MODEL" --bin "$BIN" --out "$out/cell" --gpu-lock /tmp/memra-5090.lock "$@" \
      > "$out-driver.log" 2>&1
  rc=$?
  echo $rc > "$out.exit"
  [ -d "$out" ] && { git rev-parse HEAD > "$out/gate-source.txt"; sha256sum "$BIN" > "$out/binary.sha256"; }
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -d "$out/cell" ]; then
    echo "attempt $attempt: lock busy at $(date -u +%T), sleeping 90s" >> "$R/$cell-retries.log"
    sleep 90
    continue
  fi
  exit $rc
done
exit 3
