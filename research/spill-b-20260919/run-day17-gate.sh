#!/usr/bin/env bash
# Day 17 twin-gate cell driver, local RTX 5090: tools/prefix-newest-turn-fits-gate.py through the
# collector with the inherited canonical lock (`--rig rtx5090`, `/tmp/memra-5090.lock`), the lane-tip
# release binary (build-day17.sh receipt in rtx5090-day17/build-tip/). Nothing pre-creates --out: the
# collector creates it and refuses one that exists; the gate creates <out>/cell. Bounded lock retries,
# never kills a holder. The whole cell runs under the owner's CPU quota.
# usage: run-day17-gate.sh <cell-name> <collector-timeout-s> [gate args...]
set -uo pipefail
cell=${1:?cell}; tmo=${2:?timeout}; shift 2
WT=${WT:-$HOME/projects/wt-spill-b}
R=${R:-$WT/research/spill-b-20260919/rtx5090-day17}
BIN=${BIN:-$WT/target/release/memra-server}
MODEL=${MODEL:-/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
run() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@"; }
cd "$WT" || exit 1
for attempt in 0 1 2 3 4 5; do
  out=$R/$cell
  [ $attempt -gt 0 ] && out=$R/$cell-retry$attempt
  run python3 tools/tier-battery.py --rig rtx5090 --timeout "$tmo" --out "$out" --external-lock \
    --execute python3 tools/prefix-newest-turn-fits-gate.py --external-lock @COLLECTOR_LOCK_FD@ \
      --model "$MODEL" --bin "$BIN" --out "$out/cell" --gpu-lock /tmp/memra-5090.lock "$@" \
      > "$out-driver.log" 2>&1
  rc=$?
  echo $rc > "$out.exit"
  [ -d "$out" ] && git rev-parse HEAD > "$out/gate-source.txt"
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -d "$out/cell" ]; then
    echo "attempt $attempt: lock busy at $(date -u +%T), sleeping 90s" >> "$R/$cell-retries.log"
    sleep 90
    continue
  fi
  exit $rc
done
exit 3
