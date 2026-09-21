#!/usr/bin/env bash
# Day 19 twin-gate cell driver, local RTX 5090: tools/prefix-newest-turn-fits-gate.py (V5 identity and V6
# grid are verdict clauses since day 18) through the collector with the inherited canonical lock
# (`--rig rtx5090`, `/tmp/memra-5090.lock`), one arm's release binary (target/bins/<arm>/memra-server,
# build receipts in rtx5090-day19/build-<arm>/). The collector creates --out and refuses one that exists;
# the gate creates <out>/cell. Bounded lock retries, never kills a holder. Under the owner's CPU quota.
# usage: run-day18-gate.sh <arm base|fix> <cell-name> <collector-timeout-s> [gate args...]
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
    --execute python3 tools/prefix-newest-turn-fits-gate.py --external-lock @COLLECTOR_LOCK_FD@ \
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
