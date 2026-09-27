#!/usr/bin/env bash
# Day 15 A/B cell driver: the policy A/B through the collector with the inherited canonical lock.
# usage: run-ab.sh <cell-name> <pairs-per-order> <collector-timeout-s> [arm-bin (default tip)]
# The collector creates --out itself and refuses one that exists: nothing here pre-creates it.
# Bounded lock retries (never kills a holder); every attempt leaves its own driver log.
set -uo pipefail
cell=${1:?cell}; pairs=${2:?pairs}; tmo=${3:?timeout}; arm=${4:-tip}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
R=/root/spill-receipts/b-day15
cd /root/wt-b
for attempt in 0 1 2 3 4 5; do
  out=$R/$cell
  [ $attempt -gt 0 ] && out=$R/$cell-retry$attempt
  python3 tools/tier-battery.py --rig pro-single --timeout "$tmo" --out "$out" --external-lock \
    --execute python3 tools/prefix-policy-ab.py --external-lock @COLLECTOR_LOCK_FD@ \
      --model /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf \
      --bin $R/bins/$arm/memra-server \
      --out "$out/cell" --budget-mib 2048 --cohort-tokens 7800,8000,8200,8400 \
      --turns 12 --start-tokens 27300 --grow-tokens 300 --return-every 3 --pairs "$pairs" \
      --ctx 32768 --max-tokens 8 > "$out-driver.log" 2>&1
  rc=$?
  echo $rc > "$out.exit"
  [ -d "$out" ] && git rev-parse HEAD > "$out/gate-source.txt"
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -d "$out/cell" ]; then
    echo "attempt $attempt: lock busy, sleeping 90s" >> "$R/$cell-retries.log"
    sleep 90
    continue
  fi
  exit $rc
done
exit 3
