#!/usr/bin/env bash
# Day 14 twin-gate cell driver: one arm (main|fix) through the collector with the inherited canonical lock.
# Bounded lock retries (never kills a holder); every attempt leaves its own driver log.
# The gate source is the checked-out /root/wt-b (recorded in <cell>/gate-source.txt).
set -uo pipefail
arm=${1:?arm main|fix}; suffix=${2:-}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
R=/root/spill-receipts/b-day14
cd /root/wt-b
for attempt in 0 1 2 3 4 5; do
  out=$R/gate-$arm$suffix
  [ $attempt -gt 0 ] && out=$R/gate-$arm$suffix-retry$attempt
  mkdir -p "$out"
  git rev-parse HEAD > "$out/gate-source.txt"
  python3 tools/tier-battery.py --rig pro-single --timeout 1800 --out "$out" --external-lock \
    --execute python3 tools/prefix-newest-turn-fits-gate.py --external-lock @COLLECTOR_LOCK_FD@ \
      --model /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf \
      --bin $R/bins/$arm/memra-server \
      --out "$out/cell" --budget-mib 1024 --cohort-tokens 2800,3000,3200 \
      --turns 8 --start-tokens 9200 --grow-tokens 300 > "$out-driver.log" 2>&1
  rc=$?
  echo $rc > "$out.exit"
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -d "$out/cell" ]; then
    echo "attempt $attempt: lock busy, sleeping 90s" >> "$R/gate-$arm-retries.log"
    sleep 90
    continue
  fi
  exit $rc
done
exit 3
