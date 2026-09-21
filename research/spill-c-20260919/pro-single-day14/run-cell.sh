#!/usr/bin/env bash
# Day 14 cell runner: one command through the canonical collector (pro-single, /tmp/memra-gpu.lock),
# bounded lock retries (6 x 120 s), never kills a holder; every attempt keeps its own driver log.
# usage: run-cell.sh <name> <timeout_s> <external-lock 0|1> -- <argv...>
set -uo pipefail
name=$1; timeout=$2; ext=$3; shift 3; [ "$1" = "--" ] && shift
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
R=/root/spill-receipts/c-day14
cd /root/wt-c
for attempt in 0 1 2 3 4 5; do
  out=$R/$name; [ $attempt -gt 0 ] && out=$R/$name-retry$attempt
  extra=(); [ "$ext" = 1 ] && extra=(--external-lock)
  python3 tools/tier-battery.py --rig pro-single --timeout "$timeout" --out "$out" "${extra[@]}" --execute "$@" > "$out-driver.log" 2>&1
  rc=$?
  echo $rc > "$out.exit"
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -f "$out/CELL.jsonl" ]; then
    echo "$(date -u +%FT%TZ) $name attempt $attempt: lock busy, sleeping 120s" >> "$R/lock-retries.log"
    sleep 120; continue
  fi
  exit $rc
done
exit 3
