#!/usr/bin/env bash
# Day 14 cell runner on the local RTX 5090 Laptop GPU: one command through the canonical collector
# (--rig rtx5090, /tmp/memra-5090.lock), bounded lock retries (15 x 120 s; lane B may run cells on
# this card), never kills a holder; every attempt keeps its own driver log. The collector creates
# --out itself and refuses one that exists, so a retry gets its own name; the cell script writes
# its evidence under CELL_OUT/ev of the attempt. The whole cell runs inside the caller's
# systemd-run scope (CPUQuota=1200%, MemoryMax=28G: the owner's local-rig cap).
# usage: run-cell.sh <name> <timeout_s> -- <argv...>
set -uo pipefail
name=$1; timeout=$2; shift 2; [ "$1" = "--" ] && shift
W=/home/avifenesh/projects/wt-spill-a
R=$W/research/spill-a-20260919/rtx5090-day14
cd "$W"
for attempt in $(seq 0 14); do
  out=$R/$name; [ $attempt -gt 0 ] && out=$R/$name-retry$attempt
  CELL_OUT=$out python3 tools/tier-battery.py --rig rtx5090 --timeout "$timeout" --out "$out" --external-lock --execute "$@" > "$out-driver.log" 2>&1
  rc=$?
  echo $rc > "$out.exit"
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -f "$out/CELL.jsonl" ]; then
    echo "$(date -u +%FT%TZ) $name attempt $attempt: lock busy, sleeping 120s" >> "$R/lock-retries.log"
    sleep 120; continue
  fi
  exit $rc
done
exit 3
