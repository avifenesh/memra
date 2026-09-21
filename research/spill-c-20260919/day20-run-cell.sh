#!/usr/bin/env bash
# Day 20 cell runner: one cell through the canonical collector with bounded lock retries (up to 30 x 120 s;
# another lane may hold the card), never signals a holder; every attempt keeps its own driver log.
# Environment: D20_RIG (rtx5090|pro-single), D20_R receipts root, D20_TREE worktree, plus the day20-cell.sh
# variables. usage: day20-run-cell.sh <cell> <timeout_s>
set -uo pipefail
cell=$1; timeout=$2
: "${D20_RIG:?}" "${D20_R:?}" "${D20_TREE:?}"
cd "$D20_TREE" || exit 1
for attempt in $(seq 0 30); do
  out=$D20_R/$cell; [ "$attempt" -gt 0 ] && out=$D20_R/$cell-retry$attempt
  D20_OUT=$out python3 tools/tier-battery.py --rig "$D20_RIG" --timeout "$timeout" --out "$out" --external-lock \
      --execute bash research/spill-c-20260919/day20-cell.sh "$cell" @COLLECTOR_LOCK_FD@ > "$out-driver.log" 2>&1
  rc=$?
  echo $rc > "$out.exit"
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -f "$out/CELL.jsonl" ]; then
    echo "$(date -u +%FT%TZ) $cell attempt $attempt: lock busy, sleeping 120s" >> "$D20_R/lock-retries.log"
    rm -f "$out.exit"; rmdir "$out" 2>/dev/null
    sleep 120; continue
  fi
  exit $rc
done
echo "$(date -u +%FT%TZ) $cell: gave up after bounded retries" >> "$D20_R/lock-retries.log"
exit 3
