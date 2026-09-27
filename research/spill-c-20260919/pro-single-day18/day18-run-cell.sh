#!/usr/bin/env bash
# Day 18 cell runner: one cell through the canonical collector with bounded lock retries (up to 75 x 120 s;
# another lane may hold the card), never signals a holder; every attempt keeps its own driver log.
# Environment: D18_RIG (pro-single|rtx5090), D18_R receipts root, D18_TREE worktree, plus the day18-cell.sh
# variables. usage: day18-run-cell.sh <cell> <timeout_s>
set -uo pipefail
cell=$1; timeout=$2
: "${D18_RIG:?}" "${D18_R:?}" "${D18_TREE:?}" "${D18_CELL_SCRIPT:?}"
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$D18_TREE" || exit 1
for attempt in $(seq 0 75); do
  out=$D18_R/$cell; [ "$attempt" -gt 0 ] && out=$D18_R/$cell-retry$attempt
  python3 tools/tier-battery.py --rig "$D18_RIG" --timeout "$timeout" --out "$out" --external-lock \
      --execute bash "$D18_CELL_SCRIPT" "$cell" @COLLECTOR_LOCK_FD@ > "$out-driver.log" 2>&1
  rc=$?
  echo $rc > "$out.exit"
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -f "$out/CELL.jsonl" ]; then
    echo "$(date -u +%FT%TZ) $cell attempt $attempt: lock busy, sleeping 120s" >> "$D18_R/lock-retries.log"
    rm -f "$out.exit"; rmdir "$out" 2>/dev/null
    sleep 120; continue
  fi
  exit $rc
done
echo "$(date -u +%FT%TZ) $cell: gave up after bounded retries" >> "$D18_R/lock-retries.log"
exit 3
