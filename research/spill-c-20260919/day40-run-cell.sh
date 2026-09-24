#!/usr/bin/env bash
# Day 40 cell runner: one cell through the canonical collector with bounded lock retries (D40_LOCK_TRIES x 120 s,
# default 30; another lane may hold the card), never signals a holder; every attempt keeps its own driver log.
# Before each attempt it waits, bounded (D40_WAITS x 120 s, default 15), for no compute app on the card and at
# least D40_MIN_AVAIL_GB of host MemAvailable (the door's cell pins the loader's 15 GB of expert slabs beside the
# page-cached artifact).
# Environment: D40_RIG (pro-single|rtx5090), D40_R receipts root, D40_TREE worktree, D40_CELL_SCRIPT, plus the
# day40-cell.sh variables. usage: day40-run-cell.sh <cell> <timeout_s>
set -uo pipefail
cell=$1; timeout=$2
: "${D40_RIG:?}" "${D40_R:?}" "${D40_TREE:?}" "${D40_CELL_SCRIPT:?}"
min_avail_gb=${D40_MIN_AVAIL_GB:-36}
waits=${D40_WAITS:-15}
lock_tries=${D40_LOCK_TRIES:-30}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$D40_TREE" || exit 1
mkdir -p "$D40_R"
wait_free() {
  for w in $(seq 1 "$waits"); do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    if [ "$apps" = 0 ] && [ "$avail_kb" -ge $((min_avail_gb * 1024 * 1024)) ]; then return 0; fi
    { echo "$(date -u +%FT%TZ) $cell wait $w: compute_apps=$apps MemAvailable_kB=$avail_kb (need 0 and >= ${min_avail_gb} GiB)"
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1; } >> "$D40_R/waits.log"
    sleep 120
  done
  echo "$(date -u +%FT%TZ) $cell: card or host never freed in $waits waits" >> "$D40_R/waits.log"
  return 1
}
for attempt in $(seq 0 "$lock_tries"); do
  wait_free || exit 4
  out=$D40_R/$cell; [ "$attempt" -gt 0 ] && out=$D40_R/$cell-retry$attempt
  python3 tools/tier-battery.py --rig "$D40_RIG" --timeout "$timeout" --out "$out" --external-lock \
      --execute bash "$D40_CELL_SCRIPT" "$cell" @COLLECTOR_LOCK_FD@ > "$out-driver.log" 2>&1
  rc=$?
  echo $rc > "$out.exit"
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -f "$out/CELL.jsonl" ]; then
    echo "$(date -u +%FT%TZ) $cell attempt $attempt: lock busy, sleeping 120s" >> "$D40_R/lock-retries.log"
    rm -f "$out.exit"; rmdir "$out" 2>/dev/null
    sleep 120; continue
  fi
  exit $rc
done
echo "$(date -u +%FT%TZ) $cell: gave up after bounded retries" >> "$D40_R/lock-retries.log"
exit 3
