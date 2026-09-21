#!/usr/bin/env bash
# Day 12 attempt 4 (integ15 review): the fix arm only, on the deferred-reclaim build, against the
# unchanged base receipts of attempt 3 (the base binary is origin/main be07f2d36 and did not change).
set -uo pipefail
R=/root/spill-receipts/a-day12
cd "$R"
[ "$(cat "$R/build/exit-fix2")" = 0 ] || { echo "fix2 build failed, no cell"; exit 1; }
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/driver.log"; }
run() { local name=$1; shift; log "$name start"; bash "$R/run-cell.sh" "$name" "$@"; local rc=$?; log "$name rc=$rc"; }
log "attempt 4 start; tree $(cd /root/wt-a && git rev-parse HEAD) fix2 $(sha256sum "$R/bins/memra-server-fix2" | cut -c1-16)"
tmux ls >> "$R/driver.log" 2>&1
run tenant-fix-r4 1500 1 -- bash "$R/tenant-cell.sh" fix @COLLECTOR_LOCK_FD@ tenant-fix-r4 fix2
log "attempt 4 done"
