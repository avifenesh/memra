#!/usr/bin/env bash
# Day 12 attempt 2: the same three cells under new names (the collector refuses an existing --out).
# Attempt 1 (`tenant-base`, `tenant-fix`, `arena`) stays on disk as the record: the gate died on an
# unauthenticated /metrics scrape (401 under the keyring) after all seven completions returned 200,
# and the arena cell ran the tree's harness before the --basis commit reached the box.
set -uo pipefail
R=/root/spill-receipts/a-day12
cd "$R"
[ "$(cat "$R/build/exit-base")" = 0 ] && [ "$(cat "$R/build/exit-fix")" = 0 ] || { echo "build failed, no cells"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,power.max_limit,driver_version --format=csv > "$R/card.csv"
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/driver.log"; }
run() { local name=$1; shift; log "$name start"; bash "$R/run-cell.sh" "$name" "$@"; local rc=$?; log "$name rc=$rc"; }
log "attempt 2 start; tree $(cd /root/wt-a && git rev-parse HEAD) base $(sha256sum "$R/bins/memra-server-base" | cut -c1-16) fix $(sha256sum "$R/bins/memra-server-fix" | cut -c1-16)"
tmux ls >> "$R/driver.log" 2>&1
run tenant-base-r2 1500 1 -- bash "$R/tenant-cell.sh" base @COLLECTOR_LOCK_FD@ tenant-base-r2
run tenant-fix-r2 1500 1 -- bash "$R/tenant-cell.sh" fix @COLLECTOR_LOCK_FD@ tenant-fix-r2
run arena-r2 3600 0 -- bash "$R/arena-cell.sh" arena-r2
log "attempt 2 done"
