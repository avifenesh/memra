#!/usr/bin/env bash
# Day 12 attempt 3: the two gate cells again after the gate's matchers moved to ERE (attempt 2's
# `[prefix-host]` patterns were basic-regex bracket expressions). The arena cell of attempt 2 stands.
set -uo pipefail
R=/root/spill-receipts/a-day12
cd "$R"
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/driver.log"; }
run() { local name=$1; shift; log "$name start"; bash "$R/run-cell.sh" "$name" "$@"; local rc=$?; log "$name rc=$rc"; }
log "attempt 3 start; tree $(cd /root/wt-a && git rev-parse HEAD) base $(sha256sum "$R/bins/memra-server-base" | cut -c1-16) fix $(sha256sum "$R/bins/memra-server-fix" | cut -c1-16)"
tmux ls >> "$R/driver.log" 2>&1
run tenant-base-r3 1500 1 -- bash "$R/tenant-cell.sh" base @COLLECTOR_LOCK_FD@ tenant-base-r3
run tenant-fix-r3 1500 1 -- bash "$R/tenant-cell.sh" fix @COLLECTOR_LOCK_FD@ tenant-fix-r3
log "attempt 3 done"
