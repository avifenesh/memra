#!/usr/bin/env bash
# Day 15 driver: the two #385 arena cells (chunked, thp) through the collector, then the door's
# cached-destination pair on lane C's harness if the memra-server build finished green. Every cell is one
# lock hold; a busy lock is retried by run-cell.sh (15 x 120 s), never signalled.
set -uo pipefail
R=/root/spill-receipts/a-day15
cd "$R"
nvidia-smi --query-gpu=name,power.limit,power.max_limit,driver_version --format=csv > "$R/card.csv"
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/driver.log"; }
run() { local name=$1; shift; log "$name start"; bash "$R/run-cell.sh" "$name" "$@"; local rc=$?; log "$name rc=$rc"; }
log "start tree $(cd /root/wt-a && git rev-parse HEAD)"
tmux ls >> "$R/driver.log" 2>&1
run arena-chunked 3600 0 -- bash "$R/arena-cell.sh" arena-chunked chunked
run arena-thp 3600 0 -- bash "$R/arena-cell.sh" arena-thp thp
if [ "$(cat "$R/build/exit" 2>/dev/null)" = 0 ]; then
  log "wc-pair binary $(sha256sum "$R/bins/memra-server" | cut -c1-16) source $(cat "$R/build/source.txt")"
  run wc-pair 2400 1 -- bash "$R/wc-cell.sh" @COLLECTOR_LOCK_FD@ wc-pair
else
  log "wc-pair skipped: build exit $(cat "$R/build/exit" 2>/dev/null || echo missing)"
fi
log "done"
