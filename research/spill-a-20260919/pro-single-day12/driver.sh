#!/usr/bin/env bash
# Day 12 cell sequence on the target card: the memra#384 gate on the base binary, then on the fix
# binary (same gate script, same prompts, same knobs), then the memra#385 harness cell.
set -uo pipefail
R=/root/spill-receipts/a-day12
cd "$R"
[ "$(cat "$R/build/exit-base")" = 0 ] && [ "$(cat "$R/build/exit-fix")" = 0 ] || { echo "build failed, no cells"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,power.max_limit,driver_version --format=csv > "$R/card.csv"
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/driver.log"; }
run() { local name=$1; shift; log "$name start"; bash "$R/run-cell.sh" "$name" "$@"; local rc=$?; log "$name rc=$rc"; }
log "start base $(sha256sum "$R/bins/memra-server-base" | cut -c1-16) fix $(sha256sum "$R/bins/memra-server-fix" | cut -c1-16) source $(cat "$R/build/source-fix.txt")"
tmux ls >> "$R/driver.log" 2>&1
run tenant-base 1500 1 -- bash "$R/tenant-cell.sh" base @COLLECTOR_LOCK_FD@
run tenant-fix 1500 1 -- bash "$R/tenant-cell.sh" fix @COLLECTOR_LOCK_FD@
run arena 3600 0 -- bash "$R/arena-cell.sh"
log "done"
