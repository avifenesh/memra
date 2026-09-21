#!/usr/bin/env bash
# Day 13 second sitting (the flags check tests the write-combined bit; the same cells under -s2
# names, each one collector lock hold): the arm-honoured unit cell, conformance and roundtrip
# (default arm regression), the 160 MiB decision cell, the 16 MiB context cell.
set -uo pipefail
R=/root/spill-receipts/a-day13
cd "$R"
[ "$(cat "$R/build2/exit-gate")" = 0 ] || { echo "gate build failed, no cell"; exit 1; }
[ "$(cat "$R/build2/exit-test")" = 0 ] || { echo "test build failed, no cell"; exit 1; }
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/driver.log"; }
run() { local name=$1; shift; log "$name start"; bash "$R/run-cell.sh" "$name" "$@"; local rc=$?; log "$name rc=$rc"; }
log "second sitting start; tree $(cd /root/wt-a && git rev-parse HEAD) gate $(sha256sum "$R/bins/tier-transfer-gate" | cut -c1-16)"
tmux ls >> "$R/driver.log" 2>&1
nvidia-smi --query-gpu=name,power.limit,driver_version --format=csv,noheader > "$R/card.csv"
run gputest-s2 1500 1 -- bash "$R/gputest-cell.sh" @COLLECTOR_LOCK_FD@
run conformance-s2 900 1 -- bash "$R/gate-cell.sh" @COLLECTOR_LOCK_FD@ conformance
run roundtrip-s2 900 1 -- bash "$R/gate-cell.sh" @COLLECTOR_LOCK_FD@ roundtrip
run pinned-ab-160m-s2 1500 1 -- bash "$R/pinned-cell.sh" @COLLECTOR_LOCK_FD@ 167772160 5
run pinned-ab-16m-s2 900 1 -- bash "$R/pinned-cell.sh" @COLLECTOR_LOCK_FD@ 16777216 5
log "second sitting done"
