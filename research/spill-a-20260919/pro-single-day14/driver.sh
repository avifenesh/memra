#!/usr/bin/env bash
# Day 14 target-card sitting (one RTX PRO 6000 Blackwell): the tree WITH the per-device default,
# each cell one collector lock hold: the unit cells on the card (the default lease reads back
# cached, the driver's record 2), then conformance and roundtrip through the new default.
set -uo pipefail
R=/root/spill-receipts/a-day14
cd "$R"
[ "$(cat "$R/build/exit-gate")" = 0 ] || { echo "gate build failed, no cell"; exit 1; }
[ "$(cat "$R/build/exit-test")" = 0 ] || { echo "test build failed, no cell"; exit 1; }
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/driver.log"; }
run() { local name=$1; shift; log "$name start"; bash "$R/run-cell.sh" "$name" "$@"; local rc=$?; log "$name rc=$rc"; }
log "sitting start; tree $(cd /root/wt-a && git rev-parse HEAD) gate $(sha256sum "$R/bins/tier-transfer-gate" | cut -c1-16)"
tmux ls >> "$R/driver.log" 2>&1
nvidia-smi --query-gpu=name,power.limit,driver_version --format=csv,noheader > "$R/card.csv"
run gputest-s2 1500 1 -- bash "$R/gputest-cell.sh" @COLLECTOR_LOCK_FD@
run conformance-s2 900 1 -- bash "$R/gate-cell.sh" @COLLECTOR_LOCK_FD@ conformance
run roundtrip-s2 900 1 -- bash "$R/gate-cell.sh" @COLLECTOR_LOCK_FD@ roundtrip
log "sitting done"
