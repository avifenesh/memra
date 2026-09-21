#!/usr/bin/env bash
# Day 13 third sitting: the rate discrepancy with lane C's whole-demote line. The gate at the server's
# per-item size (160.7 MB over 34 items = 4.7 MB) and the driver-independent ctypes probe at 160 MiB
# and at 4.7 MB. Each cell one collector lock hold.
set -uo pipefail
R=/root/spill-receipts/a-day13
cd "$R"
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/driver.log"; }
run() { local name=$1; shift; log "$name start"; bash "$R/run-cell.sh" "$name" "$@"; local rc=$?; log "$name rc=$rc"; }
log "third sitting start; tree $(cd /root/wt-a && git rev-parse HEAD) gate $(sha256sum "$R/bins/tier-transfer-gate" | cut -c1-16) probe $(sha256sum "$R/pinned-read-probe.py" | cut -c1-16)"
tmux ls >> "$R/driver.log" 2>&1
run pinned-ab-4m7-s2 900 1 -- bash "$R/pinned-cell.sh" @COLLECTOR_LOCK_FD@ 4718592 3
run probe-160m 900 1 -- bash "$R/probe-cell.sh" @COLLECTOR_LOCK_FD@ 167772160 3
run probe-4m7 900 1 -- bash "$R/probe-cell.sh" @COLLECTOR_LOCK_FD@ 4718592 3
log "third sitting done"
