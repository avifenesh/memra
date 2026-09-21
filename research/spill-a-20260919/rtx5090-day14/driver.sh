#!/usr/bin/env bash
# Day 14 first sitting on the local RTX 5090 Laptop GPU (the tree before any default moves): build
# check, then the cells in order, each one collector lock hold: the unit cells on the card,
# conformance and roundtrip (today's default arm), the 160 MiB decision cell, the 16 MiB context
# cell. Run under systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G.
set -uo pipefail
W=/home/avifenesh/projects/wt-spill-a
R=$W/research/spill-a-20260919/rtx5090-day14
cd "$R"
[ "$(cat "$R/build/exit-gate")" = 0 ] || { echo "gate build failed, no cell"; exit 1; }
[ "$(cat "$R/build/exit-test")" = 0 ] || { echo "test build failed, no cell"; exit 1; }
mkdir -p "$R/bins"; cp "$W/target/release/tier-transfer-gate" "$R/bins/tier-transfer-gate"
sha256sum "$R/bins/tier-transfer-gate" > "$R/build/binary-gate.sha256"
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/driver.log"; }
run() { local name=$1; shift; log "$name start"; bash "$R/run-cell.sh" "$name" "$@"; local rc=$?; log "$name rc=$rc"; }
log "first sitting start; tree $(cd "$W" && git rev-parse HEAD) gate $(sha256sum "$R/bins/tier-transfer-gate" | cut -c1-16)"
tmux ls >> "$R/driver.log" 2>&1
nvidia-smi --query-gpu=name,compute_cap,memory.total,power.limit,driver_version --format=csv,noheader > "$R/card.csv"
run gputest 1500 -- bash "$R/gputest-cell.sh" @COLLECTOR_LOCK_FD@
run conformance 900 -- bash "$R/gate-cell.sh" @COLLECTOR_LOCK_FD@ conformance
run roundtrip 900 -- bash "$R/gate-cell.sh" @COLLECTOR_LOCK_FD@ roundtrip
run pinned-ab-160m 1500 -- bash "$R/pinned-cell.sh" @COLLECTOR_LOCK_FD@ 167772160 5
run pinned-ab-16m 900 -- bash "$R/pinned-cell.sh" @COLLECTOR_LOCK_FD@ 16777216 5
log "first sitting done"
