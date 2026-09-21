#!/usr/bin/env bash
# Day 14 second sitting on the local RTX 5090 Laptop GPU: the tree WITH the per-device default
# (build2/): the unit cells on the card (the default lease reads back this card's resolved arm),
# then conformance and roundtrip through the new default, each one collector lock hold. Run under
# systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G.
set -uo pipefail
W=/home/avifenesh/projects/wt-spill-a
R=$W/research/spill-a-20260919/rtx5090-day14
cd "$R"
[ "$(cat "$R/build2/exit-gate")" = 0 ] || { echo "gate build failed, no cell"; exit 1; }
[ "$(cat "$R/build2/exit-test")" = 0 ] || { echo "test build failed, no cell"; exit 1; }
mkdir -p "$R/bins"; cp "$W/target/release/tier-transfer-gate" "$R/bins/tier-transfer-gate"
sha256sum "$R/bins/tier-transfer-gate" > "$R/build2/binary-gate.sha256"
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/driver.log"; }
run() { local name=$1; shift; log "$name start"; bash "$R/run-cell.sh" "$name" "$@"; local rc=$?; log "$name rc=$rc"; }
log "second sitting start; tree $(cd "$W" && git rev-parse HEAD) gate $(sha256sum "$R/bins/tier-transfer-gate" | cut -c1-16)"
tmux ls >> "$R/driver.log" 2>&1
run gputest-s2 1500 -- bash "$R/gputest-cell.sh" @COLLECTOR_LOCK_FD@
run conformance-s2 900 -- bash "$R/gate-cell.sh" @COLLECTOR_LOCK_FD@ conformance
run roundtrip-s2 900 -- bash "$R/gate-cell.sh" @COLLECTOR_LOCK_FD@ roundtrip
log "second sitting done"
