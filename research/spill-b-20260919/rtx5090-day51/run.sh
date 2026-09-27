#!/usr/bin/env bash
# WP-B DAY51 local after cell (DAY51 1.4): the DAY28 shape walk on the 9B at MEMRA_CTX=65536 with memra#476's boot
# pre-grow. Build the tree's release server under the CPU quota, then hold /tmp/memra-5090.lock, check the card idle
# under the hold (rig-hold.sh: no compute app, host memory), run run-day28-cell.sh with LOCK=none and the hold's fd
# closed, release (the cell runs day28-parse.py into after/REPORT.txt). Never a signal to anything this lane did not start.
set -uo pipefail
WT=/home/avifenesh/projects/wt-b-integ72
R=$WT/research/spill-b-20260919/rtx5090-day51
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/run.log"; }
log "start HEAD=$(git rev-parse HEAD)"
echo "$(date -u +%FT%TZ) WP-B DAY51 release build (nice 19, CPUQuota=600%, MemoryMax=12G)" >> research/spill-b-20260919/cpu-concurrency.log
systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=12G nice -n 19 \
  cargo build --release -p memra-server > "$R/build.log" 2>&1
rc=$?
log "build rc=$rc"
[ $rc = 0 ] || exit 1
BIN=$WT/target/release/memra-server
sha256sum "$BIN" > "$R/binary.sha256"
RIG_LOCK=/tmp/memra-5090.lock
MIN_HOST_GB=${MIN_HOST_GB:-16}
HOLD_IDLE_S=${HOLD_IDLE_S:-60} HOLD_RETRY_S=${HOLD_RETRY_S:-120}
export HOLD_IDLE_S HOLD_RETRY_S MIN_HOST_GB
# shellcheck source=../rig-hold.sh
. research/spill-b-20260919/rig-hold.sh
rig_hold after $((SECONDS + 4 * 3600))
r=$?
[ $r = 0 ] || { log "after: card not held idle within 4 h; not run"; exit 3; }
WT=$WT RIGDIR=$R LOCK=none bash research/spill-b-20260919/run-day28-cell.sh after "$BIN" > "$R/after.launch.log" 2>&1 8>&-
rc=$?
rig_release after
log "cell rc=$rc"
log "DAY51-LOCAL-DONE"
