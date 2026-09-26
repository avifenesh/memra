#!/usr/bin/env bash
# WP-B local 5090 queue-j (2026-09-26): after queue-i (pid argument) has exited, the DAY44 local chain on target/day44
# (built by the lane before this queue started; built here at nice 19 under 600% if absent). Reads /proc only; never a
# signal.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/rtx5090-queue-j.log"; }
pid=${1:?queue-i pid}
log "queue-j start: waiting for queue-i pid $pid"
while [ -d "/proc/$pid" ]; do sleep 120; done
cd "$WT" || exit 1
if [ ! -s target/day44/SHA256SUMS ]; then
  echo "$(date -u +%FT%TZ) WP-B queue-j build target/day44 (nice 19, CPUQuota=600%)" >> "$D/cpu-concurrency.log"
  export WT; TARGET=$WT/target WRAP="systemd-run --user --scope -q -p CPUQuota=600% -p MemoryMax=20G nice -n 19" \
    bash "$D/build-arms.sh" "$WT/target/day44" 138790651415cc55a093224e92752ee045861f39 tip offprev:day44-nodoor.patch \
    > "$D/rtx5090-day44/build.out" 2>&1
  log "day 44 build rc=$?"
fi
bash "$D/rtx5090-day44/chain.sh"; log "day 44 chain rc=$?"
log "queue-j done"
