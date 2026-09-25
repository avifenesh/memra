#!/usr/bin/env bash
# WP-B local 5090 queue-g (2026-09-26): after queue-f (pid argument) has exited, builds target/day41b (the revised
# grid-rewind tip and its offprev arm) and runs the DAY41 addenda B and C local chain. Reads /proc only; never a signal.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/rtx5090-queue-g.log"; }
pid=${1:?queue-f pid}
log "queue-g start: waiting for queue-f pid $pid"
while [ -d "/proc/$pid" ]; do sleep 120; done
cd "$WT" || exit 1
if [ ! -s target/day41b/SHA256SUMS ]; then
  export WT; TARGET=$WT/target WRAP="systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G nice -n 10" \
    bash "$D/build-arms.sh" "$WT/target/day41b" 23c296b583d0207413d4a2f3347882e729b4b29d tip offprev:day41b-nodoor.patch \
    > "$D/rtx5090-day41b/build.out" 2>&1
  log "day 41b build rc=$?"
fi
bash "$D/rtx5090-day41b/chain.sh"; log "day 41b chain rc=$?"
log "queue-g done"
