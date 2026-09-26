#!/usr/bin/env bash
# WP-B local 5090 queue-h (2026-09-26): after queue-g (pid argument) has exited, builds target/day43 (the budget-clamp tip and its offprev arm) and runs the DAY43 local chain.
# Reads /proc only; never a signal.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
log() { echo "$(date -u +%FT%TZ) $*" >> "$D/rtx5090-queue-h.log"; }
pid=${1:?queue-g pid}
log "queue-h start: waiting for queue-g pid $pid"
while [ -d "/proc/$pid" ]; do sleep 120; done
cd "$WT" || exit 1
if [ ! -s target/day43/SHA256SUMS ]; then
  export WT; TARGET=$WT/target WRAP="systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G nice -n 10" \
    bash "$D/build-arms.sh" "$WT/target/day43" 87d9e00d16ea4436c48d561d99926c0d1677c0dd tip offprev:day43-nodoor.patch \
    > "$D/rtx5090-day43/build.out" 2>&1
  log "day 43 build rc=$?"
fi
bash "$D/rtx5090-day43/chain.sh"; log "day 43 chain rc=$?"
log "queue-h done"
