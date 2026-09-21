#!/usr/bin/env bash
# Waits for the day-16 driver to finish, moves /root/wt-c to the gate-fix tip (tools only, same binary), then
# runs the fault gate once more through the collector. Never touches a running cell.
set -uo pipefail
R=/root/spill-receipts/c-day16
log() { echo "$(date -u +%FT%TZ) $*" >> $R/driver.log; }
for _ in $(seq 1 720); do grep -q " done$" $R/driver.log 2>/dev/null && break; sleep 10; done
grep -q " done$" $R/driver.log || { log "fix-waiter: driver not done after 2 h, giving up"; exit 3; }
git -C /root/wt-c fetch -q $R/c16b.bundle HEAD:refs/bundle/c16b && git -C /root/wt-c checkout -q --detach refs/bundle/c16b
log "fix-waiter: checkout $(git -C /root/wt-c rev-parse --short HEAD) (gate matcher fix, same binary)"
log "faultgate-fix start"; bash $R/run-cell.sh faultgate-fix 2400 1 -- bash $R/faultgate-fix-cell.sh @COLLECTOR_LOCK_FD@ 256; log "faultgate-fix rc=$?"
log "fix-waiter done"
