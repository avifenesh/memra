#!/usr/bin/env bash
# Day 16 review cells on the target card (PR #605 findings 1 and 2): the eight GPU unit cells of both routes, the
# six-cell contract fault gate, and the host-spill identity gate OFF then ON under the default spec env on the
# review binary. MEMRA_HOSTGATE_CACHE_MB=256.
set -uo pipefail
R=/root/spill-receipts/c-day16-review
cd $R
[ "$(cat $R/build/exit)" = 0 ] || { echo "build failed, no cells"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,power.max_limit,driver_version --format=csv > $R/card.csv
log() { echo "$(date -u +%FT%TZ) $*" >> $R/driver.log; }
run() { local name=$1; shift; log "$name start"; bash $R/run-cell.sh "$name" "$@"; local rc=$?; log "$name rc=$rc"; }
log "start binary $(sha256sum $R/bins/memra-server | cut -c1-16) source $(cat $R/build/source.txt)"
run gputests 3000 1 -- bash $R/gputests-cell.sh @COLLECTOR_LOCK_FD@
run faultgate 3000 1 -- bash $R/faultgate-cell.sh @COLLECTOR_LOCK_FD@ 256
for arm in off on; do
  run hostgate-identity-$arm-default 1500 1 -- bash $R/hostgate-cell.sh identity $arm default @COLLECTOR_LOCK_FD@ 256
done
log "done"
