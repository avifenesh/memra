#!/usr/bin/env bash
# Day 16 cell sequence on the target card (Option C): the GPU unit cells of both contract routes, the four-cell
# contract fault gate, then door OFF then ON per gate on the same binary and prompts (default spec env for the
# host-spill gates, lane A's reclaim fix arm, serve-smoke, the plain identity pair, lane B's two prefix gates),
# then the WC pair cell (one lock hold, both orders). MEMRA_HOSTGATE_CACHE_MB=256.
set -uo pipefail
R=/root/spill-receipts/c-day16
cd $R
[ "$(cat $R/build/exit)" = 0 ] || { echo "build failed, no cells"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,power.max_limit,driver_version --format=csv > $R/card.csv
log() { echo "$(date -u +%FT%TZ) $*" >> $R/driver.log; }
run() { local name=$1; shift; log "$name start"; bash $R/run-cell.sh "$name" "$@"; local rc=$?; log "$name rc=$rc"; }
log "start binary $(sha256sum $R/bins/memra-server | cut -c1-16) source $(cat $R/build/source.txt)"
run gputests 3000 1 -- bash $R/gputests-cell.sh @COLLECTOR_LOCK_FD@
run faultgate 2400 1 -- bash $R/faultgate-cell.sh @COLLECTOR_LOCK_FD@ 256
for arm in off on; do
  run hostgate-identity-$arm-default 1500 1 -- bash $R/hostgate-cell.sh identity $arm default @COLLECTOR_LOCK_FD@ 256
done
for arm in off on; do
  run hostgate-failure-$arm-default 1500 1 -- bash $R/hostgate-cell.sh failure $arm default @COLLECTOR_LOCK_FD@ 256
done
for arm in off on; do
  run tenant-$arm 1800 1 -- bash $R/tenant-cell.sh $arm @COLLECTOR_LOCK_FD@
done
run smoke-off 1800 0 -- bash $R/smoke-cell.sh off
run smoke-on 1800 0 -- bash $R/smoke-cell.sh on
for arm in off on; do
  run hostgate-identity-$arm-plain 1500 1 -- bash $R/hostgate-cell.sh identity $arm plain @COLLECTOR_LOCK_FD@ 256
done
for gate in evict newest; do
  for arm in off on; do
    run b$gate-$arm 1800 1 -- bash $R/bgate-cell.sh $gate $arm @COLLECTOR_LOCK_FD@
  done
done
run wc-pair 2400 1 -- bash $R/wc-cell.sh @COLLECTOR_LOCK_FD@
log "done"
