#!/usr/bin/env bash
# Day 14 cell sequence on the target card: door OFF then ON per gate, same binary, same prompts.
# The default spec environment is the exit criterion (lead ruling 16); the plain arms are the
# day-13 regression. MEMRA_HOSTGATE_CACHE_MB=256 (day-13 finding 1: one ~160 MB entry fits, two do not).
set -uo pipefail
R=/root/spill-receipts/c-day14
cd $R
[ "$(cat $R/build/exit)" = 0 ] || { echo "build failed, no cells"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,power.max_limit,driver_version --format=csv > $R/card.csv
log() { echo "$(date -u +%FT%TZ) $*" >> $R/driver.log; }
run() { local name=$1; shift; log "$name start"; bash $R/run-cell.sh "$name" "$@"; local rc=$?; log "$name rc=$rc"; }
log "start binary $(sha256sum $R/bins/memra-server | cut -c1-16) source $(cat $R/build/source.txt)"
run smoke-off 1800 0 -- bash $R/smoke-cell.sh off
run smoke-on 1800 0 -- bash $R/smoke-cell.sh on
for spec in default plain; do
  for arm in off on; do
    run hostgate-identity-$arm-$spec 1500 1 -- bash $R/hostgate-cell.sh identity $arm $spec @COLLECTOR_LOCK_FD@ 256
  done
done
for spec in default plain; do
  for arm in off on; do
    run hostgate-failure-$arm-$spec 1500 1 -- bash $R/hostgate-cell.sh failure $arm $spec @COLLECTOR_LOCK_FD@ 256
  done
done
log "done"
