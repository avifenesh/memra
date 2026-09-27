#!/usr/bin/env bash
# Day 13 cell sequence on the target card, door OFF then ON per gate, same binary, same prompts.
set -uo pipefail
R=/root/spill-receipts/c-day13
cd $R
[ "$(cat $R/build/exit)" = 0 ] || { echo "build failed, no cells"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,power.max_limit,driver_version --format=csv > $R/card.csv
echo "$(date -u +%FT%TZ) start" >> $R/driver.log
run() { local name=$1; shift; echo "$(date -u +%FT%TZ) $name start" >> $R/driver.log; bash $R/run-cell.sh "$name" "$@"; echo "$(date -u +%FT%TZ) $name rc=$?" >> $R/driver.log; }
run smoke-off 1800 0 -- bash $R/smoke-cell.sh off
run smoke-on 1800 0 -- bash $R/smoke-cell.sh on
for spec in default plain; do
  for arm in off on; do
    run hostgate-identity-$arm-$spec 1500 1 -- bash $R/hostgate-cell.sh identity $arm $spec @COLLECTOR_LOCK_FD@ 128
  done
done
for spec in default plain; do
  for arm in off on; do
    run hostgate-failure-$arm-$spec 1500 1 -- bash $R/hostgate-cell.sh failure $arm $spec @COLLECTOR_LOCK_FD@ 128
  done
done
for arm in off on; do
  run reclaim-$arm 2400 1 -- bash $R/reclaim-cell.sh $arm @COLLECTOR_LOCK_FD@
done
echo "$(date -u +%FT%TZ) done" >> $R/driver.log
