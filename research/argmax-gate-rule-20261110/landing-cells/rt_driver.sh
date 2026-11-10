#!/usr/bin/env bash
# Runs on the driving machine: one arm of lever 1's real-text served-batch margin cell on one GB10
# unit, under one sparklock lease. usage: rt_driver.sh UNIT T|B
set -u
UNIT=$1; ARM=$2
L=$HOME/.cache/memra-gb10-r5/rt/driver-$UNIT-$ARM.log
id=$(sparklock acquire --unit "$UNIT" --ttl 50m --for "memra landing #1188: lever 1's 27B real-text served-batch margin cell, arm $ARM (pass/fail, one cell per lease)" --wait 90m 2>>"$L")
[ -n "$id" ] || { echo "$(date -Is) no lease" >> "$L"; exit 1; }
echo "$(date -Is) lease $id" >> "$L"
/usr/bin/ssh -n "spark-$UNIT" "(cd ~/.cache/memra-split-rung && nohup bash rt/rt_cell.sh $ARM) > ~/.cache/memra-split-rung/rt/rt_$ARM.out 2>&1 < /dev/null & echo started" >> "$L" 2>&1
waitfor --ssh "spark-$UNIT" "grep -q 'RT-DONE\|RT-FAIL' ~/.cache/memra-split-rung/rt/rt_$ARM.out" --timeout 48m --every 20s --quiet >> "$L" 2>&1
sparklock release "$id" --unit "$UNIT" >> "$L" 2>&1
echo "$(date -Is) released" >> "$L"
