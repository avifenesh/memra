#!/usr/bin/env bash
# Runs on the rig: the round 5 re-scout cells (after levers 1, 3a and 2) on one GB10 unit, one
# sparklock lease per cell; the fair queue puts a landing waiter first at every cell boundary.
# usage: CELLS="decode nsys1k nsys131k nsys260k k4 k5 k3 k6 k2" r6_driver.sh UNIT BIN_DIR
set -u
UNIT=$1; BIN=$2
D=$HOME/.cache/memra-gb10-r4
L=$D/r6-driver-$UNIT.log
for c in ${CELLS:-decode nsys1k nsys131k nsys260k}; do
  id=$(sparklock acquire --unit "$UNIT" --ttl 25m --for "memra GB10 round 5 re-scout (single-user decode after levers 1, 3a, 2): $c (one cell per lease)" --wait 180m 2>>"$L")
  [ -n "$id" ] || { echo "$(date -Is) $c: no lease, stopping" >> "$L"; exit 1; }
  echo "$(date -Is) $c: lease $id" >> "$L"
  /usr/bin/ssh -n "spark-$UNIT" "(cd ~/.cache/memra-gb10-r4 && nohup bash box/r6_cell.sh $c $BIN) > ~/.cache/memra-gb10-r4/r6_$c.out 2>&1 < /dev/null & echo started" >> "$L" 2>&1
  waitfor --ssh "spark-$UNIT" "grep -q 'R6-DONE $c\|R6-FAIL' ~/.cache/memra-gb10-r4/r6_$c.out" --timeout 24m --every 20s --quiet >> "$L" 2>&1
  sparklock release "$id" --unit "$UNIT" >> "$L" 2>&1
  echo "$(date -Is) $c: released" >> "$L"
done
echo "$(date -Is) all cells done" >> "$L"
