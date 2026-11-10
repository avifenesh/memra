#!/usr/bin/env bash
# Runs on the rig: round 5 lever 2's GB10 cells on one unit, one sparklock lease per cell (the
# fair queue puts a landing waiter first at every cell boundary). A failed cell stops the driver.
# usage: CELLS="01-B 02-T ..." rows_driver.sh UNIT CAMPAIGN BASE_DIR HEAD_DIR SET
set -u
UNIT=$1; CAMP=$2; B=$3; T=$4; SET=$5
D=$HOME/.cache/memra-gb10-r4
L=$D/rows-driver-$CAMP.log
for c in ${CELLS:-01-B 02-T 03-T 04-B 05-B 06-T}; do
  id=$(sparklock acquire --unit "$UNIT" --ttl 20m --for "memra GB10 round 5 lever 2 (verify rows kernel) served B/T, $SET set: $c (one cell per lease)" --wait 180m 2>>"$L")
  [ -n "$id" ] || { echo "$(date -Is) $c: no lease, stopping" >> "$L"; exit 1; }
  echo "$(date -Is) $c: lease $id" >> "$L"
  /usr/bin/ssh -n "spark-$UNIT" "(cd ~/.cache/memra-gb10-r4 && nohup bash box/rows_cell.sh $CAMP $c $B $T $SET) > ~/.cache/memra-gb10-r4/rows_cell_$c.out 2>&1 < /dev/null & echo started" >> "$L" 2>&1
  waitfor --ssh "spark-$UNIT" "grep -q 'EOS-CELL-DONE $c\|EOS-CELL-FAIL $c' ~/.cache/memra-gb10-r4/runs/$CAMP/campaign.log" --timeout 19m --every 15s --quiet >> "$L" 2>&1
  sparklock release "$id" --unit "$UNIT" >> "$L" 2>&1
  echo "$(date -Is) $c: released" >> "$L"
  /usr/bin/ssh -n "spark-$UNIT" "grep -q 'EOS-CELL-DONE $c' ~/.cache/memra-gb10-r4/runs/$CAMP/campaign.log" || { echo "$(date -Is) $c: FAILED, stopping" >> "$L"; exit 1; }
done
echo "$(date -Is) all cells done" >> "$L"
