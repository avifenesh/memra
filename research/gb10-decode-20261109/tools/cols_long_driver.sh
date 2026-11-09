#!/usr/bin/env bash
# Runs on the rig: round 5 lever 1's GB10 cells on one unit, one sparklock lease per cell
# (first-come queue: a landing waiter goes first at every cell boundary).
# usage: CELLS="gate 01-B 02-T ..." eos_driver.sh UNIT CAMPAIGN BASE10 HEAD10
set -u
UNIT=$1; CAMP=$2; B=$3; T=$4
D=$HOME/.cache/memra-gb10-r4
L=$D/cols-long-driver-$UNIT.log
for c in ${CELLS:-01-B 02-T 03-T 04-B 05-B 06-T}; do
  # b23d is lent to another session from 22:30: no cell starts after 22:12.
  [ "$(date +%H%M)" -gt 2212 ] && { echo "$(date -Is) $c: past 22:12, stopping" >> "$L"; exit 0; }
  id=$(sparklock acquire --unit "$UNIT" --ttl 15m --for "memra GB10 round 5 lever 3a (verify column states in the scan slabs) with the 131k request: $c (one cell per lease)" --wait 3m 2>>"$L")
  [ -n "$id" ] || { echo "$(date -Is) $c: no lease, stopping" >> "$L"; exit 1; }
  echo "$(date -Is) $c: lease $id" >> "$L"
  /usr/bin/ssh -n "spark-$UNIT" "(cd ~/.cache/memra-gb10-r4 && nohup bash box/cols_cell_long.sh $CAMP $c $B $T) > ~/.cache/memra-gb10-r4/cols_long_cell_$c.out 2>&1 < /dev/null & echo started" >> "$L" 2>&1
  waitfor --ssh "spark-$UNIT" "grep -q 'EOS-CELL-DONE $c\|EOS-CELL-FAIL $c' ~/.cache/memra-gb10-r4/runs/$CAMP/campaign.log" --timeout 14m --every 15s --quiet >> "$L" 2>&1
  sparklock release "$id" --unit "$UNIT" >> "$L" 2>&1
  echo "$(date -Is) $c: released" >> "$L"
done
echo "$(date -Is) all cells done" >> "$L"
