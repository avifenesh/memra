#!/usr/bin/env bash
# Runs on the rig: round 5 lever 1's GB10 cells on one unit, one sparklock lease per cell
# (first-come queue: a landing waiter goes first at every cell boundary).
# usage: CELLS="gate 01-B 02-T ..." eos_driver.sh UNIT CAMPAIGN BASE10 HEAD10  (UNIT: the unit's ssh name and sparklock unit)
set -u
UNIT=$1; CAMP=$2; B=$3; T=$4
D=$HOME/.cache/memra-gb10-r4
L=$D/eos-driver-$UNIT.log
for c in ${CELLS:-gate 01-B 02-T 03-T 04-B 05-B 06-T}; do
  id=$(sparklock acquire --unit "$UNIT" --ttl 15m --for "memra GB10 round 5 lever 1 (EOS ends the spec burst): $c (one cell per lease)" --wait 120m 2>>"$L")
  [ -n "$id" ] || { echo "$(date -Is) $c: no lease, stopping" >> "$L"; exit 1; }
  echo "$(date -Is) $c: lease $id" >> "$L"
  /usr/bin/ssh -n "$UNIT" "(cd ~/.cache/memra-gb10-r4 && nohup bash box/eos_cell.sh $CAMP $c $B $T) > ~/.cache/memra-gb10-r4/eos_cell_$c.out 2>&1 < /dev/null & echo started" >> "$L" 2>&1
  waitfor --ssh "$UNIT" "grep -q 'EOS-CELL-DONE $c\|EOS-CELL-FAIL $c' ~/.cache/memra-gb10-r4/runs/$CAMP/campaign.log" --timeout 14m --every 15s --quiet >> "$L" 2>&1
  sparklock release "$id" --unit "$UNIT" >> "$L" 2>&1
  echo "$(date -Is) $c: released" >> "$L"
done
echo "$(date -Is) all cells done" >> "$L"
