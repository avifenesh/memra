#!/usr/bin/env bash
# Rig: the deep split rung's GB10 served cells on one unit, one sparklock lease per cell (a
# landing waiter goes first at every cell boundary, even behind an older ticket of ours). A failed cell
# stops the driver.
# usage: CELLS="01-B 02-T ..." split_driver.sh UNIT CAMPAIGN BASE_DIR HEAD_DIR SET
set -u
UNIT=$1; CAMP=$2; B=$3; T=$4; SET=$5
D=$HOME/.cache/memra-split-rung
L=$D/runs/split-driver-$CAMP.log
TTL=${TTL:-20}
for c in ${CELLS:-01-B 02-T 03-T 04-B 05-B 06-T}; do
  id=$(sparklock acquire --unit "$UNIT" --ttl "${TTL}m" --for "memra deep split rung served B/T, $SET set: $c (one cell per lease)" --wait 240m 2>>"$L")
  [ -n "$id" ] || { echo "$(date -Is) $c: no lease, stopping" >> "$L"; exit 1; }
  echo "$(date -Is) $c: lease $id" >> "$L"
  # a landing waiter goes first at a cell boundary, even behind a ticket of ours queued earlier
  while sparklock status --unit "$UNIT" | grep -i 'waiting:' | grep -qi 'landing'; do
    echo "$(date -Is) $c: a landing waiter is queued; yielding" >> "$L"
    sparklock release "$id" --unit "$UNIT" >> "$L" 2>&1
    waitfor --until "! sparklock status --unit $UNIT | grep -i 'waiting:' | grep -qi landing" --every 30s --timeout 90m --quiet >> "$L" 2>&1
    id=$(sparklock acquire --unit "$UNIT" --ttl "${TTL}m" --for "memra deep split rung served B/T, $SET set: $c (one cell per lease)" --wait 240m 2>>"$L")
    [ -n "$id" ] || { echo "$(date -Is) $c: no lease after yielding, stopping" >> "$L"; exit 1; }
    echo "$(date -Is) $c: lease $id" >> "$L"
  done
  /usr/bin/ssh -n "spark-$UNIT" "(cd ~/.cache/memra-split-rung && nohup bash box/split_cell.sh $CAMP $c $B $T $SET) > ~/.cache/memra-split-rung/split_cell_$c.out 2>&1 < /dev/null & echo started" >> "$L" 2>&1
  deadline=$(( $(date +%s) + TTL * 60 - 30 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    waitfor --ssh "spark-$UNIT" "grep -q 'SPLIT-CELL-DONE $c\|SPLIT-CELL-FAIL $c' ~/.cache/memra-split-rung/runs/$CAMP/campaign.log" --timeout 9m --every 15s --quiet >> "$L" 2>&1 && break
  done
  sparklock release "$id" --unit "$UNIT" >> "$L" 2>&1
  echo "$(date -Is) $c: released" >> "$L"
  /usr/bin/ssh -n "spark-$UNIT" "grep -q 'SPLIT-CELL-DONE $c' ~/.cache/memra-split-rung/runs/$CAMP/campaign.log" || { echo "$(date -Is) $c: FAILED, stopping" >> "$L"; exit 1; }
done
echo "$(date -Is) all cells done" >> "$L"
