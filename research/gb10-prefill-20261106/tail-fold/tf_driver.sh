#!/usr/bin/env bash
# Runs on the rig: lever 8 (the boundary tail fold) on one Spark unit, one sparklock lease per step:
# the build, the prime-fold-gate, then the served A/B B T T B B T. Stops when the unit is not free,
# when the build or the gate fails, or at a step boundary when the file tf-stop-UNIT exists.
# usage: tf_driver.sh UNIT HOST CAMPAIGN B_SHA T_SHA   (STEPS="..." runs a subset)
set -u
UNIT=$1; HOST=$2; CAMP=$3; BS=$4; TS=$5
D=$HOME/.cache/memra-gb10-r4
L=$D/tf-driver-$UNIT.log
C=runs/$CAMP/campaign.log
for step in ${STEPS:-build gate 01-B 02-T 03-T 04-B 05-B 06-T}; do
  if [ -e "$D/tf-stop-$UNIT" ]; then echo "$(date -Is) $step: stop file, stopping" >> "$L"; exit 0; fi
  ttl=25m; tmo=24m; [ "$step" = build ] && { ttl=40m; tmo=39m; }; [ "$step" = gate ] && { ttl=15m; tmo=14m; }
  id=$(sparklock acquire --unit "$UNIT" --ttl "$ttl" --for "memra GB10 prefill round 4 lever 8 (tail fold): $step (one step per lease)" --wait 3m 2>>"$L")
  if [ -z "$id" ]; then echo "$(date -Is) $step: $UNIT not free, stopping" >> "$L"; exit 0; fi
  echo "$(date -Is) $step: lease $id" >> "$L"
  /usr/bin/ssh -n "$HOST" "(cd ~/.cache/memra-gb10-r4 && nohup bash box/tf_cell.sh $CAMP $step $BS $TS) > ~/.cache/memra-gb10-r4/tf_cell_$step.out 2>&1 < /dev/null & echo started" >> "$L" 2>&1
  waitfor --ssh "$HOST" "grep -q 'TF-CELL-DONE $step\|TF-CELL-BUILD-FAIL' ~/.cache/memra-gb10-r4/$C" --timeout "$tmo" --every 30s --quiet >> "$L" 2>&1
  sparklock release "$id" --unit "$UNIT" >> "$L" 2>&1
  echo "$(date -Is) $step: released" >> "$L"
  /usr/bin/ssh -n "$HOST" "grep -q TF-CELL-BUILD-FAIL ~/.cache/memra-gb10-r4/$C" && { echo "build failed" >> "$L"; exit 1; }
  if [ "$step" = gate ] && ! /usr/bin/ssh -n "$HOST" "grep -q 'prime-fold-gate rc=0' ~/.cache/memra-gb10-r4/$C"; then
    echo "$(date -Is) prime-fold-gate did not pass, stopping" >> "$L"; exit 1
  fi
done
echo "$(date -Is) all steps done" >> "$L"
