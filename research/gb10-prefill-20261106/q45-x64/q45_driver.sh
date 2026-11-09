#!/usr/bin/env bash
# Runs on the rig: the q45k MMQ_X=64 probe on b23d, one lease per step; yields to any waiter.
set -u
D=$HOME/.cache/memra-gb10-r4
L=$D/q45-driver.log
for step in ${STEPS:-build 01-B 02-T 03-T 04-B 05-B 06-T}; do
  ttl=12m; tmo=11m; [ "$step" = build ] && { ttl=40m; tmo=39m; }
  id=$(sparklock acquire --unit b23d --ttl "$ttl" --for "memra GB10 prefill round 4 last probe (q45k MMQ_X=64): $step (one step per lease)" --wait 3m 2>>"$L")
  [ -n "$id" ] || { echo "$(date -Is) $step: b23d not free, stopping" >> "$L"; exit 0; }
  echo "$(date -Is) $step: lease $id" >> "$L"
  /usr/bin/ssh -n spark-b23d "(cd ~/.cache/memra-gb10-r4 && nohup bash box/q45_cell.sh q45-x64 $step) > ~/.cache/memra-gb10-r4/q45_$step.out 2>&1 < /dev/null & echo started" >> "$L" 2>&1
  waitfor --ssh spark-b23d "grep -q 'Q45-CELL-DONE $step\|Q45-CELL-FAIL' ~/.cache/memra-gb10-r4/runs/q45-x64/campaign.log" --timeout "$tmo" --every 20s --quiet >> "$L" 2>&1
  sparklock release "$id" --unit b23d >> "$L" 2>&1
  echo "$(date -Is) $step: released" >> "$L"
  /usr/bin/ssh -n spark-b23d "grep -q Q45-CELL-FAIL ~/.cache/memra-gb10-r4/runs/q45-x64/campaign.log" && { echo "failed" >> "$L"; exit 1; }
done
echo "$(date -Is) all steps done" >> "$L"
