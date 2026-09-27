#!/usr/bin/env bash
# WP-B second target-card sitting (2026-09-25): DAY37 addendum F's rerun, DAY38 addendum D, DAY39 addendum B, in that
# order, on one RTX PRO 6000 Blackwell Workstation Edition; each chain skipped once its DONE line is in its chain.log.
# Refuses to start without ss or lsof (the gates' port check). Run from /root/wt-b. DRY_RUN=1 runs each chain's checks.
set -uo pipefail
cd /root/wt-b || exit 1
command -v ss >/dev/null || command -v lsof >/dev/null || { echo "neither ss nor lsof on PATH (install iproute2); not run"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD || { echo "fetch/ff failed"; exit 1; }
D=research/spill-b-20260919
for step in "37f:pro-single-day37/rerun-f.sh:LANE-B-DAY37F-BOX-DONE" "38d:pro-single-day38/chain-d.sh:LANE-B-DAY38D-BOX-DONE" \
            "39b:pro-single-day39/chain-b.sh:LANE-B-DAY39B-BOX-DONE"; do
  IFS=: read -r day script done_line <<< "$step"
  log=/root/spill-receipts/b-day$day/chain.log
  if [ -f "$log" ] && command grep -q "$done_line" "$log"; then echo "day $day done already"; continue; fi
  echo "$(date -u +%FT%TZ) day $day start"
  bash "$D/$script"; rc=$?
  echo "$(date -u +%FT%TZ) day $day rc=$rc"
  [ "${DRY_RUN:-0}" = 1 ] || [ $rc = 0 ] || exit $rc
done
echo "$(date -u +%FT%TZ) LANE-B-SITTING2-DONE"
