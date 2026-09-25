#!/usr/bin/env bash
# WP-B third target-card sitting (2026-09-25): DAY40 (O3, day 36's decision cell on the final booking) then DAY41 (O11,
# the grid-checkpoint rewind arm's price), in decide-by order, on one RTX PRO 6000 Blackwell Workstation Edition; each
# chain skipped once its DONE line is in its chain.log. Refuses to start without ss or lsof. Run from /root/wt-b.
# DRY_RUN=1 runs each chain's checks only.
set -uo pipefail
cd /root/wt-b || exit 1
command -v ss >/dev/null || command -v lsof >/dev/null || { echo "neither ss nor lsof on PATH (install iproute2); not run"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD || { echo "fetch/ff failed"; exit 1; }
D=research/spill-b-20260919
for step in "40:pro-single-day40/chain.sh:LANE-B-DAY40-BOX-DONE" "41:pro-single-day41/chain.sh:LANE-B-DAY41-BOX-DONE"; do
  IFS=: read -r day script done_line <<< "$step"
  log=/root/spill-receipts/b-day$day/chain.log
  if [ -f "$log" ] && command grep -q "$done_line" "$log"; then echo "day $day done already"; continue; fi
  echo "$(date -u +%FT%TZ) day $day start"
  bash "$D/$script"; rc=$?
  echo "$(date -u +%FT%TZ) day $day rc=$rc"
  [ "${DRY_RUN:-0}" = 1 ] || [ $rc = 0 ] || exit $rc
done
echo "$(date -u +%FT%TZ) LANE-B-SITTING3-DONE"
