#!/usr/bin/env bash
# WP-B target-card sitting (2026-09-24): DAY37 (O1), DAY38 (O2) and DAY39 (O5) halves on one RTX PRO 6000 Blackwell
# Workstation Edition, in that order, each chain skipped once its DONE line is in its own chain.log. Run from the lane
# checkout /root/wt-b (fast-forwarded to the lane tip first). DRY_RUN=1 runs each chain's checks only.
set -uo pipefail
cd /root/wt-b || exit 1
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD || { echo "fetch/ff failed"; exit 1; }
D=research/spill-b-20260919
for day in 37 38 39; do
  log=/root/spill-receipts/b-day$day/chain.log
  if [ -f "$log" ] && command grep -q "LANE-B-DAY$day-BOX-DONE" "$log"; then echo "day $day done already"; continue; fi
  echo "$(date -u +%FT%TZ) day $day start"
  bash "$D/pro-single-day$day/chain.sh"; rc=$?
  echo "$(date -u +%FT%TZ) day $day rc=$rc"
  [ "${DRY_RUN:-0}" = 1 ] || [ $rc = 0 ] || exit $rc
done
echo "$(date -u +%FT%TZ) LANE-B-SITTING-DONE"
