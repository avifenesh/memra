#!/usr/bin/env bash
# WP-B sixteenth target-card sitting (2026-09-27): DAY50 stage 0 (a short prime call's time, walls and an nsys trace at
# three contexts), on one RTX PRO 6000 Blackwell Workstation Edition; skipped once its DONE line is in its chain.log.
# Run from /root/wt-b. DRY_RUN=1 runs the checks. S50 is the probe's commit (set by the lane's report).
set -uo pipefail
cd /root/wt-b || exit 1
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD || { echo "fetch/ff failed"; exit 1; }
log=/root/spill-receipts/b-day50/chain.log
if [ -f "$log" ] && command grep -q "LANE-B-DAY50-S0-BOX-DONE" "$log"; then echo "day 50 stage 0 done already"; exit 0; fi
echo "$(date -u +%FT%TZ) day 50 stage 0 start"
S50=${S50:-$(cat research/spill-b-20260919/pro-single-day50/S50)} bash research/spill-b-20260919/pro-single-day50/chain.sh
rc=$?
echo "$(date -u +%FT%TZ) day 50 stage 0 rc=$rc"
[ "${DRY_RUN:-0}" = 1 ] || [ $rc = 0 ] || exit $rc
echo "$(date -u +%FT%TZ) LANE-B-SITTING16-DONE"
