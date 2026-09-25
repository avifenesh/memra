#!/usr/bin/env bash
# WP-B sixth target-card sitting (2026-09-25): DAY42 in addenda C to F's shape (the worker-level demote queue; O12 and O10), on one
# RTX PRO 6000 Blackwell Workstation Edition with at least 96 GB of host RAM (a 32 GB pinned host tier); skipped once
# its DONE line is in its chain.log. Refuses to start without ss or lsof. Run from /root/wt-b. DRY_RUN=1 runs the checks.
set -uo pipefail
cd /root/wt-b || exit 1
command -v ss >/dev/null || command -v lsof >/dev/null || { echo "neither ss nor lsof on PATH (install iproute2); not run"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD || { echo "fetch/ff failed"; exit 1; }
log=/root/spill-receipts/b-day42e/chain.log
if [ -f "$log" ] && command grep -q "LANE-B-DAY42E-BOX-DONE" "$log"; then echo "day 42e done already"; exit 0; fi
echo "$(date -u +%FT%TZ) day 42e start"
bash research/spill-b-20260919/pro-single-day42/chain-e.sh; rc=$?
echo "$(date -u +%FT%TZ) day 42e rc=$rc"
[ "${DRY_RUN:-0}" = 1 ] || [ $rc = 0 ] || exit $rc
echo "$(date -u +%FT%TZ) LANE-B-SITTING6-DONE"
