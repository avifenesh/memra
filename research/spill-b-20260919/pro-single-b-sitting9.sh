#!/usr/bin/env bash
# WP-B ninth target-card sitting (2026-09-26): DAY44 (O11 revised, the exact and fast resume), on one RTX PRO
# 6000 Blackwell Workstation Edition; skipped once its DONE line is in its chain.log. Refuses to start without ss or lsof.
# Run from /root/wt-b. DRY_RUN=1 runs the checks.
set -uo pipefail
cd /root/wt-b || exit 1
command -v ss >/dev/null || command -v lsof >/dev/null || { echo "neither ss nor lsof on PATH (install iproute2); not run"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD || { echo "fetch/ff failed"; exit 1; }
log=/root/spill-receipts/b-day44/chain.log
if [ -f "$log" ] && command grep -q "LANE-B-DAY44-BOX-DONE" "$log"; then echo "day 44 done already"; exit 0; fi
echo "$(date -u +%FT%TZ) day 44 start"
bash research/spill-b-20260919/pro-single-day44/chain.sh; rc=$?
echo "$(date -u +%FT%TZ) day 44 rc=$rc"
[ "${DRY_RUN:-0}" = 1 ] || [ $rc = 0 ] || exit $rc
echo "$(date -u +%FT%TZ) LANE-B-SITTING9-DONE"
