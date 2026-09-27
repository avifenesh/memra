#!/usr/bin/env bash
# WP-B tenth target-card sitting (2026-09-26): DAY45 (O4, the outstanding-only W release), on one RTX PRO
# 6000 Blackwell Workstation Edition; skipped once its DONE line is in its chain.log. Refuses to start without ss or lsof.
# Run from /root/wt-b. DRY_RUN=1 runs the checks.
set -uo pipefail
cd /root/wt-b || exit 1
command -v ss >/dev/null || command -v lsof >/dev/null || { echo "neither ss nor lsof on PATH (install iproute2); not run"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD || { echo "fetch/ff failed"; exit 1; }
log=/root/spill-receipts/b-day45/chain.log
if [ -f "$log" ] && command grep -q "LANE-B-DAY45-BOX-DONE" "$log"; then echo "day 45 done already"; exit 0; fi
echo "$(date -u +%FT%TZ) day 45 start"
bash research/spill-b-20260919/pro-single-day45/chain.sh; rc=$?
echo "$(date -u +%FT%TZ) day 45 rc=$rc"
[ "${DRY_RUN:-0}" = 1 ] || [ $rc = 0 ] || exit $rc
echo "$(date -u +%FT%TZ) LANE-B-SITTING10-DONE"
