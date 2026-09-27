#!/usr/bin/env bash
# WP-B fifteenth target-card sitting (2026-09-27): DAY46 (O6, the enforcing predictive door on the fuller charge, with
# DAY45's W release as the second enforcing arm), on one RTX PRO 6000 Blackwell Workstation Edition; skipped once its
# DONE line is in its chain.log. Refuses to start without ss or lsof. Run from /root/wt-b. DRY_RUN=1 runs the checks.
set -uo pipefail
cd /root/wt-b || exit 1
command -v ss >/dev/null || command -v lsof >/dev/null || { echo "neither ss nor lsof on PATH (install iproute2); not run"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD || { echo "fetch/ff failed"; exit 1; }
log=/root/spill-receipts/b-day46/chain.log
if [ -f "$log" ] && command grep -q "LANE-B-DAY46-BOX-DONE" "$log"; then echo "day 46 done already"; exit 0; fi
echo "$(date -u +%FT%TZ) day 46 start"
bash research/spill-b-20260919/pro-single-day46/chain.sh
rc=$?
echo "$(date -u +%FT%TZ) day 46 rc=$rc"
[ "${DRY_RUN:-0}" = 1 ] || [ $rc = 0 ] || exit $rc
echo "$(date -u +%FT%TZ) LANE-B-SITTING15-DONE"
