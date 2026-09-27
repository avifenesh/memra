#!/usr/bin/env bash
# WP-B nineteenth target-card sitting (2026-09-27): DAY48 addendum C (O8 on the fix 521fdbbbc, the peek; V5 one debt per admission; on
# Ornith-1.5-35B-A3B NVFP4 MTP), on one RTX PRO 6000 Blackwell Workstation Edition; skipped once its
# DONE line is in its chain.log. Refuses to start without ss or lsof. Run from /root/wt-b. DRY_RUN=1 runs the checks.
set -uo pipefail
cd /root/wt-b || exit 1
command -v ss >/dev/null || command -v lsof >/dev/null || { echo "neither ss nor lsof on PATH (install iproute2); not run"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD || { echo "fetch/ff failed"; exit 1; }
log=/root/spill-receipts/b-day48b/chain.log
if [ -f "$log" ] && command grep -q "LANE-B-DAY48-BOX-DONE" "$log"; then echo "day 48b done already"; exit 0; fi
echo "$(date -u +%FT%TZ) day 48b start"
R48=/root/spill-receipts/b-day48b S48=521fdbbbce474de19f0e55be97c7539bced70f9d bash research/spill-b-20260919/pro-single-day48/chain.sh
rc=$?
echo "$(date -u +%FT%TZ) day 48b rc=$rc"
[ "${DRY_RUN:-0}" = 1 ] || [ $rc = 0 ] || exit $rc
echo "$(date -u +%FT%TZ) LANE-B-SITTING19-DONE"
