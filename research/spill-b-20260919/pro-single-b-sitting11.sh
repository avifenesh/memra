#!/usr/bin/env bash
# WP-B eleventh target-card sitting (2026-09-26): DAY45 addendum B (O4), then DAY47 (O7, the health gate's arms g and h), on one RTX PRO
# 6000 Blackwell Workstation Edition; skipped once its DONE line is in its chain.log. Refuses to start without ss or lsof.
# Run from /root/wt-b. DRY_RUN=1 runs the checks.
set -uo pipefail
cd /root/wt-b || exit 1
command -v ss >/dev/null || command -v lsof >/dev/null || { echo "neither ss nor lsof on PATH (install iproute2); not run"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD || { echo "fetch/ff failed"; exit 1; }
log=/root/spill-receipts/b-day45b/chain.log
if [ -f "$log" ] && command grep -q "LANE-B-DAY45B-BOX-DONE" "$log"; then echo "day 45b done already"; else
  echo "$(date -u +%FT%TZ) day 45b start"
  bash research/spill-b-20260919/pro-single-day45/chain-b.sh; rc=$?
  echo "$(date -u +%FT%TZ) day 45b rc=$rc"
  [ "${DRY_RUN:-0}" = 1 ] || [ $rc = 0 ] || exit $rc
fi
log47=/root/spill-receipts/b-day47/chain.log
if [ -f "$log47" ] && command grep -q "LANE-B-DAY47-BOX-DONE" "$log47"; then echo "day 47 done already"; else
  echo "$(date -u +%FT%TZ) day 47 start"
  bash research/spill-b-20260919/pro-single-day47/chain.sh; rc=$?
  echo "$(date -u +%FT%TZ) day 47 rc=$rc"
  [ "${DRY_RUN:-0}" = 1 ] || [ $rc = 0 ] || exit $rc
fi
echo "$(date -u +%FT%TZ) LANE-B-SITTING11-DONE"
