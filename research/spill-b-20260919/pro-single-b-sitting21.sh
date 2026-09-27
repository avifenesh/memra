#!/usr/bin/env bash
# WP-B twenty-first target-card sitting (2026-09-27): DAY51 (memra#476: the DAY28 walk after the boot pre-grow, on 11ee6ed77; the 27B
# Qwen3.8-27B NVFP4 MTP), on one RTX PRO 6000 Blackwell Workstation Edition; skipped once its
# DONE line is in its chain.log. Refuses to start without ss or lsof. Run from /root/wt-b. DRY_RUN=1 runs the checks.
set -uo pipefail
cd /root/wt-b || exit 1
command -v ss >/dev/null || command -v lsof >/dev/null || { echo "neither ss nor lsof on PATH (install iproute2); not run"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD || { echo "fetch/ff failed"; exit 1; }
log=/root/spill-receipts/b-day51/chain.log
if [ -f "$log" ] && command grep -q "LANE-B-DAY51-BOX-DONE" "$log"; then echo "day 51 done already"; exit 0; fi
echo "$(date -u +%FT%TZ) day 51 start"
R51=/root/spill-receipts/b-day51 S51=11ee6ed778bafe9d1ac1be47f2993c6cd8fdc936 bash research/spill-b-20260919/pro-single-day51/chain.sh
rc=$?
echo "$(date -u +%FT%TZ) day 51 rc=$rc"
[ "${DRY_RUN:-0}" = 1 ] || [ $rc = 0 ] || exit $rc
echo "$(date -u +%FT%TZ) LANE-B-SITTING21-DONE"
