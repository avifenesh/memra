#!/usr/bin/env bash
# WP-B fourteenth target-card sitting (2026-09-26): DAY49 addendum D (O14's batched-OOM recovery with the fault aimed at
# a multi-session batched chunk, arm j, its VMM pair, the serving shape on both routes), on one RTX PRO 6000 Blackwell
# Workstation Edition; skipped once its DONE line is in its chain.log. Refuses to start without ss or lsof. Run from
# /root/wt-b. DRY_RUN=1 runs the checks.
set -uo pipefail
cd /root/wt-b || exit 1
command -v ss >/dev/null || command -v lsof >/dev/null || { echo "neither ss nor lsof on PATH (install iproute2); not run"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD || { echo "fetch/ff failed"; exit 1; }
# BOOTS_ONLY=1: the serving boots alone (the fourteenth sitting's boots did not start), receipts b-day49d-boots.
if [ "${BOOTS_ONLY:-0}" = 1 ]; then log=/root/spill-receipts/b-day49d-boots/chain.log; done_line=LANE-B-DAY49D-BOOTS-DONE
else log=/root/spill-receipts/b-day49d/chain.log; done_line=LANE-B-DAY49D-BOX-DONE; fi
if [ -f "$log" ] && command grep -q "$done_line" "$log"; then echo "day 49d ($done_line) done already"; exit 0; fi
echo "$(date -u +%FT%TZ) day 49d start"
bash research/spill-b-20260919/pro-single-day49d/chain.sh
rc=$?
echo "$(date -u +%FT%TZ) day 49d rc=$rc"
[ "${DRY_RUN:-0}" = 1 ] || [ $rc = 0 ] || exit $rc
if [ "${BOOTS_ONLY:-0}" = 1 ]; then echo "$(date -u +%FT%TZ) LANE-B-SITTING14-BOOTS-DONE"; else echo "$(date -u +%FT%TZ) LANE-B-SITTING14-DONE"; fi
