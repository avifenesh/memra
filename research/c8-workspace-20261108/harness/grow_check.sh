#!/usr/bin/env bash
# The DFlash-route check before the records (2026-11-09): the grow lane's agentic replay
# (grow_ab.sh: DFlash2 + FR-Spec, ctx 32,768, 8 sessions, prefix cache 4,000 MB, T=1, c1 c2 c4 c8
# per boot) with A = main 0c25263389 (grow-rewind landed) and B = the rebased lane head, arms
# base fix fix base in one lease.
#   BASE=.. FIX=.. grow_check.sh
set -u
R=<runs>
[ -e <lane>/YIELD ] && { echo yield; exit 0; }
AB_ARMS="1 2 3 4" gpulock run --ttl 110m --wait 6h --for "memra c8r2 DFlash-route check: grow replay main vs lane, ABBA" -- \
  bash <growrewind-harness>/grow_ab.sh "$BASE" "$FIX" "$R/grow"
echo GROW_CHECK_DONE
