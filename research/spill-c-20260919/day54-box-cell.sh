#!/usr/bin/env bash
# Day 54: the collector's entry point for cell `slices` (called by day40-run-cell.sh as `<script> <cell> <lockfd>`),
# handing the hold to day54-slice-cell.sh with the sitting's paths. Environment: D40_R, D40_TREE, D40_BINS, D54_MODEL.
set -uo pipefail
cell=$1; fd=$2
: "${D40_R:?}" "${D40_TREE:?}" "${D40_BINS:?}" "${D54_MODEL:?}"
[ "$cell" = slices ] || { echo "unknown cell $cell"; exit 2; }
exec bash "$D40_TREE/research/spill-c-20260919/day54-slice-cell.sh" "$fd" "$D40_TREE" "$D40_R" "$D54_MODEL" "$D40_BINS"
