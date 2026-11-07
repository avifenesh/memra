#!/usr/bin/env bash
# Rig-side: run several gb10_chain.sh modes back to back on one held unit, renewing its lease.
#   gb10_plan.sh UNIT LEASE BINTAG OUTROOT MODE...
set -u
UNIT=${1:?}; LEASE=${2:?}; BINTAG=${3:?}; ROOT=${4:?}; shift 4
G=$HOME/.cache/memra-gdnbf16-real/gates
mkdir -p "$ROOT"
for mode in "$@"; do
  sparklock renew "$LEASE" --unit "$UNIT" --ttl 6h >> "$ROOT/plan.log" 2>&1
  echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) start $mode" >> "$ROOT/plan.log"
  bash "$G/${CHAIN:-gb10_chain.sh}" "$UNIT" "$LEASE" "$BINTAG" "$mode" "$ROOT/$mode"
  echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) end $mode rc=$?" >> "$ROOT/plan.log"
done
echo PLAN_DONE >> "$ROOT/plan.log"
