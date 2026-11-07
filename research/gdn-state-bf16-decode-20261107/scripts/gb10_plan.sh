#!/usr/bin/env bash
# Rig-side: run gb10_chain.sh steps back to back on one held unit, renewing its lease before each.
#   gb10_plan.sh UNIT LEASE BINTAG OUTROOT STEP...
# STEP: build:bins | build:tests | gates | quality:CLASS:REP | probes | bench | cells
set -u
UNIT=${1:?}; LEASE=${2:?}; BINTAG=${3:?}; ROOT=${4:?}; shift 4
S=$HOME/.cache/memra-gdnbf16-decode/scripts
mkdir -p "$ROOT"
for step in "$@"; do
  sparklock renew "$LEASE" --unit "$UNIT" --ttl 5h >> "$ROOT/plan.log" 2>&1
  echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) start $step" >> "$ROOT/plan.log"
  IFS=: read -r mode a b <<< "$step"
  case "$mode" in
    build) WHAT=$a JOBS=10 bash "$S/gb10_chain.sh" "$UNIT" "$LEASE" "$BINTAG" build "$ROOT/build" ;;
    quality) QCLASS=$a QREP=$b bash "$S/gb10_chain.sh" "$UNIT" "$LEASE" "$BINTAG" quality "$ROOT/quality" ;;
    gates) MAINBIN='$HOME/.cache/memra-gdnbf16-decode/bins/main-cf85a23ff5-121a' bash "$S/gb10_chain.sh" "$UNIT" "$LEASE" "$BINTAG" gates "$ROOT/gates" ;;
    *) bash "$S/gb10_chain.sh" "$UNIT" "$LEASE" "$BINTAG" "$mode" "$ROOT/$mode" ;;
  esac
  echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) end $step rc=$?" >> "$ROOT/plan.log"
done
echo PLAN_DONE >> "$ROOT/plan.log"
