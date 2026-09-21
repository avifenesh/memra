#!/usr/bin/env bash
# Day 23 local chain (lead ruling 26, item 3), merged tree 5f2060381: the five day-22 cells re-run on main's mechanism,
# then run-gen/run-spec. Serial: every cell takes /tmp/memra-5090.lock through the collector (the hit gate takes it itself).
set -uo pipefail
cd "$HOME/projects/wt-spill-b" || exit 1
S=research/spill-b-20260919; L=$S/rtx5090-day23/chain.log
step() { echo "== $(date -u +%FT%TZ) $*" >> "$L"; "$@" >> "$L" 2>&1; echo "rc=$? at $(date -u +%FT%TZ)" >> "$L"; }
step $S/run-day23-walk.sh walk 1800 17 16 48
step $S/run-day23-gate.sh gate 2400
step $S/run-day23-kc.sh kc 1800
step $S/run-day23-hitgate.sh hitgate
step $S/run-day23-twin.sh twin 1800
step $S/run-day23-genspec.sh genspec 3000
echo "CHAIN DONE $(date -u +%FT%TZ)" >> "$L"
