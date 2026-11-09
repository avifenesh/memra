#!/usr/bin/env bash
# The rest of the scored campaign on C1c after the c8 phase (2026-11-09 00:20), in four leases
# of my arms plus the grow-rewind replay one arm per lease.
#   S=.. C1=.. campaign5.sh
set -u
H=<harness>
R=<runs>
export A=$S B=$C1
y() { [ -e <lane>/YIELD ] && { echo "yield"; exit 0; }; }
y; bash "$H/group2.sh" "g8|0|A B|agentic:8:0:256|" "mixed|0|A B B A A B|mixed|MEMRA_MAX_SESSIONS=12"
y; bash "$H/group2.sh" "c4|0|A B B A A B|agentic:4:1.0:768|"
y; bash "$H/group2.sh" "c1|0|A B|agentic:1:1.0:768|"
for a in 1 2 3 4; do
  y; AB_ARMS=$a gpulock run --ttl 50m --wait 6h --for "memra c8r2 grow-rewind replay arm $a (S vs C1c)" -- \
    bash <growrewind-harness>/grow_ab.sh "$S" "$C1" "$R/grow"
done
echo CAMPAIGN5_DONE
