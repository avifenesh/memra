#!/usr/bin/env bash
# The scored campaign on the final tree (2026-11-09 from 06:30), A = main 0c25263389 (the
# landing base), B = the lane head rebased on it: agentic c8 (prime pass and measured pass), greedy
# c8 and the mixed shape, c4 and c1, one lease per group; then the DFlash-route grow replay.
#   M=.. L=.. campaign10.sh
set -u
H=<harness>
y() { [ -e <lane>/YIELD ] && { echo "yield"; exit 0; }; }
export A=$M B=$L
y; bash "$H/group2.sh" "c8|0|A B B A A B|agentic:8:1.0:768|"
y; bash "$H/group2.sh" "g8|0|A B|agentic:8:0:256|" "mixed|0|A B B A A B|mixed|MEMRA_MAX_SESSIONS=12"
y; bash "$H/group2.sh" "c4|0|A B B A A B|agentic:4:1.0:768|"
y; bash "$H/group2.sh" "c1|0|A B|agentic:1:1.0:768|"
y; BASE=$M FIX=$L bash "$H/grow_check.sh"
echo CAMPAIGN10_DONE
