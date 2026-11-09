#!/usr/bin/env bash
# campaign10 after its c8 group, resumed once the local checks ran (2026-11-09).
#   M=.. L=.. campaign11.sh
set -u
H=<harness>
y() { [ -e <lane>/YIELD ] && { echo "yield"; exit 0; }; }
export A=$M B=$L
y; bash "$H/group2.sh" "g8|0|A B|agentic:8:0:256|" "mixed|0|A B B A A B|mixed|MEMRA_MAX_SESSIONS=12"
y; bash "$H/group2.sh" "c4|0|A B B A A B|agentic:4:1.0:768|"
y; bash "$H/group2.sh" "c1|0|A B|agentic:1:1.0:768|"
y; BASE=$M FIX=$L bash "$H/grow_check.sh"
echo CAMPAIGN11_DONE
